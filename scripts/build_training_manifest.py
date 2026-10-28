"""Build the training manifest: candidate entries, leakage filter, cluster split.

    python scripts/build_training_manifest.py [--resolution 2.5] [--limit N]

This reconstructs BC40, because the published split is not downloadable --
AttnPacker's repository releases pre-trained models and result PDBs but neither
the training set nor the 37,266/1,500 chain split that DiffPack follows. The
reconstruction is a known divergence and belongs in the write-up.

Three stages, each of which can be inspected on its own:

1. **Candidates.** X-ray protein entries from RCSB, deduplicated to one
   representative per 40% sequence cluster -- which is what "BC40" means. The
   cluster file is RCSB's own ``clusters-by-entity-40.txt``, the resource BC40
   derives from, so this is the same clustering rather than a substitute.

2. **Leakage filter.** Every training entry within 40% sequence identity of any
   CASP test target is excluded, *with its whole cluster*. Dropping only the
   matching entries would leave their homologs in training.

   The identity search runs **MMseqs2 on RCSB's servers**, through
   ``rcsb.SequenceQuery``. That is the same tool FlowPacker and the Invariant
   Point Message Passing paper used for this filter, so the comparison does not
   hinge on us having reimplemented sequence search.

3. **Split.** Train and validation by cluster group, never splitting a group.

**No structures are downloaded here.** This writes a manifest; fetching the
coordinates is a separate and much larger step, deliberately kept apart so the
decision to pull tens of thousands of files is explicit.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import biotite.database.rcsb as rcsb
import requests

from aanca.data import ClusterIndex, StructureSpec, split_by_cluster, write_manifest
from aanca.structure import load_protein

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data" / "external"
CLUSTER_URL = (
    "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt"
)
BENCHMARK = EXTERNAL / "attnpacker" / "extracted" / "standard_results" / "native"

#: Minimum residues for a sequence to be worth an identity query. Very short
#: chains match far too much at 40% to be informative.
MIN_QUERY_LENGTH = 30


def cluster_file() -> Path:
    path = EXTERNAL / "rcsb" / "clusters-by-entity-40.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"clusters: cached ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    print("clusters: downloading RCSB's 40% identity clusters...")
    with requests.get(CLUSTER_URL, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    print(f"clusters: {path.stat().st_size / 1e6:.1f} MB")
    return path


def candidate_entries(
    resolution: float, min_residues: int, max_residues: int
) -> list[str]:
    """X-ray protein entries within the resolution and any size bounds.

    **The bounds are ours, not theirs.** No packing paper documents a resolution
    cutoff or a size range for the training pool, so they are parameters rather
    than buried constants.

    **Both size bounds default to off, because cropping replaced them.** An
    earlier version capped the pool at 40-1000 residues, for a real reason:
    retained activations are linear in residue count -- measured ~0.41 MB per
    residue per rollout step -- so a 3,570-residue assembly needs several GB for
    one truncation window. But the cap removes whole entries, and it removes the
    largest ones, which are the multi-chain assemblies: the interface-rich
    structures most worth having. ``src/aanca/crop.py`` bounds the *sample*
    instead, so a large entry contributes a ball of residues around a random
    seed and the rest of it stays available for another sample. Counts at 2.5 A
    for reference: no bounds 161,997; 30+ loses only 428 entries; 40-1000 keeps
    138,765 (86%); 40-600 keeps 109,637 (68%).

    A floor is still defensible -- a 12-residue peptide is not a packing target
    in any useful sense -- but it is left off here too, because
    ``RepackingDataset`` has to guard the floor at load time regardless: RCSB's
    ``deposited_polymer_monomer_count`` counts nucleotides, so a nucleic-acid
    complex passes any residue filter and still yields almost no protein. Two
    mechanisms for one job would only disagree.
    """
    query = (
        rcsb.FieldQuery(
            "rcsb_entry_info.resolution_combined", less_or_equal=resolution
        )
        & rcsb.FieldQuery(
            "rcsb_entry_info.polymer_entity_count_protein", greater_or_equal=1
        )
        & rcsb.FieldQuery("exptl.method", exact_match="X-RAY DIFFRACTION")
    )
    if min_residues:
        query = query & rcsb.FieldQuery(
            "rcsb_entry_info.deposited_polymer_monomer_count",
            greater_or_equal=min_residues,
        )
    if max_residues:
        query = query & rcsb.FieldQuery(
            "rcsb_entry_info.deposited_polymer_monomer_count",
            less_or_equal=max_residues,
        )
    ids = rcsb.search(query, return_type="entry")
    bounds = (
        f"{min_residues or 'any'}-{max_residues or 'any'} residues"
        if (min_residues or max_residues)
        else "no size bound (samples are cropped instead)"
    )
    print(
        f"candidates: {len(ids)} X-ray protein entries at <= {resolution} A, "
        f"{bounds}"
    )
    return [i.upper() for i in ids]


def test_target_sequences() -> dict[str, str]:
    """One sequence per CASP test target, from the benchmark archive."""
    if not BENCHMARK.is_dir():
        raise SystemExit(
            f"{BENCHMARK} missing -- run scripts/fetch_benchmark.py first. The "
            "leakage filter needs the test target sequences."
        )
    out: dict[str, str] = {}
    for sub in ("casp_all", "non_native"):
        directory = BENCHMARK / sub
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pdb")):
            if path.stem in out:
                continue
            try:
                protein = load_protein(path, repair_columns=True)
            except Exception:
                continue
            chains = protein.sequences()
            if chains:
                out[path.stem] = max(chains.values(), key=len)
    print(f"test targets: {len(out)} sequences from the benchmark archive")
    return out


def leaking_entries(
    sequences: dict[str, str], pause: float = 0.2, attempts: int = 4
) -> tuple[set[str], list[str]]:
    """Entries within 40% identity of any test target, via RCSB's MMseqs2.

    **A failed query is retried, and one that keeps failing is handed back to
    the caller rather than absorbed.** This was found the hard way. On a
    rebuild, one of the 210 queries hit a transient ``RequestError``; the
    original version logged a line and carried on, and the holdout came back
    one entry smaller than the previous run -- which is to say the leakage
    guarantee quietly weakened, leaving no trace except console output nobody
    would diff. A network blip must not get to decide what is in the training
    set, so failures now retry with backoff and the caller refuses to write a
    manifest built on an incomplete search.
    """
    hits: set[str] = set()
    failed: list[str] = []
    long_enough = {
        name: seq for name, seq in sequences.items() if len(seq) >= MIN_QUERY_LENGTH
    }
    print(
        f"leakage filter: querying {len(long_enough)} target sequences "
        f"(skipping {len(sequences) - len(long_enough)} under {MIN_QUERY_LENGTH} residues)"
    )
    for index, (name, sequence) in enumerate(sorted(long_enough.items()), start=1):
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                query = rcsb.SequenceQuery(sequence, "protein", min_identity=0.4)
                hits.update(i.upper() for i in rcsb.search(query, return_type="entry"))
                last = None
                break
            except Exception as exc:
                last = exc
                # Back off: a transient failure here is usually rate limiting.
                time.sleep(pause * (2**attempt))
        if last is not None:
            failed.append(f"{name}: {type(last).__name__}")
            print(f"  {name} failed after {attempts} attempts: {type(last).__name__}")
        if index % 25 == 0:
            print(f"  {index}/{len(long_enough)} queried, {len(hits)} entries so far")
        time.sleep(pause)
    print(f"leakage filter: {len(hits)} entries within 40% of a test target")
    if failed:
        print(f"  {len(failed)} queries failed after retries: {failed}")
    return hits, failed


def stored_leakage() -> set[str]:
    """The leaking-entry list from a previous build.

    Persisted precisely so the guarantee is auditable rather than taken on
    trust; reading it back is the same artifact an auditor would read. Refuses
    a meta file written without the filter, since that list is empty for a
    reason that must not propagate silently.
    """
    path = ROOT / "data" / "manifest_meta.json"
    if not path.exists():
        raise SystemExit(f"{path} missing -- nothing to reuse, drop --reuse-leakage")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not meta.get("leakage_filter_applied"):
        raise SystemExit(
            f"{path} was written with the leakage filter skipped, so its entry "
            "list is empty by construction. Re-run without --reuse-leakage."
        )
    entries = {str(e).upper() for e in meta.get("leaking_entries", ())}
    if not entries:
        raise SystemExit(f"{path} records no leaking entries; refusing to reuse it")
    print(f"leakage filter: reusing {len(entries)} entries from {path.name}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=float, default=2.5)
    parser.add_argument(
        "--min-residues", type=int, default=0, help="0 for no lower bound"
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=0,
        help="0 for no upper bound; large entries are cropped per sample instead",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.04)
    parser.add_argument("--limit", type=int, default=0, help="cap candidates (testing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reuse-leakage",
        action="store_true",
        help="read the leaking-entry list from data/manifest_meta.json instead "
             "of re-querying. The list depends only on the test targets, not on "
             "the candidate pool, so it is valid for a rebuild that changes the "
             "pool -- and re-running 210 MMseqs2 queries to get the same answer "
             "is not a check of anything.",
    )
    parser.add_argument(
        "--allow-failed-queries",
        action="store_true",
        help="write the manifest even if an identity query failed after retries "
             "and no stored list is available to union in. Weakens the leakage "
             "guarantee, so it has to be asked for.",
    )
    parser.add_argument(
        "--skip-leakage",
        action="store_true",
        help="DANGEROUS: build without the identity filter, for pipeline testing only",
    )
    args = parser.parse_args()

    clusters = ClusterIndex.from_file(cluster_file())
    print(f"clusters: {len(clusters)} entries indexed")

    entries = candidate_entries(
        args.resolution, args.min_residues, args.max_residues
    )
    if args.limit:
        entries = entries[: args.limit]
        print(f"candidates: capped at {len(entries)}")

    # BC40 means one representative per 40% cluster, not every entry.
    by_group: dict[str, str] = {}
    for entry in entries:
        by_group.setdefault(clusters.group_of(entry), entry)
    representatives = sorted(by_group.values())
    print(
        f"deduplicated: {len(representatives)} representatives "
        f"({len(entries) - len(representatives)} dropped as cluster duplicates)"
    )

    histogram = clusters.group_size_histogram(entries)
    biggest = max(histogram) if histogram else 0
    print(f"group sizes over the candidate pool: largest {biggest}")
    for size in sorted(histogram)[:6]:
        print(f"  size {size:<6}{histogram[size]} groups")
    if biggest > 500:
        print(
            f"  WARNING: a group of {biggest} entries suggests a promiscuous "
            "entity chained unrelated entries together"
        )

    if args.skip_leakage:
        print("\nWARNING: leakage filter SKIPPED -- this manifest is not usable "
              "for any reported number")
        holdout: set[str] = set()
    elif args.reuse_leakage:
        holdout = stored_leakage()
    else:
        holdout, failed = leaking_entries(test_target_sequences())
        # A query that failed is a target whose homologs were never looked for,
        # so the split would look clean without being clean. Union the stored
        # list in, which is the cheap way to recover a blip, and refuse outright
        # if there is nothing to fall back on.
        if failed:
            if args.allow_failed_queries:
                print(
                    f"  WARNING: {len(failed)} queries failed and were allowed. "
                    "The leakage guarantee is weaker than the manifest implies."
                )
            else:
                before = len(holdout)
                holdout |= stored_leakage()
                print(
                    f"  {len(failed)} queries failed; unioned the stored list in "
                    f"({before} -> {len(holdout)} entries)"
                )

    specs = [
        StructureSpec(
            pdb_id=entry,
            path=ROOT / "data" / "pdb" / f"{entry}.cif",
            group=clusters.group_of(entry),
        )
        for entry in representatives
    ]
    splits = split_by_cluster(
        specs,
        holdout,
        clusters,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        require_holdout=not args.skip_leakage,
    )

    out = ROOT / "data" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(out, splits)
    # Persist the actual entry and cluster lists, not just counts. The
    # test-leakage guarantee is the one claim in this pipeline that cannot be
    # re-derived cheaply -- it costs 210 MMseqs2 queries -- so it has to be
    # auditable from the artifact rather than taken on trust.
    banned_clusters: set[str] = set()
    for pdb_id in holdout:
        banned_clusters |= clusters.clusters_of(pdb_id) or {pdb_id.upper()}

    meta = {
        "resolution_max": args.resolution,
        "min_residues": args.min_residues,
        "max_residues": args.max_residues,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "candidates": len(entries),
        "representatives": len(representatives),
        "leakage_excluded_entries": len(holdout),
        "leakage_filter_applied": not args.skip_leakage,
        "counts": {name: len(specs) for name, specs in splits.items()},
        "leaking_entries": sorted(holdout),
        "banned_clusters": sorted(banned_clusters),
    }
    (ROOT / "data" / "manifest_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\nmanifest: {out}")
    for name, group in splits.items():
        print(f"  {name:<12}{len(group):>7}")
    print("\nNo coordinates downloaded. That is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
