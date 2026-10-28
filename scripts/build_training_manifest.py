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


def candidate_entries(resolution: float) -> list[str]:
    """X-ray protein entries at or better than ``resolution``.

    The resolution cutoff is **ours**, not theirs: no packing paper documents
    one. It is recorded as a parameter so the choice is visible rather than
    buried.
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
    ids = rcsb.search(query, return_type="entry")
    print(f"candidates: {len(ids)} X-ray protein entries at <= {resolution} A")
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
    sequences: dict[str, str], pause: float = 0.2
) -> tuple[set[str], list[str]]:
    """Entries within 40% identity of any test target, via RCSB's MMseqs2."""
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
        try:
            query = rcsb.SequenceQuery(sequence, "protein", min_identity=0.4)
            found = rcsb.search(query, return_type="entry")
            hits.update(i.upper() for i in found)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}")
        if index % 25 == 0:
            print(f"  {index}/{len(long_enough)} queried, {len(hits)} entries so far")
        time.sleep(pause)
    print(f"leakage filter: {len(hits)} entries within 40% of a test target")
    if failed:
        print(f"  {len(failed)} queries failed, e.g. {failed[:3]}")
    return hits, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=float, default=2.5)
    parser.add_argument("--validation-fraction", type=float, default=0.04)
    parser.add_argument("--limit", type=int, default=0, help="cap candidates (testing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-leakage",
        action="store_true",
        help="DANGEROUS: build without the identity filter, for pipeline testing only",
    )
    args = parser.parse_args()

    clusters = ClusterIndex.from_file(cluster_file())
    print(f"clusters: {len(clusters)} entries indexed")

    entries = candidate_entries(args.resolution)
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
    else:
        holdout, _failed = leaking_entries(test_target_sequences())

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
    meta = {
        "resolution_max": args.resolution,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "candidates": len(entries),
        "representatives": len(representatives),
        "leakage_excluded_entries": len(holdout),
        "leakage_filter_applied": not args.skip_leakage,
        "counts": {name: len(specs) for name, specs in splits.items()},
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
