"""Find pairs of PDB depositions of the *same* protein, for the missing anchor.

**Why this exists.** `rmsd_dmg` is anchored at *do nothing* (10.77 A, the
model-free baseline) and at *reproduce the crystal* (0.00 A, the native). It has
**no anchor for a good non-native answer**, which is precisely what the project
is for: regrowing a binder interface into something that binds better than the
original would score badly against the crystal and be the result wanted. That is
a missing *positive control*, and missing positives are the error that produces
optimistic bias -- see
``docs/claims/the-success-criterion-has-no-positive-anchor.md``.

**The anchor.** Measure what the same window reads between two independent
depositions of one protein. If a 12-residue window differs by X A between two
crystal forms of the same sequence, then X is the scale at which "different but
real" lives, and an arm within X of native is as close as the crystal is
reproducible. The method is borrowed from the pMHC binder session, where a
cross-deposition null was what made a 0.441 A measurement interpretable.

**Why the existing pool cannot supply it, which was checked before downloading
anything.** The 412-structure pool holds no two structures from one cluster, and
neither do the 1903 structures on disk: across all 25,303 manifest entries there
are **zero** groups with two available members. The dataset is deduplicated by
cluster *by construction*, which is a stronger reason than the sequence-
comparison difficulty recorded earlier. So the anchor needs structures the
project does not have.

**Identity comes from RCSB's clustering, not from an alignment written here.**
``clusters-by-entity-100.txt`` groups entities at 100 % sequence identity, so two
PDB ids in one line are the same sequence by the authority that publishes the
structures. The 40 % file already on disk is what the training manifest used and
is far too loose for "the same protein": a 40 % cluster is a fold family.

Usage:

    python scripts/deposition_pairs.py                 # report candidates
    python scripts/deposition_pairs.py --fetch 12      # download partners
"""

from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "data" / "pdb"
EXTERNAL = ROOT / "data" / "external" / "rcsb"
CLUSTER_URL = (
    "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-100.txt"
)
#: Partners land here rather than in ``data/pdb``, so the training pool cannot
#: silently acquire near-duplicates of structures it already holds. A pool with
#: two depositions of one protein in it would leak the anchor into the data the
#: arms train on, which would make the anchor worthless and the split wrong.
PAIRS = ROOT / "data" / "deposition_pairs"


def cluster_file() -> Path:
    path = EXTERNAL / "clusters-by-entity-100.txt"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {CLUSTER_URL}")
    with requests.get(CLUSTER_URL, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    print(f"  {path.stat().st_size / 1e6:.1f} MB")
    return path


def on_disk() -> set[str]:
    if not PDB.exists():
        return set()
    return {name.split(".")[0].upper() for name in os.listdir(PDB)}


def is_deposition(token: str) -> bool:
    """A real PDB entry id, not a computed model.

    **This filter is load-bearing and its absence produced a wrong answer.**
    ``clusters-by-entity-100.txt`` includes AlphaFold (``AF_AFP12345F1_1``) and
    ModelArchive (``MA_...``) computed-structure entries alongside depositions.
    Splitting those on ``_`` yields the token ``AF``, which the first run of this
    script dutifully reported as the partner for nineteen of its twenty best
    candidates.

    Had they been fetched, the anchor would have measured **how far a prediction
    sits from a crystal structure** and been reported as how far two crystal
    structures sit from each other. That is a plausible number of roughly the
    right size answering a different question -- the most dangerous kind, and the
    same shape as the probe that reported ``birth free`` for a ``--birth none``
    run.

    A PDB id is four characters, starts with a digit, and is alphanumeric.
    """
    return len(token) == 4 and token[0].isdigit() and token.isalnum()


def candidates(limit: int = 40) -> list[tuple[str, list[str]]]:
    """``(held, partners)`` for each cluster with a structure we already have.

    Ranked by fewest partners first, because a protein deposited three times is
    a better anchor than one deposited four hundred times: the enormous clusters
    are lysozyme and trypsin point mutants, where "the same protein" starts to
    mean "the same protein with a different residue in it".
    """
    have = on_disk()
    rows = []
    with open(cluster_file(), encoding="utf-8") as handle:
        for line in handle:
            ids = {
                token.split("_")[0].upper()
                for token in line.split()
                if is_deposition(token.split("_")[0].upper())
            }
            if len(ids) < 2:
                continue
            held = sorted(ids & have)
            if not held:
                continue
            partners = sorted(ids - have)
            if not partners:
                continue
            rows.append((held[0], partners))
    rows.sort(key=lambda row: len(row[1]))
    return rows[:limit]


def fetch(pdb_id: str) -> Path | None:
    PAIRS.mkdir(parents=True, exist_ok=True)
    target = PAIRS / f"{pdb_id}.cif"
    if target.exists():
        return target
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        print(f"  {pdb_id}: HTTP {response.status_code}")
        return None
    target.write_bytes(response.content)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", type=int, default=0, metavar="N",
        help="download one partner for each of the N best candidate clusters",
    )
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    rows = candidates(args.limit)
    print(f"\n{len(rows)} clusters hold a structure we have plus >=1 we do not")
    print(f"{'held':>6}  {'partners':>8}  first few")
    for held, partners in rows[:20]:
        print(f"{held:>6}  {len(partners):>8}  {' '.join(partners[:6])}")

    if not args.fetch:
        print("\n--fetch N downloads one partner each, into data/deposition_pairs/")
        print("(deliberately NOT data/pdb: a training pool holding two")
        print(" depositions of one protein would leak the anchor into the data")
        print(" the arms train on, and the anchor would measure nothing.)")
        return 0

    got = []
    for held, partners in rows[: args.fetch]:
        path = fetch(partners[0])
        if path is not None:
            got.append((held, partners[0]))
            print(f"  {held} <-> {partners[0]}")
    print(f"\n{len(got)} pairs ready in {PAIRS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
