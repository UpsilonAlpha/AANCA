"""Download the small, high-resolution structures the geometry tests validate against.

    python scripts/fetch_test_structures.py

Default set, chosen to be small and to cover all 20 residue types between them:

* ``193L`` hen egg-white lysozyme, 1.33 A, 129 residues -- contains all 20 types
  on its own, including the ones that are easy to get wrong (TRP, HIS, MET, CYS).
* ``3NIR`` crambin, 0.48 A -- the tightest geometry available, so a round-trip
  error here is the code's and not the model's.
* ``1UBQ`` ubiquitin, 1.8 A -- the standard small test case.

These go to ``data/raw/``, which is gitignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from biotite.database import rcsb

DEFAULT_IDS = ("193L", "3NIR", "1UBQ")
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_ids", nargs="*", default=list(DEFAULT_IDS))
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--format", default="pdb", choices=["pdb", "cif", "bcif"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ids = args.pdb_ids or list(DEFAULT_IDS)
    for pdb_id in ids:
        path = args.out / f"{pdb_id.upper()}.{args.format}"
        if path.exists():
            print(f"{path.name}: already present")
            continue
        rcsb.fetch(pdb_id, args.format, target_path=str(args.out), overwrite=False)
        print(f"{path.name}: fetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
