"""Download the coordinates named in the training manifest.

    python scripts/fetch_structures.py --split validation          # small, start here
    python scripts/fetch_structures.py --split train --limit 2000
    python scripts/fetch_structures.py --split train               # everything

Kept separate from ``build_training_manifest.py`` on purpose: the manifest is
cheap and reproducible, while this pulls tens of thousands of files, so the
decision to do it should be explicit rather than a side effect of planning the
split.

Format is **BinaryCIF** (``bcif``), which is several times smaller than mmCIF or
PDB for the same content and parses faster. It also avoids PDB's hard limits on
atom and chain counts, which matter for the large multi-chain entries that carry
the interfaces this project cares about.

Run with ``--estimate`` first: it samples a handful of entries and reports the
projected total, so a multi-gigabyte download is a decision rather than a
surprise.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import biotite.database.rcsb as rcsb

from aanca.data import read_manifest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
TARGET = ROOT / "data" / "pdb"
FORMAT = "bcif"


def present(directory: Path) -> set[str]:
    return {p.stem.upper() for p in directory.glob(f"*.{FORMAT}")}


def estimate(ids: list[str], sample: int = 8) -> None:
    """Download a few entries and project the total from their mean size."""
    scratch = TARGET / "_estimate"
    scratch.mkdir(parents=True, exist_ok=True)
    sizes = []
    for pdb_id in ids[:sample]:
        try:
            rcsb.fetch(pdb_id, FORMAT, target_path=str(scratch), overwrite=True)
            path = scratch / f"{pdb_id}.{FORMAT}"
            if path.exists():
                sizes.append(path.stat().st_size)
        except Exception as exc:
            print(f"  {pdb_id}: {type(exc).__name__}")
    if not sizes:
        print("could not sample any entries")
        return
    mean = sum(sizes) / len(sizes)
    print(
        f"sampled {len(sizes)} entries, mean {mean / 1e3:.0f} kB\n"
        f"projected for {len(ids)} entries: {mean * len(ids) / 1e9:.2f} GB"
    )
    print("\nRe-run without --estimate to download.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--pause", type=float, default=0.05)
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"{MANIFEST} missing -- run scripts/build_training_manifest.py first")
        return 1

    splits = read_manifest(MANIFEST)
    if args.split not in splits:
        print(f"no split {args.split!r}; available: {sorted(splits)}")
        return 1

    ids = [spec.pdb_id for spec in splits[args.split]]
    if args.limit:
        ids = ids[: args.limit]

    TARGET.mkdir(parents=True, exist_ok=True)
    have = present(TARGET)
    todo = [i for i in ids if i not in have]
    print(
        f"split {args.split}: {len(ids)} entries, {len(ids) - len(todo)} already "
        f"present, {len(todo)} to fetch"
    )
    if not todo:
        return 0

    if args.estimate:
        estimate(todo)
        return 0

    failed: list[tuple[str, str]] = []
    for index, pdb_id in enumerate(todo, start=1):
        try:
            rcsb.fetch(pdb_id, FORMAT, target_path=str(TARGET), overwrite=False)
        except Exception as exc:
            failed.append((pdb_id, type(exc).__name__))
        if index % 250 == 0:
            done = len(present(TARGET))
            print(f"  {index}/{len(todo)} attempted, {done} on disk, {len(failed)} failed")
        time.sleep(args.pause)

    total = sum(p.stat().st_size for p in TARGET.glob(f"*.{FORMAT}"))
    print(f"\n{len(present(TARGET))} entries on disk, {total / 1e9:.2f} GB")
    if failed:
        print(f"{len(failed)} failed, e.g. {failed[:5]}")
        print("Obsolete or withdrawn entries are expected; they are simply skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
