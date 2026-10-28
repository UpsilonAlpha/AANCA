"""What does cropping cost? Supervised fraction against crop size, on real entries.

    python scripts/crop_report.py [--structures 60] [--min-residues 600]

A crop's boundary shell has incomplete perception and is excluded from the loss
(see ``src/aanca/crop.py``). The shell is roughly one interaction cutoff deep
whatever the crop size, so the share of a crop spent on context falls as the
crop grows -- and that share is what ``crop_size`` trades against memory.

This measures it rather than assuming it. Memory is the other half of the trade
and is already measured: ~0.41 MB of retained activations per residue per step,
so a truncation window of 12 steps costs roughly ``0.41 * crop_size * 12`` MB.

Only entries above ``--min-residues`` are included, since smaller ones are taken
whole and cost nothing.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from aanca.crop import crop_report
from aanca.data import read_manifest
from aanca.structure import load_protein

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"

#: Retained activations per residue per rollout step, in MB. Measured, not
#: estimated -- see scripts/measure_rollout_memory.py.
MB_PER_RESIDUE_STEP = 0.41


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=60)
    parser.add_argument("--min-residues", type=int, default=600)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--truncate", type=int, default=12)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    sizes = (192, 256, 384, 512, 768, 1024)

    if not MANIFEST.exists():
        print("no manifest -- run scripts/build_training_manifest.py")
        return 1

    rng = random.Random(0)
    totals: dict[int, list[float]] = {s: [] for s in sizes}
    used = 0
    skipped_small = 0

    for spec in read_manifest(MANIFEST)[args.split]:
        if used >= args.structures:
            break
        path = next(
            (
                PDB / f"{spec.pdb_id}{suffix}"
                for suffix in (".bcif", ".cif", ".pdb")
                if (PDB / f"{spec.pdb_id}{suffix}").exists()
            ),
            None,
        )
        if path is None:
            continue
        try:
            protein = load_protein(path)
        except Exception:
            continue
        if len(protein) < args.min_residues:
            skipped_small += 1
            continue
        report = crop_report(
            protein, sizes=sizes, cutoff=args.cutoff, seeds=args.seeds, rng=rng
        )
        for size, fraction in report.items():
            totals[size].append(fraction)
        used += 1
        print(f"  {spec.pdb_id}  {len(protein):>5} residues  "
              + "  ".join(f"{s}:{report[s]:.2f}" for s in sizes))

    if not used:
        print(f"no entries over {args.min_residues} residues with coordinates on disk")
        return 1

    print(
        f"\n{used} entries over {args.min_residues} residues "
        f"({skipped_small} smaller ones skipped -- those are taken whole)"
    )
    print(f"\n{'crop':>6}{'supervised':>13}{'effective':>11}{'MB/window':>11}")
    for size in sizes:
        fraction = sum(totals[size]) / len(totals[size])
        print(
            f"{size:>6}{fraction:>12.1%}{fraction * size:>11.0f}"
            f"{MB_PER_RESIDUE_STEP * size * args.truncate:>11.0f}"
        )
    print(
        "\n'effective' is supervised residues per sample: the number that "
        "actually carries gradient. 'MB/window' is retained activations for a "
        f"{args.truncate}-step truncation window."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
