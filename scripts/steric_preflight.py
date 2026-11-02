"""Does the steric term survive every structure in the split, before a card does?

v7 died on its first training update with "residue 149 (GLU) has no CA", after
the warm start had already loaded and the update-0 evaluation had already run.
The two crystal structures the tests use (1UBQ, 193L) are complete, so no test
could have produced that case, and every discovery of this kind otherwise costs
a GPU queue wait to find one bad residue.

So this walks the whole split on CPU and reports *all* of them at once:
anything that makes ``steric_topology`` or ``clash_loss`` raise, plus the
things that would not raise but are worth knowing -- residues with no CA,
unexpected atom names, and what a native structure actually scores. A crystal
structure must score 0.000 at the default tolerance; anything above that is a
radius or an exemption being wrong on chemistry the test structures do not
contain.

    sbatch scripts/bunya_cpu.slurm scripts/steric_preflight.py --split train

Reports and exits 0 even when it finds problems: the point is the inventory,
not a gate. Exit 1 is reserved for not being able to run at all.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aanca import residue_constants as rc  # noqa: E402
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.steric import (  # noqa: E402
    DEFAULT_TOLERANCE,
    clash_loss,
    steric_topology,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    manifest = ROOT / "data" / "manifest.json"
    if not manifest.exists():
        print(f"{manifest} missing -- run scripts/build_training_manifest.py")
        return 1
    splits = read_manifest(manifest)
    if args.split not in splits:
        print(f"no split {args.split!r}; have {sorted(splits)}")
        return 1
    specs = splits[args.split]
    if args.limit:
        specs = specs[: args.limit]
    print(f"split {args.split}: {len(specs)} structures, tolerance {args.tolerance:g}")
    print(flush=True)

    dataset = RepackingDataset(specs, concentrations=(0.0,))
    no_ca: list[str] = []
    odd_atoms: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    overlaps: list[tuple[float, str]] = []

    for i, spec in enumerate(specs):
        try:
            topo, native, _indices, _seed = dataset.load_crop(i, seed_residue=0)
            torsion = topo.torsion
            for r, names in enumerate(torsion.atom_names):
                if "CA" not in names:
                    no_ca.append(f"{spec.pdb_id}:{r}:{torsion.resnames[r]}")
                for name in names:
                    try:
                        rc.element(name)
                    except ValueError:
                        odd_atoms[name] += 1
            steric = steric_topology(torsion, native)
            value = float(
                clash_loss(native, torsion, steric, tolerance=args.tolerance)
            )
            overlaps.append((value, spec.pdb_id))
        except Exception:  # noqa: BLE001 -- an inventory, not a gate
            failures.append((spec.pdb_id, traceback.format_exc(limit=3)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(specs)} done", flush=True)

    print()
    print(f"structures that raised : {len(failures)}")
    for pdb_id, text in failures[:8]:
        print(f"  --- {pdb_id}\n{text}")
    print(f"residues with no CA    : {len(no_ca)}")
    for item in no_ca[:20]:
        print(f"  {item}")
    if len(no_ca) > 20:
        print(f"  ... and {len(no_ca) - 20} more")
    print(f"unassignable atom names: {len(odd_atoms)}")
    for name, count in odd_atoms.most_common(20):
        print(f"  {name!r} x{count}")

    if overlaps:
        overlaps.sort(reverse=True)
        nonzero = [o for o in overlaps if o[0] > 0]
        print()
        print(
            f"native overlap/atom   : {len(nonzero)}/{len(overlaps)} structures "
            f"above zero at tolerance {args.tolerance:g}"
        )
        print("  worst:")
        for value, pdb_id in overlaps[:10]:
            print(f"    {pdb_id}  {value:.6f}")
        print(
            "  A deposited structure should read 0.000000. Anything above it is "
            "a radius or an exemption that is wrong on chemistry 1UBQ and 193L "
            "do not contain -- read it as a finding, not as noise."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
