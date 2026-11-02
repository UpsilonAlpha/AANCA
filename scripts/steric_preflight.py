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
    parser.add_argument("--limit", type=int, default=600, help="0 for all")
    parser.add_argument("--seed", type=int, default=0)
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
    # **Re-root the paths.** The manifest stores whatever absolute path the
    # machine that built it used -- Windows paths, in this repo -- so the
    # stored ``path`` is vestigial and every consumer is expected to point it
    # at the local file first. ``train.py:84 on_disk`` is that step, and its
    # docstring says so; the first version of this script skipped it and all
    # 23,761 entries raised FileNotFoundError on a C:\ path. Replicated rather
    # than imported because ``train.py`` is a script, not a module.
    stated = splits[args.split]
    specs = []
    for spec in stated:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = ROOT / "data" / "pdb" / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                specs.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    print(f"split {args.split}: {len(stated)} listed, {len(specs)} present on disk")
    if not specs:
        print(
            "nothing is on disk, so there is nothing to check -- this is a "
            "failure to run, not an all-clear. Check data/pdb and that the "
            "paths were re-rooted."
        )
        return 1
    # A limit is a *sample*, not a prefix: the manifest is ordered, so the
    # first 600 entries are not representative of 23,761.
    if args.limit and args.limit < len(specs):
        import random as _random

        _random.Random(args.seed).shuffle(specs)
        specs = specs[: args.limit]
        print(f"sampling {len(specs)} of them, seed {args.seed}")
    print(f"tolerance {args.tolerance:g}")
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
    print(f"structures checked     : {len(overlaps)} of {len(specs)}")
    if not overlaps:
        # The first version of this script printed "residues with no CA: 0"
        # and exited 0 while having loaded nothing at all -- precisely the
        # false all-clear it exists to prevent. An inventory of nothing is a
        # failure to run.
        print(
            "NOTHING LOADED. Every structure raised, so every count below is "
            "vacuous. This is a failure to run, not an all-clear."
        )
        for pdb_id, text in failures[:3]:
            print(f"  --- {pdb_id}\n{text}")
        return 1
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
