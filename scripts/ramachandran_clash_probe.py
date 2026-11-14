"""Does `clash` actually see the Ramachandran plot, or only pretend to?

`docs/claims/the-objective-should-restrain-the-output-not-teach-the-placement.md`
rejected a Ramachandran restraint as redundant, on the argument that the
Ramachandran plot is the empirical shadow of backbone sterics -- the disallowed
regions are disallowed because adjacent backbone atoms collide -- so a proper
clash term subsumes its hard constraint. That argument has been standing since
7 Nov and has never been tested.

It is testable without a checkpoint. Build a damaged window at a grid of fixed
(phi, psi), with the rest of the protein at its crystal pose, and read
`backbone_clash_loss` over N, CA, C, O, CB. If the argument holds, the map over
the (phi, psi) plane should light up where the real Ramachandran plot is empty
and stay near its crystal floor where the plot is populated. If it is flat, the
clash term does NOT subsume Ramachandran, nothing in the objective forbids an
impossible backbone, and the restraint that was rejected is needed after all.

The scan works by patching the two module constants `baseline_coords` reads for
`mode="helix"`, which is the only place the walk's torsions come from -- so this
measures exactly the geometry the trainer would start a rollout on, not a
reimplementation of it.

    python scripts/ramachandran_clash_probe.py
    python scripts/ramachandran_clash_probe.py --step 10 --structures 6

Omega stays at `BACKBONE_INIT_OMEGA` throughout: a non-planar peptide is a
different objection and `bond` already carries it.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca import model as model_module  # noqa: E402
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.growth import (  # noqa: E402
    baseline_coords,
    damage_window,
    growth_links,
)
from aanca.steric import backbone_clash_loss  # noqa: E402

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"

# The regions a MolProbity-style plot calls favoured, as coarse boxes in
# degrees: (phi_lo, phi_hi, psi_lo, psi_hi). Deliberately crude -- the question
# is whether clash separates populated from empty at all, not whether it
# reproduces a contour.
FAVOURED = (
    (-160.0, -45.0, -75.0, -5.0),     # alpha-R
    (-180.0, -45.0, 100.0, 180.0),    # beta / polyproline II
    (-180.0, -45.0, -180.0, -160.0),  # beta, wrapped in psi
    (35.0, 90.0, 10.0, 90.0),         # alpha-L
)


def favoured(phi: float, psi: float) -> bool:
    return any(
        lo_phi <= phi <= hi_phi and lo_psi <= psi <= hi_psi
        for lo_phi, hi_phi, lo_psi, hi_psi in FAVOURED
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--structures", type=int, default=6)
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--pool", type=int, default=400)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--damage", type=int, default=12)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument(
        "--scope", choices=("all", "window"), default="window",
        help=(
            "which residues `alive` scores. `all` includes the intact context, "
            "which CONFOUNDS the question: a window built at one fixed (phi, "
            "psi) is a rigid ideal helix or strand that collides with the "
            "surrounding protein whether or not those torsions are "
            "Ramachandran-allowed, and that placement collision dominates. "
            "`window` scores the damaged residues alone, which is the local "
            "steric question the Ramachandran plot is about"
        ),
    )
    parser.add_argument(
        "--bonded", action="store_true",
        help=(
            "pass bonded_pairs=True, i.e. score the 15 i/i+1 atom pairs at or "
            "beyond CLASH_MIN_SEPARATION instead of dropping the bonded "
            "residue pair whole. This is the switch the probe exists to "
            "evaluate"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")
    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[args.split]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = args.pool + args.eval_structures
    pool = available[:wanted] if len(available) >= wanted else available
    if not pool:
        print("no structures on disk for that split")
        return 1
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size, device=device,
        cache_limit=max(len(pool), 1),
    )
    order = list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    indices = order[: args.eval_structures][: args.structures]
    print(f"{len(pool)} on disk, {len(indices)} crops, "
          f"ids {[pool[i].pdb_id for i in indices]}")
    print(f"window {args.damage}, clash tolerance {args.tolerance} A, "
          f"grid {args.step} deg, scope {args.scope}, "
          f"bonded_pairs {args.bonded}")

    rng = random.Random(args.seed)
    crops = []
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 12:
                continue
            links = growth_links(topo)
            generator = torch.Generator().manual_seed(args.seed + index)
            damaged = damage_window(
                topo.n_residues, links, args.damage, generator, device
            )
            if not bool(damaged.any()):
                continue
            crops.append((topo, native, links, ~damaged, generator, damaged))
    if not crops:
        print("no usable crops")
        return 1

    # The crystal floor on these same crops, so the map has a reference that is
    # not zero: the term reads nonzero on real structures because backbone
    # hydrogen bonds are counted as overlap.
    with torch.no_grad():
        floor = []
        for topo, native, links, anchors, _g, damaged in crops:
            alive = (
                torch.ones(topo.n_residues, dtype=torch.bool)
                if args.scope == "all" else damaged
            )
            floor.append(float(backbone_clash_loss(
                native, topo, alive, links, args.tolerance,
                bonded_pairs=args.bonded,
            )))
    crystal = sum(floor) / len(floor)
    print(f"crystal floor on these crops: {crystal:.5f} A\n")

    phi_saved = model_module.BACKBONE_INIT_PHI
    psi_saved = model_module.BACKBONE_INIT_PSI
    grid = list(range(-180, 180, args.step))
    rows: list[tuple[float, float, float, bool]] = []
    try:
        with torch.no_grad():
            for phi in grid:
                for psi in grid:
                    model_module.BACKBONE_INIT_PHI = float(phi)
                    model_module.BACKBONE_INIT_PSI = float(psi)
                    scores = []
                    for topo, native, links, anchors, generator, damaged in crops:
                        start = baseline_coords(
                            native, topo, links, anchors, "helix", generator
                        )
                        alive = (
                            torch.ones(topo.n_residues, dtype=torch.bool)
                            if args.scope == "all" else damaged
                        )
                        scores.append(float(backbone_clash_loss(
                            start, topo, alive, links, args.tolerance,
                            bonded_pairs=args.bonded,
                        )))
                    mean = sum(scores) / len(scores)
                    rows.append((float(phi), float(psi), mean,
                                 favoured(float(phi), float(psi))))
    finally:
        model_module.BACKBONE_INIT_PHI = phi_saved
        model_module.BACKBONE_INIT_PSI = psi_saved

    # The map, as a text grid. psi across, phi down, in milli-angstroms of mean
    # overlap so the numbers fit.
    print("clash, milli-A of mean overlap per scored atom. psi across, phi down.")
    print("      " + "".join(f"{p:>6d}" for p in grid))
    for phi in grid:
        cells = []
        for psi in grid:
            value = next(v for a, b, v, _f in rows if a == phi and b == psi)
            cells.append(f"{value * 1000:>6.1f}")
        print(f"{phi:>5d} " + "".join(cells))

    good = [v for _a, _b, v, f in rows if f]
    bad = [v for _a, _b, v, f in rows if not f]
    print(f"\nfavoured boxes:     n={len(good):>3d}  mean "
          f"{sum(good) / len(good):.5f} A  max {max(good):.5f}")
    print(f"everywhere else:    n={len(bad):>3d}  mean "
          f"{sum(bad) / len(bad):.5f} A  max {max(bad):.5f}")
    print(f"crystal floor:                 {crystal:.5f} A")
    separation = (sum(bad) / len(bad)) / max(sum(good) / len(good), 1e-9)
    print(f"\ndisallowed / favoured ratio: {separation:.2f}x")
    print("A ratio near 1 means clash does NOT subsume Ramachandran and the")
    print("restraint rejected on 7 Nov is needed. A large ratio means it does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
