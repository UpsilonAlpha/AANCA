"""What RMSD does satisfying the geometry alone force, with no network at all?

**The denominator every regeneration number in this project has been missing.**

`scripts/baseline_probe.py` measures where the rollout *starts* -- the window
left where the reference walk put it, 12.23 A for a helix on Bunya's pool. That
is the "do nothing" line. It is not the floor, because a 12-residue window
pinned at both ends by intact flanks **cannot go anywhere**: closing the chain
to its neighbours and not overlapping itself already forces the window into some
RMSD range, with no knowledge whatever of the native pose. If that range turns
out to be ~9 A then `g21_pool`'s 8.5 A was never a signal and neither would 5 A
be. If it is ~14 A then a 4 A result is real.

So this optimises the window against **bond closure and sterics only** and
reports what it lands at. There is no native information in the objective -- not
a distance to native, not a torsion prior, not a fragment library. The only
inputs are the topology, the covalent links and the intact flanks' positions.

**The degrees of freedom are deliberately the rule's own.** Each window
residue gets a free rigid move about its CA -- `aanca.growth.rigid_relax`, the
same function the rollout applies, with ideal internal geometry throughout.
Not torsions. That matters: if the floor were measured in torsion space and the
rule works in frame space, a gap between them could be the parameterisation
rather than the learning, and the comparison would prove nothing. Here the
optimiser and the rule search *the same set of structures* and differ only in
that one of them is told where native is.

**Two superposition conventions, both reported, because they disagree.**
`fit_all` fits on every alive residue and is what every arm in this record was
scored by; `fit_intact` fits on the surviving context alone, which is what
`backbone_rmsd_on`'s docstring prescribes and what `g22_random`'s absolute
pass/fail bar is a line on. The second is larger -- the first lets the displaced
window help choose the frame that measures it.

**Read the MEAN, not the best.** A native-blind method has no way to pick which
valid completion to emit, so the expected RMSD of a random valid solution is the
honest floor. `best` is reported as the geometric lower bound: if even the best
of N restarts cannot beat the rule, the rule is at the ceiling of what the
constraints permit and has learned nothing. If the mean is far worse than the
rule while the best is comparable, the rule has learned to *select* among valid
completions, which is the interesting claim.

    python scripts/geometric_floor.py --damage 12 --structures 12
    python scripts/geometric_floor.py --damage 12 --restarts 16 --iters 800

`bond` and `clash` are printed as achieved values, not hidden: a floor from an
optimiser that failed to close the chain is not a floor, and the reader has to
be able to see that it converged.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.fape import backbone_rmsd_on  # noqa: E402
from aanca.growth import (  # noqa: E402
    BASELINE_MODES,
    baseline_coords,
    damage_window,
    growth_links,
    peptide_bond_restraint,
    rigid_relax,
)
from aanca.steric import backbone_clash_loss  # noqa: E402

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"


def relax_to_geometry(
    start: torch.Tensor,
    topo,
    links: torch.Tensor,
    alive: torch.Tensor,
    damaged: torch.Tensor,
    *,
    iters: int,
    lr: float,
    tolerance: float,
    clash_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """Move the window's frames to close the chain and stop overlapping.

    Returns the relaxed coordinates and the achieved (bond, clash). The
    objective is the sum of exactly those two terms and nothing else -- adding
    anything that reads ``native`` would make this a different measurement and
    a useless one.
    """
    axis_angle = torch.zeros(
        (topo.n_residues, 3), dtype=start.dtype, requires_grad=True
    )
    translation = torch.zeros(
        (topo.n_residues, 3), dtype=start.dtype, requires_grad=True
    )
    optimiser = torch.optim.Adam([axis_angle, translation], lr=lr)
    bond = clash = torch.zeros((), dtype=start.dtype)
    for _ in range(iters):
        optimiser.zero_grad(set_to_none=True)
        coords = rigid_relax(
            start, axis_angle, translation, topo, damaged,
            local=True, lab_rotation=False,
        )
        bond = peptide_bond_restraint(coords, topo, alive, links)
        clash = backbone_clash_loss(
            coords, topo, alive, links, tolerance=tolerance
        )
        (bond + clash_weight * clash).backward()
        optimiser.step()
    with torch.no_grad():
        coords = rigid_relax(
            start, axis_angle, translation, topo, damaged,
            local=True, lab_rotation=False,
        )
        bond = peptide_bond_restraint(coords, topo, alive, links)
        clash = backbone_clash_loss(
            coords, topo, alive, links, tolerance=tolerance
        )
    return coords.detach(), float(bond), float(clash)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--pool", type=int, default=400)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--damage", type=int, default=12)
    parser.add_argument(
        "--initial", default="random", choices=list(BASELINE_MODES),
        help="the start state the optimiser is launched from. `random` is the "
             "neutral one and is what g22_random runs; `helix` is what every "
             "arm from g1 to g21 ran, so the floor differs between them and "
             "an arm has to be read against its own",
    )
    parser.add_argument(
        "--restarts", type=int, default=8,
        help="independent random starts per crop. The MEAN over these is the "
             "floor; the BEST is the geometric lower bound",
    )
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--clash-tolerance", type=float, default=0.25,
        help="backbone_clash_loss reads 0.0258 A on crystal ubiquitin at "
             "tolerance 0, so a nonzero tolerance is what its own docstring "
             "asks for when the floor matters",
    )
    parser.add_argument("--clash-weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.damage < 2:
        print("--damage must be at least 2; a one-residue window has no floor")
        return 1

    device = torch.device(args.device)
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
    # Which structures, not just how many -- the same trap baseline_probe.py
    # documents: `available[:412]` is a different 412 on the laptop and on
    # Bunya, and that difference put the helix baseline at 10.77 locally
    # against 12.23 on the cluster.
    print(f"{len(pool)} structures on disk, {len(indices)} held out at "
          f"crop {args.crop_size}, window {args.damage}, start {args.initial!r}")
    print(f"pool: {len(available)} available; held-out ids "
          f"{[pool[i].pdb_id for i in indices]}")
    print(f"optimiser: Adam lr {args.lr}, {args.iters} iters, "
          f"{args.restarts} restarts/crop, clash tolerance "
          f"{args.clash_tolerance}, clash weight {args.clash_weight}")
    print("objective: peptide_bond_restraint + clash. NO native information.")
    if args.initial != "random" and args.restarts > 1:
        # `helix` and `extended` are fixed reference walks: the generator is
        # passed but nothing draws from it, so every restart rebuilds the same
        # start and reaches the same optimum. The floor and the lower bound
        # then print as the SAME number, which reads like a converged
        # distribution and is not one.
        print(f"  NOTE: start {args.initial!r} is DETERMINISTIC, so the "
              f"{args.restarts} restarts are identical and `floor` == `best` "
              f"by construction. There is no distribution to read and no "
              f"lower bound; use --restarts 1, or --initial random for a "
              f"spread.")
    print()

    start_all: list[float] = []
    start_intact: list[float] = []
    mean_all: list[float] = []
    mean_intact: list[float] = []
    best_all: list[float] = []
    best_intact: list[float] = []
    bonds: list[float] = []
    clashes: list[float] = []

    header = (
        f"{'id':>6}  {'start_all':>9}  {'start_int':>9}  "
        f"{'floor_all':>9}  {'floor_int':>9}  {'best_int':>8}  "
        f"{'bond':>7}  {'clash':>7}"
    )
    print(header)
    for index in indices:
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=random.Random(args.seed + index))
        except Exception:
            continue
        if topo.n_residues < args.damage:
            continue
        links = growth_links(topo)
        alive = torch.ones(topo.n_residues, dtype=torch.bool, device=device)
        generator = torch.Generator().manual_seed(args.seed + index)
        damaged = damage_window(
            topo.n_residues, links, args.damage, generator, device
        )
        if not bool(damaged.any()):
            continue
        anchors = ~damaged
        per_all: list[float] = []
        per_intact: list[float] = []
        first_start = None
        for restart in range(args.restarts):
            draw = torch.Generator().manual_seed(
                args.seed + index * 1000 + restart
            )
            start = baseline_coords(
                native, topo, links, anchors, args.initial, draw
            )
            if first_start is None:
                first_start = (
                    float(backbone_rmsd_on(start, native, topo, alive, damaged)),
                    float(
                        backbone_rmsd_on(start, native, topo, anchors, damaged)
                    ),
                )
            relaxed, bond, clash = relax_to_geometry(
                start, topo, links, alive, damaged,
                iters=args.iters, lr=args.lr,
                tolerance=args.clash_tolerance,
                clash_weight=args.clash_weight,
            )
            per_all.append(
                float(backbone_rmsd_on(relaxed, native, topo, alive, damaged))
            )
            per_intact.append(
                float(backbone_rmsd_on(relaxed, native, topo, anchors, damaged))
            )
            bonds.append(bond)
            clashes.append(clash)
        if not per_all or first_start is None:
            continue
        start_all.append(first_start[0])
        start_intact.append(first_start[1])
        mean_all.append(statistics.fmean(per_all))
        mean_intact.append(statistics.fmean(per_intact))
        best_all.append(min(per_all))
        best_intact.append(min(per_intact))
        print(
            f"{pool[index].pdb_id:>6}  {first_start[0]:>9.2f}  "
            f"{first_start[1]:>9.2f}  {statistics.fmean(per_all):>9.2f}  "
            f"{statistics.fmean(per_intact):>9.2f}  {min(per_intact):>8.2f}  "
            f"{bonds[-1]:>7.4f}  {clashes[-1]:>7.4f}"
        )

    if not mean_all:
        print("no crop produced a window; nothing measured")
        return 1

    print()
    print(f"over {len(mean_all)} crops x {args.restarts} restarts:")
    print(f"  start state        fit_all {statistics.fmean(start_all):>6.2f}   "
          f"fit_intact {statistics.fmean(start_intact):>6.2f}")
    print(f"  GEOMETRIC FLOOR    fit_all {statistics.fmean(mean_all):>6.2f}   "
          f"fit_intact {statistics.fmean(mean_intact):>6.2f}   <- the floor")
    print(f"  best of restarts   fit_all {statistics.fmean(best_all):>6.2f}   "
          f"fit_intact {statistics.fmean(best_intact):>6.2f}   <- lower bound")
    print(f"  achieved bond {statistics.fmean(bonds):.4f}, "
          f"clash {statistics.fmean(clashes):.4f}")
    print()
    print("Read the floor line against the arm, and the arm's own start state.")
    print("A rule at or above the floor has learned nothing: the constraints")
    print("alone put a window there without being told where native is. A rule")
    print("between the floor and the lower bound has learned to SELECT among")
    print("valid completions, which is the claim worth making. Below the lower")
    print("bound would mean it found completions this optimiser could not, in")
    print("which case raise --restarts before believing it.")
    print()
    print("If `bond` above is not small, the optimiser did not close the chain")
    print("and this is not a floor -- raise --iters or --lr and run it again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
