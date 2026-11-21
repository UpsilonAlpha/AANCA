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
    python scripts/geometric_floor.py runs/g22_random/best.pt --restarts 16

**Pass the arm's checkpoint and the floor is measured on the arm's crops.**
Without one this script picks its own held-out slice, which is how the
corrected random-start floor came to report a 14.76 / 16.28 start state against
`g22_random`'s own 12.38 / 13.44: same convention, harder windows, and a floor
measured on harder windows flatters whatever it is the denominator for. Given a
`best.pt` the crop ids come from the checkpoint's `eval_indices` and the pool,
crop size, window and start mode come from its recorded `args`, so the two
numbers describe the same task. The header says which of the two happened.

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
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="an arm's best.pt. Given one, the crops come from its "
             "`eval_indices` and the pool/crop/window/start come from its "
             "recorded `args`, so the floor is measured on exactly the "
             "windows that arm was judged on. Omit it and this script picks "
             "its own slice, which is comparable to nothing in particular",
    )
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

    # **The checkpoint's own args, never this script's defaults.** `best.pt`
    # carries `args` precisely so a probe pointed at it does not substitute its
    # own; `train_growth.py`'s save site names the two 6 Nov runs that were
    # wasted that way, and the floor is the third: it reshuffled its own slice
    # and was quoted against an arm it never shared a crop with.
    #
    # Read the blob DIRECTLY rather than through
    # `regrowth_dispersion.rule_from_checkpoint`. That helper is the right one
    # for a probe that rolls the rule out, but it builds an `AANCARule` and
    # `load_state_dict`s it, raising `KeyError` when the blob holds neither
    # `rule` (last.pt) nor `state_dict` (best.pt). This optimiser is
    # network-free -- the objective is bond closure plus sterics and there is
    # no forward pass anywhere in it -- so making the floor depend on weights
    # loading would let an architecture mismatch, or a checkpoint written by a
    # different trainer, block a measurement that needs none of it. All that is
    # wanted here is `args` and `eval_indices`.
    checkpoint_label = "none -- this script chose its own crops"
    recorded_indices: list[int] = []
    if args.checkpoint:
        blob = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        run_args = dict(blob.get("args") or {})
        checkpoint_label = (
            f"{args.checkpoint} at update {blob.get('update', '?')}"
        )
        missing: list[str] = []
        # (floor's own attribute, checkpoint key). `pool` and `structures`
        # collide across the two scripts and the mapping is not the identity:
        # the trainer's `--structures` is the size of the TRAINING pool, which
        # is what this script calls `--pool`, while this script's
        # `--structures` is how many of the held-out crops to actually measure
        # and is the reader's own knob. Getting the pool size right matters as
        # much as the indices do -- `eval_indices` are positions in
        # `available[:structures + eval_structures]`, so a differently sized
        # pool makes the same integer name a different protein.
        for attribute, key in (
            ("split", "split"),
            ("pool", "structures"),
            ("eval_structures", "eval_structures"),
            ("seed", "seed"),
            ("crop_size", "crop_size"),
            ("initial", "initial"),
            ("damage", "damage"),
        ):
            if key not in run_args:
                missing.append(key)
                continue
            value = run_args[key]
            if attribute == "damage" and (not isinstance(value, int)
                                          or value < 2):
                # A `--damage 0` run is not a regeneration run and has no
                # window to put a floor under. Say so instead of exiting on
                # the generic guard below, which would read as a bad CLI flag.
                print(f"  !! WARNING: checkpoint records damage {value!r}, "
                      f"which is not a window. KEEPING --damage "
                      f"{args.damage}. The floor and that run are then NOT "
                      f"measuring the same task.")
                continue
            was = getattr(args, attribute)
            setattr(args, attribute, value)
            note = "" if was == value else f" (CLI/default was {was!r})"
            print(f"  from checkpoint args: {attribute} = {value!r}{note}")
        if args.seed != 0:
            # `train_growth.score` is declared `seed: int = 0` and the training
            # loop calls it WITHOUT a seed, so the trainer draws its held-out
            # crops from `random.Random(0)` no matter what `--seed` the run
            # used. This script follows baseline_probe.py and draws from
            # `random.Random(args.seed)`. The two agree only at seed 0 -- which
            # every recorded arm so far used -- so a nonzero-seed run gets the
            # right PROTEINS from `eval_indices` but a different spatial ball
            # inside each of them, which is the same uncomparability in a
            # smaller place.
            print(f"  !! WARNING: checkpoint seed is {args.seed}, not 0. "
                  f"train_growth.score() draws eval crops from "
                  f"random.Random(0) regardless of --seed, so the crop "
                  f"WINDOWS here will not match that run's even though the "
                  f"structures do. Pass --seed 0 to match the trainer's eval, "
                  f"or treat this floor as same-proteins-different-windows.")
        recorded_indices = [int(i) for i in (blob.get("eval_indices") or [])]
        if not recorded_indices:
            missing.append("eval_indices")
        if missing:
            # Loud, named, and never a silent fall-through to defaults: a floor
            # that quietly reshuffled is exactly the defect this argument
            # exists to remove, and it would look identical in the log.
            print()
            print("  !! WARNING: checkpoint is MISSING "
                  f"{', '.join(sorted(set(missing)))}")
            print("  !! Those values come from this script's own CLI/defaults "
                  "instead.")
            if "eval_indices" in missing:
                print("  !! THE CROPS ARE A RESHUFFLE, NOT THE RUN'S HELD-OUT "
                      "SET. This floor is not comparable to that run.")
            print("  !! Checkpoints written by scripts/train_growth.py carry "
                  "all of these; one that does not is from another trainer.")
        print()

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
    if recorded_indices:
        # The run's OWN held-out crops. `train_growth.score` walks
        # `eval_indices` in this order with one shared rng, which is the
        # convention this loop below now follows, so the windows match.
        indices = recorded_indices[: args.structures]
        provenance = "checkpoint eval_indices"
    else:
        order = list(range(len(pool)))
        random.Random(args.seed).shuffle(order)
        indices = order[: args.eval_structures][: args.structures]
        provenance = "reshuffle of this script's own pool slice"
    # Which structures, not just how many -- the same trap baseline_probe.py
    # documents: `available[:412]` is a different 412 on the laptop and on
    # Bunya, and that difference put the helix baseline at 10.77 locally
    # against 12.23 on the cluster.
    if args.checkpoint:
        print(f"checkpoint: {checkpoint_label}")
    print(f"{len(pool)} structures on disk, {len(indices)} held out at "
          f"crop {args.crop_size}, window {args.damage}, start {args.initial!r}")
    in_range = max(indices, default=-1) < len(pool)
    ids = [pool[i].pdb_id for i in indices] if in_range else "<out of range>"
    print(f"pool: {len(available)} available; held-out ids {ids}")
    # Which crops, and WHERE THEY CAME FROM. A floor on a reshuffle and a floor
    # on a run's own held-out set print the same table and mean different
    # things, so the two cases have to be distinguishable in the log.
    print(f"crops: {provenance}; indices {list(indices)}")
    if max(indices, default=-1) >= len(pool):
        print("  !! WARNING: an index is past the end of this pool "
              f"({len(pool)} entries). The checkpoint was trained where more "
              f"structures were on disk; those crops CANNOT be measured here "
              f"and the comparison is void. Run this where the run ran.")
        return 1
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
    # **ONE rng for every crop, advanced across them -- baseline_probe.py's
    # convention and the trainer's.** This loop used to pass a freshly seeded
    # `random.Random(args.seed + index)` per crop. The rng is what `crop()`
    # draws the SEED RESIDUE from, and the crop is a spatial ball around it, so
    # a fresh stream put the window in a different region of the same protein
    # than every recorded baseline and arm number was measured on: same
    # held-out ids, different 128-residue balls, different damage windows. On
    # the laptop pool at --damage 12 that read 9.21 A of helix start where
    # baseline_probe reads 10.77 A. Measured over 46 crops the two conventions
    # agree inside their standard error (12.11 vs 11.41, se 0.8), so the fresh
    # stream was not an easier task -- it was an uncomparable sample, which for
    # a floor quoted against an arm's start state is the same problem.
    rng = random.Random(args.seed)
    for index in indices:
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
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
