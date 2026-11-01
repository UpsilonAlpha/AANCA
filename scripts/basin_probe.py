"""Is the 22.7 % wrong-well rate an error, or the shape of the landscape?

    python scripts/basin_probe.py descent  runs/v5_native/best.pt --casp 13
    python scripts/basin_probe.py bestof   runs/v5_native/best.pt --seeds 8
    python scripts/basin_probe.py sweep    runs/v5_native/best.pt
    python scripts/basin_probe.py converge runs/v5_native/best.pt --limit 8

`scripts/chi1_histogram.py` established what the CASP13 chi1 row is made of:
the rule puts side chains on real rotamer wells and picks the native well
77.3 % of the time, so the deficit against DiffPack is a well-selection rate
and nothing else. That measurement says nothing about *why* a well is picked.

Two readings survive it, and they recommend opposite next moves.

**A classifier that is wrong 22.7 % of the time.** The rule is trying to name
the native rotamer from the backbone and its neighbourhood, and it misses.
Then the fix is capacity, data, or co-solving, and the number is a defect.

**A forcefield with basins.** The rule is descending a learned energy, and from
a uniformly randomised start it lands in whichever basin it was dropped into.
Then 22.7 % is a basin-of-attraction measurement under a uniform prior over
starting angles -- a property of the landscape, not a mistake -- and the
published protocol is measuring something other than what the table's column
heading claims.

These are distinguishable, cheaply, with no retraining, because
``aanca.casp.randomised_start`` derives the starting structure deterministically
from the seed: the thing the rule was dropped into is recoverable exactly.

**descent** -- the transition matrix from starting well to predicted well. Pure
descent would leave almost every residue in the well it started in, and would
be right only as often as the start happened to be right (about a third, since
a uniform draw puts a third of residues in each 120-degree arc). Pure
classification would make the predicted well independent of the start. The
measured matrix sits somewhere between, and where it sits is the answer.

**bestof** -- the same residues packed from several independent starts. If the
native well is reachable from *some* start, the rule knows more than one
deterministic descent reports. Three selectors are scored: a single seed (the
published protocol), the modal well across seeds (free -- it needs no energy
function and no native answer), and an oracle that takes any seed that was
right (an upper bound, not a method). Unanimity across seeds is also reported
as a confidence signal, since a residue every start agrees on is one with a
single basin.

**sweep** -- well-match rate against perturbation magnitude, from mild von
Mises noise to the uniform draw. The basin-size curve. The start's own
well-match rate is reported in the same row, so the gap between what was handed
to the rule and what it returned is readable directly.

**converge** -- the two numbers the prior-art search called this work's own and
which have never been measured here: does the proposed update decay or
oscillate, and does the structure hold when the rollout runs ten times longer
than the benchmark's. Segmenting the rollout with one persistent generator
visits exactly the states an unsegmented run would, so the step-100 row is the
benchmark's own structure and the drift column is measured against it.

**Proline is excluded throughout**, as in the histogram: its chi1 is pinned by
the pyrrolidine ring to about +/-25 degrees, so the three-well model does not
describe it.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.casp import REFERENCE, RESULTS, pack_from_random, randomised_start, targets_for
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.rotamers import WELL_NAMES, nearest_well
from aanca.structure import load_protein
from aanca.torch_geometry import measure_chis, wrap_to_pi
from aanca.torch_graph import feature_topology

#: Offset between seed replicates in ``bestof``. Large enough that replicate 0
#: with ``--seed 0`` reproduces the benchmark's own stream exactly, and no
#: replicate ever collides with another target's seed.
SEED_STRIDE = 100_000


def chi1_degrees(coords: torch.Tensor, torsion) -> torch.Tensor:
    """chi1 of every residue, in degrees, on the CPU."""
    return torch.rad2deg(measure_chis(coords, torsion)[:, 0]).cpu()


def chi1_residues(torsion) -> torch.Tensor:
    """Residues with a real chi1 that the three-well model describes."""
    has_chi1 = torsion.chi_mask[:, 0].cpu()
    is_pro = torch.tensor(
        [name == "PRO" for name in torsion.resnames], dtype=torch.bool
    )
    return has_chi1 & ~is_pro


def circular_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``|predicted - target|`` in degrees, by the shortest arc."""
    return wrap_to_pi(torch.deg2rad(predicted - target)).rad2deg().abs()


def each_target(stems, backbone: str, device: str):
    """``(index, stem, topo, native_coords)``, skipping unreadable files.

    The index is the seed offset, so it counts every stem whether or not the
    file loaded -- a target that starts failing to parse must not shift the
    seeds of the targets after it, or two runs of this script stop being
    comparable.
    """
    for index, stem in enumerate(stems):
        try:
            protein = load_protein(
                RESULTS / "native" / backbone / f"{stem}.pdb", repair_columns=True
            )
            topo, native_coords = feature_topology(protein, device=device)
        except Exception:  # malformed or unreadable; benchmark.py reports these
            continue
        yield index, stem, topo, native_coords


def percent(numerator: torch.Tensor) -> float:
    return 100.0 * float(numerator.float().mean())


# --------------------------------------------------------------------------
# 1. descent: does the rule stay in the well it was dropped into?
# --------------------------------------------------------------------------


def run_descent(rule, config, stems, args) -> int:
    starts, predictions, natives = [], [], []
    for index, _stem, topo, native_coords in each_target(
        stems, args.backbone, args.device
    ):
        torsion = topo.torsion
        keep = chi1_residues(torsion)
        seed = args.seed + index
        # Recomputed rather than threaded out of ``pack_from_random``: one
        # definition of the starting structure, so these two cannot drift
        # apart. The cost is one extra rigid rotation per target.
        start = randomised_start(native_coords, topo, seed)
        packed = pack_from_random(rule, native_coords, topo, config, seed)
        starts.append(chi1_degrees(start, torsion)[keep])
        predictions.append(chi1_degrees(packed, torsion)[keep])
        natives.append(chi1_degrees(native_coords, torsion)[keep])

    if not starts:
        print("no usable targets")
        return 1

    start_deg = torch.cat(starts)
    pred_deg = torch.cat(predictions)
    native_deg = torch.cat(natives)

    start_well, _ = nearest_well(start_deg)
    pred_well, _ = nearest_well(pred_deg)
    native_well, _ = nearest_well(native_deg)

    n = start_well.numel()
    stayed = pred_well == start_well
    start_right = start_well == native_well
    pred_right = pred_well == native_well

    print(f"\n{n} residues with a chi1, prolines excluded\n")
    print("--- how often each is right ---\n")
    print(f"  the start we handed the rule : {percent(start_right):5.1f}%")
    print(f"  the rule's answer            : {percent(pred_right):5.1f}%")
    print(f"  the rule stayed where it started : {percent(stayed):5.1f}%\n")

    print("--- the transition matrix: where it started, where it ended ---\n")
    header = "  start \\ predicted  " + "".join(f"{name:>8}" for name in WELL_NAMES)
    print(header + "       n")
    for i, name in enumerate(WELL_NAMES):
        row = start_well == i
        total = int(row.sum())
        cells = ""
        for j in range(len(WELL_NAMES)):
            share = 100.0 * float((pred_well[row] == j).float().mean()) if total else 0.0
            cells += f"{share:7.1f}%"
        print(f"  {name:<18}" + cells + f"  {total:7d}")

    print("\n--- the number that decides it ---\n")
    print(
        f"  right when the start was already right "
        f"({int(start_right.sum())} residues): {percent(pred_right[start_right]):5.1f}%"
    )
    print(
        f"  right when the start was wrong         "
        f"({int((~start_right).sum())} residues): {percent(pred_right[~start_right]):5.1f}%"
    )
    wrong = ~pred_right
    if bool(wrong.any()):
        print(
            f"\n  of the {int(wrong.sum())} wrong answers, "
            f"{percent(stayed[wrong]):5.1f}% stayed in the well they started in"
        )
    print(
        f"  of the {int(pred_right.sum())} right answers, "
        f"{percent(stayed[pred_right]):5.1f}% stayed in the well they started in"
    )

    print(
        "\nReading it: pure descent leaves everything where it started, so"
        "\n'stayed' would be near 100% and the answer would be right only as"
        "\noften as the start was -- about a third. Pure classification makes"
        "\nthe predicted well independent of the start, so every row of the"
        "\nmatrix would be the same and would reproduce the native well"
        "\nfrequencies (g- 50.2%, t 32.9%, g+ 16.9%). The gap between the two"
        "\nconditional rates above is how much the start still matters once"
        "\nthe rule has run."
    )
    return 0


# --------------------------------------------------------------------------
# 2. bestof: is the native well reachable from some start?
# --------------------------------------------------------------------------


def run_bestof(rule, config, stems, args) -> int:
    columns: list[list[torch.Tensor]] = [[] for _ in range(args.seeds)]
    natives: list[torch.Tensor] = []
    for index, _stem, topo, native_coords in each_target(
        stems, args.backbone, args.device
    ):
        torsion = topo.torsion
        keep = chi1_residues(torsion)
        natives.append(chi1_degrees(native_coords, torsion)[keep])
        for replicate in range(args.seeds):
            seed = args.seed + index + SEED_STRIDE * replicate
            packed = pack_from_random(rule, native_coords, topo, config, seed)
            columns[replicate].append(chi1_degrees(packed, torsion)[keep])

    if not natives:
        print("no usable targets")
        return 1

    native_deg = torch.cat(natives)
    pred_deg = torch.stack([torch.cat(column) for column in columns])  # (S, R)
    native_well, _ = nearest_well(native_deg)
    pred_well = torch.stack([nearest_well(row)[0] for row in pred_deg])  # (S, R)
    errors = torch.stack([circular_error(row, native_deg) for row in pred_deg])

    right = pred_well == native_well  # (S, R)
    counts = torch.stack(
        [(pred_well == w).sum(dim=0) for w in range(len(WELL_NAMES))]
    )  # (W, R)
    consensus = counts.argmax(dim=0)
    unanimous = counts.max(dim=0).values == args.seeds

    print(f"\n{native_well.numel()} residues, {args.seeds} independent starts\n")
    print("--- well-match rate, by how the answer is chosen ---\n")
    per_seed = right.float().mean(dim=1) * 100
    print(
        f"  one seed, the published protocol : {per_seed.mean():5.1f}% mean "
        f"({per_seed.min():.1f} to {per_seed.max():.1f} across seeds)"
    )
    print(
        f"  modal well across seeds          : "
        f"{percent(consensus == native_well):5.1f}%   (no native answer needed)"
    )
    print(
        f"  any seed right                   : "
        f"{percent(right.any(dim=0)):5.1f}%   (an oracle, not a method)"
    )
    print(
        f"  every seed right                 : {percent(right.all(dim=0)):5.1f}%"
    )

    print("\n--- chi1 MAE, degrees ---\n")
    per_seed_mae = errors.mean(dim=1)
    print(
        f"  one seed  : {per_seed_mae.mean():6.2f} mean "
        f"({per_seed_mae.min():.2f} to {per_seed_mae.max():.2f})"
    )
    print(f"  best seed per residue : {errors.min(dim=0).values.mean():6.2f}  (oracle)")

    print("\n--- unanimity as a confidence signal ---\n")
    print(
        f"  all {args.seeds} starts agree on the well : "
        f"{percent(unanimous):5.1f}% of residues"
    )
    if bool(unanimous.any()):
        print(
            f"    and there the first seed is right     : "
            f"{percent(right[0][unanimous]):5.1f}%"
        )
    if bool((~unanimous).any()):
        print(
            f"    where they disagree, right            : "
            f"{percent(right[0][~unanimous]):5.1f}%"
        )

    print(
        "\nReading it: a large gap between one seed and 'any seed right' means"
        "\nthe native well is in reach and the single-shot number is a sampling"
        "\nartefact. A large gap between one seed and the modal well means that"
        "\nartefact is recoverable for free, since voting needs no energy"
        "\nfunction. If unanimity separates right from wrong, the rule has a"
        "\nusable confidence estimate it is not currently being asked for."
    )
    return 0


# --------------------------------------------------------------------------
# 3. sweep: recovery against how badly the start was damaged
# --------------------------------------------------------------------------


def run_sweep(rule, config, stems, args) -> int:
    print(
        f"\nkappa is a von Mises concentration, so larger is milder; "
        f"0 is the uniform draw.\n"
    )
    print(
        "   kappa   start right   rule right    start MAE    rule MAE   residues"
    )
    for kappa in args.kappa:
        starts, predictions, natives = [], [], []
        for index, _stem, topo, native_coords in each_target(
            stems, args.backbone, args.device
        ):
            torsion = topo.torsion
            keep = chi1_residues(torsion)
            seed = args.seed + index
            start = randomised_start(native_coords, topo, seed, kappa)
            packed = pack_from_random(rule, native_coords, topo, config, seed, kappa)
            starts.append(chi1_degrees(start, torsion)[keep])
            predictions.append(chi1_degrees(packed, torsion)[keep])
            natives.append(chi1_degrees(native_coords, torsion)[keep])

        if not natives:
            print("no usable targets")
            return 1

        start_deg = torch.cat(starts)
        pred_deg = torch.cat(predictions)
        native_deg = torch.cat(natives)
        native_well, _ = nearest_well(native_deg)
        start_ok = nearest_well(start_deg)[0] == native_well
        pred_ok = nearest_well(pred_deg)[0] == native_well
        print(
            f"  {kappa:6.2f}   {percent(start_ok):10.1f}%  {percent(pred_ok):10.1f}%"
            f"   {circular_error(start_deg, native_deg).mean():9.2f}"
            f"   {circular_error(pred_deg, native_deg).mean():9.2f}"
            f"   {native_well.numel():8d}"
        )

    print(
        "\nReading it: the gap between the two percentage columns is what the"
        "\nrule adds to the structure it was handed. A gap that holds up as"
        "\nkappa falls to 0 is a wide basin; one that collapses means the rule"
        "\nonly repairs starts that were nearly right already."
    )
    return 0


# --------------------------------------------------------------------------
# 4. converge: does it settle, and does it stay settled?
# --------------------------------------------------------------------------


def run_converge(rule, config, stems, args) -> int:
    checkpoints = sorted(set(args.checkpoints))
    reference = 100 if 100 in checkpoints else checkpoints[0]
    at_step: dict[int, list[torch.Tensor]] = {step: [] for step in checkpoints}
    natives: list[torch.Tensor] = []
    curves: list[list[float]] = []

    for index, _stem, topo, native_coords in each_target(
        stems, args.backbone, args.device
    ):
        torsion = topo.torsion
        keep = chi1_residues(torsion)
        natives.append(chi1_degrees(native_coords, torsion)[keep])
        seed = args.seed + index
        start = randomised_start(native_coords, topo, seed)
        # One generator across every segment. ``rollout`` draws one mask per
        # step from it in order, so the state at cumulative step N is the one
        # an unsegmented N-step rollout would reach -- which is what makes the
        # step-100 row below the benchmark's own structure.
        generator = torch.Generator().manual_seed(seed)
        state = initial_state(start, rule)
        curve: list[float] = []
        reached = 0
        for step in checkpoints:
            segment = dataclasses.replace(config, steps=step - reached)
            result = rollout(
                rule, state, topo, segment, generator=generator, record=True
            )
            state = result.state
            curve.extend(result.mean_abs_delta)
            reached = step
            at_step[step].append(chi1_degrees(state.coords, torsion)[keep])
        curves.append(curve)

    if not natives:
        print("no usable targets")
        return 1

    native_deg = torch.cat(natives)
    native_well, _ = nearest_well(native_deg)
    reference_deg = torch.cat(at_step[reference])

    print(f"\n{native_well.numel()} residues over {len(curves)} targets\n")
    print(f"--- the structure at each rollout length ---\n")
    print(
        f"   steps   well right     chi1 MAE   drift from step {reference}"
    )
    for step in checkpoints:
        pred_deg = torch.cat(at_step[step])
        pred_ok = nearest_well(pred_deg)[0] == native_well
        drift = circular_error(pred_deg, reference_deg).mean()
        print(
            f"  {step:6d}   {percent(pred_ok):9.1f}%   {circular_error(pred_deg, native_deg).mean():10.2f}"
            f"   {drift:18.2f}"
        )

    # The curves are per-target and all the same length, so a mean over targets
    # is the convergence curve of the rule rather than of one protein.
    width = min(len(curve) for curve in curves)
    mean_curve = (
        torch.tensor([curve[:width] for curve in curves]).mean(dim=0).rad2deg()
    )
    print("\n--- proposed |delta chi| per step, degrees, mean over targets ---\n")
    print("  (the rule's output, before the asynchronous mask gates it)\n")
    shown = [s for s in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 400, 700, 1000) if s <= width]
    # A perfectly silent rule is a real outcome, not an impossible one: an
    # untrained rule is the identity map by construction, and a converged one
    # can get close. Dividing by the peak to scale the bars must not be the
    # thing that ends a thousand-step job.
    peak = float(mean_curve.max()) or 1.0
    for step in shown:
        value = float(mean_curve[step - 1])
        bar = "#" * round(48 * value / peak)
        print(f"  step {step:5d} | {bar:<48} {value:7.3f}")

    tail = mean_curve[max(width - 100, 0) :]
    early = mean_curve[max(width // 2 - 5, 0) : width // 2 + 5]
    swing = float((tail[1:] - tail[:-1]).abs().max()) if tail.numel() > 1 else 0.0
    print(
        f"\n  mean over the last 100 steps : {float(tail.mean()):7.3f} deg"
        f"\n  mean around the midpoint     : {float(early.mean()):7.3f} deg"
        f"\n  largest step-to-step change in the last 100 : {swing:7.3f} deg"
    )
    print(
        "\nReading it: a curve decaying toward zero with a small final swing is"
        "\na fixed point. A curve that flattens at a non-zero value is a rule"
        "\nstill pushing at convergence, held in place only by the mask; a"
        "\nlarge final swing is the oscillation some Graph NCA rules show. The"
        "\ndrift column is the stability claim -- it should stay near zero well"
        "\npast the length the rule was trained at."
    )
    return 0


MEASUREMENTS = {
    "descent": run_descent,
    "bestof": run_bestof,
    "sweep": run_sweep,
    "converge": run_converge,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", choices=sorted(MEASUREMENTS))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--casp", type=int, default=13, choices=[13, 14])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backbone", default="casp_all")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N targets only")
    parser.add_argument(
        "--seeds", type=int, default=8, help="bestof: independent starts per residue"
    )
    parser.add_argument(
        "--kappa",
        type=float,
        nargs="+",
        default=[16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.0],
        help="sweep: von Mises concentrations, larger is milder",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50, 100, 200, 400, 700, 1000],
        help="converge: cumulative rollout lengths to measure at",
    )
    args = parser.parse_args()

    if not RESULTS.is_dir():
        print(f"{RESULTS} missing -- run scripts/fetch_benchmark.py first")
        return 1

    args.device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    state = ckpt.load(args.checkpoint, map_location=args.device)
    stored = state.get("args") or {}
    rule = AANCARule().to(args.device)
    rule.load_state_dict(ckpt.rule_weights(state))
    rule.eval()
    config = RolloutConfig(
        steps=args.steps,
        k=stored.get("k", 64),
        cutoff=10.0,
        device_neighbours=args.device != "cpu",
    )

    # The same target set the published row is averaged over.
    covered = [m for m in REFERENCE[args.casp] if (RESULTS / m / args.backbone).is_dir()]
    stems = targets_for(RESULTS, args.casp, covered, args.backbone)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        print(f"no CASP{args.casp} targets covered by all of {covered}")
        return 1

    print(
        f"{args.measurement}: {args.checkpoint} at update {state.get('update', '?')}, "
        f"CASP{args.casp}, {len(stems)} targets, {args.steps} steps, {args.device}"
    )

    with torch.no_grad():
        return MEASUREMENTS[args.measurement](rule, config, stems, args)


if __name__ == "__main__":
    raise SystemExit(main())
