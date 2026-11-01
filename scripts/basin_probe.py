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
from aanca.diagnostics import (
    chi1_degrees,
    chi1_residues,
    circular_error,
    each_target,
    percent,
)
from aanca.metrics import BURIAL_RADIUS, cb_neighbour_counts, classify_burial
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.rotamers import WELL_NAMES, nearest_well
from aanca.torch_graph import frame_atom_coords

#: Offset between seed replicates in ``bestof``. Large enough that replicate 0
#: with ``--seed 0`` reproduces the benchmark's own stream exactly, and no
#: replicate ever collides with another target's seed.
SEED_STRIDE = 100_000


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


# --------------------------------------------------------------------------
# 5. stationary: handed the right answer, does it keep it?
# --------------------------------------------------------------------------


def run_stationary(rule, config, stems, args) -> int:
    """Start at the crystal structure and see what the rule breaks.

    `sweep` found the rule is net-destructive in well terms at low damage --
    at kappa 16 it takes a start that is 99.5 % right down to 97.7 % -- which
    says the native structure is not a fixed point of the rule. That matters
    because v5 set ``--noop-weight 0`` and leaned on ``--native-probability``
    to supervise stationarity instead, after ``scripts/noop_probe.py`` showed
    the no-op term was vacuous at step 0. This is the rollout version of that
    question rather than the one-step version, and kappa 16 only approximates
    it: here the start *is* native, so every break is unambiguous.
    """
    broken, kept, names, burials = [], [], [], []
    curves = []
    for index, _stem, topo, native_coords in each_target(
        stems, args.backbone, args.device
    ):
        torsion = topo.torsion
        scored = chi1_residues(torsion)
        generator = torch.Generator().manual_seed(args.seed + index)
        result = rollout(
            rule,
            initial_state(native_coords, rule),
            topo,
            config,
            generator=generator,
            record=True,
        )
        native_deg = chi1_degrees(native_coords, torsion)
        final_deg = chi1_degrees(result.state.coords, torsion)
        native_well = nearest_well(native_deg)[0]
        final_well = nearest_well(final_deg)[0]

        broke = (native_well != final_well)[scored]
        broken.append(broke)
        kept.append(circular_error(final_deg, native_deg)[scored])
        names.extend(
            name for name, keep in zip(torsion.resnames, scored.tolist()) if keep
        )
        counts = cb_neighbour_counts(native_coords, topo).cpu()
        burials.append(classify_burial(counts)["core"][scored])
        curves.append(result.mean_abs_delta)

    if not broken:
        print("no usable targets")
        return 1

    broke = torch.cat(broken)
    error = torch.cat(kept)
    core = torch.cat(burials)

    print(f"\n{broke.numel()} residues started at their crystal chi1\n")
    print("--- what a rollout from native costs ---\n")
    print(f"  wells broken        : {percent(broke):5.2f}%  ({int(broke.sum())} residues)")
    print(f"  chi1 MAE from native: {float(error.mean()):6.2f} deg")
    print(f"  within 20 deg       : {percent(error <= 20):5.1f}%")
    print(f"\n  broken, core surface split : core {percent(broke[core]):5.2f}%, "
          f"non-core {percent(broke[~core]):5.2f}%")

    print("\n--- which residues lose their well ---\n")
    by_name: dict[str, list[int]] = {}
    for name, flag in zip(names, broke.tolist()):
        row = by_name.setdefault(name, [0, 0])
        row[0] += int(flag)
        row[1] += 1
    ranked = sorted(
        ((n, b, t) for n, (b, t) in by_name.items() if t >= 100),
        key=lambda r: -r[1] / r[2],
    )
    for name, bad, total in ranked[:8]:
        print(f"  {name}  {100 * bad / total:5.2f}%  of {total:5d}")

    width = min(len(c) for c in curves)
    mean_curve = torch.tensor([c[:width] for c in curves]).mean(dim=0).rad2deg()
    print("\n--- proposed |delta chi| per step from native, degrees ---\n")
    for step in [s for s in (1, 2, 5, 10, 20, 50, 100) if s <= width]:
        print(f"  step {step:4d} : {float(mean_curve[step - 1]):7.3f}")

    print(
        "\nReading it: a rule whose attractor is the native state would break"
        "\nalmost no wells and sit near zero MAE. Whatever it breaks here is the"
        "\nsame failure the kappa 16 row of `sweep` shows, measured without the"
        "\nambiguity of a perturbed start, and it is a training problem rather"
        "\nthan a capacity one -- the fixed point at native is what"
        "\n``--native-probability`` was meant to supervise."
    )
    return 0


# --------------------------------------------------------------------------
# 6. breakdown: who are the 22.7 %?
# --------------------------------------------------------------------------


def neighbour_wrong_share(
    near: torch.Tensor, wrong: torch.Tensor, eligible: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(share, valid)``: the wrong-well share of each residue's neighbours.

    ``valid`` marks residues with at least one eligible neighbour. They have
    to be excluded rather than counted as 0.0, because a residue the statistic
    cannot speak about would otherwise drag both conditional means toward
    zero -- and unevenly, since isolated residues are surface residues and
    surface residues are the wrong ones more often.
    """
    selected = near & eligible.unsqueeze(0)
    total = selected.sum(dim=1)
    share = (selected & wrong.unsqueeze(0)).sum(dim=1) / total.clamp_min(1)
    return share, total > 0


def run_breakdown(rule, config, stems, args) -> int:
    """Split the wrong-well rate by burial, by residue type, and spatially.

    The third is the one with teeth. If wrong-well residues sit next to other
    wrong-well residues more often than chance, the failure is cooperative --
    an early wrong commitment propagating through the neighbourhood, which is
    the co-solving story. If they are spatially independent, each residue is
    failing on its own and the rule is closer to a per-residue lookup.
    """
    flags, names, cores = [], [], []
    # One (share, valid, wrong) triple per stratum, so the spatial statistic
    # can be read within a burial class as well as over everything.
    strata: dict[str, list] = {"all": [], "core": [], "non-core": []}
    for index, _stem, topo, native_coords in each_target(
        stems, args.backbone, args.device
    ):
        torsion = topo.torsion
        scored = chi1_residues(torsion)
        packed = pack_from_random(
            rule, native_coords, topo, config, args.seed + index
        )
        native_well = nearest_well(chi1_degrees(native_coords, torsion))[0]
        pred_well = nearest_well(chi1_degrees(packed, torsion))[0]
        wrong = native_well != pred_well

        counts = cb_neighbour_counts(native_coords, topo).cpu()
        core_mask = classify_burial(counts)["core"].cpu()

        # Neighbours by CB distance at the rule's own cutoff.
        cb = frame_atom_coords(native_coords, topo)[:, 4, :].cpu()
        near = torch.cdist(cb, cb) <= BURIAL_RADIUS
        near.fill_diagonal_(False)

        groups = {
            "all": torch.ones_like(scored),
            "core": core_mask,
            "non-core": ~core_mask,
        }
        for name, group in groups.items():
            eligible = scored & group
            share, valid = neighbour_wrong_share(near, wrong, eligible)
            # The subject is restricted to the same class as its neighbours,
            # so burial cannot produce the correlation on its own.
            subject = eligible & valid
            strata[name].append((share[subject], wrong[subject]))

        flags.append(wrong[scored])
        names.extend(
            name for name, keep in zip(torsion.resnames, scored.tolist()) if keep
        )
        cores.append(core_mask[scored])

    if not flags:
        print("no usable targets")
        return 1

    wrong = torch.cat(flags)
    core = torch.cat(cores)
    base = percent(wrong)

    print(f"\n{wrong.numel()} residues, {base:.1f}% in the wrong well overall\n")
    print("--- by burial ---\n")
    print(f"  core ({int(core.sum())} residues)      : {percent(wrong[core]):5.1f}% wrong")
    print(f"  non-core ({int((~core).sum())} residues) : {percent(wrong[~core]):5.1f}% wrong")

    print("\n--- by residue type, worst first, types with 100+ residues ---\n")
    by_name: dict[str, list[int]] = {}
    for name, flag in zip(names, wrong.tolist()):
        row = by_name.setdefault(name, [0, 0])
        row[0] += int(flag)
        row[1] += 1
    ranked = sorted(
        ((n, b, t) for n, (b, t) in by_name.items() if t >= 100),
        key=lambda r: -r[1] / r[2],
    )
    for name, bad, total in ranked:
        print(f"  {name}  {100 * bad / total:5.1f}%  of {total:5d}")

    print("\n--- spatially: is the failure cooperative? ---\n")
    print(
        "  Share of a residue's chi1 neighbours within 10 A that are also in"
        "\n  the wrong well. Stratified because burial predicts wrongness AND is"
        "\n  spatially clumped, so the 'all' row cannot tell a cooperative"
        "\n  failure from two surface residues happening to be neighbours. In"
        "\n  the core and non-core rows both the subject and its neighbours are"
        "\n  restricted to that class, which removes the confound.\n"
    )
    print("  stratum     self wrong   self right   difference        n")
    for name in ("all", "core", "non-core"):
        share = torch.cat([s for s, _w in strata[name]])
        flag = torch.cat([w for _s, w in strata[name]])
        if not bool(flag.any()) or not bool((~flag).any()):
            continue
        bad = 100 * float(share[flag].mean())
        good = 100 * float(share[~flag].mean())
        print(
            f"  {name:<10}  {bad:9.1f}%  {good:10.1f}%  {bad - good:+10.1f}"
            f"  {flag.numel():7d}"
        )
    print(f"\n  base rate over everything : {base:5.1f}%")
    print(
        "\nReading it: a difference near zero in the *stratified* rows means"
        "\nwrong wells are spatially independent and each residue fails alone."
        "\nA positive difference that survives stratification is a cooperative"
        "\nfailure -- an early wrong commitment propagating -- which is the"
        "\nco-solving story and would make the clamp test's native arm the"
        "\nplace to look for the fix. The identity rule gives this statistic a"
        "\nflat null, since its errors are independent uniform draws."
    )
    return 0


# --------------------------------------------------------------------------
# 7. hbond: is the SER/THR deficit a missing-physics problem?
# --------------------------------------------------------------------------

#: Protein atoms that can donate or accept a hydrogen bond.
POLAR_ELEMENTS = ("N", "O")

#: Generous heavy-atom donor-acceptor distance. Nothing here depends on the
#: exact value; 3.5 A is the usual cutoff and the buckets are coarse.
HBOND_CUTOFF = 3.5


def run_hbond(rule, config, stems, args) -> int:
    """Split the wrong-well rate by whether the atom chi1 places can H-bond.

    `breakdown` found the error is polar: SER 43.7 %, GLU 33.5 %, ASP 28.5 %
    against PHE 9.2 %, TYR 12.7 %, ILE 13.0 %. SER and THR alone are 24.3 % of
    all chi1 error from 16.4 % of the residues. The obvious reading is that an
    aromatic's chi1 is pinned by what it collides with -- which distances and
    relative orientations describe well -- while a serine's is pinned by where
    its hydroxyl can hydrogen bond, which nothing in the feature set names.

    That reading makes a prediction, and the prediction can go either way,
    which is what makes it worth running. The atom chi1 places is the fourth
    of its four defining atoms: ``OG`` for SER, ``OG1`` for THR, ``SG`` for
    CYS, ``CG`` for LEU. Count the N and O atoms of *other* residues near it
    in the crystal structure, and split the error by that count.

    If hydrogen-bonded hydroxyls are **worse**, their chi1 is set by a partner
    the rule cannot recognise as a partner, and the fix is a feature rather
    than more parameters. If they are the **same or better**, the rule is
    already reading the acceptor geometry, the hydrogen-bonding story is wrong,
    and SER is simply small and under-determined -- in which case some of that
    43.7 % is not recoverable by any method, and the honest move is to check
    what the baselines score on SER before treating it as our deficit.
    """
    flags, partners, polar_tip, names = [], [], [], []
    for index, _stem, topo, native_coords in each_target(
        stems, args.backbone, args.device
    ):
        torsion = topo.torsion
        scored = chi1_residues(torsion)
        packed = pack_from_random(
            rule, native_coords, topo, config, args.seed + index
        )
        native_well = nearest_well(chi1_degrees(native_coords, torsion))[0]
        pred_well = nearest_well(chi1_degrees(packed, torsion))[0]
        wrong = (native_well != pred_well)[scored]

        coords = native_coords.cpu()
        tip_index = torsion.chi_atoms[:, 0, 3].cpu()
        tips = coords[torch.arange(coords.shape[0]), tip_index]

        # First character of a protein atom name is its element: no side-chain
        # atom of a standard residue has a two-letter symbol.
        tip_is_polar, residue_of, polar_xyz = [], [], []
        for r, row in enumerate(torsion.atom_names):
            tip_is_polar.append(row[int(tip_index[r])][:1] in POLAR_ELEMENTS)
            for a, name in enumerate(row):
                if name[:1] in POLAR_ELEMENTS and bool(torsion.atom_mask[r, a]):
                    residue_of.append(r)
                    polar_xyz.append(coords[r, a])

        owner = torch.tensor(residue_of)
        near = torch.cdist(tips.float(), torch.stack(polar_xyz).float())
        other = owner.unsqueeze(0) != torch.arange(tips.shape[0]).unsqueeze(1)
        count = ((near <= HBOND_CUTOFF) & other).sum(dim=1)

        flags.append(wrong)
        partners.append(count[scored])
        polar_tip.append(torch.tensor(tip_is_polar)[scored])
        names.extend(
            name for name, keep in zip(torsion.resnames, scored.tolist()) if keep
        )

    if not flags:
        print("no usable targets")
        return 1

    wrong = torch.cat(flags)
    count = torch.cat(partners)
    polar = torch.cat(polar_tip)
    names_t = names

    print(
        f"\n{wrong.numel()} residues. The atom chi1 places is polar (N or O) in"
        f"\n{int(polar.sum())} of them.\n"
    )
    print(
        "  Scope: the polar group is essentially SER and THR, because they are"
        "\n  the only types whose chi1 places an N or O. ASP, GLU, ASN and GLN"
        "\n  are polar residues but chi1 places their CG, and chi2 positions the"
        "\n  atoms that hydrogen bond -- so this test says nothing about them,"
        "\n  and they are another 31.3% of the error. CYS is excluded too: its"
        "\n  SG is a sulphur, which these element buckets do not count.\n"
    )
    print(f"--- wrong-well rate by nearby N/O of other residues, within {HBOND_CUTOFF} A ---\n")
    print("  group                 0 partners   1   2 or more        n")
    for label, subset in (
        ("polar tip", polar),
        ("non-polar tip", ~polar),
        ("SER or THR", torch.tensor([n in ("SER", "THR") for n in names_t])),
    ):
        if not bool(subset.any()):
            continue
        cells = ""
        for bucket in (count == 0, count == 1, count >= 2):
            pick = subset & bucket
            cells += f"{percent(wrong[pick]):9.1f}%" if bool(pick.any()) else "        -"
        print(f"  {label:<18} {cells}  {int(subset.sum()):7d}")

    print("\n--- the same split, counts in each bucket ---\n")
    for label, subset in (
        ("polar tip", polar),
        ("non-polar tip", ~polar),
        ("SER or THR", torch.tensor([n in ("SER", "THR") for n in names_t])),
    ):
        sizes = [int((subset & b).sum()) for b in (count == 0, count == 1, count >= 2)]
        print(f"  {label:<18} {sizes[0]:9d} {sizes[1]:9d} {sizes[2]:11d}")

    print(
        "\nReading it: if the rate climbs with partner count among polar tips,"
        "\nthe rule is failing exactly where a hydrogen bond determines the"
        "\nanswer, and the missing ingredient is a feature rather than capacity."
        "\nIf it is flat or falls, the hydrogen-bonding story is wrong and SER"
        "\nis simply small and under-determined -- in which case compare"
        "\nagainst what the baselines score on SER before calling it ours."
        "\nThe non-polar row is the control: a CG carbon cannot hydrogen bond,"
        "\nso any trend there is crowding, not chemistry."
    )
    return 0


MEASUREMENTS = {
    "descent": run_descent,
    "bestof": run_bestof,
    "sweep": run_sweep,
    "converge": run_converge,
    "stationary": run_stationary,
    "breakdown": run_breakdown,
    "hbond": run_hbond,
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
