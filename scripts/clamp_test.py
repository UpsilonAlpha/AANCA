"""Does the rule read its neighbours' side chains, or only the backbone?

    python scripts/clamp_test.py runs/v5_native/best.pt --casp 13

`scripts/basin_probe.py descent` showed that the rule ignores its *own*
starting chi: the transition matrix's rows are its output distribution whatever
well it was dropped into, and a perfect start would move the chi1 error from
22.7 % to 20.9 %. That is a statement about one residue's own conformation. It
is **not** the same statement as "the rule ignores its neighbours' side
chains", and only this test separates them.

Why the difference matters. A weight-shared local rule on a residue graph is
the architecture's whole claim: each residue negotiates with its neighbourhood
and the fixed point is the mutually consistent packing. Well choice is exactly
the kind of problem that needs it, since which rotamer fits depends on what the
neighbours chose. But a cheaper function would fit the training loss just as
well. Rollouts start from fully randomised chis, so for most of training every
neighbour's side chain is wrong; if that input is mostly noise, the rule can do
no worse by learning the **per-residue marginal** -- the best well given the
backbone, the neighbourhood's identities and their CB positions, ignoring their
conformations. A rule that learned the marginal is a cellular automaton in form
and a lookup in function, and its dynamics would be decorative. Note that
77.3 % is about what a backbone-dependent rotamer library reaches from backbone
alone, so this is not a remote possibility.

**The experiment.** Pack from a randomised start as the benchmark does, but pin
a fraction of residues for the whole rollout and score only the ones left free.
If the rule reads neighbour conformations, giving it correct neighbours must
make the free residues better.

**The control is the point.** Pinning residues changes two things at once: it
supplies information *and* it quiets the neighbourhood, since fewer side chains
are moving. Those are separated by running two arms at the same fraction:

- ``native`` -- pinned residues held at their crystal conformation.
- ``random`` -- pinned residues held at an independent uniform draw, wrong and
  motionless.

The fraction frozen is identical, so the difference between the arms isolates
whether being *right* is what helped. A gain in both arms is a quieter
neighbourhood, not co-solving. A gain only in ``native`` is co-solving. No gain
in either, at any fraction, is the marginal predictor.

**Pinning gates geometry only** (``aanca.rollout.step_once``). A pinned residue
keeps updating its hidden state, because every message is built from the
*sender's* ``cat([node_features, hidden])`` -- freezing that too would remove a
channel and hand its neighbours an all-zero hidden vector at step 50, which
never happens in training.

**Proline is excluded from the scoring** as in the other diagnostics, but it is
still eligible to be pinned: it is part of the neighbourhood whether or not the
three-well model describes it.

---

**The silencing hypothesis, added 2026-11-02 after the 0.75 row.** At 0.75 the
``native`` arm scores 63.4 % against ``random``'s 75.3 %: being given *correct*
neighbours is twelve points *worse* than being given wrong ones. Both arms are
drawn with the same seed, so the pinned set and the scored set are identical
and the free residues start from the same conformations -- the only difference
is what the frozen quarter-to-three-quarters is holding. So this is not a
sampling artefact, and "the rule ignores its neighbours" does not explain it
either: ignoring them would give zero difference, not a large negative one.

The explanation that does fit is that the rule has learned **"a native-looking
neighbourhood means you are done"**. The no-op term supervises silence at the
native fixed point, and in training a neighbourhood only looks native near
convergence, when the residue itself is already right. Three quarters of the
structure sitting motionless at its crystal conformation is therefore read as a
converged state, and the rule stops moving a residue that is still at its
random start. The ``random`` arm does not trigger it, because a mostly-wrong
motionless field is what step 0 of every training rollout looks like.

That hypothesis makes a measurable prediction the accuracy columns cannot
distinguish from ordinary error, so this test now reports, over the same scored
free residues:

- ``start right`` -- the well-match rate of the randomised *start* itself, the
  floor both arms are working up from. One number per fraction; the arms share
  the start.
- ``moved`` -- mean circular chi1 distance from each free residue's own start.
  Silencing drives this toward zero.
- ``stuck`` -- the share of free residues that end in the well they began in.

Under silencing, the ``native`` arm has a much smaller ``moved`` and a much
larger ``stuck`` than ``random``, and its ``free right`` sits near ``start
right``. Under genuine-but-wrong co-solving it moves just as much and lands
somewhere worse, and then the no-op term is not the culprit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.casp import REFERENCE, RESULTS, randomised_start, targets_for
from aanca.diagnostics import (
    chi1_degrees,
    chi1_residues,
    circular_error,
    each_target,
    percent,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.rotamers import nearest_well

#: Keeps the pinned set's draw independent of the start's, so the two do not
#: share a stream and correlate for no reason.
CLAMP_STRIDE = 7_000_000


def frozen_mask(
    n_residues: int, fraction: float, seed: int, device: torch.device | str
) -> torch.Tensor:
    """A reproducible random subset of residues to pin, as a bool mask.

    Drawn on the CPU whatever the device, for the same reason the damage is:
    which residues are pinned has to be a property of the seed and not of
    where the job landed.
    """
    if fraction <= 0.0:
        return torch.zeros(n_residues, dtype=torch.bool, device=device)
    draws = torch.rand(
        n_residues, generator=torch.Generator().manual_seed(seed)
    )
    return (draws < fraction).to(device)


def clamped_pack(
    rule,
    native_coords: torch.Tensor,
    topo,
    config: RolloutConfig,
    seed: int,
    fraction: float,
    arm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(packed coords, frozen mask, start coords)`` for one target.

    Every residue starts from the benchmark's uniform draw. The pinned ones are
    then moved to the conformation this arm holds them at -- their native one,
    or a second independent uniform draw -- and held there for the rollout.
    """
    frozen = frozen_mask(
        topo.n_residues, fraction, seed + CLAMP_STRIDE, native_coords.device
    )
    start = randomised_start(native_coords, topo, seed)

    if bool(frozen.any()):
        if arm == "native":
            held = native_coords
        elif arm == "random":
            # A *different* seed, so the held conformation is not the one the
            # free residues were started from.
            held = randomised_start(native_coords, topo, seed + 2 * CLAMP_STRIDE)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        # Atom-level selection: only the pinned residues' atoms are replaced,
        # and the start's own geometry is kept everywhere else.
        rows = frozen.unsqueeze(-1).unsqueeze(-1).expand_as(start)
        start = torch.where(rows, held, start)

    result = rollout(
        rule,
        initial_state(start, rule),
        topo,
        config,
        generator=torch.Generator().manual_seed(seed),
        record=False,
        frozen=frozen,
    )
    return result.state.coords, frozen, start


def measure(rule, config, stems, args, fraction: float, arm: str) -> dict:
    """Well-match rate and chi1 MAE over the residues left free."""
    free_pred, free_native, free_start, pinned_moved = [], [], [], []
    for index, _stem, topo, native_coords in each_target(
        stems, args.backbone, args.device
    ):
        torsion = topo.torsion
        seed = args.seed + index
        packed, frozen, start = clamped_pack(
            rule, native_coords, topo, config, seed, fraction, arm
        )
        scored = chi1_residues(torsion) & ~frozen.cpu()
        free_pred.append(chi1_degrees(packed, torsion)[scored])
        free_native.append(chi1_degrees(native_coords, torsion)[scored])
        free_start.append(chi1_degrees(start, torsion)[scored])

        # The invariant: a pinned residue's chi1 must not have moved. Measured
        # rather than asserted, because a silent failure here would look like
        # a clean result -- the whole test would be comparing a rollout to
        # itself.
        pinned = chi1_residues(torsion) & frozen.cpu()
        if bool(pinned.any()):
            held = native_coords if arm == "native" else None
            if held is not None:
                pinned_moved.append(
                    circular_error(
                        chi1_degrees(packed, torsion)[pinned],
                        chi1_degrees(held, torsion)[pinned],
                    )
                )

    if not free_pred:
        return {}
    pred = torch.cat(free_pred)
    native = torch.cat(free_native)
    start = torch.cat(free_start)
    native_well, _ = nearest_well(native)
    pred_well, _ = nearest_well(pred)
    start_well, _ = nearest_well(start)
    right = pred_well == native_well
    drift = torch.cat(pinned_moved) if pinned_moved else torch.zeros(1)
    return {
        "n": int(native_well.numel()),
        "right": percent(right),
        "mae": float(circular_error(pred, native).mean()),
        "pinned_drift": float(drift.max()),
        # The silencing diagnostics. ``start_right`` is a property of the
        # start and the scored set, so the two arms must agree on it to
        # within nothing at all -- if they disagree, the arms are not
        # scoring the same residues and the comparison is void.
        "start_right": percent(start_well == native_well),
        "moved": float(circular_error(pred, start).mean()),
        "stuck": percent(pred_well == start_well),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--casp", type=int, default=13, choices=[13, 14])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backbone", default="casp_all")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N targets only")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="shares of residues to pin; 0 is also run unless --no-anchor",
    )
    parser.add_argument(
        "--no-anchor",
        action="store_true",
        help=(
            "skip the 0.00 row. It scores every residue in the benchmark, so "
            "it costs more than any pinned row and it is the reason a "
            "seven-row sweep never fitted a 4 h card. 'native - random' does "
            "not need it; only 'native - anchor' does."
        ),
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

    covered = [m for m in REFERENCE[args.casp] if (RESULTS / m / args.backbone).is_dir()]
    stems = targets_for(RESULTS, args.casp, covered, args.backbone)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        print(f"no CASP{args.casp} targets covered by all of {covered}")
        return 1

    print(
        f"clamp: {args.checkpoint} at update {state.get('update', '?')}, "
        f"CASP{args.casp}, {len(stems)} targets, {args.steps} steps, {args.device}"
    )
    print(
        "\nScored on the residues left free. The 0.00 row pins nothing and is"
        "\nthe benchmark's own number, so it anchors the two arms together."
        "\n'start' is the randomised start's own well-match rate on the same"
        "\nresidues, 'moved' their mean chi1 distance from that start, and"
        "\n'stuck' the share that never left the well they began in."
    )
    print(
        "\n  pinned      arm   free right    free MAE   free n   pinned drift"
        "    start   moved   stuck"
    )

    def show(label: str, arm: str, out: dict) -> None:
        drift = f"{out['pinned_drift']:11.2e}" if arm == "native" else "          -"
        print(
            f"   {label}  {arm:>7}   {out['right']:9.1f}%"
            f"  {out['mae']:10.2f}  {out['n']:7d}  {drift}"
            f"   {out['start_right']:5.1f}%  {out['moved']:6.2f}"
            f"  {out['stuck']:5.1f}%"
        )

    with torch.no_grad():
        anchor = {}
        if not args.no_anchor:
            anchor = measure(rule, config, stems, args, 0.0, "native")
            if not anchor:
                print("no usable targets")
                return 1
            show("0.00", "-", anchor)
        rows = {}
        for fraction in args.fractions:
            for arm in ("native", "random"):
                out = measure(rule, config, stems, args, fraction, arm)
                if not out:
                    print("no usable targets")
                    return 1
                rows[(fraction, arm)] = out
                show(f"{fraction:4.2f}", arm, out)

    print("\n--- the comparison that decides it ---\n")
    print("  pinned     native - random    native - anchor      moved n/r")
    for fraction in args.fractions:
        native_row = rows[(fraction, "native")]
        random_row = rows[(fraction, "random")]
        n_arm = native_row["right"]
        r_arm = random_row["right"]
        raw = (
            f"{n_arm - anchor['right']:+15.1f}"
            if anchor
            else "              -"
        )
        ratio = native_row["moved"] / max(random_row["moved"], 1e-9)
        print(
            f"   {fraction:4.2f}   {n_arm - r_arm:+15.1f}    {raw}"
            f"    {ratio:13.2f}"
        )
        if abs(native_row["start_right"] - random_row["start_right"]) > 0.05:
            print(
                "   ^ the arms disagree on 'start', so they are NOT scoring "
                "the same residues and this row is void"
            )

    print(
        "\nReading it: 'native - random' holds the fraction frozen fixed and"
        "\nvaries only whether the frozen conformations are correct, so it is"
        "\nthe co-solving signal with the quieter-neighbourhood effect divided"
        "\nout. Near zero at every fraction means the rule is not reading its"
        "\nneighbours' side chains and 77.3 % is a per-residue marginal."
        "\n'native - anchor' is the raw gain, which includes both effects."
        "\n\nThe 'pinned drift' column must be ~0: it is the largest distance any"
        "\npinned residue's chi1 moved from where it was held. Anything else"
        "\nmeans the pin leaked and the whole table is void."
        "\n\n'moved n/r' is the native arm's mean movement over the random"
        "\narm's. Well below 1 with a negative 'native - random' is the"
        "\nsilencing result: correct motionless neighbours read as a converged"
        "\nstate and the rule stops, leaving free residues near their random"
        "\nstart. Near 1 means the rule moved them just as far and landed"
        "\nsomewhere worse, which is a different problem and not the no-op"
        "\nterm's doing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
