"""Where do the births go? Frontier to actual births, step by step.

    python scripts/birth_funnel_probe.py --checkpoint runs/g1/last.pt

`scripts/growth_rate_probe.py` left a factor of two unexplained. At four seeds
on 1UBQ the trained rule completes in 47 steps, so the mean birth rate is
`72/47 = 1.5` residues per step -- against a step-1 frontier of 6.7 and a cap of
8. The update mask ticks a residue with probability 0.5, which predicts ~3.35
births a step, still about twice what is observed. That gap is a factor of two
on the step budget for every future growth run, so it is worth an hour.

**The funnel, read off the code rather than guessed.** In ``rollout.step_once``:

1. ``frontier(alive, links)`` gives (residue, direction) pairs with a dormant
   covalent neighbour.
2. ``limit_births(nward & moving, max_births)`` -- intersected with the **update
   mask**, so only frontier residues that tick this step may give birth, and
   capped **per direction**, meaning the effective ceiling is ``2 x max_births``
   rather than ``max_births``.
3. ``straight_through_bernoulli(activation)`` draws the wake decision; for g1
   this is ~1.0, so it should remove nothing.
4. ``give_birth``'s ``fresh = ~alive[children]`` drops any child already alive
   **and silently deduplicates two parents targeting the same slot** -- which is
   exactly what closing a one-residue gap looks like, so a gap of length 1
   offers two frontier pairs and can only ever yield one birth.

Step 4 is the candidate the rate probe could not see, and so is the thing the
rate probe *did* flag as unmeasured: the frontier at step 1 is not the frontier
through the run. This traces both per step, so the attribution is arithmetic
rather than argument.

**What each outcome would mean.** If ``births`` tracks ``frontier x 0.5``
throughout, the factor of two is simply frontier decay and the step budget
should be computed from the mean frontier, not its initial value. If the ratio
is persistently below 0.5, parents are competing for slots and the fix is a
birth rule that lets one parent serve a gap -- or more seeds spaced further
apart, since adjacent seeds create short gaps and short gaps are where the
duplication happens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.growth import GrowthConfig, frontier, growth_links, seed_alive
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    initial_identity,
    initial_state,
    step_once,
    update_mask,
)
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--structure", default="data/raw/1UBQ.pdb")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--bias",
        type=float,
        default=None,
        help="activation bias for an untrained rule. The point of setting it "
        "high is that it emulates a trained head's saturated wake probability "
        "-- 4.0 gives p = 0.98 against g1's 1.000 -- so the funnel can be "
        "traced through a run that actually *completes* without needing the "
        "trained weights. At the default -2 growth is too slow to fill the "
        "structure, which is precisely why the first trace could not see "
        "whether the frontier collapses at the end.",
    )
    args = parser.parse_args()

    topo, native = feature_topology(load_protein(ROOT / args.structure))
    links = growth_links(topo)

    stored: dict = {}
    trained = None
    if args.checkpoint:
        state = ckpt.load(Path(args.checkpoint), map_location="cpu")
        stored = state.get("args") or {}
        trained = ckpt.rule_weights(state)
    identity_weight = float(stored.get("identity_weight", 0.0) or 0.0)

    # Accumulated across draws, indexed by step.
    front_at = [0.0] * args.steps
    births_at = [0.0] * args.steps
    alive_at = [0.0] * args.steps
    gap1_at = [0.0] * args.steps
    draws = 0

    for draw in range(args.repeats):
        rule = AANCARule(
            growth=True,
            birth=str(stored.get("birth", "cartesian")),
            activation_init=(
                args.bias
                if args.bias is not None
                else float(stored.get("activation_init", -2.0))
            ),
            identity=bool(identity_weight),
            identity_keep=float(stored.get("identity_keep", 2.0)),
        )
        if trained is not None:
            rule.load_state_dict(trained)
        rule.eval()

        generator = torch.Generator().manual_seed(draw)
        alive = seed_alive(topo.n_residues, args.seeds, generator)
        config = RolloutConfig(
            steps=args.steps, truncate=args.steps, k=24,
            blind=bool(identity_weight),
            identity_start=str(stored.get("identity_start", "masked")),
            growth=GrowthConfig(
                seeds=args.seeds, birth=str(stored.get("birth", "cartesian")),
                max_births=int(stored.get("max_births", 8)),
            ),
        )
        start_identity = (
            initial_identity(topo, native, config.identity_start, generator, alive=alive)
            if identity_weight
            else None
        )
        state = initial_state(native, rule, identity=start_identity, alive=alive)

        with torch.no_grad():
            for step in range(args.steps):
                nward, cward = frontier(state.alive, links)
                size = int(nward.sum()) + int(cward.sum())
                # A gap of exactly one dormant slot: both flanking residues are
                # on the frontier and target the same child, so the pair can
                # only produce one birth. Counting these separates "the
                # frontier shrank" from "parents competed".
                dormant = ~state.alive
                gap1 = 0
                if dormant.shape[0] > 2:
                    interior = dormant[1:-1] & state.alive[:-2] & state.alive[2:]
                    gap1 = int((interior & links[:-1] & links[1:]).sum())
                before = int(state.alive.sum())
                mask = update_mask(
                    topo.n_residues, config.update_probability,
                    native.device, generator,
                )
                state, _mu, _lk = step_once(
                    rule, state, topo, config, mask, generator=generator
                )
                front_at[step] += size
                gap1_at[step] += gap1
                births_at[step] += int(state.alive.sum()) - before
                alive_at[step] += float(state.alive.sum()) / state.alive.shape[0]
        draws += 1

    n = max(draws, 1)
    label = args.checkpoint or "untrained"
    print(
        f"{Path(args.structure).stem}, {topo.n_residues} residues, "
        f"{args.seeds} seeds, {args.steps} steps, mean of {n} draws\nrule: {label}\n"
    )
    print(
        f"  {'step':>5}{'alive':>8}{'frontier':>10}{'gap-1':>7}"
        f"{'births':>8}{'b/(f*0.5)':>11}"
    )
    for step in range(args.steps):
        f = front_at[step] / n
        b = births_at[step] / n
        if f == 0 and b == 0 and step > 2:
            continue
        expected = f * 0.5
        ratio = b / expected if expected > 0 else float("nan")
        print(
            f"  {step + 1:>5}{alive_at[step] / n:>7.1%}{f:>10.2f}"
            f"{gap1_at[step] / n:>7.2f}{b:>8.2f}"
            f"{'    n/a' if expected == 0 else f'{ratio:>11.2f}'}"
        )

    total_f = sum(front_at) / n
    total_b = sum(births_at) / n
    active = sum(1 for s in range(args.steps) if births_at[s] > 0)
    print(
        f"\n  births {total_b:.1f} over {args.steps} steps "
        f"({total_b / args.steps:.2f} per step); mean frontier "
        f"{total_f / args.steps:.2f}, and {total_f / max(active, 1):.2f} over the "
        f"{active} steps where a birth happened."
    )
    print(
        f"  funnel: births / (summed frontier x 0.5) = "
        f"{total_b / max(total_f * 0.5, 1e-9):.3f}"
    )
    print(
        "\n  A ratio near 1 means the update mask explains everything and the\n"
        "  residual factor was frontier decay -- the step budget should then be\n"
        "  computed from the mean frontier rather than its step-1 value. Well\n"
        "  below 1 means parents compete for slots; `gap-1` counts the\n"
        "  one-dormant-slot gaps where two parents must target the same child,\n"
        "  which is the mechanism and is capped at one birth per gap by\n"
        "  give_birth's `fresh` filter."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
