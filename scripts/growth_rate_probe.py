"""Does growth finish, and what sets the rate?

    python scripts/growth_rate_probe.py                      # untrained, bias sweep
    python scripts/growth_rate_probe.py --checkpoint runs/g1/last.pt

**Originally** this asked whether growth completes inside the loss window, which
mattered because reason 3 against a joint growth-and-packing objective was that
a growing structure has incomplete neighbourhoods where growth is happening, so
the chi target is unlearnable there. That objection only bites while growth is
still going, and ``loss_window = 0.5`` supervises the back half, so the question
was a number: when is the structure complete? It swept the **activation bias**,
because on an untrained rule that bias is what sets the wake probability -- at
the default -2 a frontier residue wakes with probability 0.12 per step.

**That framing is obsolete for a trained rule, and g1 is why.**
``scripts/growth_diagnose.py`` measured g1's wake probability at **1.000**: the
activation head is saturated, so the initialisation bias no longer sets
anything and sweeping it says nothing. What limits g1 is the **frontier** -- it
measured 1.4 growth opportunities per structure, and 1.4 births a step over 40
steps is the 57 residues that were actually alive, so growth ran at exactly the
rate available to it and the rollout simply ended.

So with ``--checkpoint`` this sweeps what now matters -- seeds and the birth cap
-- and reports the **frontier and segment trajectory** rather than only the
endpoint. The frontier is the rate, and the reason it is small is visible in the
segment count: four seeds growing both ways should offer eight opportunities,
and every time two segments coalesce two growable ends are consumed. A run whose
segment count falls while its alive fraction stalls is one that ate its own
frontier.

Both modes stay available and run on the same structure, so the trained numbers
are directly comparable with the untrained ones already recorded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.growth import (
    GrowthConfig,
    completion,
    frontier,
    growth_links,
    seed_alive,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_identity, initial_state, step_once, update_mask
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent


def segment_count(alive: torch.Tensor, links: torch.Tensor) -> int:
    """Maximal runs of alive residues joined by covalent links."""
    starts = alive.clone()
    starts[1:] &= ~(alive[:-1] & links)
    return int(starts.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--structure", default="data/raw/1UBQ.pdb")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    topo, native = feature_topology(load_protein(ROOT / args.structure))
    links = growth_links(topo)
    steps = args.steps

    stored: dict = {}
    trained = None
    if args.checkpoint:
        state = ckpt.load(Path(args.checkpoint), map_location="cpu")
        stored = state.get("args") or {}
        trained = ckpt.rule_weights(state)

    identity_weight = float(stored.get("identity_weight", 0.0) or 0.0)

    def build(bias: float) -> AANCARule:
        rule = AANCARule(
            growth=True,
            birth=str(stored.get("birth", "cartesian")),
            activation_init=bias,
            identity=bool(identity_weight),
            identity_keep=float(stored.get("identity_keep", 2.0)),
        )
        if trained is not None:
            rule.load_state_dict(trained)
        rule.eval()
        return rule

    name = Path(args.structure).stem
    label = args.checkpoint or "untrained"
    print(
        f"{name}, {topo.n_residues} residues, {steps} steps, mean of "
        f"{args.repeats} draws\nrule: {label}\n"
    )

    if trained is None:
        configs = [(b, s, m) for b in (-2.0, 0.0, 2.0, 4.0)
                   for s, m in ((1, 8), (4, 8), (4, 2))]
        print(f"{'bias':>6}{'p(wake)':>9}{'seeds':>7}{'cap':>5}"
              f"{'complete':>10}{'final':>8}"
              f"{'front@1':>9}{'front@mid':>11}{'front@end':>11}{'segs@end':>10}")
    else:
        # The bias is dead once the head is trained, so it is held at the run's
        # own value and only the things that still move the rate are swept.
        bias = float(stored.get("activation_init", -2.0))
        configs = [(bias, s, m) for s, m in
                   ((1, 8), (2, 8), (4, 8), (8, 8), (4, 2), (4, 16), (16, 16))]
        print(f"{'seeds':>7}{'cap':>5}{'complete':>10}{'final':>8}"
              f"{'front@1':>9}{'front@mid':>11}{'front@end':>11}{'segs@end':>10}")

    for bias, n_seeds, cap in configs:
        firsts, finals = [], []
        f_first, f_mid, f_end, s_end = [], [], [], []
        for draw in range(args.repeats):
            rule = build(bias)
            generator = torch.Generator().manual_seed(draw)
            alive = seed_alive(topo.n_residues, n_seeds, generator)
            config = RolloutConfig(
                steps=steps, truncate=steps, k=24,
                blind=bool(identity_weight),
                identity_start=str(stored.get("identity_start", "masked")),
                growth=GrowthConfig(
                    seeds=n_seeds, birth=str(stored.get("birth", "cartesian")),
                    max_births=cap,
                ),
            )
            start_identity = (
                initial_identity(topo, native, config.identity_start, generator, alive=alive)
                if identity_weight
                else None
            )
            state = initial_state(native, rule, identity=start_identity, alive=alive)
            done = None
            with torch.no_grad():
                for step in range(steps):
                    mask = update_mask(
                        topo.n_residues, config.update_probability,
                        native.device, generator,
                    )
                    state, _mu, _lk = step_once(
                        rule, state, topo, config, mask, generator=generator
                    )
                    nward, cward = frontier(state.alive, links)
                    size = int(nward.sum()) + int(cward.sum())
                    if step == 0:
                        f_first.append(size)
                    if step == steps // 2 - 1:
                        f_mid.append(size)
                    if step == steps - 1:
                        f_end.append(size)
                        s_end.append(segment_count(state.alive, links))
                    if done is None and bool(state.alive.all()):
                        done = step + 1
            firsts.append(done)
            finals.append(completion(state.alive))

        got = [f for f in firsts if f is not None]
        when = (
            f"{sum(got) / len(got):.0f}" if len(got) == len(firsts)
            else (f">{steps}({len(got)}/{args.repeats})" if got else f">{steps}")
        )
        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")
        row = (
            f"{when:>10}{mean(finals):>7.1%}"
            f"{mean(f_first):>9.1f}{mean(f_mid):>11.1f}"
            f"{mean(f_end):>11.1f}{mean(s_end):>10.1f}"
        )
        if trained is None:
            print(f"{bias:>6.1f}{torch.sigmoid(torch.tensor(bias)):>9.2f}"
                  f"{n_seeds:>7}{cap:>5}{row}")
        else:
            print(f"{n_seeds:>7}{cap:>5}{row}")

    print(
        "\ncomplete = first step where every slot is alive; final = alive fraction\n"
        "at the end. front@1 / @mid / @end are the number of growth opportunities\n"
        "-- (residue, direction) pairs with a dormant covalent neighbour -- which\n"
        "is the per-step birth ceiling. segs@end counts maximal alive runs.\n\n"
        "The rate is the frontier, not the cap: with a saturated wake probability\n"
        "a run grows front@t residues per step, so a frontier that decays faster\n"
        "than the structure fills cannot finish however many steps it is given.\n"
        "Watch segs@end against the seed count -- a fall means segments coalesced\n"
        "and each merge destroyed two growable ends."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
