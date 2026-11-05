"""What is each growth term actually contributing, at the weights being trained?

    python scripts/growth_balance_live.py --checkpoint runs/g1/last.pt

`scripts/growth_balance_probe.py` isolates a term by **zeroing every other
weight**, which is what made the original measurement possible -- and which
stops working the moment the rule is strong enough for the ablation to matter.
Run on g1's trained weights it returned `inf` for every gradient norm, because
removing the peptide restraint from a rule that has learned to move coordinates
lets the geometry diverge; the `fape`-only row was already infinite, and FAPE
clamps at 10 Å so it cannot produce an infinity itself. That measured the probe
rather than the rule.

**This measures the real thing.** Every weight stays at its training value, the
rollout runs **once**, and each weighted term's gradient is taken separately
with `retain_graph=True`. So the state the gradients are evaluated at is the
state training actually visits, restraints included, and no counterfactual
geometry is involved.

Two consequences worth stating, because they are the difference between this
and the ablation probe:

* The numbers are **contributions at a shared point**, not independent
  measurements. They do not have to sum to the total gradient -- gradient norms
  are not additive -- so read them as relative magnitudes at that point.
* A term whose weight is currently zero contributes exactly zero and is
  reported as such, rather than being switched on to see what it would do. If
  the question is "should this term be on", that is a different experiment.

The question it exists to answer: `BOND_WEIGHT = 0.022` was fitted at
initialisation, the note recording it says the bond term "should shrink as the
chain improves" and to re-measure a few thousand updates in, and g1 and g2 then
ran 9,000 updates with `bond` climbing 2.779 -> 3.903 against a crystal's 0.002.
So either the weight is too small to hold the chain, or the chain is diverging
for a reason the weight cannot fix. This separates those.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.growth import (
    BOND_WEIGHT,
    COMPLETION_WEIGHT,
    REACH_WEIGHT,
    GrowthConfig,
    growth_links,
    peptide_bond_restraint,
    relative_frame_restraint,
    pending_closures,
    reachability_restraint,
    seed_alive,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_identity, initial_state, rollout
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HEADS = ("activation_head", "birth_head", "frame_head", "identity_head")


def head_gradients(rule) -> dict[str, float]:
    out = {}
    for name in HEADS:
        head = getattr(rule, name, None)
        if head is None:
            out[name] = float("nan")
            continue
        total = 0.0
        for p in head.parameters():
            if p.grad is not None:
                total += float((p.grad**2).sum())
        out[name] = total**0.5
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pdb", default="1UBQ")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--truncate",
        type=int,
        default=0,
        help="backprop-through-time window; 0 takes the checkpoint's own value, "
        "falling back to 12. **Not `steps`, which is what both balance probes "
        "were doing and why they reported infinite gradients.** Training "
        "truncates at 12 while the probes backpropagated through all 30, so "
        "the infinity was the probe's configuration and not a property of the "
        "rule -- the third time in this project that a probe diverging from "
        "the training setup produced a number that looked like a finding.",
    )
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bond-weight", type=float, default=BOND_WEIGHT)
    parser.add_argument("--reach-weight", type=float, default=REACH_WEIGHT)
    parser.add_argument("--completion-weight", type=float, default=COMPLETION_WEIGHT)
    parser.add_argument("--identity-weight", type=float, default=93.7)
    parser.add_argument(
        "--orientation-weight", type=float, default=1.0,
        help=(
            "weight on relative_frame_restraint. Set to the value a run uses; "
            "1.0 was chosen by matching the term's *value* against FAPE's, "
            "which is exactly the mistake the completion weight was corrected "
            "for -- a term's value is not its influence. This probe is how to "
            "replace it with a measured one"
        ),
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="run the backward pass under torch.autograd.set_detect_anomaly, "
        "which raises at the first non-finite gradient and names the forward "
        "op that produced it. **This is the diagnostic for g1's non-finite "
        "gradients**: on g1's weights every `draws` count is 0 while the term "
        "*values* are finite -- fape 0.8826, bond 0.3255, identity 244.8 -- so "
        "the forward pass is healthy and the backward is not, which makes it a "
        "gradient singularity rather than a diverging rollout. Guessing at the "
        "op is what this replaces: `_bounded` was the obvious candidate and is "
        "demonstrably safe even at an exactly-zero vector, gradient (1, 1, 1). "
        "Slow, so use one repeat.",
    )
    args = parser.parse_args()
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    path = RAW / f"{args.pdb}.pdb"
    if not path.exists():
        print(f"{path} missing")
        return 1
    topo, native = feature_topology(load_protein(path))
    links = growth_links(topo)

    stored: dict = {}
    trained = None
    if args.checkpoint:
        state = ckpt.load(Path(args.checkpoint), map_location="cpu")
        stored = state.get("args") or {}
        trained = ckpt.rule_weights(state)

    truncate = args.truncate or int(stored.get("truncate") or 12)

    from aanca.fape import fape as fape_loss
    from aanca.identity import library_index, type_loss
    from aanca.torch_geometry import measure_chis

    names = ("fape", "bond", "completion", "identity", "reach", "twist")
    totals = {n: {h: 0.0 for h in HEADS} for n in names}
    values = {n: 0.0 for n in names}
    finite = {n: 0 for n in names}

    for repeat in range(args.repeats):
        rule = AANCARule(
            growth=True,
            birth=str(stored.get("birth", "cartesian")),
            identity=True,
            identity_keep=float(stored.get("identity_keep", 2.0)),
        )
        if trained is not None:
            rule.load_state_dict(trained)
        config = RolloutConfig(
            steps=args.steps, truncate=truncate, k=args.k, blind=True,
            identity_start=str(stored.get("identity_start", "masked")),
            growth=GrowthConfig(
                seeds=args.seeds, birth=str(stored.get("birth", "cartesian"))
            ),
        )
        generator = torch.Generator().manual_seed(repeat)
        alive = seed_alive(
            topo.n_residues, args.seeds, generator, native.device, links=links
        )
        start_identity = initial_identity(
            topo, native, config.identity_start, generator, alive=alive
        )
        result = rollout(
            rule,
            initial_state(native, rule, identity=start_identity, alive=alive),
            topo, config, generator=generator, record=False,
        )
        final = result.state
        scored = final.alive & topo.supervised_mask

        # Each term rebuilt on the SAME forward graph, so the gradients are all
        # evaluated at one state -- the state training visits.
        terms: dict[str, torch.Tensor] = {}
        terms["fape"] = fape_loss(final.coords, native, topo, final.alive)
        terms["bond"] = args.bond_weight * peptide_bond_restraint(
            final.coords, topo, final.alive, links
        ).to(terms["fape"].dtype)
        terms["reach"] = args.reach_weight * reachability_restraint(
            final.coords, topo, final.alive, links
        ).to(terms["fape"].dtype)
        terms["twist"] = args.orientation_weight * relative_frame_restraint(
            final.coords, native, topo, final.alive, links
        )[0].to(terms["fape"].dtype)

        from aanca.growth import frontier as frontier_of

        nward, cward = frontier_of(final.alive, links)
        unfinished = torch.stack([nward, cward], dim=-1)
        logits = final.activation_logits
        if bool(unfinished.any()):
            wanted = logits[unfinished]
            terms["completion"] = (
                args.completion_weight
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    wanted, torch.ones_like(wanted)
                ).to(terms["fape"].dtype)
            )
        else:
            terms["completion"] = torch.zeros_like(terms["fape"])

        native_chi1 = torch.rad2deg(measure_chis(native, topo.torsion)[:, 0])
        target = library_index(topo.torsion.resnames, native_chi1)
        terms["identity"] = args.identity_weight * type_loss(
            final.identity_logits, target, scored
        ).to(terms["fape"].dtype)

        for name in names:
            term = terms[name]
            values[name] += float(term.detach()) if torch.isfinite(term) else 0.0
            rule.zero_grad(set_to_none=True)
            if not (term.requires_grad and torch.isfinite(term)):
                continue
            term.backward(retain_graph=True)
            grads = head_gradients(rule)
            if all(
                not (g != g or g == float("inf")) for g in grads.values()
            ):
                finite[name] += 1
                for h in HEADS:
                    totals[name][h] += grads[h]

    print(
        f"{args.pdb}, {args.seeds} seeds, {args.steps} steps, mean of "
        f"{args.repeats} runs\nrule: {args.checkpoint or 'at initialisation'}\n"
        f"truncate {truncate} (training's, not steps)\n"
        f"weights: bond {args.bond_weight:g}, reach {args.reach_weight:g}, "
        f"completion {args.completion_weight:g}, identity {args.identity_weight:g}\n"
        "Every weight at its training value; one rollout; per-term gradients "
        "via retain_graph.\n"
    )
    print(
        f"  {'term':<12}{'weighted':>10}" + "".join(
            f"{h.replace('_head',''):>12}" for h in HEADS
        ) + f"{'draws':>7}"
    )
    for name in names:
        n = max(finite[name], 1)
        print(
            f"  {name:<12}{values[name] / args.repeats:>10.4f}"
            + "".join(f"{totals[name][h] / n:>12.2e}" for h in HEADS)
            + f"{finite[name]:>7d}"
        )
    print(
        "\n  weighted = the term as it enters the loss, so these are directly\n"
        "  comparable with each other. Gradient norms are not additive, so the\n"
        "  columns are relative magnitudes at a shared point rather than a\n"
        "  decomposition of the total. 'draws' counts how many repeats gave a\n"
        "  finite gradient -- anything below the repeat count means the rollout\n"
        "  is already producing non-finite values and the row is not\n"
        "  trustworthy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
