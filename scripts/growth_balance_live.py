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
    baseline_coords,
    run_anchors,
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
PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"

#: Above this, a gradient norm is not a measurement. Trained weights run on a
#: rollout they were not trained on diverge, and the backward pass then reports
#: numbers like 1.95e+11 with every value in the forward pass still finite --
#: which looks exactly like a finding and is not one. Measured: the first
#: trained-weights run of this probe, on 1UBQ with 6 seeds and 30 steps against
#: g5_ctrl's crop-128/16-seed/120-step training, returned 1e+08 to 1e+11 on
#: every row. Flagged rather than hidden, because the fix is to match the
#: training configuration (``--crop``) and the reader has to know which it is
#: looking at.
SANE_GRADIENT = 1e5
HEADS = (
    "activation_head", "birth_head", "frame_head", "identity_head", "trunk",
)

#: Everything that is not an output head: the message passing, the gating, the
#: edge embedding -- the shared representation every head reads.
#:
#: **This is the column the head columns cannot replace.** The trainer uses
#: Adam, which divides each parameter's step by that parameter's own gradient
#: RMS, so a head whose gradient is uniformly a hundred times smaller than
#: another's still takes full-size steps. What Adam cannot undo is a *shared*
#: parameter receiving the sum of several objectives' signals: there the ratio
#: is real, because the sum is what gets normalised and the larger contribution
#: decides the direction. So "which objective is shaping the representation the
#: frame head reads" is a question about the trunk and not about the heads.
TRUNK = (
    "aggregate_norm", "candidate", "delta_head", "edge_embed", "gate",
    "kappa_head", "message",
)


def head_gradients(rule) -> dict[str, float | None]:
    """Gradient norm per head. ``None`` for a head this rule does not have.

    **``None`` and not ``nan``, and that distinction was a silent bug.** A
    ``free`` or ``none`` rule has no birth head, this returned ``nan`` for it,
    and the caller's "every gradient is finite" check then rejected *every row
    of the table* -- so the probe printed a complete grid of zeros with
    ``draws 0`` and nothing said the architecture was the reason. An absent
    head is not a non-finite gradient; it is the absence of a question.
    """
    out: dict[str, float | None] = {}
    for name in HEADS:
        if name == "trunk":
            total = 0.0
            for parameter_name, p in rule.named_parameters():
                if parameter_name.split(".")[0] in TRUNK and p.grad is not None:
                    total += float((p.grad**2).sum())
            out[name] = total**0.5
            continue
        head = getattr(rule, name, None)
        if head is None:
            out[name] = None
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
    parser.add_argument(
        "--crop", action="store_true",
        help=(
            "load a held-out crop from the pool at the checkpoint's own crop "
            "size and split, instead of a whole PDB from data/raw. **Use this "
            "with --checkpoint.** Trained weights on a structure unlike the "
            "ones they were trained on diverge in the backward pass, and the "
            "table then prints gradient norms of 1e+11 with every forward "
            "value still finite"
        ),
    )
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
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
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument(
        "--birth", default=None,
        choices=("torsion", "cartesian", "free", "none"),
        help=(
            "override the checkpoint's birth mode, or set one when there is no "
            "checkpoint. ``none`` is the self-placing rollout: every slot alive "
            "from step 0 on a baseline conformation, no births, and the frame "
            "head doing the whole job -- which is the configuration this probe "
            "is most needed for, because 'the rule holds still' and 'the "
            "geometry gradient is swamped' look identical in a training log"
        ),
    )
    parser.add_argument(
        "--initial", default="helix", choices=("helix", "extended", "random"),
        help="baseline conformation for --birth none; see growth.BASELINE_MODES",
    )
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

    stored: dict = {}
    trained = None
    if args.checkpoint:
        path = Path(args.checkpoint)
        state = ckpt.load(path, map_location="cpu")
        stored = state.get("args") or {}
        trained = ckpt.rule_weights(state)
        if not stored:
            # **`best.pt` carries only the weights.** The trainer writes the
            # run's arguments into `last.pt` and, until today, nowhere else --
            # so a probe pointed at `best.pt` read an empty dict and fell back
            # to its own defaults *silently*: 6 seeds, 30 steps, birth "free",
            # and no eval_indices, so it reshuffled its own held-out split.
            # That is the real cause of the two useless tables of 6 Nov, and
            # the fourth time a probe configured differently from training has
            # produced a number that looked like a finding. The sibling is the
            # same run, so read it; and if there is none, refuse, because a
            # default is indistinguishable from a measurement once it is
            # printed in a table.
            sibling = path.with_name("last.pt")
            if sibling.exists():
                beside = ckpt.load(sibling, map_location="cpu")
                stored = beside.get("args") or {}
                if not state.get("eval_indices"):
                    # The held-out split too, so the crop is the one this run
                    # never trained on rather than one a reshuffle picked.
                    state["eval_indices"] = beside.get("eval_indices") or []
                print(f"run arguments read from {sibling} ({path.name} has none)")
        if not stored:
            print(
                f"{path} stores no run arguments and "
                f"{path.with_name('last.pt')} is missing, so every rollout "
                "setting would be this probe's default rather than the run's "
                "-- which is how the last three wrong numbers in this project "
                "were produced. Point --checkpoint at the run's last.pt, or "
                "pass --birth/--steps/--seeds/--k explicitly if you mean to "
                "probe off-configuration."
            )
            return 1

    # **Which heads the rule has is read from the weights, not from the stored
    # arguments**, the same lesson segment_placement_probe.py records: g6_self
    # stores birth "none" and this probe read "cartesian" from somewhere, built
    # a rule with a birth head, and ``load_state_dict`` refused -- exit 1 with
    # nothing said about why. The weights cannot disagree with themselves.
    has_birth_head = trained is not None and "birth_head.0.weight" in trained
    has_identity = trained is not None and "identity_head.0.weight" in trained
    if args.birth:
        birth = args.birth
    elif trained is None:
        birth = str(stored.get("birth") or "cartesian")
    elif not has_birth_head:
        # No birth head means ``free`` or ``none``, and those differ in whether
        # the rollout grows at all -- which the weights cannot say. The stored
        # argument is the only source for that, and defaulting it to ``none``
        # would silently turn a grown run into a self-placing one.
        birth = str(stored.get("birth") or "free")
        if birth not in ("free", "none"):
            birth = "free"
    else:
        birth = str(stored.get("birth") or "cartesian")
        if birth in ("free", "none"):
            birth = "cartesian"

    # Every rollout setting defaults to the *run's own*, because a probe
    # configured differently from training measures the probe. Paid for twice
    # now: once by backpropagating through 30 steps where training truncates at
    # 12, and once by running trained weights on 1UBQ at 6 seeds.
    truncate = args.truncate or int(stored.get("truncate") or 12)
    steps = args.steps or int(stored.get("steps") or 30)
    seeds = args.seeds or int(stored.get("seeds") or 6)
    k = args.k or int(stored.get("k") or 24)

    if args.crop:
        # A held-out crop from the pool, which is what training actually sees.
        import random as _random

        from aanca.data import RepackingDataset, read_manifest

        splits = read_manifest(MANIFEST)
        available = []
        for spec in splits[str(stored.get("split") or "train")]:
            for suffix in (".bcif", ".cif", ".pdb"):
                candidate = PDB / f"{spec.pdb_id}{suffix}"
                if candidate.exists():
                    available.append(
                        type(spec)(spec.pdb_id, candidate, spec.group)
                    )
                    break
        wanted = int(stored.get("structures", 400)) + int(
            stored.get("eval_structures", 12)
        )
        pool = available[:wanted] if len(available) >= wanted else available
        if not pool:
            print("no structures on disk for that split")
            return 1
        dataset = RepackingDataset(
            pool, concentrations=(0.0,),
            crop_size=int(stored.get("crop_size") or 128),
            device=torch.device("cpu"), cache_limit=max(len(pool), 1),
        )
        indices = list(state.get("eval_indices") or []) if args.checkpoint else []
        if not indices:
            order = list(range(len(pool)))
            _random.Random(int(stored.get("seed", 0))).shuffle(order)
            indices = order[: int(stored.get("eval_structures", 12))]
        topo, native, _crop, _used = dataset.load_crop(
            indices[0], rng=_random.Random(int(stored.get("seed", 0)))
        )
        source = f"held-out crop {indices[0]} of {len(pool)}"
    else:
        path = RAW / f"{args.pdb}.pdb"
        if not path.exists():
            print(f"{path} missing")
            return 1
        topo, native = feature_topology(load_protein(path))
        source = args.pdb
    links = growth_links(topo)

    from aanca.fape import fape as fape_loss
    from aanca.identity import library_index, type_loss
    from aanca.torch_geometry import measure_chis

    names = ("fape", "bond", "completion", "identity", "reach", "twist")
    totals = {n: {h: 0.0 for h in HEADS} for n in names}
    #: Heads this rule does not have, so the table can print ``-`` for them
    #: rather than a zero that reads as "measured, and nothing there".
    missing = set(HEADS)
    values = {n: 0.0 for n in names}
    finite = {n: 0 for n in names}

    self_placing = birth == "none"
    print(
        f"{source}: birth {birth}, identity head "
        f"{has_identity or trained is None}"
        + (f", {args.initial} baseline" if self_placing else "")
    )

    for repeat in range(args.repeats):
        rule = AANCARule(
            growth=True,
            birth=birth,
            identity=has_identity or trained is None,
            morphogen=trained is not None and "emit_head.0.weight" in trained,
            identity_keep=float(stored.get("identity_keep", 2.0)),
        )
        if trained is not None:
            rule.load_state_dict(trained)
        config = RolloutConfig(
            steps=steps, truncate=truncate, k=k, blind=True,
            identity_start=str(stored.get("identity_start", "masked")),
            growth=GrowthConfig(
                seeds=args.seeds, birth=birth, initial=args.initial
            ),
        )
        generator = torch.Generator().manual_seed(repeat)
        if self_placing:
            # The same start ``growth_objective`` builds: all alive, one native
            # anchor per covalent run, everything else on the reference walk.
            alive = torch.ones(
                topo.n_residues, dtype=torch.bool, device=native.device
            )
            anchors = run_anchors(
                topo.n_residues, links, generator, native.device
            )
            start = baseline_coords(
                native, topo, links, anchors, args.initial, generator
            )
        else:
            alive = seed_alive(
                topo.n_residues, seeds, generator, native.device, links=links
            )
            start = native
        start_identity = initial_identity(
            topo, native, config.identity_start, generator, alive=alive
        )
        result = rollout(
            rule,
            initial_state(start, rule, identity=start_identity, alive=alive),
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
            present = [g for g in grads.values() if g is not None]
            if all(not (g != g or g == float("inf")) for g in present):
                finite[name] += 1
                for h in HEADS:
                    if grads[h] is not None:
                        totals[name][h] += grads[h]
                        missing.discard(h)

    print(
        f"{source}, {seeds} seeds, {steps} steps, mean of "
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
    exploded = any(
        totals[name][h] / max(finite[name], 1) > SANE_GRADIENT
        for name in names for h in HEADS if h not in missing
    )
    for name in names:
        n = max(finite[name], 1)
        print(
            f"  {name:<12}{values[name] / args.repeats:>10.4f}"
            + "".join(
                f"{'-':>12}" if h in missing else f"{totals[name][h] / n:>12.2e}"
                for h in HEADS
            )
            + f"{finite[name]:>7d}"
        )
    print(
        "\n  weighted = the term as it enters the loss, so these are directly\n"
        "  comparable with each other. Gradient norms are not additive, so the\n"
        "  columns are relative magnitudes at a shared point rather than a\n"
        "  decomposition of the total. 'draws' counts how many repeats gave a\n"
        "  finite gradient -- anything below the repeat count means the rollout\n"
        "  is already producing non-finite values and the row is not\n"
        "  trustworthy. A '-' column is a head this rule does not have."
    )
    if exploded:
        print(
            f"\n  ** A gradient norm above {SANE_GRADIENT:g} means this is not a\n"
            "  measurement. The rollout has diverged, which trained weights do\n"
            "  on a configuration they were not trained on while every forward\n"
            "  value stays finite. Re-run with --crop so the structure, crop\n"
            "  size, seeds and steps are the run's own."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
