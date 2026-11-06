"""Read the ceiling check's own loss off a trained checkpoint, on the CPU.

`g8_ceiling` (32405172) ran its whole 2,000 updates with the held-out `move`
term structurally absent -- `evaluate()` never passed `move_weight`, and
`growth_objective` computes the term only `if weight:`. So the run cannot answer
the question it exists for, and the two readings need opposite fixes:

  * **`move` low** -- the rule learned the right per-step moves and they do not
    accumulate into placement. The suspect is then the rollout's composition:
    the bernoulli update schedule, rotating each residue about its own CA, or
    moves applied and undone.
  * **`move` high** -- the placement gradient never reached the frame head, and
    the architecture cannot place residues under any supervision.

Re-running the arm with working instrumentation costs 71 minutes of a GPU slot
and gives the trajectory. **This gives the endpoint for free**, off the
checkpoint the run already wrote, which is enough to pick the branch -- so run
this first and only spend the slot if the endpoint is ambiguous.

**And `move` alone cannot pick the branch, which is why this reports two more
numbers.** Measured on a local checkpoint: `move` tracks `rmsd_dmg` at a near
constant ratio (0.0206 against 1.11 A, 0.2269 against 11.12 A -- about 50x
throughout), because its position half is the same per-residue distance the
RMSD is, averaged over live steps and diluted by the intact residues. So "move
fell but rmsd_dmg did not" is close to impossible by construction, and a loss
value was the wrong instrument.

What separates the two readings is **direction**, not magnitude:

  * ``align`` -- the cosine between the emitted translation and the ideal one,
    over damaged residues and live steps. 1.0 is a perfect bearing, 0.0 is
    orthogonal, negative is away from the target. This is the number that says
    whether the rule knows *which way* to go, which is exactly what the reach
    probes left open for the reconstruction arms.
  * ``step`` -- the emitted translation's length against the 0.3 A cap, so a
    rule that knows the direction but asks for nothing is distinguishable from
    one that asks for the wrong thing.

    python scripts/ceiling_readout.py --checkpoint runs/g8_ceiling/last.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.growth import GrowthConfig, growth_objective
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"


def held_out(stored: dict, state: dict):
    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[str(stored.get("split") or "train")]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = int(stored.get("structures", 400)) + int(
        stored.get("eval_structures", 12)
    )
    pool = available[:wanted] if len(available) >= wanted else available
    if not pool:
        raise SystemExit("no structures on disk for that split")
    dataset = RepackingDataset(
        pool, concentrations=(0.0,),
        crop_size=int(stored.get("crop_size") or 128),
        device=torch.device("cpu"), cache_limit=max(len(pool), 1),
    )
    indices = list(state.get("eval_indices") or [])
    if not indices:
        order = list(range(len(pool)))
        random.Random(int(stored.get("seed", 0))).shuffle(order)
        indices = order[: int(stored.get("eval_structures", 12))]
    return dataset, indices, len(pool)


def _bearing(result, native, topo, damage, stored, seed, index):
    """``(align, step, net_over_path, series)`` over the damaged window.

    ``align`` is the cosine between the emitted translation and the direction
    the residue must travel to reach its native CA -- the quantity a loss value
    cannot give, since a rule can have the right magnitude and the wrong
    bearing.

    **``net_over_path`` and ``series`` exist because a correct bearing is not
    sufficient.** The pMHC binder session put it as the two-frames question
    pointed at time rather than space: if the bearing is right but *re-aimed*
    every step from a frame that is itself drifting, the walk can be correct at
    every step and still arrive nowhere. ``net_over_path`` is the window's net
    displacement over its total path length, so 1.0 is a straight line and
    values near 0 are a walk that spends its budget turning. ``series`` is the
    per-step cosine, because a mean hides the difference between "wrong
    throughout" and "right at first, then lost".

    That matters most for ``--equivariant-move``, which this measurement can
    embarrass: reading the output in the residue's own frame is what makes the
    same neighbourhood give the same *local* move, but if the residue's frame
    drifts across steps then a constant local output traces a curve in the lab.
    The fix removes one failure and can introduce the other, and only the path
    ratio distinguishes them.
    """
    from aanca.growth import backbone_atoms, damage_window, growth_links

    moves = getattr(result, "moves", [])
    nan = float("nan")
    if not moves:
        return nan, nan, nan, []
    links = growth_links(topo)
    damaged = damage_window(
        topo.n_residues, links, damage,
        torch.Generator().manual_seed(seed + index), native.device,
    )
    if not bool(damaged.any()):
        return nan, nan, nan, []
    ca_native = backbone_atoms(native, topo)[:, 1]

    cosines, lengths, series = [], [], []
    path = None
    first = None
    previous = None
    for _axis_angle, translation, coords, moving in moves:
        scored = moving & damaged
        ca_now = backbone_atoms(coords, topo)[:, 1]
        if first is None:
            first = ca_now
            path = torch.zeros_like(ca_now[:, 0])
        if previous is not None:
            path = path + (ca_now - previous).norm(dim=-1)
        previous = ca_now
        if not bool(scored.any()):
            continue
        ideal = (ca_native - ca_now)[scored]
        emitted = translation.to(coords.dtype)[scored]
        ideal_unit = ideal / ideal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        emitted_norm = emitted.norm(dim=-1, keepdim=True)
        # **Skip a step that emitted nothing instead of scoring it zero.** A
        # cosine against a zero vector divides by the clamp and comes back 0.0,
        # which reads as "uncorrelated" when the truth is "no move was made".
        # That artefact was reported as evidence once already.
        if float(emitted_norm.mean()) < 1e-6:
            lengths.append(torch.zeros_like(emitted_norm.mean()))
            continue
        cosine = (ideal_unit * (emitted / emitted_norm.clamp_min(1e-8))).sum(-1)
        cosines.append(cosine.mean())
        series.append(round(float(cosine.mean()), 3))
        lengths.append(emitted.norm(dim=-1).mean())

    if not cosines:
        return nan, nan, nan, []
    net = (previous - first).norm(dim=-1)[damaged].mean()
    walked = path[damaged].mean()
    return (
        float(torch.stack(cosines).mean()),
        float(torch.stack(lengths).mean()),
        float(net / walked.clamp_min(1e-9)),
        series,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--untrained", action="store_true",
        help="score a fresh rule instead, for the value these terms start at",
    )
    parser.add_argument(
        "--perturb", type=float, default=0.0,
        help=(
            "with --untrained, add noise of this scale to the frame head's "
            "last layer. **This is the control `--untrained` alone cannot "
            "be.** A fresh rule's frame head is zero-initialised, so it emits "
            "the zero vector and `align` comes back exactly 0.000 -- which is "
            "an artefact of taking a cosine against a zero norm, not a "
            "measurement that the bearing is random. Perturbing the head gives "
            "non-zero moves with no training, so the bearing is defined and "
            "the question 'does the lab-frame defect give an uncorrelated "
            "bearing independently of training?' can actually be asked. 0.05 "
            "reproduces the scale used in tests/test_move_equivariance.py"
        ),
    )
    args = parser.parse_args()

    path = Path(args.checkpoint)
    state = ckpt.load(path, map_location="cpu")
    stored = state.get("args") or {}
    if not stored:
        sibling = path.with_name("last.pt")
        if sibling.exists():
            beside = ckpt.load(sibling, map_location="cpu")
            stored = beside.get("args") or {}
            if not state.get("eval_indices"):
                state["eval_indices"] = beside.get("eval_indices") or []
    if not stored:
        print(f"{path} stores no run arguments; point at the run's last.pt")
        return 1

    weights = ckpt.rule_weights(state)
    dataset, indices, pool_size = held_out(stored, state)
    damage = int(stored.get("damage") or 0)
    if damage <= 0:
        print("this run is not a regeneration run; nothing to read")
        return 1

    config = RolloutConfig(
        steps=int(stored.get("steps") or 120),
        truncate=int(stored.get("truncate") or 12),
        k=int(stored.get("k") or 32),
        blind=bool(stored.get("blind", True)),
        growth=GrowthConfig(
            birth="none", initial=str(stored.get("initial") or "helix")
        ),
    )
    seed = int(stored.get("seed", 0))
    rows = []
    rng = random.Random(seed)
    for index in indices:
        try:
            topo, native, _c, _u = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 12:
            continue
        rule = AANCARule(
            growth=True, birth="none",
            identity="identity_head.0.weight" in weights,
            morphogen="emit_head.0.weight" in weights,
            identity_keep=float(stored.get("identity_keep", 2.0)),
        )
        if not args.untrained:
            rule.load_state_dict(weights)
        elif args.perturb:
            # Seeded per crop so the perturbation is a property of the run and
            # not of the order the crops happen to load in.
            generator = torch.Generator().manual_seed(seed + index)
            for parameter in rule.frame_head[-1].parameters():
                parameter.data.add_(
                    torch.randn(
                        parameter.shape, generator=generator
                    ) * args.perturb
                )
        rule.eval()
        with torch.no_grad():
            terms, result = growth_objective(
                rule, native, topo, config,
                generator=torch.Generator().manual_seed(seed + index),
                identity_weight=float(stored.get("identity_weight") or 0.0),
                identity_start=str(stored.get("identity_start", "masked")),
                damage=damage,
                move_weight=1.0,
                clash_weight=1.0,
                clash_tolerance=float(stored.get("clash_tolerance") or 0.5),
            )
        if terms.rmsd_damaged is None:
            continue
        align, step, ratio, series = _bearing(
            result, native, topo, damage, stored, seed, index
        )
        rows.append((
            index,
            float(terms.move),
            float(terms.clash),
            float(terms.rmsd_damaged),
            align,
            step,
            ratio,
            series,
        ))

    if not rows:
        print("no crops produced a damaged window")
        return 1
    label = str(path)
    if args.untrained:
        label = (
            f"UNTRAINED, frame head perturbed by {args.perturb:g}"
            if args.perturb
            else "UNTRAINED (zero-initialised head: `align` is vacuous, see "
            "--perturb)"
        )
    print(f"{label}, {len(rows)} held-out crops of {pool_size}, "
          f"{damage}-residue window, {config.steps} steps\n")
    print(f"{'crop':>6} {'move':>8} {'clash':>9} {'rmsd_dmg':>9} "
          f"{'align':>7} {'step':>7} {'net/path':>9}")
    for index, move, clash, dmg, align, step, ratio, _s in rows:
        print(f"{index:>6} {move:>8.4f} {clash:>9.5f} {dmg:>9.2f} "
              f"{align:>7.3f} {step:>7.4f} {ratio:>9.3f}")
    n = len(rows)
    print(f"\n{'mean':>6} {sum(r[1] for r in rows)/n:>8.4f} "
          f"{sum(r[2] for r in rows)/n:>9.5f} {sum(r[3] for r in rows)/n:>9.2f} "
          f"{sum(r[4] for r in rows)/n:>7.3f} {sum(r[5] for r in rows)/n:>7.4f} "
          f"{sum(r[6] for r in rows)/n:>9.3f}")
    cap = 0.3
    print(f"\n  step against the {cap:g} A cap: "
          f"{100 * sum(r[5] for r in rows) / n / cap:.1f}% used")
    print("\n  per-step bearing, first crop (a mean hides 'right then lost'):")
    print("   ", rows[0][7][:14] or "(nothing emitted)")
    print(
        "\n  Read `align` first -- it is the one that picks the branch.\n"
        "    near 1.0  the rule knows the bearing, so a flat rmsd_dmg means\n"
        "              the moves are not accumulating: suspect the update\n"
        "              schedule, rotating about each residue's own CA, or\n"
        "              moves applied and undone.\n"
        "    near 0.0  the rule does not know which way to go even when every\n"
        "              step is supervised, and the capacity question answers\n"
        "              no.\n"
        "    negative  it is being driven away from the target, which no\n"
        "              amount of objective design fixes.\n"
        "  `step` against the 0.3 A cap separates 'wrong direction' from\n"
        "  'asking for nothing'. `move` is reported for continuity but is\n"
        "  roughly rmsd_dmg / 50 and settles nothing on its own."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
