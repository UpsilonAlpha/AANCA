"""Why did growth stop? Read a growth checkpoint's stall diagnostics, no training.

    python scripts/growth_diagnose.py runs/g1/last.pt [--steps 40]

g1 stalled at about 48 % of slots alive while the completion term read **0.004**
-- essentially satisfied. Two situations produce that pair and they need
opposite fixes:

* **The frontier is empty.** Then the completion term is a mean over nothing and
  returns exactly zero by its own ``else`` branch, so a half-built structure is
  indistinguishable from a finished one. The fix is a term that charges for
  dormant slots, or a frontier that does not vanish.
* **The frontier is non-empty and its logits are saturated.** Then the rule
  already wants to grow everywhere it can, the loss is correctly near zero, and
  something *else* limits births -- in which case adding a dormant-slot penalty
  would push on an already-maximised quantity and change nothing.

``GrowthTerms`` now carries ``frontier``, ``wake_probability``, ``segments`` and
``unreachable`` to separate these. This script applies them to a finished or
in-flight checkpoint without spending a GPU on another training run, because the
job that produced the checkpoint is running the *older* code and cannot report
them itself.

``unreachable`` is the number that matters most: dormant slots with no covalently
linked alive neighbour can never be reached from the current state, so they make
a stall permanent rather than slow. Note that ``frontier`` is gated on ``links``,
so a segment ending at a chain break or a crop boundary is **not** on the
frontier -- which is a third possible cause, and one the ``unreachable`` count
exposes directly.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.growth import GrowthConfig, growth_objective
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"


def on_disk(specs):
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--steps", type=int, default=0, help="0 uses the run's own")
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    state = ckpt.load(Path(args.checkpoint), map_location="cpu")
    stored = state.get("args") or {}
    if not stored:
        print("this checkpoint carries no args; cannot rebuild the run's config")
        return 1

    split = stored.get("split", "train")
    pool = on_disk(read_manifest(MANIFEST)[split])
    indices = list(state.get("eval_indices") or [])[: args.structures]
    if not indices:
        print("no eval_indices in this checkpoint")
        return 1

    dataset = RepackingDataset(
        pool,
        concentrations=(0.0,),
        crop_size=int(stored.get("crop_size", 128)),
        device=device,
        cache_limit=max(len(pool), 1),
    )

    identity_weight = float(stored.get("identity_weight", 0.0))
    rule = AANCARule(
        growth=True,
        birth=str(stored.get("birth", "cartesian")),
        activation_init=float(stored.get("activation_init", -2.0)),
        max_rotation=float(stored.get("max_rotation", 0.1)),
        max_translation=float(stored.get("max_translation", 0.3)),
        identity=bool(identity_weight),
        identity_keep=float(stored.get("identity_keep", 2.0)),
    ).to(device)
    weights = Path(args.weights) if args.weights else Path(args.checkpoint)
    rule.load_state_dict(ckpt.rule_weights(ckpt.load(weights, map_location="cpu")))
    rule.eval()

    steps = args.steps or int(stored.get("steps") or 40)
    config = RolloutConfig(
        steps=steps,
        truncate=int(stored.get("truncate", 12)),
        k=int(stored.get("k", 32)),
        update_schedule=str(stored.get("update_schedule", "bernoulli")),
        coupling_radius=float(stored.get("coupling_radius", 5.0)),
        blind=bool(identity_weight),
        identity_start=str(stored.get("identity_start", "masked")),
        identity_sampling=str(stored.get("identity_sampling", "sample")),
        growth=GrowthConfig(
            seeds=int(stored.get("seeds", 4)),
            cward_only=bool(stored.get("cward_only", False)),
            birth=str(stored.get("birth", "cartesian")),
            activation="sample",
            max_births=int(stored.get("max_births", 8)),
            max_rotation=float(stored.get("max_rotation", 0.1)),
            max_translation=float(stored.get("max_translation", 0.3)),
        ),
    )

    print(
        f"{weights}, update {state.get('update', '?')}, {len(indices)} held-out "
        f"structures, {steps} steps, {config.growth.seeds} seeds, {device}\n"
    )
    print(
        f"  {'entry':>6}{'alive':>8}{'frontier':>10}{'wake p':>9}"
        f"{'segments':>10}{'unreach':>9}{'done':>8}{'rmsd':>9}{'slots':>7}"
    )

    rng = random.Random(0)
    totals = {k: 0.0 for k in ("alive", "frontier", "wake", "segments", "unreach", "done", "slots")}
    n = 0
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _c, _u = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            generator = torch.Generator().manual_seed(index)
            terms, _result = growth_objective(
                rule, native, topo, config,
                generator=generator,
                bond_weight=float(stored.get("bond_weight", 0.022)),
                completion_weight=float(stored.get("completion_weight", 27.2)),
                identity_weight=identity_weight,
                identity_start=str(stored.get("identity_start", "masked")),
                reach_weight=float(stored.get("reach_weight", 0.086)),
            )
            wake = terms.wake_probability
            print(
                f"  {index:>6}{terms.alive_fraction:>7.1%}{terms.frontier:>10d}"
                f"{'   n/a' if math.isnan(wake) else f'{wake:>9.3f}'}"
                f"{terms.segments:>10d}{terms.unreachable:>9d}"
                f"{float(terms.completion):>8.3f}{terms.rmsd:>7.2f} A"
                f"{topo.n_residues:>7d}"
            )
            totals["alive"] += terms.alive_fraction
            totals["frontier"] += terms.frontier
            totals["wake"] += 0.0 if math.isnan(wake) else wake
            totals["segments"] += terms.segments
            totals["unreach"] += terms.unreachable
            totals["done"] += float(terms.completion)
            totals["slots"] += topo.n_residues
            n += 1

    if not n:
        print("nothing scored")
        return 1
    print(
        f"\n  {'mean':>6}{totals['alive'] / n:>7.1%}{totals['frontier'] / n:>10.1f}"
        f"{totals['wake'] / n:>9.3f}{totals['segments'] / n:>10.1f}"
        f"{totals['unreach'] / n:>9.1f}{totals['done'] / n:>8.3f}"
        f"{'':>9}{totals['slots'] / n:>7.1f}"
    )
    print(
        "\n  How to read it. A near-zero `done` with `frontier` at or near 0 means\n"
        "  the completion term saw nothing and cannot object to a half-built\n"
        "  structure -- the objective needs a term that charges for dormant slots.\n"
        "  A near-zero `done` with a real frontier and `wake p` near 1 means the\n"
        "  rule already wants to grow everywhere it can and the limit is\n"
        "  elsewhere, so such a term would push on a maximised quantity.\n"
        "  `unreach` counts slots with no covalently linked alive neighbour: if it\n"
        "  is large the stall is permanent from this state, and if it is ~0 the\n"
        "  structure could still have grown and chose not to."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
