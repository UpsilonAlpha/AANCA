"""Score a trained rule's identity head on the exact held-out set its run used.

    python scripts/score_identity.py runs/v8c_if/last.pt
    python scripts/score_identity.py runs/v8c_if/last.pt --weights runs/v8c_if/best.pt

**Why this exists.** v8c_if trained inverse folding for 20,000 updates and
produced **no held-out identity number at all** — ``evaluate`` was carrying no
labels, so every table it printed was packing-only, and the only evidence the
head had learned was the ``id`` column in the progress line. That column is a
running mean over *training* updates, and with 400 structures seen about fifty
times each under different crops, "this backbone is structure X, whose residue
47 is LEU" is a memorisable fact. So the one number that mattered was the one
nothing measured.

``evaluate`` now carries identity and reports joint NLL, type-marginal NLL and
sequence recovery. This script applies that to a finished run without spending
another seven GPU-hours to re-derive it.

**The held-out set is read from the checkpoint, not reconstructed.** The split
comes from a shuffle and a load-success filter, so ``train.py`` stores
``eval_indices`` rather than trusting a reseed to reproduce it
(``aanca.checkpoint``'s module docstring argues the point). Rebuilding the set
from ``--seed`` would quietly score a *different* twelve structures, some of
which the run trained on — which is the same class of error as the thing this
script exists to correct, one level up.

``--weights`` scores a different file's parameters against those indices, which
is what ``best.pt`` needs: it holds the weights alone, under ``state_dict``,
and carries no split.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.evaluate import build_eval_set, evaluate
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
    parser.add_argument("checkpoint", help="a last.pt, which carries the split")
    parser.add_argument(
        "--weights",
        default=None,
        help="score these parameters instead (e.g. best.pt, which has no split)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    state = ckpt.load(Path(args.checkpoint), map_location="cpu")
    stored = state.get("args") or {}
    if "eval_indices" not in state:
        print(
            f"{args.checkpoint} carries no eval_indices, so the held-out set "
            "cannot be recovered. Pass a last.pt from the run itself."
        )
        return 1
    eval_indices = list(state["eval_indices"])
    if not stored.get("identity_weight"):
        print(
            f"warning: this run has identity_weight "
            f"{stored.get('identity_weight')!r}, so there may be no head to score"
        )

    split = stored.get("split", "validation")
    splits = read_manifest(MANIFEST)
    # **The dataset spans the WHOLE split, exactly as train.py builds it.** Its
    # ``usable()`` helper shuffles ``range(len(dataset))`` over the full
    # available list and keeps the first that load, so a stored index is a
    # position in *that* list. Truncating the pool to ``structures +
    # eval_structures`` -- which looks like the obvious thing, since that is how
    # many the run used -- renumbers every index and scores a different set of
    # structures, some of which the run trained on. It raised an IndexError
    # here rather than scoring the wrong twelve silently, which was luck.
    pool = on_disk(splits[split])

    dataset = RepackingDataset(
        pool,
        concentrations=(100.0, 20.0, 5.0, 0.0),
        damage_fractions=tuple(stored.get("damage_fractions", (1.0,))),
        crop_size=int(stored.get("crop_size", 384)),
        device=device,
        cache_limit=max(len(pool), 1),
    )
    cases = build_eval_set(dataset, eval_indices, concentrations=(100.0, 20.0, 0.0))

    rule = AANCARule(
        identity=bool(stored.get("identity_weight")),
        identity_keep=float(stored.get("identity_keep", 8.0)),
    ).to(device)
    weights = Path(args.weights) if args.weights else Path(args.checkpoint)
    rule.load_state_dict(ckpt.rule_weights(ckpt.load(weights, map_location="cpu")))

    # ``or``-chained, not ``.get(key, default)``: ``--steps-max`` defaults to
    # **0** meaning "unset, use --steps", so a stored 0 is *present* and falsy
    # rather than absent, and a default only fires on absence. Reading it as a
    # real value gave a zero-step rollout, which never calls ``step_once`` --
    # so it produced no identity logits, reported "the rule has no identity
    # head" when the head was built and correct, and printed the *damaged
    # inputs* as though they were predictions. One root cause, three symptoms,
    # and the misleading one was the message: it sent me to the model instead
    # of the rollout length.
    steps = (
        args.steps
        or int(stored.get("steps_max") or 0)
        or int(stored.get("steps") or 0)
        or 30
    )
    if steps < 1:
        print(f"refusing to score with {steps} steps")
        return 1
    config = RolloutConfig(
        steps=steps,
        truncate=int(stored.get("truncate", 12)),
        k=int(stored.get("k", 64)),
        blind=bool(stored.get("identity_weight")),
        identity_start=str(stored.get("identity_start", "masked")),
        identity_sampling=str(stored.get("identity_sampling", "sample")),
        identity_temperature=float(stored.get("identity_temperature", 1.0)),
        update_schedule=str(stored.get("update_schedule", "bernoulli")),
        coupling_radius=float(stored.get("coupling_radius", 5.0)),
        clash_weight=float(stored.get("clash_weight", 0.0)),
    )

    print(
        f"{weights} on {split}'s held-out {len(eval_indices)} structures "
        f"({len(cases)} cases), {steps} steps, {device}"
    )
    print(
        f"update {state.get('update', '?')}, blind={config.blind}, "
        f"identity_start={config.identity_start}, "
        f"schedule={config.update_schedule}\n"
    )
    result = evaluate(rule, cases, config)
    print(result.report())

    if result.recovery is not None:
        # The floors, spelled out, because a NLL means nothing without them and
        # a recovery means nothing without the background composition.
        print(
            f"\n  joint NLL floor ln 54 = {math.log(54):.3f}; "
            f"type NLL floor ln 20 = {math.log(20):.3f}."
        )
        print(
            "  Sequence recovery is comparable with PottsMPNN (~44 %) and "
            "ProteinMPNN (~58 %)\n  only in kind, not on this set: those are "
            "published on their own benchmarks."
        )
    elif not getattr(rule, "identity", False):
        print("\n  no identity numbers: this rule has no identity head.")
    else:
        print(
            "\n  no identity numbers: the rule HAS a head, but the rollout "
            "produced no\n  logits -- which means it ran zero steps. Check "
            "--steps and the stored steps_max."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
