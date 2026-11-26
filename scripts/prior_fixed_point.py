"""Does the DENOISER reach a fixed point from the pure Gaussian prior?

    python scripts/prior_fixed_point.py --checkpoint runs/mvp/best.pt
    python scripts/prior_fixed_point.py --tolerance 0.1 --max-steps 500

**This runs the registered bucket-1 test at its registered numbers, which the
trainer does not.** `docs/claims/a-local-rule-with-an-attractor-is-a-diffusion
-model-without-a-schedule.md` registers *"max per-step displacement < 0.1 A
within 500 steps"*. `scripts/train_denoise.py` samples at its own
`--sample-tolerance 0.01` and `--sample-max-steps 200` -- **stricter on both
axes** -- so `conv 0/7` in the training log is not the registered measurement
and reporting it as one would be answering a question nobody asked. The
registered numbers are this script's defaults for exactly that reason.

The distinction this probe exists to keep sharp. `scripts/persistence_readout.py`
answered bucket 1 YES for the port: damage a 12-residue window, hold ~116
residues at their crystal pose, and the rule returns to the same endpoint within
0.0105 A. That is a fixed point of a *conditional* map, with the answer pinned
at its boundary. Option 2 needs a fixed point of the *unconditional* map --
nothing held, the graph rebuilt from where the residues currently are, the start
state a Gaussian cloud at `2.2 N^0.38` A. **Two different basins, and only the
second one makes the rule a generator.** Nothing else in this project measures
the second.

Every rollout setting is read out of the checkpoint and checked key-by-key
against `scripts/train_denoise.py`'s own schema, through
`scripts/persistence_readout.py`'s `settings_from_checkpoint`, `load_rule` and
`held_out` -- the same four refusals, for the same reason, recorded in
`docs/claims/a-probe-pointed-at-best-pt-silently-used-its-own-defaults.md`.
`--tolerance` and `--max-steps` are flags because they are *this probe's
question* rather than the run's, and `--native-edges` is not a flag at all: it
comes from the checkpoint, because a sample from the prior message-passing over
the native contact graph would be given the answer.

**`n` and `diverged` are printed before anything else**, this project's standing
rule: `last_move` is `nan` on a non-finite state and a mean over the samples
that stayed finite would be the best-looking number in the table.

`last_move` is the column that says how *close* a non-converged sample came, and
it is the one that decides whether the cap or the tolerance is what refuted the
claim. A rule stalling at 8 A per step has no fixed point; a rule at 0.11 A per
step has one just outside the window.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _module(name: str):
    """A `scripts/` file as a module -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/mvp/best.pt")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="the registered threshold, A per step (max over residues)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="the registered cap",
    )
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.add_argument("--device", default="cpu")
    return p


def main() -> int:
    options = build_parser().parse_args()
    probe = _module("persistence_readout")
    trainer = _module("train_denoise")
    device = torch.device(options.device)

    path = Path(options.checkpoint)
    if not path.exists():
        print(f"no checkpoint at {path}", file=sys.stderr)
        return 1
    state = torch.load(path, map_location="cpu", weights_only=False)

    try:
        args = probe.settings_from_checkpoint(state, trainer.build_schema())
        rule = probe.load_rule(state, args, trainer, device)
        dataset, catalogue, eval_indices = probe.held_out(
            state, args, trainer, device
        )
    except probe.ProbeRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 1

    params = sum(p.numel() for p in rule.parameters())
    print(
        f"checkpoint  {path}  update {state.get('update')}  params {params}\n"
        f"recorded    context_fraction {args.context_fraction}  "
        f"native_edges {args.native_edges}  k {args.k}  cutoff {args.cutoff}\n"
        f"registered  tolerance {options.tolerance:g} A/step, "
        f"cap {options.max_steps} steps  ({options.samples} samples)\n"
    )

    args.sample_count = options.samples
    samples = []
    for position in range(options.samples):
        index = eval_indices[position % len(eval_indices)]
        rng = random.Random(options.seed * 7_919 + position)
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 3:
            continue
        generator = torch.Generator().manual_seed(options.seed * 7_919 + position)
        start = trainer.prior_coordinates(
            topo.n_residues,
            generator=generator,
            dtype=rule.dtype,
            device=native.device,
        )
        sample = trainer.run_to_fixed_point(
            rule,
            topo,
            native,
            start,
            tolerance=options.tolerance,
            max_steps=options.max_steps,
            k=args.k,
            cutoff=args.cutoff,
            device_neighbours=args.device_neighbours,
            native_edges=args.native_edges,
        )
        samples.append(
            {
                "pdb_id": catalogue[index].pdb_id,
                "n_residues": int(topo.n_residues),
                "steps": sample.steps,
                "converged": bool(sample.converged),
                "last_move": sample.last_move,
                "rg": sample.radius_of_gyration,
            }
        )

    finite = [s for s in samples if s["last_move"] == s["last_move"]]
    diverged = len(samples) - len(finite)
    reached = [s for s in samples if s["converged"]]
    print(f"n {len(samples)}   diverged {diverged}   converged {len(reached)}\n")
    print("  pdb     res  steps  conv   last_move      rg")
    for s in samples:
        print(
            f"  {s['pdb_id']:6s} {s['n_residues']:4d} {s['steps']:6d}"
            f"  {'YES' if s['converged'] else ' no'}"
            f"  {s['last_move']:10.4f} {s['rg']:7.2f}"
        )

    if finite:
        moves = [s["last_move"] for s in finite]
        print(
            f"\nlast_move over {len(finite)} finite samples: "
            f"median {statistics.median(moves):.4f}  min {min(moves):.4f}  "
            f"max {max(moves):.4f} A/step"
        )
        print(
            f"the registered tolerance is {options.tolerance:g} A/step, so the "
            f"median sample is {statistics.median(moves) / options.tolerance:.0f}x "
            "outside it"
        )

    verdict = (
        "YES -- a fixed point from the prior"
        if reached and len(reached) == len(samples)
        else f"NO -- {len(samples) - len(reached)} of {len(samples)} never settle"
    )
    print(f"\nBUCKET 1, unconditional: {verdict}")

    if options.json:
        Path(options.json).write_text(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "update": state.get("update"),
                    "tolerance": options.tolerance,
                    "max_steps": options.max_steps,
                    "n": len(samples),
                    "diverged": diverged,
                    "converged": len(reached),
                    "samples": samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
