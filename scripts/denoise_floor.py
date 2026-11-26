"""What does the denoising RMSD read for a rule that has learned NOTHING?

    python scripts/denoise_floor.py --checkpoint runs/mvp/best.pt

**`scripts/train_denoise.py` prints `rmsd` against `start` and nothing else, so
`15.205 -> 9.1 A` reads as a 40 % improvement.** It is only an improvement
against a baseline, and the trainer carries none. This script supplies three,
all weight-free, on the run's own held-out crops at the run's own `sigma`:

  * **`start`** -- the corrupted coordinates, untouched. The trainer's own
    column, recomputed here as the check that this script is scoring the same
    quantity on the same crops.
  * **`centroid`** -- shrink every coordinate toward its own centroid by the
    single scalar that makes the radius of gyration match the native's.
    Isotropic noise inflates Rg by a known factor, so **this estimator uses one
    number about the target and no model at all.**
  * **`oracle_scale`** -- the best such scalar found by search rather than from
    Rg. An upper bound on everything a pure radial rescaling can do, so a rule
    that does not beat it has learned no angular information whatsoever: no
    chain, no contacts, no fold. Only a size.

**Why this matters more than it sounds.** A diffusion model's job is to turn a
Gaussian cloud into a protein. A rule that matches the Rg of a protein and gets
the angles from the prior has done nothing a one-line rescaling cannot, and its
loss curve will still fall steeply in the first few hundred updates and then sit
flat -- which is exactly the shape
`docs/claims/a-local-rule-with-an-attractor-is-a-diffusion-model-without-a-schedule.md`
has to interpret. Without this floor that shape is ambiguous. With it, it is not.

Every setting is read out of the checkpoint through
`scripts/persistence_readout.py`'s `settings_from_checkpoint` and `held_out`, so
the crops are the run's own held-out crops and the pool check refuses a mismatch
(`docs/claims/a-probe-pointed-at-best-pt-silently-used-its-own-defaults.md`).
The rule's weights are never loaded: nothing here is a measurement of the rule,
which is the point.

**`--trainer` IS DECLARED AND NEVER DETECTED.**
`scripts/train_conditioned.py`'s checkpoints record two keys --- `label_dim`
and `label_dropout` --- that `scripts/train_denoise.py`'s schema does not
define, so pointing this script at one without saying so is a refusal, and
that refusal is the mechanism of the claim cited above. Detecting the trainer
from the recorded keys would make every checkpoint readable and the check
vacuous, so the trainer is named:

    python scripts/denoise_floor.py --trainer train_conditioned \
        --checkpoint runs/cond/best.pt

Nothing else about this script changes for the conditioned arm, and that is a
result rather than a convenience: **the floor is weight-free, so the label
cannot reach it.** The corruption, the crops and the two rescalings are the
same functions on the same held-out split at the same sigma, which is exactly
what makes the conditioned arm's `rmsd` comparable with the MVP's against one
shared floor.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aanca.egnca import ca_positions, with_ca_positions  # noqa: E402
from aanca.fape import backbone_rmsd_on  # noqa: E402


def _module(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


#: The two trainers whose checkpoints this script can honour. A closed set: the
#: schema a checkpoint is checked against is a declaration, not a guess.
TRAINERS = ("train_denoise", "train_conditioned")


def radius_of_gyration(x: torch.Tensor) -> torch.Tensor:
    centred = x - x.mean(dim=0, keepdim=True)
    return centred.pow(2).sum(-1).mean().clamp_min(1e-12).sqrt()


def rescaled(x: torch.Tensor, factor: float) -> torch.Tensor:
    centre = x.mean(dim=0, keepdim=True)
    return centre + (x - centre) * factor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/mvp/best.pt")
    p.add_argument("--crops", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="default: the mean sigma the run's own history records",
    )
    p.add_argument("--json", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--trainer",
        default="train_denoise",
        choices=TRAINERS,
        help=(
            "which trainer's schema the checkpoint is checked against, and "
            "whose `corrupt` and `on_disk` are used. DECLARED, never detected"
        ),
    )
    return p


def main() -> int:
    options = build_parser().parse_args()
    probe = _module("persistence_readout")
    trainer = _module(options.trainer)
    device = torch.device(options.device)

    path = Path(options.checkpoint)
    if not path.exists():
        print(f"no checkpoint at {path}", file=sys.stderr)
        return 1
    state = torch.load(path, map_location="cpu", weights_only=False)

    try:
        args = probe.settings_from_checkpoint(state, trainer.build_schema())
        dataset, catalogue, eval_indices = probe.held_out(
            state, args, trainer, device
        )
    except probe.ProbeRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 1

    sigma = options.sigma
    if sigma is None:
        history = path.parent / "history.json"
        if not history.exists():
            print(
                "no --sigma and no history.json beside the checkpoint, so the "
                "noise level would be this script's own guess",
                file=sys.stderr,
            )
            return 1
        rows = json.loads(history.read_text(encoding="utf-8").replace("NaN", "null"))
        levels = {round(float(r["sigma"]), 9) for r in rows if r.get("sigma")}
        if len(levels) != 1:
            print(
                f"history.json records {len(levels)} distinct sigma values "
                f"{sorted(levels)}; the eval corruption is not single-valued and "
                "the rows the trainer printed are not comparable with each other, "
                "let alone with this floor. Pass --sigma explicitly.",
                file=sys.stderr,
            )
            return 1
        sigma = levels.pop()

    print(
        f"checkpoint  {path}  update {state.get('update')}  (weights NOT loaded)\n"
        f"trainer     scripts/{options.trainer}.py  (declared, not detected)\n"
        f"sigma       {sigma:.4f} A per coordinate, from "
        f"{'--sigma' if options.sigma else 'history.json'}\n"
        f"crops       {options.crops} of the run's own held-out split\n"
    )

    rows = []
    for position in range(options.crops):
        index = eval_indices[position % len(eval_indices)]
        rng = random.Random(options.seed * 7_919 + position)
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        # The trainer's own accessors and the trainer's own corruption, so the
        # `start` column below has to reproduce the one in the log.
        ca = ca_positions(native, topo).detach()
        everything = torch.ones(topo.n_residues, dtype=torch.bool)
        corrupted = torch.ones(topo.n_residues, dtype=torch.bool)
        generator = torch.Generator().manual_seed(options.seed * 7_919 + position)
        with torch.no_grad():
            start = trainer.corrupt(ca, sigma, corrupted, generator=generator)

        def score(x):
            placed = with_ca_positions(native, x, topo)
            return float(
                backbone_rmsd_on(placed, native, topo, everything, corrupted)
            )

        target_rg = float(radius_of_gyration(ca))
        start_rg = float(radius_of_gyration(start))
        factor = target_rg / max(start_rg, 1e-9)

        best, best_f = float("inf"), None
        for i in range(1, 161):
            f = i / 80.0
            v = score(rescaled(start, f))
            if v < best:
                best, best_f = v, f

        rows.append(
            {
                "pdb_id": catalogue[index].pdb_id,
                "n": int(len(ca)),
                "start": score(start),
                "centroid": score(rescaled(start, factor)),
                "factor": factor,
                "oracle_scale": best,
                "oracle_factor": best_f,
            }
        )

    if not rows:
        print("no crops loaded", file=sys.stderr)
        return 1

    print("  pdb      n     start  centroid  factor  oracle  o-factor")
    for r in rows:
        print(
            f"  {r['pdb_id']:6s} {r['n']:4d} {r['start']:9.3f} {r['centroid']:9.3f}"
            f" {r['factor']:7.3f} {r['oracle_scale']:7.3f} {r['oracle_factor']:9.3f}"
        )

    def mean(key):
        return statistics.mean(r[key] for r in rows)

    print(
        f"\nn {len(rows)}\n"
        f"  start         {mean('start'):7.3f} A   the trainer's own `start` column\n"
        f"  centroid      {mean('centroid'):7.3f} A   Rg-matched rescaling, no model\n"
        f"  oracle_scale  {mean('oracle_scale'):7.3f} A   the best radial rescaling there is\n"
    )
    print(
        "READ THE RULE'S `rmsd` AGAINST `oracle_scale`, not against `start`. A "
        "rule at or above the oracle has learned a size and no angles."
    )

    if options.json:
        Path(options.json).write_text(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "trainer": options.trainer,
                    "sigma": sigma,
                    "n": len(rows),
                    "start": mean("start"),
                    "centroid": mean("centroid"),
                    "oracle_scale": mean("oracle_scale"),
                    "crops": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
