"""Find the *first* non-finite tensor in a training run and name it.

The run on Bunya died at ``cKDTree: data must be finite``, which is the
neighbour search complaining about coordinates it was handed. That is a report
from the scene of the symptom: by then the rule has already emitted NaN, which
means either its weights are gone or some forward pass produced NaN from finite
inputs, and the two have different fixes. Worse, the sample pool stores states
the rule produced, so the update that fails need not be the update that erred.

So this script does three things ``train.py`` deliberately does not:

* **Hooks every leaf module** and stops on the first non-finite output, inside
  the forward pass, with the module's own name. A traceback from the hook
  points at the layer rather than at scipy.
* **Wraps ``step_once``** so the rollout step index is known, and records the
  largest magnitude seen in the hidden state, the mean angle and the
  coordinates. Divergence before the overflow is the thing worth seeing: a
  growing-NCA cell has nothing normalising its hidden state, and "it went NaN"
  and "it grew by 1.4x a step for twenty steps" are the same event described
  from different distances.
* **Prints the maxima as a trajectory**, not just at the failure, because if
  the quantity is already climbing at update 10 the answer does not need the
  failure at all.

It is a diagnostic, not a trainer: no checkpointing, no evaluation, no pool
re-seeding unless asked. Run it with ``-u`` so the trajectory arrives while it
is still running.

    python -u scripts/find_nan.py --updates 2000 --noop-weight 4
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aanca import rollout as rollout_module  # noqa: E402
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import RolloutConfig, SamplePool, training_objective  # noqa: E402


class FirstNonFinite(Exception):
    """Raised inside the forward pass, where the information still exists."""


def load_train_module():
    """``scripts/train.py`` as a module, for its dataset wiring.

    Imported rather than copied so this script cannot drift from the run it is
    meant to explain -- a probe that builds a slightly different dataset is
    worse than no probe, because it answers a question nobody asked.
    """
    spec = importlib.util.spec_from_file_location("aanca_train", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def describe(tensor: torch.Tensor) -> str:
    finite = tensor[torch.isfinite(tensor)]
    nan = int(torch.isnan(tensor).sum())
    inf = int(torch.isinf(tensor).sum())
    largest = float(finite.abs().max()) if finite.numel() else float("nan")
    return (
        f"shape {tuple(tensor.shape)}, {nan} NaN, {inf} inf, "
        f"largest finite |x| {largest:.4g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--structures", type=int, default=400)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--truncate", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--pool-probability", type=float, default=0.5)
    parser.add_argument("--noop-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument(
        "--scan",
        action="store_true",
        help="one update per structure at fixed weights: is any entry itself toxic?",
    )
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    train = load_train_module()
    splits = read_manifest(train.MANIFEST)
    if args.split not in splits:
        print(f"no split {args.split!r}; available: {sorted(splits)}")
        return 1
    available = train.on_disk(splits[args.split])
    if not available:
        print(f"no coordinates on disk for {args.split!r}")
        return 1

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    dataset = RepackingDataset(
        available,
        concentrations=(100.0, 20.0, 5.0, 0.0),
        crop_size=args.crop_size,
        device="cpu",
        cache_limit=args.structures + 16,
    )
    indices = train.usable(dataset, args.structures, rng)
    print(f"{len(indices)} structures from {args.split}; {args.updates} updates")

    rule = AANCARule()
    optimiser = torch.optim.Adam(rule.parameters(), lr=args.lr)
    pool = SamplePool(capacity=256, seed=args.seed)
    config = RolloutConfig(
        steps=args.steps,
        truncate=args.truncate,
        k=args.k,
        update_probability=args.pool_probability,
    )

    #: What the forward pass was doing when something went wrong. A plain dict
    #: rather than an exception payload because the hook that fires is several
    #: frames below the code that knows the update number.
    context: dict = {"update": 0, "step": None, "from_pool": False}
    worst = {"hidden": 0.0, "mu": 0.0, "log_kappa": 0.0, "coords": 0.0, "moved": 0.0}

    def hook(name: str):
        def check(_module, _inputs, output):
            tensors = output if isinstance(output, tuple) else (output,)
            for position, tensor in enumerate(tensors):
                if not isinstance(tensor, torch.Tensor):
                    continue
                if torch.isfinite(tensor).all():
                    continue
                raise FirstNonFinite(
                    f"{name} output[{position}]: {describe(tensor)}\n"
                    f"    update {context['update']}, rollout step "
                    f"{context['step']}, seeded from "
                    f"{'the pool' if context['from_pool'] else 'fresh damage'}"
                )
        return check

    for name, module in rule.named_modules():
        if not list(module.children()):
            module.register_forward_hook(hook(name or "<root>"))

    original_step = rollout_module.step_once

    def watched_step(rule_, state, topo, config_, mask):
        context["step"] = state.step
        # The absolute coordinate magnitude says little -- it is mostly where
        # the entry happens to sit in its own frame. **Displacement from the
        # state the rollout started in** is the quantity with a physical
        # bound: these are side-chain torsions, so no atom should move more
        # than the length of its own side chain, about 8 A. A number far above
        # that is the geometry reconstruction failing, and it is visible long
        # before anything overflows.
        if state.step == 0:
            context["origin"] = state.coords.detach().clone()
        origin = context.get("origin")
        if origin is not None and origin.shape == state.coords.shape:
            with torch.no_grad():
                moved = (state.coords.detach() - origin).abs()
                worst["moved"] = max(worst["moved"], _finite_max(moved))
        worst["hidden"] = max(worst["hidden"], _finite_max(state.hidden))
        worst["coords"] = max(worst["coords"], _finite_max(state.coords))
        new_state, mu, log_kappa = original_step(rule_, state, topo, config_, mask)
        worst["mu"] = max(worst["mu"], _finite_max(mu))
        worst["log_kappa"] = max(worst["log_kappa"], _finite_max(log_kappa))
        return new_state, mu, log_kappa

    rollout_module.step_once = watched_step

    if args.scan:
        # One update per structure, with the *same untrained weights* every
        # time, so the only thing varying is the entry. If the data alone can
        # produce an enormous gradient, it shows up here in a couple of
        # minutes and the answer needs no training run at all; if every entry
        # is well behaved, the cause is something the weights do over time,
        # and the long run is the only way to see it.
        reference = {k: v.detach().clone() for k, v in rule.state_dict().items()}
        rows = []
        for position, index in enumerate(indices, start=1):
            rule.load_state_dict(reference)
            context.update({"update": position, "from_pool": False})
            sample = dataset.sample(index, generator=generator, rng=rng)
            try:
                terms, _result = training_objective(
                    rule, sample.damaged, sample.native, sample.topology,
                    config, generator=generator, rng=rng,
                    noop_weight=args.noop_weight,
                )
                rule.zero_grad()
                terms.total.backward()
                norm = float(
                    torch.nn.utils.clip_grad_norm_(rule.parameters(), float("inf"))
                )
                rows.append((norm, index, dataset.specs[index].pdb_id, float(terms.total)))
            except FirstNonFinite as failure:
                rows.append((float("inf"), index, dataset.specs[index].pdb_id, str(failure)))
            if position % 50 == 0:
                print(f"  scanned {position}/{len(indices)}", flush=True)

        rows.sort(key=lambda row: (row[0] != row[0], -row[0]))
        print("\nworst gradient norms, untrained weights, one update each:")
        for norm, index, pdb_id, loss in rows[:15]:
            print(f"  {pdb_id:>6} (index {index:>6}): grad {norm:>12.4g}  loss {loss}")
        finite = [row[0] for row in rows if row[0] == row[0] and row[0] != float("inf")]
        if finite:
            finite.sort()
            print(
                f"\n{len(finite)}/{len(rows)} finite; median {finite[len(finite)//2]:.4g}, "
                f"max {finite[-1]:.4g}"
            )
        rollout_module.step_once = original_step
        return 0

    try:
        for update in range(1, args.updates + 1):
            context["update"] = update
            index = rng.choice(indices)
            sample = dataset.sample(index, generator=generator, rng=rng)

            seed_state = None
            if rng.random() < args.pool_probability:
                seed_state = pool.sample(index)
            from_pool = (
                seed_state is not None
                and seed_state.coords.shape == sample.damaged.shape
            )
            context["from_pool"] = from_pool
            start = seed_state.coords if from_pool else sample.damaged

            if not torch.isfinite(start).all():
                print(f"\nupdate {update}: the START state is already non-finite")
                print(f"  {describe(start)}")
                print(f"  seeded from {'the pool' if from_pool else 'fresh damage'}")
                return 2

            terms, result = training_objective(
                rule,
                start,
                sample.native,
                sample.topology,
                config,
                generator=generator,
                rng=rng,
                noop_weight=args.noop_weight,
            )
            if not torch.isfinite(terms.total):
                print(f"\nupdate {update}: the LOSS is non-finite")
                print(f"  total {float(terms.total)}")
                for field in ("reconstruction", "noop", "restraint", "total"):
                    value = getattr(terms, field, None)
                    if isinstance(value, torch.Tensor):
                        print(f"  {field:>14}: {float(value)}")
                print(f"  worst so far: {worst}")
                return 3

            optimiser.zero_grad()
            terms.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                print(f"\nupdate {update}: the GRADIENT is non-finite")
                print("  the loss was finite, so this came out of the backward pass")
                bad = [
                    n for n, p in rule.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                print(f"  parameters with non-finite grad: {bad}")
                print(f"  worst so far: {worst}")
                return 4
            optimiser.step()

            if not all(torch.isfinite(p).all() for p in rule.parameters()):
                print(f"\nupdate {update}: the WEIGHTS went non-finite on the step")
                print(f"  grad norm {float(grad_norm):.4g}, worst so far: {worst}")
                return 5

            pool.insert(index, result.state)

            if update % args.report_every == 0:
                print(
                    f"  {update:>6}  loss {float(terms.total.detach()):>9.3f}  "
                    f"grad {float(grad_norm):>9.3f}  "
                    f"max|hidden| {worst['hidden']:>8.4g}  "
                    f"max|mu| {worst['mu']:>8.4g}  "
                    f"max moved {worst['moved']:>10.4g} A"
                )
    except FirstNonFinite as failure:
        print(f"\nfirst non-finite tensor, inside the forward pass:\n  {failure}")
        print(f"  worst finite magnitudes before this: {worst}")
        return 6
    finally:
        rollout_module.step_once = original_step

    print(f"\n{args.updates} updates, nothing non-finite. worst: {worst}")
    return 0


def _finite_max(tensor: torch.Tensor) -> float:
    with torch.no_grad():
        finite = tensor[torch.isfinite(tensor)]
        return float(finite.abs().max()) if finite.numel() else float("inf")


if __name__ == "__main__":
    raise SystemExit(main())
