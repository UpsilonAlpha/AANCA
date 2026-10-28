"""Train the rule, and judge it on held-out accuracy rather than on the loss.

    python scripts/train.py --updates 2000 --eval-every 200
    python scripts/train.py --updates 50 --eval-every 25 --eval-structures 4

**The training loss is not the progress signal.** Its target is measured from
the rule's own input at the supervised step, so the question gets easier as the
rule improves and a falling loss conflates that with better prediction. See
``src/aanca/evaluate.py``. Everything reported as progress here comes from a
fixed held-out set at fixed difficulty, and the loss is printed only to show
that optimisation is happening at all.

Three numbers decide whether the idea works, and they can fail separately:

* **chi MAE and accuracy** against native, per damage level, in the published
  layout. Compared against the MAE of the *damaged input*, because at kappa=100
  the input is already close and an unimpressive rule would look good.
* **mean |delta| per step**, which must decay. A rule that oscillates can still
  score well at the step it was supervised at, and that is precisely the
  failure a fixed-point objective exists to rule out.
* **drift from a native start**, the "does it stay" half. An untrained
  identity-initialised rule scores a perfect zero, so it only means something
  once training has moved the weights.

Checkpoints are kept by best held-out pooled chi1 MAE, not by last.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from aanca.data import RepackingDataset, read_manifest
from aanca.evaluate import build_eval_set, evaluate
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, SamplePool, initial_state, training_objective

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
RUNS = ROOT / "runs"


def on_disk(specs):
    """Specs whose coordinates are present, with the path pointed at the file."""
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def usable(dataset, limit: int, rng: random.Random) -> list[int]:
    """Indices that actually load, shuffled.

    Checked up front rather than caught mid-training: the size and padding
    guards reject a couple of percent of real entries, and discovering that
    inside the training loop turns a data problem into a mysterious stall.
    """
    order = list(range(len(dataset)))
    rng.shuffle(order)
    good: list[int] = []
    for index in order:
        if len(good) >= limit:
            break
        try:
            dataset.load(index)
        except Exception:
            continue
        good.append(index)
    return good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--structures", type=int, default=400, help="training pool")
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--truncate", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--pool-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print("no manifest -- run scripts/build_training_manifest.py")
        return 1

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    splits = read_manifest(MANIFEST)

    # The validation split is what is downloaded, so it provides both pools
    # here -- **disjoint by structure**, which is the only property that
    # matters for an honest progress curve. The real training split is a
    # separate 23,761-entry download; this makes the loop runnable before
    # paying for it. A number from this run is a development signal, never a
    # benchmark result.
    available = on_disk(splits["validation"])
    if not available:
        print(f"no coordinates in {PDB} -- run scripts/fetch_structures.py")
        return 1

    dataset = RepackingDataset(
        available,
        concentrations=(100.0, 20.0, 5.0, 0.0),
        crop_size=args.crop_size,
    )
    wanted = args.structures + args.eval_structures
    indices = usable(dataset, wanted, rng)
    if len(indices) < args.eval_structures + 1:
        print(f"only {len(indices)} structures load; need more")
        return 1
    eval_indices = indices[: args.eval_structures]
    train_indices = indices[args.eval_structures :]
    print(
        f"{len(train_indices)} training structures, "
        f"{len(eval_indices)} held out (disjoint)"
    )

    rule = AANCARule()
    n_params = sum(p.numel() for p in rule.parameters())
    optimiser = torch.optim.Adam(rule.parameters(), lr=args.lr)
    pool = SamplePool(capacity=256, seed=args.seed)

    train_config = RolloutConfig(
        steps=args.steps, truncate=args.truncate, k=64, cutoff=10.0
    )
    eval_config = RolloutConfig(
        steps=args.eval_steps, truncate=args.truncate, k=64, cutoff=10.0
    )

    print("building the held-out set (fixed crops, fixed damage)...")
    cases = build_eval_set(dataset, eval_indices, concentrations=(100.0, 20.0, 0.0))
    print(f"{len(cases)} cases over {len(eval_indices)} structures; {n_params} parameters")

    RUNS.mkdir(exist_ok=True)
    out_dir = RUNS / args.tag
    out_dir.mkdir(exist_ok=True)
    history: list[dict] = []
    best = float("inf")
    started = time.perf_counter()

    baseline = evaluate(rule, cases, eval_config)
    print("\nuntrained:")
    print(baseline.report())
    print(
        "  damaged-input chi MAE: "
        + ", ".join(f"{k} {v:.1f} deg" for k, v in sorted(baseline.input_mae.items()))
    )

    print(f"\n{args.updates} updates")
    recent: list[float] = []
    for update in range(1, args.updates + 1):
        index = rng.choice(train_indices)
        sample = dataset.sample(index, generator=generator, rng=rng)

        # Growing-NCA re-seeding: sometimes start from a state the rule
        # produced itself rather than from fresh damage, which is what asks it
        # to *stay* converged rather than merely arrive once.
        seed_state = None
        if rng.random() < args.pool_probability:
            seed_state = pool.sample(index)
        start = (
            seed_state.coords
            if seed_state is not None
            and seed_state.coords.shape == sample.damaged.shape
            else sample.damaged
        )

        terms, result = training_objective(
            rule,
            start,
            sample.native,
            sample.topology,
            train_config,
            generator=generator,
            rng=rng,
        )
        optimiser.zero_grad()
        terms.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)
        optimiser.step()
        pool.insert(index, result.state)
        recent.append(float(terms.total.detach()))

        if update % max(args.eval_every // 4, 1) == 0:
            print(
                f"  {update:>6}  loss {sum(recent) / len(recent):>7.3f}  "
                f"grad {float(grad_norm):>7.3f}  "
                f"{(time.perf_counter() - started) / update:>5.2f} s/update"
            )
            recent.clear()

        if update % args.eval_every == 0 or update == args.updates:
            result = evaluate(rule, cases, eval_config)
            print(f"\nheld out after {update} updates:")
            print(result.report())
            chi1 = result.rows["pooled"].chi_mae[0]
            history.append(
                {
                    "update": update,
                    "pooled_chi1_mae": chi1,
                    "pooled_accuracy": result.rows["pooled"].accuracy,
                    "pooled_rmsd": result.rows["pooled"].rmsd,
                    "convergence": result.convergence,
                    "native_drift": result.native_drift,
                    "seconds": time.perf_counter() - started,
                }
            )
            (out_dir / "history.json").write_text(
                json.dumps(
                    {
                        "args": vars(args),
                        "input_mae": baseline.input_mae,
                        "history": history,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if chi1 == chi1 and chi1 < best:
                best = chi1
                torch.save(
                    {"state_dict": rule.state_dict(), "update": update, "chi1": chi1},
                    out_dir / "best.pt",
                )
                print(f"  new best pooled chi1 MAE {chi1:.2f} deg -- checkpointed")
            print()

    print(f"done in {(time.perf_counter() - started) / 60:.1f} min; {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
