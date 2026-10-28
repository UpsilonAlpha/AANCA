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

**Interruption is the normal case, not an error.** On a shared cluster the run
ends when the walltime does, or when the job is pre-empted, or when someone
cancels it. So ``runs/<tag>/last.pt`` carries the whole training state --
weights, optimiser, every RNG, the sample pool, the structure split and the
history -- and ``--resume`` continues the same trajectory rather than a
differently-seeded one; see ``src/aanca/checkpoint.py`` for why that
distinction matters to the diagnostics. ``--max-seconds`` stops cleanly before
a walltime does, and SIGTERM/SIGUSR1/Ctrl-C each finish the update in flight,
checkpoint, and exit with code 64 meaning "not finished". Anything else
non-zero is a real failure, which is what lets the batch script resubmit on 64
alone and stop on a crash.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.evaluate import build_eval_set, evaluate
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, SamplePool, training_objective

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
RUNS = ROOT / "runs"

#: Set by the signal handlers. Checked once per update, so a stop always lands
#: between updates with a consistent optimiser state rather than inside a
#: backward pass.
STOP: dict[str, object] = {"asked": False, "why": ""}


def _install_handlers() -> None:
    def handler(signum, _frame):
        STOP["asked"] = True
        STOP["why"] = signal.Signals(signum).name

    for name in ("SIGTERM", "SIGINT", "SIGUSR1", "SIGUSR2"):
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, handler)
            except (ValueError, OSError):
                pass  # not the main thread, or not supported on this platform


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


def resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--pool-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--split", default="validation", help="manifest split to draw from")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--threads", type=int, default=0, help="torch CPU threads; 0 leaves the default"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="updates between full-state checkpoints; the cost of an interruption",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop cleanly after this long (0 = no limit). Set it below the "
        "job walltime so the run checkpoints rather than being killed.",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help="path checked once per update; when it appears the run "
        "checkpoints and exits 0, which ends a resubmission chain. Default "
        "runs/<tag>/STOP.",
    )
    parser.add_argument(
        "--resume",
        choices=("auto", "never"),
        default="auto",
        help="auto continues runs/<tag>/last.pt if it exists",
    )
    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="resume even though the pinned settings differ (joins two runs)",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        print("no manifest -- run scripts/build_training_manifest.py")
        return 1

    _install_handlers()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    splits = read_manifest(MANIFEST)
    if args.split not in splits:
        print(f"no split {args.split!r}; available: {sorted(splits)}")
        return 1

    # The validation split is what is downloaded first, so it can provide both
    # pools here -- **disjoint by structure**, which is the only property that
    # matters for an honest progress curve. The real training split is a
    # separate 23,761-entry download; this makes the loop runnable before
    # paying for it. A number from a run on ``--split validation`` is a
    # development signal, never a benchmark result.
    available = on_disk(splits[args.split])
    if not available:
        print(f"no coordinates in {PDB} -- run scripts/fetch_structures.py")
        return 1

    dataset = RepackingDataset(
        available,
        concentrations=(100.0, 20.0, 5.0, 0.0),
        crop_size=args.crop_size,
        device=device,
    )

    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "STOP"
    if stop_file.exists():
        print(
            f"{stop_file} exists, so this run was stopped on purpose. "
            "Delete it to continue."
        )
        return 0

    rule = AANCARule().to(device)
    optimiser = torch.optim.Adam(rule.parameters(), lr=args.lr)
    pool = SamplePool(capacity=256, seed=args.seed)
    history: list[dict] = []
    best = float("inf")
    start_update = 0
    elapsed_before = 0.0

    state = None
    if args.resume == "auto" and last_path.exists():
        state = ckpt.load(last_path, map_location=device)
        if not args.force_resume:
            ckpt.check_fingerprint(state["fingerprint"], ckpt.fingerprint(args))

    if state is not None:
        train_indices = list(state["train_indices"])
        eval_indices = list(state["eval_indices"])
    else:
        wanted = args.structures + args.eval_structures
        indices = usable(dataset, wanted, rng)
        if len(indices) < args.eval_structures + 1:
            print(f"only {len(indices)} structures load; need more")
            return 1
        eval_indices = indices[: args.eval_structures]
        train_indices = indices[args.eval_structures :]

    print(
        f"device {device}, split {args.split}: {len(train_indices)} training "
        f"structures, {len(eval_indices)} held out (disjoint)"
    )

    n_params = sum(p.numel() for p in rule.parameters())
    train_config = RolloutConfig(
        steps=args.steps, truncate=args.truncate, k=args.k, cutoff=10.0
    )
    eval_config = RolloutConfig(
        steps=args.eval_steps, truncate=args.truncate, k=args.k, cutoff=10.0
    )

    print("building the held-out set (fixed crops, fixed damage)...")
    cases = build_eval_set(dataset, eval_indices, concentrations=(100.0, 20.0, 0.0))
    print(f"{len(cases)} cases over {len(eval_indices)} structures; {n_params} parameters")

    if state is not None:
        # Order matters: the RNGs are restored *after* the deterministic setup
        # above, so nothing between here and the loop can consume a draw.
        rule.load_state_dict(state["rule"])
        optimiser.load_state_dict(state["optimiser"])
        pool.load_state_dict(state["pool"])
        ckpt.restore_python_rng(rng, state["rng_python"])
        ckpt.restore_generator(generator, state["rng_generator"])
        ckpt.set_rng_state(state["rng_global"])
        history = list(state["history"])
        best = float(state["best"])
        start_update = int(state["update"])
        elapsed_before = float(state["elapsed"])
        baseline_input_mae = dict(state["input_mae"])
        print(
            f"resumed {last_path} at update {start_update} "
            f"({elapsed_before / 60:.1f} min spent, best chi1 {best:.2f} deg)"
        )
        if start_update >= args.updates:
            print(f"already at {start_update} of {args.updates} updates -- nothing to do")
            return 0
    else:
        baseline = evaluate(rule, cases, eval_config)
        baseline_input_mae = baseline.input_mae
        print("\nuntrained:")
        print(baseline.report())
        print(
            "  damaged-input chi MAE: "
            + ", ".join(f"{k} {v:.1f} deg" for k, v in sorted(baseline_input_mae.items()))
        )

    started = time.perf_counter()

    def segment_seconds() -> float:
        """Time spent in *this* process. What ``--max-seconds`` governs."""
        return time.perf_counter() - started

    def elapsed() -> float:
        """Time spent on this run across every segment. What gets reported."""
        return elapsed_before + segment_seconds()

    def write_history() -> None:
        (out_dir / "history.json").write_text(
            json.dumps(
                {"args": vars(args), "input_mae": baseline_input_mae, "history": history},
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_checkpoint(update: int) -> None:
        ckpt.save(
            last_path,
            {
                "update": update,
                "elapsed": elapsed(),
                "best": best,
                "fingerprint": ckpt.fingerprint(args),
                "args": vars(args),
                "rule": rule.state_dict(),
                "optimiser": optimiser.state_dict(),
                "pool": pool.state_dict(),
                "rng_python": rng.getstate(),
                "rng_generator": generator.get_state(),
                "rng_global": ckpt.rng_state(),
                "history": history,
                "input_mae": baseline_input_mae,
                "train_indices": train_indices,
                "eval_indices": eval_indices,
            },
        )

    def run_evaluation(update: int) -> None:
        nonlocal best
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
                "seconds": elapsed(),
            }
        )
        write_history()
        if chi1 == chi1 and chi1 < best:
            best = chi1
            torch.save(
                {"state_dict": rule.state_dict(), "update": update, "chi1": chi1},
                out_dir / "best.pt",
            )
            print(f"  new best pooled chi1 MAE {chi1:.2f} deg -- checkpointed")
        print()

    print(f"\n{args.updates - start_update} updates to go (target {args.updates})")
    recent: list[float] = []
    update = start_update
    for update in range(start_update + 1, args.updates + 1):
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
            done = update - start_update
            print(
                f"  {update:>6}  loss {sum(recent) / len(recent):>7.3f}  "
                f"grad {float(grad_norm):>7.3f}  "
                f"{(time.perf_counter() - started) / done:>5.2f} s/update",
                flush=True,
            )
            recent.clear()

        if update % args.eval_every == 0 or update == args.updates:
            run_evaluation(update)

        # A deliberate stop and an interruption are different events: this one
        # ends the chain (exit 0), while a signal or a time limit asks for a
        # resubmission (exit 64). Conflating them would have ``scancel``
        # followed by a resubmitted job, which is the opposite of cancelling.
        if stop_file.exists():
            write_checkpoint(update)
            write_history()
            print(
                f"\nstopped at update {update} of {args.updates} by {stop_file}; "
                f"state in {last_path}. Delete the file and resubmit to continue.",
                flush=True,
            )
            return 0

        # Against this segment's own clock, not the run's: a budget measured
        # from the start of the *run* would make every resumed segment stop at
        # its first update, which is a chain of jobs making no progress.
        out_of_time = args.max_seconds > 0 and segment_seconds() >= args.max_seconds
        if STOP["asked"] or out_of_time:
            why = STOP["why"] if STOP["asked"] else f"--max-seconds {args.max_seconds:g}"
            write_checkpoint(update)
            write_history()
            print(
                f"\nstopped at update {update} of {args.updates} ({why}); "
                f"state in {last_path}. Resume with the same --tag.",
                flush=True,
            )
            return ckpt.UNFINISHED

        if update % args.checkpoint_every == 0:
            write_checkpoint(update)

    write_checkpoint(update)
    write_history()
    print(f"done in {elapsed() / 60:.1f} min; {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
