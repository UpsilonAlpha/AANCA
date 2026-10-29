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
from dataclasses import replace
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.evaluate import build_eval_set, evaluate
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, SamplePool, training_objective
from aanca.schedule import learning_rate

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


#: What a warm start may not differ on. Shorter than ``checkpoint.PINNED``
#: on purpose: a warm start exists to change the rollout length or the
#: learning rate, so those are exactly what it must be free to change. These
#: are the settings that decide *which problem* the restored pool and the
#: restored split describe. Get one of them wrong and the pool hands the new
#: run states belonging to structures it is not training on -- the same class
#: of silent mismatch that the crop/pool bug was.
WARM_START_MUST_MATCH = (
    "split", "structures", "eval_structures", "seed", "crop_size", "k",
)


def check_warm_start(source: dict, args) -> None:
    """Refuse a warm start whose pool and split describe another problem."""
    stored = source.get("args") or {}
    current = vars(args)
    changed = {
        key: (stored.get(key), current.get(key))
        for key in WARM_START_MUST_MATCH
        if stored.get(key) != current.get(key)
    }
    if changed:
        detail = ", ".join(
            f"{key}: checkpoint {was!r}, now {now!r}"
            for key, (was, now) in sorted(changed.items())
        )
        raise ValueError(
            "--init-from needs a checkpoint over the same structures, because "
            "its sample pool and its train/eval split are restored with the "
            f"weights ({detail}). Drop --init-from to start from scratch."
        )


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
    parser.add_argument(
        "--cache",
        type=int,
        default=0,
        help="parsed entries to keep (0: enough for the whole training pool)",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--steps-max",
        type=int,
        default=0,
        help="if set, each update draws its rollout length uniformly from "
        "[--steps, --steps-max]. A fixed horizon lets the rule learn where "
        "the horizon is; the 3D NCA prior art samples 64-96 per iteration. "
        "Truncated backpropagation keeps the memory cost flat, so a longer "
        "rollout costs time and not memory.",
    )
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--truncate", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-drop",
        type=int,
        nargs="*",
        default=[],
        help="update numbers at which the learning rate is multiplied by "
        "--lr-gamma. Absolute, not fractions of --updates, so extending a run "
        "cannot move the schedule it already trained under.",
    )
    parser.add_argument("--lr-gamma", type=float, default=0.3)
    parser.add_argument(
        "--init-from",
        default="",
        help="start a NEW run from the weights, optimiser and sample pool of "
        "this checkpoint, at update 0. For changing a setting the fingerprint "
        "pins -- rollout length, learning rate -- without discarding the "
        "training already done. The settings that define the data (split, "
        "structures, seed, crop size, k) must match, or the restored pool and "
        "split would describe a different problem.",
    )
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument(
        "--max-skipped",
        type=int,
        default=200,
        help="give up after this many non-finite updates (0 to never give up)",
    )
    parser.add_argument("--pool-probability", type=float, default=0.5)
    parser.add_argument(
        "--native-probability",
        type=float,
        default=0.0,
        help="fraction of fresh samples started from the native structure "
        "instead of a damaged one. This is how the fixed point is supervised: "
        "the end state goes into the sample pool like any other, so near-native "
        "states become future seeds and 'stay put' is trained over the whole "
        "rollout rather than asserted at one step. Replaces --noop-weight; see "
        "scripts/noop_probe.py for why that term could not do the job.",
    )
    parser.add_argument(
        "--noop-weight",
        type=float,
        default=1.0,
        help="weight on the no-op term, which asks for zero output at the "
        "native structure. **Measured vacuous at its current placement**: it "
        "is evaluated at step 0, where the hidden state is empty and the rule "
        "proposes ~0.04 deg whatever it is shown, so it cannot tell a native "
        "structure from a wrecked one. Kept at its old default so a v4 run "
        "still resumes; set it to 0 and use --native-probability instead.",
    )
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

    # Hold the whole training pool in the parse cache. The dataset's default
    # bound is 256, which is right for a 23,761-entry split and wrong here:
    # the run draws from ``--structures`` entries at random, so a cache
    # smaller than that pool thrashes and the expensive part -- parsing -- is
    # paid again on most updates. It is transparent to the results either way,
    # which is why it is not pinned in the fingerprint.
    cache = args.cache if args.cache > 0 else args.structures + args.eval_structures
    dataset = RepackingDataset(
        available,
        concentrations=(100.0, 20.0, 5.0, 0.0),
        crop_size=args.crop_size,
        device=device,
        cache_limit=max(cache, 1),
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
    resumed_skipped: dict = {}

    state = None
    if args.resume == "auto" and last_path.exists():
        state = ckpt.load(last_path, map_location=device)
        if not args.force_resume:
            # The defaults go in so that pinning a *new* setting does not
            # strand a run already chaining through its segments: see
            # ``check_fingerprint``.
            ckpt.check_fingerprint(
                state["fingerprint"],
                ckpt.fingerprint(args),
                ckpt.fingerprint(parser.parse_args([])),
            )

    #: A warm start is not a resume, and the difference is the point. A resume
    #: continues one curve and must match on everything pinned; a warm start
    #: deliberately changes something pinned -- the rollout length, the
    #: learning rate -- and so begins a new curve at update 0, keeping only
    #: what is still true under the new settings: the weights, the optimiser
    #: moments, and the sample pool. The pool is the valuable part. It is the
    #: long-horizon memory the prior art says persistence comes from, and
    #: throwing it away would restart the part of the objective that takes
    #: longest to mean anything.
    warm = None
    if state is None and args.init_from:
        warm = ckpt.load(Path(args.init_from), map_location=device)
        check_warm_start(warm, args)

    #: Carried forward on resume, so a chained run's later segments still say
    #: where the weights originally came from.
    init_record = (
        {"path": str(args.init_from), "update": int(warm["update"])}
        if warm is not None
        else (state or {}).get("init_from")
    )

    source = state if state is not None else warm
    if source is not None:
        train_indices = list(source["train_indices"])
        eval_indices = list(source["eval_indices"])
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

    if warm is not None:
        # Weights, moments and pool, and nothing else: the RNGs, the update
        # count, the history and the best score all belong to the run that
        # wrote this checkpoint, not to the one starting here.
        rule.load_state_dict(warm["rule"])
        optimiser.load_state_dict(warm["optimiser"])
        pool.load_state_dict(warm["pool"])
        for group in optimiser.param_groups:
            group["lr"] = args.lr
        print(
            f"warm start: {len(pool)} pooled states and the weights from "
            f"{args.init_from} at its update {warm['update']}, continuing at "
            "update 0 under this run's settings"
        )

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
        # ``.get`` because checkpoints written before the counters existed are
        # still resumable; a missing tally is zero, not an error.
        resumed_skipped = dict(state.get("skipped") or {})
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
        # Only an untrained model is untrained. After a warm start this row is
        # the *source* run's result measured under this run's eval settings,
        # which is the number every later row should be read against -- and
        # calling it "untrained" would make the whole curve look like it began
        # from nothing.
        if warm is not None:
            print(f"\nat update 0, warm-started from {args.init_from}:")
        else:
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
                {
                    "args": vars(args),
                    "init_from": init_record,
                    "input_mae": baseline_input_mae,
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    #: Non-finite events, counted rather than fatal. ``update`` is an update
    #: whose loss or gradient was not finite and was therefore not applied;
    #: ``state`` is a rollout whose result the pool refused. They are
    #: different failures and worth separating: the first leaves the weights
    #: untouched, the second means the forward pass produced NaN from weights
    #: that were finite going in. Carried in the checkpoint so the tally is a
    #: property of the run rather than of the segment.
    skipped = {"update": 0, "state": 0, **resumed_skipped}

    #: How many fresh samples were started from the native structure rather
    #: than a damaged one. Cumulative across segments, like ``skipped``,
    #: because the share that matters is the run's and not this segment's --
    #: and because a pool whose seeds came from native starts is a different
    #: pool, so the count belongs with the weights.
    native_starts = int((source or {}).get("native_starts", 0))

    def write_checkpoint(update: int) -> None:
        ckpt.save(
            last_path,
            {
                "update": update,
                "skipped": dict(skipped),
                "native_starts": native_starts,
                "elapsed": elapsed(),
                "best": best,
                "fingerprint": ckpt.fingerprint(args),
                "args": vars(args),
                # Where the weights came from, if not from scratch. Without
                # this a warm-started run looks like one that learned
                # unusually fast in its first thousand updates.
                "init_from": init_record,
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
                "native_drift_mean": result.native_drift_mean,
                "native_starts": native_starts,
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
        # Set from the update number rather than stepped, so a resume lands on
        # the right rate without a scheduler to restore. Costs one float per
        # update; buys one fewer thing that can be wrong at a segment boundary.
        lr_now = learning_rate(args.lr, update, args.lr_drop, args.lr_gamma)
        for group in optimiser.param_groups:
            group["lr"] = lr_now

        index = rng.choice(train_indices)

        # Drawn from ``rng`` and drawn here, before anything else consumes it,
        # so the sequence is a property of the run and survives a resume.
        if args.steps_max > args.steps:
            step_config = replace(
                train_config, steps=rng.randint(args.steps, args.steps_max)
            )
        else:
            step_config = train_config

        # Growing-NCA re-seeding: sometimes start from a state the rule
        # produced itself rather than from fresh damage, which is what asks it
        # to *stay* converged rather than merely arrive once.
        #
        # **The pool is consulted before the sample is drawn**, because a
        # pooled state belongs to one particular crop and the sample has to be
        # rebuilt on that same crop. Drawing the sample first and then
        # overriding its coordinates pairs one set of residues with another's
        # topology and native target; the shapes match, so nothing complains.
        pooled = pool.sample(index) if rng.random() < args.pool_probability else None
        crop_seed = pooled[0] if pooled is not None else None
        sample = dataset.sample(
            index, generator=generator, rng=rng, crop_seed=crop_seed
        )

        # The finiteness test is belt to the pool's own braces: the pool
        # refuses to store a non-finite state, but a checkpoint written
        # before it did so can still carry one, and a resume must not
        # reinstate a poisoned seed. The shape test should now be redundant --
        # the same crop seed gives the same number of residues -- and is kept
        # as an assertion that it is.
        reusable = (
            pooled is not None
            and pooled[1].coords.shape == sample.damaged.shape
            and bool(torch.isfinite(pooled[1].coords).all())
        )
        # A fresh sample is sometimes the *native* structure rather than a
        # damaged one. This is the fixed point as a training case instead of
        # an assertion: the reconstruction loss at a random late step then
        # asks the rule to still be at native after 64-96 steps, with a hidden
        # state that has had time to say something -- which is exactly what
        # the no-op term at step 0 cannot ask (scripts/noop_probe.py). The end
        # state is pooled like any other, so near-native states become seeds
        # and the horizon compounds across updates.
        #
        # The draw is skipped entirely at probability 0 so that a run started
        # before this existed consumes the same random stream and resumes
        # bit-identically; within a run the probability is pinned, so the
        # stream is consistent across segments either way.
        from_native = (
            args.native_probability > 0.0
            and rng.random() < args.native_probability
        )
        if reusable:
            start = pooled[1].coords
        elif from_native:
            start = sample.native
            native_starts += 1
        else:
            start = sample.damaged

        terms, result = training_objective(
            rule,
            start,
            sample.native,
            sample.topology,
            step_config,
            generator=generator,
            rng=rng,
            noop_weight=args.noop_weight,
        )
        optimiser.zero_grad()
        terms.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)

        # **A non-finite update is skipped, not clipped.** Clipping does not
        # contain this: ``clip_grad_norm_`` divides by a total norm that is
        # itself NaN, so every gradient becomes NaN and one bad update writes
        # NaN into every weight permanently. After that the rule emits NaN
        # coordinates for ever, the pool fills with them, and the run dies
        # much later at a neighbour search with a traceback pointing at
        # scipy. The guard is two reductions and it turns an unrecoverable
        # run into a counted, reported event.
        healthy = bool(torch.isfinite(terms.total) and torch.isfinite(grad_norm))
        if healthy:
            optimiser.step()
            if not pool.insert(index, result.state, crop_seed=sample.crop_seed):
                skipped["state"] += 1
        else:
            optimiser.zero_grad(set_to_none=True)
            skipped["update"] += 1
            if skipped["update"] in (1, 10, 100) or skipped["update"] % 1000 == 0:
                print(
                    f"  update {update}: non-finite loss/gradient "
                    f"(loss {float(terms.total):.4g}, grad {float(grad_norm):.4g}); "
                    f"skipped, {skipped['update']} so far",
                    flush=True,
                )
            # Persistent, not transient: something is wrong with the setup or
            # the data, and a run that skips every update is not training.
            if args.max_skipped > 0 and skipped["update"] >= args.max_skipped:
                # Which of the two failures this is, said plainly, because
                # they need different responses. Non-finite weights mean the
                # rule is gone and the checkpoint is only useful as evidence;
                # finite weights mean the forward pass is producing NaN from
                # particular inputs, and the run is recoverable by finding
                # them.
                bad = [
                    name
                    for name, p in rule.named_parameters()
                    if not torch.isfinite(p).all()
                ]
                print(
                    f"\n{skipped['update']} non-finite updates (limit "
                    f"--max-skipped {args.max_skipped}); stopping rather than "
                    "spending a walltime on a run that is not learning."
                )
                print(
                    f"  weights: {len(bad)} non-finite parameters"
                    + (f" ({', '.join(bad[:4])})" if bad else "")
                    + "; "
                    + (
                        "the rule itself is destroyed -- the cause is earlier "
                        "than this message."
                        if bad
                        else "the weights are intact, so the NaN is coming out "
                        "of the forward pass on particular inputs."
                    )
                )
                write_checkpoint(update)
                write_history()
                return 1
        if healthy:
            recent.append(float(terms.total.detach()))

        if update % max(args.eval_every // 4, 1) == 0:
            done = update - start_update
            print(
                f"  {update:>6}  loss "
                f"{sum(recent) / len(recent) if recent else float('nan'):>7.3f}  "
                f"grad {float(grad_norm):>7.3f}  "
                f"{(time.perf_counter() - started) / done:>5.2f} s/update"
                + (
                    f"  [skipped {skipped['update']} updates, "
                    f"{skipped['state']} states]"
                    if skipped["update"] or skipped["state"]
                    else ""
                ),
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
