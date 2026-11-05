"""Train growth: from a seed, can the rule rebuild the backbone and sequence?

    python scripts/train_growth.py --tag g1 --updates 4000
    python scripts/train_growth.py --tag g1_id --identity-weight 93.7 --updates 8000

``aanca.growth`` has had a complete objective with measured weights since the
growth stack landed, and nothing trained it: ``scripts/train.py`` has no growth
path, so every number this project holds about growth comes from probes on an
**untrained** rule. This is the script that closes that, and it is deliberately
the cheapest thing that can: one structure, one rollout, one backward pass per
update, no sample pool.

**What the two headline numbers are, and why neither alone is a figure of
merit.** ``growth_objective`` reports backbone RMSD over the residues that are
alive and the fraction of slots that are. Under FAPE alone the cheapest
structure is the seed by itself forever -- FAPE is a mean over what exists, so
growing can only add error -- which means a run can drive RMSD towards zero by
refusing to grow. An RMSD quoted without its alive fraction is therefore not a
result, and the table below always prints the pair. ``completion_weight`` is
what forbids the degenerate optimum, and at 27.2 it is the largest weight in
the objective for exactly that reason.

**Checkpoint selection is on the held-out objective total, not on RMSD**, for
the same reason. The total is the only scalar here that prices growing and
placing against each other, and inventing a composite of RMSD and alive
fraction would be choosing that price by hand after the balance probe already
measured it.

**The eval runs the sampled activation, matching training.** A deterministic
``threshold`` probe is available through ``--eval-activation`` and is worth
reading separately: the activation head initialises at a bias of -2, so a
threshold decision wakes *nothing* until training has pushed the logits
positive, and the step at which threshold growth becomes possible at all is a
cleaner statement about what the head has learned than any loss value. It is
not the default because a run that reports zero growth for its first few
thousand updates reports nothing useful about its gradient.

Crop size matters more here than in packing, and not for memory. ``max_births``
caps a step at 8 births, so a 40-step rollout can wake at most 320 slots and
in practice far fewer -- at the initial bias the untrained wake probability is
0.12 per frontier residue per step. A 384-residue crop cannot be completed
inside the rollout at any setting, so the default crop is small enough that
completion is reachable and the alive fraction is a quantity the rule can
actually move.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.growth import (
    BOND_WEIGHT,
    COMPLETION_WEIGHT,
    REACH_WEIGHT,
    GrowthConfig,
    count_runs,
    growth_links,
    growth_objective,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
RUNS = ROOT / "runs"

#: The same set scripts/train.py pins, and for the same reason: these are the
#: settings that decide which structures exist and how they are split, so a
#: warm start that changed one would restore a split describing another pool.
#: ``steps`` is deliberately absent -- changing it is the whole point.
WARM_START_MUST_MATCH = (
    "split", "structures", "eval_structures", "seed", "crop_size", "k",
    # Architecture, not data. ``morphogen`` widens the node dimension and
    # ``birth`` decides whether a birth head exists at all, so a mismatch is a
    # shape error at the first forward pass -- and "mat1 and mat2 shapes cannot
    # be multiplied" is the wrong thing to tell someone who warm-started from
    # the wrong run.
    "birth", "morphogen",
)


def on_disk(specs) -> list:
    """Specs whose coordinates are present, with the path pointed at the file.

    The same helper as ``scripts/train.py``, and it has to redirect the path
    rather than just filter: the manifest records a ``.cif`` path while what is
    downloaded is ``.bcif``.
    """
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_config(args, activation: str) -> RolloutConfig:
    return RolloutConfig(
        steps=args.steps,
        truncate=args.truncate,
        k=args.k,
        update_schedule=args.update_schedule,
        coupling_radius=args.coupling_radius,
        blind=bool(args.identity_weight),
        identity_start=args.identity_start,
        identity_sampling=args.identity_sampling,
        growth=GrowthConfig(
            seeds=args.seeds,
            cward_only=args.cward_only,
            birth=args.birth,
            activation=activation,
            max_births=args.max_births,
            max_rotation=args.max_rotation,
            max_translation=args.max_translation,
            placement_steps=args.placement_steps,
            placement_rotation=args.placement_rotation,
            placement_translation=args.placement_translation,
        ),
    )


def score(rule, dataset, indices, args, config, seed: int = 0) -> dict[str, float]:
    """The held-out objective and its parts, averaged over structures.

    Every term is reported *before* its weight, so the numbers are comparable
    against physical values -- a crystal structure scores ~0.002 on the bond
    term and exactly 0 on reach -- and against the balance probe's table.
    """
    rule.eval()
    keys = (
        "total", "fape", "bond", "completion", "reach", "twist",
        "identity", "rmsd", "alive", "recovery", "closures",
        "frontier", "wake", "segments", "unreachable",
    )
    sums = {key: 0.0 for key in keys}
    n = 0
    rng = random.Random(seed)
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 12:
                continue
            generator = torch.Generator().manual_seed(seed + index)
            terms, _result = growth_objective(
                rule, native, topo, config,
                generator=generator,
                bond_weight=args.bond_weight,
                completion_weight=args.completion_weight,
                identity_weight=args.identity_weight,
                identity_start=args.identity_start,
                reach_weight=args.reach_weight,
                orientation_weight=args.orientation_weight,
            )
            sums["total"] += float(terms.total)
            sums["fape"] += float(terms.fape)
            sums["bond"] += float(terms.bond)
            sums["completion"] += float(terms.completion)
            sums["reach"] += float(terms.reach)
            # Degrees, not the loss. The loss is ``1 - cos theta`` and a
            # reader cannot tell 0.034 from 0.009; fifteen degrees against
            # eight is the comparison the probes set up.
            sums["twist"] += float(terms.orientation_degrees)
            sums["identity"] += float(terms.identity)
            sums["rmsd"] += float(terms.rmsd)
            sums["alive"] += float(terms.alive_fraction)
            sums["closures"] += float(terms.closures)
            sums["frontier"] += float(terms.frontier)
            sums["segments"] += float(terms.segments)
            sums["unreachable"] += float(terms.unreachable)
            if not math.isnan(terms.wake_probability):
                sums["wake"] += float(terms.wake_probability)
            if not math.isnan(terms.recovery):
                sums["recovery"] += float(terms.recovery)
            n += 1
    rule.train()
    out = {key: value / max(n, 1) for key, value in sums.items()}
    out["n"] = float(n)
    return out


def report(scores: dict[str, float], identity: bool) -> str:
    lines = [
        f"  {'total':>8}{'fape':>8}{'rmsd':>9}{'alive':>8}"
        f"{'bond':>8}{'twist':>8}{'reach':>8}{'closures':>10}{'done':>7}"
    ]
    lines.append(
        f"  {scores['total']:>8.3f}{scores['fape']:>8.3f}"
        f"{scores['rmsd']:>7.2f} A{scores['alive']:>7.1%}"
        f"{scores['bond']:>8.3f}{scores['twist']:>7.1f} deg"
        f"{scores['reach']:>8.3f}"
        f"{scores['closures']:>10.1f}{scores['completion']:>7.3f}"
    )
    # The stall diagnostics. Read `done` beside `frontier`: a small completion
    # loss means "the rule wants to grow" only if the frontier is non-empty,
    # and with an empty frontier it means the term saw nothing at all.
    # `unreachable` is the one that says whether a stall is permanent.
    lines.append(
        f"\n  {'frontier':>9}{'wake p':>9}{'segments':>10}{'unreachable':>13}"
    )
    lines.append(
        f"  {scores['frontier']:>9.1f}{scores['wake']:>9.3f}"
        f"{scores['segments']:>10.1f}{scores['unreachable']:>13.1f}"
    )
    if identity:
        lines.append(
            f"\n  type NLL {scores['identity']:>6.3f} "
            f"(floor ln 20 = {math.log(20):.3f}), "
            f"recovery {scores['recovery']:>5.1%}"
        )
    lines.append(
        f"\n  Read rmsd and alive together: FAPE is a mean over what exists, so\n"
        f"  a low rmsd at a low alive fraction is the rule declining to grow."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--structures", type=int, default=400)
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
        help="a compromise between two opposing pressures, measured rather "
        "than guessed. max_births caps a step at 8, so a long crop cannot be "
        "completed inside the rollout and its alive fraction stops being a "
        "quantity the rule can move; but the crop's supervised interior -- "
        "which is what the identity loss scores -- is only 12.8 %% of a "
        "48-residue crop and 19.9 %% of a 64, against 54.8 %% at 128 and "
        "97.2 %% at 384. 128 keeps completion reachable (8 births a step over "
        "40 steps covers 107 residues) while leaving the sequence loss a real "
        "set to score. Geometry scores every alive residue regardless; see "
        "growth_objective on why those two sets differ.",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--truncate", type=int, default=12)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--update-schedule", default="bernoulli")
    parser.add_argument("--coupling-radius", type=float, default=5.0)

    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--cward-only", action="store_true")
    parser.add_argument(
        "--birth", choices=("torsion", "cartesian", "free"), default="cartesian"
    )
    # Zero by default so every mode but ``free`` is unchanged; the free mode
    # sets it below unless the caller asked for something else, because free
    # placement with no placement window is the Cartesian birth with its head
    # frozen and would measure nothing.
    parser.add_argument("--placement-steps", type=int, default=0)
    parser.add_argument("--placement-rotation", type=float, default=3.14159265)
    parser.add_argument("--placement-translation", type=float, default=4.0)
    parser.add_argument(
        "--morphogen",
        action="store_true",
        help=(
            "carry the chain-transported field from aanca.morphogen and feed "
            "it to perception: a route between two residues that does not "
            "depend on them being within the perception cutoff"
        ),
    )
    parser.add_argument("--max-births", type=int, default=8)
    parser.add_argument("--max-rotation", type=float, default=0.1)
    parser.add_argument("--max-translation", type=float, default=0.3)
    parser.add_argument("--activation-init", type=float, default=-2.0)
    parser.add_argument(
        "--activation",
        choices=("sample", "threshold"),
        default="sample",
        help="how training draws the wake decision. 'sample' is the only one "
        "that carries a gradient to the activation head: the straight-through "
        "Bernoulli's derivative is (placed - parked).dL/dx, which is the whole "
        "credit signal for growth order. 'threshold' is for a probe.",
    )

    parser.add_argument("--bond-weight", type=float, default=BOND_WEIGHT)
    parser.add_argument("--completion-weight", type=float, default=COMPLETION_WEIGHT)
    parser.add_argument("--reach-weight", type=float, default=REACH_WEIGHT)
    parser.add_argument(
        "--orientation-weight", type=float, default=0.0,
        help=(
            "weight on relative_frame_restraint: how wrongly a peptide bond is "
            "rotated, against the crystal. Zero reproduces every run before "
            "2026-11-05 exactly, and the quantity is reported either way -- "
            "the probes put it at 15-17 degrees on g4, which alone accounts "
            "for the measured error at every length scale"
        ),
    )
    parser.add_argument(
        "--identity-weight",
        type=float,
        default=0.0,
        help="grow the sequence as well as the backbone. The balance probe "
        "measured 93.7 as the weight that matches FAPE's gradient. Turning "
        "this on implies blind perception, so the state carries no side chains "
        "and a newborn is not secretly its native type.",
    )
    parser.add_argument(
        "--identity-keep",
        type=float,
        default=2.0,
        help="bonus added to the currently-committed label's logit. **Not the "
        "model default of 8.0, deliberately.** Measured at initialisation on "
        "four validation crops, the growth type NLL reads 7.017 at keep 8.0, "
        "4.234 at 4.0, 3.753 at 2.0 and 3.659 at 0.0 against a floor of ln 20 "
        "= 3.00 -- so a keep of 8 costs 3.36 nats of pure self-reinforcement. "
        "It is not merely a bad starting point: the head's logits have a "
        "standard deviation near 1, so a bonus of 8 can never be overcome and "
        "the sequence freezes on whichever label was first sampled at random. "
        "That matters more here than in packing because a newborn arrives "
        "unlabelled and must decide. 2.0 is what v8c_if used to reach 36.8 %% "
        "recovery.",
    )
    parser.add_argument("--identity-start", default="masked")
    parser.add_argument("--identity-sampling", default="sample")
    parser.add_argument(
        "--eval-activation",
        choices=("sample", "threshold"),
        default="sample",
        help="'sample' matches training. 'threshold' takes logit > 0, which "
        "wakes nothing at the initial bias of -2 -- so it reports whether the "
        "head has learned to grow under a deterministic decision, which is a "
        "sharper claim and a useless one early.",
    )

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="growth")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-from",
        default="",
        help="start a NEW run from this checkpoint's weights and optimiser at "
        "update 0, for changing the rollout length without discarding the "
        "training already done. g2 needs it because g1's identity head reached "
        "22.7 %% recovery and its activation head reached a wake probability of "
        "1.000, and both are worth keeping while only --steps changes. The "
        "settings that define the data must match, for the same reason as in "
        "scripts/train.py: the eval split is restored with the weights, so a "
        "different pool would renumber it and score structures the source run "
        "trained on.",
    )
    args = parser.parse_args()

    if args.birth == "free" and args.placement_steps == 0:
        # Free placement with no placement window is the Cartesian birth with
        # its head held at zero: the newborn would sit at the reference pose
        # and be allowed 0.1 rad and 0.3 A per step to leave it, which over a
        # rollout cannot undo an inherited orientation (the coalescence probe's
        # r = 0.970 between junction gaps and accumulated drift). Defaulted
        # rather than required, because a run with the window at zero would
        # produce a plausible-looking null that measures nothing.
        args.placement_steps = 8
        print(
            "note: --birth free with no --placement-steps; defaulting to 8. "
            "Placement *is* relaxation under this mode, so a newborn needs a "
            "window in which it may travel further than a settled residue."
        )

    if args.identity_weight and args.birth == "torsion":
        # Not fatal, but worth saying: a torsion birth reads the parent's type
        # to pick its geometry, and under a blind rollout the newborn has no
        # type yet. Cartesian birth is type-independent, which is why it is
        # the default here.
        print(
            "note: --birth torsion with an identity weight places a newborn "
            "from torsions while its type is still unknown; cartesian birth "
            "is type-independent and is the default for this reason."
        )

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    splits = read_manifest(MANIFEST)
    if args.split not in splits:
        print(f"no split {args.split!r}; available: {sorted(splits)}")
        return 1
    available = on_disk(splits[args.split])
    if not available:
        print(f"no coordinates in {PDB} -- run scripts/fetch_structures.py")
        return 1

    wanted = args.structures + args.eval_structures
    pool = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        pool,
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=max(len(pool), 1),
    )
    indices = list(range(len(pool)))
    random.Random(args.seed).shuffle(indices)
    eval_indices = indices[: args.eval_structures]
    train_indices = indices[args.eval_structures :]
    if not train_indices:
        print("no training entries left after holding some out")
        return 1

    rule = AANCARule(
        growth=True,
        birth=args.birth,
        morphogen=args.morphogen,
        activation_init=args.activation_init,
        max_rotation=args.max_rotation,
        max_translation=args.max_translation,
        identity=bool(args.identity_weight),
        identity_keep=args.identity_keep,
    ).to(device)
    optimiser = torch.optim.Adam(rule.parameters(), lr=args.lr)

    config = build_config(args, args.activation)
    eval_config = build_config(args, args.eval_activation)

    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "STOP"

    start_update = 0
    best = float("inf")
    if args.init_from and not (args.resume and last_path.exists()):
        warm = ckpt.load(Path(args.init_from), map_location=device)
        stored = warm.get("args") or {}
        changed = {
            key: (stored.get(key), vars(args).get(key))
            for key in WARM_START_MUST_MATCH
            if stored.get(key) != vars(args).get(key)
        }
        if changed:
            detail = ", ".join(
                f"{key}: checkpoint {was!r}, now {now!r}"
                for key, (was, now) in sorted(changed.items())
            )
            print(
                "--init-from needs a checkpoint over the same structures, "
                "because its eval split is restored with the weights "
                f"({detail}). Drop --init-from to start from scratch."
            )
            return 1
        rule.load_state_dict(ckpt.rule_weights(warm))
        if "optimiser" in warm:
            optimiser.load_state_dict(warm["optimiser"])
        # The source run's split, so the held-out set stays held out. Without
        # this the new run would reshuffle and score structures g1 trained on,
        # which is the same fault scripts/score_identity.py was written to fix.
        if warm.get("eval_indices"):
            eval_indices = list(warm["eval_indices"])
            train_indices = [i for i in indices if i not in set(eval_indices)]
        print(
            f"warm-started from {args.init_from} at its update "
            f"{warm.get('update', '?')}, continuing at update 0"
        )
    if args.resume and last_path.exists():
        state = ckpt.load(last_path, map_location=device)
        rule.load_state_dict(state["rule"])
        optimiser.load_state_dict(state["optimiser"])
        start_update = int(state["update"])
        best = float(state.get("best", float("inf")))
        ckpt.set_rng_state(state["rng"])
        ckpt.restore_python_rng(rng, state["python_rng"])
        print(f"resumed from {last_path} at update {start_update}")

    print(
        f"device {device}, split {args.split}: {len(train_indices)} training, "
        f"{len(eval_indices)} held out (disjoint)"
    )
    print(
        f"{rule.parameter_count()} parameters, {args.seeds} seed(s), "
        f"{args.steps} steps, {args.birth} birth, crop {args.crop_size}, "
        f"activation bias {args.activation_init}"
    )
    print(
        f"weights: bond {args.bond_weight:g}, completion "
        f"{args.completion_weight:g}, reach {args.reach_weight:g}, identity "
        f"{args.identity_weight:g}"
    )
    # **Say whether the seed count can reach the whole crop.** A crop is a
    # spatial ball, so it contains several covalent runs, and growth cannot
    # cross a chain break -- g2 stalled at 55.5 % alive with an empty frontier
    # for exactly this reason. Measuring it here means a seed-starved run
    # announces itself rather than being diagnosed afterwards from a flat
    # alive fraction.
    sampled = []
    probe_rng = random.Random(args.seed)
    for index in train_indices[: min(8, len(train_indices))]:
        try:
            probe_topo, _n, _c, _u = dataset.load_crop(index, rng=probe_rng)
        except Exception:
            continue
        sampled.append(count_runs(growth_links(probe_topo)))
    if sampled:
        mean_runs = sum(sampled) / len(sampled)
        print(
            f"covalent runs per crop: mean {mean_runs:.1f} "
            f"(range {min(sampled)}-{max(sampled)}) over {len(sampled)} crops; "
            f"{args.seeds} seeds"
        )
        if args.seeds < mean_runs:
            print(
                f"  WARNING: fewer seeds than runs, so about "
                f"{mean_runs - args.seeds:.1f} runs per crop get no seed and "
                "can never be reached.\n  The alive fraction is capped below 1 "
                "by topology and no step budget will lift it."
            )
    print(
        "The alive fraction is the number to watch first. Completion at 27.2 "
        "is\nthe weight that forbids the degenerate optimum of a seed alone "
        "forever,\nand whether it is enough is the first thing this run "
        "settles.\n"
    )

    started = time.perf_counter()
    #: Updates dropped for a non-finite gradient. Reported at the end, because
    #: a run that silently skipped a third of its updates is not the run its
    #: own log claims to be.
    skipped_updates = 0
    recent: dict[str, list[float]] = {
        key: []
        for key in ("total", "fape", "rmsd", "alive", "recovery", "twist")
    }

    for update in range(start_update + 1, args.updates + 1):
        index = rng.choice(train_indices)
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 12:
            continue
        generator = torch.Generator().manual_seed(args.seed * 1_000_003 + update)
        terms, _result = growth_objective(
            rule, native, topo, config,
            generator=generator,
            bond_weight=args.bond_weight,
            completion_weight=args.completion_weight,
            identity_weight=args.identity_weight,
            identity_start=args.identity_start,
            reach_weight=args.reach_weight,
            orientation_weight=args.orientation_weight,
        )
        if not torch.isfinite(terms.total):
            continue
        optimiser.zero_grad()
        terms.total.backward()
        # **A finite loss does not imply a finite gradient, and clipping turns
        # one infinity into NaN parameters.** This is what killed g2 at update
        # ~1375 and g3 within its first sixty, and the guard above cannot see
        # it: ``terms.total`` was finite in both cases.
        #
        # Measured directly. Give one entry of one gradient an infinity, then
        # clip to 1.0: ``clip_grad_norm_`` computes ``total_norm = inf``, so the
        # scale factor is ``clip / inf = 0``, and ``0 * inf = NaN``. The
        # parameters holding that gradient go NaN on ``step()``, and if they sit
        # in the shared trunk every logit downstream is NaN -- which is exactly
        # the "6912 of 6912 identity logits non-finite" the rollout guard
        # reported, rather than the handful a local fault would give.
        #
        # So the gradient is checked before it is allowed to touch the weights.
        # Skipping the update is right rather than clamping: an infinite
        # gradient means this rollout produced something the objective cannot
        # price, and a clamped version of it is a fabricated direction.
        finite_grad = all(
            torch.isfinite(p.grad).all()
            for p in rule.parameters()
            if p.grad is not None
        )
        if not finite_grad:
            skipped_updates += 1
            optimiser.zero_grad(set_to_none=True)
            if skipped_updates in (1, 10, 100) or skipped_updates % 500 == 0:
                print(
                    f"  update {update}: non-finite gradient with a finite "
                    f"loss ({float(terms.total.detach()):.3f}); skipped. "
                    f"{skipped_updates} so far.",
                    flush=True,
                )
            continue
        torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)
        optimiser.step()

        recent["total"].append(float(terms.total.detach()))
        recent["fape"].append(float(terms.fape))
        recent["rmsd"].append(float(terms.rmsd))
        recent["alive"].append(float(terms.alive_fraction))
        recent["twist"].append(float(terms.orientation_degrees))
        if not math.isnan(terms.recovery):
            recent["recovery"].append(float(terms.recovery))

        if update % max(args.eval_every // 4, 1) == 0:
            done = update - start_update

            def mean(key: str) -> float:
                values = recent[key]
                return sum(values) / max(len(values), 1)

            line = (
                f"  {update:>6}  total {mean('total'):>8.3f}  "
                f"fape {mean('fape'):>6.3f}  rmsd {mean('rmsd'):>6.2f} A  "
                f"alive {mean('alive'):>5.1%}"
            )
            line += f"  twist {mean('twist'):>4.1f}d"
            if args.identity_weight:
                line += f"  rec {mean('recovery'):>5.1%}"
            line += f"  {(time.perf_counter() - started) / max(done, 1):>5.2f} s/u"
            print(line, flush=True)
            for values in recent.values():
                values.clear()

        if update % args.eval_every == 0 or update == args.updates:
            scores = score(rule, dataset, eval_indices, args, eval_config)
            print(f"\nheld out after {update} updates "
                  f"({args.eval_activation} activation):")
            print(report(scores, bool(args.identity_weight)) + "\n", flush=True)
            if scores["total"] < best:
                best = scores["total"]
                ckpt.save(out_dir / "best.pt", {"state_dict": rule.state_dict()})
                print("  new best, checkpointed\n", flush=True)
            ckpt.save(
                last_path,
                {
                    "rule": rule.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "update": update,
                    "best": best,
                    "rng": ckpt.rng_state(),
                    "python_rng": rng.getstate(),
                    "args": vars(args),
                    "eval_indices": eval_indices,
                },
            )

        if stop_file.exists():
            print(f"stopped at update {update} by {stop_file}")
            return ckpt.UNFINISHED
        if args.max_seconds and time.perf_counter() - started > args.max_seconds:
            print(f"stopped at update {update} after {args.max_seconds:g} s")
            return ckpt.UNFINISHED

    minutes = (time.perf_counter() - started) / 60
    print(f"done in {minutes:.1f} min; {out_dir}")
    if skipped_updates:
        done = max(args.updates - start_update, 1)
        print(
            f"{skipped_updates} of {done} updates were skipped for a "
            f"non-finite gradient ({skipped_updates / done:.1%}). A large "
            "share means the run trained on far fewer updates than its tag "
            "says and the objective is producing gradients it cannot price."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
