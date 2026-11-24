"""Train the E(n)-GNCA port: Gala's rule, verbatim, on the regeneration task.

    .venv/Scripts/python.exe scripts/train_egnca.py configs/egnca.toml --section port
    .venv/Scripts/python.exe scripts/train_egnca.py configs/egnca.toml --list-sections

Settings are a named section of a config file, never a flag, exactly as in
``scripts/train_growth.py``: an option may only address the config or name a
resource. ``aanca.config.check_base_is_complete`` holds ``[base.inputs]`` to
naming every key, so a change to a default here cannot reinterpret a recorded
run.

**This is a separate trainer because it trains a separate rule.**
``train_growth.py`` builds an ``AANCARule`` and calls ``growth_objective``,
whose namespace is twenty-odd weights over frames, births, identity, sterics and
closures -- none of which this rule has. Threading a frameless rule through that
schema would mean a dozen keys that are silently inert, which is the exact class
of fault this project has lost results to. Three settings carry the whole
objective here, and the banner prints all three.

**What the port changes relative to every arm in ``configs/regeneration.toml``,
and it changes several at once on purpose** (a replication target, not an
ablation -- ``docs/claims/the-task-is-to-realise-coordinates-for-a-given-contact-graph.md``):

  * the rule: frameless, ~5.3 kparam, against ``AANCARule``'s 229,588;
  * the rollout: ~15-25 steps, differentiated **end to end**, against 120 steps
    truncated at 12;
  * no step cap, where the project's actuator bounded a move at 0.3 A;
  * the native contact graph as the task statement;
  * the all-pairs distance restraint over the **full** pair set, both halves
    weighted;
  * a replay pool, as published.

**What it cannot report.** ``fape``, ``twist``, ``bond``, ``clash``,
``identity``, ``alive`` and ``recovery`` are absent from the table rather than
zero: this rule places CA positions and no orientation, so none of them is a
quantity it produces. See ``aanca.egnca``'s module docstring. ``rmsd_dmg`` and
the two halves of the pair restraint are the whole readout, and ``move`` is
there so a flat ``rmsd_dmg`` can be told from a rule that emits nothing.

**WHAT 941 UPDATES ON A LAPTOP SAID, AND IT IS A WARNING ABOUT THE READOUT.**
First run of this trainer at the ``port`` section, CPU, 0.21 s/update::

    update  total  anchor  window  rmsd_dmg  (fit intact, start)   n  diverged
       250  6.075   2.960   3.115     2.501   (2.629,    8.774)   15         0
       500  4.240   1.769   2.471     2.194   (2.302,    7.203)   11         4
       750  3.769   1.676   2.093     2.140   (2.223,    5.402)    7         8

The loss falls and the window improves, **and half the held-out set has gone
non-finite by update 750.** Look at ``start``: it falls from 8.77 to 5.40, and
the start state cannot depend on the weights -- what changed is *which crops
were averaged*, because the hard ones diverged out of the mean. So the
``rmsd_dmg`` column is partly survivorship, and a row with a high ``diverged``
is not a result no matter how good its other columns look.

That is the unbounded rule's own failure mode, reported by its authors, and it
is the first thing to fix before an arm of this is read: a lower ``lr``, a
tighter ``clip``, or a per-crop guard. It is recorded here rather than left to
be rediscovered from a log, and it is why ``diverged`` is printed at all.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca import config as run_config
from aanca.data import RepackingDataset, read_manifest
from aanca.egnca import (
    EGNCAPoolEntry,
    EGNCARule,
    egnca_objective,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
RUNS = ROOT / "runs"

#: Settings a warm start may not cross. The two widths because they change the
#: shapes and ``load_state_dict`` would raise a size mismatch several screens
#: into a run; ``split``, ``structures`` and ``eval_structures`` because the
#: eval split is restored with the weights, so a checkpoint over a different
#: pool would hold back different proteins. The weights are deliberately absent:
#: continuing training under a changed objective is something this project does
#: on purpose and no stored tensor changes shape.
WARM_START_MUST_MATCH = (
    "hidden_dim",
    "message_dim",
    "split",
    "structures",
    "eval_structures",
    "crop_size",
)


def build_schema() -> argparse.ArgumentParser:
    """Every setting the trainer reads, with its type and its default."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument("--split", default="train")
    p.add_argument("--structures", type=int, default=400)
    p.add_argument("--eval-structures", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)

    # The rule. Gala's widths, and the reason they are the defaults is that the
    # published parameter count is quoted against them.
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--message-dim", type=int, default=32)

    # The rollout. ``steps-min`` with ``steps`` is the published "15 to 25
    # steps, sampled uniformly"; equal values fix the length.
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--steps-min", type=int, default=15)
    p.add_argument("--damage", type=int, default=12)
    p.add_argument(
        "--initial", default="helix", choices=("helix", "extended", "random")
    )
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--device-neighbours", action="store_true")
    #: The task statement, not a ceiling. False recomputes the selection from
    #: the live CA every step, which is a strictly harder problem than the
    #: published one.
    p.add_argument("--native-edges", action="store_true", default=True)

    # The objective. Both halves of the unclamped all-pairs CA restraint, and
    # nothing else -- there is no third weight to mis-set.
    p.add_argument("--pairwise-weight", type=float, default=1.0)
    p.add_argument("--pairwise-window-weight", type=float, default=1.0)

    # Training. ``weight-decay`` and ``clip`` are the two things the authors
    # name as what stabilised their training, and they report exploding
    # gradients even so.
    p.add_argument("--updates", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--pool-probability", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=250)

    for dest in run_config.RUNTIME_ONLY:
        p.add_argument(f"--{dest.replace('_', '-')}", default=None)
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--section", default=None)
    p.add_argument("--list-sections", action="store_true")
    p.add_argument("--diff", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--stop-file", default=None)
    p.add_argument("--resume", action="store_true")
    return p


def recorded_args(args: argparse.Namespace) -> dict:
    """What the run *is*, for the checkpoint: settings without the resources."""
    return {
        key: value
        for key, value in vars(args).items()
        if key not in run_config.RUNTIME_ONLY
    }


def build_rule(args) -> EGNCARule:
    """The rule, built from the section. One call, so the keys cannot miss it."""
    return EGNCARule(hidden_dim=args.hidden_dim, message_dim=args.message_dim)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def on_disk(specs) -> list:
    """Specs whose coordinates are present, with the path pointed at the file.

    Lifted verbatim from ``scripts/train_growth.py``, and it has to **redirect**
    the path rather than only filter: the manifest records a ``.cif`` path while
    what is downloaded is ``.bcif``. A filter that only matched ``.pdb`` finds
    nothing in a populated ``data/pdb`` and the run refuses with "no
    coordinates" -- which is what the first version of this function did.
    """
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def objective_kwargs(args) -> dict:
    """Every setting that reaches :func:`aanca.egnca.egnca_objective`.

    One dict, built once and passed to both the training step and the held-out
    evaluation. **That is the point of it**: the project's own record has a
    retraction in it from weights that reached training and not evaluation
    (job 32405172), so a held-out column was structurally absent and printed as
    a measurement. There is no second place here for a key to go missing from.
    """
    return dict(
        damage=args.damage,
        pairwise_weight=args.pairwise_weight,
        pairwise_window_weight=args.pairwise_window_weight,
        k=args.k,
        cutoff=args.cutoff,
        device_neighbours=args.device_neighbours,
        native_edges=args.native_edges,
        initial=args.initial,
    )


KEYS = (
    "total",
    "pair_anchor",
    "pair_window",
    "rmsd_dmg",
    "rmsd_dmg_fit_intact",
    "start_rmsd_dmg",
    "move",
)


def score(rule, dataset, indices, args, seed: int = 0) -> dict[str, float]:
    """The held-out objective and its parts, averaged over crops that had a window.

    A crop whose longest covalent run is too short to hold a 12-residue window
    is undamaged, and folding its absent score in as a zero would read as a
    perfect fill -- so those crops are counted out rather than averaged in.

    **``n`` and ``diverged`` are part of the readout, not diagnostics.** This
    rule is unbounded and its authors report exploding gradients, so a
    non-finite rollout is an ordinary outcome -- and a mean taken over the three
    crops that survived, printed without saying so, would be the best-looking
    column in the log. ``diverged`` counts the crops whose rollout went
    non-finite; a row with a low ``n`` and a high ``diverged`` is not a result.
    """
    rule.eval()
    sums = {key: 0.0 for key in KEYS}
    n = 0
    diverged = 0
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
            terms, _state = egnca_objective(
                rule,
                native,
                topo,
                steps=args.steps,
                generator=generator,
                **objective_kwargs(args),
            )
            if terms.rmsd_damaged is None or not torch.isfinite(terms.total):
                diverged += 1
                continue
            sums["total"] += float(terms.total)
            sums["pair_anchor"] += float(terms.pair_anchor)
            sums["pair_window"] += float(terms.pair_window)
            sums["rmsd_dmg"] += terms.rmsd_damaged
            sums["rmsd_dmg_fit_intact"] += terms.rmsd_damaged_fit_intact
            sums["start_rmsd_dmg"] += terms.start_rmsd_damaged
            sums["move"] += terms.move or 0.0
            n += 1
    rule.train()
    tail = {"n": float(n), "diverged": float(diverged)}
    if not n:
        return {key: float("nan") for key in KEYS} | tail
    return {key: value / n for key, value in sums.items()} | tail


def report(scores: dict[str, float]) -> str:
    return (
        f"total {scores['total']:9.3f}  anchor {scores['pair_anchor']:9.3f}  "
        f"window {scores['pair_window']:9.3f}  rmsd_dmg {scores['rmsd_dmg']:6.3f}  "
        f"(fit intact {scores['rmsd_dmg_fit_intact']:6.3f}, start "
        f"{scores['start_rmsd_dmg']:6.3f})  move {scores['move']:6.3f}  "
        f"n {int(scores['n'])}"
        + (f"  DIVERGED {int(scores['diverged'])}" if scores["diverged"] else "")
    )


class _SkipUpdate(Exception):
    """This rollout produced nothing the objective can price."""


def main() -> int:
    runtime = build_parser().parse_args()
    schema = build_schema()

    data = run_config.load(runtime.config)
    if runtime.list_sections:
        for name in run_config.sections(data):
            print(name)
        return 0
    if not runtime.section:
        print(
            "error: --section is required. This config defines: "
            + (", ".join(run_config.sections(data)) or "(none)")
        )
        return 2
    run_config.check_base_is_complete(data, schema)
    args = run_config.to_namespace(
        schema, run_config.resolve(data, runtime.section), runtime.section, runtime
    )

    if runtime.diff:
        other = run_config.to_namespace(
            schema, run_config.resolve(data, runtime.diff), runtime.diff, runtime
        )
        rows = [
            (key, getattr(other, key), getattr(args, key))
            for key in sorted(vars(args))
            if key != "tag" and getattr(args, key) != getattr(other, key)
        ]
        print(f"{runtime.diff} -> {runtime.section}")
        for key, was, now in rows:
            print(f"  {key}: {was!r} -> {now!r}")
        print(f"  {len(rows)} setting(s) differ")
        return 0

    if args.steps_min > args.steps:
        print(
            f"steps_min {args.steps_min} exceeds steps {args.steps}: the "
            "rollout length is drawn uniformly from [steps_min, steps] and an "
            "empty range cannot be drawn from."
        )
        return 2

    print(f"config     : {runtime.config} [{runtime.section}]")
    print(
        "E(n)-GNCA PORT: Gala's rule verbatim -- frameless, one hop per step, "
        "scalar phi_x, no step cap, FULL BPTT over "
        + (
            f"{args.steps_min}-{args.steps} steps"
            if args.steps_min != args.steps
            else f"{args.steps} steps"
        )
        + ".\n  It places CA positions and produces NO residue orientation, so "
        "fape, twist, bond, clash and identity are not reported -- they are not "
        "quantities this rule produces. Read rmsd_dmg against start_rmsd_dmg."
    )
    print(
        "  "
        + (
            "NATIVE EDGES: message passing runs over the contact graph the "
            "NATIVE coordinates select, held fixed for the rollout. That is "
            "the task statement."
            if args.native_edges
            else "LIVE EDGES: the selection is recomputed from the current CA "
            "every step. This is a HARDER problem than the published one and "
            "is not the port."
        )
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
    catalogue = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        catalogue,
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=max(len(catalogue), 1),
    )
    indices = list(range(len(catalogue)))
    random.Random(args.seed).shuffle(indices)
    eval_indices = indices[: args.eval_structures]
    train_indices = indices[args.eval_structures :]
    if not train_indices:
        print("no training entries left after holding some out")
        return 1

    rule = build_rule(args).to(device)
    optimiser = torch.optim.Adam(
        rule.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_text = run_config.snapshot(
        args, args.tag, runtime.config, trainer="scripts/train_egnca.py"
    )
    snapshot_path = out_dir / "config.toml"
    if (
        snapshot_path.exists()
        and snapshot_path.read_text(encoding="utf-8") != snapshot_text
    ):
        snapshot_path = out_dir / "config.resumed.toml"
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "STOP"

    start_update = 0
    best = float("inf")
    if args.resume and last_path.exists():
        state = ckpt.load(last_path, map_location=device)
        drifted = run_config.resume_drift(state.get("args") or {}, args)
        if drifted:
            detail = "\n".join(
                f"    {key}: checkpoint {was!r}, config now {now!r}"
                for key, (was, now) in drifted.items()
            )
            print(
                f"refusing to resume {last_path}: the config no longer "
                f"describes the run in it.\n{detail}\n"
                "Resuming would splice two experiments into one curve."
            )
            return 1
        rule.load_state_dict(state["rule"])
        optimiser.load_state_dict(state["optimiser"])
        start_update = int(state["update"])
        best = float(state.get("best", float("inf")))
        ckpt.set_rng_state(state["rng"])
        ckpt.restore_python_rng(rng, state["python_rng"])
        if state.get("eval_indices"):
            eval_indices = list(state["eval_indices"])
            train_indices = [i for i in indices if i not in set(eval_indices)]
        print(f"resumed from {last_path} at update {start_update}")

    try:
        eval_pdb_ids = [catalogue[i].pdb_id for i in eval_indices]
    except (IndexError, KeyError, AttributeError) as problem:
        print(f"cannot resolve eval_indices against this catalogue: {problem!r}")
        return 1

    print(
        f"device {device}, split {args.split}: {len(train_indices)} training, "
        f"{len(eval_indices)} held out (disjoint)"
    )
    print(
        f"{rule.parameter_count()} parameters (hidden {rule.hidden_dim}, message "
        f"{rule.message_dim}), crop {args.crop_size}, k {args.k}, "
        f"{args.damage}-residue window, {args.initial} start"
    )
    print(
        f"weights: pairwise {args.pairwise_weight:g} (window-to-context), "
        f"pairwise-window {args.pairwise_window_weight:g} (window-to-window) "
        "-- unclamped, A^2"
        + (
            "  FULL PAIR SET"
            if args.pairwise_weight and args.pairwise_window_weight
            else "  ONE HALF ONLY, which is not the published loss"
        )
    )

    started = time.perf_counter()
    skipped_updates = 0
    history_path = out_dir / "history.json"
    history: list[dict] = []
    if args.resume and history_path.exists():
        try:
            history = [
                row
                for row in json.loads(history_path.read_text(encoding="utf-8"))
                if row.get("update", 0) <= start_update
            ]
        except (OSError, ValueError):
            history = []

    #: Keyed by ``(structure index, crop seed)``, like ``train_growth``'s and
    #: for the same reason: ``load_crop`` re-crops on every visit, so a state
    #: kept under the index alone would be resumed against a topology it was
    #: never built on.
    pool: dict[tuple[int, int | None], EGNCAPoolEntry] = {}

    for update in range(start_update + 1, args.updates + 1):
        try:
            resume: EGNCAPoolEntry | None = None
            key: tuple[int, int | None]
            if (
                args.pool_probability > 0
                and pool
                and rng.random() < args.pool_probability
            ):
                key = rng.choice(sorted(pool, key=lambda k: (k[0], k[1] or -1)))
                index, crop_seed = key
                try:
                    topo, native, _crop, _used = dataset.load_crop(
                        index, seed_residue=crop_seed, rng=rng
                    )
                except Exception:
                    pool.pop(key, None)
                else:
                    resume = pool[key]
                    if resume.ca.shape[0] != topo.n_residues:
                        pool.pop(key, None)
                        resume = None
            if resume is None:
                index = rng.choice(train_indices)
                try:
                    topo, native, _crop, crop_seed = dataset.load_crop(index, rng=rng)
                except Exception:
                    raise _SkipUpdate
                key = (index, crop_seed)
            if topo.n_residues < 12:
                raise _SkipUpdate

            generator = torch.Generator().manual_seed(
                args.seed * 1_000_003 + update
            )
            steps = (
                rng.randint(args.steps_min, args.steps)
                if args.steps_min != args.steps
                else args.steps
            )
            terms, state = egnca_objective(
                rule,
                native,
                topo,
                steps=steps,
                generator=generator,
                resume_from=resume,
                **objective_kwargs(args),
            )
            # **An undamaged crop carries a finite loss with no graph behind
            # it.** ``damage_window`` returns an all-false mask when the crop's
            # longest covalent run is shorter than ``damage``, and
            # ``pairwise_distance_restraint`` over an empty window is a
            # constant zero with no ``grad_fn`` -- so ``isfinite`` passes and
            # ``backward()`` raises "does not require grad", which killed a
            # 1,000-update run inside its first 250. ``score`` already counts
            # these crops out of the held-out mean; this is the training side
            # of the same statement, and it is a skip rather than a zero
            # because a crop with nothing to fill is not a filled crop.
            if terms.damaged_mask is None or not bool(terms.damaged_mask.any()):
                raise _SkipUpdate
            if not torch.isfinite(terms.total):
                raise _SkipUpdate
            optimiser.zero_grad()
            terms.total.backward()
            # **A finite loss does not imply a finite gradient**, and
            # ``clip_grad_norm_`` turns one infinity into NaN parameters: the
            # total norm is inf, the scale factor is clip/inf = 0, and
            # 0 * inf = NaN. That is what killed two of this project's growth
            # arms, and this rule is the one most likely to produce it -- the
            # authors report exploding gradients and ``phi_x`` is unbounded.
            # Skipping is right rather than clamping: a clamped infinite
            # gradient is a fabricated direction.
            if not all(
                torch.isfinite(p.grad).all()
                for p in rule.parameters()
                if p.grad is not None
            ):
                skipped_updates += 1
                optimiser.zero_grad(set_to_none=True)
                if skipped_updates in (1, 10, 100) or skipped_updates % 500 == 0:
                    print(
                        f"  update {update}: non-finite gradient with a finite "
                        f"loss ({float(terms.total.detach()):.3f}); skipped. "
                        f"{skipped_updates} so far.",
                        flush=True,
                    )
                raise _SkipUpdate
            torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)
            optimiser.step()

            # **The pool's bar is the crop's own fresh baseline**, remembered
            # the first time the crop was seen, so there is no threshold to
            # pick and it self-calibrates per protein. A state worse than not
            # trying is not worth resuming, and without the guard the pool
            # fills with whatever the rule diverges into.
            if args.pool_probability > 0:
                bar = (
                    resume.baseline_rmsd
                    if resume is not None
                    else terms.start_rmsd_damaged
                )
                reached = terms.rmsd_damaged
                if (
                    bar is not None
                    and reached is not None
                    and reached <= bar
                    and terms.damaged_mask is not None
                ):
                    pool[key] = EGNCAPoolEntry(
                        ca=state.ca.detach().clone(),
                        hidden=state.hidden.detach().clone(),
                        damaged=terms.damaged_mask.clone(),
                        baseline_rmsd=bar,
                    )
        except _SkipUpdate:
            pass

        if update % args.eval_every == 0 or update == args.updates:
            scores = score(rule, dataset, eval_indices, args, seed=args.seed)
            elapsed = time.perf_counter() - started
            print(
                f"update {update:7d}  {report(scores)}  "
                f"{elapsed / max(update - start_update, 1):.2f}s/update",
                flush=True,
            )
            history.append({"update": update, **scores})
            history_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
            payload = {
                "rule": rule.state_dict(),
                "optimiser": optimiser.state_dict(),
                "update": update,
                "best": best,
                "args": recorded_args(args),
                "runtime": {
                    key: getattr(args, key)
                    for key in run_config.RUNTIME_ONLY
                    if hasattr(args, key)
                },
                "eval_indices": eval_indices,
                "eval_pdb_ids": eval_pdb_ids,
                "scores": scores,
                "rng": ckpt.rng_state(),
                "python_rng": rng.getstate(),
                "skipped_updates": skipped_updates,
            }
            ckpt.save(last_path, payload)
            # Selection is on the held-out `total`, which is the only scalar
            # here that prices the two halves of the restraint against each
            # other.
            if scores["total"] == scores["total"] and scores["total"] < best:
                best = scores["total"]
                payload["best"] = best
                ckpt.save(best_path, payload)

        if stop_file.exists():
            print(f"stopping at update {update}: {stop_file} exists")
            break
        if args.max_seconds and time.perf_counter() - started > float(
            args.max_seconds
        ):
            print(f"stopping at update {update}: out of time")
            break

    if skipped_updates:
        print(
            f"{skipped_updates} update(s) skipped for a non-finite gradient -- "
            "this run trained on fewer updates than its curve's x-axis says."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
