"""Does the trained E(n)-GNCA port have an ATTRACTOR, or only a trajectory?

    .venv/Scripts/python.exe scripts/persistence_readout.py --checkpoint runs/port/best.pt
    .venv/Scripts/python.exe scripts/persistence_readout.py --steps 15,25,50,100,250,1000

**This is the published rule's headline claim and nothing else in this project
measures it.** Gala train over 15-25 steps and then evaluate to 10,000-100,000;
everything here -- every arm, every readout, the port's own 2.17/2.53 A pass --
is read at the step count the rule was trained on. A rule that merely *passes
through* a good configuration at step 25 and then drifts away is a different
object from one that *converges to* it and stays, and the two are
indistinguishable from any number in `HANDOFF.md`.

So this probe asks three questions, per held-out crop, from the same damaged
window the trainer evaluates on:

  * **Does it hold?** `rmsd_dmg` and `rmsd_dmg_fit_intact` at 15, 25, 50, 100,
    250, 1,000 and 10,000 steps -- the same two quantities
    `train_egnca.score` prints, through `fape.backbone_rmsd_on` with the same
    two superposition conventions.
  * **Does it stop?** The first step at which the largest per-residue
    displacement over one step falls below 0.1 A. `None` is a rule that never
    settles inside the cap, which is a *trajectory* and not a fixed point
    however good its step-25 value looks.
  * **Does it stay stopped?** `rmsd_dmg` at convergence against `rmsd_dmg` at
    the last step count run. A genuine attractor holds; a rule without one
    drifts away from its own best configuration, and the drift column is where
    that shows.

And it asks all of it a second time under **the published perturbation
protocol**: run 24 steps, re-apply the damage to the state the rule reached,
and keep running. That is a perturbation of a *lived-in* state -- the hidden
channels are the ones the rule built, not the published all-ones start -- which
is the harder half of the claim and the one the replay pool trains towards.

**WHY THIS SCRIPT REFUSES RATHER THAN DEFAULTS.** `docs/claims/a-probe-pointed
-at-best-pt-silently-used-its-own-defaults.md` records four invalid results
from one cause: a probe read a `best.pt` with no `args`, fell through to its
own argparse defaults, and printed 30 steps for a run trained at 120 under a
column heading. So no rollout setting here has a default. Every one is read out
of the checkpoint, checked key-by-key against `train_egnca.build_schema`, and
handed to the rule through `train_egnca.build_rule` and
`train_egnca.objective_kwargs` -- *the trainer's own two functions*, so a
setting cannot reach the trainer and miss this probe. A recorded key the schema
does not know, a schema key the checkpoint does not carry, a value outside its
`choices`, or an eval split that does not match the catalogue on disk is a
refusal with exit 1. The step counts and the paths are flags because they are
the probe's own question, not the run's.

**`n` and `diverged` are printed before any other column**, which is this
project's standing rule: the rule is unbounded, its authors report exploding
gradients, and a mean over the crops that survived to step 10,000 would be the
best-looking row in the table. They are reported at *every* step count
separately, because a crop can be finite at 25 and gone at 1,000 -- which is
precisely the failure this probe exists to catch.

Nothing here writes to `src/aanca/egnca.py` or `scripts/train_egnca.py`: the
rollout is re-run step by step rather than through `egnca_objective`, because
the objective reports only its endpoint and this question is about the shape of
the whole curve.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca import config as run_config
from aanca.data import RepackingDataset, read_manifest
from aanca.egnca import (
    EGNCAState,
    ca_positions,
    egnca_neighbours,
    egnca_step,
    with_ca_positions,
)
from aanca.fape import backbone_rmsd_on
from aanca.growth import baseline_coords, damage_window, growth_links

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
RUNS = ROOT / "runs"

#: The step counts the readout is taken at. 15 and 25 are the training horizon;
#: 10,000 is the published evaluation length. Everything between is there so a
#: drift can be *located* rather than only detected.
STEP_MARKS = (15, 25, 50, 100, 250, 1000, 10000)

#: A step this small is a fixed point for this task's purposes: the window has
#: ~13 A to travel and CA-CA is 3.8 A, so 0.1 A per step is two orders of
#: magnitude below the quantity being measured. It is a *threshold on the
#: largest* per-residue move, not on the mean -- a mean hides one residue
#: orbiting forever.
FIXED_POINT_TOL = 0.1

#: The published protocol's perturbation point: damage at t=24, then keep
#: running. Inside the 15-25 training horizon on purpose -- the state being
#: perturbed is one the rule was trained to produce.
DAMAGE_AT = 24

#: `tag` is not a setting: `config.to_namespace` derives it from the section
#: name, so it is in the checkpoint's `args` and not in the schema.
DERIVED_KEYS = frozenset(run_config.DERIVED)


class ProbeRefusal(Exception):
    """This checkpoint cannot be honoured, so nothing is measured.

    Raised rather than warned, for the reason in this module's docstring: a
    default is indistinguishable from a measurement once it is printed under a
    column heading.
    """


def trainer_module():
    """`scripts/train_egnca.py` as a module, so its own functions do the work.

    Loaded by path rather than imported because `scripts/` is not a package.
    This probe then builds the rule with `build_rule` and the objective
    settings with `objective_kwargs` -- the exact two functions the trainer
    uses -- so there is no second place for a setting to go missing from. That
    is `objective_kwargs`' own stated purpose, and this script is the second
    caller it was written for.
    """
    spec = importlib.util.spec_from_file_location(
        "train_egnca", ROOT / "scripts" / "train_egnca.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("train_egnca", module)
    spec.loader.exec_module(module)
    return module


def settings_from_checkpoint(
    state: dict,
    schema: argparse.ArgumentParser,
    schema_name: str = "scripts/train_egnca.py",
):
    """The run's own settings, or a refusal. Never a default.

    ``schema_name`` is what the refusals call the schema they were handed. It
    exists because the message used to say ``scripts/train_egnca.py`` whatever
    was passed, and once a probe could declare ``--trainer`` that was a refusal
    naming the wrong file -- the most misleading thing a refusal can do, since
    its whole job is to say which two things disagree. The default is the
    historical caller, so nothing that does not pass it changes.

    Four ways this refuses, each one a shape the cited incident had:

      * no `args` at all -- the original fault;
      * a recorded key the schema does not know, so this probe could not act
        on it even if it read it;
      * a schema key the checkpoint does not carry, which is where a silent
        default would enter;
      * a value the schema would reject (outside `choices`, or of the wrong
        type).
    """
    recorded = state.get("args")
    if not recorded:
        raise ProbeRefusal(
            "this checkpoint carries no `args`, so every rollout setting would "
            "be this probe's own default. That is the fault in "
            "docs/claims/a-probe-pointed-at-best-pt-silently-used-its-own-"
            "defaults.md. Point at a checkpoint written by "
            "scripts/train_egnca.py, or at the sibling last.pt."
        )
    if not isinstance(recorded, dict):
        raise ProbeRefusal(f"`args` is {type(recorded).__name__}, not a dict")

    actions = {
        action.dest: action
        for action in schema._actions
        if action.dest not in ("help", argparse.SUPPRESS)
    }
    runtime = set(run_config.RUNTIME_ONLY)

    unknown = sorted(set(recorded) - set(actions) - DERIVED_KEYS)
    if unknown:
        raise ProbeRefusal(
            f"this checkpoint records settings {schema_name}'s schema "
            f"does not define: {', '.join(unknown)}. The probe cannot honour a "
            "setting it has no name for, and running anyway would measure a "
            "rule configured unlike the run."
        )
    missing = sorted(set(actions) - runtime - set(recorded))
    if missing:
        raise ProbeRefusal(
            "this checkpoint is missing settings the schema defines: "
            f"{', '.join(missing)}. Each one would fall through to a default, "
            "which is the fault this refusal exists for."
        )

    for key, value in sorted(recorded.items()):
        action = actions.get(key)
        if action is None:  # `tag`, the only derived key
            continue
        if action.choices is not None and value not in action.choices:
            raise ProbeRefusal(
                f"recorded {key}={value!r} is not one of this schema's choices "
                f"{tuple(action.choices)}: the checkpoint and the code disagree "
                "about what this setting can be."
            )
        if action.type is not None and value is not None:
            try:
                coerced = action.type(value)
            except (TypeError, ValueError) as problem:
                raise ProbeRefusal(
                    f"recorded {key}={value!r} is not a "
                    f"{action.type.__name__}: {problem!r}"
                ) from problem
            if isinstance(value, bool) is not isinstance(coerced, bool):
                raise ProbeRefusal(
                    f"recorded {key}={value!r} does not survive this schema's "
                    f"{action.type.__name__}"
                )

    args = argparse.Namespace(**recorded)
    for key in run_config.RUNTIME_ONLY:
        if not hasattr(args, key):
            setattr(args, key, None)
    return args


def load_rule(state: dict, args, trainer, device: torch.device):
    """The rule the run trained, built by the trainer's own `build_rule`.

    `strict=True` on purpose: a width or a switch that changed the shapes is a
    refusal here rather than a partially loaded rule several screens into a
    rollout.
    """
    rule = trainer.build_rule(args).to(device)
    weights = ckpt.rule_weights(state)
    try:
        rule.load_state_dict(weights, strict=True)
    except RuntimeError as problem:
        raise ProbeRefusal(
            f"the recorded settings build a rule these weights do not fit: "
            f"{problem}"
        ) from problem
    rule.eval()
    return rule


def held_out(state: dict, args, trainer, device: torch.device):
    """The dataset, catalogue and eval indices the run itself held out.

    Rebuilt exactly as `train_egnca.main` builds them -- same manifest, same
    `on_disk` redirect, same `wanted` slice, same `RepackingDataset`
    arguments -- and then **checked against the recorded `eval_pdb_ids`**. A
    laptop pool and a cluster pool give `available[:416]` a different 416
    proteins, so the same index means a different protein and a number from one
    is not comparable with a number from the other. That check is why this
    probe cannot quietly score the wrong proteins.
    """
    splits = read_manifest(MANIFEST)
    if args.split not in splits:
        raise ProbeRefusal(f"no split {args.split!r}; available: {sorted(splits)}")
    available = trainer.on_disk(splits[args.split])
    if not available:
        raise ProbeRefusal(
            f"no coordinates in {PDB} -- run scripts/fetch_structures.py"
        )
    wanted = args.structures + args.eval_structures
    catalogue = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        catalogue,
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=max(len(catalogue), 1),
    )

    eval_indices = list(state.get("eval_indices") or [])
    if not eval_indices:
        raise ProbeRefusal(
            "this checkpoint records no `eval_indices`, so the held-out split "
            "would be reshuffled here and the probe would score crops the run "
            "trained on."
        )
    recorded_ids = list(state.get("eval_pdb_ids") or [])
    if recorded_ids:
        try:
            now = [catalogue[i].pdb_id for i in eval_indices]
        except IndexError as problem:
            raise ProbeRefusal(
                f"eval_indices do not fit this catalogue of {len(catalogue)}: "
                f"{problem!r}"
            ) from problem
        if now != recorded_ids:
            raise ProbeRefusal(
                "the held-out indices name different proteins in this "
                f"catalogue than the run recorded: {recorded_ids} became "
                f"{now}. This pool is not the run's pool."
            )
    return dataset, catalogue, eval_indices


# -- the measurement --------------------------------------------------------


def step_displacement(
    before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[float, float]:
    """``(max, mean)`` per-residue displacement in Angstrom over one step.

    Read over ``mask`` -- the moving window -- because the surviving context is
    held at its crystal pose and contributes exact zeros: a mean over the whole
    structure would be the window's mean divided by the crop size, which falls
    as the crop grows rather than as the rule settles. The *max* is the same
    number either way.

    Zero for an identity step, which is what makes the fixed-point test an
    assertion about the rule and not about this function.
    """
    delta = (after - before).norm(dim=-1)
    if mask is not None:
        delta = delta[mask]
    if delta.numel() == 0:
        return 0.0, 0.0
    return float(delta.max()), float(delta.mean())


@dataclass
class CropTrace:
    """One crop under one protocol: the curve, where it settled, and the drift."""

    marks: dict[int, dict] = field(default_factory=dict)
    converged_at: int | None = None
    converged_rmsd: float | None = None
    nonfinite_at: int | None = None
    steps_run: int = 0
    seconds: float = 0.0

    def drift(self) -> float | None:
        """`rmsd_dmg` at the last step run minus `rmsd_dmg` at convergence.

        `None` when it never converged, because there is then no fixed point to
        drift away from -- and that is the finding, not a missing number.
        """
        if self.converged_at is None or self.converged_rmsd is None:
            return None
        finite = [
            (step, row)
            for step, row in sorted(self.marks.items())
            if row.get("finite")
        ]
        if not finite:
            return None
        return finite[-1][1]["rmsd_dmg"] - self.converged_rmsd


def _measure(rule_ca, native, topo, damaged, intact) -> dict:
    """`rmsd_dmg` and `rmsd_dmg_fit_intact`, the trainer's own two conventions.

    Both through `fape.backbone_rmsd_on`, the whole structure then the intact
    context, exactly as `egnca_objective` reads them -- so a number here is
    comparable with a number in the trainer's log and not merely similar to one.
    """
    if not bool(torch.isfinite(rule_ca).all()):
        return {"finite": False, "rmsd_dmg": None, "rmsd_dmg_fit_intact": None}
    final = with_ca_positions(native, rule_ca, topo)
    everything = torch.ones_like(damaged)
    return {
        "finite": True,
        "rmsd_dmg": float(
            backbone_rmsd_on(final, native, topo, everything, damaged)
        ),
        "rmsd_dmg_fit_intact": float(
            backbone_rmsd_on(final, native, topo, intact, damaged)
        ),
    }


def _selector(args, native, topo):
    """The neighbour selection, under the run's own `native_edges` setting.

    Lifted from `egnca_rollout` rather than called through it: the rollout
    returns only its end state and this probe has to look between the steps.
    The *selection* is the same construction with the same `k` and `cutoff`.
    """
    if args.native_edges:
        fixed = egnca_neighbours(
            native, topo, args.k, args.cutoff, args.device_neighbours
        )
        return lambda state: fixed

    def live(state):
        return egnca_neighbours(
            with_ca_positions(native, state.ca.detach(), topo),
            topo,
            args.k,
            args.cutoff,
            args.device_neighbours,
        )

    return live


def trace_rollout(
    rule,
    state: EGNCAState,
    selector,
    native,
    topo,
    damaged,
    marks,
    tol: float = FIXED_POINT_TOL,
) -> tuple[CropTrace, EGNCAState | None]:
    """Step the rule to `max(marks)`, reading out at each mark.

    Returns the trace and the end state -- the state so the published protocol
    can perturb what the rule actually reached without stepping those same 24
    steps a second time, and `None` for it when the rollout went non-finite.

    Stops at the first non-finite state and records where: every later mark is
    then `finite: False` rather than absent, so the `diverged` count at step
    10,000 is a count and not an inference from a missing row.
    """
    intact = ~damaged
    trace = CropTrace()
    cap = max(marks)
    wanted = set(marks)
    started = time.perf_counter()
    diverged = False
    for step in range(1, cap + 1):
        previous = state.ca
        selection = selector(state)
        state = egnca_step(
            rule, state, selection.neighbours, selection.edge_mask, damaged
        )
        trace.steps_run = step
        if not bool(torch.isfinite(state.ca).all()):
            trace.nonfinite_at = step
            diverged = True
            break
        largest, average = step_displacement(previous, state.ca, damaged)
        if trace.converged_at is None and largest < tol:
            trace.converged_at = step
            trace.converged_rmsd = _measure(
                state.ca, native, topo, damaged, intact
            )["rmsd_dmg"]
        if step in wanted:
            trace.marks[step] = {
                **_measure(state.ca, native, topo, damaged, intact),
                "max_step": largest,
                "mean_step": average,
            }
    trace.seconds = time.perf_counter() - started
    for step in sorted(wanted):
        trace.marks.setdefault(
            step,
            {
                "finite": False,
                "rmsd_dmg": None,
                "rmsd_dmg_fit_intact": None,
                "max_step": None,
                "mean_step": None,
            },
        )
    return trace, (None if diverged else state)


def crop_readout(
    rule, args, native, topo, generator, marks, damage_at: int, tol: float
) -> dict | None:
    """Both protocols on one crop, from the trainer's own damaged start.

    The start state is built by the same three calls in the same order as
    `egnca_objective` -- `growth_links`, `damage_window`, `baseline_coords` --
    off a generator seeded as `train_egnca.score` seeds it, so the window this
    probe fills is the window the trainer's held-out column was read on.

    `None` when the crop has no window to fill (its longest covalent run is
    shorter than `damage`), which `score` also counts out rather than averaging
    in as a perfect fill.
    """
    links = growth_links(topo)
    damaged = damage_window(
        topo.n_residues, links, args.damage, generator, native.device
    )
    if not bool(damaged.any()) or int((~damaged).sum()) < 3:
        return None
    with torch.no_grad():
        start = baseline_coords(
            native, topo, links, ~damaged, args.initial, generator
        )
    start_ca = ca_positions(start, topo).detach().to(rule.dtype)
    hidden = rule.initial_hidden(
        topo.n_residues, device=native.device, dtype=rule.dtype
    )
    intact = ~damaged
    selector = _selector(args, native, topo)

    begin = with_ca_positions(native, start_ca, topo)
    everything = torch.ones_like(damaged)
    out = {
        "n_residues": int(topo.n_residues),
        "window": int(damaged.sum()),
        "start_rmsd_dmg": float(
            backbone_rmsd_on(begin, native, topo, everything, damaged)
        ),
        "start_rmsd_dmg_fit_intact": float(
            backbone_rmsd_on(begin, native, topo, intact, damaged)
        ),
    }

    from_start, _end = trace_rollout(
        rule,
        EGNCAState(ca=start_ca, hidden=hidden),
        selector,
        native,
        topo,
        damaged,
        marks,
        tol,
    )
    out["from_start"] = _as_json(from_start)

    # The published protocol: run to `damage_at`, perturb the state the rule
    # reached, keep going. The hidden channels are **not** reset -- the
    # perturbation is on the coordinates, as their Gaussian noise is, so what
    # is being tested is whether a lived-in state returns rather than whether a
    # fresh one arrives. The displacement re-applied is the same baseline walk
    # the start used, i.e. the project's own damage operator on the same window.
    if damage_at > 0:
        warm, warm_state = trace_rollout(
            rule,
            EGNCAState(ca=start_ca, hidden=hidden),
            selector,
            native,
            topo,
            damaged,
            (damage_at,),
            tol,
        )
        reached = warm.marks[damage_at]
        out["at_damage_point"] = {
            "step": damage_at,
            **{k: reached[k] for k in ("finite", "rmsd_dmg", "rmsd_dmg_fit_intact")},
        }
    else:
        warm_state = EGNCAState(ca=start_ca, hidden=hidden)
    if warm_state is None:
        out["after_damage"] = None
        return out
    perturbed = warm_state.ca.clone()
    perturbed[damaged] = start_ca[damaged]
    after, _end = trace_rollout(
        rule,
        EGNCAState(ca=perturbed, hidden=warm_state.hidden),
        selector,
        native,
        topo,
        damaged,
        marks,
        tol,
    )
    out["after_damage"] = _as_json(after)
    return out


def _as_json(trace: CropTrace) -> dict:
    return {
        "marks": {str(step): row for step, row in sorted(trace.marks.items())},
        "converged_at": trace.converged_at,
        "converged_rmsd": trace.converged_rmsd,
        "drift": trace.drift(),
        "nonfinite_at": trace.nonfinite_at,
        "steps_run": trace.steps_run,
        "seconds": trace.seconds,
    }


# -- aggregation and the table ----------------------------------------------

COLUMNS = ("rmsd_dmg", "rmsd_dmg_fit_intact", "max_step", "mean_step")


def aggregate(crops: list[dict], protocol: str, marks) -> list[dict]:
    """Per step count: `n`, `diverged`, then mean, sd and minimum per column.

    `n` and `diverged` first and at every step count, because this rule is
    unbounded and a crop finite at 25 can be gone at 1,000 -- a single
    `diverged` for the whole table would hide exactly the thing being measured.
    A mean over survivors is not a result, so the two counts are what the row
    is read on.
    """
    rows = []
    for step in sorted(marks):
        values = {key: [] for key in COLUMNS}
        n = 0
        diverged = 0
        for crop in crops:
            trace = crop.get(protocol)
            row = (trace or {}).get("marks", {}).get(str(step))
            if not row or not row.get("finite"):
                diverged += 1
                continue
            n += 1
            for key in COLUMNS:
                values[key].append(row[key])
        entry = {"steps": step, "n": n, "diverged": diverged}
        for key in COLUMNS:
            series = values[key]
            entry[key] = {
                "mean": statistics.fmean(series) if series else None,
                "sd": statistics.stdev(series) if len(series) > 1 else 0.0
                if series
                else None,
                "min": min(series) if series else None,
            }
        rows.append(entry)
    return rows


def fixed_point_summary(crops: list[dict], protocol: str) -> dict:
    """How many crops reached a fixed point, where, and what the drift was."""
    converged = [
        crop[protocol]["converged_at"]
        for crop in crops
        if crop.get(protocol) and crop[protocol]["converged_at"] is not None
    ]
    drifts = [
        crop[protocol]["drift"]
        for crop in crops
        if crop.get(protocol) and crop[protocol]["drift"] is not None
    ]
    return {
        "crops": sum(1 for crop in crops if crop.get(protocol)),
        "converged": len(converged),
        "never_converged": sum(
            1
            for crop in crops
            if crop.get(protocol) and crop[protocol]["converged_at"] is None
        ),
        "steps_to_fixed_point": {
            "min": min(converged) if converged else None,
            "median": statistics.median(converged) if converged else None,
            "max": max(converged) if converged else None,
        },
        "drift_after_fixed_point": {
            "mean": statistics.fmean(drifts) if drifts else None,
            "sd": statistics.stdev(drifts) if len(drifts) > 1 else 0.0
            if drifts
            else None,
            "max_abs": max((abs(d) for d in drifts), default=None),
        },
    }


def _cell(value, width=7, places=3) -> str:
    if value is None:
        return "      -".rjust(width)
    if isinstance(value, float) and not math.isfinite(value):
        return "    nan".rjust(width)
    return f"{value:{width}.{places}f}"


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(
        "  steps    n  diverged   rmsd_dmg       sd      min   "
        "fit_intact       sd      min   max_step  mean_step"
    )
    for row in rows:
        print(
            f"  {row['steps']:5d} {row['n']:4d} {row['diverged']:9d}  "
            + _cell(row["rmsd_dmg"]["mean"], 9)
            + _cell(row["rmsd_dmg"]["sd"], 9)
            + _cell(row["rmsd_dmg"]["min"], 9)
            + "  "
            + _cell(row["rmsd_dmg_fit_intact"]["mean"], 9)
            + _cell(row["rmsd_dmg_fit_intact"]["sd"], 9)
            + _cell(row["rmsd_dmg_fit_intact"]["min"], 9)
            + "  "
            + _cell(row["max_step"]["mean"], 9, 4)
            + _cell(row["mean_step"]["mean"], 9, 4)
        )


def build_parser() -> argparse.ArgumentParser:
    """Paths and step counts only. Every setting that changes a result is read
    out of the checkpoint -- see this module's docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(RUNS / "port" / "best.pt"))
    p.add_argument(
        "--steps",
        default=",".join(str(step) for step in STEP_MARKS),
        help="comma-separated step counts to read the rollout out at",
    )
    p.add_argument(
        "--damage-at",
        type=int,
        default=DAMAGE_AT,
        help="re-apply the damage at this step and continue (0 disables)",
    )
    p.add_argument("--tolerance", type=float, default=FIXED_POINT_TOL)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="score only the first N held-out crops (correctness checks)",
    )
    return p


def main() -> int:
    options = build_parser().parse_args()
    marks = tuple(
        sorted({int(piece) for piece in options.steps.split(",") if piece.strip()})
    )
    if not marks or marks[0] < 1:
        print("--steps must be positive step counts")
        return 2

    device = torch.device(options.device)
    trainer = trainer_module()
    schema = trainer.build_schema()
    path = Path(options.checkpoint)
    if not path.exists():
        print(f"no checkpoint at {path}")
        return 1
    state = ckpt.load(path, map_location=device)
    try:
        args = settings_from_checkpoint(state, schema)
        rule = load_rule(state, args, trainer, device)
        dataset, catalogue, eval_indices = held_out(state, args, trainer, device)
    except ProbeRefusal as refusal:
        print(f"refusing to measure {path}:\n  {refusal}")
        return 1

    print(f"checkpoint : {path} (update {state.get('update')})")
    print(
        "settings   : read from the checkpoint and checked key-by-key against "
        "scripts/train_egnca.py's schema; the rule is built by the trainer's "
        "own build_rule."
    )
    print(
        f"  rule: hidden {args.hidden_dim}, message {args.message_dim}, "
        f"coord_act {args.has_coord_act}, residual {args.is_residual}, "
        f"act {args.activation}, norm {args.norm_type} "
        f"({rule.parameter_count()} parameters)"
    )
    print(
        f"  task: crop {args.crop_size}, k {args.k}, cutoff {args.cutoff}, "
        f"{args.damage}-residue window, {args.initial} start, native_edges "
        f"{args.native_edges}, seed {args.seed}"
    )
    print(
        f"  trained over {args.steps_min}-{args.steps} steps; this probe reads "
        f"out at {', '.join(str(m) for m in marks)}"
    )
    objective = trainer.objective_kwargs(args)
    print(
        "  objective settings honoured: "
        + ", ".join(f"{k}={v!r}" for k, v in sorted(objective.items()))
    )

    rng = random.Random(args.seed)
    indices = eval_indices[: options.limit] if options.limit else eval_indices
    crops: list[dict] = []
    skipped = 0
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
            except Exception:
                skipped += 1
                continue
            if topo.n_residues < 12:
                skipped += 1
                continue
            generator = torch.Generator().manual_seed(args.seed + index)
            readout = crop_readout(
                rule,
                args,
                native,
                topo,
                generator,
                marks,
                options.damage_at,
                options.tolerance,
            )
            if readout is None:
                skipped += 1
                continue
            readout["index"] = index
            readout["pdb_id"] = catalogue[index].pdb_id
            crops.append(readout)
            print(
                f"  crop {index:4d} {readout['pdb_id']}: start "
                f"{readout['start_rmsd_dmg']:.3f} A, "
                f"{readout['from_start']['steps_run']} steps in "
                f"{readout['from_start']['seconds']:.1f} s",
                flush=True,
            )

    if not crops:
        print("no held-out crop had a window to fill: nothing measured")
        return 1

    start_mean = statistics.fmean(crop["start_rmsd_dmg"] for crop in crops)
    protocols = {"from_start": "FROM THE DAMAGED START (step 0 = the trainer's own)"}
    if options.damage_at > 0:
        protocols["after_damage"] = (
            f"AFTER RE-DAMAGE AT STEP {options.damage_at} (the published protocol)"
        )

    payload = {
        "checkpoint": str(path),
        "update": state.get("update"),
        "settings": vars(args),
        "step_marks": list(marks),
        "tolerance": options.tolerance,
        "damage_at": options.damage_at,
        "crops": crops,
        "skipped_crops": skipped,
        "start_rmsd_dmg_mean": start_mean,
        "aggregate": {
            name: aggregate(crops, name, marks) for name in protocols
        },
        "fixed_point": {
            name: fixed_point_summary(crops, name) for name in protocols
        },
    }
    out_path = Path(options.out) if options.out else path.parent / "persistence.json"
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(
        f"\n{len(crops)} held-out crops measured, {skipped} counted out (no "
        f"window or no crop). Mean start_rmsd_dmg {start_mean:.3f} A -- this is "
        "the trainer's own `start` column, and it is the survivorship check: it "
        "cannot depend on the weights, so a value away from the run's means "
        "these are not the run's crops."
    )
    for name, title in protocols.items():
        rows = payload["aggregate"][name]
        print_table(rows, title)
        summary = payload["fixed_point"][name]
        print(
            f"  steps to fixed point (max per-residue step < "
            f"{options.tolerance} A): {summary['converged']} of "
            f"{summary['crops']} crops converged, "
            f"{summary['never_converged']} never did; "
            f"min {summary['steps_to_fixed_point']['min']}, median "
            f"{summary['steps_to_fixed_point']['median']}, max "
            f"{summary['steps_to_fixed_point']['max']}"
        )
        drift = summary["drift_after_fixed_point"]
        print(
            "  drift after the fixed point (rmsd_dmg at the last step run minus "
            "at convergence): mean "
            + (f"{drift['mean']:+.4f}" if drift["mean"] is not None else "-")
            + ", sd "
            + (f"{drift['sd']:.4f}" if drift["sd"] is not None else "-")
            + ", largest |drift| "
            + (f"{drift['max_abs']:.4f}" if drift["max_abs"] is not None else "-")
            + " A"
        )

    seconds = sum(crop["from_start"]["seconds"] for crop in crops)
    steps = sum(crop["from_start"]["steps_run"] for crop in crops)
    if steps:
        print(
            f"\ncost: {seconds:.1f} s for {steps} steps over {len(crops)} crops "
            f"= {1000 * seconds / steps:.1f} s per crop per 1,000 steps on "
            f"{device}. All {len(crops)} crops to 10,000 steps is about "
            f"{10_000 * len(crops) * seconds / steps / 60:.0f} min per protocol."
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
