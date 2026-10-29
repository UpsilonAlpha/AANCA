"""Held-out evaluation at fixed difficulty, which the training loss cannot give.

**The training objective is not a progress signal, and this is not a detail.**
The reconstruction term scores the rule's output against
``wrap_to_pi(native_chis - measure_chis(seen_coords))``, where ``seen_coords``
is the rule's own input at a randomly drawn late step. Early in training the
rule barely moves, so ``seen`` is still the damaged structure and the target is
the whole damage. As it improves, ``seen`` approaches native and the targets
shrink toward zero. A falling loss therefore conflates two unrelated things --
better conditional prediction, and an easier question -- and it can rise while
the model improves, if improvement moves the sampled step into a regime where
little delta remains and the concentration head is penalised for overconfidence.

So progress is measured somewhere else, under conditions that do not move:

* a **fixed** set of structures, cropped with a **fixed** seed;
* **fixed** damage, drawn once and stored, not resampled per evaluation;
* a **fixed** asynchronous update mask sequence, since the mask is random and
  run-to-run mask noise would otherwise contaminate the curve;
* the published metric definitions, so the number is on the same scale as the
  table it will eventually be compared against.

Three things are reported, because the objective makes three distinct claims
and they can fail separately:

**Does it pack?** chi MAE and accuracy against native, per damage level. The
headline.

**Does it converge?** Mean ``|delta|`` per step. A rule that oscillates or
drifts without settling can still score well at the step it was supervised at,
which is exactly the failure a fixed point is supposed to rule out.

**Does it stay?** Drift from a native start, over a rollout much longer than
anything the loss sees at one step. Reported as a mean *and* a worst atom, and
the mean is the one to read as a curve: the worst atom is a max over atoms of a
max over structures, so a single flipped terminal chi in one structure sets it
and it swings by a factor of two between evaluations that agree on everything
else. An identity-initialised untrained rule scores a perfect zero on both, so
these numbers only mean something after training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .data import RepackingDataset, von_mises_deltas
from .metrics import BenchmarkRow, Convention, benchmark_table
from .model import AANCARule
from .rollout import RolloutConfig, initial_state, rollout
from .torch_geometry import apply_chi_deltas, measure_chis, wrap_to_pi
from .torch_graph import FeatureTopology


@dataclass(frozen=True)
class EvalCase:
    """One evaluation case, built once and never resampled.

    Holding the damaged coordinates rather than a seed is deliberate: a seed
    only reproduces the damage if every sampling call upstream of it stays
    identical, and that is a promise about unrelated code. The coordinates are
    the thing that must not change, so they are what is stored.
    """

    pdb_id: str
    topology: FeatureTopology
    native: Tensor
    damaged: Tensor
    concentration: float
    #: Seed for this case's update-mask sequence, so the asynchronous schedule
    #: is the same at every evaluation.
    mask_seed: int


def build_eval_set(
    dataset: RepackingDataset,
    indices: list[int],
    concentrations: tuple[float, ...] = (100.0, 20.0, 0.0),
    seed: int = 0,
) -> list[EvalCase]:
    """One case per (structure, damage level), all of it deterministic.

    Crops use ``random.Random(index)``, which is :meth:`RepackingDataset.load`'s
    own default, so a case's region is pinned to its structure rather than to
    the order cases happen to be built in.
    """
    cases: list[EvalCase] = []
    for index in indices:
        topo, native = dataset.load(index)
        for level in concentrations:
            # One generator per (index, level), derived arithmetically rather
            # than with hash(): adding a level or a structure must not change
            # the damage of any existing case, and Python's hash is not a
            # promise about anything.
            stream = (seed * 1_000_003 + index * 1009 + int(level * 10)) % (2**31)
            generator = torch.Generator().manual_seed(stream)
            deltas = von_mises_deltas(
                topo.torsion.chi_mask, level, generator=generator
            )
            cases.append(
                EvalCase(
                    pdb_id=dataset.specs[index].pdb_id,
                    topology=topo,
                    native=native,
                    damaged=apply_chi_deltas(native, deltas, topo.torsion),
                    concentration=float(level),
                    mask_seed=stream,
                )
            )
    return cases


@dataclass
class EvalResult:
    """What one evaluation says, split by the claim it bears on."""

    #: Published-layout row per damage level, plus ``"pooled"`` over all levels.
    rows: dict[str, BenchmarkRow] = field(default_factory=dict)
    #: Mean ``|delta|`` in degrees at each rollout step, averaged over cases.
    #: Should decay. Oscillation here is a fixed-point failure the headline
    #: accuracy can hide.
    convergence: list[float] = field(default_factory=list)
    #: Max coordinate drift in Angstrom over a rollout started from native,
    #: worst case over the evaluation set. A guard, not a trend: it is a max
    #: over atoms of a max over structures, so one side chain flipping a
    #: terminal chi in one structure sets it, and it moves by more than a
    #: factor of two between evaluations that are otherwise indistinguishable.
    native_drift: float = 0.0
    #: Mean displacement in Angstrom over the same rollouts, over supervised
    #: side-chain atoms and structures. This is the one to read as a curve;
    #: measured at 0.015 A while the max said 1.7, which is the gap between
    #: "the rule stays put" and "one arginine did not".
    native_drift_mean: float = 0.0
    #: Worst-atom drift per structure, so the tail is visible rather than
    #: collapsed into the max.
    native_drift_by_structure: dict[str, float] = field(default_factory=dict)
    #: Mean chi MAE in degrees of the *damaged* inputs against native, per
    #: level. The baseline a prediction has to beat; without it a good-looking
    #: MAE at kappa=100 says nothing, since mild damage is already close.
    input_mae: dict[str, float] = field(default_factory=dict)

    def report(self) -> str:
        lines = [self.rows["pooled"].header()]
        for label in sorted(self.rows, key=lambda k: (k == "pooled", k)):
            lines.append(self.rows[label].row(label))
        if self.convergence:
            trace = "  ".join(f"{v:.1f}" for v in self.convergence)
            lines.append(f"mean |delta| per step (deg): {trace}")
        lines.append(
            f"drift from a native start: {self.native_drift_mean:.4f} A mean, "
            f"{self.native_drift:.3f} A worst atom"
        )
        return "\n".join(lines)


@torch.no_grad()
def evaluate(
    rule: AANCARule,
    cases: list[EvalCase],
    config: RolloutConfig,
    convention: Convention | None = None,
) -> EvalResult:
    """Roll out every case from its stored damage and score the result.

    No gradient, no resampling, no sample pool: this measures the rule, not the
    training procedure. Each case uses its own fixed mask sequence.
    """
    if not cases:
        raise ValueError("no evaluation cases")

    was_training = rule.training
    rule.eval()
    try:
        by_level: dict[str, list] = {}
        curves: list[list[float]] = []
        input_mae_num: dict[str, float] = {}
        input_mae_den: dict[str, int] = {}
        drift = 0.0

        for case in cases:
            topo, torsion = case.topology, case.topology.torsion
            result = rollout(
                rule,
                initial_state(case.damaged, rule),
                topo,
                config,
                generator=torch.Generator().manual_seed(case.mask_seed),
            )
            predicted = result.state.coords
            label = f"kappa {case.concentration:g}"
            by_level.setdefault(label, []).append((predicted, case.native, topo))
            curves.append([torch.rad2deg(torch.tensor(v)).item()
                           for v in result.mean_abs_delta])

            # How far the input already was. Restricted to supervised residues
            # so the baseline and the prediction are scored on the same set.
            scored = topo.supervised_mask.unsqueeze(-1) & torsion.chi_mask
            residual = wrap_to_pi(
                measure_chis(case.damaged, torsion)
                - measure_chis(case.native, torsion)
            ).abs()
            input_mae_num[label] = input_mae_num.get(label, 0.0) + float(
                torch.rad2deg(residual[scored]).sum()
            )
            input_mae_den[label] = input_mae_den.get(label, 0) + int(scored.sum())

        out = EvalResult()
        for label, pairs in by_level.items():
            out.rows[label] = benchmark_table(pairs, convention=convention)
        out.rows["pooled"] = benchmark_table(
            [p for pairs in by_level.values() for p in pairs], convention=convention
        )
        out.input_mae = {
            label: input_mae_num[label] / max(input_mae_den[label], 1)
            for label in input_mae_num
        }
        if curves:
            width = min(len(c) for c in curves)
            out.convergence = [
                sum(c[step] for c in curves) / len(curves) for step in range(width)
            ]

        # "Does it stay": one rollout per distinct structure from native.
        seen: set[str] = set()
        moved_total, moved_count = 0.0, 0
        for case in cases:
            if case.pdb_id in seen:
                continue
            seen.add(case.pdb_id)
            result = rollout(
                rule,
                initial_state(case.native, rule),
                case.topology,
                config,
                generator=torch.Generator().manual_seed(case.mask_seed),
                record=False,
            )
            moved = (result.state.coords - case.native).abs()
            atoms = case.topology.torsion.atom_mask & (
                case.topology.supervised_mask.unsqueeze(-1)
            )
            if not bool(atoms.any()):
                continue
            worst = float(moved[atoms].max())
            out.native_drift_by_structure[case.pdb_id] = worst
            drift = max(drift, worst)
            moved_total += float(moved[atoms].sum())
            moved_count += int(atoms.sum())
        out.native_drift = drift
        out.native_drift_mean = moved_total / max(moved_count, 1)
        return out
    finally:
        if was_training:
            rule.train()
