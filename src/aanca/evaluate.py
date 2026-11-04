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
from .metrics import (
    BenchmarkRow,
    Convention,
    benchmark_table,
    cb_neighbour_counts,
    classify_burial,
)
from .model import AANCARule
from .identity import (
    identity_loss,
    library_index,
    type_index,
    type_loss,
    type_marginal,
)
from .rollout import RolloutConfig, initial_identity, initial_state, rollout
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
    #: Fraction of positions whose committed label changed, at each rollout
    #: step, averaged over cases. Empty unless the identity head is on.
    #:
    #: Reported next to ``convergence`` because neither is readable without the
    #: other once a label can flip: a plateau in ``|delta|`` with *high* churn is
    #: the labels refusing to settle and the geometry chasing them, which is the
    #: ripple failing to converge; the same plateau with *low* churn is the
    #: geometry oscillating on its own, which is a different fault. v8c_if
    #: plateaued at 21 deg a step and could not be told apart, because nothing
    #: called ``identity.label_churn``.
    churn: list[float] = field(default_factory=list)
    #: Held-out identity numbers, present only when the rule has the head.
    #: ``identity_nll`` is over the joint 54-entry library, whose uninformative
    #: floor is ln 54 = 3.99; ``type_nll`` marginalises over the chi1 wells and
    #: so floors at ln 20 = 3.00; ``recovery`` is the fraction of supervised
    #: positions whose most likely *type* is the native one, which is the
    #: quantity ProteinMPNN (~58 %) and PottsMPNN (~44 %) report.
    identity_nll: float | None = None
    type_nll: float | None = None
    recovery: float | None = None
    #: Recovery split by burial on ``aanca.metrics``' convention, the same one
    #: the packing columns use. Not a partition: ``intermediate`` is neither,
    #: so these two do not average back to ``recovery``.
    recovery_core: float | None = None
    recovery_surface: float | None = None
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
        if self.churn:
            trace = "  ".join(f"{v:.2f}" for v in self.churn)
            lines.append(f"label churn per step (must decay to 0): {trace}")
        if self.identity_nll is not None:
            lines.append(
                f"identity: joint NLL {self.identity_nll:.3f} (floor ln 54 = "
                f"3.99), type NLL {self.type_nll:.3f} (floor ln 20 = 3.00), "
                f"sequence recovery {self.recovery:.1%}"
            )
            if self.recovery_core is not None or self.recovery_surface is not None:
                core = (
                    f"{self.recovery_core:.1%}"
                    if self.recovery_core is not None
                    else "n/a"
                )
                surface = (
                    f"{self.recovery_surface:.1%}"
                    if self.recovery_surface is not None
                    else "n/a"
                )
                lines.append(
                    f"  recovery by burial: core {core}, surface {surface} "
                    f"(intermediate is neither, so these do not average back)"
                )
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
        churn_curves: list[list[float]] = []
        input_mae_num: dict[str, float] = {}
        input_mae_den: dict[str, int] = {}
        drift = 0.0
        identity_nll_total = 0.0
        type_nll_total = 0.0
        recovered_num = 0.0
        recovered_den = 0.0
        band_num: dict[str, float] = {}
        band_den: dict[str, float] = {}
        identity_cases = 0

        for case in cases:
            topo, torsion = case.topology, case.topology.torsion
            # **Carry the labels when the rule has an identity head.** Until
            # v8c_if finished there was no held-out identity number of any
            # kind: this called ``initial_state`` without an identity, so the
            # eval never exercised the head, and the only evidence the head had
            # learned was the ``id`` column in the progress line -- which is a
            # running mean over *training* updates. With 400 structures seen
            # about fifty times each under different crops, "this backbone is
            # structure X, whose residue 47 is LEU" is a memorisable fact, so a
            # training-set NLL is exactly the number that cannot settle the
            # question it was being used to settle.
            start_identity = (
                initial_identity(
                    topo, case.native, config.identity_start,
                    torch.Generator().manual_seed(case.mask_seed),
                )
                if getattr(rule, "identity", False)
                else None
            )
            result = rollout(
                rule,
                initial_state(case.damaged, rule, identity=start_identity),
                topo,
                config,
                generator=torch.Generator().manual_seed(case.mask_seed),
            )
            if start_identity is not None and result.state.identity_logits is not None:
                supervised = topo.supervised_mask
                chi1 = torch.rad2deg(measure_chis(case.native, torsion)[:, 0])
                target = library_index(torsion.resnames, chi1)
                identity_nll_total += float(
                    identity_loss(result.state.identity_logits, target, supervised)
                )
                type_nll_total += float(
                    type_loss(result.state.identity_logits, target, supervised)
                )
                predicted_type = type_marginal(
                    torch.softmax(result.state.identity_logits, dim=-1)
                ).argmax(-1)
                right = (predicted_type == type_index(target)) & supervised
                recovered_num += float(right.sum())
                recovered_den += float(supervised.sum())
                # **Recovery split by burial, using the same convention the
                # packing columns already use** (``aanca.metrics``: core at 20
                # or more CB neighbours within 10 A, surface at 15 or fewer).
                # Sequence recovery is not uniform over a protein -- a buried
                # position is pinned by the size of the cavity it has to fill,
                # while an exposed polar position is satisfied by several
                # residues at once -- so a single pooled figure averages two
                # different problems and hides which one the rule is bad at.
                # ProteinMPNN and PottsMPNN both report the split for that
                # reason, so this is also what makes the comparison like for
                # like rather than pooled-against-stratified.
                bands = classify_burial(cb_neighbour_counts(case.native, topo))
                for name in ("core", "surface"):
                    in_band = supervised & bands[name]
                    band_num[name] = band_num.get(name, 0.0) + float(
                        (right & in_band).sum()
                    )
                    band_den[name] = band_den.get(name, 0.0) + float(in_band.sum())
                identity_cases += 1
            predicted = result.state.coords
            label = f"kappa {case.concentration:g}"
            by_level.setdefault(label, []).append((predicted, case.native, topo))
            curves.append([torch.rad2deg(torch.tensor(v)).item()
                           for v in result.mean_abs_delta])
            if result.label_churn:
                churn_curves.append(list(result.label_churn))

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
        if identity_cases:
            out.identity_nll = identity_nll_total / identity_cases
            out.type_nll = type_nll_total / identity_cases
            out.recovery = recovered_num / max(recovered_den, 1.0)
            out.recovery_core = (
                band_num.get("core", 0.0) / band_den["core"]
                if band_den.get("core")
                else None
            )
            out.recovery_surface = (
                band_num.get("surface", 0.0) / band_den["surface"]
                if band_den.get("surface")
                else None
            )
        if churn_curves:
            width = min(len(c) for c in churn_curves)
            out.churn = [
                sum(c[step] for c in churn_curves) / len(churn_curves)
                for step in range(width)
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
