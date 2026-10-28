"""Unrolling the rule, and the training objective built on top of it.

This is where the NCA machinery lives, and where the fixed-point objective is
actually expressed. Four pieces, each of which `docs/PLAN_V1.md` §3 argues for:

**Asynchronous updates.** Each residue updates with probability ~0.5 per step.
This does *not* cost the fixed point: the no-op term forces every residue to
output zero at the native structure, and any subset of zeros is still zero, so
the mask randomises the path and not the attractor. It also removes the shared
clock -- a synchronous update is a weak global channel, since every cell
implicitly knows it is step *t* at the same instant -- which strengthens the
locality claim rather than merely helping convergence.

**Loss at a random late step, not the last.** This is the mechanism that
produces an attractor rather than a schedule. A rule supervised only at step N
learns a trajectory of length N; a rule supervised at an unpredictable late step
has to be correct *and stay* correct.

**Truncated BPTT.** Measured, not optional: at ~124 MB of retained activations
per step for a 300-residue target, a 30-step rollout is 3.7 GB per sample. The
graph is detached every ``truncate`` steps, so memory is bounded by the window
rather than the rollout length.

**A sample pool.** Final states are persisted and used as future starting
points, so the rule is trained from partially-converged states it produced
itself and has to stay stable rather than merely arrive once.

Diagnostics come out of the rollout for free: mean ``|mu|`` per step is the
tier-2 convergence curve.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from . import residue_constants as rc
from .losses import chi_nll, harmonic_restraint, masked_mean, noop_nll
from .model import AANCARule
from .torch_geometry import (
    TorsionTopology,
    apply_internal_deltas,
    measure_angles,
    measure_bonds,
    measure_chis,
    wrap_to_pi,
)
from .torch_graph import FeatureTopology, build_graph


@dataclass(frozen=True)
class RolloutConfig:
    steps: int = 30
    update_probability: float = 0.5
    truncate: int = 12
    k: int = 64
    cutoff: float = 10.0
    feature_dtype: torch.dtype = torch.float32
    #: Fraction of the rollout the loss step is drawn from, measured from the
    #: end. 0.5 means "the back half", which is what the plan specifies.
    loss_window: float = 0.5
    bond_stiffness: float = 100.0
    angle_stiffness: float = 10.0


@dataclass
class RolloutState:
    """Everything carried between steps. ``coords`` is float64, hidden float32."""

    coords: Tensor
    hidden: Tensor
    step: int = 0

    def detached(self) -> "RolloutState":
        return RolloutState(
            coords=self.coords.detach(), hidden=self.hidden.detach(), step=self.step
        )


@dataclass
class RolloutResult:
    state: RolloutState
    #: Mean |mu| over real chis at each step, in radians. The tier-2 convergence
    #: curve: does it decay, or oscillate as some Graph NCA rules do?
    mean_abs_delta: list[float] = field(default_factory=list)
    #: Mean kappa at each step, as a confidence trace.
    mean_kappa: list[float] = field(default_factory=list)
    #: ``(mu, log_kappa, coords_seen)`` at the supervised step, if one was asked
    #: for. ``coords_seen`` is the rule's *input* at that step, which is what the
    #: target delta has to be measured from -- scoring against the final
    #: coordinates would ask the rule for a delta it was never in a position to
    #: predict.
    supervised: tuple[Tensor, Tensor, Tensor] | None = None
    supervised_step: int | None = None


def initial_state(
    coords: Tensor, rule: AANCARule, dtype: torch.dtype = torch.float32
) -> RolloutState:
    return RolloutState(
        coords=coords,
        hidden=rule.initial_hidden(coords.shape[0], coords.device, dtype),
        step=0,
    )


def update_mask(
    n_residues: int,
    probability: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Bernoulli mask over residues. ``probability >= 1`` gives a synchronous step."""
    if probability >= 1.0:
        return torch.ones(n_residues, dtype=torch.bool, device=device)
    # Drawn on the CPU and moved, for the same reason the damage is: the mask
    # sequence has to be a property of the seed, not of the device.
    draws = torch.rand(n_residues, generator=generator)
    return (draws < probability).to(device)


def step_once(
    rule: AANCARule,
    state: RolloutState,
    topo: FeatureTopology,
    config: RolloutConfig,
    mask: Tensor,
) -> tuple[RolloutState, Tensor, Tensor]:
    """One application of the rule. Returns ``(new_state, mu, log_kappa)``.

    The mask gates the hidden state as well as the coordinates. Gating only the
    coordinates would let a frozen residue still advance its own memory, which
    is not what asynchrony means -- the cell either ticks or it does not.
    """
    graph = build_graph(
        state.coords.to(config.feature_dtype),
        topo,
        k=config.k,
        cutoff=config.cutoff,
    )
    node = graph.node_features.to(torch.float32)

    mu, log_kappa, new_hidden = rule(
        node,
        state.hidden,
        graph.edge_features.to(torch.float32),
        graph.neighbours,
        graph.edge_mask,
        graph.envelope.to(torch.float32),
    )

    gate = mask.unsqueeze(-1)
    hidden = torch.where(gate, new_hidden, state.hidden)

    chi = torch.where(gate, mu, torch.zeros_like(mu)).to(state.coords.dtype)
    bond, angle = rule.internal_deltas(node, hidden)
    if bond is not None:
        bond = torch.where(gate, bond, torch.zeros_like(bond)).to(state.coords.dtype)
        angle = torch.where(gate, angle, torch.zeros_like(angle)).to(state.coords.dtype)

    coords = apply_internal_deltas(
        state.coords, topo.torsion, chi=chi, angle=angle, bond=bond
    )
    return RolloutState(coords=coords, hidden=hidden, step=state.step + 1), mu, log_kappa


def rollout(
    rule: AANCARule,
    state: RolloutState,
    topo: FeatureTopology,
    config: RolloutConfig,
    generator: torch.Generator | None = None,
    supervise_step: int | None = None,
    record: bool = True,
) -> RolloutResult:
    """Unroll ``config.steps`` times, detaching every ``config.truncate`` steps.

    ``supervise_step`` captures ``(mu, log_kappa)`` at that step for the loss.
    The graph is never detached at or after that step, so the captured outputs
    keep a gradient path back at least to the preceding truncation boundary.
    """
    result = RolloutResult(state=state)
    chi_mask = topo.torsion.chi_mask

    for step in range(config.steps):
        mask = update_mask(
            topo.n_residues, config.update_probability, state.coords.device, generator
        )
        seen = state.coords
        state, mu, log_kappa = step_once(rule, state, topo, config, mask)

        if record:
            with torch.no_grad():
                result.mean_abs_delta.append(
                    float(masked_mean(wrap_to_pi(mu).abs(), chi_mask))
                )
                result.mean_kappa.append(
                    float(masked_mean(torch.exp(log_kappa), chi_mask))
                )

        if supervise_step is not None and step == supervise_step:
            result.supervised = (mu, log_kappa, seen)
            result.supervised_step = step

        boundary = (step + 1) % config.truncate == 0
        past_supervision = supervise_step is None or step < supervise_step
        if boundary and past_supervision and step + 1 < config.steps:
            state = state.detached()

    result.state = state
    return result


def sample_loss_step(
    config: RolloutConfig, rng: random.Random | None = None
) -> int:
    """Draw the supervised step uniformly from the back ``loss_window`` fraction.

    Supervising the *last* step teaches a trajectory of that length. Supervising
    an unpredictable late step is what forces a fixed point.
    """
    rng = rng or random
    first = int(config.steps * (1.0 - config.loss_window))
    return rng.randrange(max(first, 0), config.steps)


@dataclass
class ObjectiveTerms:
    total: Tensor
    reconstruction: Tensor
    noop: Tensor
    restraint: Tensor
    supervised_step: int


def training_objective(
    rule: AANCARule,
    damaged: Tensor,
    native_coords: Tensor,
    topo: FeatureTopology,
    config: RolloutConfig,
    generator: torch.Generator | None = None,
    rng: random.Random | None = None,
    noop_weight: float = 1.0,
) -> tuple[ObjectiveTerms, RolloutResult]:
    """The v1 objective: reconstruct from damage, do nothing when already native.

    The reconstruction term asks, at a random late step, how surprised the rule
    was by the delta that would actually have taken it to the native torsions.
    The no-op term runs *one* step on the undamaged structure and requires a
    zero update with high confidence.
    """
    torsion: TorsionTopology = topo.torsion
    native_chis = measure_chis(native_coords, torsion)

    # Every loss term is restricted to residues with complete perception. On an
    # intact structure that is all of them; on a spatial crop the boundary shell
    # is excluded, because its native torsions depend on contacts that were
    # cropped away and asking for them is unlearnable rather than merely hard.
    # The shell stays in the graph as context for the interior.
    supervised = topo.supervised_mask
    if not bool(supervised.any()):
        raise ValueError(
            "no residue has complete perception, so there is nothing to "
            "supervise -- the crop is smaller than one interaction cutoff"
        )
    chi_loss_mask = torsion.chi_mask & supervised.unsqueeze(-1)

    supervise_step = sample_loss_step(config, rng)
    result = rollout(
        rule,
        initial_state(damaged, rule),
        topo,
        config,
        generator=generator,
        supervise_step=supervise_step,
    )
    assert result.supervised is not None
    mu, log_kappa, seen_coords = result.supervised

    # The delta that would have been correct from the coordinates the rule saw
    # at that step -- not from the final ones.
    target = wrap_to_pi(native_chis - measure_chis(seen_coords, torsion))
    reconstruction = chi_nll(
        mu, log_kappa, target, chi_loss_mask, torsion.periodic
    )

    native_state = initial_state(native_coords, rule)
    native_mask = torch.ones(
        topo.n_residues, dtype=torch.bool, device=native_coords.device
    )
    _after, native_mu, native_log_kappa = step_once(
        rule, native_state, topo, config, native_mask
    )
    noop = noop_nll(
        native_mu, native_log_kappa, chi_loss_mask, torsion.periodic
    )

    restraint = torch.zeros((), dtype=reconstruction.dtype, device=mu.device)
    if rule.bond_geometry:
        restraint = harmonic_restraint(
            measure_bonds(result.state.coords, torsion),
            measure_bonds(native_coords, torsion),
            torsion.bond_mask & supervised.unsqueeze(-1),
            config.bond_stiffness,
        ) + harmonic_restraint(
            measure_angles(result.state.coords, torsion),
            measure_angles(native_coords, torsion),
            torsion.angle_mask & supervised.unsqueeze(-1),
            config.angle_stiffness,
        )

    total = reconstruction + noop_weight * noop + restraint
    return (
        ObjectiveTerms(
            total=total,
            reconstruction=reconstruction.detach(),
            noop=noop.detach(),
            restraint=restraint.detach(),
            supervised_step=supervise_step,
        ),
        result,
    )


class SamplePool:
    """Growing-NCA sample pool: persist intermediate states as future seeds.

    Training only from freshly damaged structures teaches the rule to *arrive*.
    Re-seeding from states it produced itself, which are partially converged and
    somewhat off-distribution, is what teaches it to *stay* -- and is the cheap
    half of the fixed-point objective.
    """

    def __init__(self, capacity: int = 256, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        self._entries: list[tuple[int, RolloutState]] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._entries)

    def insert(self, structure_id: int, state: RolloutState) -> None:
        """Store a detached copy, evicting at random once full."""
        entry = (structure_id, state.detached())
        if len(self._entries) < self.capacity:
            self._entries.append(entry)
        else:
            self._entries[self._rng.randrange(self.capacity)] = entry

    def sample(self, structure_id: int) -> RolloutState | None:
        """A stored state for this structure, or ``None`` to start from damage.

        Keyed by structure because a pooled state is only a valid seed for the
        structure it came from -- the topology and the native target differ.
        """
        matching = [s for sid, s in self._entries if sid == structure_id]
        if not matching:
            return None
        return self._rng.choice(matching)

    def clear(self) -> None:
        self._entries.clear()

    def state_dict(self) -> dict:
        """Everything needed to resume, including this pool's own RNG.

        The pool is part of the training state, not a cache: a resumed run that
        started with an empty pool would train for a while on fresh damage
        only, which is exactly the half of the objective the pool exists to
        supply. Tensors go in as they are and are moved on load.
        """
        return {
            "capacity": self.capacity,
            "rng": self._rng.getstate(),
            "entries": [
                {
                    "structure_id": sid,
                    "coords": state.coords,
                    "hidden": state.hidden,
                    "step": state.step,
                }
                for sid, state in self._entries
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"pool capacity changed: checkpoint has {state['capacity']}, "
                f"this pool has {self.capacity}"
            )
        self._rng.setstate(_tuple_states(state["rng"]))
        self._entries = [
            (
                int(entry["structure_id"]),
                RolloutState(
                    coords=entry["coords"],
                    hidden=entry["hidden"],
                    step=int(entry["step"]),
                ),
            )
            for entry in state["entries"]
        ]


def _tuple_states(state):
    """``random.setstate`` demands tuples; a round trip through torch.save
    leaves the inner state as a list."""
    if isinstance(state, (list, tuple)):
        return tuple(_tuple_states(item) for item in state)
    return state
