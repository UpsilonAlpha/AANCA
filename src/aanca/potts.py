"""A Potts energy over amino-acid identity, and the chains that sample it.

Why this module exists, from ``HANDOFF.md`` ("Inverse folding: the design
settled on 2026-11-02"): an update map ``x <- f(x)`` descends a scalar only if
its Jacobian is symmetric, which a generic net's is not. The AANCA rule reaches
a fixed point, but that fixed point is the minimum of nothing -- there is no
function it is stationary with respect to, and the oscillation risk is the
observable symptom. So an explicit, evaluable energy is needed next to the
network, for two jobs: as the target distribution ``pi(s) ~ exp(-E(s)/kT)`` of a
Metropolis-Hastings chain whose *proposal* is the network's conditional, and as
the scalar whose difference is point-mutant ddG.

A Potts model was chosen over an MD forcefield and over a learned scalar energy
for one structural reason, and that reason is the shape of this file: **a
single-position mutation changes only the terms involving that position, so
``delta_energy`` is O(neighbours) rather than O(protein)**. Both MH acceptance
and ddG are differences under one-position moves, so the whole cost model of
sampling rests on that locality. :func:`delta_energy` is the function this
module exists for; :func:`energy` is mostly the reference the fast path is
checked against.

Two consequences for the data layout:

* The couplings are stored **sparsely over an explicit edge list**, never as a
  dense ``(n_res, n_res, n_types, n_types)`` array. At 300 residues and 20
  types that dense form is 1.4e10 entries, and all but the contacting pairs are
  zero anyway.
* Incidence is precomputed once per model (:meth:`PottsModel.incident`), so a
  move at position *i* never scans the edge list.

Scope: this file is the energy and the samplers only. Where ``h`` and ``J``
come from -- a learned head on the encoder, or a fit -- is a separate task, so
everything here takes them as supplied tensors.

Temperature convention: ``temperature`` *is* ``kT``, in whatever units the
energy is in (Boltzmann's constant is folded in). ``temperature=0`` is allowed
and means the zero-temperature limit -- the conditional collapses onto the
argmax, which makes :func:`gibbs_sweep` iterated conditional modes and lets
:func:`anneal` quench exactly rather than merely nearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .rng import draw_multinomial

__all__ = [
    "AnnealResult",
    "PottsModel",
    "anneal",
    "conditional",
    "delta_energy",
    "edges_from_adjacency",
    "energy",
    "gibbs_sweep",
    "metropolis_ratio",
    "metropolis_sweep",
]


def edges_from_adjacency(adjacency: Tensor) -> Tensor:
    """``(n_edges, 2)`` long tensor of the upper triangle of ``adjacency``.

    Only ``i < j`` is read, so a one-sided (or asymmetric) contact matrix gives
    the same edge list as its symmetrisation. The diagonal is ignored: a
    self-coupling is not a pair term, it is a field, and it belongs in ``h``.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"adjacency must be square (n_res, n_res), got {tuple(adjacency.shape)}"
        )
    upper = torch.triu(adjacency.bool(), diagonal=1)
    return upper.nonzero(as_tuple=False).to(torch.long)


@dataclass(eq=False)
class PottsModel:
    """``E(s) = sum_i h_i(s_i) + sum_{i<j} J_ij(s_i, s_j)``.

    ``h``
        ``(n_res, n_types)`` fields.
    ``edges``
        ``(n_edges, 2)`` long, each row ``(i, j)`` with ``i < j``. The
        orientation is part of the contract, not a convention: ``J[e]`` is read
        as ``J[e, s_i, s_j]``, so a flipped row silently transposes that
        edge's coupling.
    ``J``
        ``(n_edges, n_types, n_types)`` couplings, one matrix per edge.

    Shapes and the ``i < j`` ordering are validated on construction, because
    every error they catch is one that otherwise shows up as a wrong number
    rather than as an exception.
    """

    h: Tensor
    edges: Tensor
    J: Tensor

    #: Per-position incidence, built in ``__post_init__``: for position ``p``,
    #: ``_incident[p]`` is ``(edge_indices, neighbour_positions, p_is_first)``.
    #: Derived, so it is not a constructor argument. ``eq=False`` above because
    #: the default dataclass ``__eq__`` would compare tensors elementwise and
    #: then fail on the ambiguous truth value.
    _incident: list[tuple[Tensor, Tensor, Tensor]] = field(
        default_factory=list, repr=False, init=False
    )

    def __post_init__(self) -> None:
        if self.h.ndim != 2:
            raise ValueError(f"h must be (n_res, n_types), got {tuple(self.h.shape)}")
        n_res, n_types = self.h.shape
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(
                f"edges must be (n_edges, 2), got {tuple(self.edges.shape)}"
            )
        if self.J.shape != (self.edges.shape[0], n_types, n_types):
            raise ValueError(
                f"J must be (n_edges, n_types, n_types) = "
                f"{(self.edges.shape[0], n_types, n_types)}, "
                f"got {tuple(self.J.shape)}"
            )
        self.edges = self.edges.to(torch.long)
        if self.edges.numel():
            if int(self.edges.min()) < 0 or int(self.edges.max()) >= n_res:
                raise ValueError(
                    f"edge endpoints must be in [0, {n_res}), got "
                    f"[{int(self.edges.min())}, {int(self.edges.max())}]"
                )
            bad = (self.edges[:, 0] >= self.edges[:, 1]).nonzero(as_tuple=False)
            if bad.numel():
                row = int(bad[0, 0])
                raise ValueError(
                    "edges must satisfy i < j (J[e] is indexed J[e, s_i, s_j], "
                    f"so a flipped row transposes a coupling); row {row} is "
                    f"{tuple(int(v) for v in self.edges[row])}"
                )
        self._build_incidence()

    def _build_incidence(self) -> None:
        index = torch.arange(self.n_edges, dtype=torch.long, device=self.edges.device)
        self._incident = []
        for position in range(self.n_res):
            first = self.edges[:, 0] == position
            second = self.edges[:, 1] == position
            touches = first | second
            edge_ids = index[touches]
            is_first = first[touches]
            others = torch.where(
                is_first, self.edges[touches, 1], self.edges[touches, 0]
            )
            self._incident.append((edge_ids, others, is_first))

    @classmethod
    def from_adjacency(cls, h: Tensor, adjacency: Tensor, J: Tensor) -> "PottsModel":
        """Build from a boolean contact matrix; ``J`` is then given in the
        order :func:`edges_from_adjacency` returns (row-major upper triangle)."""
        return cls(h=h, edges=edges_from_adjacency(adjacency), J=J)

    @property
    def n_res(self) -> int:
        return int(self.h.shape[0])

    @property
    def n_types(self) -> int:
        return int(self.h.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def incident(self, position: int) -> tuple[Tensor, Tensor, Tensor]:
        """``(edge_indices, neighbour_positions, position_is_first_endpoint)``
        for the edges touching ``position``. Precomputed, so the local moves
        below never scan the edge list."""
        if not 0 <= position < self.n_res:
            raise IndexError(f"position {position} outside [0, {self.n_res})")
        return self._incident[position]

    def neighbours(self, position: int) -> Tensor:
        """Positions coupled to ``position``, in edge-list order."""
        return self.incident(position)[1]


def energy(model: PottsModel, sequence: Tensor) -> Tensor:
    """Total energy of ``sequence``, a ``(..., n_res)`` long tensor of type
    indices.

    A leading batch dimension falls out of the gathers, so it is supported;
    the return has shape ``sequence.shape[:-1]`` (a 0-d tensor for one
    sequence). This is the O(n_edges) reference. Sampling must not call it --
    that is what :func:`delta_energy` and :func:`conditional` are for.
    """
    if sequence.shape[-1] != model.n_res:
        raise ValueError(
            f"sequence must have last dimension n_res={model.n_res}, "
            f"got {tuple(sequence.shape)}"
        )
    flat = sequence.reshape(-1, model.n_res).to(torch.long)
    batch = flat.shape[0]

    fields = model.h.unsqueeze(0).expand(batch, -1, -1)
    total = fields.gather(2, flat.unsqueeze(-1)).squeeze(-1).sum(-1)

    if model.n_edges:
        left = flat[:, model.edges[:, 0]]
        right = flat[:, model.edges[:, 1]]
        edge_index = torch.arange(model.n_edges, device=flat.device)
        total = total + model.J[edge_index.unsqueeze(0), left, right].sum(-1)

    return total.reshape(sequence.shape[:-1])


def _incident_pair_terms(
    model: PottsModel, sequence: Tensor, position: int
) -> Tensor:
    """``(n_types,)``: the sum of this position's pair terms as a function of
    the type placed there, with every neighbour held at its current value.

    This is the one piece of arithmetic behind :func:`delta_energy` and
    :func:`conditional` both, which is why it is factored out -- two
    implementations of the same locality argument would be two things to keep
    right.
    """
    edge_ids, others, is_first = model.incident(position)
    if edge_ids.numel() == 0:
        return torch.zeros(model.n_types, dtype=model.h.dtype, device=model.h.device)
    neighbour_types = sequence[others]
    blocks = model.J[edge_ids]  # (deg, n_types, n_types)
    # Where this position is endpoint i, the neighbour indexes the columns and
    # the free type indexes the rows; where it is endpoint j, the other way.
    as_first = blocks.gather(
        2, neighbour_types.view(-1, 1, 1).expand(-1, model.n_types, 1)
    ).squeeze(-1)
    as_second = blocks.gather(
        1, neighbour_types.view(-1, 1, 1).expand(-1, 1, model.n_types)
    ).squeeze(1)
    return torch.where(is_first.unsqueeze(-1), as_first, as_second).sum(0)


def _effective_energy(model: PottsModel, sequence: Tensor, position: int) -> Tensor:
    """``(n_types,)`` energy as a function of the type at ``position``, up to
    the additive constant contributed by everything that does not touch it."""
    return model.h[position] + _incident_pair_terms(model, sequence, position)


def delta_energy(
    model: PottsModel, sequence: Tensor, position: int, new_type: int | Tensor
) -> Tensor:
    """``E(mutant) - E(sequence)`` for setting ``position`` to ``new_type``.

    The function the module exists for. It reads ``h[position]`` and the edges
    incident on ``position`` and nothing else, so the cost is O(degree) and not
    O(protein) -- which is what makes a Metropolis sweep and a ddG scan
    affordable. :func:`energy` recomputes the same number the slow way, and
    ``tests/test_potts.py`` asserts they agree; that test is the whole
    justification for this function existing.

    Returns a 0-d tensor. Exactly ``0`` when ``new_type`` is the current type:
    the differences cancel term by term, not approximately.
    """
    old_type = int(sequence[position])
    new = int(new_type)
    if not 0 <= new < model.n_types:
        raise ValueError(f"new_type {new} outside [0, {model.n_types})")
    if new == old_type:
        return torch.zeros((), dtype=model.h.dtype, device=model.h.device)
    effective = _effective_energy(model, sequence, position)
    return effective[new] - effective[old_type]


def conditional(
    model: PottsModel, sequence: Tensor, position: int, temperature: float
) -> Tensor:
    """The ``(n_types,)`` Gibbs conditional ``p(t) ~ exp(-E_eff(t) / kT)``.

    ``E_eff(t)`` is the energy with ``position`` set to ``t`` and every other
    position held fixed; the constant from the untouched terms drops out of the
    normalisation, so this is computed from the same incident-edge arithmetic
    as :func:`delta_energy` rather than by evaluating :func:`energy` ``n_types``
    times.

    The softmax subtracts the maximum *before* exponentiating, and it does so
    on the energies rather than on the already-divided logits: at kT = 1e-6 the
    raw ``exp(-E/kT)`` overflows to ``inf`` for every type at once, and
    ``inf - inf`` is the NaN that shifting afterwards would not prevent.

    ``temperature=0`` returns the zero-temperature limit -- uniform over the
    argmin of the energy, which is a single type unless there is an exact tie.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    effective = _effective_energy(model, sequence, position)
    shifted = effective.min() - effective  # <= 0, with 0 at the best type
    if _is_quench(temperature, model.h.dtype):
        weights = (shifted == 0).to(model.h.dtype)
    else:
        weights = torch.exp(shifted / temperature)
    return weights / weights.sum()


def _is_quench(temperature: float, dtype: torch.dtype) -> bool:
    """Whether to take the zero-temperature branch.

    Deliberately **not** ``temperature == 0``. A positive temperature smaller
    than the working dtype's smallest normal becomes exactly zero when torch
    casts the Python scalar to the tensor's dtype, so dividing by it gives 0/0
    = NaN at the best type and -inf at the rest -- a silent NaN at a
    temperature that merely *looks* positive. 1e-300 is positive in Python and
    exactly zero in float32, and that gap between 0 and ~1.2e-38 is the whole
    bug; it survived a test at 1e-6 and a test at exactly 0, because it lives
    between them.

    Treating such a temperature as a quench is also the physically right
    answer: a kT below the dtype's resolution *is* zero here, and no annealing
    schedule that reaches it wanted anything else. Comparing against
    ``finfo.tiny`` closes the hole without a magic constant, and keeps the
    division out of the subnormal range where it would be finite but enormous.
    """
    return not temperature > torch.finfo(dtype).tiny


def metropolis_ratio(
    delta_e: Tensor | float,
    q_old: Tensor | float,
    q_new: Tensor | float,
    temperature: float,
) -> Tensor:
    """``min(1, exp(-dE / kT) * q_old / q_new)``: the acceptance probability.

    ``q_old`` and ``q_new`` are the proposal probabilities of the current and
    the proposed type at the position being moved. This is the single-site
    independence-proposal form: the proposal depends on the rest of the
    sequence but not on the value currently at this position, so the reverse
    proposal density ``q(old | new)`` is just ``q_old`` and the Hastings
    correction is the plain ratio.

    One function spans the two samplers that matter here:

    * A **symmetric** proposal (uniform over types, say) has
      ``q_old == q_new``, the ratio cancels, and this reduces to the Metropolis
      criterion ``min(1, exp(-dE / kT))`` -- Rosetta's simulated annealing.
    * A proposal **equal to the exact conditional** has
      ``q_new / q_old == exp(-dE / kT)``, so the whole ratio is identically 1
      and every move is accepted -- that is Gibbs sampling.

    Everything in between -- in particular the AANCA network's conditional,
    which is neither symmetric nor exact -- is a valid chain *because* of the
    ``q`` ratio. Getting the right stationary distribution out of a wrong
    proposal is the entire point, and the test that proves it is the biased-
    proposal Boltzmann comparison in ``tests/test_potts.py``.

    At ``temperature=0`` the Boltzmann factor is a step: downhill and flat
    moves keep the ``q`` ratio, uphill moves are rejected outright.
    """
    delta = torch.as_tensor(delta_e, dtype=torch.get_default_dtype()).reshape(())
    old = torch.as_tensor(q_old, dtype=delta.dtype).reshape(())
    new = torch.as_tensor(q_new, dtype=delta.dtype).reshape(())
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if new <= 0:
        # Unreachable from a proposal that actually generated this move; a
        # zero-probability target would make the ratio infinite rather than
        # undefined, so refusing is the conservative reading.
        return torch.zeros((), dtype=delta.dtype)
    if old <= 0:
        return torch.zeros((), dtype=delta.dtype)
    if _is_quench(temperature, delta.dtype):
        if delta > 0:
            return torch.zeros((), dtype=delta.dtype)
        log_ratio = torch.log(old) - torch.log(new)
    else:
        log_ratio = -delta / temperature + torch.log(old) - torch.log(new)
    return torch.clamp(torch.exp(torch.clamp(log_ratio, max=0.0)), max=1.0)


def _sample(probabilities: Tensor, generator: torch.Generator | None) -> int:
    # See ``aanca.rng``: a bare ``torch.multinomial`` requires the generator's
    # device to match the probabilities'. Reached on CPU today, but the same
    # latent fault that killed job 31516313.
    return int(draw_multinomial(probabilities.reshape(1, -1), generator)[0])


def gibbs_sweep(
    model: PottsModel,
    sequence: Tensor,
    positions: Sequence[int] | Tensor,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int]:
    """Resample each of ``positions``, in the order given, from its conditional.

    Returns ``(sequence, n_changed)`` -- a new tensor, and the number of
    positions whose type actually differs afterwards. That count is the
    convergence diagnostic ``HANDOFF.md`` asks for in place of
    ``mean_abs_delta``: a discrete label can flip while a continuous residual
    settles, so the fraction of labels that changed is what must decay.

    The updates are **sequential**, so each conditional already sees the
    neighbours resampled earlier in the same sweep. ``positions`` is passed in
    rather than chosen here precisely so that a caller can hand it a
    conditionally independent set (``aanca.schedule``) and get a correct
    *parallel* sweep instead; this module deliberately does not import that
    module, and does not assume the set it is given is independent.
    """
    updated = sequence.clone()
    changed = 0
    for position in _as_positions(positions, model):
        old = int(updated[position])
        probabilities = conditional(model, updated, position, temperature)
        new = _sample(probabilities, generator)
        updated[position] = new
        changed += int(new != old)
    return updated, changed


def metropolis_sweep(
    model: PottsModel,
    sequence: Tensor,
    positions: Sequence[int] | Tensor,
    temperature: float,
    proposal: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int]:
    """Propose from ``proposal`` at each of ``positions`` and accept by
    :func:`metropolis_ratio`.

    ``proposal`` is ``(n_types,)``, used at every position, or
    ``(n_res, n_types)`` for a per-position proposal -- the latter is the shape
    a network head emits. It is not required to resemble the conditional; the
    Hastings ratio is what makes a bad proposal correct rather than merely
    cheap, and it is what the network's conditional will eventually be plugged
    into.

    Returns ``(sequence, n_changed)``. Note that ``n_changed`` counts changed
    types, not accepted moves: a proposal that re-proposes the current type is
    accepted with probability 1 and changes nothing, which is why the
    diagnostic is the former.
    """
    if proposal.ndim == 1:
        if proposal.shape[0] != model.n_types:
            raise ValueError(
                f"proposal must be (n_types,) = ({model.n_types},) or "
                f"(n_res, n_types); got {tuple(proposal.shape)}"
            )
        per_position = proposal.unsqueeze(0).expand(model.n_res, -1)
    elif proposal.shape == (model.n_res, model.n_types):
        per_position = proposal
    else:
        raise ValueError(
            f"proposal must be ({model.n_types},) or "
            f"{(model.n_res, model.n_types)}; got {tuple(proposal.shape)}"
        )

    updated = sequence.clone()
    changed = 0
    for position in _as_positions(positions, model):
        q = per_position[position]
        old = int(updated[position])
        new = _sample(q, generator)
        if new == old:
            continue
        delta = delta_energy(model, updated, position, new)
        accept = metropolis_ratio(delta, q[old], q[new], temperature)
        draw = torch.rand((), generator=generator, dtype=accept.dtype)
        if draw < accept:
            updated[position] = new
            changed += 1
    return updated, changed


@dataclass
class AnnealResult:
    """``sequence`` after the last sweep, and ``changed`` -- the number of
    positions that changed on each sweep, in order.

    ``changed`` is the diagnostic, not a log line. It must decay to zero; a
    plateau at a nonzero value is a limit cycle, which is the failure mode an
    asymmetric-Jacobian update map has and the reason this energy exists.
    """

    sequence: Tensor
    changed: list[int]


def anneal(
    model: PottsModel,
    sequence: Tensor,
    schedule: Iterable[float],
    generator: torch.Generator | None = None,
    positions: Sequence[int] | Tensor | None = None,
    proposal: Tensor | None = None,
) -> AnnealResult:
    """Run one sweep per temperature in ``schedule``.

    ``positions`` defaults to every position, swept in index order. With
    ``proposal`` given the sweeps are Metropolis with that proposal; without
    it they are Gibbs. Ending the schedule at ``0.0`` quenches exactly --
    :func:`conditional` takes the argmin there -- so a schedule's last entry
    being zero is the difference between "nearly a local minimum" and "a local
    minimum".
    """
    swept = (
        torch.arange(model.n_res, dtype=torch.long)
        if positions is None
        else _as_positions(positions, model)
    )
    current = sequence.clone()
    changed: list[int] = []
    for temperature in schedule:
        if proposal is None:
            current, count = gibbs_sweep(
                model, current, swept, float(temperature), generator
            )
        else:
            current, count = metropolis_sweep(
                model, current, swept, float(temperature), proposal, generator
            )
        changed.append(count)
    return AnnealResult(sequence=current, changed=changed)


def _as_positions(positions: Sequence[int] | Tensor, model: PottsModel) -> Tensor:
    index = torch.as_tensor(positions, dtype=torch.long).reshape(-1)
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= model.n_res):
        raise IndexError(
            f"positions must lie in [0, {model.n_res}), got "
            f"[{int(index.min())}, {int(index.max())}]"
        )
    return index
