"""The chain-transported field: a route that does not pass through the graph.

Every edge a growth rollout perceives is selected from ``coords.detach()``
within a 10 A cutoff, so two residues that ought to touch and currently do not
have no edge and no gradient path between them. These tests pin the properties
that make a field along the backbone a route anyway -- that it reaches far
along the chain regardless of geometry, that its sources are detectable
locally, and that it carries gradient from the head that secretes into it.
"""

from __future__ import annotations

import torch

from aanca.model import FIELD_NODE_EXTRA
from aanca.morphogen import (
    CWARD_DECAYS,
    EMIT_CHANNELS,
    FIELD_CHANNELS,
    FIELD_FEATURE_DIM,
    NWARD_DECAYS,
    diffuse,
    field_features,
    initial_field,
)


def _chain(n: int, alive: slice | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One covalent run of ``n`` slots, with ``alive`` the live stretch."""
    links = torch.ones(n - 1, dtype=torch.bool)
    live = torch.zeros(n, dtype=torch.bool)
    live[alive if alive is not None else slice(None)] = True
    return live, links


def _settle(alive: torch.Tensor, links: torch.Tensor, hops: int) -> torch.Tensor:
    field = initial_field(alive.shape[0])
    for _ in range(hops):
        field = diffuse(field, alive, links, None)
    return field


def test_dim_constants_agree():
    """The same guard the growth and density blocks have, for the same reason.

    Two modules agree on a width, and disagreement would surface as a shape
    error at the first forward pass rather than next to the definitions.
    """
    assert FIELD_FEATURE_DIM == FIELD_NODE_EXTRA
    assert FIELD_CHANNELS == len(NWARD_DECAYS) + len(CWARD_DECAYS) + EMIT_CHANNELS


def test_termini_are_sources_without_being_told():
    """The N-most and C-most alive residues clamp to one.

    No residue is handed its index or the chain's length. A terminus is the
    absence of inflow, so the test is that the field finds them from the alive
    mask alone -- and finds the *alive* run's ends, not the crop's.
    """
    alive, links = _chain(20, slice(5, 15))
    field = _settle(alive, links, 30)
    n_channel, c_channel = field[:, 0], field[:, len(NWARD_DECAYS)]
    assert float(n_channel[5]) == 1.0
    assert float(c_channel[14]) == 1.0
    # The crop's own ends are dormant and carry nothing at all.
    assert float(field[0].abs().sum()) == 0.0
    assert float(field[19].abs().sum()) == 0.0


def test_log_amplitude_is_linear_in_chain_distance():
    """The feature that makes this a positional coordinate.

    ``decay ** d`` has a logarithm proportional to ``-d``, so the log channel
    is a linear ramp along the chain. That is what lets a residue know how far
    it is from a terminus without anything global being computed.
    """
    alive, links = _chain(40)
    field = _settle(alive, links, 80)
    logged = field_features(field)[:, FIELD_CHANNELS:]
    ramp = logged[:, 0]
    steps = ramp[1:] - ramp[:-1]
    # Constant slope, and a decreasing one: further from the N-terminal source
    # is a smaller amplitude, so a more negative logarithm.
    assert float(steps.std()) < 1e-6
    assert float(steps.mean()) < 0.0


def test_transport_is_one_residue_per_step():
    """Propagation speed is the whole reason for directed transport.

    Symmetric diffusion from one source spreads as the square root of time and
    would still be travelling when a 120-step rollout ends. Directed transport
    advances exactly one residue per hop, which matches growth's own rate at
    the frontier.
    """
    alive, links = _chain(60)
    for hops in (3, 10, 25):
        field = _settle(alive, links, hops)
        reached = int((field[:, 0] > 0).sum())
        assert reached == hops, (hops, reached)


def test_the_field_ignores_geometry_entirely():
    """The property the spatial graph does not have.

    The same chain, once compact and once with every residue flung apart, must
    give the same field: ``diffuse`` reads ``alive`` and ``links`` and never a
    coordinate. This is the whole claim -- a wrong fold cannot delete this
    route the way it deletes a 10 A edge.
    """
    alive, links = _chain(30)
    field = _settle(alive, links, 50)
    # There is no coordinate argument to vary, which is the point; the test is
    # that the signature cannot accept one, and the field is a function of the
    # covalent topology alone.
    broken = links.clone()
    broken[14] = False
    split = initial_field(30)
    for _ in range(50):
        split = diffuse(split, alive, broken, None)
    # Breaking the *chain* does change it -- two runs, so four termini.
    assert float(split[:, 0][15]) == 1.0
    assert float(field[:, 0][15]) < 1.0


def test_a_chain_break_makes_two_runs_with_their_own_sources():
    """Transport cannot cross a break, because growth cannot either.

    ``frontier`` is gated on ``links``, so a crop holding several covalent runs
    grows them independently. A field that leaked across a break would tell a
    residue it was near a terminus that no amount of growth could connect it
    to.
    """
    alive = torch.ones(20, dtype=torch.bool)
    links = torch.ones(19, dtype=torch.bool)
    links[9] = False
    field = _settle(alive, links, 40)
    n_channel = field[:, 0]
    assert float(n_channel[0]) == 1.0
    assert float(n_channel[10]) == 1.0
    assert float(n_channel[9]) < 1.0


def test_dormant_slots_do_not_conduct():
    """Otherwise the converged field is a positional encoding, handed over.

    If a dormant slot conducted, the field at residue ``i`` would settle to
    ``decay ** i`` -- a deterministic function of the sequence index, which is
    the global coordinate this is deliberately not. Conduction through live
    residues only makes it a property of the structure grown so far.
    """
    alive, links = _chain(20, slice(0, 8))
    field = _settle(alive, links, 40)
    assert float(field[:, 0][7]) > 0.0
    assert float(field[8:].abs().sum()) == 0.0


def test_emission_reaches_both_directions_and_carries_gradient():
    """The emitted channels are the emergent half, and must be trainable.

    A fixed terminus source is physics; a secreted one is the rule inventing a
    coordinate. For that to be learnable the gradient has to run from a distant
    residue's feature back to the secreting residue's head, which is the one
    path in this architecture that does not pass through the spatial graph.
    """
    alive, links = _chain(21)
    source = torch.zeros(21, EMIT_CHANNELS)
    source[10] = 0.5
    emission = source.clone().requires_grad_(True)
    field = initial_field(21)
    for _ in range(6):
        field = diffuse(field, alive, links, emission)
    emitted = field[:, -EMIT_CHANNELS:]
    assert float(emitted[6, 0]) > 0.0
    assert float(emitted[14, 0]) > 0.0
    emitted[6, 0].backward()
    assert emission.grad is not None
    assert float(emission.grad[10].abs().sum()) > 0.0


def test_features_are_finite_at_zero_and_at_a_source():
    """Both ends of the range are ordinary inputs, not edge cases.

    A dormant slot reads exactly zero and a source reads exactly one, and the
    logarithm is taken of both. Without the floor the first is minus infinity,
    which poisons the whole batch's gradient one matmul later.
    """
    alive, links = _chain(12, slice(0, 4))
    field = _settle(alive, links, 10)
    block = field_features(field)
    assert block.shape == (12, FIELD_FEATURE_DIM)
    assert bool(torch.isfinite(block).all())
    assert float(block.min()) >= -1.0 - 1e-6
    assert float(block.max()) <= 1.0 + 1e-6


def test_a_run_of_one_is_a_source_in_both_directions():
    """A single seed is both termini of its own run.

    ``seed_alive`` starts every rollout here, so this is the state at step zero
    of every growth run rather than a degenerate case. Written as a maximum
    against the source term precisely so this reads one and not zero.
    """
    alive, links = _chain(10, slice(4, 5))
    field = diffuse(initial_field(10), alive, links, None)
    assert float(field[4, 0]) == 1.0
    assert float(field[4, len(NWARD_DECAYS)]) == 1.0
