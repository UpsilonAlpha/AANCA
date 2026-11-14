"""A chemical the rule secretes, spreading over the graph it already perceives.

The module these test replaced transported two groups along the covalent chain,
and its tests pinned exactly the properties that turned out to be the problem:
that termini are sources whether or not the rule wants them, that the log
amplitude is linear in chain distance, and that the field *ignores geometry
entirely*. Under ``birth="none"`` that converges to ``decay ** index``, which is
a positional encoding
(``docs/claims/the-chain-transported-morphogen-is-a-positional-encoding.md``).

So these tests pin the opposite, and the first two are the ones that matter: the
field is **zero until the rule secretes**, and it spreads over **spatial**
neighbours rather than bonded ones.

The third arrived a day later. The first spatial version recomputed each
residue's concentration from its neighbourhood every step, which is a relay: one
hop around a *persistent* source, and nothing carried through a region that is
not itself secreting. Ulban, 15 Nov, chose real diffusion, so the field now
retains and relaxes -- ``c <- (1 - lambda)(c + D(mean - c)) + secretion`` -- and
:func:`test_a_pulse_travels_because_the_field_has_memory` is what pins the
difference.
"""

from __future__ import annotations

import torch

from aanca.model import FIELD_NODE_EXTRA
from aanca.morphogen import (
    DEGRADATION,
    DIFFUSION_RATE,
    EMIT_CHANNELS,
    FIELD_CHANNELS,
    FIELD_FEATURE_DIM,
    diffuse,
    field_features,
    initial_field,
)


def _line_graph(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each residue perceives its two index-neighbours. ``(neighbours, mask)``.

    A deliberately chain-shaped *spatial* graph, used only where the question is
    about hops and decay rather than about which topology is in play. It is the
    graph, not ``links``: nothing in :func:`diffuse` knows about bonds.
    """
    neighbours = torch.zeros(n, 2, dtype=torch.long)
    mask = torch.zeros(n, 2, dtype=torch.bool)
    for i in range(n):
        if i > 0:
            neighbours[i, 0] = i - 1
            mask[i, 0] = True
        if i + 1 < n:
            neighbours[i, 1] = i + 1
            mask[i, 1] = True
    return neighbours, mask


def _emit(n: int, where: dict[int, float]) -> torch.Tensor:
    """``(n, EMIT_CHANNELS)`` secreting ``where[i]`` at residue ``i``."""
    emission = torch.zeros(n, EMIT_CHANNELS)
    for i, amount in where.items():
        emission[i] = amount
    return emission


def test_dim_constants_agree():
    """The node block's width and the field's cannot drift apart.

    ``FIELD_NODE_EXTRA`` is what the rule reserves and ``FIELD_FEATURE_DIM`` is
    what ``field_features`` returns; a mismatch is a shape error at the first
    forward pass, which is the wrong way to find out.
    """
    assert FIELD_CHANNELS == EMIT_CHANNELS
    assert FIELD_FEATURE_DIM == 2 * FIELD_CHANNELS
    assert FIELD_NODE_EXTRA == FIELD_FEATURE_DIM


def test_nothing_secreting_means_the_field_stays_exactly_zero():
    """The null the old module could not express.

    Its termini were clamped to one on the first hop whatever the rule did, so
    a rule with a dead emit head still handed every residue a gradient in chain
    distance. Here a rollout that never secretes carries zeros from start to
    finish, so the block costs four features and says nothing.
    """
    n = 24
    neighbours, mask = _line_graph(n)
    field = initial_field(n)
    for _ in range(200):
        field = diffuse(field, neighbours, mask)
    assert torch.equal(field, torch.zeros_like(field))


def test_the_field_is_not_a_function_of_the_residue_index():
    """The property whose failure retired the old module.

    One residue secretes, in the middle of a uniform line graph. The converged
    field is symmetric about it and says nothing about where the chain begins
    or ends -- in particular residue 0 and residue n-1 are equal, where the old
    transported groups made them 1.0 and ``decay ** (n-1)``.
    """
    n = 21
    neighbours, mask = _line_graph(n)
    field = initial_field(n)
    emission = _emit(n, {n // 2: 1.0})
    for _ in range(300):
        field = diffuse(field, neighbours, mask, emission)
    assert float(field[0, 0]) == torch.tensor(float(field[-1, 0]))
    for offset in range(1, n // 2):
        assert abs(float(field[n // 2 - offset, 0]) - float(field[n // 2 + offset, 0])) < 1e-6


def test_the_chemical_spreads_over_spatial_neighbours_not_bonded_ones():
    """The whole point of the rewrite, as a single assertion.

    A graph where residue 0 perceives residue 9 and nothing else, and residues
    1 and 2 -- 0's *sequence* neighbours -- perceive each other and not 0. One
    hop from a secreting residue 0 reaches 9 and never reaches 1, which is the
    opposite of what the chain-transported field did.
    """
    n = 10
    neighbours = torch.zeros(n, 1, dtype=torch.long)
    mask = torch.zeros(n, 1, dtype=torch.bool)
    neighbours[9, 0], mask[9, 0] = 0, True   # 9 perceives 0
    neighbours[1, 0], mask[1, 0] = 2, True   # 1 perceives 2, not 0
    neighbours[2, 0], mask[2, 0] = 1, True

    field = initial_field(n)
    field = diffuse(field, neighbours, mask, _emit(n, {0: 1.0}))
    assert float(field[0, 0]) == 1.0          # it secreted
    field = diffuse(field, neighbours, mask)  # one hop, no further secretion
    assert float(field[9, 0]) > 0.0           # spatially adjacent: reached
    assert float(field[1, 0]) == 0.0          # bonded to 0: not reached
    assert float(field[2, 0]) == 0.0


def test_a_masked_edge_does_not_conduct():
    """``edge_mask`` is honoured, so a padded or dead slot is not a sender.

    The graph pads its neighbour lists, and a padded entry points at a real
    index -- so a diffusion that ignored the mask would read a residue chosen
    by the padding.
    """
    n = 4
    neighbours = torch.zeros(n, 2, dtype=torch.long)
    neighbours[:, 0] = 1  # everyone "perceives" residue 1 in slot 0
    mask = torch.zeros(n, 2, dtype=torch.bool)
    mask[3, 0] = True     # but only residue 3 really does

    field = initial_field(n)
    field = diffuse(field, neighbours, mask, _emit(n, {1: 1.0}))
    field = diffuse(field, neighbours, mask)
    assert float(field[3, 0]) > 0.0
    assert float(field[0, 0]) == 0.0
    assert float(field[2, 0]) == 0.0


def test_one_step_moves_d_of_the_gap_and_the_source_keeps_the_rest():
    """The spreading is a fixed local operator, with a known rate, and it keeps.

    Residue 0 secretes once and stops; residue 1 perceives only 0, so a mean
    over one neighbour is that neighbour and the arithmetic is exact. Both
    halves are asserted, because the second is the whole change: residue 0 does
    not drop to zero the moment it stops secreting, it retains ``1 - D`` of
    what it held. Residue 0 perceives nobody, so its own mean is zero and it
    reads back exactly the number residue 1 receives.
    """
    n = 3
    neighbours = torch.zeros(n, 1, dtype=torch.long)
    mask = torch.zeros(n, 1, dtype=torch.bool)
    neighbours[1, 0], mask[1, 0] = 0, True
    neighbours[2, 0], mask[2, 0] = 1, True

    field = initial_field(n)
    field = diffuse(field, neighbours, mask, _emit(n, {0: 1.0}))
    assert float(field[0, 0]) == 1.0

    field = diffuse(field, neighbours, mask)
    one_step = (1.0 - DEGRADATION) * DIFFUSION_RATE
    assert abs(float(field[1, 0]) - one_step) < 1e-6
    assert abs(float(field[0, 0]) - one_step) < 1e-6  # retained, not forgotten


def test_transport_uses_the_receivers_list_and_is_not_symmetrised():
    """A one-way edge moves the chemical one way, which is a sensing rule.

    Residue 1 perceives residue 0 and residue 0 perceives nobody. With 0
    secreting steadily, 1 holds a concentration and 0 never reads anything
    back, so nothing here quietly symmetrises the graph -- worth pinning
    because a kNN graph is not symmetric and a conservative diffusion would
    have to make it so.
    """
    n = 2
    neighbours = torch.zeros(n, 1, dtype=torch.long)
    mask = torch.zeros(n, 1, dtype=torch.bool)
    neighbours[1, 0], mask[1, 0] = 0, True

    field = initial_field(n)
    emission = _emit(n, {0: 0.5})
    for _ in range(200):
        field = diffuse(field, neighbours, mask, emission)

    # And the comparison that makes it an asymmetry rather than an inequality:
    # a lone residue with no neighbours at all, secreting the same amount,
    # arrives at the same concentration residue 0 does. So residue 0 read
    # nothing back from residue 1 -- a symmetrised transport would have left it
    # higher.
    lone_n, lone_e = torch.zeros(1, 1, dtype=torch.long), _emit(1, {0: 0.5})
    lone_mask = torch.zeros(1, 1, dtype=torch.bool)
    alone = initial_field(1)
    for _ in range(200):
        alone = diffuse(alone, lone_n, lone_mask, lone_e)
    assert float(field[1, 0]) > 0.5
    assert abs(float(field[0, 0]) - float(alone[0, 0])) < 1e-6


def test_a_pulse_travels_because_the_field_has_memory():
    """The property Ulban chose on 15 Nov, and the reason for the second term.

    One residue secretes once, at step 0, and nothing secretes again. Twelve
    hops away the concentration is nonzero by step 40 -- a front propagating
    through residues that are not themselves secreting, which is the thing a
    relay cannot do: recomputing each residue's concentration from its
    neighbourhood gives one hop and then silence (``git show 247d068`` for that
    version, and the claim node for why it went).

    Degradation is asserted in the same test, because a travelling signal that
    is not bounded is the other failure: the total falls at every step and the
    pulse is gone after 400 of them.

    That total is a line-graph property and is stated as one. Transport is a
    gather, so a residue read by fifteen neighbours is counted fifteen times
    and the *sum* is not conserved -- on 1UBQ's real k=16 graph a single pulse's
    total rises from 1.116 to 1.124 before it falls. What is bounded whatever
    the degree is each residue's own concentration, because the relaxation is a
    convex combination; and the sum is not a quantity anything reads.
    """
    n = 16
    neighbours, mask = _line_graph(n)
    field = initial_field(n)
    field = diffuse(field, neighbours, mask, _emit(n, {0: 1.0}))

    totals = [float(field[:, 0].sum())]
    reach = []
    for step in range(1, 401):
        field = diffuse(field, neighbours, mask)
        totals.append(float(field[:, 0].sum()))
        if step in (1, 40):
            reach.append(float(field[12, 0]))

    assert reach[0] == 0.0        # one step: the front has not got there yet
    assert reach[1] > 1e-5        # forty: it has, with nothing secreting since
    assert all(b < a for a, b in zip(totals, totals[1:]))
    assert totals[-1] < 1e-3


def test_a_persistent_source_reaches_a_gradient_not_a_saturated_graph():
    """What degradation buys: a steady state that still carries information.

    Diffusion alone conserves, so a source that keeps secreting into a finite
    graph would raise every residue to the same concentration and the channel
    would say nothing. With ``DEGRADATION`` the steady state falls off
    geometrically, at the rate the module's own docstring predicts from ``D``
    and ``lambda``: on a chain ``1 / (1 - lambda) = (1 - D) + D(1/r + r)/2``,
    which is ``r = 0.519494`` at the current constants against 0.51949
    measured, constant to five figures over residues 4 to 14.

    Forty residues, not sixteen, because the prediction is for a half-infinite
    chain and the far end reflects; the ratio is checked in the middle for the
    same reason. The assertion is written against the constants rather than
    against 0.5195, so retuning ``DEGRADATION`` moves the prediction with it
    and the test keeps checking the physics instead of a number.
    """
    n = 40
    neighbours, mask = _line_graph(n)
    field = initial_field(n)
    emission = _emit(n, {0: 1.0})
    for _ in range(4000):
        field = diffuse(field, neighbours, mask, emission)

    column = [float(field[i, 0]) for i in range(n)]
    assert abs(column[0] - 1.0) < 1e-6
    assert all(b < a for a, b in zip(column, column[1:]))  # a gradient
    assert column[-1] < 0.05                               # a finite reach

    gap = (1.0 / (1.0 - DEGRADATION)) - (1.0 - DIFFUSION_RATE)
    root = gap * 2.0 / DIFFUSION_RATE
    predicted = (root - (root**2 - 4.0) ** 0.5) / 2.0
    for i in range(4, 13):
        assert abs(column[i + 1] / column[i] - predicted) < 0.001


def test_emission_carries_gradient_to_the_residues_that_read_it():
    """The one route by which a loss on geometry trains the secretion head.

    A loss on residue 2's concentration, two hops from residue 0, has a nonzero
    derivative with respect to what residue 0 secreted.
    """
    n = 3
    neighbours, mask = _line_graph(n)
    emission = torch.zeros(n, EMIT_CHANNELS, requires_grad=True)
    with torch.no_grad():
        emission[0] = 0.5

    field = initial_field(n)
    field = diffuse(field, neighbours, mask, emission)
    for _ in range(3):
        field = diffuse(field, neighbours, mask, emission)
    field[2, 0].backward()
    assert emission.grad is not None
    assert float(emission.grad[0, 0]) > 0.0


def test_dormant_slots_are_zeroed_when_a_mask_is_passed():
    """``alive`` still zeroes dormant slots, and ``None`` means all alive.

    The mask is no longer load-bearing -- the field has no sources to place --
    but a dormant slot must not carry a concentration something could read.
    """
    n = 6
    neighbours, mask = _line_graph(n)
    alive = torch.ones(n, dtype=torch.bool)
    alive[4:] = False

    field = initial_field(n)
    emission = _emit(n, {0: 1.0, 5: 1.0})
    for _ in range(10):
        field = diffuse(field, neighbours, mask, emission, alive)
    assert float(field[0, 0]) > 0.0
    assert float(field[4, 0]) == 0.0
    assert float(field[5, 0]) == 0.0


def test_features_are_finite_at_zero_and_at_saturation():
    """``field_features`` never returns minus infinity.

    An unreached residue is the common case -- the field is zero until
    something secretes -- so the log needs its floor, and a saturated channel
    has to stay in range.
    """
    field = torch.zeros(3, FIELD_CHANNELS)
    field[1] = 1.0
    field[2] = 0.5
    features = field_features(field)
    assert features.shape == (3, FIELD_FEATURE_DIM)
    assert torch.isfinite(features).all()
    assert float(features[0, FIELD_CHANNELS]) < 0.0
    assert abs(float(features[1, FIELD_CHANNELS])) < 1e-6


def test_the_field_shape_and_the_emission_shape_are_checked():
    """A wrong width is a loud error rather than a broadcast.

    ``(R, k)`` against ``(R, C)`` broadcasts in several wrong ways, and the
    quiet ones produce a field that trains.
    """
    import pytest

    n = 4
    neighbours, mask = _line_graph(n)
    with pytest.raises(ValueError, match="field must be"):
        diffuse(torch.zeros(n, FIELD_CHANNELS + 1), neighbours, mask)
    with pytest.raises(ValueError, match="neighbours and edge_mask"):
        diffuse(initial_field(n), neighbours, mask[:-1])
    with pytest.raises(ValueError, match="emission must be"):
        diffuse(initial_field(n), neighbours, mask, torch.zeros(n, 1))
