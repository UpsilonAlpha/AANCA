"""The density block: the neighbour count the aggregation computes and discards.

``model.aggregate_norm`` is a ``LayerNorm`` over the summed messages, and a
LayerNorm's output norm is ``sqrt(channels)`` whatever went in -- so the degree,
which lives in the sum's magnitude, is removed exactly rather than attenuated.
Since burial *is* neighbour density, that deleted the one global-sounding
quantity that is actually local. These tests pin the block that hands it back.
"""

from __future__ import annotations

import torch

from aanca.model import (
    DENSITY_NODE_EXTRA,
    GROWTH_NODE_EXTRA,
    NODE_FEATURE_DIM,
    AANCARule,
)
from aanca.torch_graph import DENSITY_DIM, density_features


def test_dim_constants_agree():
    """The same guard GROWTH_NODE_EXTRA has, for the same reason.

    Two modules have to agree on a width, and disagreement would surface as a
    shape error in training rather than next to the definitions.
    """
    assert DENSITY_DIM == DENSITY_NODE_EXTRA


def test_node_dim_composes_with_growth():
    assert AANCARule(density=True).node_dim == NODE_FEATURE_DIM + DENSITY_NODE_EXTRA
    assert AANCARule(growth=True, density=True).node_dim == (
        NODE_FEATURE_DIM + GROWTH_NODE_EXTRA + DENSITY_NODE_EXTRA
    )
    # Off by default, so every existing checkpoint still loads.
    assert AANCARule().node_dim == NODE_FEATURE_DIM
    assert AANCARule(growth=True).node_dim == NODE_FEATURE_DIM + GROWTH_NODE_EXTRA


def test_density_is_monotone_in_degree():
    """More neighbours must read as more crowded.

    The scalar channel carries the ordering; without it the expansion alone
    would represent the count without representing "more".
    """
    previous = -1.0
    for degree in (2, 8, 16, 24, 32):
        envelope = torch.zeros(1, 40)
        envelope[0, :degree] = 0.8
        mask = torch.zeros(1, 40, dtype=torch.bool)
        mask[0, :degree] = True
        value = float(density_features(envelope, mask)[0, 0])
        assert value > previous, (degree, value, previous)
        previous = value


def test_masked_edges_do_not_count():
    """A dormant or out-of-cutoff neighbour occupies a slot and is not a neighbour.

    This is the same distinction ``step_once`` makes when it ands the edge mask
    with ``alive``: a slot in the neighbour list is not evidence of a contact.
    """
    envelope = torch.full((1, 10), 0.5)
    all_on = torch.ones(1, 10, dtype=torch.bool)
    half = torch.zeros(1, 10, dtype=torch.bool)
    half[0, :5] = True
    assert float(density_features(envelope, half)[0, 0]) < float(
        density_features(envelope, all_on)[0, 0]
    )
    none = torch.zeros(1, 10, dtype=torch.bool)
    assert float(density_features(envelope, none)[0, 0]) == 0.0


def test_block_is_finite_at_zero_and_saturation():
    """No NaN at an isolated residue, and no blow-up at an implausible degree.

    An isolated residue is real -- the first seed of a growth rollout has no
    live neighbours at all -- so zero has to be an ordinary input rather than an
    edge case.
    """
    for degree in (0, 1, 200):
        envelope = torch.full((1, 256), 0.9)
        mask = torch.zeros(1, 256, dtype=torch.bool)
        mask[0, :degree] = True
        block = density_features(envelope, mask)
        assert block.shape == (1, DENSITY_DIM)
        assert bool(torch.isfinite(block).all())


def test_gradient_reaches_the_envelope():
    """The block must not be a dead end.

    ``weight = envelope * edge_mask`` is the route by which gradient reaches
    the geometry in the aggregation, and the density block is built from the
    same envelope -- so it should carry gradient too, or it is an input the
    rule can read but never influence.
    """
    envelope = torch.full((1, 12), 0.5, requires_grad=True)
    mask = torch.ones(1, 12, dtype=torch.bool)
    density_features(envelope, mask).sum().backward()
    assert envelope.grad is not None
    assert float(envelope.grad.abs().sum()) > 0.0
