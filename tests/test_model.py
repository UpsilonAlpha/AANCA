"""The rule: identity at init, strictly one hop, bounded confidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aanca import residue_constants as rc
from aanca.graph import build_graph
from aanca.model import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, AANCARule
from aanca.structure import load_protein

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def inputs():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    protein = load_protein(path)
    graph = build_graph(protein, cutoff=10.0)
    return {
        "node": torch.from_numpy(graph.node_features).float(),
        "edge": torch.from_numpy(graph.edge_features).float(),
        "neighbours": torch.from_numpy(graph.neighbours),
        "edge_mask": torch.from_numpy(graph.edge_mask),
        "envelope": torch.from_numpy(graph.envelope).float(),
        "graph": graph,
        "protein": protein,
    }


def _run(rule, inputs, node=None, hidden=None):
    node = inputs["node"] if node is None else node
    n = node.shape[0]
    hidden = rule.initial_hidden(n) if hidden is None else hidden
    return rule(
        node,
        hidden,
        inputs["edge"],
        inputs["neighbours"],
        inputs["edge_mask"],
        inputs["envelope"],
    )


def _randomise_heads(rule, seed=0):
    """Identity init makes mu identically zero, which would make some tests vacuous."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (rule.delta_head[-1], rule.kappa_head[-1]):
            layer.weight.normal_(0.0, 0.2)
            layer.bias.normal_(0.0, 0.2)


def test_feature_dims_match_the_featuriser(inputs):
    assert inputs["node"].shape[1] == NODE_FEATURE_DIM
    assert inputs["edge"].shape[2] == EDGE_FEATURE_DIM


def test_output_shapes(inputs):
    rule = AANCARule()
    mu, log_kappa, hidden = _run(rule, inputs)
    n = inputs["node"].shape[0]
    assert mu.shape == (n, rc.MAX_CHI)
    assert log_kappa.shape == (n, rc.MAX_CHI)
    assert hidden.shape == (n, rule.hidden_dim)
    assert torch.isfinite(mu).all() and torch.isfinite(log_kappa).all()


def test_untrained_rule_is_the_identity(inputs):
    """Growing-NCA init. A rule that starts by moving atoms randomly, applied
    thirty times, diverges before it learns anything."""
    rule = AANCARule()
    mu, _log_kappa, _h = _run(rule, inputs)
    assert torch.equal(mu, torch.zeros_like(mu)), "initial update must be exactly zero"


def test_initial_confidence_is_moderate(inputs):
    rule = AANCARule()
    _mu, log_kappa, _h = _run(rule, inputs)
    assert torch.allclose(log_kappa, torch.full_like(log_kappa, 1.0), atol=1e-4)


def test_log_kappa_is_capped(inputs):
    """The cap is what stops the likelihood being reduced by claiming confidence."""
    rule = AANCARule()
    with torch.no_grad():
        rule.kappa_head[-1].bias.fill_(1e4)
    _mu, log_kappa, _h = _run(rule, inputs)
    log_kappa = log_kappa.detach()
    assert float(log_kappa.max()) <= rule.max_log_kappa + 1e-5
    assert float(log_kappa.min()) >= -rule.max_log_kappa - 1e-5


def test_parameter_count_is_nca_scale(inputs):
    rule = AANCARule()
    count = rule.parameter_count()
    # The plan budgets ~220k. Flag drift in either direction -- a jump means an
    # accidental extra layer, a drop means a head went missing.
    assert 150_000 < count < 320_000, count


# -- the locality claim ----------------------------------------------------


def test_one_step_is_exactly_one_hop(inputs):
    """The scientific claim, asserted: a residue's update depends only on itself
    and its graph neighbours. A second message-passing layer would break this."""
    rule = AANCARule()
    _randomise_heads(rule)
    graph = inputs["graph"]
    n = inputs["node"].shape[0]

    target = 10
    neighbours = set(graph.neighbours[target][graph.edge_mask[target]].tolist())
    outsiders = [j for j in range(n) if j != target and j not in neighbours]
    assert outsiders, "every residue is a neighbour; pick a smaller cutoff to test"

    mu_before, _k, _h = _run(rule, inputs)

    perturbed = inputs["node"].clone()
    perturbed[outsiders[0]] += 5.0
    mu_after, _k, _h = _run(rule, inputs, node=perturbed)

    assert torch.equal(mu_before[target], mu_after[target]), (
        "a non-neighbour changed this residue's update"
    )


def test_a_neighbour_does_change_the_update(inputs):
    """Guards against the previous test passing because nothing matters at all."""
    rule = AANCARule()
    _randomise_heads(rule)
    graph = inputs["graph"]

    target = 10
    neighbours = graph.neighbours[target][graph.edge_mask[target]].tolist()
    mu_before, _k, _h = _run(rule, inputs)

    perturbed = inputs["node"].clone()
    perturbed[neighbours[0]] += 5.0
    mu_after, _k, _h = _run(rule, inputs, node=perturbed)

    assert not torch.equal(mu_before[target], mu_after[target])


def test_layer_norm_introduces_no_cross_residue_pathway(inputs):
    """LayerNorm is over feature channels within one residue. If it were over
    the batch it would be a global information channel and this would fail."""
    rule = AANCARule()
    _randomise_heads(rule, seed=1)
    n = inputs["node"].shape[0]
    graph = inputs["graph"]

    target = 0
    reachable = set(graph.neighbours[target][graph.edge_mask[target]].tolist()) | {target}
    outsiders = [j for j in range(n) if j not in reachable]

    mu_before, kappa_before, _h = _run(rule, inputs)
    perturbed = inputs["node"].clone()
    perturbed[outsiders] += torch.randn(len(outsiders), NODE_FEATURE_DIM) * 3.0
    mu_after, kappa_after, _h = _run(rule, inputs, node=perturbed)

    assert torch.equal(mu_before[target], mu_after[target])
    assert torch.equal(kappa_before[target], kappa_after[target])


# -- training mechanics ----------------------------------------------------


def test_gradient_reaches_the_head_weights_at_init(inputs):
    """The zero-initialised final layer means no gradient reaches the trunk on
    the very first step -- only the head's own weights. Once those move, the
    trunk trains. Documented here because it looks like a bug otherwise."""
    rule = AANCARule()
    mu, log_kappa, _h = _run(rule, inputs)
    (mu.sum() + log_kappa.sum()).backward()

    final = rule.delta_head[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.abs().sum()) > 0.0

    first = rule.message[0]
    assert float(first.weight.grad.abs().sum()) == pytest.approx(0.0, abs=1e-12)


def test_gradient_reaches_the_trunk_once_the_head_is_nonzero(inputs):
    rule = AANCARule()
    _randomise_heads(rule, seed=2)
    mu, log_kappa, _h = _run(rule, inputs)
    (mu.sum() + log_kappa.sum()).backward()
    assert float(rule.message[0].weight.grad.abs().sum()) > 0.0
    assert float(rule.edge_embed.weight.grad.abs().sum()) > 0.0


def test_hidden_state_evolves_between_steps(inputs):
    rule = AANCARule()
    _randomise_heads(rule, seed=3)
    _mu, _k, h1 = _run(rule, inputs)
    assert float(h1.detach().abs().sum()) > 0.0
    _mu, _k, h2 = _run(rule, inputs, hidden=h1)
    assert not torch.equal(h1, h2)


def test_rule_is_deterministic(inputs):
    rule = AANCARule().eval()
    _randomise_heads(rule, seed=4)
    a = _run(rule, inputs)
    b = _run(rule, inputs)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_padded_edges_contribute_nothing(inputs):
    """Padded slots carry zero features and zero envelope. Corrupting them must
    not change the output, or the mask is being ignored somewhere."""
    rule = AANCARule()
    _randomise_heads(rule, seed=5)
    mu_before, _k, _h = _run(rule, inputs)

    edge = inputs["edge"].clone()
    padded = ~inputs["edge_mask"]
    edge[padded] = 7.0
    corrupted = dict(inputs)
    corrupted["edge"] = edge
    mu_after, _k, _h = _run(rule, corrupted)

    assert torch.allclose(mu_before, mu_after, atol=1e-6)
