"""The identity head's neutral element is "keep", not "uniform"."""

from __future__ import annotations

import torch

from aanca.identity import LIBRARY_SIZE
from aanca.model import IDENTITY_KEEP_INIT, AANCARule


def _state(rule, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    node = torch.randn(n, rule.node_dim, generator=g)
    hidden = torch.randn(n, rule.hidden_dim, generator=g)
    return node, hidden


def test_the_v1_path_has_no_identity_head_at_all():
    """Same contract as the bond heads: off means absent, not zeroed."""
    rule = AANCARule()
    node, hidden = _state(rule)
    assert rule.identity_logits(node, hidden) is None
    assert not any("identity" in k for k in rule.state_dict())


def test_the_head_emits_one_logit_per_library_entry():
    rule = AANCARule(identity=True)
    node, hidden = _state(rule)
    logits = rule.identity_logits(node, hidden)
    assert logits.shape == (12, LIBRARY_SIZE)


def test_an_untrained_head_is_uniform_without_a_current_label():
    """Nothing in the logits but the keep term, so with no keep they are flat."""
    rule = AANCARule(identity=True)
    node, hidden = _state(rule)
    logits = rule.identity_logits(node, hidden)
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-7)


def test_an_untrained_head_holds_the_current_label():
    """The identity initialisation, for a discrete label.

    Without this the rule would resample every residue's type from a uniform
    distribution on step 1 and do it again thirty times.
    """
    rule = AANCARule(identity=True)
    node, hidden = _state(rule)
    current = torch.arange(12) % LIBRARY_SIZE
    logits = rule.identity_logits(node, hidden, current)
    assert torch.equal(logits.argmax(dim=-1), current)
    kept = logits.detach().softmax(dim=-1).gather(-1, current.unsqueeze(-1))
    assert float(kept.min()) > 0.97


def test_indices_and_one_hot_mean_the_same_thing():
    rule = AANCARule(identity=True)
    node, hidden = _state(rule, seed=3)
    current = torch.arange(12) % LIBRARY_SIZE
    one_hot = torch.zeros(12, LIBRARY_SIZE).scatter_(-1, current.unsqueeze(-1), 1.0)
    assert torch.allclose(
        rule.identity_logits(node, hidden, current),
        rule.identity_logits(node, hidden, one_hot),
        atol=1e-6,
    )


def test_a_soft_current_label_carries_gradient_to_the_head_that_chose_it():
    """Why the rollout passes a straight-through one-hot and not indices.

    The label committed at step t is an input at step t+1. If it goes in as an
    index, the chain stops there and the loss at t+1 can never reach the head
    that chose it -- which is the entire mechanism of a ripple.
    """
    rule = AANCARule(identity=True)
    node, hidden = _state(rule, seed=5)
    first = rule.identity_logits(node, hidden)
    soft = torch.softmax(first, dim=-1)
    second = rule.identity_logits(node, hidden, soft)
    second.square().sum().backward()
    grad = rule.identity_head[-1].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0
    assert rule.identity_keep.grad is not None


def test_the_keep_strength_is_learnable_and_starts_where_it_says():
    rule = AANCARule(identity=True)
    assert rule.identity_keep.requires_grad
    assert float(rule.identity_keep.detach()) == IDENTITY_KEEP_INIT
