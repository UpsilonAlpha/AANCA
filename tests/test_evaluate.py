"""The held-out protocol: fixed difficulty, or it measures nothing.

The whole reason this module exists is that the training loss moves its own
target as the rule improves. If the *evaluation* also moved -- resampled
damage, a fresh crop, a different asynchronous mask sequence -- it would have
the same defect and the progress curve would be unreadable. So the tests here
are mostly about things staying exactly the same.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from aanca.data import RepackingDataset, StructureSpec
from aanca.evaluate import build_eval_set, evaluate
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture
def dataset():
    paths = [DATA / f"{i}.pdb" for i in ("1UBQ", "193L")]
    if not all(p.exists() for p in paths):
        pytest.skip("run scripts/fetch_test_structures.py")
    return RepackingDataset(
        [StructureSpec(p.stem, p, p.stem) for p in paths],
        concentrations=(50.0, 0.0),
    )


@pytest.fixture
def config():
    return RolloutConfig(steps=4, truncate=2, k=24, cutoff=10.0)


def test_cases_cover_every_structure_and_level(dataset):
    cases = build_eval_set(dataset, [0, 1], concentrations=(100.0, 0.0))
    assert len(cases) == 4
    assert {c.pdb_id for c in cases} == {"1UBQ", "193L"}
    assert {c.concentration for c in cases} == {100.0, 0.0}


def test_damage_is_stored_not_reseeded(dataset):
    """Built twice, the damage must be identical to the last bit.

    Storing coordinates rather than a seed is the point: a seed only reproduces
    the damage while every sampling call upstream stays identical, which is a
    promise about unrelated code.
    """
    a = build_eval_set(dataset, [0], concentrations=(100.0,))
    b = build_eval_set(dataset, [0], concentrations=(100.0,))
    assert torch.equal(a[0].damaged, b[0].damaged)


def test_adding_a_level_does_not_disturb_an_existing_case(dataset):
    """Otherwise extending the protocol silently invalidates every earlier
    number on the curve."""
    one = build_eval_set(dataset, [0], concentrations=(100.0,))
    two = build_eval_set(dataset, [0], concentrations=(100.0, 20.0, 0.0))
    kept = next(c for c in two if c.concentration == 100.0)
    assert torch.equal(one[0].damaged, kept.damaged)


def test_adding_a_structure_does_not_disturb_an_existing_case(dataset):
    one = build_eval_set(dataset, [1], concentrations=(20.0,))
    two = build_eval_set(dataset, [0, 1], concentrations=(20.0,))
    kept = next(c for c in two if c.pdb_id == one[0].pdb_id)
    assert torch.equal(one[0].damaged, kept.damaged)


def test_damage_actually_damages(dataset):
    """A protocol that silently evaluated on native structures would report
    excellent numbers forever."""
    cases = build_eval_set(dataset, [0], concentrations=(0.0,))
    assert not torch.allclose(cases[0].damaged, cases[0].native, atol=1e-6)


def test_evaluation_is_deterministic(dataset, config):
    """Same weights, same cases, same numbers -- including the asynchronous
    update mask, which is random and would otherwise add run-to-run noise to
    the progress curve."""
    rule = AANCARule()
    cases = build_eval_set(dataset, [0], concentrations=(100.0, 0.0))
    first = evaluate(rule, cases, config)
    second = evaluate(rule, cases, config)
    assert first.rows["pooled"].chi_mae == second.rows["pooled"].chi_mae
    assert first.convergence == second.convergence
    assert first.native_drift == second.native_drift


def test_evaluation_leaves_no_gradient_and_restores_training_mode(dataset, config):
    rule = AANCARule()
    rule.train()
    cases = build_eval_set(dataset, [0], concentrations=(100.0,))
    evaluate(rule, cases, config)
    assert rule.training
    assert all(p.grad is None for p in rule.parameters())


def test_untrained_rule_reproduces_its_input(dataset, config):
    """The identity initialisation means an untrained rule returns the damaged
    structure, so its chi MAE must equal the damaged input's MAE.

    This is the calibration for every later number: it pins what "no skill at
    all" scores on this protocol, so an improvement cannot be confused with
    the damage having been mild to begin with.
    """
    rule = AANCARule()
    cases = build_eval_set(dataset, [0], concentrations=(100.0,))
    result = evaluate(rule, cases, config)
    label = "kappa 100"
    assert result.rows[label].chi_mae[0] == pytest.approx(
        result.input_mae[label], abs=2.0
    )


def test_native_drift_is_zero_for_an_untrained_rule(dataset, config):
    """And therefore says nothing until the weights have moved -- which is
    exactly why it is reported next to the accuracy rather than alone."""
    rule = AANCARule()
    cases = build_eval_set(dataset, [0], concentrations=(100.0,))
    assert evaluate(rule, cases, config).native_drift == pytest.approx(0.0, abs=1e-9)


def test_convergence_trace_has_one_entry_per_step(dataset, config):
    rule = AANCARule()
    cases = build_eval_set(dataset, [0], concentrations=(100.0,))
    result = evaluate(rule, cases, config)
    assert len(result.convergence) == config.steps
    assert all(v >= 0.0 for v in result.convergence)


def test_evaluation_refuses_an_empty_set(config):
    with pytest.raises(ValueError, match="no evaluation cases"):
        evaluate(AANCARule(), [], config)


def test_a_perturbed_rule_moves_the_numbers(dataset, config):
    """Guards against an evaluation that is deterministic because it is inert.

    Every test above asks for something to stay the same, which a function
    returning a constant would also satisfy.
    """
    cases = build_eval_set(dataset, [0], concentrations=(100.0,))
    rule = AANCARule()
    before = evaluate(rule, cases, config)
    with torch.no_grad():
        for p in rule.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    after = evaluate(rule, cases, config)
    assert before.rows["pooled"].chi_mae[0] != after.rows["pooled"].chi_mae[0]
    assert after.native_drift > 0.0
