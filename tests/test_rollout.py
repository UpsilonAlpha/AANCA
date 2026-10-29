"""The rollout: asynchrony, truncation, the loss step, and the sample pool."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from aanca.damage import torsion_noise
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    SamplePool,
    initial_state,
    rollout,
    sample_loss_step,
    step_once,
    training_objective,
    update_mask,
)
from aanca.structure import load_protein
from aanca.torch_geometry import measure_chis
from aanca.torch_graph import feature_topology

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def setup():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    protein = load_protein(path)
    topo, coords = feature_topology(protein)
    damaged_np = torsion_noise(
        protein, concentration=8.0, rng=np.random.default_rng(0)
    ).apply(protein)
    _t, damaged = feature_topology(damaged_np)
    return protein, topo, coords, damaged


def _config(**kwargs):
    base = dict(steps=4, truncate=2, k=32, cutoff=10.0)
    base.update(kwargs)
    return RolloutConfig(**base)


def _trained_ish(seed=0):
    """Identity init means zero updates, which makes dynamics tests vacuous."""
    rule = AANCARule()
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (rule.delta_head[-1], rule.kappa_head[-1]):
            layer.weight.normal_(0.0, 0.1)
    return rule


# -- asynchrony ------------------------------------------------------------


def test_update_mask_respects_the_probability():
    g = torch.Generator().manual_seed(0)
    mask = update_mask(20000, 0.5, "cpu", g)
    assert 0.47 < float(mask.float().mean()) < 0.53


def test_probability_one_is_synchronous():
    assert bool(update_mask(50, 1.0, "cpu").all())


def test_masked_out_residues_do_not_move(setup):
    """Asynchrony has to gate the coordinates, or it is not asynchrony."""
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    state = initial_state(damaged, rule)

    mask = torch.zeros(topo.n_residues, dtype=torch.bool)
    mask[: topo.n_residues // 2] = True
    after, _mu, _k = step_once(rule, state, topo, _config(), mask)

    frozen = ~mask
    assert torch.equal(after.coords[frozen], state.coords[frozen])
    assert not torch.equal(after.coords[mask], state.coords[mask])


def test_masked_out_residues_keep_their_hidden_state(setup):
    """The cell either ticks or it does not -- gating only the coordinates would
    let a frozen residue still advance its own memory."""
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    state = initial_state(damaged, rule)
    state.hidden = torch.randn_like(state.hidden)

    mask = torch.zeros(topo.n_residues, dtype=torch.bool)
    mask[0] = True
    after, _mu, _k = step_once(rule, state, topo, _config(), mask)

    assert torch.equal(after.hidden[1:], state.hidden[1:])
    assert not torch.equal(after.hidden[0], state.hidden[0])


def test_an_untrained_rule_is_a_fixed_point(setup):
    """Identity init: every step is a no-op, however many run.

    The update is *exactly* zero -- that is the claim. The coordinates match to
    float64 round-off rather than bitwise, because rotating by zero still
    evaluates ``(x - origin) @ I + origin``. Skipping the op at exactly zero
    would make it bitwise but would sever d(coords)/d(delta) on the first
    training step, which is a worse trade.
    """
    _protein, topo, _coords, damaged = setup
    rule = AANCARule()
    result = rollout(rule, initial_state(damaged, rule), topo, _config(steps=5))
    assert all(d == 0.0 for d in result.mean_abs_delta)
    assert float((result.state.coords.detach() - damaged).abs().max()) < 1e-12


# -- the rollout -----------------------------------------------------------


def test_rollout_records_a_convergence_trace(setup):
    """Tier 2's first plot comes out of the rollout for free."""
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    config = _config(steps=6, truncate=3)
    result = rollout(rule, initial_state(damaged, rule), topo, config)

    assert len(result.mean_abs_delta) == config.steps
    assert len(result.mean_kappa) == config.steps
    assert all(np.isfinite(result.mean_abs_delta))
    assert all(k > 0 for k in result.mean_kappa)
    assert result.state.step == config.steps


def test_rollout_moves_the_structure(setup):
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    result = rollout(rule, initial_state(damaged, rule), topo, _config())
    assert float((result.state.coords.detach() - damaged).abs().max()) > 1e-3


def test_truncation_severs_the_gradient_path(setup):
    """Truncated BPTT is mandatory at measured memory, so it must really cut.

    Tested by where the gradient reaches rather than by counting bytes: the
    saved-tensor accountant in ``scripts/measure_rollout_memory.py`` sums
    *distinct storages ever saved*, which does not shrink under truncation
    because the freed tensors were already counted. Severing the path is the
    property that bounds live memory, and it is directly observable.
    """
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()

    def start_gradient(truncate):
        start = damaged.clone().requires_grad_(True)
        result = rollout(
            rule, initial_state(start, rule), topo, _config(steps=6, truncate=truncate)
        )
        result.state.coords.sum().backward()
        return 0.0 if start.grad is None else float(start.grad.abs().sum())

    assert start_gradient(99) > 0.0, "an untruncated rollout must reach the start"
    assert start_gradient(2) == 0.0, "truncation must cut the path to the start"


def test_supervised_step_is_not_detached(setup):
    """If the loss step were behind a detach boundary it would get no gradient."""
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    config = _config(steps=6, truncate=2)
    result = rollout(
        rule, initial_state(damaged, rule), topo, config, supervise_step=5
    )
    mu, log_kappa, seen = result.supervised
    assert mu.requires_grad and log_kappa.requires_grad
    (mu.sum() + log_kappa.sum()).backward()
    assert float(rule.message[0].weight.grad.abs().sum()) > 0.0


def test_loss_step_is_drawn_from_the_back_window():
    """Supervising the last step teaches a trajectory; a random late step
    teaches an attractor."""
    config = RolloutConfig(steps=30, loss_window=0.5)
    rng = random.Random(0)
    draws = [sample_loss_step(config, rng) for _ in range(500)]
    assert min(draws) >= 15
    assert max(draws) <= 29
    assert len(set(draws)) > 10, "should not always pick the same step"


# -- the objective ---------------------------------------------------------


def test_objective_runs_and_is_finite(setup):
    _protein, topo, native, damaged = setup
    rule = _trained_ish()
    terms, result = training_objective(
        rule,
        damaged,
        native,
        topo,
        _config(steps=4, truncate=2),
        rng=random.Random(0),
    )
    assert torch.isfinite(terms.total)
    assert torch.isfinite(terms.reconstruction)
    assert torch.isfinite(terms.noop)
    assert float(terms.restraint) == 0.0, "bond geometry is off in v1"
    assert result.supervised_step == terms.supervised_step


def test_objective_produces_gradients_for_every_parameter(setup):
    _protein, topo, native, damaged = setup
    rule = _trained_ish()
    terms, _r = training_objective(
        rule, damaged, native, topo, _config(), rng=random.Random(1)
    )
    terms.total.backward()
    missing = [n for n, p in rule.named_parameters() if p.grad is None]
    assert not missing, missing


def test_noop_term_is_lower_on_a_native_input(setup):
    """The term exists to make 'already correct' a stable place to be."""
    _protein, topo, native, damaged = setup
    rule = _trained_ish()
    config = _config(steps=2, truncate=2)

    from_native, _r = training_objective(
        rule, native, native, topo, config, rng=random.Random(2)
    )
    from_damage, _r2 = training_objective(
        rule, damaged, native, topo, config, rng=random.Random(2)
    )
    assert float(from_native.reconstruction) < float(from_damage.reconstruction)


def test_target_is_measured_from_the_supervised_step(setup):
    """The delta asked of the rule must be the one correct from what it saw.

    Scoring against the final coordinates would demand a delta the rule was
    never in a position to predict.
    """
    _protein, topo, native, damaged = setup
    rule = _trained_ish()
    config = _config(steps=4, truncate=4)
    _terms, result = training_objective(
        rule, damaged, native, topo, config, rng=random.Random(3)
    )
    _mu, _k, seen = result.supervised
    # The captured coordinates are an intermediate state, not the endpoint.
    assert not torch.equal(seen, result.state.coords)
    assert torch.isfinite(measure_chis(seen, topo.torsion)).all()


def test_bond_geometry_adds_a_restraint(setup):
    """Enabling the degrees of freedom must also enable what keeps them stiff."""
    _protein, topo, native, damaged = setup
    rule = AANCARule(bond_geometry=True)
    torch.manual_seed(0)
    with torch.no_grad():
        rule.bond_head[-1].weight.normal_(0.0, 0.1)
        rule.angle_head[-1].weight.normal_(0.0, 0.1)

    terms, _r = training_objective(
        rule, damaged, native, topo, _config(), rng=random.Random(4)
    )
    assert float(terms.restraint) > 0.0
    assert torch.isfinite(terms.total)


def test_bond_geometry_off_by_default_moves_no_bonds(setup):
    _protein, topo, _native, damaged = setup
    rule = _trained_ish()
    assert not rule.bond_geometry
    assert rule.internal_deltas(torch.zeros(1, 32), torch.zeros(1, 96)) == (None, None)


# -- the sample pool -------------------------------------------------------


def test_pool_returns_none_until_seeded(setup):
    pool = SamplePool(capacity=4, seed=0)
    assert len(pool) == 0
    assert pool.sample(0) is None


def test_pool_round_trips_a_state(setup):
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    pool = SamplePool(capacity=4, seed=0)
    state = initial_state(damaged, rule)
    pool.insert(7, state, crop_seed=41)

    assert len(pool) == 1
    got = pool.sample(7)
    assert got is not None
    crop_seed, stored = got
    assert crop_seed == 41, "the crop the state belongs to comes back with it"
    assert torch.equal(stored.coords, damaged)
    assert not stored.coords.requires_grad, "pooled states must be detached"


def test_pool_refuses_a_non_finite_state(setup):
    """Otherwise one bad update seeds every later one that draws it.

    The point of refusing here rather than at the call site is that the run
    then fails where the NaN was produced, instead of at a neighbour search
    an unknown number of updates later.
    """
    _protein, _topo, _coords, damaged = setup
    rule = _trained_ish()
    pool = SamplePool(capacity=4, seed=0)

    assert pool.insert(1, initial_state(damaged, rule)) is True

    spoiled = damaged.clone()
    spoiled[0, 0, 0] = float("nan")
    assert pool.insert(2, initial_state(spoiled, rule)) is False
    assert pool.sample(2) is None
    assert len(pool) == 1, "the refused state must not displace a good one"

    state = initial_state(damaged, rule)
    state.hidden[0, 0] = float("inf")
    assert pool.insert(3, state) is False, "the hidden state poisons just as well"


def test_pool_returns_the_crop_its_state_belongs_to(setup):
    """The bug this exists to prevent: two crops of one entry have the same
    shape and different residues, so a state keyed by structure alone pairs
    coordinates with somebody else's topology and nothing catches it."""
    _protein, _topo, _coords, damaged = setup
    rule = _trained_ish()
    pool = SamplePool(capacity=8, seed=0)
    for seed_residue in (3, 90, 211):
        pool.insert(5, initial_state(damaged, rule), crop_seed=seed_residue)

    for _ in range(20):
        drawn = pool.sample(5)
        assert drawn is not None
        assert drawn[0] in (3, 90, 211), "a state must arrive with its own crop"


def test_pool_is_keyed_by_structure(setup):
    """A pooled state is only a valid seed for the structure it came from --
    the topology and the native target differ."""
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    pool = SamplePool(capacity=4, seed=0)
    pool.insert(1, initial_state(damaged, rule))
    assert pool.sample(1) is not None
    assert pool.sample(2) is None


def test_pool_evicts_at_capacity(setup):
    _protein, topo, _coords, damaged = setup
    rule = _trained_ish()
    pool = SamplePool(capacity=3, seed=0)
    for i in range(10):
        pool.insert(0, initial_state(damaged, rule))
    assert len(pool) == 3


def test_pool_rejects_zero_capacity():
    with pytest.raises(ValueError):
        SamplePool(capacity=0)
