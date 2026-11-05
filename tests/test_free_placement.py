"""Free placement: a residue that places itself instead of being placed.

Under ``birth="torsion"`` and ``birth="cartesian"`` a newborn's frame is
``rot_parent @ rot_reference @ rot_spin``, so the learned part of placement sits
inside a composition and an orientation error at one birth rotates every
residue grown through it -- about twenty times the angle at twenty Angstrom.
Under ``birth="free"`` there is no birth head at all: the newborn goes to the
unlearned reference pose and then relaxes into position with the frame head,
reading its own neighbourhood. Learned error is then additive over residues
rather than multiplicative along a path.

These tests pin the three things that has to mean: that the head is really
gone, that a free birth starts where the other two start so a comparison
between them is fair, and that the placement window lets a newborn travel far
enough to be doing the placing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.backbone import backbone_atoms
from aanca.growth import (
    GrowthConfig,
    give_birth,
    growth_links,
    growth_objective,
    park_coords,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(path))
    return topo, coords, growth_links(topo)


def test_the_birth_head_does_not_exist():
    """Not zeroed, not frozen -- absent.

    A frozen head still has parameters an optimiser steps and a checkpoint
    stores, and "the birth head is not being trained" is then a claim about a
    run's configuration rather than about its architecture. Removing it makes
    the claim structural.
    """
    free = AANCARule(growth=True, birth="free")
    assert not hasattr(free, "birth_head")
    assert hasattr(AANCARule(growth=True, birth="cartesian"), "birth_head")
    node = torch.randn(8, free.node_dim)
    hidden = torch.randn(8, free.hidden_dim)
    _activation, birth, _axis, _translation = free.growth_outputs(node, hidden)
    assert birth is None


def test_a_free_birth_lands_where_an_untrained_cartesian_birth_lands(ubiquitin):
    """The comparison has to start from the same structure to mean anything.

    ``free`` is the Cartesian birth with its head's output held at zero, which
    is exactly the helical reference. So at initialisation the three modes put
    a newborn in the same place, and what differs between them is only what is
    learnable about where it ends up.
    """
    topo, coords, links = ubiquitin
    parent = 20
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[parent] = True
    parked = park_coords(coords, alive)
    decided = torch.zeros(topo.n_residues)
    decided[parent] = 1.0
    zero = torch.zeros(topo.n_residues)

    rule = AANCARule(growth=True, birth="cartesian")
    _a, birth, _aa, _t = rule.growth_outputs(
        torch.zeros(topo.n_residues, rule.node_dim),
        torch.zeros(topo.n_residues, rule.hidden_dim),
    )
    cartesian, _alive = give_birth(
        parked, topo, alive, links, zero, decided, birth.detach(),
        birth_mode="cartesian",
    )
    free, _alive = give_birth(
        parked, topo, alive, links, zero, decided, None, birth_mode="free"
    )
    child = parent + 1
    assert torch.allclose(cartesian[child], free[child], atol=1e-9)


def test_the_free_birth_keeps_the_exact_ca_ca_distance(ubiquitin):
    """The one invariant a Cartesian birth has, and free placement keeps it.

    3.804 A for every trans peptide whatever the torsions, because it is fixed
    by the peptide's own bond geometry. A birth chooses a direction, not a
    distance, so the newborn's first pose is wrong in orientation only.
    """
    topo, coords, links = ubiquitin
    parent = 20
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[parent] = True
    decided = torch.zeros(topo.n_residues)
    decided[parent] = 1.0
    placed, _alive = give_birth(
        park_coords(coords, alive), topo, alive, links,
        torch.zeros(topo.n_residues), decided, None, birth_mode="free",
    )
    ca = backbone_atoms(placed, topo)[:, 1]
    assert float((ca[parent + 1] - ca[parent]).norm()) == pytest.approx(3.804, abs=5e-3)


def test_the_placement_window_is_the_only_loosened_bound():
    """A young residue may travel; a settled one may not.

    This is what makes free placement different from the Cartesian birth with
    a frozen head. Without it a newborn is allowed 0.1 rad and 0.3 A per step
    to leave a pose it inherited, which over a rollout does not undo an
    inherited orientation -- the coalescence probe measured r = 0.970 between
    junction gaps and accumulated drift.
    """
    rule = AANCARule(growth=True, birth="free")
    torch.manual_seed(0)
    for p in rule.frame_head.parameters():
        p.data.add_(torch.randn_like(p) * 5.0)
    node = torch.randn(64, rule.node_dim)
    hidden = torch.randn(64, rule.hidden_dim)

    placing = torch.zeros(64, dtype=torch.bool)
    placing[:32] = True
    rotation = torch.where(placing, 3.14159265, rule.max_rotation).unsqueeze(-1)
    translation = torch.where(placing, 4.0, rule.max_translation).unsqueeze(-1)
    _a, _b, axis, move = rule.growth_outputs(node, hidden, rotation, translation)

    young_rotation = axis[:32].norm(dim=-1)
    settled_rotation = axis[32:].norm(dim=-1)
    assert float(settled_rotation.max()) <= rule.max_rotation + 1e-6
    assert float(young_rotation.max()) > rule.max_rotation
    assert float(young_rotation.max()) <= 3.14159265 + 1e-5
    assert float(move[32:].norm(dim=-1).max()) <= rule.max_translation + 1e-6
    assert float(move[:32].norm(dim=-1).max()) > rule.max_translation


def test_age_resets_at_birth_and_counts_up(ubiquitin):
    """A slot parked for eighty steps and just woken is new, not old.

    The placement window is read off the age, so a newborn inheriting its
    slot's age would be denied the window it exists for -- and would be denied
    it precisely in the late rollout where most births happen.
    """
    topo, coords, _links = ubiquitin
    # ``activation_init`` well above the default -2.0, so births are near
    # certain rather than merely likely: this is a test of the age bookkeeping
    # and a run that happened to grow nothing would pass it vacuously.
    rule = AANCARule(growth=True, birth="free", activation_init=1.0)
    config = RolloutConfig(
        steps=8, truncate=99, k=16,
        growth=GrowthConfig(seeds=2, birth="free", placement_steps=3),
    )
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[10] = alive[40] = True
    state = initial_state(coords, rule, alive=alive)
    assert state.age is not None
    assert int(state.age.max()) == 0
    result = rollout(
        rule, state, topo, config, torch.Generator().manual_seed(0), record=False
    )
    age, grown = result.state.age, result.state.alive
    assert int(age[~grown].abs().max()) == 0, "a dormant slot has no age"
    assert int(age[10]) == config.steps, "a seed ages every step"
    newborns = grown & (age < config.steps)
    assert int(newborns.sum()) > 0, "nothing was born, so nothing is being tested"
    assert int(age[newborns].max()) < config.steps


def test_free_placement_trains_the_heads_that_remain(ubiquitin):
    """Removing a head must not strand the others.

    The activation head's credit signal is the straight-through path through a
    newborn's coordinates, and under free placement those coordinates are a
    *constant* offset from the parent's frame. So the gradient reaching the
    activation head still exists -- it says "waking this neighbour changed the
    loss by this much" -- but it no longer passes through anything learned at
    the birth itself, which is the change. This asserts it survives.
    """
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True, birth="free", morphogen=True)
    # **Every head's output layer is zero-initialised**, which means that at
    # initialisation no gradient returns *into* the node features at all:
    # ``W.T @ g`` is zero for a zero ``W``. So the emit head -- whose only
    # route to a loss is through what other residues perceive -- reads exactly
    # zero gradient on an untrained rule. That is an initialisation artefact
    # and not a dead path, which this perturbation is here to demonstrate
    # rather than to paper over. Small, so the geometry does not diverge and
    # the finiteness assertion stays meaningful.
    torch.manual_seed(0)
    for head in (rule.frame_head, rule.activation_head):
        for p in head[-1].parameters():
            p.data.add_(torch.randn_like(p) * 0.01)
    config = RolloutConfig(
        steps=16, truncate=99, k=24,
        growth=GrowthConfig(seeds=2, birth="free", placement_steps=4),
    )
    terms, _result = growth_objective(
        rule, coords, topo, config, torch.Generator().manual_seed(0)
    )
    assert torch.isfinite(terms.total)
    terms.total.backward()
    for name in ("activation_head", "frame_head", "emit_head"):
        head = getattr(rule, name)
        total = sum(
            float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None
        )
        assert total > 0.0, f"{name} is not being trained"


def test_the_field_is_carried_and_reaches_beyond_the_perception_cutoff(ubiquitin):
    """The field's whole purpose, asserted on a real rollout.

    A 10 A cutoff reaches three or four residues along a helix. The field has
    to reach further than that along the chain, because the pairs it exists to
    connect are the ones the cutoff has already dropped.
    """
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True, birth="free", morphogen=True)
    config = RolloutConfig(
        steps=20, truncate=99, k=24,
        growth=GrowthConfig(seeds=1, birth="free", placement_steps=4),
    )
    # **The field conducts through alive residues only, so it reaches exactly
    # as far as the chain has been grown and no further.** That is a real limit
    # and it is recorded here rather than hidden: a field cannot connect two
    # residues the growing structure has not yet connected covalently, so two
    # separately seeded segments have separate fields until they fuse. What it
    # does connect is two residues far apart *in sequence* on one grown run --
    # which is the case where the 10 A cutoff has dropped the edge and the
    # chain between them exists. So the reach is measured on a grown stretch
    # rather than on one the rollout has to grow first.
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[20:50] = True
    result = rollout(
        rule,
        initial_state(coords, rule, alive=alive),
        topo,
        config,
        torch.Generator().manual_seed(0),
        record=False,
    )
    field = result.state.morphogen
    assert field is not None
    carrying = int((field[:, 0] > 0).sum())
    # One residue per hop from the run's N-terminus, so twenty steps carries it
    # twenty residues -- five times what a 10 A cutoff reaches along a helix,
    # and the pairs this exists for are the ones that cutoff has dropped.
    assert carrying == 20, f"the field reached {carrying} residues in 20 steps"
    assert bool(torch.isfinite(field).all())


def test_a_rule_without_the_field_carries_none(ubiquitin):
    """Off by default, and off means absent rather than zero.

    An all-zero field would still cost the transport each step and would still
    widen nothing, so it would look like a working feature while being a
    constant. ``None`` is the state every run before this one was in.
    """
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True, birth="cartesian")
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[20] = True
    state = initial_state(coords, rule, alive=alive)
    assert state.morphogen is None
    config = RolloutConfig(
        steps=4, truncate=99, k=16,
        growth=GrowthConfig(seeds=1, birth="cartesian"),
    )
    result = rollout(
        rule, state, topo, config, torch.Generator().manual_seed(0), record=False
    )
    assert result.state.morphogen is None
