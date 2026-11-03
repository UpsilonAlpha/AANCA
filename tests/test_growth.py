"""The alive mask: does a dormant slot exist, and can it be perceived?"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.backbone import C_N, backbone_atoms
from aanca.growth import (
    GrowthConfig,
    alive_topology,
    completion,
    frontier,
    give_birth,
    growth_links,
    limit_births,
    park_coords,
    peptide_bond_restraint,
    rigid_relax,
    seed_alive,
    straight_through_bernoulli,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout, step_once, update_mask
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(path))
    return topo, coords, growth_links(topo)


def _alive(n: int, indices) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    mask[list(indices)] = True
    return mask


# -- the frontier ----------------------------------------------------------


def test_the_frontier_is_the_ends_of_the_segments(ubiquitin):
    topo, _coords, links = ubiquitin
    alive = _alive(topo.n_residues, [10, 11, 12, 40])
    nward, cward = frontier(alive, links)
    assert torch.nonzero(nward).flatten().tolist() == [10, 40]
    assert torch.nonzero(cward).flatten().tolist() == [12, 40]
    # A segment of one is on both frontiers, which is what nucleation means.
    assert bool(nward[40] and cward[40])
    # The interior is on neither, so growth is O(surface) and not O(N).
    assert not bool(nward[11] or cward[11])


def test_cward_only_is_the_synthesis_order_control(ubiquitin):
    topo, _coords, links = ubiquitin
    alive = _alive(topo.n_residues, [10, 11])
    nward, cward = frontier(alive, links, cward_only=True)
    assert not bool(nward.any())
    assert bool(cward[11])


def test_limit_births_caps_the_frontier(ubiquitin):
    mask = torch.zeros(50, dtype=torch.bool)
    mask[::2] = True
    capped = limit_births(mask, 4, torch.Generator().manual_seed(0))
    assert int(capped.sum()) == 4
    assert bool((capped & ~mask).sum() == 0), "only frontier residues may be kept"
    assert torch.equal(limit_births(mask, 99, None), mask)


# -- perception ------------------------------------------------------------


def test_a_dormant_residue_holds_no_atoms_and_owner_agrees(ubiquitin):
    """``owner`` is indexed by surviving atom, so it has to be re-derived.

    If it is not, every atom after the first dormant residue is attributed to
    the wrong residue -- and neighbour selection still returns a plausible
    graph, which is why this is a test rather than a comment.
    """
    topo, _coords, _links = ubiquitin
    alive = _alive(topo.n_residues, [5, 6, 7])
    masked = alive_topology(topo, alive)
    assert int(masked.atom_mask.sum()) == int(topo.atom_mask[alive].sum())
    assert masked.owner.shape[0] == int(masked.atom_mask.sum())
    assert set(masked.owner.tolist()) == {5, 6, 7}
    assert not bool(masked.supervised_mask[~alive].any())


def test_parking_moves_the_dormant_and_keeps_their_shape(ubiquitin):
    topo, coords, _links = ubiquitin
    alive = _alive(topo.n_residues, [5, 6])
    parked = park_coords(coords, alive)
    assert torch.equal(parked[alive], coords[alive])
    assert float((parked[3] - coords[3]).norm(dim=-1).min()) > 100.0
    # Internal geometry survives, which is what makes a parked residue a
    # usable source triad when it is born.
    assert torch.allclose(
        torch.cdist(parked[3], parked[3]), torch.cdist(coords[3], coords[3]), atol=1e-9
    )


def test_the_whole_claim_a_dormant_residue_cannot_be_perceived(ubiquitin):
    """Randomise every dormant coordinate and the alive rule output is identical.

    The growth analogue of ``tests/test_blind.py``: it is not enough for
    dormant residues to be masked out of the loss, because neighbour selection
    and the minimum-heavy-atom envelope would still read them. If this fails,
    a growth rollout is quietly conditioned on the answer it is building.
    """
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True)
    torch.manual_seed(0)
    for p in rule.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    alive = _alive(topo.n_residues, range(20, 30))

    other = coords.clone()
    other[~alive] += torch.randn_like(other[~alive]) * 7.0

    config = RolloutConfig(steps=1, k=24, growth=GrowthConfig())
    mask = update_mask(topo.n_residues, 1.0, coords.device)
    outputs = []
    for start in (coords, other):
        state = initial_state(start, rule, alive=alive)
        _after, mu, log_kappa = step_once(
            rule, state, topo, config, mask, generator=torch.Generator().manual_seed(1)
        )
        outputs.append((mu[alive], log_kappa[alive]))
    assert torch.equal(outputs[0][0], outputs[1][0]), "dormant coordinates leaked"
    assert torch.equal(outputs[0][1], outputs[1][1])


# -- the wake decision -----------------------------------------------------


def test_the_wake_decision_is_hard_forward_and_soft_backward(ubiquitin):
    logit = torch.tensor([-3.0, 0.0, 2.0], requires_grad=True)
    out = straight_through_bernoulli(logit, sample=False)
    assert out.detach().tolist() == [0.0, 0.0, 1.0]
    out.sum().backward()
    expected = torch.sigmoid(logit.detach()) * (1 - torch.sigmoid(logit.detach()))
    assert torch.allclose(logit.grad, expected, atol=1e-6)


def test_a_high_logit_wakes_and_a_low_one_does_not(ubiquitin):
    logit = torch.full((500,), 4.0)
    drawn = straight_through_bernoulli(logit, True, torch.Generator().manual_seed(0))
    assert float(drawn.mean()) > 0.9
    drawn = straight_through_bernoulli(-logit, True, torch.Generator().manual_seed(0))
    assert float(drawn.mean()) < 0.1


# -- birth -----------------------------------------------------------------


def test_a_newborn_is_bonded_to_its_parent(ubiquitin):
    topo, coords, links = ubiquitin
    alive = _alive(topo.n_residues, [20])
    parked = park_coords(coords, alive)
    torsions = torch.zeros(topo.n_residues, 2, 3, dtype=parked.dtype)
    torsions[:, :, 1] = torch.pi  # planar trans omega
    born_c = torch.zeros(topo.n_residues, dtype=parked.dtype)
    born_c[20] = 1.0
    grown, alive_after = give_birth(
        parked, topo, alive, links, torch.zeros_like(born_c), born_c, torsions
    )
    assert bool(alive_after[21]) and int(alive_after.sum()) == 2
    bb = backbone_atoms(grown, topo)
    # Ideal plus the child's own N-CA deviation, because the rigid transfer
    # lands CA exactly and lets N follow the residue's own geometry -- see
    # ``give_birth``. A hundredth of an Angstrom is the right scale; a tenth
    # would mean the triad or the transfer is wrong.
    assert float((bb[21, 0] - bb[20, 2]).norm()) == pytest.approx(C_N, abs=0.02)
    # The child's own internal geometry is its crystal one, carried rigidly.
    assert torch.allclose(
        torch.cdist(grown[21], grown[21]), torch.cdist(coords[21], coords[21]), atol=1e-8
    )


def test_a_refused_birth_leaves_the_child_parked(ubiquitin):
    topo, coords, links = ubiquitin
    alive = _alive(topo.n_residues, [20])
    parked = park_coords(coords, alive)
    zero = torch.zeros(topo.n_residues, dtype=parked.dtype)
    grown, alive_after = give_birth(
        parked, topo, alive, links, zero, zero,
        torch.zeros(topo.n_residues, 2, 3, dtype=parked.dtype),
    )
    assert torch.equal(grown, parked)
    assert torch.equal(alive_after, alive)


def test_the_peptide_restraint_is_small_on_a_crystal_and_large_when_torn(ubiquitin):
    topo, coords, links = ubiquitin
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    intact = float(peptide_bond_restraint(coords, topo, alive, links))
    assert intact < 0.02, intact
    torn = coords.clone()
    torn[::2] += 3.0
    # A ratio rather than a margin: the term is a mean over bonded pairs, so
    # one broken bond in seventy-five is a small absolute number however badly
    # it is broken, and an absolute threshold would be testing the chain
    # length instead of the term.
    assert float(peptide_bond_restraint(torn, topo, alive, links)) > 50.0 * intact


def test_the_rigid_relax_is_rigid_and_only_touches_the_moving(ubiquitin):
    topo, coords, _links = ubiquitin
    moving = _alive(topo.n_residues, [4, 5])
    axis = torch.zeros(topo.n_residues, 3)
    axis[:, 2] = 0.1
    shift = torch.full((topo.n_residues, 3), 0.2)
    moved = rigid_relax(coords, axis, shift, topo, moving)
    assert torch.equal(moved[~moving], coords[~moving])
    assert not torch.allclose(moved[4], coords[4])
    assert torch.allclose(
        torch.cdist(moved[4], moved[4]), torch.cdist(coords[4], coords[4]), atol=1e-9
    )


# -- the rollout -----------------------------------------------------------


def test_growth_only_ever_adds(ubiquitin):
    """A slot that is alive stays alive. There is no death in this model."""
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True)
    generator = torch.Generator().manual_seed(0)
    alive = seed_alive(topo.n_residues, 2, generator)
    config = RolloutConfig(
        steps=12, truncate=99, k=24, growth=GrowthConfig(activation="threshold")
    )
    state = initial_state(coords, rule, alive=alive)
    mask = update_mask(topo.n_residues, 1.0, coords.device)
    seen = alive.clone()
    for _ in range(6):
        state, _mu, _lk = step_once(
            rule, state, topo, config, mask, generator=generator
        )
        assert bool((seen & ~state.alive).sum() == 0), "a residue died"
        seen = state.alive.clone()


def test_a_growth_rollout_grows_and_stays_finite(ubiquitin):
    topo, coords, links = ubiquitin
    rule = AANCARule(growth=True)
    generator = torch.Generator().manual_seed(0)
    alive = seed_alive(topo.n_residues, 3, generator)
    config = RolloutConfig(steps=30, truncate=10, k=24, growth=GrowthConfig(seeds=3))
    out = rollout(
        rule,
        initial_state(coords, rule, alive=alive),
        topo,
        config,
        generator=generator,
        record=False,
    )
    assert completion(out.state.alive) > completion(alive)
    assert bool(torch.isfinite(out.state.coords).all())
    # Built from ideal geometry with an untrained frame head, so the chain it
    # grows must still be a chain.
    assert float(
        peptide_bond_restraint(out.state.coords, topo, out.state.alive, links).detach()
    ) < 0.1


def test_an_unknown_activation_mode_is_refused(ubiquitin):
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True)
    alive = _alive(topo.n_residues, [10, 11])
    config = RolloutConfig(
        steps=1, k=24, growth=GrowthConfig(activation="softmax")
    )
    with pytest.raises(ValueError, match="activation"):
        step_once(
            rule,
            initial_state(coords, rule, alive=alive),
            topo,
            config,
            update_mask(topo.n_residues, 1.0, coords.device),
        )


def test_a_growth_rollout_without_growth_heads_is_refused(ubiquitin):
    topo, coords, _links = ubiquitin
    rule = AANCARule()
    alive = _alive(topo.n_residues, [10, 11])
    config = RolloutConfig(steps=1, k=24, growth=GrowthConfig())
    with pytest.raises(ValueError, match="growth"):
        step_once(
            rule,
            initial_state(coords, rule, alive=alive),
            topo,
            config,
            update_mask(topo.n_residues, 1.0, coords.device),
        )


def test_the_v1_path_carries_no_alive_mask(ubiquitin):
    topo, coords, _links = ubiquitin
    rule = AANCARule()
    out = rollout(
        rule,
        initial_state(coords, rule),
        topo,
        RolloutConfig(steps=3, k=24),
        generator=torch.Generator().manual_seed(0),
        record=False,
    )
    assert out.state.alive is None


def test_seeding_is_reproducible_and_bounded(ubiquitin):
    topo, _coords, _links = ubiquitin
    a = seed_alive(topo.n_residues, 4, torch.Generator().manual_seed(7))
    b = seed_alive(topo.n_residues, 4, torch.Generator().manual_seed(7))
    assert torch.equal(a, b) and int(a.sum()) == 4
    with pytest.raises(ValueError, match="seeds"):
        seed_alive(10, 0)
    with pytest.raises(ValueError, match="slots"):
        seed_alive(10, 11)


# -- the objective ---------------------------------------------------------


def test_the_bound_is_alive_at_the_origin(ubiquitin):
    """The frame head is zero-initialised, so this is the first update's gradient."""
    from aanca.model import _bounded

    vector = torch.zeros(3, 3, requires_grad=True)
    _bounded(vector, 0.1).sum().backward()
    assert float(vector.grad.abs().sum()) > 0.0, "the frame head can never learn"
    assert float(_bounded(torch.full((1, 3), 100.0), 0.1).norm()) == pytest.approx(
        0.1, abs=1e-6
    )


def test_every_growth_head_receives_gradient(ubiquitin):
    """Three heads, three distinct jobs; a dead one would be silent otherwise."""
    from aanca.growth import growth_objective

    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True)
    config = RolloutConfig(steps=16, truncate=99, k=24, growth=GrowthConfig(seeds=2))
    terms, _result = growth_objective(
        rule, coords, topo, config, torch.Generator().manual_seed(0)
    )
    terms.total.backward()
    for name in ("activation_head", "birth_head", "frame_head"):
        head = getattr(rule, name)
        total = sum(
            float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None
        )
        assert total > 0.0, f"{name} is not being trained"


def test_the_completion_term_is_zero_once_there_is_nothing_left_to_grow(ubiquitin):
    """It pushes only while the frontier exists, which is what makes it safe."""
    from aanca.growth import growth_objective

    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True)
    config = RolloutConfig(
        steps=1, truncate=99, k=24, growth=GrowthConfig(seeds=topo.n_residues)
    )
    terms, result = growth_objective(
        rule, coords, topo, config, torch.Generator().manual_seed(0)
    )
    assert terms.alive_fraction == 1.0
    assert float(terms.completion) == 0.0


def test_a_fully_seeded_growth_rollout_starts_at_the_native_answer(ubiquitin):
    """The control: seeded everywhere, nothing is parked and FAPE starts at zero."""
    from aanca.fape import fape

    topo, coords, _links = ubiquitin
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    state = initial_state(coords, AANCARule(growth=True), alive=alive)
    assert torch.equal(state.coords, coords)
    assert float(fape(state.coords, coords, topo, alive)) == pytest.approx(0.0, abs=1e-9)


def test_the_objective_refuses_a_config_without_growth(ubiquitin):
    from aanca.growth import growth_objective

    topo, coords, _links = ubiquitin
    with pytest.raises(ValueError, match="growth"):
        growth_objective(
            AANCARule(growth=True), coords, topo, RolloutConfig(steps=1, k=24)
        )
