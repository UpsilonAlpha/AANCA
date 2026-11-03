"""Closure-aware growth: the reach bound, the restraint, and what it credits."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.backbone import backbone_atoms
from aanca.growth import (
    CLOSURE_BINS,
    CLOSURE_DIM,
    REACH,
    GrowthConfig,
    closure_features,
    closure_separation,
    growth_links,
    growth_objective,
    pending_closures,
    reach,
    reachability_restraint,
    seed_alive,
)
from aanca.model import GROWTH_NODE_EXTRA, NODE_FEATURE_DIM, AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"

#: The trans peptide's CA-CA distance, measured constant over every (phi, psi).
TRANS_CA_CA = 3.804


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


# -- the bound -------------------------------------------------------------


def test_the_reach_table_is_a_protein_backbone():
    """Monotone, and its first entry is a peptide bond rather than an artifact.

    ``scripts/reach_probe.py`` reports the maximum at g = 1 as 4.32 A, which is
    no peptide bond at all -- trans is 3.80 and cis 2.9 -- so the table is
    taken at the 99.99th percentile instead. If a future measurement puts
    ``REACH[1]`` back above 4.1 it has picked up broken models again.
    """
    assert REACH[0] == 0.0
    assert all(REACH[g] > REACH[g - 1] for g in range(2, len(REACH)))
    assert TRANS_CA_CA < REACH[1] < 4.1, REACH[1]
    # The fall-off with separation is the whole point: a long gap cannot be
    # spanned at a peptide bond a residue.
    assert REACH[1] / 1 > REACH[10] / 10 > REACH[20] / 20


def test_reach_extrapolates_beyond_the_table():
    inside = reach(torch.tensor([0, 1, 5, len(REACH) - 1]))
    assert float(inside[0]) == 0.0
    assert float(inside[1]) == pytest.approx(REACH[1])
    beyond = reach(torch.tensor([len(REACH) + 4]))
    assert float(beyond) > REACH[-1]
    # Generous is the safe direction for a feasibility test: it never calls a
    # closure impossible that a real chain could have made.
    assert float(beyond) > float(reach(torch.tensor([len(REACH) - 1])))


def test_the_feature_width_cannot_drift_from_the_rules_node_dim():
    """Two modules have to agree on a number; this is the thing that says so."""
    assert CLOSURE_DIM == GROWTH_NODE_EXTRA
    assert CLOSURE_DIM == 2 * (len(CLOSURE_BINS) + 2)


# -- which pairs are pending -----------------------------------------------


def test_pending_closures_are_consecutive_alive_pairs(ubiquitin):
    topo, _coords, links = ubiquitin
    alive = _alive(topo.n_residues, [5, 6, 20, 50])
    left, right = pending_closures(alive, links)
    assert list(zip(left.tolist(), right.tolist())) == [(5, 6), (6, 20), (20, 50)]


def test_a_covalent_break_is_not_a_closure(ubiquitin):
    """Two molecules do not have to be joined, so they must not be charged."""
    topo, _coords, links = ubiquitin
    broken = links.clone()
    broken[10] = False
    alive = _alive(topo.n_residues, [5, 30])
    assert pending_closures(alive, broken)[0].numel() == 0
    assert pending_closures(alive, links)[0].numel() == 1


def test_one_alive_residue_has_nothing_pending(ubiquitin):
    topo, _coords, links = ubiquitin
    assert pending_closures(_alive(topo.n_residues, [7]), links)[0].numel() == 0


# -- the restraint ---------------------------------------------------------


def test_a_crystal_structure_charges_exactly_nothing(ubiquitin):
    """A real chain is closed everywhere, so every pending closure is feasible."""
    topo, coords, links = ubiquitin
    full = torch.ones(topo.n_residues, dtype=torch.bool)
    assert float(reachability_restraint(coords, topo, full, links)) == 0.0
    # And on any subset of it, because the native geometry is reachable
    # whatever is dormant.
    sparse = _alive(topo.n_residues, range(0, topo.n_residues, 7))
    assert float(reachability_restraint(coords, topo, sparse, links)) == 0.0


def test_a_real_peptide_bond_needs_no_special_case(ubiquitin):
    """A pair at separation 1 is included and charges nothing, by arithmetic."""
    topo, coords, links = ubiquitin
    alive = _alive(topo.n_residues, [30, 31])
    left, _right = pending_closures(alive, links)
    assert left.numel() == 1, "the adjacent pair must be a pending closure"
    bb = backbone_atoms(coords, topo)
    assert float((bb[31, 1] - bb[30, 1]).norm()) == pytest.approx(TRANS_CA_CA, abs=0.1)
    assert float(reachability_restraint(coords, topo, alive, links)) == 0.0


def test_a_displaced_segment_is_charged_for_what_it_cannot_span(ubiquitin):
    topo, coords, links = ubiquitin
    alive = _alive(topo.n_residues, [20, 24])
    moved = coords.clone()
    moved[24:] += 30.0
    before = float(reachability_restraint(coords, topo, alive, links))
    after = float(reachability_restraint(moved, topo, alive, links))
    assert before == 0.0
    # Four residues of separation reach 14.37 A; the displacement is ~52.
    assert after > 30.0, after


def test_the_gradient_asks_for_the_gap_to_close(ubiquitin):
    """The sign, which is the one thing a restraint must not get wrong."""
    topo, coords, links = ubiquitin
    alive = _alive(topo.n_residues, [20, 24])
    moved = coords.clone()
    moved[24:] += 30.0
    moved.requires_grad_(True)
    reachability_restraint(moved, topo, alive, links).backward()

    bb = backbone_atoms(moved.detach(), topo)
    separation = bb[24, 1] - bb[20, 1]
    ca_index = topo.frame_idx[24, 1]
    # Descending the loss moves residue 24 along -grad, which must have a
    # component *towards* residue 20.
    step = -moved.grad[24, ca_index]
    assert float((step * separation).sum()) < 0.0


def test_single_seed_growth_is_never_charged(ubiquitin):
    """It costs nothing when there is nothing pending, which is why it is on."""
    topo, coords, links = ubiquitin
    rule = AANCARule(growth=True, birth="cartesian")
    generator = torch.Generator().manual_seed(0)
    alive = seed_alive(topo.n_residues, 1, generator)
    config = RolloutConfig(
        steps=30, truncate=30, k=24,
        growth=GrowthConfig(seeds=1, birth="cartesian"),
    )
    with torch.no_grad():
        out = rollout(
            rule, initial_state(coords, rule, alive=alive), topo, config,
            generator=generator, record=False,
        )
    final = out.state
    assert float(reachability_restraint(final.coords, topo, final.alive, links)) == 0.0


# -- the feature -----------------------------------------------------------


def test_the_separation_feature_says_how_far_the_nearest_alive_slot_is(ubiquitin):
    topo, _coords, links = ubiquitin
    alive = _alive(topo.n_residues, [10, 11, 15, 30])
    separation = closure_separation(alive, links)
    assert separation[10].tolist() == [0, 1]      # chain start side, then 11
    assert separation[11].tolist() == [1, 4]      # 10 behind, 15 ahead
    assert separation[15].tolist() == [4, 15]
    assert separation[30].tolist() == [15, 0]     # nothing alive ahead
    # A dormant slot is not an endpoint of any closure, so it has none.
    assert separation[12].tolist() == [0, 0]


def test_the_separation_block_is_two_one_hots(ubiquitin):
    topo, _coords, links = ubiquitin
    alive = _alive(topo.n_residues, [10, 11, 15, 30])
    block = closure_features(closure_separation(alive, links), torch.float32)
    assert tuple(block.shape) == (topo.n_residues, CLOSURE_DIM)
    assert torch.equal(block.sum(-1), torch.full((topo.n_residues,), 2.0))
    assert set(block.unique().tolist()) == {0.0, 1.0}


def test_a_growth_rule_is_wider_and_a_packing_rule_is_not():
    assert AANCARule().node_dim == NODE_FEATURE_DIM
    assert AANCARule(growth=True).node_dim == NODE_FEATURE_DIM + GROWTH_NODE_EXTRA
    # An explicit value still wins, so the default cannot trap a caller.
    assert AANCARule(growth=True, node_dim=99).node_dim == 99


# -- the objective ---------------------------------------------------------


def test_the_reach_term_credits_the_activation_head(ubiquitin):
    """Waking a residue is what spends the budget, so this is the decision it
    has to reach -- and it is the one the peptide restraint can never inform,
    because by the time that fires the budget is already gone."""
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True, birth="cartesian")
    config = RolloutConfig(
        steps=40, truncate=40, k=24,
        growth=GrowthConfig(seeds=8, birth="cartesian"),
    )
    terms, _result = growth_objective(
        rule, coords, topo, config, torch.Generator().manual_seed(0),
        bond_weight=0.0, completion_weight=0.0, reach_weight=1.0,
    )
    assert float(terms.reach) > 0.0, "no closure was foreclosed, so nothing is tested"
    assert terms.closures > 0
    terms.total.backward()
    for name in ("activation_head", "birth_head", "frame_head"):
        head = getattr(rule, name)
        total = sum(
            float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None
        )
        assert total > 0.0, f"{name} gets no closure signal"


def test_the_closure_count_is_reported_alongside_the_mean(ubiquitin):
    """The term is a mean, so without the count it cannot be read."""
    topo, coords, _links = ubiquitin
    terms, _result = growth_objective(
        AANCARule(growth=True, birth="cartesian"),
        coords,
        topo,
        RolloutConfig(
            steps=20, truncate=20, k=24,
            growth=GrowthConfig(seeds=4, birth="cartesian"),
        ),
        torch.Generator().manual_seed(0),
    )
    assert terms.closures >= 1
    assert float(terms.reach) >= 0.0
