"""Growing the sequence with the structure: does a newborn know what it is?

The leak these tests exist for: a newborn's non-backbone atoms arrive by rigid
transform from its parked pose, so without the collapse onto CB it *is* its
native amino acid, sitting in the state where the clash term and anything
downstream can read it. Masking it out of perception is not enough, because
the state is what the next step is built from.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from aanca.backbone import backbone_atoms
from aanca.blind import beyond_cb_mask, blind_coords
from aanca.growth import GrowthConfig, growth_links, growth_objective, seed_alive
from aanca.identity import LIBRARY_SIZE, library_index, type_index
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    initial_identity,
    initial_state,
    rollout,
    step_once,
    update_mask,
)
from aanca.structure import load_protein
from aanca.torch_geometry import measure_chis, set_chis
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


def _rule(**kw):
    base = dict(growth=True, birth="cartesian", identity=True, identity_keep=2.0)
    base.update(kw)
    return AANCARule(**base)


def _config(**kw):
    base = dict(
        steps=12,
        truncate=99,
        k=24,
        blind=True,
        identity_sampling="argmax",
        growth=GrowthConfig(seeds=2, birth="cartesian"),
    )
    base.update(kw)
    return RolloutConfig(**base)


# -- the newborn's label ---------------------------------------------------


def test_a_dormant_slot_has_no_label_however_the_start_is_asked_for(ubiquitin):
    """Including ``native``, which is the one that would hand over the answer."""
    topo, coords, _links = ubiquitin
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[[10, 11, 40]] = True
    for start in ("native", "random"):
        identity = initial_identity(
            topo, coords, start, torch.Generator().manual_seed(0), alive=alive
        )
        assert float(identity[~alive].abs().max()) == 0.0, start
        assert float(identity[alive].sum()) == pytest.approx(3.0), start


def test_a_newborn_arrives_unlabelled(ubiquitin):
    """It inherits its slot's row, and the unknown token is a row of zeros.

    So the first thing it does, once it can perceive anything at all, is draw
    a label from its neighbourhood -- which is the ripple. If this fails the
    newborn is committing to a label chosen before it existed.
    """
    topo, coords, _links = ubiquitin
    # A positive activation bias and a threshold decision, so a birth is
    # certain rather than a 12 % draw -- otherwise the assertion below passes
    # by not testing anything.
    rule = _rule(activation_init=4.0)
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[20] = True
    identity = initial_identity(topo, coords, "native", alive=alive)
    state = initial_state(coords, rule, identity=identity, alive=alive)

    generator = torch.Generator().manual_seed(0)
    mask = update_mask(topo.n_residues, 1.0, coords.device)
    after, _mu, _lk = step_once(
        rule,
        state,
        topo,
        _config(growth=GrowthConfig(birth="cartesian", activation="threshold")),
        mask,
        generator=generator,
    )
    newly = after.alive & ~alive
    assert bool(newly.any()), "nothing was born, so the test is vacuous"
    assert float(after.identity[newly].detach().abs().max()) == 0.0


# -- the representation ----------------------------------------------------


def test_a_blind_growth_rollout_holds_no_side_chains_at_all(ubiquitin):
    """Not merely hidden from perception -- absent from the state.

    Every atom beyond CB must sit exactly on its own CB, so collapsing again
    changes nothing. That is the property that makes the structure honestly
    backbone-and-CB, and it is what lets the identity head *choose* a type
    instead of confirming one.
    """
    topo, coords, _links = ubiquitin
    rule = _rule()
    generator = torch.Generator().manual_seed(0)
    alive = seed_alive(topo.n_residues, 3, generator)
    out = rollout(
        rule,
        initial_state(
            coords,
            rule,
            identity=initial_identity(topo, coords, "masked", alive=alive),
            alive=alive,
        ),
        topo,
        _config(steps=16),
        generator=generator,
        record=False,
    )
    grown = out.state.coords
    assert torch.allclose(blind_coords(grown, topo), grown, atol=1e-9)
    beyond = beyond_cb_mask(topo)
    assert bool(beyond.any()), "1UBQ has side chains, so the mask must be non-empty"


def test_the_whole_claim_the_native_side_chains_cannot_reach_the_grown_backbone(
    ubiquitin,
):
    """Randomise every native chi and the grown backbone is bit-identical.

    The growth analogue of ``tests/test_blind.py``'s headline. A de novo growth
    run must not be conditioned on the conformation -- or, through it, on the
    identity -- of the sequence it is supposed to be inventing.
    """
    topo, coords, _links = ubiquitin
    generator = torch.Generator().manual_seed(3)
    chis = measure_chis(coords, topo.torsion)
    scrambled = set_chis(
        coords,
        torch.rand(chis.shape, generator=generator, dtype=coords.dtype) * 2 * torch.pi
        - torch.pi,
        topo.torsion,
    )
    assert not torch.allclose(coords, scrambled), "the fixture must actually move"

    results = []
    for start in (coords, scrambled):
        rule = _rule()
        run_generator = torch.Generator().manual_seed(0)
        alive = seed_alive(topo.n_residues, 2, run_generator)
        out = rollout(
            rule,
            initial_state(
                start,
                rule,
                identity=initial_identity(topo, start, "masked", alive=alive),
                alive=alive,
            ),
            topo,
            _config(steps=10),
            generator=run_generator,
            record=False,
        )
        results.append(
            (backbone_atoms(out.state.coords, topo)[out.state.alive], out.state.alive)
        )
    assert torch.equal(results[0][1], results[1][1]), "a different set was grown"
    assert torch.equal(results[0][0], results[1][0]), "the side chains leaked"


def test_the_chi_head_is_inert_under_blind_growth(ubiquitin):
    """And that is correct, not a bug, which is why it is written down.

    With every side-chain atom collapsed onto CB, the chi1 axis CA-CB passes
    through all of them, so a chi rotation is the identity map and the delta
    head receives no gradient. The consequence worth knowing: growth and
    packing cannot share one rollout, because in the representation growth
    needs there is nothing for the packing head to move.
    """
    topo, coords, _links = ubiquitin
    rule = _rule()
    config = _config(steps=10)
    terms, _result = growth_objective(
        rule,
        coords,
        topo,
        config,
        torch.Generator().manual_seed(0),
        identity_weight=1.0,
    )
    terms.total.backward()
    chi = sum(
        float(p.grad.abs().sum())
        for p in rule.delta_head.parameters()
        if p.grad is not None
    )
    assert chi == 0.0, chi
    head = sum(
        float(p.grad.abs().sum())
        for p in rule.identity_head.parameters()
        if p.grad is not None
    )
    assert head > 0.0, "the identity head must be trained instead"


# -- the loss --------------------------------------------------------------


def test_the_identity_term_is_the_type_marginal_not_the_joint_label(ubiquitin):
    """So it is comparable with sequence recovery, and its floor is ln 20.

    An untrained head starts near the uninformative floor. Asserting the floor
    rather than a target is the point: it is the number that says whether the
    head is reading anything at all, the way ln 54 did for the joint label.
    """
    topo, coords, _links = ubiquitin
    terms, _result = growth_objective(
        _rule(),
        coords,
        topo,
        _config(),
        torch.Generator().manual_seed(0),
        identity_weight=1.0,
    )
    assert float(terms.identity) == pytest.approx(math.log(20), abs=0.6)
    assert 0.0 <= terms.recovery <= 1.0


def test_recovery_counts_only_the_residues_that_exist(ubiquitin):
    """A dormant slot has no label, so scoring one would be scoring nothing."""
    topo, coords, _links = ubiquitin
    terms, result = growth_objective(
        _rule(),
        coords,
        topo,
        _config(),
        torch.Generator().manual_seed(0),
        identity_weight=1.0,
    )
    assert terms.alive_fraction < 1.0, "the test needs some slots still dormant"
    assert not math.isnan(terms.recovery)
    # The denominator is the alive-and-supervised set, so a recovery of 1.0
    # would mean every grown residue was typed correctly, not every slot.
    assert result.state.alive.sum() >= 2


def test_growth_without_identity_reports_no_recovery(ubiquitin):
    topo, coords, _links = ubiquitin
    terms, _result = growth_objective(
        AANCARule(growth=True, birth="cartesian"),
        coords,
        topo,
        RolloutConfig(
            steps=8, truncate=99, k=24,
            growth=GrowthConfig(seeds=2, birth="cartesian"),
        ),
        torch.Generator().manual_seed(0),
    )
    assert math.isnan(terms.recovery)
    assert float(terms.identity) == 0.0


def test_the_identity_term_is_refused_without_the_head(ubiquitin):
    topo, coords, _links = ubiquitin
    with pytest.raises(ValueError, match="identity"):
        growth_objective(
            AANCARule(growth=True, birth="cartesian"),
            coords,
            topo,
            _config(),
            torch.Generator().manual_seed(0),
            identity_weight=1.0,
        )


def test_the_library_target_is_a_real_library_index(ubiquitin):
    """Guards the one thing a silent mismatch here would produce: a wrong loss."""
    topo, coords, _links = ubiquitin
    chi1 = torch.rad2deg(measure_chis(coords, topo.torsion)[:, 0])
    target = library_index(topo.torsion.resnames, chi1)
    assert int(target.min()) >= 0 and int(target.max()) < LIBRARY_SIZE
    types = type_index(target)
    assert int(types.min()) >= 0 and int(types.max()) < 20
