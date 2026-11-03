"""Cartesian birth: a newborn placed from a frame, with no torsion anywhere.

Kept in its own file because these tests are a *comparison* between the two
birth modes rather than tests of one of them, and the thing they mostly assert
is that the comparison is fair.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from aanca.backbone import (
    backbone_atoms,
    canonical_triad,
    extend_cward,
    frame_from_triad,
    helical_birth_reference,
    ideal_geometry,
    triad_from_frame,
)
from aanca.growth import (
    GrowthConfig,
    _cartesian_triad,
    give_birth,
    growth_links,
    growth_objective,
    park_coords,
)
from aanca.model import (
    BACKBONE_INIT_OMEGA,
    BACKBONE_INIT_PHI,
    BACKBONE_INIT_PSI,
    AANCARule,
)
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_geometry import rotation_matrices
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"

#: The CA-CA distance of a trans peptide, Angstrom. Measured, not looked up:
#: it comes out at 3.804 for every (phi, psi), because it is fixed by the
#: peptide's own bond geometry and not by the torsions.
CA_CA = 3.804


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(path))
    return topo, coords, growth_links(topo)


def _one_birth(topo, coords, links, mode, perturb=0.0, seed=0, parent=20):
    """Wake ``parent + 1`` and return the whole coordinate tensor."""
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[parent] = True
    parked = park_coords(coords, alive)
    rule = AANCARule(growth=True, birth=mode)
    if perturb:
        torch.manual_seed(seed)
        for p in rule.birth_head.parameters():
            p.data.add_(torch.randn_like(p) * perturb)
    _a, birth, _aa, _t = rule.growth_outputs(
        torch.zeros(topo.n_residues, rule.node_dim),
        torch.zeros(topo.n_residues, rule.hidden_dim),
    )
    born = torch.zeros(topo.n_residues, dtype=parked.dtype)
    born[parent] = 1.0
    grown, _after = give_birth(
        parked,
        topo,
        alive,
        links,
        torch.zeros_like(born),
        born,
        birth.to(parked.dtype).detach(),
        birth_mode=mode,
    )
    return grown


def test_the_two_births_are_the_same_birth_on_an_ideal_parent():
    """The equivalence that makes comparing the two modes mean anything.

    Both are initialised to the helical reference, so from a parent whose own
    internal geometry is ideal they must agree to float precision. If they did
    not, the probe's torsion-versus-Cartesian table would be comparing two
    different starting structures rather than two parameterisations, and any
    difference in the result would be uninterpretable.
    """
    parent = canonical_triad(dtype=torch.float64)
    radians = [
        torch.tensor(math.radians(v))
        for v in (BACKBONE_INIT_PSI, BACKBONE_INIT_OMEGA, BACKBONE_INIT_PHI)
    ]
    by_torsion = extend_cward(
        *parent,
        radians[0],
        radians[1],
        radians[2],
        ideal_geometry(dtype=torch.float64),
    )

    axis_angle, offset = helical_birth_reference(
        BACKBONE_INIT_PHI, BACKBONE_INIT_PSI, BACKBONE_INIT_OMEGA, cward=True
    )
    rot = frame_from_triad(*parent)
    ca = parent[1] + (rot @ offset.unsqueeze(-1)).squeeze(-1)
    by_frame = triad_from_frame(
        rot @ rotation_matrices(axis_angle, axis_angle.norm()), ca
    )
    for a, b in zip(by_torsion, by_frame):
        assert float((a - b).norm()) < 1e-7


def test_on_a_crystal_parent_they_differ_by_the_parents_own_deviation(ubiquitin):
    """And the difference is accounted for, not merely observed to be small.

    Torsion birth reads the parent's *measured* N, CA and C through the NeRF.
    Cartesian birth reads only the parent's frame, which is blind to the
    N-CA-C angle. Residue 20 of 1UBQ has 109.84 deg against an ideal 111.00,
    and 1.16 deg at a 3.8 A lever is 0.077 A -- which is the gap observed.

    That blindness is the whole point: a Cartesian birth depends on nothing
    about either residue's own internal geometry, and therefore on nothing
    about its amino-acid type.
    """
    topo, coords, links = ubiquitin
    gap = (
        backbone_atoms(_one_birth(topo, coords, links, "torsion"), topo)[21]
        - backbone_atoms(_one_birth(topo, coords, links, "cartesian"), topo)[21]
    ).norm(dim=-1)
    assert 0.01 < float(gap.max()) < 0.15, float(gap.max())


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_ca_ca_distance_is_exact_whatever_the_head_asks_for(ubiquitin, seed):
    """The one invariant that genuinely is invariant, held by construction.

    The offset is the reference offset *rotated*, and a rotation preserves
    norm, so no head output -- however large or however badly trained -- can
    tear the peptide at the moment of birth. Only relaxation can, which is
    what the restraint is there to charge for.
    """
    topo, coords, links = ubiquitin
    parent_ca = backbone_atoms(coords, topo)[20, 1]
    grown = _one_birth(topo, coords, links, "cartesian", perturb=0.8, seed=seed)
    child_ca = backbone_atoms(grown, topo)[21, 1]
    assert float((child_ca - parent_ca).norm()) == pytest.approx(CA_CA, abs=2e-3)


def test_cartesian_birth_is_equivariant(ubiquitin):
    """Rotate the structure and the birth rotates with it.

    True because the offset and the rotation are both expressed in the
    parent's own frame. A birth predicted in the crystal's coordinate system
    would not be, and the failure would look like a model that works on one
    orientation of a protein and not on another -- which is hard to diagnose
    and easy to mistake for undertraining.
    """
    topo, coords, links = ubiquitin
    rot = rotation_matrices(
        torch.tensor([0.3, -0.8, 0.5], dtype=coords.dtype),
        torch.tensor(1.3, dtype=coords.dtype),
    )
    shift = torch.tensor([11.0, -4.0, 7.0], dtype=coords.dtype)
    moved = (coords.reshape(-1, 3) @ rot.transpose(-1, -2) + shift).reshape(
        coords.shape
    )

    here = backbone_atoms(_one_birth(topo, coords, links, "cartesian"), topo)[21, :3]
    there = backbone_atoms(_one_birth(topo, moved, links, "cartesian"), topo)[21, :3]
    expected = here @ rot.transpose(-1, -2) + shift
    assert float((there - expected).norm(dim=-1).max()) < 1e-8


def test_the_cartesian_triad_takes_no_argument_about_the_child(ubiquitin):
    """Type-independence, stated as a signature rather than as a measurement.

    ``_cartesian_triad`` is given the parent's triad and the head's output.
    There is nothing about the child to vary, which is the claim; moving the
    *parent* does move the birth, so the test can fail.
    """
    topo, coords, _links = ubiquitin
    bb = backbone_atoms(coords, topo)
    birth = torch.zeros(1, 6, dtype=coords.dtype)
    first = _cartesian_triad((bb[20, 0:1], bb[20, 1:2], bb[20, 2:3]), birth, 1)
    second = _cartesian_triad((bb[40, 0:1], bb[40, 1:2], bb[40, 2:3]), birth, 1)
    assert float((first[1] - second[1]).norm()) > 1.0


def test_an_unknown_birth_mode_is_refused(ubiquitin):
    topo, coords, links = ubiquitin
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[20] = True
    parked = park_coords(coords, alive)
    born = torch.zeros(topo.n_residues, dtype=parked.dtype)
    born[20] = 1.0
    with pytest.raises(ValueError, match="birth mode"):
        give_birth(
            parked,
            topo,
            alive,
            links,
            torch.zeros_like(born),
            born,
            torch.zeros(topo.n_residues, 2, 6, dtype=parked.dtype),
            birth_mode="frames",
        )


def test_the_rule_refuses_an_unknown_birth_parameterisation():
    with pytest.raises(ValueError, match="birth"):
        AANCARule(growth=True, birth="nonsense")


def test_the_head_width_differs_between_the_modes():
    """A shape difference, so a checkpoint cannot be loaded into the other mode."""
    torsion = AANCARule(growth=True, birth="torsion")
    cartesian = AANCARule(growth=True, birth="cartesian")
    assert torsion.birth_head[-1].out_features == 2 * 3 * 2
    assert cartesian.birth_head[-1].out_features == 2 * 6
    node = torch.zeros(5, cartesian.node_dim)
    hidden = torch.zeros(5, cartesian.hidden_dim)
    assert tuple(torsion.growth_outputs(node, hidden)[1].shape) == (5, 2, 3)
    assert tuple(cartesian.growth_outputs(node, hidden)[1].shape) == (5, 2, 6)


def test_the_untrained_cartesian_head_asks_for_no_deviation():
    """Zeroed, so the reference birth is what an untrained rule performs."""
    rule = AANCARule(growth=True, birth="cartesian")
    _a, birth, _aa, _t = rule.growth_outputs(
        torch.zeros(3, rule.node_dim), torch.zeros(3, rule.hidden_dim)
    )
    assert float(birth.detach().abs().max()) == 0.0


def test_the_direction_swing_is_bounded_and_the_orientation_is_not():
    """Measured bounds: 0.72 rad of swing reaches sheet, so 1.0 is the cap.

    The orientation is deliberately uncapped -- polyproline II sits 3.07 rad
    from helix, nearly antipodal, so any cap tight enough to matter would
    exclude real secondary structure.
    """
    rule = AANCARule(growth=True, birth="cartesian")
    torch.manual_seed(0)
    for p in rule.birth_head.parameters():
        p.data.add_(torch.randn_like(p) * 5.0)
    _a, birth, _aa, _t = rule.growth_outputs(
        torch.randn(64, rule.node_dim), torch.randn(64, rule.hidden_dim)
    )
    birth = birth.detach()
    swing = birth[..., :3].norm(dim=-1)
    assert float(swing.max()) <= rule.max_birth_direction + 1e-6
    assert float(birth[..., 3:].norm(dim=-1).max()) > rule.max_birth_direction


def test_every_growth_head_receives_gradient_under_cartesian_birth(ubiquitin):
    topo, coords, _links = ubiquitin
    rule = AANCARule(growth=True, birth="cartesian")
    config = RolloutConfig(
        steps=16,
        truncate=99,
        k=24,
        growth=GrowthConfig(seeds=2, birth="cartesian"),
    )
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
