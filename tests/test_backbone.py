"""The placement primitive, and the size of the approximation built on it."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.backbone import (
    BackboneGeometry,
    C_N,
    backbone_atoms,
    extend_cward,
    extend_nward,
    ideal_geometry,
    measure_backbone_torsions,
    measured_geometry,
    place_atom,
    place_oxygen,
    rigid_onto_triad,
    _angle,
)
from aanca.structure import load_protein
from aanca.torch_geometry import dihedral
from aanca.torch_graph import feature_topology, sequence_links

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(path))
    return topo, coords, sequence_links(topo, coords.device)


def _per_residue(geometry: BackboneGeometry, parent: int, child: int) -> BackboneGeometry:
    """The six values one peptide bond needs, pulled out for a single pair.

    The bond across the peptide belongs to the parent and the two inside the
    residue belong to the child, which is the asymmetry
    :func:`extend_cward` documents. Getting this wrong is the one way the
    measured round trip below can fail for a reason other than the primitive.
    """
    return BackboneGeometry(
        n_ca=geometry.n_ca[child],
        ca_c=geometry.ca_c[child],
        c_n=geometry.c_n[parent],
        n_ca_c=geometry.n_ca_c[child],
        ca_c_n=geometry.ca_c_n[parent],
        c_n_ca=geometry.c_n_ca[parent],
    )


@pytest.mark.parametrize("torsion", [-3.0, -1.7, -0.4, 0.0, 0.9, 2.5, 3.1])
def test_the_primitive_produces_the_internal_coordinates_it_was_asked_for(torsion):
    """A sign error here mirrors every grown fold and nothing else reveals it."""
    torch.manual_seed(0)
    p0, p1, p2 = torch.randn(3, 3, dtype=torch.float64)
    bond = torch.tensor(1.33, dtype=torch.float64)
    angle = torch.tensor(2.03, dtype=torch.float64)
    asked = torch.tensor(torsion, dtype=torch.float64)
    x = place_atom(p0, p1, p2, bond, angle, asked)
    assert float((x - p2).norm()) == pytest.approx(1.33, abs=1e-9)
    assert float(_angle(p1, p2, x)) == pytest.approx(2.03, abs=1e-9)
    assert float(dihedral(p0, p1, p2, x)) == pytest.approx(torsion, abs=1e-9)


def test_measured_geometry_regrows_the_crystal_backbone_exactly(ubiquitin):
    """The control. With the structure's own bonds and angles this is a round trip."""
    topo, coords, links = ubiquitin
    phi, psi, omega = measure_backbone_torsions(coords, topo, links)
    geometry = measured_geometry(coords, topo, links)
    bb = backbone_atoms(coords, topo)

    n, ca, c = bb[0, 0], bb[0, 1], bb[0, 2]
    worst = 0.0
    for i in range(coords.shape[0] - 1):
        if not bool(links[i]):
            break
        n, ca, c = extend_cward(
            n, ca, c, psi[i], omega[i], phi[i + 1], _per_residue(geometry, i, i + 1)
        )
        worst = max(worst, float((ca - bb[i + 1, 1]).norm()))
    assert worst < 1e-8, f"the primitive is wrong: {worst:.2e} A"


def test_ideal_geometry_does_not_and_the_error_is_the_lever_arm(ubiquitin):
    """The measurement that shaped the design, pinned as a bound.

    Torsion-space growth on ideal bonds is exact at the first residue and
    metres off down the chain, because an angle error at residue i swings
    everything after it. This is why an alive residue relaxes by a rigid frame
    move rather than by another torsion -- so if this test ever starts passing
    at a tight tolerance, that decision should be revisited rather than
    celebrated.
    """
    topo, coords, links = ubiquitin
    phi, psi, omega = measure_backbone_torsions(coords, topo, links)
    geometry = ideal_geometry(coords.device, coords.dtype)
    bb = backbone_atoms(coords, topo)

    n, ca, c = bb[0, 0], bb[0, 1], bb[0, 2]
    errors = []
    for i in range(coords.shape[0] - 1):
        if not bool(links[i]):
            break
        n, ca, c = extend_cward(n, ca, c, psi[i], omega[i], phi[i + 1], geometry)
        errors.append(float((ca - bb[i + 1, 1]).norm()))

    assert errors[0] < 0.05, "one ideal peptide bond should be close"
    assert errors[9] > 0.5, "ten should not be"
    assert max(errors) > 5.0, "the lever arm is the whole point"


def test_growing_out_and_back_returns_to_the_start(ubiquitin):
    """``extend_nward`` is ``extend_cward`` run backwards, or growth is a prefix."""
    topo, coords, links = ubiquitin
    phi, psi, omega = measure_backbone_torsions(coords, topo, links)
    geometry = ideal_geometry(coords.device, coords.dtype)
    bb = backbone_atoms(coords, topo)

    # Grown twice, not once. Coming back N-ward rebuilds the parent's *own*
    # N-CA and N-CA-C from the geometry it was handed, so starting the round
    # trip at a crystal residue would measure the ideal-versus-measured gap in
    # that residue's internals (about 0.1 A) rather than the inverse property.
    # The first grown residue has ideal internals by construction, so the
    # round trip is exact there or the two functions disagree.
    first = extend_cward(
        bb[5, 0], bb[5, 1], bb[5, 2], psi[5], omega[5], phi[6], geometry
    )
    second = extend_cward(*first, psi[6], omega[6], phi[7], geometry)
    # From the second residue's point of view the same three torsions are its
    # own phi and its parent's omega and psi.
    back = extend_nward(*second, phi[7], omega[6], psi[6], geometry)
    for original, returned in zip(first, back):
        assert float((original - returned).norm()) < 1e-9


def test_a_grown_peptide_bond_has_the_length_it_claims(ubiquitin):
    topo, coords, links = ubiquitin
    phi, psi, omega = measure_backbone_torsions(coords, topo, links)
    geometry = ideal_geometry(coords.device, coords.dtype)
    bb = backbone_atoms(coords, topo)
    n1, _ca1, _c1 = extend_cward(
        bb[5, 0], bb[5, 1], bb[5, 2], psi[5], omega[5], phi[6], geometry
    )
    assert float((n1 - bb[5, 2]).norm()) == pytest.approx(C_N, abs=1e-9)


def test_the_oxygen_follows_psi_and_the_crystal_agrees(ubiquitin):
    """O is not in the backbone's rigid group, and this is the evidence."""
    topo, coords, links = ubiquitin
    _phi, psi, _omega = measure_backbone_torsions(coords, topo, links)
    bb = backbone_atoms(coords, topo)
    index = torch.nonzero(links, as_tuple=False).flatten()
    placed = place_oxygen(bb[index, 0], bb[index, 1], bb[index, 2], psi[index])
    error = (placed - bb[index, 3]).norm(dim=-1)
    # Ideal C-O length and CA-C-O angle against a crystal, so a few hundredths
    # of an Angstrom is the right scale; anything near 1 A would mean the
    # dihedral offset is wrong.
    assert float(error.mean()) < 0.15, float(error.mean())


def test_a_rigid_transfer_keeps_the_residue_rigid(ubiquitin):
    topo, coords, _links = ubiquitin
    bb = backbone_atoms(coords, topo)
    source = (bb[10, 0], bb[10, 1], bb[10, 2])
    target = (bb[40, 0], bb[40, 1], bb[40, 2])
    atoms = coords[10]
    moved = rigid_onto_triad(atoms, source, target)

    before = torch.cdist(atoms, atoms)
    after = torch.cdist(moved, moved)
    assert torch.allclose(before, after, atol=1e-9), "not a rigid motion"
    # CA lands exactly; N and C land to within the two triads' own geometry.
    ca_index = topo.frame_idx[10, 1]
    assert float((moved[ca_index] - target[1]).norm()) < 1e-9


def test_the_measured_torsions_are_a_protein(ubiquitin):
    """omega near planar trans, and phi in the left half, or something is mirrored."""
    topo, coords, links = ubiquitin
    phi, psi, omega = measure_backbone_torsions(coords, topo, links)
    joined = torch.nonzero(links, as_tuple=False).flatten()
    planar = torch.cos(omega[joined])
    assert float(planar.mean()) < -0.95, "omega should be trans"
    negative = (phi[joined + 1] < 0).to(torch.float64).mean()
    assert float(negative) > 0.85, "phi should be negative for an L-protein"
