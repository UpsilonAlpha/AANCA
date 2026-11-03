"""FAPE: zero on itself, blind to rigid motion, and able to see a wrong fold."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from aanca.fape import CLAMP, SCALE, backbone_rmsd, fape
from aanca.structure import load_protein
from aanca.torch_geometry import rotation_matrices
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    return feature_topology(load_protein(path))


def _rigidly_moved(coords: torch.Tensor, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    axis = torch.randn(3, generator=generator, dtype=coords.dtype)
    rot = rotation_matrices(axis, torch.tensor(1.1, dtype=coords.dtype))
    shift = torch.randn(3, generator=generator, dtype=coords.dtype) * 20.0
    flat = coords.reshape(-1, 3) @ rot.transpose(-1, -2) + shift
    return flat.reshape(coords.shape)


def test_fape_is_zero_on_the_structure_itself(ubiquitin):
    topo, coords = ubiquitin
    mask = torch.ones(topo.n_residues, dtype=torch.bool)
    assert float(fape(coords, coords, topo, mask)) == pytest.approx(0.0, abs=1e-9)


def test_fape_cannot_see_a_global_rigid_motion(ubiquitin):
    """The property that makes it usable without an alignment step."""
    topo, coords = ubiquitin
    mask = torch.ones(topo.n_residues, dtype=torch.bool)
    moved = _rigidly_moved(coords)
    assert float((moved - coords).norm(dim=-1).mean()) > 1.0, "the fixture must move"
    assert float(fape(moved, coords, topo, mask)) == pytest.approx(0.0, abs=1e-8)


def test_fape_sees_a_wrong_fold_that_local_geometry_would_hide(ubiquitin):
    """A hinge: every residue keeps its own geometry, the fold is wrong anyway.

    This is the error a torsion loss cannot charge for, and the reason growth
    needs this loss at all -- rotating half the protein about one point leaves
    every bond, angle and torsion inside each residue untouched.
    """
    topo, coords = ubiquitin
    mask = torch.ones(topo.n_residues, dtype=torch.bool)
    half = coords.shape[0] // 2
    hinged = coords.clone()
    pivot = coords[half, 1]
    rot = rotation_matrices(
        torch.tensor([0.0, 0.0, 1.0], dtype=coords.dtype),
        torch.tensor(1.0, dtype=coords.dtype),
    )
    hinged[half:] = (coords[half:] - pivot) @ rot.transpose(-1, -2) + pivot
    assert float(fape(hinged, coords, topo, mask)) > 0.2


def test_fape_saturates_at_the_clamp(ubiquitin):
    """Why the probe also reports RMSD: beyond the clamp this stops counting."""
    topo, coords = ubiquitin
    mask = torch.ones(topo.n_residues, dtype=torch.bool)
    near = coords.clone()
    near[40:] += 40.0
    far = coords.clone()
    far[40:] += 400.0
    assert float(fape(near, coords, topo, mask)) == pytest.approx(
        float(fape(far, coords, topo, mask)), rel=1e-6
    )


def test_fape_scores_only_the_masked_residues(ubiquitin):
    topo, coords = ubiquitin
    mask = torch.zeros(topo.n_residues, dtype=torch.bool)
    mask[:20] = True
    wrecked = coords.clone()
    wrecked[20:] += 30.0
    assert float(fape(wrecked, coords, topo, mask)) == pytest.approx(0.0, abs=1e-9)


def test_fape_needs_two_residues_to_mean_anything(ubiquitin):
    topo, coords = ubiquitin
    mask = torch.zeros(topo.n_residues, dtype=torch.bool)
    mask[3] = True
    assert float(fape(coords, coords, topo, mask)) == 0.0


def test_the_clamp_and_scale_are_af2s(ubiquitin):
    """Pinned so a later tuning pass has to be deliberate about it."""
    assert (CLAMP, SCALE) == (10.0, 10.0)


def test_rmsd_is_zero_on_itself_and_blind_to_rigid_motion(ubiquitin):
    topo, coords = ubiquitin
    mask = torch.ones(topo.n_residues, dtype=torch.bool)
    assert float(backbone_rmsd(coords, coords, topo, mask)) == pytest.approx(
        0.0, abs=1e-6
    )
    moved = _rigidly_moved(coords, seed=3)
    assert float(backbone_rmsd(moved, coords, topo, mask)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_rmsd_reports_a_displacement_it_cannot_superpose_away(ubiquitin):
    topo, coords = ubiquitin
    mask = torch.ones(topo.n_residues, dtype=torch.bool)
    hinged = coords.clone()
    half = coords.shape[0] // 2
    hinged[half:] += 5.0
    value = float(backbone_rmsd(hinged, coords, topo, mask))
    # Half the CAs moved 5*sqrt(3) A; the best superposition splits the
    # difference, so the answer is a few Angstrom rather than zero or nine.
    assert 1.0 < value < 9.0, value


def test_rmsd_refuses_to_superpose_two_points(ubiquitin):
    topo, coords = ubiquitin
    mask = torch.zeros(topo.n_residues, dtype=torch.bool)
    mask[:2] = True
    assert math.isnan(float(backbone_rmsd(coords, coords, topo, mask)))
