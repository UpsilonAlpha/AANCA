"""The Torch torsion path must match the verified NumPy reference, and differentiate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aanca import residue_constants as rc
from aanca.structure import load_protein
from aanca.torch_geometry import (
    MAX_ATOMS,
    apply_chi_deltas,
    chi_residual,
    measure_chis,
    rotation_matrices,
    set_chis,
    topology_from_protein,
    wrap_to_pi,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module", params=["193L", "1UBQ"])
def loaded(request):
    path = DATA / f"{request.param}.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    protein = load_protein(path)
    topo, coords = topology_from_protein(protein)
    return protein, topo, coords


def test_topology_matches_the_numpy_masks(loaded):
    protein, topo, _coords = loaded
    assert np.array_equal(topo.chi_mask.numpy(), protein.chi_mask())
    assert topo.n_residues == len(protein)
    assert topo.atom_mask.shape[1] == MAX_ATOMS


def test_measured_chis_match_the_reference(loaded):
    protein, topo, coords = loaded
    reference, mask = protein.chis()
    got = measure_chis(coords, topo).numpy()
    assert np.allclose(got[mask], reference[mask], atol=1e-10)


def test_masked_chis_read_zero_not_nan(loaded):
    _protein, topo, coords = loaded
    got = measure_chis(coords, topo)
    assert torch.isfinite(got).all(), "a NaN here would poison a whole batch"
    assert (got[~topo.chi_mask] == 0.0).all()


def test_applying_deltas_matches_the_reference(loaded):
    protein, topo, coords = loaded
    rng = np.random.default_rng(0)
    mask = protein.chi_mask()
    deltas = np.where(mask, rng.uniform(-np.pi, np.pi, size=mask.shape), 0.0)

    expected = protein.apply_chi_deltas(deltas)
    got = apply_chi_deltas(coords, torch.from_numpy(deltas), topo)

    for i, res in enumerate(expected.residues):
        n = len(res.atom_names)
        assert np.allclose(got[i, :n].numpy(), res.coords, atol=1e-9), i


def test_round_trip_is_exact(loaded):
    """The Milestone 1 guarantee, carried into the training path."""
    _protein, topo, coords = loaded
    native = measure_chis(coords, topo)
    torch.manual_seed(0)
    damage = torch.where(
        topo.chi_mask,
        torch.rand(topo.chi_mask.shape, dtype=torch.float64) * 2 * torch.pi - torch.pi,
        torch.zeros(topo.chi_mask.shape, dtype=torch.float64),
    )
    damaged = apply_chi_deltas(coords, damage, topo)
    assert (damaged - coords).abs().max() > 1.0

    restored = set_chis(damaged, native, topo)
    deviation = float((restored - coords).abs().max())
    assert deviation < 1e-9, f"round trip off by {deviation:.2e} A"


def test_zero_delta_is_a_no_op(loaded):
    _protein, topo, coords = loaded
    out = apply_chi_deltas(coords, torch.zeros_like(topo.chi_mask, dtype=coords.dtype), topo)
    assert torch.equal(out, coords)


def test_backbone_never_moves(loaded):
    protein, topo, coords = loaded
    torch.manual_seed(1)
    deltas = torch.where(
        topo.chi_mask, torch.randn(topo.chi_mask.shape, dtype=torch.float64), 0.0
    )
    moved = apply_chi_deltas(coords, deltas, topo)
    for i, res in enumerate(protein.residues):
        for name in ("N", "CA", "C", "O"):
            if name in res.index:
                j = res.index[name]
                assert torch.equal(moved[i, j], coords[i, j])


def test_proline_is_rigid(loaded):
    protein, topo, coords = loaded
    pro = [i for i, r in enumerate(protein.residues) if r.resname == "PRO"]
    if not pro:
        pytest.skip("no proline")
    assert not topo.chi_mask[pro].any()
    moved = apply_chi_deltas(coords, torch.ones_like(coords[:, :rc.MAX_CHI, 0]), topo)
    for i in pro:
        assert torch.equal(moved[i], coords[i])


def test_bond_lengths_survive(loaded):
    protein, topo, coords = loaded
    torch.manual_seed(2)
    deltas = torch.where(
        topo.chi_mask, torch.randn(topo.chi_mask.shape, dtype=torch.float64), 0.0
    )
    moved = apply_chi_deltas(coords, deltas, topo)
    worst = 0.0
    for i, res in enumerate(protein.residues):
        idx = res.index
        for u, v in rc.SIDECHAIN_BONDS[res.resname]:
            if u in idx and v in idx:
                a, b = idx[u], idx[v]
                d0 = float((coords[i, a] - coords[i, b]).norm())
                d1 = float((moved[i, a] - moved[i, b]).norm())
                worst = max(worst, abs(d1 - d0))
    assert worst < 1e-9


# -- differentiability -----------------------------------------------------


def test_gradients_flow_to_the_deltas(loaded):
    _protein, topo, coords = loaded
    deltas = torch.zeros(topo.n_residues, rc.MAX_CHI, dtype=torch.float64, requires_grad=True)
    out = apply_chi_deltas(coords, deltas, topo)
    out.sum().backward()
    assert deltas.grad is not None
    assert torch.isfinite(deltas.grad).all()
    # Every real chi must receive gradient; a zero would mean a chi that cannot
    # be trained because its rotation reaches no atom.
    assert (deltas.grad[topo.chi_mask].abs() > 0).all()


def test_gradients_flow_through_the_incoming_coordinates(loaded):
    """A rollout's step t depends on step t-1, so this path must be live."""
    _protein, topo, coords = loaded
    start = coords.clone().requires_grad_(True)
    deltas = torch.where(
        topo.chi_mask, torch.full(topo.chi_mask.shape, 0.3, dtype=torch.float64), 0.0
    )
    measure_chis(apply_chi_deltas(start, deltas, topo), topo).sum().backward()
    assert start.grad is not None
    assert torch.isfinite(start.grad).all()
    assert float(start.grad.abs().sum()) > 0.0


def test_delta_then_measure_is_additive(loaded):
    _protein, topo, coords = loaded
    before = measure_chis(coords, topo)
    torch.manual_seed(3)
    deltas = torch.where(
        topo.chi_mask,
        torch.rand(topo.chi_mask.shape, dtype=torch.float64) - 0.5,
        torch.zeros(topo.chi_mask.shape, dtype=torch.float64),
    )
    after = measure_chis(apply_chi_deltas(coords, deltas, topo), topo)
    residual = wrap_to_pi(after - before - deltas)
    assert float(residual[topo.chi_mask].abs().max()) < 1e-10


# -- conventions -----------------------------------------------------------


def test_rotation_matrices_are_rotations():
    torch.manual_seed(4)
    axis = torch.randn(16, 3, dtype=torch.float64)
    theta = torch.rand(16, dtype=torch.float64) * 4 - 2
    R = rotation_matrices(axis, theta)
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(16, dtype=torch.float64))


def test_chi_residual_folds_only_periodic_entries():
    pred = torch.tensor([[torch.pi, torch.pi, 0.0, 0.0]])
    target = torch.zeros(1, 4)
    periodic = torch.tensor([[False, True, False, False]])
    out = chi_residual(pred, target, periodic)
    assert abs(float(out[0, 0])) == pytest.approx(torch.pi, abs=1e-6)
    assert float(out[0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_wrap_to_pi_is_half_open():
    got = wrap_to_pi(torch.tensor([torch.pi, -torch.pi, 3 * torch.pi, 0.5]))
    assert got[0] == pytest.approx(torch.pi, abs=1e-6)
    assert got[1] == pytest.approx(torch.pi, abs=1e-6)
    assert got[2] == pytest.approx(torch.pi, abs=1e-6)
    assert got[3] == pytest.approx(0.5, abs=1e-6)
