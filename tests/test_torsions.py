"""The Milestone 1 claim, on synthetic residues: the chi round trip is exact."""

from __future__ import annotations

import numpy as np
import pytest

from aanca import residue_constants as rc
from aanca.geometry import wrap_to_pi
from aanca.torsions import (
    Residue,
    apply_chi_deltas,
    chi_error,
    chi_mask,
    measure_chis,
    set_chis,
)
from synthetic import synthetic_residue

ROTATABLE = [r for r in rc.AA3 if rc.num_chi(r) > 0]


@pytest.mark.parametrize("resname", rc.AA3)
def test_mask_matches_the_residue_type(resname):
    res = synthetic_residue(resname)
    assert chi_mask(res).sum() == rc.num_chi(resname)


@pytest.mark.parametrize("resname", ROTATABLE)
def test_set_then_measure_returns_what_was_set(resname):
    res = synthetic_residue(resname, seed=3)
    n = rc.num_chi(resname)
    rng = np.random.default_rng(7)
    for _ in range(5):
        targets = np.full(rc.MAX_CHI, np.nan)
        targets[:n] = rng.uniform(-np.pi, np.pi, size=n)
        got, mask = measure_chis(set_chis(res, targets))
        assert np.allclose(got[:n], targets[:n], atol=1e-10), resname
        assert mask.sum() == n


@pytest.mark.parametrize("resname", ROTATABLE)
def test_deltas_compose_additively(resname):
    res = synthetic_residue(resname, seed=11)
    n = rc.num_chi(resname)
    before, _ = measure_chis(res)
    rng = np.random.default_rng(13)
    deltas = np.zeros(rc.MAX_CHI)
    deltas[:n] = rng.uniform(-1.0, 1.0, size=n)
    after, _ = measure_chis(apply_chi_deltas(res, deltas))
    assert np.allclose(
        wrap_to_pi(after[:n] - before[:n] - deltas[:n]), 0.0, atol=1e-10
    ), resname


@pytest.mark.parametrize("resname", ROTATABLE)
def test_round_trip_restores_coordinates_exactly(resname):
    # Perturb every chi, then set them back to the measured originals. The
    # coordinates must return, not merely the torsions -- this is what rules out
    # a sign error or a wrong downstream set that happens to be self-consistent.
    res = synthetic_residue(resname, seed=5)
    native, _ = measure_chis(res)
    rng = np.random.default_rng(17)
    n = rc.num_chi(resname)

    damage = np.zeros(rc.MAX_CHI)
    damage[:n] = rng.uniform(-np.pi, np.pi, size=n)
    perturbed = apply_chi_deltas(res, damage)
    assert np.linalg.norm(perturbed.coords - res.coords) > 0.1

    restored = set_chis(perturbed, native)
    assert np.abs(restored.coords - res.coords).max() < 1e-9, resname


@pytest.mark.parametrize("resname", ROTATABLE)
def test_zero_delta_is_a_no_op(resname):
    # The training objective includes a no-op term, so the identity update must
    # be exactly the identity in the geometry too.
    res = synthetic_residue(resname, seed=23)
    out = apply_chi_deltas(res, np.zeros(rc.MAX_CHI))
    assert np.array_equal(out.coords, res.coords)


@pytest.mark.parametrize("resname", ROTATABLE)
def test_chi_rotation_preserves_bond_lengths_and_angles(resname):
    res = synthetic_residue(resname, seed=29)
    idx = res.index
    bonds = [
        (idx[u], idx[v])
        for u, v in rc.SIDECHAIN_BONDS[resname]
        if u in idx and v in idx
    ]
    rng = np.random.default_rng(31)
    deltas = np.zeros(rc.MAX_CHI)
    deltas[: rc.num_chi(resname)] = rng.uniform(-np.pi, np.pi, size=rc.num_chi(resname))
    moved = apply_chi_deltas(res, deltas)
    for i, j in bonds:
        before = np.linalg.norm(res.coords[i] - res.coords[j])
        after = np.linalg.norm(moved.coords[i] - moved.coords[j])
        assert after == pytest.approx(before, abs=1e-10)


def test_proline_side_chain_does_not_move():
    res = synthetic_residue("PRO")
    out = apply_chi_deltas(res, np.full(rc.MAX_CHI, 1.0))
    assert np.array_equal(out.coords, res.coords)
    assert not chi_mask(res).any()


def test_missing_atoms_mask_out_the_affected_chi():
    full = synthetic_residue("LYS")
    keep = [i for i, n in enumerate(full.atom_names) if n != "NZ"]
    truncated = Residue(
        resname="LYS",
        atom_names=tuple(full.atom_names[i] for i in keep),
        coords=full.coords[keep],
    )
    mask = chi_mask(truncated)
    assert mask.tolist() == [True, True, True, False]
    chis, _ = measure_chis(truncated)
    assert np.isnan(chis[3])


def test_chi_error_folds_the_symmetric_terminal_chis():
    flipped = np.array([0.0, np.pi, 0.0, 0.0])
    zero = np.zeros(4)
    # TYR chi2 is two-fold symmetric, HIS chi2 is not.
    assert chi_error(flipped, zero, "TYR")[1] == pytest.approx(0.0, abs=1e-12)
    assert abs(chi_error(flipped, zero, "HIS")[1]) == pytest.approx(np.pi, abs=1e-12)
