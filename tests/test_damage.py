"""The geometric damage operator: reproducible, invertible, and graded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aanca import residue_constants as rc
from aanca.damage import (
    chi_mae_degrees,
    mutate,
    patch_damage,
    randomise_rotamers,
    torsion_noise,
)
from aanca.structure import load_protein, sidechain_rmsd

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def protein():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    return load_protein(path)


def _max_dev(a, b) -> float:
    return max(
        float(np.abs(x.coords - y.coords).max()) for x, y in zip(a.residues, b.residues)
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda p, rng: torsion_noise(p, concentration=20.0, rng=rng),
        lambda p, rng: randomise_rotamers(p, rng=rng),
        lambda p, rng: patch_damage(p, centre=10, radius=10.0, rng=rng),
    ],
    ids=["torsion_noise", "randomise_rotamers", "patch_damage"],
)
def test_damage_is_exactly_invertible(protein, factory):
    damage = factory(protein, np.random.default_rng(0))
    damaged = damage.apply(protein)
    restored = damage.invert().apply(damaged)
    assert _max_dev(restored, protein) < 1e-9


@pytest.mark.parametrize(
    "factory",
    [
        lambda p, rng: torsion_noise(p, concentration=20.0, rng=rng),
        lambda p, rng: randomise_rotamers(p, rng=rng),
    ],
    ids=["torsion_noise", "randomise_rotamers"],
)
def test_damage_is_reproducible_from_a_seed(protein, factory):
    a = factory(protein, np.random.default_rng(42)).apply(protein)
    b = factory(protein, np.random.default_rng(42)).apply(protein)
    assert _max_dev(a, b) == 0.0


def test_damage_never_touches_a_chi_that_does_not_exist(protein):
    mask = protein.chi_mask()
    for factory in (torsion_noise, randomise_rotamers):
        damage = factory(protein, rng=np.random.default_rng(1))
        assert np.all(damage.deltas[~mask] == 0.0)


def test_concentration_grades_the_perturbation(protein):
    """The basin-of-attraction sweep needs magnitude to be monotone in kappa."""
    rmsds = []
    for kappa in (200.0, 50.0, 10.0, 2.0):
        damaged = torsion_noise(
            protein, concentration=kappa, rng=np.random.default_rng(0)
        ).apply(protein)
        rmsds.append(float(np.nanmean(sidechain_rmsd(damaged, protein))))
    assert rmsds == sorted(rmsds), rmsds
    assert rmsds[0] < 0.2, "kappa 200 should be a mild perturbation"
    assert rmsds[-1] > 0.8, "kappa 2 should be a severe one"


def test_patch_damage_is_local(protein):
    damage = patch_damage(protein, centre=10, radius=8.0, rng=np.random.default_rng(0))
    damaged = damage.apply(protein)
    rmsd = sidechain_rmsd(damaged, protein)
    touched = np.flatnonzero(np.nan_to_num(rmsd) > 1e-6)
    assert len(touched) > 0, "patch damaged nothing"
    assert len(touched) < len(protein), "patch damaged the whole protein"


def test_fraction_zero_is_a_no_op(protein):
    damage = randomise_rotamers(protein, fraction=0.0, rng=np.random.default_rng(0))
    assert np.all(damage.deltas == 0.0)
    assert _max_dev(damage.apply(protein), protein) == 0.0


def test_patch_centre_is_bounds_checked(protein):
    with pytest.raises(IndexError):
        patch_damage(protein, centre=len(protein))


def test_chemical_damage_is_an_explicit_stub(protein):
    with pytest.raises(NotImplementedError, match="tier-3 readout"):
        mutate(protein, 0, "ALA")


def test_chi_mae_is_zero_for_a_perfect_prediction(protein):
    native, _ = protein.chis()
    mae = chi_mae_degrees(native, native, protein)
    assert np.allclose(np.nan_to_num(mae), 0.0, atol=1e-9)
    assert mae.shape == (rc.MAX_CHI,)


def test_chi_mae_folds_symmetric_chis(protein):
    """Flipping every symmetric terminal chi by 180 must cost nothing."""
    native, mask = protein.chis()
    flipped = native.copy()
    for i, res in enumerate(protein.residues):
        for j, periodic in enumerate(rc.chi_periodic_mask(res.resname)):
            if periodic and mask[i, j]:
                flipped[i, j] += np.pi
    mae = chi_mae_degrees(flipped, native, protein)
    # Chi 2 of ASP/PHE/TYR and chi 3 of GLU are the folded ones; all entries
    # should still read zero because the flip is physically invisible.
    assert np.allclose(np.nan_to_num(mae), 0.0, atol=1e-6), mae
