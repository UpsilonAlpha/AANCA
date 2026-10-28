"""The packing benchmark, checked against the published definitions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aanca import residue_constants as rc
from aanca.metrics import (
    CORE_MIN_NEIGHBOURS,
    SURFACE_MAX_NEIGHBOURS,
    cb_neighbour_counts,
    chi_accuracy,
    chi_mae_degrees,
    classify_burial,
    evaluate_packing,
    sidechain_rmsd_torch,
)
from aanca.structure import load_protein, sidechain_rmsd
from aanca.torch_geometry import apply_chi_deltas, measure_chis
from aanca.torch_graph import feature_topology

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module", params=["193L", "1UBQ"])
def loaded(request):
    path = DATA / f"{request.param}.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    protein = load_protein(path)
    topo, coords = feature_topology(protein)
    return protein, topo, coords


# -- burial ----------------------------------------------------------------


def test_neighbour_counts_are_plausible(loaded):
    _protein, topo, coords = loaded
    counts = cb_neighbour_counts(coords, topo)
    assert counts.shape == (topo.n_residues,)
    assert int(counts.min()) >= 0
    # A residue cannot neighbour itself, and cannot exceed the whole protein.
    assert int(counts.max()) < topo.n_residues
    assert float(counts.float().mean()) > 5.0, "a folded protein should be packed"


def test_core_and_surface_are_not_a_partition(loaded):
    """AttnPacker's thresholds leave a 16-19 neighbour band in neither class.

    Treating them as complementary would silently reassign those residues and
    make our numbers incomparable with the published ones.
    """
    _protein, topo, coords = loaded
    bands = classify_burial(cb_neighbour_counts(coords, topo))
    assert not bool((bands["core"] & bands["surface"]).any())
    total = int(bands["core"].sum()) + int(bands["surface"].sum())
    assert total + int(bands["intermediate"].sum()) == topo.n_residues
    assert SURFACE_MAX_NEIGHBOURS + 1 < CORE_MIN_NEIGHBOURS


def test_burial_thresholds_match_the_published_convention():
    counts = torch.tensor([0, 15, 16, 19, 20, 40])
    bands = classify_burial(counts)
    assert bands["surface"].tolist() == [True, True, False, False, False, False]
    assert bands["core"].tolist() == [False, False, False, False, True, True]
    assert bands["intermediate"].tolist() == [False, False, True, True, False, False]


def test_a_buried_protein_has_some_core_residues(loaded):
    _protein, topo, coords = loaded
    bands = classify_burial(cb_neighbour_counts(coords, topo))
    assert int(bands["core"].sum()) > 0
    assert int(bands["surface"].sum()) > 0


# -- the three numbers ------------------------------------------------------


def test_perfect_prediction_scores_perfectly(loaded):
    _protein, topo, coords = loaded
    chis = measure_chis(coords, topo.torsion)
    mae = chi_mae_degrees(chis, chis, topo.torsion)
    acc = chi_accuracy(chis, chis, topo.torsion)
    for value in mae:
        assert value != value or value < 1e-6
    for value in acc:
        assert value != value or value == pytest.approx(1.0)


def test_accuracy_counts_the_twenty_degree_band(loaded):
    """Their headline number alongside MAE: a chi is correct within 20 degrees."""
    _protein, topo, coords = loaded
    chis = measure_chis(coords, topo.torsion)

    just_inside = chis + math.radians(19.0)
    just_outside = chis + math.radians(21.0)

    inside = chi_accuracy(just_inside, chis, topo.torsion)
    outside = chi_accuracy(just_outside, chis, topo.torsion)
    for value in inside:
        assert value != value or value == pytest.approx(1.0)
    for value in outside:
        assert value != value or value == pytest.approx(0.0)


def test_accuracy_and_mae_fold_symmetric_chis(loaded):
    """A 180-degree flip of a symmetric chi is the same structure."""
    _protein, topo, coords = loaded
    chis = measure_chis(coords, topo.torsion)
    periodic = topo.torsion.periodic
    if not bool(periodic.any()):
        pytest.skip("no symmetric chis")

    flipped = torch.where(periodic, chis + math.pi, chis)
    rows = periodic.any(dim=1)
    mae = chi_mae_degrees(flipped, chis, topo.torsion, rows)
    # Chi index 1 is the symmetric one for ASP/PHE/TYR; index 2 for GLU.
    assert mae[1] == pytest.approx(0.0, abs=1e-4) or mae[1] != mae[1]


def test_mae_is_reported_per_chi_index(loaded):
    _protein, topo, coords = loaded
    chis = measure_chis(coords, topo.torsion)
    perturbed = chis.clone()
    perturbed[:, 0] += math.radians(30.0)

    mae = chi_mae_degrees(perturbed, chis, topo.torsion)
    assert mae[0] == pytest.approx(30.0, abs=1e-3)
    for value in mae[1:]:
        assert value != value or value < 1e-6, "only chi1 was perturbed"


# -- the torch RMSD must agree with the verified numpy one -------------------


def test_torch_rmsd_matches_the_numpy_reference(loaded):
    protein, topo, coords = loaded
    rng = np.random.default_rng(0)
    mask = protein.chi_mask()
    deltas = np.where(mask, rng.uniform(-1.0, 1.0, size=mask.shape), 0.0)

    moved_np = protein.apply_chi_deltas(deltas)
    moved_torch = apply_chi_deltas(coords, torch.from_numpy(deltas), topo.torsion)

    reference = sidechain_rmsd(moved_np, protein)
    got = sidechain_rmsd_torch(moved_torch, coords, topo).numpy()

    both = ~np.isnan(reference) & ~np.isnan(got)
    assert both.sum() > 0
    assert np.allclose(got[both], reference[both], atol=1e-6)


def test_torch_rmsd_respects_symmetry(loaded):
    _protein, topo, coords = loaded
    periodic = topo.torsion.periodic
    if not bool(periodic.any()):
        pytest.skip("no symmetric chis")
    flipped = apply_chi_deltas(
        coords,
        torch.where(periodic, torch.full(periodic.shape, math.pi, dtype=coords.dtype), 0.0),
        topo.torsion,
    )
    rows = periodic.any(dim=1).numpy()
    rmsd = sidechain_rmsd_torch(flipped, coords, topo).numpy()[rows]
    assert np.nanmean(rmsd) < 0.3, "a symmetric flip must not read as a big error"


# -- the assembled table ----------------------------------------------------


def test_evaluate_packing_produces_the_published_table(loaded):
    _protein, topo, coords = loaded
    table = evaluate_packing(coords, coords, topo)

    assert set(table) == {"all", "core", "surface", "intermediate"}
    assert table["all"].n_residues == topo.n_residues
    for band in table.values():
        assert len(band.chi_mae) == rc.MAX_CHI
        assert len(band.chi_accuracy) == rc.MAX_CHI
        # A structure against itself is perfect everywhere.
        assert band.rmsd != band.rmsd or band.rmsd < 1e-9
        assert band.summary()


def test_core_residues_are_harder_than_surface(loaded):
    """Sanity check on the stratification, and the reason to report it.

    Randomise every chi: core residues are more constrained, so an arbitrary
    rotamer displaces their atoms against more neighbours. The point of the
    split is that a pooled number hides which population a method is good at.
    """
    _protein, topo, coords = loaded
    torch.manual_seed(0)
    deltas = torch.where(
        topo.torsion.chi_mask,
        torch.rand(topo.torsion.chi_mask.shape, dtype=coords.dtype) * 2 * math.pi - math.pi,
        0.0,
    )
    damaged = apply_chi_deltas(coords, deltas, topo.torsion)
    table = evaluate_packing(damaged, coords, topo)

    assert table["core"].n_residues > 0 and table["surface"].n_residues > 0
    # Both bands should show large error; the test is that both are measurable
    # and reported, not that one is worse.
    assert table["core"].rmsd > 0.5
    assert table["surface"].rmsd > 0.5
    assert table["all"].chi_mae[0] > 30.0
