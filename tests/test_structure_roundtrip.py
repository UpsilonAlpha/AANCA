"""The Milestone 1 claim on real crystal structures.

Run ``python scripts/fetch_test_structures.py`` first; these skip without the
files. Synthetic residues check the maths, but only real entries exercise PDB
atom naming, altlocs, disordered side chains and the odd residue types.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aanca import residue_constants as rc
from aanca.structure import load_protein, sidechain_rmsd

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
ENTRIES = ["193L", "3NIR", "1UBQ"]

#: The exactness bar from the milestone plan.
TOLERANCE_A = 0.01


def _load(pdb_id: str):
    path = DATA / f"{pdb_id}.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    return load_protein(path)


@pytest.fixture(scope="module", params=ENTRIES)
def protein(request):
    return _load(request.param)


def test_parses_residues(protein):
    assert len(protein) > 20
    assert all(r.resname in rc.AA3 for r in protein.residues)


def test_chis_are_measured_wherever_the_atoms_are_present(protein):
    chis, mask = protein.chis()
    assert mask.any()
    assert np.isfinite(chis[mask]).all()
    assert np.isnan(chis[~mask]).all()
    # Every chi the residue type defines and the model contains must be measured.
    for i, res in enumerate(protein.residues):
        present = set(res.atom_names)
        for j in range(rc.num_chi(res.resname)):
            if all(a in present for a in rc.CHI_ATOMS[res.resname][j]):
                assert mask[i, j], f"{res.resname}{protein.res_ids[i]} chi{j + 1}"


def test_zero_delta_leaves_every_coordinate_untouched(protein):
    out = protein.apply_chi_deltas(np.zeros((len(protein), rc.MAX_CHI)))
    assert np.array_equal(out.atoms.coord, protein.atoms.coord)


def test_randomise_then_restore_is_exact(protein):
    """Damage every side chain, then set the native chis back.

    This is the test the milestone plan asks for: the geometric damage operator
    and its exact inverse, on real coordinates, under the 0.01 A bar. It would
    catch a sign error, a wrong downstream atom set, or a chi applied in the
    wrong order -- each of which would otherwise surface much later as a model
    that will not converge.
    """
    native, mask = protein.chis()
    rng = np.random.default_rng(0)

    damage = np.where(mask, rng.uniform(-np.pi, np.pi, size=mask.shape), 0.0)
    damaged = protein.apply_chi_deltas(damage)

    moved = np.abs(damaged.atoms.coord - protein.atoms.coord).max()
    assert moved > 1.0, "the damage operator did not move anything"

    damaged_chis, damaged_mask = damaged.chis()
    assert np.array_equal(mask, damaged_mask), "damage changed which chis exist"

    restore = np.where(mask, native - damaged_chis, 0.0)
    restored = damaged.apply_chi_deltas(restore)

    # Measure on the float64 residue coordinates, not on atoms.coord: biotite
    # stores an AtomArray as float32, where a 1e-14 A error rounds to exactly
    # zero and the test would pass for the wrong reason.
    max_dev = max(
        float(np.abs(a.coords - b.coords).max())
        for a, b in zip(restored.residues, protein.residues)
    )
    assert max_dev < TOLERANCE_A, f"round trip off by {max_dev:.2e} A"
    # It should in fact be at double-precision noise, not merely under the bar.
    assert max_dev < 1e-9, f"round trip only accurate to {max_dev:.2e} A"

    rmsd = sidechain_rmsd(restored, protein)
    assert np.nanmax(rmsd, initial=0.0) < TOLERANCE_A


def test_damage_preserves_bond_lengths(protein):
    """A rotation cannot stretch a bond; check it on the real geometry."""
    _chis, mask = protein.chis()
    rng = np.random.default_rng(1)
    damaged = protein.apply_chi_deltas(
        np.where(mask, rng.uniform(-np.pi, np.pi, size=mask.shape), 0.0)
    )
    worst = 0.0
    for before, after in zip(protein.residues, damaged.residues):
        idx = before.index
        for u, v in rc.SIDECHAIN_BONDS[before.resname]:
            if u in idx and v in idx:
                i, j = idx[u], idx[v]
                d0 = np.linalg.norm(before.coords[i] - before.coords[j])
                d1 = np.linalg.norm(after.coords[i] - after.coords[j])
                worst = max(worst, abs(d1 - d0))
    assert worst < 1e-9, f"bond length changed by {worst:.2e} A"


def test_backbone_never_moves(protein):
    _chis, mask = protein.chis()
    rng = np.random.default_rng(2)
    damaged = protein.apply_chi_deltas(
        np.where(mask, rng.uniform(-np.pi, np.pi, size=mask.shape), 0.0)
    )
    bb = np.isin(protein.atoms.atom_name, ["N", "CA", "C", "O"])
    assert np.array_equal(damaged.atoms.coord[bb], protein.atoms.coord[bb])


def test_proline_side_chains_are_never_touched(protein):
    _chis, mask = protein.chis()
    pro = [i for i, r in enumerate(protein.residues) if r.resname == "PRO"]
    if not pro:
        pytest.skip("no proline in this entry")
    assert not mask[pro].any()
    damaged = protein.apply_chi_deltas(np.full((len(protein), rc.MAX_CHI), 1.0))
    for i in pro:
        assert np.array_equal(
            damaged.residues[i].coords, protein.residues[i].coords
        )


def test_rmsd_respects_chemical_symmetry(protein):
    """A 180-degree flip of a symmetric chi is the same structure chemically.

    Comparing by atom name scores it as a ~1.8 A error. Published packer RMSDs
    are 0.4-0.8 A, so the unsymmetrised number would inflate the whole-protein
    mean by 0.26-0.36 A -- comparable to the entire gap between competing
    methods, in exactly the comparison that has to be auditable.
    """
    _chis, mask = protein.chis()
    periodic = np.zeros_like(mask)
    for i, res in enumerate(protein.residues):
        flags = rc.chi_periodic_mask(res.resname)
        periodic[i, : len(flags)] = flags
    periodic &= mask
    if not periodic.any():
        pytest.skip("no symmetric chis in this entry")

    flipped = protein.apply_chi_deltas(np.where(periodic, np.pi, 0.0))
    rows = periodic.any(axis=1)

    naive = sidechain_rmsd(flipped, protein, respect_symmetry=False)[rows]
    aware = sidechain_rmsd(flipped, protein, respect_symmetry=True)[rows]

    assert np.nanmean(naive) > 1.0, "the flip should look large when named naively"
    # Not zero: the residual is real non-ideal geometry, the same source as the
    # approximate feature invariance. But an order of magnitude smaller.
    assert np.nanmean(aware) < 0.3
    assert np.nanmean(aware) < np.nanmean(naive) / 5.0


def test_rmsd_of_a_structure_against_itself_is_zero(protein):
    assert np.allclose(np.nan_to_num(sidechain_rmsd(protein, protein)), 0.0, atol=1e-12)


def test_lysozyme_covers_all_twenty_residue_types():
    protein = _load("193L")
    seen = {r.resname for r in protein.residues}
    assert seen == set(rc.AA3), f"missing: {sorted(set(rc.AA3) - seen)}"
