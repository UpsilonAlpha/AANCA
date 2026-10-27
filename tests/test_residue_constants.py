"""Invariants on the hand-written tables, so a typo in one shows up here."""

from __future__ import annotations

import pytest

from aanca import residue_constants as rc


def test_twenty_residue_types():
    assert len(rc.AA3) == 20


def test_chi_atoms_and_bond_graph_agree():
    for res in rc.AA3:
        bonded = {n for bond in rc.SIDECHAIN_BONDS[res] for n in bond}
        for i, (a, b, c, d) in enumerate(rc.CHI_ATOMS[res]):
            assert len({a, b, c, d}) == 4, f"{res} chi{i + 1} repeats an atom"
            # The axis and the distal atom must be real bonds of this residue.
            for u, v in ((a, b), (b, c), (c, d)):
                assert (u, v) in rc.SIDECHAIN_BONDS[res] or (
                    v,
                    u,
                ) in rc.SIDECHAIN_BONDS[res] or {u, v} == {"N", "CA"}, (
                    f"{res} chi{i + 1}: {u}-{v} is not a bond"
                )
            assert d in bonded


def test_consecutive_chis_share_their_axis_atoms():
    # chi_(i+1) must hinge on the bond immediately distal to chi_i's, or the
    # outward application order in apply_chi_deltas is wrong.
    for res in rc.AA3:
        atoms = rc.CHI_ATOMS[res]
        for i in range(len(atoms) - 1):
            assert atoms[i][1:] == atoms[i + 1][:3], f"{res} chi{i + 1}/chi{i + 2}"


def test_downstream_sets_contain_the_distal_atom_and_exclude_the_axis():
    for res in rc.AA3:
        if res in rc.RIGID_RESIDUES:
            continue
        for i, (a, b, c, d) in enumerate(rc.CHI_ATOMS[res]):
            moving = rc.CHI_DOWNSTREAM[res][i]
            assert d in moving, f"{res} chi{i + 1} does not move {d}"
            assert not moving & {a, b, c}, f"{res} chi{i + 1} moves its own axis"
            assert not moving & {"N", "CA", "C", "O"}, (
                f"{res} chi{i + 1} moves the backbone"
            )


def test_downstream_sets_are_nested_outward():
    # Rotating chi1 must move everything chi2 moves, and more.
    for res in rc.AA3:
        if res in rc.RIGID_RESIDUES:
            continue
        sets = rc.CHI_DOWNSTREAM[res]
        for i in range(len(sets) - 1):
            assert sets[i + 1] < sets[i], f"{res} chi{i + 2} not nested in chi{i + 1}"


def test_proline_is_detected_as_rigid_by_its_ring_not_by_name():
    assert "PRO" in rc.RIGID_RESIDUES
    assert "ring" in rc.RIGID_RESIDUES["PRO"]
    assert rc.num_chi("PRO") == 0
    # And it is the only one.
    assert set(rc.RIGID_RESIDUES) == {"PRO"}


def test_chi_counts():
    expected = {
        "ALA": 0, "GLY": 0, "PRO": 0,
        "CYS": 1, "SER": 1, "THR": 1, "VAL": 1,
        "ASN": 2, "ASP": 2, "HIS": 2, "ILE": 2, "LEU": 2,
        "PHE": 2, "TRP": 2, "TYR": 2,
        "GLN": 3, "GLU": 3, "MET": 3,
        "ARG": 4, "LYS": 4,
    }
    for res, n in expected.items():
        assert rc.num_chi(res) == n, res


def test_only_the_four_symmetric_terminal_chis_are_periodic():
    periodic = {
        (res, i + 1)
        for res in rc.AA3
        for i, flag in enumerate(rc.chi_periodic_mask(res))
        if flag
    }
    assert periodic == {("ASP", 2), ("GLU", 3), ("PHE", 2), ("TYR", 2)}


@pytest.mark.parametrize("res,expected", [("GLY", 0), ("ALA", 1), ("TRP", 10)])
def test_sidechain_atom_counts(res, expected):
    assert len(rc.sidechain_atoms(res)) == expected
