"""The sign and wrapping conventions, asserted rather than assumed."""

from __future__ import annotations

import numpy as np
import pytest

from aanca.geometry import (
    angle_difference,
    dihedral,
    rotate_about_bond,
    rotation_matrix,
    wrap_to_pi,
)
from synthetic import place_atom


def test_dihedral_matches_the_torsion_an_atom_was_placed_at():
    # place_atom and dihedral are independent implementations of the same
    # convention; if they agree, both are IUPAC.
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.5, 0.0, 0.0])
    for phi in np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 23):
        d = place_atom(a, b, c, bond=1.5, angle=np.deg2rad(110.0), torsion=phi)
        assert dihedral(a, b, c, d) == pytest.approx(phi, abs=1e-12)


def test_rotating_downstream_atoms_increases_the_dihedral_by_theta():
    # The convention the whole torsion module depends on: +theta about the
    # directed axis b -> c adds +theta to the dihedral.
    a = np.array([0.3, 1.1, -0.2])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.5, 0.1, 0.05])
    d = place_atom(a, b, c, bond=1.5, angle=np.deg2rad(107.0), torsion=0.4)

    for theta in (0.1, -0.7, 2.3, np.pi):
        moved = rotate_about_bond(d, origin=c, direction=c - b, theta=theta)
        got = dihedral(a, b, c, moved)
        assert got == pytest.approx(wrap_to_pi(0.4 + theta), abs=1e-12)


def test_rotation_preserves_distances_to_the_axis_atoms():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(12, 3))
    b, c = rng.normal(size=3), rng.normal(size=3)
    moved = rotate_about_bond(pts, origin=c, direction=c - b, theta=1.234)
    for origin in (b, c):
        before = np.linalg.norm(pts - origin, axis=-1)
        after = np.linalg.norm(moved - origin, axis=-1)
        assert np.allclose(before, after, atol=1e-12)


def test_rotation_matrix_is_a_rotation():
    rng = np.random.default_rng(1)
    for _ in range(10):
        r = rotation_matrix(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_wrap_to_pi_is_half_open_at_pi():
    assert wrap_to_pi(np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(-np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(3 * np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(0.5) == pytest.approx(0.5)


def test_angle_difference_folds_only_the_periodic_entries():
    a = np.array([np.deg2rad(170.0), np.deg2rad(170.0)])
    b = np.array([np.deg2rad(-10.0), np.deg2rad(-10.0)])
    diff = angle_difference(a, b, periodic=np.array([False, True]))
    # 170 and -10 differ by 180 degrees: a full error for a non-symmetric chi,
    # and zero error for a two-fold symmetric one.
    assert abs(diff[0]) == pytest.approx(np.pi, abs=1e-12)
    assert diff[1] == pytest.approx(0.0, abs=1e-12)
