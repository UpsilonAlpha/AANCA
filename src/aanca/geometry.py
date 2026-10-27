"""Vectorised dihedral measurement and rigid rotation about an arbitrary axis.

Sign convention, fixed here and relied on everywhere else: ``dihedral`` returns
the IUPAC signed torsion, and rotating the atoms downstream of ``p2`` by
``+theta`` about the directed axis ``p1 -> p2`` increases that dihedral by
``+theta``. ``tests/test_geometry.py`` asserts it rather than trusting it.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def dihedral(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
    """Signed dihedral in radians, in ``(-pi, pi]``. Broadcasts over leading axes."""
    b0 = np.asarray(p0, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    b1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    b2 = np.asarray(p3, dtype=np.float64) - np.asarray(p2, dtype=np.float64)

    b1 = _unit(b1)
    # Components of b0 and b2 perpendicular to the axis.
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def rotation_matrix(axis: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    """Right-handed rotation about ``axis`` by ``theta`` radians (Rodrigues)."""
    k = _unit(np.asarray(axis, dtype=np.float64))
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [
            np.stack([zero, -kz, ky], axis=-1),
            np.stack([kz, zero, -kx], axis=-1),
            np.stack([-ky, kx, zero], axis=-1),
        ],
        axis=-2,
    )
    theta = np.asarray(theta, dtype=np.float64)[..., None, None]
    eye = np.eye(3, dtype=np.float64)
    return eye + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotate_about_bond(
    coords: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Rotate ``coords`` by ``theta`` about the line through ``origin`` along ``direction``.

    This is the only operation that moves atoms. Because it is a rigid rotation
    of a whole downstream atom set about the bond it hinges on, bond lengths and
    bond angles -- including the crystal's own slightly non-ideal ones -- are
    preserved exactly. That is what makes the chi round-trip exact by
    construction instead of exact to within a rebuild tolerance.
    """
    coords = np.asarray(coords, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    R = rotation_matrix(direction, theta)
    return (coords - origin) @ R.T + origin


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    """Map angles into ``(-pi, pi]``."""
    return -(-(np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi)) + np.pi


def angle_difference(
    a: np.ndarray | float,
    b: np.ndarray | float,
    periodic: np.ndarray | bool = False,
) -> np.ndarray:
    """Shortest signed difference ``a - b``, optionally modulo 180 degrees.

    ``periodic`` selects the two-fold symmetric chis (ASP chi2, GLU chi3, PHE
    chi2, TYR chi2), for which a 180-degree flip is the same structure. The
    result for those is wrapped into ``(-pi/2, pi/2]``.
    """
    diff = wrap_to_pi(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    periodic = np.asarray(periodic, dtype=bool)
    if not periodic.any():
        return diff
    folded = wrap_to_pi(2.0 * diff) / 2.0
    return np.where(periodic, folded, diff)
