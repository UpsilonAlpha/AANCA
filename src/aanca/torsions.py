"""Chi extraction and chi modification for a single residue.

The modification path is deliberately *not* a rebuild from ideal internal
coordinates. ``apply_chi_deltas`` rotates the downstream atom set about the chi
bond, starting from the residue's existing coordinates, so:

* the round trip measure -> set -> measure is exact by construction;
* the crystal's own bond lengths and angles survive untouched;
* nothing depends on a rotamer library or an idealised geometry table.

Residues whose chi axis lies in a ring (PRO) are rigid here and carry an
all-false chi mask -- see ``residue_constants.RIGID_RESIDUES``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import residue_constants as rc
from .geometry import angle_difference, dihedral, rotate_about_bond, wrap_to_pi


@dataclass(frozen=True)
class Residue:
    """One residue's heavy atoms. ``coords`` is ``(n_atoms, 3)`` in Angstrom."""

    resname: str
    atom_names: tuple[str, ...]
    coords: np.ndarray

    def __post_init__(self) -> None:
        if self.coords.shape != (len(self.atom_names), 3):
            raise ValueError(
                f"coords {self.coords.shape} does not match "
                f"{len(self.atom_names)} atom names"
            )

    @property
    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.atom_names)}

    def with_coords(self, coords: np.ndarray) -> "Residue":
        return replace(self, coords=np.asarray(coords, dtype=np.float64))


def chi_mask(residue: Residue) -> np.ndarray:
    """``(MAX_CHI,)`` bool array: which chis are both defined and measurable.

    A chi is masked out when the residue type does not have it, when the residue
    is rigid, or when any of the four defining atoms is missing from the model
    (disordered side chains in crystal structures are common).
    """
    mask = np.zeros(rc.MAX_CHI, dtype=bool)
    n = rc.num_chi(residue.resname)
    if n == 0:
        return mask
    present = set(residue.atom_names)
    for i in range(n):
        mask[i] = all(a in present for a in rc.CHI_ATOMS[residue.resname][i])
    return mask


def measure_chis(residue: Residue) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(chis, mask)``: chi1..chi4 in radians, NaN where masked out."""
    mask = chi_mask(residue)
    chis = np.full(rc.MAX_CHI, np.nan, dtype=np.float64)
    if not mask.any():
        return chis, mask
    idx = residue.index
    for i in np.flatnonzero(mask):
        a, b, c, d = rc.CHI_ATOMS[residue.resname][i]
        chis[i] = dihedral(
            residue.coords[idx[a]],
            residue.coords[idx[b]],
            residue.coords[idx[c]],
            residue.coords[idx[d]],
        )
    return chis, mask


def apply_chi_deltas(residue: Residue, deltas: np.ndarray) -> Residue:
    """Rotate the side chain by ``deltas`` (radians, ``(MAX_CHI,)``).

    This is the model's output path: a residual update to the residue's current
    torsions. Entries that are NaN, zero, or outside the residue's chi count are
    skipped. Chis are applied chi1 outward; because each step is a rigid rotation
    of everything downstream, the later chis' values are unaffected by the
    earlier rotations.
    """
    deltas = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if deltas.size != rc.MAX_CHI:
        raise ValueError(f"expected {rc.MAX_CHI} deltas, got {deltas.size}")

    mask = chi_mask(residue)
    coords = residue.coords.copy()
    idx = residue.index

    for i in np.flatnonzero(mask):
        theta = deltas[i]
        if not np.isfinite(theta) or theta == 0.0:
            continue
        _a, b, c, _d = rc.CHI_ATOMS[residue.resname][i]
        moving = [
            idx[name]
            for name in rc.CHI_DOWNSTREAM[residue.resname][i]
            if name in idx
        ]
        if not moving:
            continue
        pb, pc = coords[idx[b]], coords[idx[c]]
        coords[moving] = rotate_about_bond(
            coords[moving], origin=pc, direction=pc - pb, theta=float(theta)
        )
    return residue.with_coords(coords)


def set_chis(residue: Residue, targets: np.ndarray) -> Residue:
    """Set chi1..chi4 to ``targets`` (radians). NaN entries are left alone."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    current, mask = measure_chis(residue)
    deltas = np.zeros(rc.MAX_CHI, dtype=np.float64)
    for i in np.flatnonzero(mask):
        if np.isfinite(targets[i]):
            # Literal difference, not the symmetry-folded one: set_chis means
            # what it says, and folding here would silently refuse 180-degree
            # moves on the symmetric chis.
            deltas[i] = wrap_to_pi(targets[i] - current[i])
    return apply_chi_deltas(residue, deltas)


def chi_error(
    pred: np.ndarray, target: np.ndarray, resname: str
) -> np.ndarray:
    """Signed per-chi error, with two-fold symmetric chis folded.

    Use this for both the loss and the reported MAE. Using a plain difference
    instead costs up to 180 degrees of error on ASP chi2, GLU chi3, PHE chi2 and
    TYR chi2, and the resulting numbers look plausible.
    """
    periodic = np.zeros(rc.MAX_CHI, dtype=bool)
    flags = rc.chi_periodic_mask(resname)
    periodic[: len(flags)] = flags
    return angle_difference(pred, target, periodic=periodic)
