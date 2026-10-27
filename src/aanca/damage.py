"""The geometric damage operator, and the chemical one's interface.

These are kept separate on purpose. They are different objectives with
different supervision, and merging them into one loss term should be a
deliberate decision rather than an accident of the code layout:

* **Geometric damage** perturbs side-chain torsions. The native packing is the
  exact ground truth, so this trains with a reconstruction loss. It is the
  operator the headline basin-of-attraction figure sweeps over.
* **Chemical damage** substitutes a residue's identity. There is no ground-truth
  structure, so it cannot be trained the same way and is evaluated against
  measured ddG instead. Only the perturbation side lives here; the readout does
  not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import residue_constants as rc
from .structure import Protein


@dataclass(frozen=True)
class GeometricDamage:
    """A reproducible side-chain perturbation and its exact inverse.

    ``deltas`` is ``(n_res, MAX_CHI)`` in radians, zero wherever a chi does not
    exist. Because every perturbation is a rigid rotation, ``-deltas`` restores
    the original coordinates to double-precision noise -- which is what makes
    this a usable training target rather than an approximation.
    """

    deltas: np.ndarray
    kind: str
    magnitude: float

    def apply(self, protein: Protein) -> Protein:
        return protein.apply_chi_deltas(self.deltas)

    def invert(self) -> "GeometricDamage":
        return GeometricDamage(
            deltas=-self.deltas, kind=f"inverse({self.kind})", magnitude=self.magnitude
        )


def _von_mises_deltas(
    mask: np.ndarray, concentration: float, rng: np.random.Generator
) -> np.ndarray:
    """Circular noise. Large concentration is a small perturbation."""
    noise = rng.vonmises(mu=0.0, kappa=concentration, size=mask.shape)
    return np.where(mask, noise, 0.0)


def torsion_noise(
    protein: Protein,
    concentration: float = 20.0,
    rng: np.random.Generator | None = None,
) -> GeometricDamage:
    """Mild damage: von Mises noise on every measurable chi.

    ``concentration`` is the von Mises kappa. Roughly, kappa 100 is a few
    degrees, 20 is ~13 degrees, 2 is most of a rotamer well, and 0 is uniform.
    Sweeping it is how the basin-of-attraction curve gets its x axis.
    """
    rng = np.random.default_rng() if rng is None else rng
    mask = protein.chi_mask()
    return GeometricDamage(
        deltas=_von_mises_deltas(mask, concentration, rng),
        kind="torsion_noise",
        magnitude=float(concentration),
    )


def randomise_rotamers(
    protein: Protein,
    fraction: float = 1.0,
    rng: np.random.Generator | None = None,
) -> GeometricDamage:
    """Maximal damage: replace the chis of a random residue subset uniformly.

    This is the far end of the perturbation sweep -- the chis carry no
    information about the native state at all, so recovery is pure blind
    denoising from the neighbourhood.
    """
    rng = np.random.default_rng() if rng is None else rng
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    mask = protein.chi_mask()
    chosen = rng.random(len(protein)) < fraction
    mask = mask & chosen[:, None]
    deltas = np.where(mask, rng.uniform(-np.pi, np.pi, size=mask.shape), 0.0)
    return GeometricDamage(
        deltas=deltas, kind="randomise_rotamers", magnitude=float(fraction)
    )


def patch_damage(
    protein: Protein,
    centre: int,
    radius: float = 10.0,
    rng: np.random.Generator | None = None,
) -> GeometricDamage:
    """Randomise the side chains within ``radius`` Angstrom of one residue.

    The protein analogue of cutting a patch out of a Growing-NCA image: the
    chain's connectivity is untouched -- a residue cannot be deleted the way a
    voxel can -- but a contiguous region of packing is destroyed and the rest of
    the structure has to drive its recovery.
    """
    rng = np.random.default_rng() if rng is None else rng
    if not 0 <= centre < len(protein):
        raise IndexError(f"centre {centre} out of range for {len(protein)} residues")

    anchors = np.stack([_anchor(r) for r in protein.residues])
    within = np.linalg.norm(anchors - anchors[centre], axis=-1) <= radius

    mask = protein.chi_mask() & within[:, None]
    deltas = np.where(mask, rng.uniform(-np.pi, np.pi, size=mask.shape), 0.0)
    return GeometricDamage(
        deltas=deltas, kind="patch_damage", magnitude=float(radius)
    )


def _anchor(residue) -> np.ndarray:
    """CB where it exists, else CA -- a stable per-residue reference point."""
    idx = residue.index
    for name in ("CB", "CA"):
        if name in idx:
            return residue.coords[idx[name]]
    return residue.coords.mean(axis=0)


def mutate(protein: Protein, position: int, new_resname: str) -> Protein:
    """Chemical damage -- not implemented, and deliberately left as a stub.

    Substituting a residue identity needs a rule for which atoms survive the
    swap and how the new side chain is seeded, and that choice interacts with
    the ddG readout chosen for tier 3 (strain, stochastic-head entropy, or a
    ProteinEBM energy difference). Implementing it before that decision would
    bake in an answer by accident.
    """
    raise NotImplementedError(
        "chemical damage is pending the tier-3 readout decision; "
        "see HANDOFF.md open question 4"
    )


def chi_mae_degrees(
    pred_chis: np.ndarray, target_chis: np.ndarray, protein: Protein
) -> np.ndarray:
    """Per-chi MAE in degrees, with two-fold symmetric chis folded.

    Reported per chi index rather than pooled, because chi1 accuracy and chi4
    accuracy differ by a lot and a single pooled number hides which one moved.
    """
    from .torsions import chi_error

    mask = protein.chi_mask()
    errors = np.full(mask.shape, np.nan)
    for i, res in enumerate(protein.residues):
        errors[i] = chi_error(pred_chis[i], target_chis[i], res.resname)
    errors = np.where(mask, np.abs(errors), np.nan)
    with np.errstate(invalid="ignore"):
        return np.rad2deg(np.nanmean(errors, axis=0))
