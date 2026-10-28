"""Differentiable torsion geometry in Torch.

The NumPy path in ``torsions.py`` is the verified reference; this is the training
path. It must agree with that reference to float tolerance, and
``tests/test_torch_geometry.py`` checks that it does.

Two things make this a port rather than a rewrite:

**The topology is static.** Which atoms exist, which four define each chi, and
which atoms a chi rotation moves all depend only on the sequence, never on the
coordinates. So they are resolved once into index tensors
(:class:`TorsionTopology`) and the per-step work is pure tensor arithmetic with
no Python dict lookups.

**Gradients must flow through the coordinates.** A rollout applies the rule
many times, and step *t*'s features are computed from coordinates that depend on
step *t-1*'s outputs. So ``apply_chi_deltas`` has to be differentiable with
respect to both the deltas and the incoming coordinates. Neighbour *selection*
does not and cannot be -- it is an argsort -- which is why the graph carries a
smooth envelope weight for gradients to flow through instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from . import residue_constants as rc
from .structure import Protein

EPS = 1e-8

#: Heavy-atom count of the largest residue (TRP: N CA C O CB CG CD1 CD2 NE1
#: CE2 CE3 CZ2 CZ3 CH2). Padding width for the atom axis.
MAX_ATOMS = 14


@dataclass(frozen=True)
class TorsionTopology:
    """Static per-residue index tensors. Depends on sequence, not coordinates.

    ``atom_mask`` marks real atoms in the padded ``(n_res, MAX_ATOMS)`` layout.
    ``chi_atoms[r, i]`` holds the four atom indices defining chi ``i+1``, and
    ``downstream[r, i]`` marks the atoms that chi rotation moves. Padded entries
    are zero-filled and guarded by ``chi_mask``.
    """

    atom_mask: Tensor
    chi_mask: Tensor
    chi_atoms: Tensor
    downstream: Tensor
    periodic: Tensor
    resnames: tuple[str, ...]
    atom_names: tuple[tuple[str, ...], ...]

    @property
    def n_residues(self) -> int:
        return self.atom_mask.shape[0]

    def to(self, device: torch.device | str) -> "TorsionTopology":
        return TorsionTopology(
            atom_mask=self.atom_mask.to(device),
            chi_mask=self.chi_mask.to(device),
            chi_atoms=self.chi_atoms.to(device),
            downstream=self.downstream.to(device),
            periodic=self.periodic.to(device),
            resnames=self.resnames,
            atom_names=self.atom_names,
        )


def topology_from_protein(
    protein: Protein, device: torch.device | str = "cpu"
) -> tuple[TorsionTopology, Tensor]:
    """Resolve ``protein`` into a static topology plus a padded coordinate tensor.

    Returns ``(topology, coords)`` where ``coords`` is ``(n_res, MAX_ATOMS, 3)``
    float64. Double precision deliberately: the exactness guarantee from
    Milestone 1 is a float64 result, and a rollout applies hundreds of rotations.
    Cast to float32 at the network boundary, not here.
    """
    n_res = len(protein)
    atom_mask = torch.zeros(n_res, MAX_ATOMS, dtype=torch.bool)
    chi_mask = torch.zeros(n_res, rc.MAX_CHI, dtype=torch.bool)
    chi_atoms = torch.zeros(n_res, rc.MAX_CHI, 4, dtype=torch.long)
    downstream = torch.zeros(n_res, rc.MAX_CHI, MAX_ATOMS, dtype=torch.bool)
    periodic = torch.zeros(n_res, rc.MAX_CHI, dtype=torch.bool)
    coords = torch.zeros(n_res, MAX_ATOMS, 3, dtype=torch.float64)

    for r, res in enumerate(protein.residues):
        if len(res.atom_names) > MAX_ATOMS:
            raise ValueError(
                f"residue {r} ({res.resname}) has {len(res.atom_names)} atoms, "
                f"more than MAX_ATOMS={MAX_ATOMS}"
            )
        n_atoms = len(res.atom_names)
        atom_mask[r, :n_atoms] = True
        coords[r, :n_atoms] = torch.from_numpy(np.ascontiguousarray(res.coords))

        index = res.index
        flags = rc.chi_periodic_mask(res.resname)
        for i in range(rc.num_chi(res.resname)):
            names = rc.CHI_ATOMS[res.resname][i]
            if not all(a in index for a in names):
                continue
            chi_mask[r, i] = True
            chi_atoms[r, i] = torch.tensor([index[a] for a in names])
            moving = [
                index[n] for n in rc.CHI_DOWNSTREAM[res.resname][i] if n in index
            ]
            downstream[r, i, moving] = True
            periodic[r, i] = flags[i] if i < len(flags) else False

    topology = TorsionTopology(
        atom_mask=atom_mask,
        chi_mask=chi_mask,
        chi_atoms=chi_atoms,
        downstream=downstream,
        periodic=periodic,
        resnames=tuple(r.resname for r in protein.residues),
        atom_names=tuple(r.atom_names for r in protein.residues),
    )
    return topology.to(device), coords.to(device)


def _gather_atoms(coords: Tensor, index: Tensor) -> Tensor:
    """``coords`` (R, A, 3) at per-residue atom indices ``index`` (R, ...) -> (R, ..., 3)."""
    flat = index.reshape(index.shape[0], -1)
    rows = torch.arange(coords.shape[0], device=coords.device).unsqueeze(-1)
    picked = coords[rows, flat]
    return picked.reshape(*index.shape, 3)


def dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Signed IUPAC dihedral in radians. Broadcasts over leading axes.

    Same convention as ``geometry.dihedral``: rotating the atoms downstream of
    ``p2`` by ``+theta`` about the directed axis ``p1 -> p2`` adds ``+theta``.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(EPS)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x = (v * w).sum(-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


def measure_chis(coords: Tensor, topo: TorsionTopology) -> Tensor:
    """``(n_res, MAX_CHI)`` torsions in radians. Masked-out entries read zero.

    Zero rather than NaN, unlike the NumPy reference: a NaN here would poison
    gradients for every residue sharing a batch. Always read alongside
    ``topo.chi_mask``.
    """
    pts = _gather_atoms(coords, topo.chi_atoms)
    chis = dihedral(pts[..., 0, :], pts[..., 1, :], pts[..., 2, :], pts[..., 3, :])
    return torch.where(topo.chi_mask, chis, torch.zeros_like(chis))


def rotation_matrices(axis: Tensor, theta: Tensor) -> Tensor:
    """Right-handed rotations about ``axis`` (..., 3) by ``theta`` (...) -> (..., 3, 3)."""
    k = axis / axis.norm(dim=-1, keepdim=True).clamp_min(EPS)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    rows = [
        torch.stack([zero, -kz, ky], dim=-1),
        torch.stack([kz, zero, -kx], dim=-1),
        torch.stack([-ky, kx, zero], dim=-1),
    ]
    K = torch.stack(rows, dim=-2)
    theta = theta.unsqueeze(-1).unsqueeze(-1)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def apply_chi_deltas(
    coords: Tensor, deltas: Tensor, topo: TorsionTopology
) -> Tensor:
    """Rotate side chains by ``deltas`` ``(n_res, MAX_CHI)`` radians.

    Chis are applied chi1 outward. Each is a rigid rotation of everything
    downstream, so an earlier rotation carries the later chis' axes with it and
    leaves their values unchanged -- which is what makes the composition
    additive and the inverse exact.

    Differentiable in both ``coords`` and ``deltas``.
    """
    if deltas.shape != (topo.n_residues, rc.MAX_CHI):
        raise ValueError(
            f"expected deltas of shape {(topo.n_residues, rc.MAX_CHI)}, "
            f"got {tuple(deltas.shape)}"
        )
    deltas = torch.where(topo.chi_mask, deltas, torch.zeros_like(deltas))

    for i in range(rc.MAX_CHI):
        moving = topo.downstream[:, i, :]
        if not bool(moving.any()):
            continue
        pb = _gather_atoms(coords, topo.chi_atoms[:, i, 1:2])[:, 0, :]
        pc = _gather_atoms(coords, topo.chi_atoms[:, i, 2:3])[:, 0, :]
        R = rotation_matrices(pc - pb, deltas[:, i])
        origin = pc.unsqueeze(1)
        rotated = (coords - origin) @ R.transpose(-1, -2) + origin
        coords = torch.where(moving.unsqueeze(-1), rotated, coords)
    return coords


def set_chis(coords: Tensor, targets: Tensor, topo: TorsionTopology) -> Tensor:
    """Set chis to ``targets``. The literal difference, not the folded one."""
    current = measure_chis(coords, topo)
    return apply_chi_deltas(coords, wrap_to_pi(targets - current), topo)


def wrap_to_pi(angle: Tensor) -> Tensor:
    """Map angles into ``(-pi, pi]``."""
    return -torch.remainder(-angle + torch.pi, 2 * torch.pi) + torch.pi


def chi_residual(
    pred: Tensor, target: Tensor, periodic: Tensor | None = None
) -> Tensor:
    """Shortest signed difference ``pred - target``, folding two-fold chis.

    For the symmetric terminal chis (ASP chi2, GLU chi3, PHE chi2, TYR chi2) a
    180-degree flip is the same structure, so the result is folded into
    ``(-pi/2, pi/2]``. Use this for reported MAE. The *loss* handles the
    symmetry as a proper two-component mixture instead -- see ``losses.py``.
    """
    diff = wrap_to_pi(pred - target)
    if periodic is None:
        return diff
    folded = wrap_to_pi(2.0 * diff) / 2.0
    return torch.where(periodic, folded, diff)
