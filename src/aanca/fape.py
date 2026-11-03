"""Frame-aligned point error, which is the loss a grown structure needs.

Every loss in this project so far is on an *internal* coordinate -- a chi, a
bond, an angle -- and that works because the backbone is given, so a residue's
internal coordinates determine where its atoms are. Growth breaks that: the
backbone is the thing being built, and a rule that gets every torsion right to
a degree can still put residue 75 thirty Angstrom away (measured; see
``aanca.growth``). A loss on torsions cannot see that error, and a loss on raw
coordinates cannot be computed at all without first choosing a global
superposition.

FAPE is the standard answer and it is AF2's (Jumper et al. 2021, eq. 10): score
each atom in *every* residue's local frame, and average. It is invariant to a
global rigid motion by construction, needs no alignment step, and is sensitive
to exactly the error torsions hide -- if the fold is wrong, distant pairs are
wrong in each other's frames even when every local geometry is perfect.

Taken rather than invented, deliberately: the brief on this project is to copy
the field's benchmarks rather than reinvent them, and a growth model that
reported a bespoke error measure could not be compared with anything.

**The clamp is part of the definition, not a safeguard.** Unclamped, a single
badly placed segment dominates the average and the gradient points at it to the
exclusion of everything else. AF2 clamps at 10 A; the clamp is also why FAPE
says nothing about *how* wrong a very wrong structure is, which is why the
probe reports unclamped RMSD alongside it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .backbone import frame_from_triad
from .torch_graph import FeatureTopology

#: AF2's clamp, Angstrom.
CLAMP = 10.0
#: AF2's length scale, Angstrom. Divides the error so the loss is O(1).
SCALE = 10.0
EPS = 1e-8


def fape(
    coords: Tensor,
    native: Tensor,
    topo: FeatureTopology,
    mask: Tensor,
    atoms: Tensor | None = None,
    clamp: float = CLAMP,
    scale: float = SCALE,
) -> Tensor:
    """Mean frame-aligned error over residue pairs in ``mask``.

    ``mask`` is ``(R,)``: which residues both supply a frame and have their
    atoms scored. ``atoms`` is ``(R, A)`` bool, defaulting to the backbone's
    N, CA, C -- the three atoms that define a frame, which keeps this a measure
    of the *fold* and leaves side chains to the chi loss that already handles
    them.

    Both structures must be in the same slot order, which they are: growth
    pre-allocates the slots, so residue ``i`` of the grown structure is residue
    ``i`` of the native one and no correspondence has to be guessed.
    """
    index = torch.nonzero(mask, as_tuple=False).flatten()
    if index.numel() < 2:
        return torch.zeros((), device=coords.device, dtype=coords.dtype)

    rot, origin = _frames(coords, topo, index)
    rot_native, origin_native = _frames(native, topo, index)

    if atoms is None:
        points = _backbone_points(coords, topo, index)
        points_native = _backbone_points(native, topo, index)
        flat = points.reshape(-1, 3)
        flat_native = points_native.reshape(-1, 3)
    else:
        keep = atoms[index]
        flat = coords[index][keep]
        flat_native = native[index][keep]

    local = _to_local(rot, origin, flat)
    local_native = _to_local(rot_native, origin_native, flat_native)
    error = (local - local_native).norm(dim=-1).clamp_max(clamp)
    return error.mean() / scale


def _frames(coords: Tensor, topo: FeatureTopology, index: Tensor):
    rows = index.unsqueeze(-1)
    picked = coords[rows, topo.frame_idx[index, :3]]
    rot = frame_from_triad(picked[:, 0], picked[:, 1], picked[:, 2])
    return rot, picked[:, 1]


def _backbone_points(coords: Tensor, topo: FeatureTopology, index: Tensor) -> Tensor:
    rows = index.unsqueeze(-1)
    return coords[rows, topo.frame_idx[index, :3]]


def _to_local(rot: Tensor, origin: Tensor, points: Tensor) -> Tensor:
    """``(F, P, 3)``: every point expressed in every frame."""
    offset = points.unsqueeze(0) - origin.unsqueeze(1)
    return offset @ rot


def backbone_rmsd(
    coords: Tensor, native: Tensor, topo: FeatureTopology, mask: Tensor
) -> Tensor:
    """CA RMSD after an optimal superposition, for reporting only.

    FAPE is the loss; this is the number a reader of a paper recognises. It is
    unclamped on purpose -- the point of reporting both is that FAPE saturates
    and this does not.
    """
    index = torch.nonzero(mask, as_tuple=False).flatten()
    if index.numel() < 3:
        return torch.full((), float("nan"), device=coords.device, dtype=coords.dtype)
    rows = index.unsqueeze(-1)
    a = coords[rows, topo.frame_idx[index, 1:2]][:, 0]
    b = native[rows, topo.frame_idx[index, 1:2]][:, 0]
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    u, _s, vh = torch.linalg.svd(a.transpose(0, 1) @ b)
    d = torch.sign(torch.det(u @ vh))
    correction = torch.diag(
        torch.tensor([1.0, 1.0, 1.0], device=coords.device, dtype=coords.dtype)
    ).clone()
    correction[2, 2] = d
    rot = u @ correction @ vh
    aligned = a @ rot
    return ((aligned - b) ** 2).sum(dim=-1).mean().sqrt()
