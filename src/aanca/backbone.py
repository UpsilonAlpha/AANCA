"""Building backbone, rather than reading it.

Everything up to v8 takes the backbone as given: the rule rotates side chains
inside a fixed scaffold, and no code in this project has ever had to *place* a
residue. Growth does, and this is the module that can. It is kept apart from
``torch_geometry`` on purpose -- that module is the verified port of the
torsion machinery and its exactness guarantee is about rotations of existing
atoms, which is a different claim from the one made here.

**One primitive.** :func:`place_atom` is NeRF: given three placed atoms and an
internal coordinate triple (bond, angle, torsion), put the fourth atom down.
Every placement below is three calls to it. It is differentiable in the
coordinates and in all three internal coordinates, which is what makes a
predicted phi/psi a trainable quantity rather than a sampled one.

**Ideal geometry is a stated approximation, not a convenience.** The bond
lengths and angles are Engh & Huber means. Regrowing a real protein from its
own *measured* torsions and these ideal bonds does not reproduce the crystal
backbone exactly, and the error accumulates along the chain -- which is a
property of torsion-space building, not a bug, and the reason
``scripts/growth_probe.py`` measures it rather than assuming it is small.
:func:`measured_geometry` exists so a test can separate the two questions: with
measured bonds and angles the round trip is exact to float tolerance, so a
failure there is the primitive being wrong, and a failure only under ideal
geometry is the approximation being too coarse.

**The torsion names.** A residue's own phi is C(i-1)-N-CA-C and its own psi is
N-CA-C-N(i+1), so both are shared with a neighbour: phi belongs to the peptide
bond on the N side and psi to the one on the C side. Growing C-ward across that
bond therefore sets psi(parent), omega(parent) and phi(child) -- three torsions
per birth, and exactly the three the Ramachandran plot and the planarity of the
peptide are about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .torch_geometry import EPS, dihedral
from .torch_graph import FeatureTopology

#: Engh & Huber ideal bond lengths, Angstrom.
N_CA = 1.458
CA_C = 1.525
C_N = 1.329
C_O = 1.231

#: Engh & Huber ideal bond angles, degrees.
N_CA_C = 111.0
CA_C_N = 116.2
C_N_CA = 121.7
CA_C_O = 120.8


@dataclass(frozen=True)
class BackboneGeometry:
    """The six internal coordinates a peptide bond needs, as tensors.

    Scalars broadcast, so the ideal instance is six zero-dimensional tensors
    and a per-residue measured instance is six ``(R,)`` tensors. Angles are in
    radians here even though the constants above are in degrees, because
    everything downstream is.
    """

    n_ca: Tensor
    ca_c: Tensor
    c_n: Tensor
    n_ca_c: Tensor
    ca_c_n: Tensor
    c_n_ca: Tensor


def ideal_geometry(
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64
) -> BackboneGeometry:
    def scalar(value: float) -> Tensor:
        return torch.tensor(value, device=device, dtype=dtype)

    return BackboneGeometry(
        n_ca=scalar(N_CA),
        ca_c=scalar(CA_C),
        c_n=scalar(C_N),
        n_ca_c=scalar(math.radians(N_CA_C)),
        ca_c_n=scalar(math.radians(CA_C_N)),
        c_n_ca=scalar(math.radians(C_N_CA)),
    )


def place_atom(
    p0: Tensor, p1: Tensor, p2: Tensor, bond: Tensor, angle: Tensor, torsion: Tensor
) -> Tensor:
    """NeRF. Put an atom at ``bond`` from ``p2``, making ``angle`` at ``p2``.

    The dihedral ``p0-p1-p2-X`` of the result is ``torsion``, in the same
    IUPAC sign convention as :func:`torch_geometry.dihedral` -- which
    ``tests/test_backbone.py`` checks by measuring it back, because a sign
    error here is invisible until a grown chain mirrors.

    Broadcasts over leading axes. ``bond``, ``angle`` and ``torsion`` may be
    scalars.
    """
    bc = p2 - p1
    bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(EPS)
    ab = p1 - p0
    # Component of the previous bond perpendicular to the current one. The
    # clamp keeps the direction finite rather than correct when the three
    # atoms are collinear, which does not occur in a peptide.
    normal = torch.cross(ab, bc, dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(EPS)
    inplane = torch.cross(normal, bc, dim=-1)

    # The local basis is (-bc, inplane, normal): the angle opens away from the
    # incoming bond and the torsion rotates about it.
    along = -bond * torch.cos(angle)
    across = bond * torch.sin(angle) * torch.cos(torsion)
    out = bond * torch.sin(angle) * torch.sin(torsion)
    offset = (
        along.unsqueeze(-1) * bc
        + across.unsqueeze(-1) * inplane
        + out.unsqueeze(-1) * normal
    )
    return p2 + offset


def extend_cward(
    n: Tensor,
    ca: Tensor,
    c: Tensor,
    psi: Tensor,
    omega: Tensor,
    phi: Tensor,
    geometry: BackboneGeometry,
) -> tuple[Tensor, Tensor, Tensor]:
    """The next residue's ``(N, CA, C)``, placed off this one's.

    ``psi`` and ``omega`` belong to the parent and ``phi`` to the child, which
    is the asymmetry of the peptide bond rather than a choice: psi is
    N-CA-C-N(next), so it is not measurable until the child's N exists, and
    growing is precisely the act of deciding it.
    """
    n_next = place_atom(n, ca, c, geometry.c_n, geometry.ca_c_n, psi)
    ca_next = place_atom(ca, c, n_next, geometry.n_ca, geometry.c_n_ca, omega)
    c_next = place_atom(c, n_next, ca_next, geometry.ca_c, geometry.n_ca_c, phi)
    return n_next, ca_next, c_next


def extend_nward(
    n: Tensor,
    ca: Tensor,
    c: Tensor,
    phi: Tensor,
    omega: Tensor,
    psi: Tensor,
    geometry: BackboneGeometry,
) -> tuple[Tensor, Tensor, Tensor]:
    """The previous residue's ``(N, CA, C)``, placed off this one's.

    The mirror of :func:`extend_cward`: ``phi`` is the parent's, ``omega`` and
    ``psi`` the child's. Growth has to be able to run both ways or the alive
    set is a prefix, and a prefix is the synthesis order this project argues
    against.
    """
    c_prev = place_atom(c, ca, n, geometry.c_n, geometry.c_n_ca, phi)
    ca_prev = place_atom(ca, n, c_prev, geometry.ca_c, geometry.ca_c_n, omega)
    n_prev = place_atom(n, c_prev, ca_prev, geometry.n_ca, geometry.n_ca_c, psi)
    return n_prev, ca_prev, c_prev


def place_oxygen(n: Tensor, ca: Tensor, c: Tensor, psi: Tensor) -> Tensor:
    """The carbonyl O, which is a function of psi and therefore of the child.

    O is often described as part of the backbone's rigid group, and it is not:
    its dihedral N-CA-C-O is psi minus 180 degrees, so a residue's O cannot be
    placed correctly until its C-ward neighbour has been. A newborn carries its
    parked O until that happens; this is what corrects it.
    """
    bond = torch.as_tensor(C_O, device=c.device, dtype=c.dtype)
    angle = torch.as_tensor(math.radians(CA_C_O), device=c.device, dtype=c.dtype)
    return place_atom(n, ca, c, bond, angle, psi - math.pi)


def frame_from_triad(n: Tensor, ca: Tensor, c: Tensor) -> Tensor:
    """``(..., 3, 3)`` rotation whose columns are an orthonormal basis at CA.

    The same construction as :func:`torch_graph.backbone_frames`, repeated here
    rather than imported because that one is tied to a ``FeatureTopology`` and
    this one takes bare points.
    """
    x = n - ca
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    y = c - ca
    y = y - (y * x).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)


def rigid_onto_triad(
    atoms: Tensor,
    source: tuple[Tensor, Tensor, Tensor],
    target: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Move ``atoms`` so the ``source`` triad lands on the ``target`` triad.

    A proper rotation composed from the two frames rather than a least-squares
    fit: CA maps onto CA exactly and N and C land to within the difference
    between the two triads' own internal geometry, which is the ideal-versus-
    measured gap and is reported by ``scripts/growth_probe.py`` instead of
    being hidden inside a residual.

    This is how a newborn gets the rest of its atoms. The alternative -- ideal
    coordinates for all twenty residue types -- is a table this project does
    not have and does not need: bond lengths and angles are frozen in v1, so a
    residue's own internal geometry is a constant of the problem and may as
    well be its crystal one, with the side chain's *torsions* left to the rule.
    """
    rot_src = frame_from_triad(*source)
    rot_dst = frame_from_triad(*target)
    rot = rot_dst @ rot_src.transpose(-1, -2)
    shifted = atoms - source[1].unsqueeze(-2)
    return shifted @ rot.transpose(-1, -2) + target[1].unsqueeze(-2)


def backbone_atoms(coords: Tensor, topo: FeatureTopology) -> Tensor:
    """``(R, 4, 3)`` for N, CA, C, O, straight off ``frame_idx``."""
    rows = torch.arange(coords.shape[0], device=coords.device).unsqueeze(-1)
    return coords[rows, topo.frame_idx[:, :4]]


def measure_backbone_torsions(
    coords: Tensor, topo: FeatureTopology, links: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """``(phi, psi, omega)``, each ``(R,)`` radians, zero where undefined.

    ``links[i]`` says residue ``i`` is bonded to ``i + 1``
    (:func:`torch_graph.sequence_links`). phi and omega need the bond on the N
    side, psi the one on the C side, so each is defined on a different subset
    and the caller needs the same ``links`` to mask them.
    """
    bb = backbone_atoms(coords, topo)
    n, ca, c = bb[:, 0], bb[:, 1], bb[:, 2]
    zero = torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
    phi, psi, omega = zero.clone(), zero.clone(), zero.clone()

    index = torch.nonzero(links, as_tuple=False).flatten()
    if index.numel() == 0:
        return phi, psi, omega
    nxt = index + 1
    psi[index] = dihedral(n[index], ca[index], c[index], n[nxt])
    omega[index] = dihedral(ca[index], c[index], n[nxt], ca[nxt])
    phi[nxt] = dihedral(c[index], n[nxt], ca[nxt], c[nxt])
    return phi, psi, omega


def _angle(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    u = a - b
    v = c - b
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.acos((u * v).sum(-1).clamp(-1.0 + 1e-12, 1.0 - 1e-12))


def measured_geometry(
    coords: Tensor, topo: FeatureTopology, links: Tensor
) -> BackboneGeometry:
    """The structure's own bond lengths and angles, per residue.

    Only for tests and for the probe: it is the control that separates "the
    primitive is wrong" from "ideal geometry is too coarse". A growth rollout
    cannot use it, because a structure being grown de novo has no measured
    geometry to read.
    """
    bb = backbone_atoms(coords, topo)
    n, ca, c = bb[:, 0], bb[:, 1], bb[:, 2]
    n_res = coords.shape[0]
    kw = {"device": coords.device, "dtype": coords.dtype}
    c_n = torch.full((n_res,), C_N, **kw)
    ca_c_n = torch.full((n_res,), math.radians(CA_C_N), **kw)
    c_n_ca = torch.full((n_res,), math.radians(C_N_CA), **kw)
    index = torch.nonzero(links, as_tuple=False).flatten()
    if index.numel():
        nxt = index + 1
        c_n[index] = (n[nxt] - c[index]).norm(dim=-1)
        ca_c_n[index] = _angle(ca[index], c[index], n[nxt])
        c_n_ca[index] = _angle(c[index], n[nxt], ca[nxt])
    return BackboneGeometry(
        n_ca=(ca - n).norm(dim=-1),
        ca_c=(c - ca).norm(dim=-1),
        c_n=c_n,
        n_ca_c=_angle(n, ca, c),
        ca_c_n=ca_c_n,
        c_n_ca=c_n_ca,
    )


def index_geometry(geometry: BackboneGeometry, index: Tensor) -> BackboneGeometry:
    """Take the rows ``index`` of a per-residue geometry; scalars pass through."""

    def pick(value: Tensor) -> Tensor:
        return value if value.dim() == 0 else value[index]

    return BackboneGeometry(
        n_ca=pick(geometry.n_ca),
        ca_c=pick(geometry.ca_c),
        c_n=pick(geometry.c_n),
        n_ca_c=pick(geometry.n_ca_c),
        ca_c_n=pick(geometry.ca_c_n),
        c_n_ca=pick(geometry.c_n_ca),
    )
