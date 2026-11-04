"""Growing a protein: pre-allocated slots, an alive mask, and a frontier.

The design, and why it is this and not something else.

**Growing NCA does not create cells.** Mordvintsev's lattice is complete from
step 0; growth is an *alive mask* that spreads from one seed, and a cell is
"alive" if anything in its neighbourhood has alpha above a threshold. That is
the trick to transplant, because it resolves the chicken-and-egg that kills the
obvious design: a residue cannot be placed by the graph it is about to join,
since it has no coordinates until it is placed and no edges until it has
coordinates. With pre-allocated slots the substrate is the ``N`` residue slots,
the sequence edges exist from step 0, and only the *coordinates* are absent.

**The alive set is a union of sequence segments, and that is forced, not
chosen.** A dormant slot has no coordinates, so "spatially adjacent to a live
cell" is undefined for it. The only adjacency a dormant slot has is covalent:
it is bonded to ``i - 1`` and ``i + 1``. So growth extends segments at either
end and nucleates new ones at new seeds. With several seeds this is nucleation
and coalescence -- segments grow in both directions and later meet -- which is
foldon-like and is emphatically *not* synthesis order. With one seed and
C-ward growth only it degenerates to synthesis order, which is the control to
run, not the model.

**Birth is the one place a torsion is unavoidable, and the one place nothing is
downstream.** A newborn's backbone comes from its parent's frame plus ideal
bond geometry and three predicted torsions (``aanca.backbone``). At the instant
of birth the lever arm is exactly zero, because the newborn has no children
yet.

**Relaxation is not more torsions.** Measured on 1UBQ: regrowing the backbone
from the crystal's own torsions with *ideal* bond lengths and angles is exact
at the first residue, 1.47 A off by the tenth and 29.4 A off by the
seventy-fifth (``scripts/growth_probe.py``). That is the lever arm, at the
backbone, and it is fatal for a torsion-space relaxation: a rule that corrects
a fold by adjusting phi in the middle swings everything downstream. So an alive
residue relaxes by a bounded **rigid move of its own atoms** -- an axis-angle
rotation and a translation, EGNN's coordinate update in its SE(3) form -- and
the chain is held together by a peptide-bond restraint instead of by
construction. No lever arm, and a global correction is a local parameter
change, which is how AF2 and every structure diffusion model do it.

What this module does *not* claim to have solved, recorded here rather than
discovered later: a grown structure is a sample from whatever the rule happens
to do, with no density attached, so "grown" says nothing yet about whether the
result is a protein; two segments that meet are not covalently closed, which is
the loop-closure problem and is left to the bond restraint; and nothing here
forbids a mirror fold, because the activation and frame heads see only
distances.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from .backbone import (
    BackboneGeometry,
    backbone_atoms,
    extend_cward,
    extend_nward,
    ideal_geometry,
    index_geometry,
    frame_from_triad,
    helical_birth_reference,
    rotation_from_vector,
    measure_backbone_torsions,
    place_oxygen,
    rigid_onto_triad,
    triad_from_frame,
)
from .torch_geometry import rotation_matrices
from .torch_graph import FeatureTopology, sequence_links

#: How far apart dormant residues are parked, Angstrom. Any value well beyond
#: the 10 A interaction cutoff works; this one also keeps dormant residues far
#: from *each other*, so a bug that let them into the graph would show up as an
#: obvious distance rather than as a plausible contact.
PARK_SPACING = 500.0


@dataclass(frozen=True)
class GrowthConfig:
    """Everything about growth that is a choice rather than a consequence."""

    #: How many segments nucleate. 1 with ``cward_only`` is the synthesis-order
    #: control; several seeds is the nucleation-and-coalescence model.
    seeds: int = 1
    #: Forbid N-ward growth. Only for the control above.
    cward_only: bool = False
    #: How a newborn is placed. ``"torsion"`` builds its backbone by NeRF from
    #: three predicted torsions, which is correct by construction and carries
    #: the lever arm. ``"cartesian"`` takes a frame the parent emitted, which
    #: has no lever arm anywhere and is type-independent -- the same state
    #: variable the SE(3) diffusion models use. Must match the rule's own
    #: ``birth``, since the head's width differs.
    birth: str = "torsion"
    #: How a wake decision is drawn from the activation logit. ``"sample"``
    #: draws a Bernoulli and passes a straight-through gradient; ``"threshold"``
    #: takes ``logit > 0``, which is what a deterministic probe wants.
    activation: str = "sample"
    #: Ceiling on how many residues may be born in one step, as a fraction of
    #: the slots. Growth is meant to be parallel over the frontier -- that is
    #: the whole O(surface) argument -- but an unbounded first step would wake
    #: every frontier residue of every segment at once before the rule has any
    #: hidden state, so the frontier is subsampled early on.
    max_births: int = 8
    #: Bounds on one relaxation step's rigid move: radians and Angstrom. Chosen
    #: so thirty steps can reorient a residue completely while one step stays
    #: local, which is the same reasoning as ``MAX_BOND_DELTA``.
    max_rotation: float = 0.1
    max_translation: float = 0.3


# -- term weights, measured rather than chosen ----------------------------
#
# ``scripts/growth_balance_probe.py``, 1UBQ at 6 seeds over 30 steps, each term
# alone with every other weight zeroed. The gradient norm each induces on the
# activation head, against FAPE's 1.83e1:
#
#     term          value   activation      birth      frame   identity
#     fape         0.6558     1.83e+01   1.37e-01   1.07e+00   0.00e+00
#     bond         0.7719     8.69e+02   1.97e+00   1.10e+01   0.00e+00
#     reach        0.1676     2.26e+02   7.24e-01   1.88e+00   0.00e+00
#     completion   2.1269     7.17e-01   0.00e+00   0.00e+00   0.00e+00
#     identity     3.1214     0.00e+00   0.00e+00   0.00e+00   2.08e-01
#
# So the weights below are what equalises each term's gradient against FAPE's.
# All four defaults were 1.0 and were wrong by one to two orders of magnitude,
# **in both directions**: the peptide restraint was 47 times too strong and
# completion 27 times too weak.
#
# **This corrects a claim recorded earlier in HANDOFF.md.** That said completion
# dominated early training and would need lowering, on the grounds that its
# *value* was 2.13 against FAPE's 0.56. A term's value is not its influence: in
# gradient, completion is 0.72 against FAPE's 18.3. The 4.8e2 activation
# gradient quoted there belonged to the reach term, which was on by default.
#
# Two things the table shows that are not about weights. Completion gives
# gradient to the activation head and *exactly zero* to the others, and identity
# only to the identity head -- clean confirmation that each term is wired to the
# head it is about. And the activation head receives two orders of magnitude
# more gradient than the birth head from every term, which is structural rather
# than a fault: a frontier residue decides whether to wake on every step, while
# the birth head only matters on the steps where a birth happens.
#
# A hypothesis tested and disproved, recorded so it is not re-tried: the
# activation head's gradient is **not** inflated by ``PARK_SPACING``. The
# straight-through gate's derivative is ``placed - parked``, which is ~500 A, so
# it looked like an artefact of an arbitrary constant -- but FAPE clamps at 10 A
# and a clamp has zero gradient beyond it, so a residue parked 500 A away
# contributes nothing. Measured flat at 1.8e1 for spacings of 500, 100 and 20 A.
#
# **These balance the terms at initialisation and nowhere else**, which is the
# limit worth stating: three of the four heads are zeroed and the structure is
# barely grown. The reach term reads 0.168 here and only bites once a closure is
# foreclosed, so it is the one most likely underweighted; the bond term reads
# 0.772 and should shrink as the chain improves. Re-measure a few thousand
# updates in, and watch the alive fraction on the first run -- completion at 27
# is a large change in the direction of growing faster.
BOND_WEIGHT = 0.022
REACH_WEIGHT = 0.086
COMPLETION_WEIGHT = 27.2
#: Measured at 93.7. Left for the caller because ``identity_weight`` defaults
#: to 0 -- the identity track is opt-in, not a weight to tune by default.
IDENTITY_WEIGHT = 93.7


def seed_alive(
    n_residues: int,
    seeds: int = 1,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """``(R,)`` bool with ``seeds`` residues alive, drawn without replacement.

    Seeds are drawn on the CPU and moved, like every other random choice in
    this project, so the sequence is a property of the seed and not of the
    device.
    """
    if seeds < 1:
        raise ValueError(f"seeds must be at least 1, got {seeds}")
    if seeds > n_residues:
        raise ValueError(f"{seeds} seeds asked for in {n_residues} slots")
    order = torch.randperm(n_residues, generator=generator)
    alive = torch.zeros(n_residues, dtype=torch.bool)
    alive[order[:seeds]] = True
    return alive.to(device)


def frontier(alive: Tensor, links: Tensor, cward_only: bool = False) -> tuple[Tensor, Tensor]:
    """``(nward, cward)``: alive residues with a dormant covalent neighbour.

    ``nward[i]`` means residue ``i`` is alive, bonded to ``i - 1``, and
    ``i - 1`` is dormant -- so ``i`` may give birth N-ward. ``cward`` is the
    mirror. A residue can be on both, which is a segment of length one.
    """
    nward = torch.zeros_like(alive)
    cward = torch.zeros_like(alive)
    if alive.shape[0] < 2:
        return nward, cward
    cward[:-1] = alive[:-1] & links & ~alive[1:]
    if not cward_only:
        nward[1:] = alive[1:] & links & ~alive[:-1]
    return nward, cward


def park_coords(coords: Tensor, alive: Tensor, spacing: float = PARK_SPACING) -> Tensor:
    """Move every dormant residue far away, keeping its internal geometry.

    A dormant slot has to hold *something*: ``node_features`` measures chis
    from it and a NaN there would poison the whole batch's gradient. Zeros
    would make every atom coincident and the dihedral of coincident points has
    no derivative, so dormant residues keep their own internal conformation and
    are simply translated out of reach -- which is also what makes them a
    usable source triad when they are born (:func:`rigid_onto_triad`).

    This is belt and braces: :func:`alive_topology` already removes their atoms
    from neighbour selection and the step masks them out of every edge. The
    parking is what makes a *missed* mask loud instead of plausible.
    """
    offset = torch.arange(
        1, coords.shape[0] + 1, device=coords.device, dtype=coords.dtype
    ) * spacing
    shift = torch.zeros_like(coords[:, :1, :])
    shift[:, 0, 0] = offset
    return torch.where(alive[:, None, None], coords, coords + shift)


def alive_topology(topo: FeatureTopology, alive: Tensor) -> FeatureTopology:
    """``topo`` with dormant residues holding no atoms and no supervision.

    ``atom_mask`` is what neighbour selection flattens the coordinates by and
    what ``pair_min_distances`` validates pairs with, so masking it is what
    actually removes a dormant residue from perception. ``owner`` has to be
    re-derived at the same time, because it is indexed by *surviving atom*, and
    an ``owner`` that disagrees with ``atom_mask`` mislabels every atom after
    the first dormant residue -- a wrong answer with no symptom.
    """
    keep = topo.atom_mask & alive.unsqueeze(-1)
    full = topo.atom_mask.detach().cpu().numpy()
    selector = (keep.detach().cpu().numpy() & full)[full]
    return replace(
        topo,
        atom_mask=keep,
        owner=topo.owner[selector],
        supervised=topo.supervised_mask & alive,
    )


def straight_through_bernoulli(
    logit: Tensor, sample: bool = True, generator: torch.Generator | None = None
) -> Tensor:
    """A hard 0/1 decision whose gradient is the sigmoid's.

    The same device as the identity head's straight-through one-hot, and for
    the same reason: the wake decision is discrete, and a newborn's
    coordinates are ``where(born, placed, parked)``. Written as
    ``born * placed + (1 - born) * parked`` with ``born`` straight-through, the
    gradient reaching the activation logit is ``(placed - parked) . dL/dx`` --
    "waking this neighbour changed the loss by this much". That is the only
    credit signal growth order has, and a hard decision would give none.
    """
    probability = torch.sigmoid(logit)
    if sample:
        draw = torch.rand(logit.shape, generator=generator, device="cpu").to(
            logit.device
        )
        hard = (draw < probability).to(probability.dtype)
    else:
        hard = (logit > 0).to(probability.dtype)
    return hard.detach() - probability.detach() + probability


def limit_births(mask: Tensor, limit: int, generator: torch.Generator | None) -> Tensor:
    """Keep at most ``limit`` of the ``True`` entries, chosen uniformly."""
    index = torch.nonzero(mask, as_tuple=False).flatten()
    if index.numel() <= limit:
        return mask
    order = torch.randperm(index.numel(), generator=generator)
    out = torch.zeros_like(mask)
    out[index[order[:limit].to(index.device)]] = True
    return out


def rigid_relax(
    coords: Tensor,
    axis_angle: Tensor,
    translation: Tensor,
    topo: FeatureTopology,
    moving: Tensor,
) -> Tensor:
    """Rotate and translate each moving residue's atoms about its own CA.

    About CA rather than the centroid so the move is a frame update in the
    sense ``aanca.fape`` scores: the residue's position *is* its CA and its
    orientation *is* its N-CA-C frame.
    """
    rot = rotation_from_vector(axis_angle).to(coords.dtype)
    ca = backbone_atoms(coords, topo)[:, 1]
    origin = ca.unsqueeze(1)
    moved = (coords - origin) @ rot.transpose(-1, -2) + origin + translation.unsqueeze(1).to(
        coords.dtype
    )
    return torch.where(moving[:, None, None], moved, coords)


def give_birth(
    coords: Tensor,
    topo: FeatureTopology,
    alive: Tensor,
    links: Tensor,
    born_n: Tensor,
    born_c: Tensor,
    torsions: Tensor,
    geometry: BackboneGeometry | None = None,
    birth_mode: str = "torsion",
) -> tuple[Tensor, Tensor]:
    """Place the residues whose parents decided to wake them.

    ``born_n`` and ``born_c`` are indexed by *parent*: ``born_c[i]`` is the
    straight-through decision that ``i`` wakes ``i + 1``. ``torsions`` is
    ``(R, 2, 3)`` -- N-ward then C-ward, three torsions each, in the order
    :func:`backbone.extend_nward` and :func:`backbone.extend_cward` take them.

    Returns ``(coords, alive)``. A newborn's backbone triad comes from the
    NeRF; the rest of its atoms ride on a rigid transform from its parked pose,
    so bond lengths and angles inside the residue stay exactly what the crystal
    said they were -- which is consistent with v1 freezing them, and avoids
    needing ideal coordinates for twenty residue types.

    **The NeRF triad sets the newborn's frame, not its atoms.** Only CA lands
    exactly on it: ``rigid_onto_triad`` is a proper rotation, so the child's N
    and C end up where its *own* N-CA and CA-C put them, and the realised
    peptide bond is therefore ideal plus that residue's own deviation --
    measured at 0.007 A on 1UBQ. The alternative is to overwrite N, CA and C
    with the NeRF triad, which moves the same discrepancy to the CB attachment
    instead; keeping it in the peptide bond is better because the peptide bond
    is exactly what :func:`peptide_bond_restraint` supervises, and a residue's
    internal geometry is meant to be frozen at its crystal values.

    **Under ``birth_mode="cartesian"`` the triad is not constructed at all**,
    and ``torsions`` is instead the ``(R, 2, 6)`` pair of axis-angle vectors
    described at :meth:`model.AANCARule.growth_outputs`. See
    :func:`_cartesian_triad`. The two modes are initialised to the same birth,
    so the probe's comparison between them is a comparison of what is
    learnable and not of where they start.
    """
    geometry = geometry or ideal_geometry(coords.device, coords.dtype)
    bb = backbone_atoms(coords, topo)
    new_coords = coords
    alive = alive.clone()

    for direction, decided in ((0, born_n), (1, born_c)):
        parents = torch.nonzero(decided > 0.5, as_tuple=False).flatten()
        if parents.numel() == 0:
            continue
        children = parents - 1 if direction == 0 else parents + 1
        # **A child may be woken from both sides in the same step**, which is
        # what closing a one-residue gap looks like, and it was a bug worth the
        # measurement that found it. ``bb`` is read once from the incoming
        # coordinates, so the second placement's *source* triad was the child's
        # parked one while its atoms had already been moved -- and
        # ``rigid_onto_triad`` then subtracts a CA five hundred Angstrom per
        # slot away, flinging the residue to roughly the park distance. It
        # showed up as a 13,495 A peptide bond at 16 seeds, in exactly the
        # coalescence regime this is all about.
        #
        # One side wins and the other becomes a junction, which the peptide
        # restraint charges for. That is the honest resolution: two
        # independently grown segments meeting is not a birth, and pretending
        # either parent placed the child would hide the gap rather than report
        # it.
        fresh = ~alive[children]
        parents, children = parents[fresh], children[fresh]
        if parents.numel() == 0:
            continue
        n, ca, c = bb[parents, 0], bb[parents, 1], bb[parents, 2]
        if birth_mode == "cartesian":
            triad = _cartesian_triad(
                (n, ca, c), torsions[parents, direction], direction
            )
        elif birth_mode == "torsion":
            geo = index_geometry(geometry, children)
            angles = torsions[parents, direction]
            extend = extend_nward if direction == 0 else extend_cward
            triad = extend(
                n, ca, c, angles[:, 0], angles[:, 1], angles[:, 2], geo
            )
        else:
            raise ValueError(
                f"unknown birth mode {birth_mode!r}; expected 'torsion' or "
                "'cartesian'"
            )
        source = (bb[children, 0], bb[children, 1], bb[children, 2])
        placed = rigid_onto_triad(new_coords[children], source, triad)
        # The straight-through gate: a born child's coordinates are the placed
        # ones, a refused child's are its parked ones, and the activation
        # logit's gradient is the difference between them.
        gate = decided[parents].to(placed.dtype)[:, None, None]
        blended = gate * placed + (1.0 - gate) * new_coords[children]
        new_coords = new_coords.index_copy(0, children, blended)
        alive[children] = alive[children] | (decided[parents] > 0.5)

    return new_coords, alive


#: The reference birth, per direction, cached because it is pure geometry.
#: Index 0 is N-ward and 1 is C-ward, matching ``give_birth``'s loop.
_REFERENCE: dict[tuple[int, torch.dtype], tuple[Tensor, Tensor]] = {}


def _reference(direction: int, dtype: torch.dtype, device) -> tuple[Tensor, Tensor]:
    key = (direction, dtype)
    if key not in _REFERENCE:
        from .model import BACKBONE_INIT_OMEGA, BACKBONE_INIT_PHI, BACKBONE_INIT_PSI

        axis_angle, offset = helical_birth_reference(
            BACKBONE_INIT_PHI, BACKBONE_INIT_PSI, BACKBONE_INIT_OMEGA,
            cward=direction == 1,
        )
        _REFERENCE[key] = (axis_angle.to(dtype), offset.to(dtype))
    axis_angle, offset = _REFERENCE[key]
    return axis_angle.to(device), offset.to(device)


def _cartesian_triad(
    parent: tuple[Tensor, Tensor, Tensor], birth: Tensor, direction: int
) -> tuple[Tensor, Tensor, Tensor]:
    """A newborn's ``(N, CA, C)`` from a frame the parent emitted, no torsion.

    The variant the measurements argue for, and it rests on two of them.

    **The CA-CA distance is not a free parameter.** It is 3.804 A for every
    (phi, psi) of a trans peptide, because it is fixed by the peptide's own
    bond geometry rather than by the torsions. So a birth chooses the
    *direction* of the child's CA and not its distance, and the offset here is
    the reference offset *rotated* -- which keeps the CA-CA distance exactly
    right by construction, the one invariant that genuinely is invariant.

    **The rotation residual is multiplicative.** Axis-angle is not a vector
    space: measured, the additive difference between the helical and sheet
    births is 4.6 rad while the geodesic between them is 1.86, and the
    representation is discontinuous near pi. So the head's output composes onto
    the reference rotation rather than adding to it.

    Everything is expressed in the **parent's own frame**, so a birth is
    equivariant: rotate the whole structure and the birth rotates with it
    instead of changing. And nothing about the child's amino-acid type enters,
    which is what torsion birth cannot say -- it reads the child's own measured
    N-CA, CA-C and N-CA-C. That is the property the CB-level representation
    needs, and it arrives here as a side effect rather than as extra work.
    """
    # Everything is done in the *coordinates'* precision, not the head's. The
    # head emits float32 and the coordinates are float64, and matmul does not
    # promote -- so casting here rather than at the call site keeps the
    # torsion arithmetic's exactness guarantee and still carries the gradient
    # back into the head.
    dtype = parent[1].dtype
    birth = birth.to(dtype)
    ref_axis_angle, ref_offset = _reference(direction, dtype, birth.device)
    rot_parent = frame_from_triad(*parent)
    rot_reference = rotation_from_vector(ref_axis_angle)
    rot_swing = rotation_from_vector(birth[:, :3])
    rot_spin = rotation_from_vector(birth[:, 3:])

    offset = (rot_swing @ ref_offset.unsqueeze(-1)).squeeze(-1)
    ca_child = parent[1] + (rot_parent @ offset.unsqueeze(-1)).squeeze(-1)
    rot_child = rot_parent @ rot_reference @ rot_spin
    return triad_from_frame(rot_child, ca_child)


def correct_oxygens(
    coords: Tensor, topo: FeatureTopology, alive: Tensor, links: Tensor
) -> Tensor:
    """Re-place the carbonyl O of every alive residue whose C-ward child exists.

    A newborn inherits its parked O, whose N-CA-C-O dihedral is the crystal's
    psi, not the one the rule just chose. Once the child is there psi is
    measurable, so the O can be put where the rule's own decision implies --
    and until then it is the one atom in the structure that is knowingly wrong.
    """
    both = torch.zeros_like(alive)
    both[:-1] = alive[:-1] & links & alive[1:]
    index = torch.nonzero(both, as_tuple=False).flatten()
    if index.numel() == 0:
        return coords
    bb = backbone_atoms(coords, topo)
    psi = measure_backbone_torsions(coords, topo, links)[1]
    fixed = place_oxygen(
        bb[index, 0], bb[index, 1], bb[index, 2], psi[index]
    )
    rows = index
    cols = topo.frame_idx[index, 3]
    out = coords.clone()
    out[rows, cols] = fixed.to(out.dtype)
    return out


def peptide_bond_restraint(
    coords: Tensor, topo: FeatureTopology, alive: Tensor, links: Tensor
) -> Tensor:
    """What holds the chain together once relaxation stopped doing it by construction.

    Two terms over every bonded pair that is alive at both ends: the C-N
    distance against its ideal 1.329 A, and the omega dihedral against planar
    trans. A rigid per-residue move is free to tear the chain apart, and this
    is the only thing that says it should not.

    Returned unweighted, like the clash term, because it is a physical quantity
    with a known value in a crystal structure and multiplying it by a weight
    would destroy the comparison.
    """
    from .backbone import C_N

    both = torch.zeros_like(alive)
    both[:-1] = alive[:-1] & links & alive[1:]
    index = torch.nonzero(both, as_tuple=False).flatten()
    if index.numel() == 0:
        return torch.zeros((), device=coords.device, dtype=coords.dtype)
    bb = backbone_atoms(coords, topo)
    nxt = index + 1
    length = (bb[nxt, 0] - bb[index, 2]).norm(dim=-1)
    omega = measure_backbone_torsions(coords, topo, links)[2][index]
    # cos(omega - pi) is -1 at planar trans, so 1 + cos is zero there and 2 at
    # cis. Cis proline is real and this term forbids it; v1 has no proline
    # exception and that is a known approximation rather than an oversight.
    planarity = 1.0 + torch.cos(omega)
    return ((length - C_N) ** 2).mean() + planarity.mean()


# -- closure-aware growth --------------------------------------------------

#: How far ``g`` bonded residues can reach, Angstrom, indexed by sequence
#: separation. ``REACH[g]`` is the greatest CA(i)-CA(i+g) distance a real
#: protein backbone was observed to span.
#:
#: Measured by ``scripts/reach_probe.py`` over 391 structures and ~188,000
#: pairs per separation, at the **99.99th percentile rather than the maximum**.
#: The maximum is a maximum over the worst modelling artifact in four hundred
#: structures: at g = 1 it is 4.32 A, which is not a peptide bond at all, since
#: trans is 3.80 and cis 2.9 -- so that pair is a chain break with contiguous
#: numbering, or a broken model. The percentile still keeps the nineteen
#: longest of 188,000.
#:
#: Note what the shape says: 3.96 A for one bond falling to 2.99 A per residue
#: by twenty. A long gap cannot be spanned at a peptide bond a residue, because
#: the CA-CA-CA pseudo-angle is capped well below straight.
REACH: tuple[float, ...] = (
    0.00, 3.96, 7.44, 10.96, 14.37, 17.75, 21.09, 24.43, 27.74, 31.03,
    33.98, 37.18, 40.14, 43.00, 45.76, 48.56, 50.64, 53.33, 54.98, 57.24,
    59.90,
)

#: Angstrom per residue beyond the table, taken from its last step
#: (59.90 - 57.24). Extrapolating generously is the safe direction for a
#: feasibility test: it never calls a closure impossible that a real chain
#: could have made.
REACH_SLOPE = REACH[-1] - REACH[-2]


def reach(separation: Tensor) -> Tensor:
    """``REACH`` at each sequence separation, extrapolated beyond the table."""
    table = torch.tensor(REACH, dtype=torch.float64, device=separation.device)
    inside = separation.clamp(0, len(REACH) - 1)
    value = table[inside]
    beyond = (separation - (len(REACH) - 1)).clamp_min(0).to(value.dtype)
    return value + beyond * REACH_SLOPE


def pending_closures(alive: Tensor, links: Tensor) -> tuple[Tensor, Tensor]:
    """``(left, right)`` indices of every pair of alive residues that must bond.

    A pending closure is two alive residues with nothing but dormant slots
    between them and an unbroken covalent path across the whole span. Taking
    *consecutive* entries of the sorted alive set makes "nothing but dormant
    slots between" true by construction, so only the covalent path has to be
    checked -- a gap that crosses a chain break or a numbering discontinuity is
    not a closure, it is two different molecules.

    A pair at separation 1 is included, which is the junction that has already
    formed. It needs no special case: a bond made by a birth sits at 3.80 A and
    ``REACH[1]`` is 3.96, so a real peptide bond charges exactly nothing.
    """
    index = torch.nonzero(alive, as_tuple=False).flatten()
    if index.numel() < 2:
        empty = torch.zeros(0, dtype=torch.long, device=alive.device)
        return empty, empty
    left, right = index[:-1], index[1:]
    # ``links[a:b]`` must be all true, which is a difference of a cumulative
    # sum rather than a loop.
    joined = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=alive.device),
            links.to(torch.long).cumsum(0),
        ]
    )
    spans = joined[right] - joined[left]
    keep = spans == (right - left)
    return left[keep], right[keep]


def reachability_restraint(
    coords: Tensor, topo: FeatureTopology, alive: Tensor, links: Tensor
) -> Tensor:
    """Angstrom by which pending closures exceed what their slots can span.

    ``relu(d - reach(g))``, averaged over pending closures. The term growth
    needs and the peptide restraint cannot be, because this one is a **leading**
    indicator: it charges the instant a birth commits to a geometry that
    *forecloses* closure, while the peptide restraint only fires once the two
    segments are adjacent and nothing local can fix the gap any more.

    Three properties are why this is the design rather than an extra term.
    It is exactly zero for single-seed growth, which has no pending closure at
    all, and zero early in multi-seed growth, where the ends are far apart but
    have slots to spare. Its gradient lands on precisely the two decisions that
    could have avoided the problem -- the birth offset, through the newborn's
    coordinates, and the activation logit, through the straight-through gate --
    because waking a residue is what spends the budget. And it arrives within a
    step or two of the offending birth, so it fits inside the twelve-step
    truncation window, where a coordinated segment translation never could.

    Same logic as the clash term: charge for a geometry that forecloses an
    outcome rather than one that is already wrong. Reported unweighted, in
    Angstrom, so it can be compared against a crystal structure's own value --
    which is zero, since a real chain is closed everywhere.
    """
    left, right = pending_closures(alive, links)
    if left.numel() == 0:
        return torch.zeros((), device=coords.device, dtype=coords.dtype)
    ca = backbone_atoms(coords, topo)[:, 1]
    distance = (ca[right] - ca[left]).norm(dim=-1)
    budget = reach(right - left).to(distance.dtype)
    return (distance - budget).clamp_min(0.0).mean()


#: Bin edges for the closure-separation feature: how many residues away the
#: nearest alive slot is, in one direction. The first eight separations get a
#: bin each because that is the regime the reach bound actually bites in -- a
#: ten Angstrom gap needs about three spare slots -- and beyond that the
#: distinction stops mattering, since REACH[9] is already 31 A.
CLOSURE_BINS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16)
#: One bin per edge above, one for "further than the last edge", one for
#: "there is no alive residue that way at all". Times two directions.
CLOSURE_DIM = 2 * (len(CLOSURE_BINS) + 2)


def closure_separation(alive: Tensor, links: Tensor) -> Tensor:
    """``(R, 2)`` residues to the nearest alive slot, N-ward then C-ward.

    Zero means there is none -- the chain ends, or a covalent break
    intervenes. A value of 1 means the adjacent residue is already alive, which
    is a closure that has already happened.

    **This cannot be inferred from the graph**, which is why it is a feature.
    A dormant slot has no atoms and is removed from perception entirely, so a
    frontier residue sees no edge to anything between it and its closure
    partner. It *can* see the partner itself, labelled by the 33-dim
    ``sequence_offset`` one-hot -- but only if the partner is inside the 10 A
    cutoff, and only by reasoning over the *absence* of edges at the
    intervening offsets, which is exactly the kind of inference
    sum-aggregation message passing does worst. Stating the count directly
    costs twenty-four node dimensions.
    """
    n = alive.shape[0]
    device = alive.device
    out = torch.zeros(n, 2, dtype=torch.long, device=device)
    left, right = pending_closures(alive, links)
    if left.numel():
        separation = right - left
        # The pair (left, right) is each other's nearest alive slot by
        # construction, so one scatter fills both directions.
        out[left, 1] = separation
        out[right, 0] = separation
    return out


def closure_features(separation: Tensor, dtype: torch.dtype) -> Tensor:
    """``(R, CLOSURE_DIM)`` one-hot of :func:`closure_separation`'s bins."""
    edges = torch.tensor(CLOSURE_BINS, device=separation.device)
    # Bin 0 is "none" (separation 0); then one bin per edge; then "beyond".
    index = torch.bucketize(separation, edges, right=False) + 1
    index = torch.where(separation == 0, torch.zeros_like(index), index)
    width = len(CLOSURE_BINS) + 2
    one_hot = torch.zeros(
        separation.shape[0], 2, width, dtype=dtype, device=separation.device
    )
    one_hot.scatter_(-1, index.unsqueeze(-1), 1.0)
    return one_hot.reshape(separation.shape[0], -1)


def growth_links(topo: FeatureTopology) -> Tensor:
    """``sequence_links`` on the topology's own device."""
    return sequence_links(topo, topo.atom_mask.device)


def completion(alive: Tensor) -> float:
    return float(alive.sum()) / max(alive.shape[0], 1)


# -- the objective ---------------------------------------------------------


@dataclass
class GrowthTerms:
    """What a growth update charges for, each reported before its weight."""

    total: Tensor
    #: Frame-aligned point error on the backbone of the alive residues.
    fape: Tensor
    #: C-N length and omega planarity over bonded alive pairs. Physical, so
    #: comparable against a crystal structure's own value, which is ~0.002.
    bond: Tensor
    #: Binary cross-entropy pushing the frontier's activation logits towards
    #: waking, evaluated at the final state only.
    completion: Tensor
    #: Fraction of slots alive at the end.
    alive_fraction: float
    #: CA RMSD after superposition, Angstrom. Reporting only.
    rmsd: float
    #: Mean cross-entropy of the *type marginal* at the final state, before
    #: ``identity_weight``. The type and not the joint (type, chi1 well):
    #: growth runs at backbone-and-CB resolution, so there is no side chain to
    #: carry a chi1 and supervising one would ask the rule to commit to a
    #: conformation the structure cannot express. The joint head is unchanged
    #: and its well half is simply unsupervised here, which is the right
    #: division of labour -- packing supervises it already. ln 20 = 3.00 is the
    #: uninformative floor.
    identity: Tensor
    #: Angstrom by which pending closures exceed what their remaining slots can
    #: span, before ``reach_weight``. Unweighted and physical, like the other
    #: restraints: a real chain is closed everywhere, so a crystal scores
    #: exactly zero. Measured on untrained growth: 0.0000 at one and two seeds,
    #: 0.073 at four, 0.439 at eight, 0.774 at sixteen.
    reach: Tensor
    #: How many pending closures there were. The reach term is a mean over
    #: these, so without the count it cannot be read -- a large mean over one
    #: closure is a different situation from a small mean over twenty.
    closures: int
    #: Fraction of alive residues whose most likely type is the native one.
    #: This is sequence recovery, and it is the number ProteinMPNN (~58 %),
    #: PottsMPNN (~44 %) and RFD4-Proteina report -- so it is the one to quote,
    #: alongside the chi accuracy nobody else has.
    recovery: float


def growth_objective(
    rule,
    native_coords: Tensor,
    topo: FeatureTopology,
    config,
    generator: torch.Generator | None = None,
    bond_weight: float = BOND_WEIGHT,
    completion_weight: float = COMPLETION_WEIGHT,
    identity_weight: float = 0.0,
    identity_start: str = "masked",
    reach_weight: float = REACH_WEIGHT,
    fape_weight: float = 1.0,
) -> tuple[GrowthTerms, object]:
    """Supervised regrowth: from a seed, rebuild the native backbone.

    De novo growth is the goal and this is not it -- a native target is
    available here, so this trains the machinery without needing the
    distribution growth does not have. It is the same relationship the
    reconstruction term has to design.

    **The reachability restraint is on by default**, unlike every other
    optional term in this project, and that is deliberate: multi-seed growth is
    not an elaboration but a requirement -- a single segment cannot finish
    inside the loss window (``scripts/growth_rate_probe.py``) -- so coalescence
    happens on every run that matters, and a run with ``reach_weight = 0``
    would be growing segments it has no reason to be able to join. It costs
    nothing when there is nothing pending: single-seed growth scores exactly
    zero.

    **The degenerate optimum, and the term that closes it.** FAPE is a mean
    over the residues that exist, so growing one more residue can only add
    error: the cheapest structure under FAPE alone is the seed, alone, forever.
    Nothing in the geometry objects to that, which makes ``completion_weight``
    load-bearing rather than a nudge. It is a cross-entropy on the frontier's
    activation logits towards waking, evaluated at the final state: if the
    structure is finished the frontier is empty and the term is exactly zero,
    so it pushes only while there is work left.

    The two signals the activation head gets are different and both are needed.
    This one says "grow"; the straight-through path through the newborn's
    coordinates (``straight_through_bernoulli``) says "that birth helped" or
    "that birth hurt". Without the first the rule never starts; without the
    second it has no reason to prefer one order over another.

    Side chains are out of scope here on purpose: ``rollout.training_objective``
    already supervises chis, and joining the two objectives is a decision to
    take with a measurement rather than by putting them in the same function.

    **With ``identity_weight`` the sequence is grown too**, which is what makes
    this de novo rather than regrowth of a known chain. It turns on
    ``config.blind`` implicitly through the caller and three things follow.
    The newborn arrives *unlabelled*, because
    :func:`rollout.initial_identity` forces every dormant slot to the unknown
    token. The state holds no side chains at all, because a blind growth step
    collapses them onto CB -- so a newborn is no longer secretly its native
    amino acid, which was the leak this closes. And the loss is on the **type
    marginal**, not the joint label, for the reason given at
    :attr:`GrowthTerms.identity`.

    What remains unbuilt, and deliberately: nothing materialises a side chain
    from a committed label. The output is a backbone and a sequence, which is
    what RFdiffusion plus ProteinMPNN produce between them and what
    RFD4-Proteina produces directly; placing the side chains afterwards is the
    existing packing rule's job, and composing the two is its own measurement.
    """
    from .fape import backbone_rmsd, fape as fape_loss
    from .identity import library_index, type_index, type_loss, type_marginal
    from .rollout import initial_identity, initial_state, rollout
    from .torch_geometry import measure_chis

    growth = config.growth
    if growth is None:
        raise ValueError("growth_objective needs a RolloutConfig with growth set")

    alive = seed_alive(
        topo.n_residues, growth.seeds, generator, native_coords.device
    )
    links = growth_links(topo)
    identity_start_state = None
    if identity_weight:
        identity_start_state = initial_identity(
            topo, native_coords, identity_start, generator, alive=alive
        )
    result = rollout(
        rule,
        initial_state(
            native_coords, rule, identity=identity_start_state, alive=alive
        ),
        topo,
        config,
        generator=generator,
        record=False,
    )
    final = result.state
    # **Two scoring sets, because the crop mask exists for a reason that does
    # not transfer to a backbone position.** ``supervised_mask`` is the crop's
    # interior: residues whose 10 A perception is complete, and so -- in
    # packing -- the only ones whose *chi* is a learnable target. A truncated
    # neighbourhood genuinely makes a rotamer unlearnable, and it genuinely
    # makes an amino-acid identity unlearnable, because both are inferences
    # *from* the environment.
    #
    # A native CA position is not. It is exact whether or not the residue can
    # see all its neighbours, and under growth every residue has a truncated
    # neighbourhood by construction -- that is the regime, not an artefact of
    # cropping. Masking the boundary shell out of FAPE therefore discards most
    # of the geometric signal for a reason that does not apply: measured over
    # eight validation entries, the supervised fraction is **12.8 % at crop 48,
    # 19.9 % at 64, 54.8 % at 128 and 97.2 % at 384**. At the small crops
    # growth needs in order to finish inside a rollout, scoring only the
    # interior left FAPE a mean over the empty set -- it read exactly 0.000
    # with the RMSD alongside it reading ``nan``, which is what caught it.
    #
    # So geometry scores every alive residue and identity keeps the mask. That
    # split is not a compromise; it is the mask's own justification applied
    # where it holds and withheld where it does not.
    geometric = final.alive
    scored = final.alive & topo.supervised_mask

    fape = fape_loss(final.coords, native_coords, topo, geometric)
    bond = peptide_bond_restraint(final.coords, topo, final.alive, links)
    reach_term = reachability_restraint(final.coords, topo, final.alive, links)
    closure_left, _closure_right = pending_closures(final.alive, links)

    logits = final.activation_logits
    if logits is None:
        raise ValueError(
            "the rollout produced no activation logits -- build the rule with "
            "AANCARule(growth=True)"
        )
    nward, cward = frontier(final.alive, links, growth.cward_only)
    unfinished = torch.stack([nward, cward], dim=-1)
    if bool(unfinished.any()):
        wanted = logits[unfinished]
        completion_term = torch.nn.functional.binary_cross_entropy_with_logits(
            wanted, torch.ones_like(wanted)
        )
    else:
        completion_term = torch.zeros((), device=logits.device, dtype=logits.dtype)

    identity_term = torch.zeros((), device=fape.device, dtype=fape.dtype)
    recovery = float("nan")
    if identity_weight:
        logits = final.identity_logits
        if logits is None:
            raise ValueError(
                "identity_weight is set but the rollout produced no identity "
                "logits -- build the rule with AANCARule(identity=True) and "
                "pass a RolloutConfig with blind=True"
            )
        native_chi1 = torch.rad2deg(measure_chis(native_coords, topo.torsion)[:, 0])
        target = library_index(topo.torsion.resnames, native_chi1)
        identity_term = type_loss(logits, target, scored)
        with torch.no_grad():
            predicted = type_marginal(torch.softmax(logits, dim=-1)).argmax(-1)
            right = predicted == type_index(target)
            recovery = float(
                (right & scored).sum() / scored.sum().clamp_min(1)
            )

    # ``fape_weight`` exists so FAPE can be switched *off*, which it could not
    # be while it was the one unweighted term. That made
    # ``scripts/growth_balance_probe.py`` unable to isolate any other term --
    # every row measured "FAPE plus that term" and reported it as the term --
    # and gradient norms do not decompose linearly, so the difference could not
    # be taken afterwards either. Nothing should ever set it to 0 in training.
    total = (
        fape_weight * fape
        + bond_weight * bond.to(fape.dtype)
        + completion_weight * completion_term.to(fape.dtype)
        + identity_weight * identity_term.to(fape.dtype)
        + reach_weight * reach_term.to(fape.dtype)
    )
    return (
        GrowthTerms(
            total=total,
            fape=fape.detach(),
            bond=bond.detach(),
            completion=completion_term.detach(),
            alive_fraction=completion(final.alive),
            rmsd=float(
                backbone_rmsd(final.coords, native_coords, topo, geometric).detach()
            ),
            identity=identity_term.detach(),
            recovery=recovery,
            reach=reach_term.detach(),
            closures=int(closure_left.numel()),
        ),
        result,
    )
