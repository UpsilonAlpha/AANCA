"""Steric exclusion: the term ``losses.py`` has been describing and not having.

The comment in :func:`losses.harmonic_restraint` has said since v1 that
"deforming a bond relieves a steric clash more cheaply than finding the right
rotamer". It was written about a clash term that did not exist. This is it.

Why it is being added now, rather than as one more plausible ingredient: the
clamp test (``scripts/clamp_test.py``, ``docs/benchmarks/clamp_v5_native.log``)
found that pinning three quarters of a protein at its *crystal* conformations
makes the free residues **worse** -- 57.1 % well-right against 77.3 % when the
whole structure is damaged. Two things predict that table. One is that the
spatially mixed state is off-distribution, which ``--damage-fractions`` fixes.
The other is that nothing in the objective has ever told the rule that two side
chains may not occupy the same space, so it has no reason to treat a correct
neighbour as an obstacle rather than as noise. That is this module.

The formulation is copied, not invented: AlphaFold2's
``between_residue_clash_loss`` -- a flat-bottom penalty ``max(0, r_i + r_j -
tol - d)`` over non-bonded heavy-atom pairs, with Bondi radii and a
deliberately slack tolerance -- which AttnPacker's steric term uses as well. A
penalty that is not the literature's is not comparable with the literature's.

Three things about it are decisions rather than transcription, and each is
argued where it is made: which pairs are exempt (see
``residue_constants.CLASH_MIN_SEPARATION``), that backbone-backbone pairs are
dropped, and that the term is normalised per atom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import residue_constants as rc
from .torch_geometry import MAX_ATOMS, TorsionTopology

#: Angstroms of overlap allowed before anything is charged. AlphaFold2's
#: ``clash_overlap_tolerance``. It is slack on purpose -- at 1.5 A two carbons
#: are only penalised below 1.9 A, which no real non-bonded pair reaches -- so
#: the term fires on genuine interpenetration and stays silent on the ordinary
#: tight contacts that hold a core together. A tighter value would turn it into
#: a packing-density prior, which is a different and much less safe thing to
#: add to an objective.
DEFAULT_TOLERANCE = 1.5

#: CA-CA cutoff for the residue-pair neighbour list, in angstroms.
#:
#: Derived rather than guessed. The furthest any side-chain atom sits from its
#: own CA is about 6.6 A (ARG's NH1/NH2), and two atoms can only be charged if
#: they are within ``r_i + r_j - tol``, at most 3.6 A at tolerance zero. So no
#: pair of residues further apart at the CA than ``6.6 + 3.6 + 6.6 = 16.8`` A
#: can contribute, whatever the tolerance. 17 A is that bound, not a tuned
#: value, which is why it is safe to leave alone.
DEFAULT_CUTOFF = 17.0

#: Upper bound on an S-S distance for two cysteines to count as bonded. A
#: disulfide is 2.03-2.05 A and the next thing up is a non-bonded S-S contact
#: at well over 3 A, so the threshold sits in an empty gap and needs no care.
DISULFIDE_MAX = 2.5

_EPS = 1e-8


@dataclass(frozen=True)
class StericTopology:
    """What the clash term needs to know that is not the coordinates.

    Built once per structure and reused, like ``TorsionTopology``. The one field
    that is not purely sequence-derived is ``linkage``, which needs the native
    coordinates to find the disulfides; see :func:`steric_topology`.
    """

    radii: Tensor
    backbone: Tensor
    exempt: Tensor
    linkage: Tensor
    partner: Tensor
    #: The atom each residue's neighbour-list distance is measured from, and
    #: how far that atom may sit from where the CA is. Zero for a real CA; see
    #: :func:`steric_topology` for why it is ever anything else.
    anchor_index: Tensor
    anchor_slack: Tensor

    @property
    def n_residues(self) -> int:
        return self.radii.shape[0]

    def to(self, device: torch.device | str) -> "StericTopology":
        return StericTopology(
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "radii",
                    "backbone",
                    "exempt",
                    "linkage",
                    "partner",
                    "anchor_index",
                    "anchor_slack",
                )
            }
        )


def _disulfide_partners(sg_index: Tensor, coords: Tensor) -> Tensor:
    """Mutually nearest cysteine pairs within :data:`DISULFIDE_MAX`.

    Mutual nearest rather than "any pair under the threshold": a cysteine has
    one SG and can make one disulfide, so a threshold alone could pair a
    cysteine with two partners in a cluster and the exemptions would stop being
    a matching.
    """
    n_res = sg_index.shape[0]
    partner = torch.full((n_res,), -1, dtype=torch.long)
    which = (sg_index >= 0).nonzero(as_tuple=True)[0]
    if which.numel() < 2:
        return partner
    sg = coords[which, sg_index[which]].detach().to(torch.float64)
    distance = torch.cdist(sg, sg)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.argmin(dim=1)
    for a in range(which.numel()):
        b = int(nearest[a])
        if int(nearest[b]) == a and float(distance[a, b]) <= DISULFIDE_MAX:
            partner[which[a]] = which[b]
    return partner


def steric_topology(
    torsion: TorsionTopology,
    native_coords: Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> StericTopology:
    """Radii, exemptions and the disulfide matching for one structure.

    ``native_coords`` is read only to find the disulfides, and only the SG
    positions of it. It is the *native* coordinates deliberately: a disulfide is
    a covalent bond, so whether one exists is a fact about the structure and not
    something the rule may decide by moving a side chain. Passing ``None``
    treats every cysteine as free, which is the right behaviour for a synthetic
    topology in a test and the wrong one for a real protein.
    """
    n_res = torsion.n_residues
    radii = torch.zeros(n_res, MAX_ATOMS, dtype=dtype)
    backbone = torch.zeros(n_res, MAX_ATOMS, dtype=torch.bool)
    exempt = torch.zeros(n_res, MAX_ATOMS, MAX_ATOMS, dtype=torch.bool)
    linkage = torch.zeros(n_res, MAX_ATOMS, dtype=torch.bool)
    anchor_index = torch.zeros(n_res, dtype=torch.long)
    anchor_slack = torch.zeros(n_res, dtype=dtype)
    sg_index = torch.full((n_res,), -1, dtype=torch.long)
    fixed = rc.BACKBONE_ATOMS | rc.TERMINAL_ATOMS

    for r, (resname, names) in enumerate(
        zip(torsion.resnames, torsion.atom_names, strict=True)
    ):
        index = {name: i for i, name in enumerate(names)}
        # The neighbour list is anchored on CA, but **a residue in the training
        # set need not have one**. v7's first training update died here on
        # "residue 149 (GLU) has no CA" after the warm start had already
        # loaded, because the two complete crystal structures the tests use
        # (1UBQ, 193L) cannot produce the case. The rest of the pipeline
        # already tolerates incomplete residues -- that is what
        # ``FeatureTopology.virtual_cb`` and ``fallback_idx`` are for -- so
        # raising here made the steric term stricter than the model it serves.
        #
        # Falling back costs a little cutoff margin rather than correctness:
        # N and C sit at most 1.52 A from CA and CB at most 1.53, so a fallback
        # anchor is within 2.5 A of where the CA is. That slack is added to the
        # pair test per residue instead of being absorbed into a bigger global
        # cutoff, so an ordinary residue pays nothing for a broken neighbour.
        for name, slack in (("CA", 0.0), ("N", 2.5), ("C", 2.5), ("CB", 2.5)):
            if name in index:
                anchor_index[r] = index[name]
                anchor_slack[r] = slack
                break
        else:
            if not names:
                raise ValueError(f"residue {r} ({resname}) has no atoms at all")
            # Nothing recognisable: anchor on whatever is there and widen the
            # slack to the furthest a side-chain atom can be from CA, so the
            # pair list is still a superset rather than silently lossy.
            anchor_index[r] = 0
            anchor_slack[r] = 6.6
        for i, name in enumerate(names):
            radii[r, i] = rc.vdw_radius(name)
            backbone[r, i] = name in fixed
        for a, b in rc.CLASH_EXEMPT_PAIRS[resname]:
            ia, ib = index.get(a), index.get(b)
            if ia is not None and ib is not None:
                exempt[r, ia, ib] = True
        if resname == "CYS" and "SG" in index:
            sg_index[r] = index["SG"]

    partner = (
        _disulfide_partners(sg_index, native_coords)
        if native_coords is not None
        else torch.full((n_res,), -1, dtype=torch.long)
    )
    for r in range(n_res):
        if int(partner[r]) >= 0:
            for i, name in enumerate(torsion.atom_names[r]):
                # CB and SG only. Across the S-S bond SG-SG is one bond, CB-SG
                # two and CB-CB three, so all four of those pairs fall under
                # CLASH_MIN_SEPARATION and none of them is a steric
                # observation. Everything further out still is.
                linkage[r, i] = name in ("CB", "SG")

    return StericTopology(
        radii=radii,
        backbone=backbone,
        exempt=exempt,
        linkage=linkage,
        partner=partner,
        anchor_index=anchor_index,
        anchor_slack=anchor_slack,
    ).to(torsion.atom_mask.device)


def _overlap(a: Tensor, b: Tensor, bound: Tensor, scored: Tensor) -> Tensor:
    """``sum(relu(bound - |a - b|))`` over the entries ``scored`` selects.

    The distance is ``sqrt(d2 + eps)`` rather than :func:`torch.cdist` on
    purpose. Padded atoms sit at the origin, so a padded pair is at distance
    zero, where ``cdist``'s backward pass is 0/0; the epsilon makes the
    derivative merely large rather than undefined, and the flat bottom then
    multiplies it by zero anyway, because a padded atom has radius zero and so
    a negative bound.
    """
    diff = a.unsqueeze(-2) - b.unsqueeze(-3)
    distance = torch.sqrt((diff * diff).sum(-1) + _EPS)
    return ((bound - distance).clamp_min(0.0) * scored).sum()


def clash_loss(
    coords: Tensor,
    torsion: TorsionTopology,
    steric: StericTopology,
    supervised: Tensor | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    cutoff: float = DEFAULT_CUTOFF,
) -> Tensor:
    """Mean angstroms of van der Waals overlap per supervised heavy atom.

    Normalised per atom rather than per scored *pair*, following AlphaFold2.
    Per pair would divide a handful of real overlaps by the hundreds of
    thousands of comfortable pairs around them and report a number that is
    always near zero and moves for the wrong reasons; per atom gives a quantity
    that means something on its own -- how deep, on average, an atom sits inside
    its neighbours -- so it can be read beside the chi MAE rather than only
    against its own history.

    ``supervised`` restricts the term the way every other term is restricted,
    but with "or" rather than "and": a pair is scored if *either* residue is
    supervised. A crop's boundary shell is native context, and a supervised
    residue driven into the shell is a real clash the rule should feel. Only
    shell-against-shell is somebody else's problem.
    """
    if coords.shape[:2] != (torsion.n_residues, MAX_ATOMS):
        raise ValueError(
            f"coords must be (n_res, {MAX_ATOMS}, 3), got {tuple(coords.shape)}"
        )
    device = coords.device
    real = torsion.atom_mask
    radii = steric.radii.to(coords.dtype)
    if supervised is None:
        supervised = torch.ones(
            torsion.n_residues, dtype=torch.bool, device=device
        )

    # --- inside one residue -------------------------------------------------
    # A long side chain folding back on itself. Backbone-backbone pairs are
    # dropped here and below: the backbone is an input, so those distances are
    # constants, and keeping them would add a fixed offset no gradient can
    # reach and that would have to be subtracted by hand to read the term.
    both_bb = steric.backbone.unsqueeze(-1) & steric.backbone.unsqueeze(-2)
    upper = torch.triu(
        torch.ones(MAX_ATOMS, MAX_ATOMS, dtype=torch.bool, device=device),
        diagonal=1,
    )
    intra = (
        real.unsqueeze(-1)
        & real.unsqueeze(-2)
        & ~steric.exempt
        & ~both_bb
        & upper
        & supervised.view(-1, 1, 1)
    )
    bound_intra = radii.unsqueeze(-1) + radii.unsqueeze(-2) - tolerance
    total = _overlap(coords, coords, bound_intra, intra)

    # --- between residues ---------------------------------------------------
    # The pair list is built from the CA positions, which are backbone and so
    # fixed, and detached regardless: which residues are candidates is a
    # bookkeeping decision and not part of the function being differentiated.
    with torch.no_grad():
        rows = torch.arange(torsion.n_residues, device=device)
        anchor = coords.detach()[rows, steric.anchor_index]
        # The bound is per pair, not global: a residue whose anchor is not a
        # real CA carries its own slack, so one broken residue widens only the
        # rows and columns it appears in.
        slack = steric.anchor_slack.to(coords.dtype)
        allowed = cutoff + slack.unsqueeze(-1) + slack.unsqueeze(-2)
        near = torch.cdist(anchor, anchor) <= allowed
        near &= torch.triu(torch.ones_like(near), diagonal=1)
        near &= supervised.unsqueeze(-1) | supervised.unsqueeze(-2)
        pairs = near.nonzero(as_tuple=False)

    if pairs.numel():
        i, j = pairs[:, 0], pairs[:, 1]
        scored = real[i].unsqueeze(-1) & real[j].unsqueeze(-2)
        scored &= ~(
            steric.backbone[i].unsqueeze(-1) & steric.backbone[j].unsqueeze(-2)
        )
        linked = (steric.partner[i] == j).view(-1, 1, 1)
        scored &= ~(
            linked
            & steric.linkage[i].unsqueeze(-1)
            & steric.linkage[j].unsqueeze(-2)
        )
        bound = radii[i].unsqueeze(-1) + radii[j].unsqueeze(-2) - tolerance
        total = total + _overlap(coords[i], coords[j], bound, scored)

    n_atoms = (real & supervised.unsqueeze(-1)).sum()
    if n_atoms == 0:
        return total * 0.0
    return total / n_atoms
