"""The local perception graph, rebuilt from the current coordinates every step.

Three decisions are made here, all of them flagged as open risks in the plan.

**The graph is rebuilt per step, not frozen.** Freezing it from the native
structure would let the rule read the answer through the topology -- at
inference time on a novel structure that topology is not available, so a frozen
graph measures something the method cannot actually do. Rebuilding is the honest
option and it is what this module implements.

**Neighbour selection sees side-chain atoms, which is what makes "per step"
mean anything.** The backbone is fixed in this milestone, so a graph built from
backbone and CB atoms alone would be *identical at every step* and could be
computed once. Only a cutoff over all heavy atoms -- side chains included --
produces a graph that actually moves as the rule works. That is the point, and
it is also the cost: see ``GraphStats.k_binding_fraction`` below.

**Edge features are invariant, not equivariant.** Every feature is a distance, a
relative orientation, or a vector expressed in a residue's own backbone frame,
so the rule is rotation- and translation-invariant without an equivariant
architecture. This is ProteinEBM's choice (invariant features plus rotation
augmentation) rather than E(n)-GNCA's, taken deliberately: feeding raw
displacement vectors is the failure mode. ``tests/test_graph.py`` asserts the
invariance rather than trusting it.

This is a NumPy reference implementation, part of the verified data layer. The
model's Torch version will be a port, and should be checked against this. Note
that neighbour *selection* is a hard argsort and is not differentiable in either
version -- gradients flow through the envelope weight, not through the choice of
neighbour, which is precisely why the envelope exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import residue_constants as rc
from .structure import Protein
from .torsions import measure_chis

#: Atoms used for the backbone distance block. CB is virtual where absent.
FRAME_ATOMS = ("N", "CA", "C", "O", "CB")

#: Radial basis centres for distance encoding, in Angstrom.
RBF_MIN, RBF_MAX, RBF_COUNT = 2.0, 22.0, 16

#: ProteinMPNN's virtual-CB coefficients, used for GLY and for any residue whose
#: CB is missing from the model. Keeps the backbone block defined everywhere.
_VCB = (-0.58273431, 0.56802827, -0.54067466)


@dataclass(frozen=True)
class GraphStats:
    """Diagnostics that say whether the graph is behaving as intended."""

    n_residues: int
    mean_degree: float
    #: Fraction of residues for which the k limit, not the distance cutoff, is
    #: what excluded a neighbour. The envelope smooths an edge that leaves
    #: through the *cutoff*; an edge dropped because k was full still pops
    #: discontinuously. Keep this near zero, by raising k, or the per-step
    #: rebuild reintroduces exactly the gradient discontinuity the envelope is
    #: there to remove.
    k_binding_fraction: float
    #: Fraction of residues missing a well-defined N/CA/C frame.
    frameless_fraction: float


@dataclass(frozen=True)
class ResidueGraph:
    """A k-NN residue graph with padded neighbour lists.

    ``neighbours`` is ``(n_res, k)`` of residue indices, ``edge_mask`` marks the
    real entries, and ``envelope`` is the smooth cutoff weight per edge. Messages
    must be multiplied by ``envelope`` (and masked by ``edge_mask``) or the
    smoothness is lost.
    """

    neighbours: np.ndarray
    edge_mask: np.ndarray
    envelope: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    feature_slices: dict[str, slice] = field(default_factory=dict)
    stats: GraphStats | None = None

    @property
    def n_residues(self) -> int:
        return self.neighbours.shape[0]

    @property
    def k(self) -> int:
        return self.neighbours.shape[1]


def _rbf(distances: np.ndarray) -> np.ndarray:
    """Expand distances onto Gaussian radial bases; appends a trailing axis."""
    centres = np.linspace(RBF_MIN, RBF_MAX, RBF_COUNT)
    sigma = (RBF_MAX - RBF_MIN) / RBF_COUNT
    z = (distances[..., None] - centres) / sigma
    return np.exp(-(z**2))


def cosine_envelope(distances: np.ndarray, cutoff: float) -> np.ndarray:
    """Smooth 1 -> 0 weight, continuous in value and first derivative at ``cutoff``.

    A neighbour crossing the cutoff therefore fades out rather than vanishing,
    which is what keeps the rebuilt-every-step graph from injecting a jump into
    the gradient. ``d/dr`` is zero at ``r = cutoff`` because ``sin(pi) == 0``.
    """
    scaled = np.clip(np.asarray(distances, dtype=np.float64) / cutoff, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * scaled))


def backbone_frames(protein: Protein) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-residue orthonormal frame from N, CA, C.

    Returns ``(rotations, origins, has_frame)``. ``rotations[i]`` has the frame's
    basis vectors as columns, so ``rotations[i].T @ (p - origins[i])`` takes a
    global point into residue ``i``'s local frame -- the operation that makes a
    feature invariant.
    """
    n_res = len(protein)
    rotations = np.zeros((n_res, 3, 3))
    origins = np.zeros((n_res, 3))
    has_frame = np.zeros(n_res, dtype=bool)

    for i, res in enumerate(protein.residues):
        idx = res.index
        if not all(a in idx for a in ("N", "CA", "C")):
            continue
        n, ca, c = (res.coords[idx[a]] for a in ("N", "CA", "C"))
        x = n - ca
        nx = np.linalg.norm(x)
        y_raw = c - ca
        if nx < 1e-8 or np.linalg.norm(y_raw) < 1e-8:
            continue
        x = x / nx
        y = y_raw - np.dot(y_raw, x) * x
        ny = np.linalg.norm(y)
        if ny < 1e-8:
            continue
        y = y / ny
        rotations[i] = np.stack([x, y, np.cross(x, y)], axis=-1)
        origins[i] = ca
        has_frame[i] = True

    return rotations, origins, has_frame


def _virtual_cb(res) -> np.ndarray:
    idx = res.index
    n, ca, c = (res.coords[idx[a]] for a in ("N", "CA", "C"))
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return _VCB[0] * a + _VCB[1] * b + _VCB[2] * cc + ca


def frame_atom_coords(protein: Protein) -> np.ndarray:
    """``(n_res, 5, 3)`` coordinates of N, CA, C, O and CB, CB virtual if absent.

    These are backbone-derived and so are *constant* across a rollout while the
    backbone is fixed. Kept separate from the side-chain summary for that
    reason: a Torch port can cache this block once per structure.
    """
    out = np.full((len(protein), len(FRAME_ATOMS), 3), np.nan)
    for i, res in enumerate(protein.residues):
        idx = res.index
        for j, name in enumerate(FRAME_ATOMS):
            if name in idx:
                out[i, j] = res.coords[idx[name]]
            elif name == "CB" and all(a in idx for a in ("N", "CA", "C")):
                out[i, j] = _virtual_cb(res)
            elif "CA" in idx:
                out[i, j] = res.coords[idx["CA"]]
    return out


def sidechain_summary(protein: Protein) -> np.ndarray:
    """``(n_res, 2, 3)``: side-chain centroid, and the mean of the most distal atoms.

    This is the part of the perception that *moves*. Two points rather than one
    because a centroid alone cannot distinguish a long side chain folded back
    from a short one -- the tip carries the reach.

    The tip is ``rc.SIDECHAIN_TIP_ATOMS``, chosen by bond count from CA rather
    than by distance: a distance-based argmax can change identity mid-rotation,
    which would put a discontinuity in the features. Averaging tied tips has a
    useful side effect -- it makes this summary invariant under the 180-degree
    flip of the two-fold symmetric chis (ASP/GLU tips average over the swapped
    pair; PHE/TYR tips sit on the rotation axis), so the features cannot leak
    which of two physically identical states a symmetric chi is in.
    """
    out = np.zeros((len(protein), 2, 3))
    for i, res in enumerate(protein.residues):
        idx = res.index
        body = [idx[n] for n in rc.sidechain_atoms(res.resname) - {"CB"} if n in idx]
        tips = [idx[n] for n in rc.SIDECHAIN_TIP_ATOMS[res.resname] if n in idx]

        fallback = res.coords[idx["CB"]] if "CB" in idx else (
            res.coords[idx["CA"]] if "CA" in idx else res.coords.mean(axis=0)
        )
        out[i, 0] = res.coords[body].mean(axis=0) if body else fallback
        out[i, 1] = res.coords[tips].mean(axis=0) if tips else fallback
    return out


def residue_distances(protein: Protein, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Minimum heavy-atom distance between residue pairs within ``cutoff``."""
    coords = np.concatenate([r.coords for r in protein.residues])
    owner = np.concatenate(
        [np.full(len(r.atom_names), i) for i, r in enumerate(protein.residues)]
    )
    return residue_distances_from_atoms(coords, owner, len(protein), cutoff)


def residue_distances_from_atoms(
    coords: np.ndarray, owner: np.ndarray, n_res: int, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum heavy-atom distance between residue pairs within ``cutoff``.

    Returns ``(pairs, distances)``. Uses every heavy atom, side chains included
    -- that is what makes the graph move between steps.

    Takes flat arrays rather than a :class:`Protein` so the Torch path can share
    it: neighbour *selection* is identical in both, and a second copy of this
    logic would be free to drift from the verified one.
    """
    tree = cKDTree(coords)
    atom_pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if len(atom_pairs) == 0:
        return np.zeros((0, 2), dtype=int), np.zeros(0)

    ri, rj = owner[atom_pairs[:, 0]], owner[atom_pairs[:, 1]]
    keep = ri != rj
    ri, rj = ri[keep], rj[keep]
    d = np.linalg.norm(
        coords[atom_pairs[keep, 0]] - coords[atom_pairs[keep, 1]], axis=-1
    )

    lo, hi = np.minimum(ri, rj), np.maximum(ri, rj)
    flat = lo * n_res + hi
    uniq, inverse = np.unique(flat, return_inverse=True)
    best = np.full(len(uniq), np.inf)
    np.minimum.at(best, inverse, d)
    return np.stack([uniq // n_res, uniq % n_res], axis=-1), best


def sequence_neighbour_sets(
    chain_ids: list[str], res_ids: list[int]
) -> list[set[int]]:
    """Covalently bonded sequence neighbours, which must never be dropped.

    Same chain and contiguous numbering only. Across a chain break i and i+1 are
    not bonded, and forcing an edge there would assert a contact that is not
    real.
    """
    n_res = len(chain_ids)
    forced: list[set[int]] = [set() for _ in range(n_res)]
    for i in range(n_res - 1):
        if chain_ids[i] == chain_ids[i + 1] and res_ids[i + 1] - res_ids[i] == 1:
            forced[i].add(i + 1)
            forced[i + 1].add(i)
    return forced


def select_neighbours(
    n_res: int,
    pairs: np.ndarray,
    dists: np.ndarray,
    forced: list[set[int]],
    k: int,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pick up to ``k`` neighbours per residue, forced edges first.

    Returns ``(neighbours, edge_mask, edge_dist, k_bound)``. Shared with the
    Torch path. ``edge_dist`` is only used for the NumPy envelope -- the Torch
    path recomputes it differentiably from the selected pairs, so that gradients
    reach the geometry through the envelope weight.
    """
    candidates: list[list[tuple[float, int]]] = [[] for _ in range(n_res)]
    for (i, j), d in zip(pairs, dists):
        candidates[i].append((d, j))
        candidates[j].append((d, i))

    neighbours = np.zeros((n_res, k), dtype=np.int64)
    edge_mask = np.zeros((n_res, k), dtype=bool)
    edge_dist = np.zeros((n_res, k))
    k_bound = np.zeros(n_res, dtype=bool)

    for i in range(n_res):
        by_index = {j: d for d, j in sorted(candidates[i])}
        for j in forced[i]:
            by_index.setdefault(j, cutoff)
        ordered = sorted(
            by_index.items(), key=lambda kv: (kv[0] not in forced[i], kv[1])
        )
        k_bound[i] = len(ordered) > k
        for slot, (j, d) in enumerate(ordered[:k]):
            neighbours[i, slot] = j
            edge_dist[i, slot] = d
            edge_mask[i, slot] = True

    return neighbours, edge_mask, edge_dist, k_bound


def build_graph(
    protein: Protein,
    k: int = 64,
    cutoff: float = 10.0,
    include_sequence_neighbours: bool = True,
) -> ResidueGraph:
    """Build the per-step perception graph for ``protein``'s current coordinates.

    ``k`` must be large enough that the distance cutoff, not ``k``, is what
    excludes neighbours -- otherwise edges are dropped by the k limit, which is
    discontinuous however smooth the envelope is, and the per-step rebuild
    reintroduces the gradient jump the envelope exists to remove. Always check
    ``graph.stats.k_binding_fraction``.

    **The default of 64 is measured, not guessed** (``scripts/graph_report.py``,
    10 A cutoff). ProteinMPNN's customary k of 32 is not enough here, because a
    cutoff over *all heavy atoms* admits more neighbours than a CA- or CB-based
    one does:

    ======  ====  ====  ====  ====
    k        16    24    32    64
    ======  ====  ====  ====  ====
    193L     98%   81%   48%    0%
    1UBQ     89%   62%   25%    0%
    3NIR     85%   48%   11%    0%
    ======  ====  ====  ====  ====

    Those are k_binding_fraction values. Observed degree saturates at ~34, so 64
    is headroom rather than cost in the neighbour count; it does cost memory in
    the edge-feature tensor.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    n_res = len(protein)
    pairs, dists = residue_distances(protein, cutoff)
    forced = (
        sequence_neighbour_sets(protein.chain_ids, protein.res_ids)
        if include_sequence_neighbours
        else [set() for _ in range(n_res)]
    )
    neighbours, edge_mask, edge_dist, k_bound = select_neighbours(
        n_res, pairs, dists, forced, k, cutoff
    )

    envelope = np.where(edge_mask, cosine_envelope(edge_dist, cutoff), 0.0)

    node_features = _node_features(protein)
    edge_features, slices = _edge_features(protein, neighbours, edge_mask, envelope)

    _rot, _org, has_frame = backbone_frames(protein)
    stats = GraphStats(
        n_residues=n_res,
        mean_degree=float(edge_mask.sum(axis=1).mean()) if n_res else 0.0,
        k_binding_fraction=float(k_bound.mean()) if n_res else 0.0,
        frameless_fraction=float((~has_frame).mean()) if n_res else 0.0,
    )
    return ResidueGraph(
        neighbours=neighbours,
        edge_mask=edge_mask,
        envelope=envelope,
        node_features=node_features,
        edge_features=edge_features,
        feature_slices=slices,
        stats=stats,
    )


def _node_features(protein: Protein) -> np.ndarray:
    """``(n_res, 32)``: residue one-hot, chi as sin/cos pairs, and the chi mask.

    Residue identity is an *input*: this is a repacker and the sequence is known.
    The chis go in as sin/cos so the representation is continuous across the
    +-pi wrap; a raw angle would put a discontinuity in the middle of the input.
    """
    aa_index = {name: i for i, name in enumerate(rc.AA3)}
    out = np.zeros((len(protein), len(rc.AA3) + 3 * rc.MAX_CHI))
    for i, res in enumerate(protein.residues):
        out[i, aa_index[res.resname]] = 1.0
        chis, mask = measure_chis(res)
        base = len(rc.AA3)
        safe = np.where(mask, np.nan_to_num(chis), 0.0)
        out[i, base : base + rc.MAX_CHI] = np.where(mask, np.sin(safe), 0.0)
        out[i, base + rc.MAX_CHI : base + 2 * rc.MAX_CHI] = np.where(
            mask, np.cos(safe), 0.0
        )
        out[i, base + 2 * rc.MAX_CHI :] = mask.astype(np.float64)
    return out


def _edge_features(
    protein: Protein,
    neighbours: np.ndarray,
    edge_mask: np.ndarray,
    envelope: np.ndarray,
) -> tuple[np.ndarray, dict[str, slice]]:
    """Assemble the invariant edge features. Every block is rotation-invariant."""
    n_res, k = neighbours.shape
    frame_xyz = frame_atom_coords(protein)
    side_xyz = sidechain_summary(protein)
    rotations, origins, has_frame = backbone_frames(protein)

    nb = neighbours

    # Block 1: the 5x5 backbone/CB distance matrix, RBF-encoded. Backbone-derived,
    # so constant across a rollout for a fixed neighbour list -- and it is 400 of
    # the 575 edge dimensions. A Torch port should cache it per structure and
    # recompute only the side-chain block; that is most of the per-step cost.
    diff = frame_xyz[:, None, :, None, :] - frame_xyz[nb][:, :, None, :, :]
    bb_dist = np.linalg.norm(diff, axis=-1).reshape(n_res, k, -1)
    bb_block = _rbf(np.nan_to_num(bb_dist, nan=RBF_MAX)).reshape(n_res, k, -1)

    # Block 2: side-chain summary distances. This is the block that changes as
    # the rule works -- the mutable input to the repacker.
    mine = np.concatenate([side_xyz, frame_xyz[:, 1:2], frame_xyz[:, 4:5]], axis=1)
    theirs = np.concatenate(
        [side_xyz[nb], frame_xyz[nb][:, :, 1:2], frame_xyz[nb][:, :, 4:5]], axis=2
    )
    sc_diff = mine[:, None, :2, None, :] - theirs[:, :, None, :, :]
    sc_dist = np.linalg.norm(sc_diff, axis=-1).reshape(n_res, k, -1)
    sc_block = _rbf(np.nan_to_num(sc_dist, nan=RBF_MAX)).reshape(n_res, k, -1)

    # Block 3: relative orientation R_i^T R_j, invariant under a global rotation.
    rel_rot = np.einsum("iab,ikac->ikbc", rotations, rotations[nb]).reshape(n_res, k, 9)

    # Block 4: the neighbour's CA direction in residue i's own frame. A unit
    # vector in a local frame is invariant; the same vector in global
    # coordinates would not be, and using it is the documented failure mode.
    delta = frame_xyz[nb][:, :, 1, :] - origins[:, None, :]
    local = np.einsum("iab,ikb->ika", np.transpose(rotations, (0, 2, 1)), delta)
    norm = np.linalg.norm(local, axis=-1, keepdims=True)
    direction = local / np.maximum(norm, 1e-8)

    # Block 5: sequence offset, clipped, plus a same-chain flag.
    offsets = np.asarray(protein.res_ids)[nb] - np.asarray(protein.res_ids)[:, None]
    clipped = np.clip(offsets, -16, 16) + 16
    offset_onehot = np.zeros((n_res, k, 33))
    np.put_along_axis(offset_onehot, clipped[..., None], 1.0, axis=-1)
    chains = np.asarray(protein.chain_ids)
    same_chain = (chains[nb] == chains[:, None]).astype(np.float64)[..., None]

    blocks = [
        ("backbone_rbf", bb_block),
        ("sidechain_rbf", sc_block),
        ("relative_rotation", rel_rot),
        ("local_direction", direction),
        ("sequence_offset", offset_onehot),
        ("same_chain", same_chain),
        ("envelope", envelope[..., None]),
    ]

    slices: dict[str, slice] = {}
    start = 0
    for name, block in blocks:
        slices[name] = slice(start, start + block.shape[-1])
        start += block.shape[-1]

    features = np.concatenate([b for _n, b in blocks], axis=-1)
    # A padded slot must carry no signal, and a frameless residue's orientation
    # features are meaningless rather than merely noisy.
    features = np.where(edge_mask[..., None], features, 0.0)
    features = np.where(has_frame[:, None, None], features, 0.0)
    return features, slices
