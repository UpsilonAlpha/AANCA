"""The perception graph in Torch, differentiable where it has to be.

``graph.py`` is the verified NumPy reference; this is the training path, and
``tests/test_torch_graph.py`` pins the two together.

Where the gradient runs, and where it deliberately does not:

* **Neighbour selection is not differentiable and cannot be.** It is an argsort
  over distances -- a discrete choice with no useful derivative. It runs on
  detached coordinates through the shared selection helpers in ``graph.py``, so
  the two paths cannot pick different neighbours.
* **The envelope is differentiable, and is the gradient path for graph
  structure.** Once the neighbour list is fixed, the minimum heavy-atom distance
  for each selected pair is recomputed *with* gradient. So moving a side chain
  changes how strongly its edges contribute, and that is what the optimiser
  feels. Without this the envelope would be decoration.
* **Everything built from coordinates is differentiable**: the side-chain
  summary, the distance blocks, the local-frame direction.
* **The backbone block is constant across a rollout** because the backbone is
  fixed. It is 400 of the 575 edge dimensions, so :func:`backbone_edge_block`
  is separated out to be computed once per neighbour list rather than per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from . import graph as np_graph
from . import residue_constants as rc
from .graph import FRAME_ATOMS, RBF_COUNT, RBF_MAX, RBF_MIN
from .structure import Protein
from .torch_geometry import MAX_ATOMS, TorsionTopology, measure_chis, topology_from_protein

EPS = 1e-8

#: Matches ``graph.py``'s block layout exactly.
EDGE_DIM = 25 * RBF_COUNT + 8 * RBF_COUNT + 9 + 3 + 33 + 1 + 1
NODE_DIM = len(rc.AA3) + 3 * rc.MAX_CHI


@dataclass(frozen=True)
class FeatureTopology:
    """Static index tensors for featurisation. Depends on sequence, not coordinates."""

    torsion: TorsionTopology
    atom_mask: Tensor
    frame_idx: Tensor
    frame_valid: Tensor
    virtual_cb: Tensor
    body_mask: Tensor
    tip_mask: Tensor
    fallback_idx: Tensor
    has_frame: Tensor
    aa_onehot: Tensor
    res_ids: Tensor
    chain_code: Tensor
    owner: np.ndarray
    chain_ids: list[str]
    res_id_list: list[int]
    #: ``(n_res,)`` bool: residues whose perception is complete and which the
    #: loss may therefore touch. All true for an intact structure; false on the
    #: boundary shell of a spatial crop, where neighbours were deleted and the
    #: native torsion depends on contacts the rule cannot see (see ``crop.py``).
    #:
    #: It rides on the topology rather than being passed alongside it because
    #: the failure mode is one-sided: forgetting the mask supervises unlearnable
    #: targets and quietly degrades training, while carrying it costs nothing.
    #: Defaults to all-true so every existing caller keeps its behaviour.
    supervised: Tensor | None = None

    @property
    def n_residues(self) -> int:
        return self.atom_mask.shape[0]

    @property
    def supervised_mask(self) -> Tensor:
        """``supervised``, materialised as all-true when unset."""
        if self.supervised is not None:
            return self.supervised
        return torch.ones(
            self.n_residues, dtype=torch.bool, device=self.atom_mask.device
        )


def feature_topology(
    protein: Protein,
    device: torch.device | str = "cpu",
    supervised: np.ndarray | Tensor | None = None,
) -> tuple[FeatureTopology, Tensor]:
    """Resolve ``protein`` into a featurisation topology plus padded coordinates.

    ``supervised`` is the crop mask from :func:`crop.crop`; ``None`` means an
    intact structure, where every residue may be supervised.
    """
    torsion, coords = topology_from_protein(protein, device=device)
    n_res = len(protein)

    frame_idx = torch.zeros(n_res, len(FRAME_ATOMS), dtype=torch.long)
    frame_valid = torch.zeros(n_res, len(FRAME_ATOMS), dtype=torch.bool)
    virtual_cb = torch.zeros(n_res, dtype=torch.bool)
    body_mask = torch.zeros(n_res, MAX_ATOMS, dtype=torch.bool)
    tip_mask = torch.zeros(n_res, MAX_ATOMS, dtype=torch.bool)
    fallback_idx = torch.zeros(n_res, dtype=torch.long)
    has_frame = torch.zeros(n_res, dtype=torch.bool)
    aa_onehot = torch.zeros(n_res, len(rc.AA3))

    aa_index = {name: i for i, name in enumerate(rc.AA3)}
    chain_lookup: dict[str, int] = {}

    for r, res in enumerate(protein.residues):
        index = res.index
        aa_onehot[r, aa_index[res.resname]] = 1.0
        has_frame[r] = all(a in index for a in ("N", "CA", "C"))

        for j, name in enumerate(FRAME_ATOMS):
            # Same fallback chain as the NumPy featuriser: the atom itself, then
            # a virtual CB, then CA.
            if name in index:
                frame_idx[r, j] = index[name]
                frame_valid[r, j] = True
            elif name == "CB" and has_frame[r]:
                virtual_cb[r] = True
                frame_valid[r, j] = True
            elif "CA" in index:
                frame_idx[r, j] = index["CA"]
                frame_valid[r, j] = True

        for name in rc.sidechain_atoms(res.resname) - {"CB"}:
            if name in index:
                body_mask[r, index[name]] = True
        for name in rc.SIDECHAIN_TIP_ATOMS[res.resname]:
            if name in index:
                tip_mask[r, index[name]] = True

        if "CB" in index:
            fallback_idx[r] = index["CB"]
        elif "CA" in index:
            fallback_idx[r] = index["CA"]

        chain_lookup.setdefault(protein.chain_ids[r], len(chain_lookup))

    owner = np.repeat(
        np.arange(n_res), [len(r.atom_names) for r in protein.residues]
    )
    topo = FeatureTopology(
        torsion=torsion,
        atom_mask=torsion.atom_mask.to(device),
        frame_idx=frame_idx.to(device),
        frame_valid=frame_valid.to(device),
        virtual_cb=virtual_cb.to(device),
        body_mask=body_mask.to(device),
        tip_mask=tip_mask.to(device),
        fallback_idx=fallback_idx.to(device),
        has_frame=has_frame.to(device),
        aa_onehot=aa_onehot.to(device),
        res_ids=torch.tensor(protein.res_ids, dtype=torch.long, device=device),
        chain_code=torch.tensor(
            [chain_lookup[c] for c in protein.chain_ids], dtype=torch.long, device=device
        ),
        owner=owner,
        chain_ids=list(protein.chain_ids),
        res_id_list=list(protein.res_ids),
        supervised=_supervised_tensor(supervised, n_res, device),
    )
    return topo, coords


def _supervised_tensor(
    supervised: np.ndarray | Tensor | None, n_res: int, device
) -> Tensor | None:
    if supervised is None:
        return None
    mask = torch.as_tensor(np.asarray(supervised, dtype=bool))
    if mask.shape != (n_res,):
        raise ValueError(
            f"supervised mask must be ({n_res},), got {tuple(mask.shape)} -- a "
            "mask built for the uncropped structure would silently supervise "
            "the wrong residues"
        )
    return mask.to(device)


# -- geometry blocks -------------------------------------------------------


def _rows(n: int, device) -> Tensor:
    return torch.arange(n, device=device).unsqueeze(-1)


def frame_atom_coords(coords: Tensor, topo: FeatureTopology) -> Tensor:
    """``(n_res, 5, 3)`` for N, CA, C, O, CB, with CB constructed where absent.

    Backbone-derived, so constant across a rollout while the backbone is fixed.
    """
    picked = coords[_rows(coords.shape[0], coords.device), topo.frame_idx]
    if bool(topo.virtual_cb.any()):
        n, ca, c = picked[:, 0], picked[:, 1], picked[:, 2]
        b, cc = ca - n, c - ca
        a = torch.cross(b, cc, dim=-1)
        cb = (
            np_graph._VCB[0] * a
            + np_graph._VCB[1] * b
            + np_graph._VCB[2] * cc
            + ca
        )
        picked = torch.cat(
            [
                picked[:, :4],
                torch.where(topo.virtual_cb.unsqueeze(-1), cb, picked[:, 4]).unsqueeze(1),
            ],
            dim=1,
        )
    return picked


def _masked_mean(coords: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
    count = mask.sum(dim=-1, keepdim=True).to(coords.dtype)
    total = (coords * mask.unsqueeze(-1).to(coords.dtype)).sum(dim=-2)
    mean = total / count.clamp_min(1.0)
    return torch.where(count > 0, mean, fallback)


def sidechain_summary(coords: Tensor, topo: FeatureTopology) -> Tensor:
    """``(n_res, 2, 3)``: side-chain centroid and the mean of the most distal atoms.

    The part of the perception that moves, and the reason the per-step rebuild
    is not a no-op. Differentiable.
    """
    fallback = coords[_rows(coords.shape[0], coords.device), topo.fallback_idx.unsqueeze(-1)][:, 0]
    body = _masked_mean(coords, topo.body_mask, fallback)
    tip = _masked_mean(coords, topo.tip_mask, fallback)
    return torch.stack([body, tip], dim=1)


def backbone_frames(coords: Tensor, topo: FeatureTopology) -> tuple[Tensor, Tensor]:
    """``(rotations, origins)``; columns of each rotation are the frame's basis."""
    frame = frame_atom_coords(coords, topo)
    n, ca, c = frame[:, 0], frame[:, 1], frame[:, 2]
    x = n - ca
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    y_raw = c - ca
    y = y_raw - (y_raw * x).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(EPS)
    rot = torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)
    rot = torch.where(topo.has_frame[:, None, None], rot, torch.zeros_like(rot))
    return rot, ca


def pair_min_distances(
    coords: Tensor, topo: FeatureTopology, neighbours: Tensor
) -> Tensor:
    """``(n_res, k)`` minimum heavy-atom distance for each selected pair.

    Differentiable, and this is the envelope's gradient path.

    The argmin is found under ``no_grad`` over the full ``(R, K, A, A)`` block
    and only the winning atom pair is then measured with gradient. That is
    exact -- the derivative of a min is the derivative of its argmin -- and it
    keeps the retained-for-backward tensors at ``(R, K, 3)`` instead of
    ``(R, K, A, A, 3)``, which matters a lot once BPTT stores one per step.
    """
    rows = _rows(coords.shape[0], coords.device)
    with torch.no_grad():
        mine = coords.unsqueeze(1).unsqueeze(3)
        theirs = coords[neighbours].unsqueeze(2)
        d = (mine - theirs).norm(dim=-1)
        valid = topo.atom_mask[:, None, :, None] & topo.atom_mask[neighbours][:, :, None, :]
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        flat = d.flatten(start_dim=2)
        best = flat.argmin(dim=-1)
        ai = best // MAX_ATOMS
        aj = best % MAX_ATOMS
        unreachable = ~valid.flatten(start_dim=2).any(dim=-1)

    pi = coords[rows, ai]
    pj = coords[neighbours.reshape(-1), aj.reshape(-1)].reshape(*ai.shape, 3)
    out = (pi - pj).norm(dim=-1)
    return torch.where(unreachable, torch.full_like(out, RBF_MAX), out)


# -- neighbour selection (not differentiable, by nature) -------------------


def select_neighbours(
    coords: Tensor,
    topo: FeatureTopology,
    k: int = 64,
    cutoff: float = 10.0,
    include_sequence_neighbours: bool = True,
) -> tuple[Tensor, Tensor]:
    """Pick neighbours on detached coordinates, via the shared NumPy helpers."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    flat = coords.detach().cpu().numpy()[topo.atom_mask.cpu().numpy()]
    pairs, dists = np_graph.residue_distances_from_atoms(
        flat, topo.owner, topo.n_residues, cutoff
    )
    forced = (
        np_graph.sequence_neighbour_sets(topo.chain_ids, topo.res_id_list)
        if include_sequence_neighbours
        else [set() for _ in range(topo.n_residues)]
    )
    neighbours, edge_mask, _dist, _bound = np_graph.select_neighbours(
        topo.n_residues, pairs, dists, forced, k, cutoff
    )
    return (
        torch.from_numpy(neighbours).to(coords.device),
        torch.from_numpy(edge_mask).to(coords.device),
    )


def sequence_links(topo: FeatureTopology, device) -> Tensor:
    """``(n_res,)`` -- is residue ``i`` covalently bonded to ``i + 1``?

    The tensor form of :func:`graph.sequence_neighbour_sets`, and a property of
    the topology rather than of the coordinates, so a caller that rebuilds the
    graph every step should hoist it out of the loop.
    """
    chain_ids, res_ids = topo.chain_ids, topo.res_id_list
    links = [
        chain_ids[i] == chain_ids[i + 1] and res_ids[i + 1] - res_ids[i] == 1
        for i in range(len(chain_ids) - 1)
    ]
    return torch.tensor(links, dtype=torch.bool, device=device)


def select_neighbours_device(
    coords: Tensor,
    topo: FeatureTopology,
    k: int = 64,
    cutoff: float = 10.0,
    include_sequence_neighbours: bool = True,
    links: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """:func:`select_neighbours` without leaving the device.

    **Why this exists.** Profiled on an MI355X at the 384-residue crop training
    uses, the NumPy version is 29.72 ms of a 37.27 ms rollout step -- 80% --
    while the rule's own forward pass is 0.34 ms. It is not slow code; it is
    code in the wrong place, because ``coords.detach().cpu().numpy()`` forces a
    device transfer and a synchronisation eighty times per update and the
    search itself cannot use the card at all.

    **It must agree with the NumPy version exactly**, which is what
    ``tests/test_torch_graph.py`` pins: a different graph is a different
    experiment, and one that would show up as nothing more than a run that
    scores slightly differently. Two details carry that agreement and are easy
    to lose:

    * the ordering is *forced neighbours first, then by distance*, and a forced
      pair that is beyond the cutoff enters at exactly ``cutoff``;
    * ties are broken by residue index, via a stable sort. In the NumPy version
      that falls out of dict insertion order; here it is asked for.

    ``links`` is :func:`sequence_links`, passed in when the caller has hoisted
    it out of a rollout loop. ``compute_mode`` is pinned to the direct
    formulation because the matmul one is not exact, and a pair sitting within
    an ulp of the cutoff would otherwise be admitted by one implementation and
    refused by the other.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    device = coords.device
    n_res = topo.n_residues
    atoms = coords.detach()[topo.atom_mask]
    owner = torch.as_tensor(topo.owner, device=device, dtype=torch.long)
    n_atoms = atoms.shape[0]

    distances = torch.cdist(
        atoms, atoms, compute_mode="donot_use_mm_for_euclid_dist"
    )
    infinity = torch.tensor(float("inf"), dtype=distances.dtype, device=device)
    # Minimum heavy-atom distance, reduced twice: over the atoms of the first
    # residue, then over those of the second.
    by_residue = torch.full(
        (n_res, n_atoms), float("inf"), dtype=distances.dtype, device=device
    ).scatter_reduce(
        0, owner.unsqueeze(-1).expand(-1, n_atoms), distances, "amin"
    )
    pair_distance = torch.full(
        (n_res, n_res), float("inf"), dtype=distances.dtype, device=device
    ).scatter_reduce(
        1, owner.unsqueeze(0).expand(n_res, -1), by_residue, "amin"
    )
    pair_distance.fill_diagonal_(float("inf"))

    forced = torch.zeros((n_res, n_res), dtype=torch.bool, device=device)
    if include_sequence_neighbours and n_res > 1:
        if links is None:
            links = sequence_links(topo, device)
        index = torch.arange(n_res - 1, device=device)
        forced[index, index + 1] = links
        forced[index + 1, index] = links

    within = pair_distance <= cutoff
    candidate = within | forced
    # A forced pair beyond the cutoff enters at the cutoff itself, which is
    # ``setdefault(j, cutoff)`` in the NumPy version.
    effective = torch.where(within, pair_distance, torch.full_like(pair_distance, cutoff))
    # Forced first, then distance, as one sort key -- and **in float64**. The
    # offset has to exceed any distance that survives the cutoff, and in
    # float32 the ulp at 1e6 is 0.0625, so adding it would round every
    # distance into 6-hundredths-of-an-Angstrom buckets and scramble the order
    # inside each one. That is not a tie-break detail: it reordered real
    # neighbours by a tenth of an Angstrom. float64 holds 1e6 plus a float32
    # distance exactly, since 20 bits of exponent and 24 of mantissa fit in 53.
    offset = torch.where(forced, 0.0, 1.0e6).to(torch.float64)
    key = torch.where(
        candidate,
        effective.to(torch.float64) + offset,
        torch.tensor(float("inf"), dtype=torch.float64, device=device),
    )

    order = torch.argsort(key, dim=-1, stable=True)[:, :k]
    chosen = torch.gather(key, 1, order)
    edge_mask = torch.isfinite(chosen)
    neighbours = order * edge_mask
    return neighbours, edge_mask


# -- features --------------------------------------------------------------


def rbf(distances: Tensor) -> Tensor:
    """Gaussian radial bases; appends a trailing axis of ``RBF_COUNT``."""
    centres = torch.linspace(
        RBF_MIN, RBF_MAX, RBF_COUNT, dtype=distances.dtype, device=distances.device
    )
    sigma = (RBF_MAX - RBF_MIN) / RBF_COUNT
    return torch.exp(-(((distances.unsqueeze(-1) - centres) / sigma) ** 2))


def cosine_envelope(distances: Tensor, cutoff: float) -> Tensor:
    """Smooth 1 -> 0 weight, continuous in value and first derivative at ``cutoff``."""
    scaled = (distances / cutoff).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.pi * scaled))


def node_features(coords: Tensor, topo: FeatureTopology) -> Tensor:
    """``(n_res, 32)``: residue one-hot, chis as sin/cos pairs, and the chi mask.

    Chis go in as sin/cos so the representation is continuous across the +-pi
    wrap; a raw angle would put a discontinuity in the middle of the input.
    """
    chis = measure_chis(coords, topo.torsion)
    mask = topo.torsion.chi_mask.to(coords.dtype)
    return torch.cat(
        [
            topo.aa_onehot.to(coords.dtype),
            torch.sin(chis) * mask,
            torch.cos(chis) * mask,
            mask,
        ],
        dim=-1,
    )


def backbone_edge_block(
    coords: Tensor, topo: FeatureTopology, neighbours: Tensor, fixed_backbone: bool = True
) -> Tensor:
    """``(n_res, k, 400)``: the 5x5 backbone/CB distance matrix, RBF-encoded.

    With ``fixed_backbone`` this is built under ``no_grad``, which is the single
    biggest memory saving available: 400 of the 575 edge dimensions never enter
    the autograd graph. It is exact, not an approximation -- only side chains
    rotate, and the backbone is excluded from every chi's downstream atom set,
    so the gradient of this block with respect to the moving coordinates is
    identically zero.

    **Detaching it afterwards does not work and was measured not to.** By then
    the full tensor has already been built inside the graph and saved for
    backward; a later ``.detach()`` changes the output's grad flag but frees
    nothing. The block has to not be *computed* under grad in the first place.

    ``fixed_backbone=False`` must be passed once the backbone-remodelling
    escalation lands, or those gradients will be silently zero.
    """
    context = torch.no_grad() if fixed_backbone else torch.enable_grad()
    with context:
        frame = frame_atom_coords(coords, topo)
        diff = frame[:, None, :, None, :] - frame[neighbours][:, :, None, :, :]
        dist = diff.norm(dim=-1)
        valid = (
            topo.frame_valid[:, None, :, None]
            & topo.frame_valid[neighbours][:, :, None, :]
        )
        dist = torch.where(valid, dist, torch.full_like(dist, RBF_MAX))
        n_res, k = neighbours.shape
        return rbf(dist.reshape(n_res, k, -1)).reshape(n_res, k, -1)


def edge_features(
    coords: Tensor,
    topo: FeatureTopology,
    neighbours: Tensor,
    edge_mask: Tensor,
    cutoff: float = 10.0,
    backbone_block: Tensor | None = None,
    fixed_backbone: bool = True,
) -> tuple[Tensor, Tensor, dict[str, slice]]:
    """``(edge_features, envelope, slices)``. Block order matches ``graph.py``."""
    n_res, k = neighbours.shape
    frame = frame_atom_coords(coords, topo)
    side = sidechain_summary(coords, topo)
    rot, origin = backbone_frames(coords, topo)

    bb_block = (
        backbone_edge_block(coords, topo, neighbours, fixed_backbone)
        if backbone_block is None
        else backbone_block
    )

    # Side-chain block: the two moving points of i against four points of j.
    mine = side
    theirs = torch.cat(
        [side[neighbours], frame[neighbours][:, :, 1:2], frame[neighbours][:, :, 4:5]],
        dim=2,
    )
    sc_dist = (mine[:, None, :, None, :] - theirs[:, :, None, :, :]).norm(dim=-1)
    sc_block = rbf(sc_dist.reshape(n_res, k, -1)).reshape(n_res, k, -1)

    rel_rot = torch.einsum("iab,ikac->ikbc", rot, rot[neighbours]).reshape(n_res, k, 9)

    delta = frame[neighbours][:, :, 1, :] - origin[:, None, :]
    local = torch.einsum("iab,ikb->ika", rot.transpose(-1, -2), delta)
    direction = local / local.norm(dim=-1, keepdim=True).clamp_min(EPS)

    offsets = topo.res_ids[neighbours] - topo.res_ids[:, None]
    clipped = offsets.clamp(-16, 16) + 16
    offset_onehot = torch.zeros(n_res, k, 33, dtype=coords.dtype, device=coords.device)
    offset_onehot.scatter_(-1, clipped.unsqueeze(-1), 1.0)
    same_chain = (
        (topo.chain_code[neighbours] == topo.chain_code[:, None])
        .to(coords.dtype)
        .unsqueeze(-1)
    )

    envelope = cosine_envelope(pair_min_distances(coords, topo, neighbours), cutoff)
    envelope = torch.where(edge_mask, envelope, torch.zeros_like(envelope))

    blocks = [
        ("backbone_rbf", bb_block),
        ("sidechain_rbf", sc_block),
        ("relative_rotation", rel_rot),
        ("local_direction", direction),
        ("sequence_offset", offset_onehot),
        ("same_chain", same_chain),
        ("envelope", envelope.unsqueeze(-1)),
    ]
    slices: dict[str, slice] = {}
    start = 0
    for name, block in blocks:
        slices[name] = slice(start, start + block.shape[-1])
        start += block.shape[-1]

    features = torch.cat([b for _n, b in blocks], dim=-1)
    features = torch.where(edge_mask.unsqueeze(-1), features, torch.zeros_like(features))
    features = torch.where(
        topo.has_frame[:, None, None], features, torch.zeros_like(features)
    )
    return features, envelope, slices


@dataclass(frozen=True)
class TorchGraph:
    """Everything one step of the rule needs."""

    neighbours: Tensor
    edge_mask: Tensor
    envelope: Tensor
    node_features: Tensor
    edge_features: Tensor
    feature_slices: dict[str, slice]


def build_graph(
    coords: Tensor,
    topo: FeatureTopology,
    k: int = 64,
    cutoff: float = 10.0,
    include_sequence_neighbours: bool = True,
    backbone_block: Tensor | None = None,
    fixed_backbone: bool = True,
    device_neighbours: bool = False,
) -> TorchGraph:
    """Select neighbours and featurise. Differentiable except for the selection.

    Featurise from ``coords.float()`` for training: float32 cuts retained
    activations by about a third (measured), and distances and RBF expansions
    do not need double precision. Keep the *torsion* arithmetic in float64,
    which is where the exactness guarantee lives.

    Pass ``backbone_block`` from a previous call with the *same* neighbour list
    to skip recomputing it. Note the neighbour list changes most steps, so this
    helps less than it looks -- ``fixed_backbone`` is the saving that matters.
    """
    select = select_neighbours_device if device_neighbours else select_neighbours
    neighbours, edge_mask = select(
        coords, topo, k, cutoff, include_sequence_neighbours
    )
    features, envelope, slices = edge_features(
        coords, topo, neighbours, edge_mask, cutoff, backbone_block, fixed_backbone
    )
    return TorchGraph(
        neighbours=neighbours,
        edge_mask=edge_mask,
        envelope=envelope,
        node_features=node_features(coords, topo),
        edge_features=features,
        feature_slices=slices,
    )
