"""The perception graph: invariant, genuinely dynamic, and smooth at the cutoff."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aanca.damage import randomise_rotamers
from aanca.geometry import rotation_matrix
from aanca.graph import (
    backbone_frames,
    build_graph,
    cosine_envelope,
    frame_atom_coords,
    residue_distances,
    sidechain_summary,
)
from aanca.structure import load_protein

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def protein():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    return load_protein(path)


@pytest.fixture(scope="module")
def graph(protein):
    return build_graph(protein, k=32, cutoff=10.0)


# -- invariance ------------------------------------------------------------
# The scientific claim: the rule is rotation- and translation-invariant because
# its features are, not because its architecture is. If these fail, the choice
# has been inherited by accident rather than made.


def test_edge_features_are_invariant_to_a_global_rigid_motion(protein, graph):
    rng = np.random.default_rng(0)
    r = rotation_matrix(rng.normal(size=3), 1.234)
    moved = protein.transformed(rotation=r, translation=rng.normal(size=3) * 25.0)
    after = build_graph(moved, k=32, cutoff=10.0)

    assert np.array_equal(graph.neighbours, after.neighbours)
    assert np.array_equal(graph.edge_mask, after.edge_mask)
    assert np.allclose(graph.envelope, after.envelope, atol=1e-9)
    assert np.allclose(graph.edge_features, after.edge_features, atol=1e-8)


def test_node_features_are_invariant_to_a_global_rigid_motion(protein, graph):
    rng = np.random.default_rng(1)
    moved = protein.transformed(
        rotation=rotation_matrix(rng.normal(size=3), -2.0),
        translation=np.array([100.0, -50.0, 7.0]),
    )
    after = build_graph(moved, k=32, cutoff=10.0)
    assert np.allclose(graph.node_features, after.node_features, atol=1e-9)


@pytest.mark.parametrize(
    "block", ["backbone_rbf", "sidechain_rbf", "relative_rotation", "local_direction"]
)
def test_every_geometric_block_is_invariant_individually(protein, graph, block):
    # Checked per block so a failure names the offending featuriser rather than
    # just the concatenation.
    rng = np.random.default_rng(2)
    moved = protein.transformed(
        rotation=rotation_matrix(rng.normal(size=3), 0.77),
        translation=rng.normal(size=3) * 10.0,
    )
    after = build_graph(moved, k=32, cutoff=10.0)
    sl = graph.feature_slices[block]
    assert np.allclose(
        graph.edge_features[..., sl], after.edge_features[..., sl], atol=1e-8
    )


def test_frames_rotate_with_the_structure(protein):
    rng = np.random.default_rng(3)
    r = rotation_matrix(rng.normal(size=3), 1.0)
    rot_a, org_a, mask_a = backbone_frames(protein)
    rot_b, org_b, mask_b = backbone_frames(protein.transformed(rotation=r))
    assert np.array_equal(mask_a, mask_b)
    # The frame is equivariant; the features built from it are invariant.
    assert np.allclose(rot_b[mask_a], r @ rot_a[mask_a], atol=1e-9)
    assert np.allclose(org_b[mask_a], org_a[mask_a] @ r.T, atol=1e-9)


# -- the graph is genuinely per-step ---------------------------------------


def test_moving_side_chains_changes_the_graph(protein):
    """Otherwise "rebuilt every step" would be an expensive no-op.

    The backbone is fixed in this milestone, so only a cutoff that sees
    side-chain atoms produces a graph that moves.
    """
    before = build_graph(protein, k=32, cutoff=10.0)
    damaged = randomise_rotamers(protein, rng=np.random.default_rng(0)).apply(protein)
    after = build_graph(damaged, k=32, cutoff=10.0)
    assert not np.array_equal(before.neighbours, after.neighbours)


def test_the_backbone_block_does_not_change_when_side_chains_move(protein):
    """The claim that justifies caching it in a Torch port."""
    damaged = randomise_rotamers(protein, rng=np.random.default_rng(0)).apply(protein)
    assert np.allclose(
        frame_atom_coords(protein), frame_atom_coords(damaged), atol=1e-9
    )


def test_the_sidechain_summary_does_change_when_side_chains_move(protein):
    damaged = randomise_rotamers(protein, rng=np.random.default_rng(0)).apply(protein)
    moved = np.abs(sidechain_summary(protein) - sidechain_summary(damaged)).max()
    assert moved > 1.0


def test_a_backbone_only_cutoff_would_have_been_static(protein):
    """Documents why neighbour selection uses all heavy atoms.

    Residue distances computed from backbone+CB atoms alone are identical before
    and after side-chain damage -- so a graph built that way could be computed
    once, and "per step" would mean nothing.
    """
    damaged = randomise_rotamers(protein, rng=np.random.default_rng(0)).apply(protein)
    bb_before = frame_atom_coords(protein)
    bb_after = frame_atom_coords(damaged)
    assert np.allclose(bb_before, bb_after, atol=1e-9)

    # Whereas the all-heavy-atom distances the builder actually uses do move.
    pairs_a, dist_a = residue_distances(protein, 10.0)
    pairs_b, dist_b = residue_distances(damaged, 10.0)
    assert not (
        pairs_a.shape == pairs_b.shape
        and np.array_equal(pairs_a, pairs_b)
        and np.allclose(dist_a, dist_b)
    )


# -- smoothness and structure ---------------------------------------------


def test_envelope_is_c1_continuous_at_the_cutoff():
    cutoff = 10.0
    assert cosine_envelope(np.array([0.0]), cutoff)[0] == pytest.approx(1.0)
    assert cosine_envelope(np.array([cutoff]), cutoff)[0] == pytest.approx(0.0)
    # Value and slope both vanish at the cutoff, so an edge fades out instead of
    # popping -- that is the whole reason the envelope exists.
    h = 1e-6
    just_inside = cosine_envelope(np.array([cutoff - h]), cutoff)[0]
    assert just_inside == pytest.approx(0.0, abs=1e-10)
    slope = (cosine_envelope(np.array([cutoff]), cutoff)[0] - just_inside) / h
    assert abs(slope) < 1e-4


def test_envelope_is_monotone_decreasing():
    d = np.linspace(0.0, 12.0, 200)
    w = cosine_envelope(d, 10.0)
    assert np.all(np.diff(w) <= 1e-12)
    assert np.all(w >= 0.0)


def test_sequence_neighbours_are_always_present(protein, graph):
    for i in range(len(protein) - 1):
        if protein.chain_ids[i] != protein.chain_ids[i + 1]:
            continue
        if protein.res_ids[i + 1] - protein.res_ids[i] != 1:
            continue
        real = graph.neighbours[i][graph.edge_mask[i]]
        assert i + 1 in real, f"residue {i} lost its sequence neighbour"


def test_sequence_neighbours_survive_a_tiny_k(protein):
    # They are forced, so even k=1 keeps one of them rather than the nearest
    # contact. A covalent bond is not negotiable.
    small = build_graph(protein, k=1, cutoff=10.0)
    first = small.neighbours[5][small.edge_mask[5]]
    assert set(first) <= {4, 6}


def test_no_self_edges(protein, graph):
    for i in range(len(protein)):
        assert i not in graph.neighbours[i][graph.edge_mask[i]]


def test_padded_slots_carry_no_signal(protein):
    graph = build_graph(protein, k=64, cutoff=10.0)
    padded = ~graph.edge_mask
    assert padded.any(), "k=64 should leave padding on ubiquitin"
    assert np.all(graph.edge_features[padded] == 0.0)
    assert np.all(graph.envelope[padded] == 0.0)


def test_graph_is_deterministic(protein):
    a = build_graph(protein, k=32, cutoff=10.0)
    b = build_graph(protein, k=32, cutoff=10.0)
    assert np.array_equal(a.neighbours, b.neighbours)
    assert np.array_equal(a.edge_features, b.edge_features)


def test_feature_slices_tile_the_feature_axis(protein, graph):
    covered = sorted(graph.feature_slices.values(), key=lambda s: s.start)
    assert covered[0].start == 0
    for a, b in zip(covered, covered[1:]):
        assert a.stop == b.start
    assert covered[-1].stop == graph.edge_features.shape[-1]


def test_larger_cutoff_gives_more_neighbours(protein):
    degrees = [
        build_graph(protein, k=64, cutoff=c).edge_mask.sum() for c in (6.0, 8.0, 12.0)
    ]
    assert degrees == sorted(degrees)


def test_k_binding_fraction_reports_the_discontinuity_risk(protein):
    """The diagnostic that says whether the envelope is actually doing its job.

    With k small, neighbours are dropped by the k limit rather than by distance,
    and those drops are discontinuous however smooth the envelope is.
    """
    tight = build_graph(protein, k=4, cutoff=10.0)
    loose = build_graph(protein, k=64, cutoff=10.0)
    assert tight.stats.k_binding_fraction > 0.9
    assert loose.stats.k_binding_fraction == 0.0


def test_the_default_k_leaves_the_cutoff_in_charge(protein):
    """Guards the measured default: at k=64 nothing is lost to the k limit.

    If a future change to the cutoff or the atom set makes k bind again, the
    envelope silently stops guaranteeing smoothness and this fails.
    """
    graph = build_graph(protein, cutoff=10.0)
    assert graph.k == 64
    assert graph.stats.k_binding_fraction == 0.0
    assert graph.edge_mask.sum(axis=1).max() < graph.k


def test_rejects_nonsense_parameters(protein):
    with pytest.raises(ValueError):
        build_graph(protein, k=0)
    with pytest.raises(ValueError):
        build_graph(protein, cutoff=0.0)
