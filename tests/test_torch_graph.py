"""The Torch featuriser: matches the NumPy reference, and differentiates where it must."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aanca.damage import randomise_rotamers
from aanca.geometry import rotation_matrix
from aanca.graph import build_graph as np_build_graph
from aanca.structure import load_protein
from aanca.torch_geometry import apply_chi_deltas
from aanca.torch_graph import (
    EDGE_DIM,
    NODE_DIM,
    build_graph,
    edge_features,
    feature_topology,
    node_features,
    pair_min_distances,
    select_neighbours,
    select_neighbours_device,
    sequence_links,
    sidechain_summary,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
CUTOFF = 10.0


@pytest.fixture(scope="module", params=["193L", "1UBQ"])
def loaded(request):
    path = DATA / f"{request.param}.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    protein = load_protein(path)
    topo, coords = feature_topology(protein)
    return protein, topo, coords


# -- agreement with the verified reference ---------------------------------


def test_dims_match_the_reference(loaded):
    _protein, _topo, _coords = loaded
    assert EDGE_DIM == 575
    assert NODE_DIM == 32


def test_same_neighbours_as_the_reference(loaded):
    protein, topo, coords = loaded
    reference = np_build_graph(protein, cutoff=CUTOFF)
    neighbours, edge_mask = select_neighbours(coords, topo, k=64, cutoff=CUTOFF)
    assert np.array_equal(neighbours.numpy(), reference.neighbours)
    assert np.array_equal(edge_mask.numpy(), reference.edge_mask)


def test_node_features_match_the_reference(loaded):
    protein, topo, coords = loaded
    reference = np_build_graph(protein, cutoff=CUTOFF)
    got = node_features(coords, topo).numpy()
    assert np.allclose(got, reference.node_features, atol=1e-10)


def test_sidechain_summary_matches_the_reference(loaded):
    protein, topo, coords = loaded
    from aanca.graph import sidechain_summary as np_summary

    got = sidechain_summary(coords, topo).numpy()
    assert np.allclose(got, np_summary(protein), atol=1e-10)


def test_edge_features_match_the_reference(loaded):
    protein, topo, coords = loaded
    reference = np_build_graph(protein, cutoff=CUTOFF)
    graph = build_graph(coords, topo, k=64, cutoff=CUTOFF)

    assert graph.edge_features.shape == reference.edge_features.shape
    assert graph.feature_slices == reference.feature_slices
    for name, sl in graph.feature_slices.items():
        got = graph.edge_features[..., sl].numpy()
        want = reference.edge_features[..., sl]
        assert np.allclose(got, want, atol=1e-8), name


def test_envelope_matches_the_reference(loaded):
    protein, topo, coords = loaded
    reference = np_build_graph(protein, cutoff=CUTOFF)
    graph = build_graph(coords, topo, k=64, cutoff=CUTOFF)
    assert np.allclose(graph.envelope.numpy(), reference.envelope, atol=1e-9)


def test_min_distances_match_a_brute_force_computation(loaded):
    """The argmin-then-measure trick must give the same answer as the full min."""
    protein, topo, coords = loaded
    neighbours, edge_mask = select_neighbours(coords, topo, k=64, cutoff=CUTOFF)
    got = pair_min_distances(coords, topo, neighbours)

    for i in (0, 5, 20):
        for slot in range(int(edge_mask[i].sum())):
            j = int(neighbours[i, slot])
            a = protein.residues[i].coords
            b = protein.residues[j].coords
            brute = float(
                np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1))
            )
            assert float(got[i, slot]) == pytest.approx(brute, abs=1e-9)


# -- invariance ------------------------------------------------------------


def test_features_are_invariant_to_a_global_rigid_motion(loaded):
    protein, _topo, _coords = loaded
    rng = np.random.default_rng(0)
    r = rotation_matrix(rng.normal(size=3), 1.1)
    moved = protein.transformed(rotation=r, translation=rng.normal(size=3) * 20.0)

    topo_a, coords_a = feature_topology(protein)
    topo_b, coords_b = feature_topology(moved)
    a = build_graph(coords_a, topo_a, cutoff=CUTOFF)
    b = build_graph(coords_b, topo_b, cutoff=CUTOFF)

    assert torch.equal(a.neighbours, b.neighbours)
    assert torch.allclose(a.edge_features, b.edge_features, atol=1e-8)
    assert torch.allclose(a.node_features, b.node_features, atol=1e-9)


def test_symmetric_chi_flip_barely_moves_the_sidechain_summary(loaded):
    """Averaging tied tip atoms makes the summary *nearly* blind to the
    180-degree flip of the two-fold symmetric chis.

    Nearly, not exactly. Exact invariance needs ideal geometry -- a carboxylate
    whose two oxygens are related precisely by a rotation about CB-CG, or an
    aromatic ring whose own two-fold axis is precisely the CB-CG bond. Real
    crystal side chains depart from that, and the lever arm to a distal atom
    amplifies the departure: measured deviations run 0.04-0.19 A for ASP/GLU but
    up to 0.39 A for PHE/TYR, whose tips sit furthest from the axis.

    The flip itself moves atoms by ~2.4 A, so the summary is an order of
    magnitude less sensitive to it than the raw coordinates -- which is the
    property wanted, just not a guarantee.
    """
    _protein, topo, coords = loaded
    periodic = topo.torsion.periodic
    if not bool(periodic.any()):
        pytest.skip("no symmetric chis present")

    flip = torch.where(
        periodic, torch.full(periodic.shape, torch.pi, dtype=coords.dtype), 0.0
    )
    flipped = apply_chi_deltas(coords, flip, topo.torsion)
    atom_motion = float((flipped - coords).abs().max())
    assert atom_motion > 2.0, "the flip moved nothing"

    rows = periodic.any(dim=1).nonzero(as_tuple=True)[0]
    deviation = float(
        (sidechain_summary(coords, topo)[rows] - sidechain_summary(flipped, topo)[rows])
        .abs()
        .max()
    )
    assert deviation < 0.5, f"summary moved {deviation:.3f} A under a symmetric flip"
    assert deviation < atom_motion / 5.0


# -- the per-step rebuild is real ------------------------------------------


def test_moving_side_chains_changes_the_graph(loaded):
    protein, topo, coords = loaded
    damaged_np = randomise_rotamers(protein, rng=np.random.default_rng(0)).apply(protein)
    _topo2, damaged = feature_topology(damaged_np)

    before = build_graph(coords, topo, cutoff=CUTOFF)
    after = build_graph(damaged, topo, cutoff=CUTOFF)
    assert not torch.equal(before.neighbours, after.neighbours)
    assert not torch.allclose(before.envelope, after.envelope)


def test_backbone_block_is_reusable_across_a_step(loaded):
    """The caching claim: with the neighbour list fixed, the backbone block does
    not change when side chains move -- so it can be computed once per selection."""
    protein, topo, coords = loaded
    from aanca.torch_graph import backbone_edge_block

    neighbours, _mask = select_neighbours(coords, topo, cutoff=CUTOFF)
    damaged_np = randomise_rotamers(protein, rng=np.random.default_rng(1)).apply(protein)
    _t, damaged = feature_topology(damaged_np)

    a = backbone_edge_block(coords, topo, neighbours)
    b = backbone_edge_block(damaged, topo, neighbours)
    assert torch.allclose(a, b, atol=1e-10)


def test_fixed_backbone_changes_gradients_not_values(loaded):
    """The memory saving must be exact, not an approximation.

    Building the backbone block under no_grad is legitimate only because the
    backbone never moves, so its gradient is identically zero. The *values* must
    be bit-identical either way -- if they are not, the flag is changing the
    model rather than its memory profile.
    """
    _protein, topo, coords = loaded
    neighbours, edge_mask = select_neighbours(coords, topo, cutoff=CUTOFF)

    with_grad, _e1, _s1 = edge_features(
        coords, topo, neighbours, edge_mask, CUTOFF, fixed_backbone=False
    )
    without, _e2, _s2 = edge_features(
        coords, topo, neighbours, edge_mask, CUTOFF, fixed_backbone=True
    )
    assert torch.equal(with_grad, without)


def test_fixed_backbone_keeps_the_block_out_of_the_graph(loaded):
    """Measured 47% of retained activations on 4HHB. Detaching afterwards does
    not work -- by then the tensor is already built and saved for backward."""
    from aanca.torch_graph import backbone_edge_block

    _protein, topo, coords = loaded
    start = coords.clone().requires_grad_(True)
    neighbours, _mask = select_neighbours(start, topo, cutoff=CUTOFF)

    fixed = backbone_edge_block(start, topo, neighbours, fixed_backbone=True)
    assert not fixed.requires_grad

    moving = backbone_edge_block(start, topo, neighbours, fixed_backbone=False)
    assert moving.requires_grad, "the escalation path must still differentiate"


def test_sidechain_gradients_survive_a_fixed_backbone(loaded):
    """The saving must not cost the gradients that actually matter."""
    _protein, topo, coords = loaded
    start = coords.clone().requires_grad_(True)
    graph = build_graph(start, topo, cutoff=CUTOFF, fixed_backbone=True)
    sl = graph.feature_slices["sidechain_rbf"]
    graph.edge_features[..., sl].sum().backward()
    assert float(start.grad.abs().sum()) > 0.0


def test_passing_a_cached_backbone_block_changes_nothing(loaded):
    _protein, topo, coords = loaded
    from aanca.torch_graph import backbone_edge_block

    neighbours, edge_mask = select_neighbours(coords, topo, cutoff=CUTOFF)
    cached = backbone_edge_block(coords, topo, neighbours)
    fresh, _env, _sl = edge_features(coords, topo, neighbours, edge_mask, CUTOFF)
    reused, _env2, _sl2 = edge_features(
        coords, topo, neighbours, edge_mask, CUTOFF, backbone_block=cached
    )
    assert torch.allclose(fresh, reused, atol=1e-12)


# -- differentiability -----------------------------------------------------


def test_gradients_reach_the_coordinates_through_the_envelope(loaded):
    """The envelope is the gradient path for graph structure. If this is zero,
    moving a side chain cannot change how strongly its edges contribute, and the
    smooth cutoff is decoration."""
    _protein, topo, coords = loaded
    start = coords.clone().requires_grad_(True)
    neighbours, edge_mask = select_neighbours(start, topo, cutoff=CUTOFF)
    _f, envelope, _s = edge_features(start, topo, neighbours, edge_mask, CUTOFF)
    envelope.sum().backward()
    assert start.grad is not None
    assert torch.isfinite(start.grad).all()
    assert float(start.grad.abs().sum()) > 0.0


def test_gradients_reach_the_coordinates_through_the_sidechain_block(loaded):
    _protein, topo, coords = loaded
    start = coords.clone().requires_grad_(True)
    graph = build_graph(start, topo, cutoff=CUTOFF)
    sl = graph.feature_slices["sidechain_rbf"]
    graph.edge_features[..., sl].sum().backward()
    assert float(start.grad.abs().sum()) > 0.0


def test_gradients_flow_from_deltas_through_the_features(loaded):
    """The rollout path: deltas -> coordinates -> features. One link broken here
    and training silently optimises only the final step."""
    _protein, topo, coords = loaded
    import aanca.residue_constants as rc

    deltas = torch.zeros(
        topo.n_residues, rc.MAX_CHI, dtype=coords.dtype, requires_grad=True
    )
    moved = apply_chi_deltas(coords, deltas, topo.torsion)
    neighbours, edge_mask = select_neighbours(moved, topo, cutoff=CUTOFF)
    features, _env, _sl = edge_features(moved, topo, neighbours, edge_mask, CUTOFF)
    features.sum().backward()
    assert deltas.grad is not None
    assert torch.isfinite(deltas.grad).all()
    assert float(deltas.grad[topo.torsion.chi_mask].abs().sum()) > 0.0


def test_selection_is_detached(loaded):
    """Selection must not try to backprop through an argsort."""
    _protein, topo, coords = loaded
    start = coords.clone().requires_grad_(True)
    neighbours, edge_mask = select_neighbours(start, topo, cutoff=CUTOFF)
    assert not neighbours.requires_grad
    assert not edge_mask.requires_grad


def test_rejects_nonsense_parameters(loaded):
    _protein, topo, coords = loaded
    with pytest.raises(ValueError):
        select_neighbours(coords, topo, k=0)
    with pytest.raises(ValueError):
        select_neighbours(coords, topo, cutoff=0.0)


# -- the on-device selection, which must be the same function --------------


def test_device_selection_matches_the_numpy_one(loaded):
    """The replacement for the 80% of a rollout step that runs on the CPU.

    Exactness is the whole requirement. A neighbour list that differs anywhere
    is a different perception graph, and the only symptom would be a two-day
    run that scores slightly differently from the one it is supposed to be
    compared with -- which is indistinguishable from a result.
    """
    _protein, topo, coords = loaded
    coords32 = coords.to(torch.float32)
    expected = select_neighbours(coords32, topo, k=64, cutoff=CUTOFF)
    actual = select_neighbours_device(coords32, topo, k=64, cutoff=CUTOFF)
    assert torch.equal(actual[1], expected[1]), "edge masks differ"
    assert torch.equal(actual[0], expected[0]), "neighbour lists differ"


def test_device_selection_matches_after_the_side_chains_move(loaded):
    """The graph is rebuilt every step because side chains move, so agreement
    on the deposited coordinates is not enough -- it has to hold on the
    off-distribution geometry a rollout actually visits."""
    protein, topo, coords = loaded
    generator = torch.Generator().manual_seed(7)
    for step in range(3):
        deltas = torch.zeros_like(topo.torsion.chi_mask, dtype=coords.dtype)
        deltas[topo.torsion.chi_mask] = (
            torch.rand(
                int(topo.torsion.chi_mask.sum()), generator=generator,
                dtype=coords.dtype,
            )
            * 2
            - 1
        ) * 3.14159
        moved = apply_chi_deltas(coords, deltas, topo.torsion).to(torch.float32)
        expected = select_neighbours(moved, topo, k=64, cutoff=CUTOFF)
        actual = select_neighbours_device(moved, topo, k=64, cutoff=CUTOFF)
        assert torch.equal(actual[1], expected[1]), f"edge masks differ at {step}"
        assert torch.equal(actual[0], expected[0]), f"neighbours differ at {step}"


def test_device_selection_matches_with_a_binding_k(loaded):
    """At k=16 the limit, not the cutoff, decides -- so the *order* matters and
    not just the membership. This is the case that catches a wrong tie-break or
    a forced neighbour sorted in the wrong place."""
    _protein, topo, coords = loaded
    coords32 = coords.to(torch.float32)
    for k in (8, 16, 32):
        expected = select_neighbours(coords32, topo, k=k, cutoff=CUTOFF)
        actual = select_neighbours_device(coords32, topo, k=k, cutoff=CUTOFF)
        assert torch.equal(actual[1], expected[1]), f"edge masks differ at k={k}"
        assert torch.equal(actual[0], expected[0]), f"neighbours differ at k={k}"


def test_device_selection_keeps_the_sequence_neighbours(loaded):
    """Forced edges come first whatever the distance.

    At k=1 that is the whole content of the slot, so a version that merely
    sorted by distance would fail here even on a compact protein -- where at
    k=8 it would pass, because the bonded neighbours are among the eight
    nearest anyway.
    """
    _protein, topo, coords = loaded
    coords32 = coords.to(torch.float32)
    links = sequence_links(topo, coords32.device)
    neighbours, edge_mask = select_neighbours_device(
        coords32, topo, k=1, cutoff=CUTOFF
    )
    for i in range(topo.n_residues):
        bonded = set()
        if i > 0 and bool(links[i - 1]):
            bonded.add(i - 1)
        if i + 1 < topo.n_residues and bool(links[i]):
            bonded.add(i + 1)
        if bonded:
            assert bool(edge_mask[i, 0])
            assert int(neighbours[i, 0]) in bonded, f"residue {i} lost its bond"


def test_device_selection_matches_without_sequence_neighbours(loaded):
    """And the switch is wired the same way in both implementations."""
    _protein, topo, coords = loaded
    coords32 = coords.to(torch.float32)
    for k in (8, 64):
        expected = select_neighbours(
            coords32, topo, k=k, cutoff=CUTOFF, include_sequence_neighbours=False
        )
        actual = select_neighbours_device(
            coords32, topo, k=k, cutoff=CUTOFF, include_sequence_neighbours=False
        )
        assert torch.equal(actual[1], expected[1]), f"edge masks differ at k={k}"
        assert torch.equal(actual[0], expected[0]), f"neighbours differ at k={k}"


def test_device_selection_refuses_the_same_arguments():
    topo = coords = None
    for bad in ({"k": 0}, {"cutoff": 0.0}):
        with pytest.raises(ValueError):
            select_neighbours_device(torch.zeros(1, 3), topo, **bad)
