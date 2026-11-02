"""One assertion carries the inverse-folding task: randomise the chis and the
blind perception must not move.

If it moves, the rule can read something about the conformation -- and the
conformation names the type -- so a sequence-recovery number measured on it
would be partly a readout of the answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.blind import VISIBLE_ATOMS, beyond_cb_mask, blind_coords
from aanca.casp import randomised_start
from aanca.structure import load_protein
from aanca.torch_graph import (
    build_graph,
    feature_topology,
    frame_atom_coords,
    node_features,
    pair_min_distances,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    return feature_topology(load_protein(path))


def _blind_graph(coords, topo):
    blind = blind_coords(coords, topo)
    graph = build_graph(blind, topo, k=24, cutoff=10.0)
    return graph, node_features(blind, topo, blind=True)


def test_the_whole_claim_randomising_every_chi_changes_nothing(ubiquitin):
    topo, coords = ubiquitin
    other = randomised_start(coords, topo, seed=3)
    assert not torch.allclose(coords, other), "the fixture must actually move"

    (graph_a, node_a) = _blind_graph(coords, topo)
    (graph_b, node_b) = _blind_graph(other, topo)

    assert torch.equal(graph_a.neighbours, graph_b.neighbours), "neighbour leak"
    assert torch.equal(graph_a.edge_mask, graph_b.edge_mask)
    assert torch.allclose(graph_a.envelope, graph_b.envelope, atol=1e-6), "envelope leak"
    assert torch.allclose(graph_a.edge_features, graph_b.edge_features, atol=1e-5)
    assert torch.allclose(node_a, node_b, atol=1e-6)


def test_the_unblinded_graph_does_change_so_the_test_above_can_fail(ubiquitin):
    """Guards against the previous test passing because nothing was measured."""
    topo, coords = ubiquitin
    other = randomised_start(coords, topo, seed=3)
    a = build_graph(coords, topo, k=24, cutoff=10.0)
    b = build_graph(other, topo, k=24, cutoff=10.0)
    assert not torch.allclose(a.envelope, b.envelope, atol=1e-6)


def test_collapsed_atoms_land_exactly_on_cb(ubiquitin):
    topo, coords = ubiquitin
    beyond = beyond_cb_mask(topo)
    blind = blind_coords(coords, topo, beyond)
    cb = frame_atom_coords(coords, topo)[:, 4]
    moved = blind[beyond]
    assert moved.numel() > 0
    expected = cb.unsqueeze(1).expand_as(blind)[beyond]
    assert torch.allclose(moved, expected, atol=1e-9)
    kept = ~beyond
    assert torch.allclose(blind[kept], coords[kept], atol=1e-9)


def test_only_backbone_and_cb_survive_the_mask(ubiquitin):
    topo, _coords = ubiquitin
    beyond = beyond_cb_mask(topo)
    for r, names in enumerate(topo.torsion.atom_names):
        for a, name in enumerate(names):
            if a >= beyond.shape[1]:
                continue
            assert bool(beyond[r, a]) == (name not in VISIBLE_ATOMS), (r, name)


def test_minimum_pair_distances_become_backbone_distances(ubiquitin):
    """The leak that is not a feature: neighbour choice is a min over atoms."""
    topo, coords = ubiquitin
    blind = blind_coords(coords, topo)
    graph = build_graph(blind, topo, k=24, cutoff=10.0)
    direct = pair_min_distances(blind, topo, graph.neighbours)
    other = blind_coords(randomised_start(coords, topo, seed=11), topo)
    assert torch.allclose(
        direct, pair_min_distances(other, topo, graph.neighbours), atol=1e-6
    )


def test_blind_node_features_keep_only_the_type_block(ubiquitin):
    topo, coords = ubiquitin
    feats = node_features(blind_coords(coords, topo), topo, blind=True)
    assert feats.shape[1] == 32
    assert torch.allclose(feats[:, :20], topo.aa_onehot.to(feats.dtype))
    assert float(feats[:, 20:].abs().max()) == 0.0


def test_a_supplied_one_hot_replaces_the_native_type(ubiquitin):
    """The ripple's pathway, and the only identity channel left open."""
    topo, coords = ubiquitin
    other = torch.zeros_like(topo.aa_onehot, dtype=torch.float32)
    other[:, 7] = 1.0
    feats = node_features(blind_coords(coords, topo), topo, aa_onehot=other, blind=True)
    assert torch.allclose(feats[:, :20], other.to(feats.dtype))
