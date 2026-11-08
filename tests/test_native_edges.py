"""Holding the neighbour selection at the native edge set.

``build_graph`` is called inside ``step_once`` on the live coordinates, so a
residue that starts 13 A from where it belongs reads the neighbourhood of the
wrong location. The regeneration task then asks it to place itself from local
information that is local to somewhere else -- and every arm scored 12.23 A,
exactly its own baseline, with no way to tell "cannot place" from "cannot find
its neighbours". The published automaton that succeeds at this task
(E(n)-equivariant Graph NCA, TMLR 2024) message-passes over the target graph.

Four things have to hold, and two of them are about not being lied to:

  * **only the selection is frozen.** If the features froze with it the rule
    would be reading a constant and the diagnostic would measure nothing, so
    both halves are asserted -- the indices identical at every step, the
    features different between steps;
  * **off is off**, and byte-identical to selecting per step;
  * **a fixed list that cannot be applied raises.** ``--damage`` is silently
    dropped outside ``--birth none`` and job 32241769 was retracted for a
    diagnostic that announced the wrong mode, so a flag that quietly no-ops is
    this project's established failure mode. Never a fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.growth import (
    GrowthConfig,
    baseline_coords,
    damage_window,
    growth_links,
    growth_objective,
)
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    fixed_neighbours_at,
    initial_state,
    rollout,
)
from aanca.structure import load_protein
from aanca.torch_graph import FixedNeighbours, build_graph, feature_topology

NATIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "1UBQ.pdb"
K = 16
STEPS = 4


@pytest.fixture(scope="module")
def ubiquitin():
    if not NATIVE.exists():
        pytest.skip(f"{NATIVE} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords, growth_links(topo)


def damaged_start(topo, coords, links, seed: int = 0):
    """A 12-residue window destroyed, everything else at its crystal pose."""
    damaged = damage_window(
        topo.n_residues, links, 12, torch.Generator().manual_seed(seed)
    )
    return damaged, baseline_coords(coords, topo, links, ~damaged, "helix")


def config() -> RolloutConfig:
    return RolloutConfig(
        steps=STEPS, truncate=99, k=K,
        growth=GrowthConfig(birth="none", initial="helix"),
    )


def moving_rule() -> AANCARule:
    """A rule that actually moves its residues.

    The frame head is zero-initialised on purpose (``model._bounded`` explains
    why the branch at the origin exists), so an untrained growth rule emits
    *exactly* zero rotation and translation and the only thing that changes
    across a rollout is the carbonyl oxygen that ``correct_oxygens`` rebuilds.
    A frozen-features bug would hide inside that, so the head's bias is set
    here and the geometry really moves -- which is also the regime a trained
    rule is in.
    """
    rule = AANCARule(growth=True, birth="none")
    with torch.no_grad():
        rule.frame_head[-1].bias.copy_(
            torch.tensor([0.05, -0.02, 0.03, 0.3, 0.2, -0.1])
        )
    return rule


def recorded_rollout(monkeypatch, coords, topo, fixed=None, seed: int = 0):
    """Run a rollout and keep every per-step graph it built.

    ``step_once`` does not return its graph, and the thing under test is an
    input to the rule rather than an output of the rollout, so the call is
    intercepted where the rollout makes it.
    """
    import aanca.rollout as rollout_module

    graphs = []
    original = rollout_module.build_graph

    def recording(*args, **kwargs):
        graph = original(*args, **kwargs)
        graphs.append(graph)
        return graph

    monkeypatch.setattr(rollout_module, "build_graph", recording)
    rule = moving_rule()
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    result = rollout(
        rule,
        initial_state(coords, rule, alive=alive),
        topo,
        config(),
        generator=torch.Generator().manual_seed(seed),
        record=False,
        fixed_neighbours=fixed,
    )
    assert len(graphs) == STEPS
    return graphs, result


def test_the_window_really_starts_far_from_where_it_belongs(ubiquitin):
    """The premise, so the rest is not testing a structure that barely moved."""
    topo, coords, links = ubiquitin
    damaged, start = damaged_start(topo, coords, links)
    index = torch.arange(topo.n_residues)
    ca = lambda c: c[index, topo.frame_idx[index, 1]]  # noqa: E731
    moved = (ca(start) - ca(coords)).norm(dim=-1)
    assert float(moved[damaged].mean()) > 5.0
    assert float(moved[~damaged].max()) == 0.0


def test_a_fixed_selection_is_identical_at_every_step(monkeypatch, ubiquitin):
    topo, coords, links = ubiquitin
    _damaged, start = damaged_start(topo, coords, links)
    fixed = fixed_neighbours_at(coords, topo, config(), alive=torch.ones(
        topo.n_residues, dtype=torch.bool
    ))

    graphs, _result = recorded_rollout(monkeypatch, start, topo, fixed=fixed)

    for step, graph in enumerate(graphs):
        assert torch.equal(graph.neighbours, fixed.neighbours), (
            f"step {step} selected its own neighbours"
        )
        assert torch.equal(graph.edge_mask, fixed.edge_mask)


def test_the_fixed_selection_is_the_one_the_native_coordinates_give(ubiquitin):
    """It is the *native* graph, not merely a constant one.

    Frozen-but-wrong would be the worst outcome available: a rollout that
    reports the diagnostic ran while every residue talks to the neighbours of
    its displaced start. So the tensor is pinned against the selection
    ``step_once`` makes when handed the native structure itself.
    """
    topo, coords, _links = ubiquitin
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    fixed = fixed_neighbours_at(coords, topo, config(), alive=alive)
    live = build_graph(
        coords.to(torch.float32), topo, k=K, cutoff=config().cutoff
    )
    assert torch.equal(fixed.neighbours, live.neighbours)
    assert torch.equal(fixed.edge_mask, live.edge_mask)


def test_selecting_per_step_does_change_the_neighbours(monkeypatch, ubiquitin):
    """Otherwise the test above passes for the wrong reason.

    With a window 13 A out of place the live selection has to differ from the
    native one -- that difference *is* the thing the flag removes.
    """
    topo, coords, links = ubiquitin
    _damaged, start = damaged_start(topo, coords, links)
    native = fixed_neighbours_at(coords, topo, config(), alive=torch.ones(
        topo.n_residues, dtype=torch.bool
    ))

    graphs, _result = recorded_rollout(monkeypatch, start, topo)

    assert not torch.equal(graphs[0].neighbours, native.neighbours), (
        "the live selection at the damaged start equals the native one, so "
        "this structure cannot tell the two apart"
    )


def test_only_the_selection_is_frozen_and_the_features_are_not(
    monkeypatch, ubiquitin
):
    """**The assertion that keeps the diagnostic meaningful.**

    The residue is supposed to measure its *true* neighbours at their
    *current* distances. Freezing the features along with the indices would
    hand the rule a constant input for the whole rollout, and a result read off
    that would be meaningless rather than merely wrong -- so the envelope, the
    distances and the direction block are required to move between steps while
    the indices do not.
    """
    topo, coords, links = ubiquitin
    _damaged, start = damaged_start(topo, coords, links)
    fixed = fixed_neighbours_at(coords, topo, config(), alive=torch.ones(
        topo.n_residues, dtype=torch.bool
    ))

    graphs, _result = recorded_rollout(monkeypatch, start, topo, fixed=fixed)

    first, last = graphs[0], graphs[-1]
    assert not torch.equal(first.edge_features, last.edge_features)
    # The three blocks the diagnostic's claim rests on: the distances, the
    # direction, and the weight the distances are turned into.
    for block in ("backbone_rbf", "sidechain_rbf", "local_direction", "envelope"):
        where = first.feature_slices[block]
        delta = first.edge_features[..., where] - last.edge_features[..., where]
        assert float(delta.detach().abs().max()) > 0.0, (
            f"the {block} block is identical across the rollout, so the "
            "features were frozen with the selection"
        )
    assert float((first.envelope - last.envelope).detach().abs().max()) > 0.0
    # And the sequence offsets, which are a property of the topology, are the
    # one block that must *not* move -- so the test above is not just reading
    # noise.
    where = first.feature_slices["sequence_offset"]
    assert torch.equal(
        first.edge_features[..., where], last.edge_features[..., where]
    )


def test_off_is_the_unchanged_path(monkeypatch, ubiquitin):
    """Flag off reproduces per-step selection exactly, to the bit.

    ``build_graph`` only consults the fixed list inside ``if
    fixed_neighbours is not None``, so passing ``None`` and not passing it at
    all are the same code; this pins the whole rollout rather than that line.
    """
    topo, coords, links = ubiquitin
    _damaged, start = damaged_start(topo, coords, links)

    graphs_a, result_a = recorded_rollout(monkeypatch, start, topo, fixed=None)
    graphs_b, result_b = recorded_rollout(monkeypatch, start, topo)

    assert torch.equal(result_a.state.coords, result_b.state.coords)
    for a, b in zip(graphs_a, graphs_b):
        assert torch.equal(a.neighbours, b.neighbours)
        assert torch.equal(a.edge_features, b.edge_features)


def test_the_objective_defaults_to_off_and_is_unchanged_by_saying_so(ubiquitin):
    topo, coords, _links = ubiquitin
    scores = []
    for kwargs in ({}, {"native_edges": False}):
        rule = AANCARule(growth=True, birth="none")
        torch.manual_seed(0)
        terms, _result = growth_objective(
            rule, coords, topo, config(),
            torch.Generator().manual_seed(0), damage=12, **kwargs,
        )
        scores.append(terms)
    assert float(scores[0].total.detach()) == float(scores[1].total.detach())
    assert scores[0].rmsd_damaged == scores[1].rmsd_damaged


def test_the_objective_freezes_the_selection_when_asked(monkeypatch, ubiquitin):
    """End to end through ``growth_objective``, which is what the CLI drives."""
    import aanca.rollout as rollout_module

    topo, coords, _links = ubiquitin
    graphs = []
    original = rollout_module.build_graph
    monkeypatch.setattr(
        rollout_module,
        "build_graph",
        lambda *a, **k: graphs.append(original(*a, **k)) or graphs[-1],
    )
    rule = AANCARule(growth=True, birth="none")
    growth_objective(
        rule, coords, topo, config(), torch.Generator().manual_seed(0),
        damage=12, native_edges=True,
    )
    assert len(graphs) == STEPS
    native = fixed_neighbours_at(coords, topo, config(), alive=torch.ones(
        topo.n_residues, dtype=torch.bool
    ))
    for graph in graphs:
        assert torch.equal(graph.neighbours, native.neighbours)


def test_a_growing_rollout_refuses_the_flag_rather_than_dropping_it(ubiquitin):
    """``--damage`` is silently ignored outside ``--birth none``; this is not.

    Under growth the alive set changes every step, so one selection taken from
    the native coordinates is neither fixed nor native -- and it would hand a
    dormant slot the finished structure's neighbourhood.
    """
    topo, coords, _links = ubiquitin
    grown = RolloutConfig(
        steps=2, truncate=99, k=K,
        growth=GrowthConfig(birth="cartesian", seeds=2),
    )
    rule = AANCARule(growth=True, birth="cartesian")
    with pytest.raises(ValueError, match="native_edges needs birth='none'"):
        growth_objective(
            rule, coords, topo, grown, torch.Generator().manual_seed(0),
            native_edges=True,
        )


@pytest.mark.parametrize(
    "broken, message",
    [
        ("not a tensor", "must be a Tensor"),
        ("short", "must be"),
        ("float", "must be torch.int64"),
        ("out of range", "outside"),
    ],
)
def test_an_inapplicable_fixed_list_raises(ubiquitin, broken, message):
    topo, coords, _links = ubiquitin
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    good = fixed_neighbours_at(coords, topo, config(), alive=alive)

    if broken == "not a tensor":
        fixed = FixedNeighbours(
            neighbours=good.neighbours.tolist(), edge_mask=good.edge_mask
        )
    elif broken == "short":
        fixed = FixedNeighbours(
            neighbours=good.neighbours[:, : K - 1],
            edge_mask=good.edge_mask[:, : K - 1],
        )
    elif broken == "float":
        fixed = FixedNeighbours(
            neighbours=good.neighbours.to(torch.float32),
            edge_mask=good.edge_mask,
        )
    else:
        fixed = FixedNeighbours(
            neighbours=good.neighbours + topo.n_residues,
            edge_mask=good.edge_mask,
        )

    with pytest.raises(ValueError, match=message):
        build_graph(
            coords.to(torch.float32), topo, k=K, fixed_neighbours=fixed
        )


def test_a_missing_fixed_list_is_not_quietly_recomputed(ubiquitin):
    """The shape of the fallback that must not exist.

    A list built for another structure is the realistic way this goes wrong --
    a crop of a different length, reused across updates -- and the only safe
    answer is to stop. Silently selecting from the live coordinates instead
    would report the diagnostic while running the arm it was meant to replace.
    """
    topo, coords, _links = ubiquitin
    half = torch.zeros(topo.n_residues // 2, K, dtype=torch.long)
    fixed = FixedNeighbours(
        neighbours=half, edge_mask=torch.ones_like(half, dtype=torch.bool)
    )
    with pytest.raises(ValueError, match=f"\\({topo.n_residues}, {K}\\)"):
        build_graph(coords.to(torch.float32), topo, k=K, fixed_neighbours=fixed)
