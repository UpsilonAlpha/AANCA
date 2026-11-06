"""Regeneration: a hole in an intact protein, filled from local context.

This is the project's objective, and it is not what the arms of 6 Nov trained
on. They started every residue from a constructed baseline with one native
anchor per covalent run -- native context at about one residue in eight -- and
all four froze at their own starting conformation. Regeneration is the opposite
ratio: the intact remainder is most of the structure and the damage is a window
in it.

The tests here pin the three things that have to hold for that to be a
different experiment rather than the same one with a flag:

  * the context is preserved **exactly**, because a regeneration run that
    perturbs the intact structure is measuring something else;
  * the window is hinged off its own neighbour and lands within the rollout's
    36 A reach, which the walk order did not do before (measured: 74 A out);
  * and the reported RMSD is the window's, superposed on the context.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.fape import backbone_rmsd_on
from aanca.growth import (
    GrowthConfig,
    baseline_coords,
    damage_window,
    growth_links,
    growth_objective,
    run_anchors,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

NATIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "1UBQ.pdb"
CA_CA = 3.804


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords, growth_links(topo)


def ca(coords, topo):
    index = torch.arange(topo.n_residues)
    return coords[index, topo.frame_idx[index, 1]]


def test_a_damage_window_is_contiguous_and_inside_one_covalent_run(ubiquitin):
    topo, _coords, links = ubiquitin
    for width in (4, 12, 24):
        mask = damage_window(topo.n_residues, links, width, torch.Generator())
        index = torch.nonzero(mask).flatten()
        assert int(mask.sum()) == width
        assert int(index.max() - index.min()) == width - 1, "not contiguous"
        # Every link inside the window is intact, so it spans no chain break.
        assert bool(links[index[:-1]].all())
        # Interior: an intact residue on each side.
        assert index.min() > 0 and index.max() < topo.n_residues - 1


def test_a_run_too_short_for_a_window_is_left_undamaged(ubiquitin):
    """An empty mask, not an exception.

    One short crop in a pool of 412 should not stop a training run, and the
    caller then trains that step on an undamaged structure -- which the
    evaluation reports separately rather than averaging in as a perfect fill.
    """
    topo, _coords, links = ubiquitin
    mask = damage_window(topo.n_residues, links, topo.n_residues, torch.Generator())
    assert not bool(mask.any())
    assert damage_window(topo.n_residues, links, 0, torch.Generator()).sum() == 0


def test_the_context_is_preserved_exactly_and_the_window_is_not(ubiquitin):
    topo, coords, links = ubiquitin
    damaged = damage_window(topo.n_residues, links, 12, torch.Generator().manual_seed(0))
    start = baseline_coords(coords, topo, links, ~damaged, "helix")

    moved = (ca(start, topo) - ca(coords, topo)).norm(dim=-1)
    assert float(moved[~damaged].max()) == 0.0, "the intact context must not move"
    assert float(moved[damaged].min()) > 0.5, "the window must actually be damaged"
    assert not start.requires_grad


def test_the_window_is_hinged_off_its_neighbour_and_stays_within_reach(ubiquitin):
    """The walk-order fix, stated as the number that was wrong.

    Before it, every anchor's walk ran to the end of its covalent run and the
    highest-indexed anchor overwrote the rest, so a window surrounded by
    anchors was placed off the far end of the chain: measured at 73.96 A mean
    displacement with the N-ward join at 64.78 A. Both are past the rollout's
    own ceiling of ``steps x max_translation`` = 36 A, so the task would have
    been impossible for a reason that had nothing to do with regeneration.
    """
    topo, coords, links = ubiquitin
    rule = AANCARule(growth=True, birth="none")
    ceiling = 120 * rule.max_translation

    for seed in range(6):
        damaged = damage_window(
            topo.n_residues, links, 12, torch.Generator().manual_seed(seed)
        )
        start = baseline_coords(coords, topo, links, ~damaged, "helix")
        index = torch.nonzero(damaged).flatten()
        lo = int(index.min())

        step = (ca(start, topo)[1:] - ca(start, topo)[:-1]).norm(dim=-1)
        assert float(step[lo - 1]) == pytest.approx(CA_CA, abs=1e-3), (
            "the window must hinge off the intact residue before it"
        )
        inside = step[lo : lo + 11]
        assert torch.allclose(inside, torch.full_like(inside, CA_CA), atol=1e-3)

        moved = (ca(start, topo) - ca(coords, topo)).norm(dim=-1)
        assert float(moved[damaged].max()) < ceiling, (
            f"the window starts {float(moved[damaged].max()):.1f} A out, past "
            f"the {ceiling:.0f} A a 120-step rollout can travel"
        )


def test_the_c_ward_join_is_left_open_for_the_rule_to_close(ubiquitin):
    """Loop closure is the task, not something to hand the rule.

    The window hinges off its N-ward neighbour, so its far end lands wherever
    ideal geometry put it. If that join were also closed the window would
    already be in the right place and there would be nothing to learn.
    """
    topo, coords, links = ubiquitin
    damaged = damage_window(topo.n_residues, links, 12, torch.Generator().manual_seed(1))
    start = baseline_coords(coords, topo, links, ~damaged, "helix")
    index = torch.nonzero(damaged).flatten()
    hi = int(index.max())
    step = (ca(start, topo)[1:] - ca(start, topo)[:-1]).norm(dim=-1)
    assert float(step[hi]) != pytest.approx(CA_CA, abs=0.5), (
        "the C-ward join must be open, or the fill is already correct"
    )


def test_one_anchor_per_run_is_unchanged_by_the_walk_order_fix(ubiquitin):
    """The old behaviour, which fifteen tests depend on, is a special case.

    With a single anchor in a run there is only ever one walk, so stopping a
    walk at the next anchor changes nothing. Pinned by equality rather than
    argued, because the fix touches the code path every earlier arm used and a
    silent change there would make the 6 Nov results unreproducible.
    """
    topo, coords, links = ubiquitin
    generator = torch.Generator().manual_seed(3)
    anchors = run_anchors(topo.n_residues, links, generator)
    assert int(anchors.sum()) >= 1
    first = baseline_coords(coords, topo, links, anchors, "helix")
    second = baseline_coords(coords, topo, links, anchors, "helix")
    assert torch.equal(first, second)
    # Every link in a run is at the ideal distance, which is the property the
    # self-placing tests assert and the one a walk-order bug would break.
    step = (ca(first, topo)[1:] - ca(first, topo)[:-1]).norm(dim=-1)
    linked = links[:-1].bool() if links.numel() == topo.n_residues else links.bool()
    inside = step[linked[: step.numel()]]
    assert torch.allclose(inside, torch.full_like(inside, CA_CA), atol=1e-2)


def test_the_damaged_rmsd_is_superposed_on_the_context_not_on_itself(ubiquitin):
    """Why the metric needs its own function.

    A window superposed onto its own native copy scores well for any plausible
    shape placed anywhere, because the superposition absorbs the error being
    asked about. Take the native structure, move the window rigidly, and the
    two metrics disagree by construction: superposing on the context sees the
    displacement, superposing the window on itself sees nothing.
    """
    topo, coords, links = ubiquitin
    damaged = damage_window(topo.n_residues, links, 12, torch.Generator().manual_seed(0))
    intact = ~damaged
    moved = coords.clone()
    moved[damaged] = moved[damaged] + torch.tensor([8.0, 0.0, 0.0])

    on_context = float(backbone_rmsd_on(moved, coords, topo, intact, damaged))
    on_itself = float(backbone_rmsd_on(moved, coords, topo, damaged, damaged))

    assert on_context == pytest.approx(8.0, abs=1e-3)
    assert on_itself < 1e-3, "superposed on itself, a rigid shift is invisible"


def test_a_regeneration_objective_reports_the_window_and_scores_better(ubiquitin):
    """End to end, and the comparison that made this worth building.

    The same loss on the same structure, reconstruction against regeneration.
    FAPE is a mean over residue pairs, so with the context intact most pairs
    are between two correct residues and the wrong ones have correct
    neighbours inside the 10 A clamp.
    """
    topo, coords, _links = ubiquitin
    config = RolloutConfig(
        steps=4, truncate=99, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )

    scores = {}
    for name, damage in (("reconstruction", 0), ("regeneration", 12)):
        rule = AANCARule(growth=True, birth="none", identity=True)
        terms, _ = growth_objective(
            rule, coords, topo, config, torch.Generator().manual_seed(0),
            identity_weight=0.0, damage=damage,
        )
        scores[name] = terms

    assert scores["reconstruction"].rmsd_damaged is None
    assert scores["reconstruction"].damaged == 0
    assert scores["regeneration"].damaged == 12
    assert scores["regeneration"].rmsd_damaged is not None
    assert float(scores["regeneration"].fape) < float(scores["reconstruction"].fape)
    # And the window's own error is far larger than the whole-structure RMSD,
    # which is the reason `rmsd` cannot be the headline for this task.
    assert scores["regeneration"].rmsd_damaged > scores["regeneration"].rmsd
