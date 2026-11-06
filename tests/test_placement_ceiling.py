"""The ceiling check: can this architecture place residues under full supervision?

`placement_loss` is a diagnostic and must never be a training objective -- it is
teacher forcing, and a rule that places residues only because it was handed each
one's target is a regression fitter arranged as a cellular automaton. It exists
because seven arms have failed at rung 1 of the information ladder and **no run
has ever established that the architecture can place residues under ANY amount
of supervision**, which makes a loose-restraint result uninterpretable either
way.

The tests here pin the two things that would make it silently useless: the
rotation convention, which if transposed trains towards a mirror image, and the
claim that the loss is actually minimised by the ideal move.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.backbone import frame_from_triad
from aanca.growth import (
    GrowthConfig,
    backbone_atoms,
    growth_links,
    growth_objective,
    placement_loss,
    rigid_relax,
    rotation_from_vector,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

NATIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "1UBQ.pdb"


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords, growth_links(topo)


def frames(coords, topo):
    bb = backbone_atoms(coords, topo)
    return frame_from_triad(bb[:, 0], bb[:, 1], bb[:, 2]), bb[:, 1]


def test_a_relax_composes_the_rotation_on_the_left(ubiquitin):
    """``rigid_relax`` gives ``rot @ frame``, not ``frame @ rot``.

    `placement_loss` compares ``rot @ rot_now`` against the native frame. If the
    composition order were the other way the term would be minimised by a
    different rotation entirely -- a plausible-looking loss that trains towards
    the wrong pose -- and nothing else in the suite would notice, because
    `rigid_relax` itself is only ever checked through distances.
    """
    topo, coords, _links = ubiquitin
    moving = torch.zeros(topo.n_residues, dtype=torch.bool)
    moving[5] = True

    axis_angle = torch.zeros(topo.n_residues, 3)
    axis_angle[5] = torch.tensor([0.3, -0.2, 0.1])
    translation = torch.zeros(topo.n_residues, 3)

    rot_before, _ca = frames(coords, topo)
    moved = rigid_relax(coords, axis_angle, translation, topo, moving)
    rot_after, _ca_after = frames(moved, topo)

    rot = rotation_from_vector(axis_angle).to(coords.dtype)
    assert torch.allclose(rot_after[5], rot[5] @ rot_before[5], atol=1e-5)
    # And the other order is measurably different, so the test has teeth.
    assert not torch.allclose(rot_after[5], rot_before[5] @ rot[5], atol=1e-3)


def test_the_loss_is_zero_for_a_move_that_lands_on_the_native(ubiquitin):
    """A residue already in place, asked to stay, scores zero on both halves."""
    topo, coords, _links = ubiquitin
    moving = torch.ones(topo.n_residues, dtype=torch.bool)
    axis_angle = torch.zeros(topo.n_residues, 3)
    translation = torch.zeros(topo.n_residues, 3)

    loss = placement_loss(
        [(axis_angle, translation, coords, moving)], coords, topo
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_the_ideal_translation_beats_any_other_of_the_same_size(ubiquitin):
    """The target is a bearing, and the loss is lowest along it.

    The complaint against FAPE is that it supplies a magnitude without a
    direction. This asserts the replacement does the opposite: hold the step
    length fixed at the 0.3 A cap and the loss is minimised by the step pointing
    at the native, and strictly larger for the same step pointing elsewhere.
    """
    topo, coords, _links = ubiquitin
    displaced = coords + torch.tensor([4.0, 1.0, -2.0])
    moving = torch.ones(topo.n_residues, dtype=torch.bool)
    axis_angle = torch.zeros(topo.n_residues, 3)

    _rot, ca_now = frames(displaced, topo)
    _rot_n, ca_native = frames(coords, topo)
    toward = ca_native - ca_now
    toward = toward / toward.norm(dim=-1, keepdim=True) * 0.3

    best = float(placement_loss(
        [(axis_angle, toward, displaced, moving)], coords, topo
    ))
    for wrong in (
        torch.tensor([0.3, 0.0, 0.0]),
        torch.tensor([0.0, 0.3, 0.0]),
        torch.tensor([-0.3, 0.0, 0.0]),
    ):
        other = wrong.expand_as(toward).contiguous()
        value = float(placement_loss(
            [(axis_angle, other, displaced, moving)], coords, topo
        ))
        assert value > best, (wrong.tolist(), value, best)


def test_the_position_half_is_linear_so_its_gradient_does_not_fade(ubiquitin):
    """Unclamped and linear on purpose.

    A squared or clamped position term would reintroduce exactly the problem it
    is replacing: FAPE's gradient is flat above 10 A, so a residue 40 A out gets
    a constant. Doubling the displacement here must double the loss, which is
    what a constant-magnitude gradient looks like from the value side.
    """
    topo, coords, _links = ubiquitin
    moving = torch.ones(topo.n_residues, dtype=torch.bool)
    axis_angle = torch.zeros(topo.n_residues, 3)
    translation = torch.zeros(topo.n_residues, 3)

    values = []
    for far in (5.0, 10.0, 20.0, 40.0):
        displaced = coords + torch.tensor([far, 0.0, 0.0])
        values.append(float(placement_loss(
            [(axis_angle, translation, displaced, moving)], coords, topo
        )))
    for a, b in zip(values, values[1:]):
        assert b == pytest.approx(2.0 * a, rel=1e-4)


def test_an_empty_or_dead_segment_scores_zero_rather_than_nan(ubiquitin):
    topo, coords, _links = ubiquitin
    assert float(placement_loss([], coords, topo)) == 0.0
    dead = torch.zeros(topo.n_residues, dtype=torch.bool)
    axis_angle = torch.zeros(topo.n_residues, 3)
    translation = torch.zeros(topo.n_residues, 3)
    value = placement_loss(
        [(axis_angle, translation, coords, dead)], coords, topo
    )
    assert float(value) == 0.0 and torch.isfinite(value)


def test_the_objective_reports_it_and_leaves_it_off_by_default(ubiquitin):
    """``move_weight`` defaults to 0, so no existing arm's loss changes."""
    topo, coords, _links = ubiquitin
    config = RolloutConfig(
        steps=4, truncate=99, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )

    out = {}
    for name, weight in (("off", 0.0), ("on", 1.0)):
        rule = AANCARule(growth=True, birth="none", identity=True)
        terms, _ = growth_objective(
            rule, coords, topo, config, torch.Generator().manual_seed(0),
            identity_weight=0.0, damage=12, move_weight=weight,
        )
        out[name] = terms

    assert float(out["off"].move) == 0.0
    assert float(out["on"].move) > 0.0, "the ceiling term must see the window"
    # The reported FAPE is unchanged by adding the term, so the two arms stay
    # comparable on the metric even though their losses differ.
    assert float(out["off"].fape) == pytest.approx(float(out["on"].fape), rel=1e-5)
