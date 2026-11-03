"""The hole task: is it well posed, equivariant, and does it exclude the answer?"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
import torch

from aanca.backbone import backbone_atoms, frame_from_triad
from aanca.hole import (
    SEE_RADIUS,
    centroid_error,
    consensus_error,
    fused_position,
    hole_loss,
    hole_predictions,
    make_task,
    placement_error,
)
from aanca.model import AANCARule
from aanca.structure import load_protein
from aanca.torch_geometry import rotation_matrices
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    return feature_topology(load_protein(path))


# -- the task --------------------------------------------------------------


def test_the_hole_is_not_alive_and_everything_else_is(ubiquitin):
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(0))
    assert not bool(task.alive[task.hole])
    assert int((~task.alive).sum()) == 1, "exactly one residue is missing"


def test_a_frontier_task_leaves_a_partial_structure(ubiquitin):
    """The case growth is actually in: the hole is bounded on one side only."""
    topo, coords = ubiquitin
    task = make_task(coords, topo, "frontier", random.Random(0), 0.5)
    assert not bool(task.alive[task.hole])
    assert int(task.alive.sum()) < topo.n_residues - 1, "nothing was cropped"
    # The surviving residues are one contiguous sequence stretch, which is the
    # shape of partial structure a growth rollout produces.
    index = torch.nonzero(task.alive, as_tuple=False).flatten()
    assert int(index.max() - index.min()) == index.numel() - 1


def test_a_frontier_hole_has_fewer_witnesses_than_a_surrounded_one(ubiquitin):
    """The whole reason both modes are measured, in one assertion."""
    topo, coords = ubiquitin
    surrounded, frontier = [], []
    for seed in range(12):
        rng = random.Random(seed)
        surrounded.append(
            int(make_task(coords, topo, "hole", rng).witnesses.sum())
        )
        frontier.append(
            int(make_task(coords, topo, "frontier", rng, 0.5).witnesses.sum())
        )
    assert sum(frontier) / 12 < sum(surrounded) / 12


def test_only_residues_that_could_have_seen_the_hole_are_asked(ubiquitin):
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(1))
    ca = backbone_atoms(coords, topo)[:, 1]
    distance = (ca - ca[task.hole]).norm(dim=-1)
    assert bool((distance[task.witnesses] < SEE_RADIUS).all())
    assert not bool(task.witnesses[task.hole]), "the hole is not its own witness"
    assert not bool((task.witnesses & ~task.alive).any())


def test_the_target_is_the_hole_in_each_witnesses_own_frame(ubiquitin):
    """Checked by mapping it back, because a frame error here is silent."""
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(2))
    bb = backbone_atoms(coords, topo)
    rotation = frame_from_triad(bb[:, 0], bb[:, 1], bb[:, 2])
    ca = bb[:, 1]
    index = torch.nonzero(task.witnesses, as_tuple=False).flatten()
    recovered = ca[index] + torch.einsum(
        "iab,ib->ia", rotation[index], task.target[index]
    )
    assert torch.allclose(
        recovered, ca[task.hole].expand_as(recovered), atol=1e-8
    )


def test_the_target_is_zero_where_nothing_is_asked(ubiquitin):
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(3))
    assert float(task.target[~task.witnesses].abs().max()) == 0.0


def test_an_unknown_mask_is_refused(ubiquitin):
    topo, coords = ubiquitin
    with pytest.raises(ValueError, match="mask"):
        make_task(coords, topo, "ball", random.Random(0))


def test_a_tiny_structure_is_refused(ubiquitin):
    """``n_residues`` comes off the topology, so the topology is what to shrink."""
    from dataclasses import replace

    topo, coords = ubiquitin
    tiny = replace(topo, atom_mask=topo.atom_mask[:4])
    with pytest.raises(ValueError, match="at least 8"):
        make_task(coords, tiny, "hole", random.Random(0))


# -- the prediction --------------------------------------------------------


def test_the_deleted_residue_cannot_influence_the_prediction(ubiquitin):
    """The headline. Move the hole anywhere and the answers must not change.

    A task that let the deleted residue into the graph would be measuring
    interpolation of a known answer, which is the one way this whole
    measurement could be meaningless while looking fine.
    """
    topo, coords = ubiquitin
    rule = AANCARule(hole=True)
    torch.manual_seed(0)
    for p in rule.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    task = make_task(coords, topo, "hole", random.Random(4))

    moved = coords.clone()
    moved[task.hole] += 25.0
    first = hole_predictions(rule, coords, topo, task)
    second = hole_predictions(rule, moved, topo, task)
    # Compared over the residues that exist. The hole's *own* row does change,
    # because its node features are measured from its own coordinates -- and
    # nothing reads that row: it is dead in the mask, zeroed as a sender by
    # ``edge_mask & alive[neighbours]``, and excluded from the loss. Measured:
    # every alive row is bit-identical, the hole's row moves by 3e-7.
    alive = task.alive
    assert torch.equal(first[0][alive], second[0][alive]), "the hole leaked"
    assert torch.equal(first[1][alive], second[1][alive])
    assert torch.equal(first[0][task.witnesses], second[0][task.witnesses])


def test_the_prediction_is_equivariant(ubiquitin):
    """It is a displacement in the residue's own frame, so it rotates with it."""
    topo, coords = ubiquitin
    rule = AANCARule(hole=True)
    torch.manual_seed(0)
    for p in rule.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    task = make_task(coords, topo, "hole", random.Random(5))

    rot = rotation_matrices(
        torch.tensor([0.4, -0.6, 0.7], dtype=coords.dtype),
        torch.tensor(0.9, dtype=coords.dtype),
    )
    shift = torch.tensor([13.0, -2.0, 5.0], dtype=coords.dtype)
    moved = (coords.reshape(-1, 3) @ rot.transpose(-1, -2) + shift).reshape(
        coords.shape
    )
    here = hole_predictions(rule, coords, topo, task)[0]
    there = hole_predictions(rule, moved, topo, task)[0]
    # The offsets are already in local frames, so they should be *identical*,
    # not rotated -- that is what makes them equivariant predictions.
    assert float((here - there).detach().abs().max()) < 2e-3


def test_the_loss_is_zero_when_the_prediction_is_right(ubiquitin):
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(6))
    perfect = task.target.to(torch.float32)
    # Detection perfect too: a large logit where a witness is, small elsewhere.
    logit = torch.where(
        task.witnesses, torch.full_like(perfect[:, 0], 30.0),
        torch.full_like(perfect[:, 0], -30.0),
    )
    total, fused, per_witness = hole_loss(perfect, logit, task, coords, topo)
    assert float(fused) == pytest.approx(0.0, abs=1e-5)
    assert float(per_witness) == pytest.approx(0.0, abs=1e-5)
    assert float(total) == pytest.approx(0.0, abs=1e-5)


def test_the_placement_term_is_huber_so_far_witnesses_cannot_dominate(ubiquitin):
    """A squared error would let the nine-Angstrom witnesses set the gradient."""
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(7))
    offset = torch.zeros(topo.n_residues, 3)
    logit = torch.zeros(topo.n_residues)
    near, _f, _p = hole_loss(offset, logit, task, coords, topo, delta=2.0)
    # Doubling every residual must less than quadruple the loss, which is what
    # separates Huber from squared error.
    scaled = make_task(coords, topo, "hole", random.Random(7))
    scaled.target = task.target * 2.0
    far, _f2, _p2 = hole_loss(offset, logit, scaled, coords, topo, delta=2.0)
    assert float(far) < 4.0 * float(near)


def test_the_consensus_beats_the_average_witness(ubiquitin):
    """Mapping every answer into one frame should help, or witnesses are useless.

    With a perfect prediction both are zero, so the test perturbs each witness
    independently: errors that are independent average away, and the consensus
    is the number a generator would actually use.
    """
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(8))
    torch.manual_seed(0)
    noisy = task.target.to(torch.float32) + torch.randn(
        topo.n_residues, 3
    ) * 1.5
    logit = torch.where(
        task.witnesses, torch.full((topo.n_residues,), 5.0),
        torch.full((topo.n_residues,), -5.0),
    )
    per_witness = float(placement_error(noisy, task))
    consensus = float(consensus_error(noisy, logit, task, coords, topo))
    assert consensus < per_witness, (consensus, per_witness)


def test_a_rule_without_the_head_is_refused(ubiquitin):
    topo, coords = ubiquitin
    with pytest.raises(ValueError, match="hole head"):
        AANCARule().hole_outputs(torch.zeros(3, 32), torch.zeros(3, 96))


def test_the_untrained_head_predicts_its_own_ca(ubiquitin):
    """Zeroed, so the answer is "the hole is where I am" -- wrong, not divergent."""
    rule = AANCARule(hole=True)
    offset, logit = rule.hole_outputs(torch.randn(5, 32), torch.randn(5, 96))
    assert float(offset.detach().abs().max()) == 0.0
    assert float(logit.detach().abs().max()) == 0.0


def test_no_witnesses_gives_a_finite_differentiable_zero(ubiquitin):
    """A crop can leave a hole nobody can see; it must not produce a NaN."""
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(9))
    task.witnesses = torch.zeros_like(task.witnesses)
    offset = torch.zeros(topo.n_residues, 3, requires_grad=True)
    total, _f, _p = hole_loss(
        offset, torch.zeros(topo.n_residues), task, coords, topo
    )
    assert float(total.detach()) == 0.0
    total.backward()
    assert torch.isfinite(offset.grad).all()
    assert math.isnan(float(placement_error(offset.detach(), task)))


# -- the fused position, and the baseline it has to beat -------------------


def test_an_untrained_head_reproduces_the_centroid_exactly(ubiquitin):
    """The invariant the rebuilt objective rests on, and it is checkable.

    A zero offset and a flat logit make the softmax uniform, so the fused
    position *is* the witnesses' centroid. That makes the centroid the right
    zero point for the headline metric -- not "no prediction" -- and it is why
    the first version of this task was measuring the wrong thing: the
    unweighted per-witness error has no such anchor, and emitting zero beat
    every closed-form guess on it (``scripts/hole_baseline.py``).
    """
    topo, coords = ubiquitin
    rule = AANCARule(hole=True)
    for mode in ("hole", "frontier"):
        task = make_task(coords, topo, mode, random.Random(11))
        offset, logit = hole_predictions(rule, coords, topo, task)
        fused = float(consensus_error(offset, logit, task, coords, topo).detach())
        centroid = float(centroid_error(task, coords, topo))
        assert fused == pytest.approx(centroid, abs=1e-6), mode


def test_the_fusion_weights_are_a_softmax_so_a_witness_can_abstain(ubiquitin):
    """A sigmoid lets every witness assert itself; a softmax makes them compete."""
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(12))
    index = torch.nonzero(task.witnesses, as_tuple=False).flatten()
    offset = task.target.to(torch.float32).clone()
    # Give one witness a wrong answer and all the weight: the fused position
    # must follow it, which a sigmoid-weighted mean would dilute.
    offset[index[0]] += 20.0
    logit = torch.full((topo.n_residues,), -10.0)
    logit[index[0]] = 10.0
    fused = fused_position(offset, logit, task, coords, topo)
    from aanca.backbone import backbone_atoms

    ca = backbone_atoms(coords, topo)[:, 1]
    assert float((fused - ca[task.hole]).norm()) > 10.0


def test_a_far_witness_is_weighted_down_in_the_per_witness_term(ubiquitin):
    """1/d^2, which is the Fisher argument and not a taste.

    A witness localises the hole to an angular precision, which converts to a
    positional error proportional to its distance, so its precision goes as the
    inverse square. Without this the twelve-Angstrom witnesses set the gradient
    and the model is paid to say nothing.
    """
    topo, coords = ubiquitin
    task = make_task(coords, topo, "hole", random.Random(13))
    index = torch.nonzero(task.witnesses, as_tuple=False).flatten()
    order = task.distance[index].argsort()
    nearest, furthest = int(index[order[0]]), int(index[order[-1]])
    assert float(task.distance[furthest]) > float(task.distance[nearest]) + 2.0

    logit = torch.zeros(topo.n_residues)
    base = task.target.to(torch.float32)
    near_wrong = base.clone()
    near_wrong[nearest] += 5.0
    far_wrong = base.clone()
    far_wrong[furthest] += 5.0
    near_loss = float(hole_loss(near_wrong, logit, task, coords, topo)[0].detach())
    far_loss = float(hole_loss(far_wrong, logit, task, coords, topo)[0].detach())
    assert near_loss > far_loss, (near_loss, far_loss)
