"""Where is a residue missing? The question growth is driven by.

Growth as a point cloud rests on one capability: a residue looking at its own
neighbourhood and saying "there should be another residue *there*". Everything
else in the scheme -- threading the chain off the peptide bond, terminating when
the density is satisfied, nucleating a new segment -- is downstream of how
accurately that position can be named. So it is worth measuring on its own,
before any of it is built, and it can be: **this task needs no rollout at all.**

**The task.** Take a native structure, delete one residue, and ask the residues
that could see it to point at the hole. The target is unique -- the deleted
residue's CA, expressed in the predictor's own frame -- so the regression is
well posed, and it is exactly the question growth asks: given my neighbourhood
as it stands, where is a residue missing?

Well posed *because* one residue is missing. A head asked "where are your
neighbours" has ~19 right answers and no way to choose; a head asked "where is
the gap" has one. That is the whole reason the task is framed as deletion
rather than as prediction of the neighbourhood.

**Two regimes, and the difference between them is the point.** A single hole in
an otherwise complete structure is the easy case and an *upper bound* on what
is achievable: the hole is surrounded, so its position is heavily
over-determined. Growth never sees that -- at the frontier, most of the
protein does not exist yet, and the hole is bounded on one side only. So
``mask="hole"`` measures the bound and ``mask="frontier"`` measures the case
that matters, and the gap between them is the real difficulty of growth.

**Equivariance.** The target is expressed in the predictor's own backbone frame,
so the prediction is of a local displacement and rotating the structure rotates
the answer rather than changing it. The same reason Cartesian birth is
parameterised in the parent's frame.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from .backbone import backbone_atoms, frame_from_triad
from .growth import alive_topology
from .torch_graph import FeatureTopology, build_graph

#: Radius within which a residue is considered able to see the hole, Angstrom.
#: The same 10 A as perception, so "could see it" means "would have had an edge
#: to it".
SEE_RADIUS = 10.0


@dataclass
class HoleTask:
    """One training example: a structure with a hole, and who can see it."""

    #: ``(R,)`` bool -- which residues exist. The hole is false here, and under
    #: ``frontier`` masking so is everything that has not been built yet.
    alive: Tensor
    #: The index of the deleted residue the question is about.
    hole: int
    #: ``(R,)`` bool -- alive residues within ``SEE_RADIUS`` of the hole, which
    #: are the ones scored. A residue that could not have seen it is not being
    #: asked.
    witnesses: Tensor
    #: ``(R, 3)`` the hole's CA in each witness's own frame. Zero elsewhere.
    target: Tensor
    #: ``(R,)`` how far each witness is from the hole, Angstrom. The loss needs
    #: it: a witness at twelve Angstrom cannot locate the hole, and weighting it
    #: equally both wastes capacity and lets it dominate the gradient, since its
    #: target is three times longer than a close witness's.
    distance: Tensor


def _frames(coords: Tensor, topo: FeatureTopology) -> tuple[Tensor, Tensor]:
    backbone = backbone_atoms(coords, topo)
    rotation = frame_from_triad(backbone[:, 0], backbone[:, 1], backbone[:, 2])
    return rotation, backbone[:, 1]


def make_task(
    coords: Tensor,
    topo: FeatureTopology,
    mask: str = "hole",
    rng: random.Random | None = None,
    frontier_fraction: float = 0.5,
) -> HoleTask:
    """Delete a residue and work out who should be able to point at it.

    ``mask="hole"`` deletes only the hole, which is the over-determined upper
    bound. ``mask="frontier"`` additionally deletes a contiguous sequence stretch
    beyond it, so the hole sits on the boundary of a partial structure, which is
    the case growth actually faces.

    The contiguous stretch is deliberately a *sequence* stretch rather than a
    spatial ball: growth's alive set is a union of sequence segments, so that is
    the shape of partial structure the rule will meet. A spatial ball would be
    an easier and less relevant problem, since it leaves every segment's
    interior intact.
    """
    rng = rng or random
    n = topo.n_residues
    if n < 8:
        raise ValueError(f"need at least 8 residues to make a hole, got {n}")

    alive = torch.ones(n, dtype=torch.bool, device=coords.device)
    if mask == "frontier":
        # A segment of the chain survives and the rest does not; the hole is
        # the first residue past the surviving end.
        length = max(int(round(n * frontier_fraction)), 4)
        start = rng.randrange(0, max(n - length, 1))
        alive[:] = False
        alive[start : start + length] = True
        hole = start + length if start + length < n else start - 1
        if hole < 0 or hole >= n:
            raise ValueError("the crop leaves no room for a hole")
        alive[hole] = False
    elif mask == "hole":
        hole = rng.randrange(n)
        alive[hole] = False
    else:
        raise ValueError(f"unknown mask {mask!r}; expected 'hole' or 'frontier'")

    rotation, centre = _frames(coords, topo)
    offset = centre - centre[hole]
    distance = offset.norm(dim=-1)
    witnesses = alive & (distance < SEE_RADIUS) & topo.has_frame
    # The hole's CA in each witness's frame: R^T (x_hole - x_witness).
    local = torch.einsum(
        "iab,ib->ia", rotation.transpose(-1, -2), centre[hole] - centre
    )
    target = torch.where(witnesses.unsqueeze(-1), local, torch.zeros_like(local))
    return HoleTask(
        alive=alive,
        hole=hole,
        witnesses=witnesses,
        target=target,
        distance=distance,
    )


def hole_predictions(
    rule,
    coords: Tensor,
    topo: FeatureTopology,
    task: HoleTask,
    k: int = 32,
    cutoff: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """``(offset, logit)`` per residue from one message-passing hop.

    One hop and no rollout: the question is what a residue can tell from its
    immediate neighbourhood, and unrolling would answer a different one. The
    graph is built on the *alive* subset, so the deleted residue contributes
    nothing -- which is what ``growth.alive_topology`` is for, and the reason
    this task needed no new perception code.
    """
    masked = alive_topology(topo, task.alive)
    graph = build_graph(
        coords.to(torch.float32), masked, k=k, cutoff=cutoff, blind=False
    )
    hidden = rule.initial_hidden(topo.n_residues, coords.device)
    _mu, _log_kappa, new_hidden = rule(
        graph.node_features.to(torch.float32),
        hidden,
        graph.edge_features.to(torch.float32),
        graph.neighbours,
        graph.edge_mask & task.alive[graph.neighbours],
        graph.envelope.to(torch.float32),
    )
    return rule.hole_outputs(graph.node_features.to(torch.float32), new_hidden)


def hole_loss(
    offset: Tensor,
    logit: Tensor,
    task: HoleTask,
    coords: Tensor,
    topo: FeatureTopology,
    delta: float = 2.0,
    fused_weight: float = 1.0,
    detection_weight: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(total, fused_error, per_witness_error)``, the errors in Angstrom.

    **Rebuilt after the baselines said the first version was measuring the
    wrong thing.** `scripts/hole_baseline.py` found that every closed-form
    predictor scored *worse* than emitting zero -- anti-centroid 10.53 A
    against zero's 7.20 -- which is not a weak baseline but a broken metric.
    Witness distances are median 7.6 A with only 30 % inside 6 A, so a target's
    *length* is mostly what the mean per-witness error measures: predicting
    zero scores the witness's own distance, and any guess with magnitude in a
    slightly wrong direction scores worse. Equal weighting therefore paid the
    model to say nothing, and the far witnesses -- which cannot locate the hole
    -- set the gradient.

    So there are three terms.

    **The fused position**, which is the objective that matters: every witness's
    answer is mapped into the shared frame and combined under a softmax of the
    logits, and *that* single position is scored. It is what a generator would
    actually use, it lets a far witness abstain by assigning itself a low
    weight, and the baseline to beat is the plain centroid of the witnesses --
    measured at **2.51 A** surrounded and **4.76 A** at a frontier.

    **A proximity-weighted per-witness term**, for dense supervision: one fused
    scalar per structure is a thin gradient, so each witness is still scored,
    weighted by ``1 / d^2``. That exponent is the Fisher argument rather than a
    taste: a witness localises the hole to an angular precision, which converts
    to a positional error proportional to its distance, so its precision goes
    as the inverse square.

    **Detection**, over every residue and not only the witnesses, because at
    growth time nothing tells a residue that a hole exists. A rule that can
    place a hole but not notice one is useless.
    """
    witnesses = task.witnesses
    if not bool(witnesses.any()):
        zero = offset.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    def huber(residual: Tensor) -> Tensor:
        return torch.where(
            residual < delta, 0.5 * residual**2, delta * (residual - 0.5 * delta)
        )

    residual = (offset[witnesses] - task.target[witnesses].to(offset.dtype)).norm(dim=-1)
    precision = 1.0 / task.distance[witnesses].to(offset.dtype).clamp_min(1.0) ** 2
    precision = precision / precision.sum().clamp_min(1e-9)
    per_witness = (huber(residual) * precision).sum()

    _rotation, centre = _frames(coords, topo)
    fused = fused_position(offset, logit, task, coords, topo)
    fused_residual = (fused - centre[task.hole]).norm()

    detection = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, witnesses.to(logit.dtype)
    )
    total = (
        per_witness
        + fused_weight * huber(fused_residual).to(per_witness.dtype)
        + detection_weight * detection.to(per_witness.dtype)
    )
    return total, fused_residual.detach(), residual.mean().detach()


def fused_position(
    offset: Tensor,
    logit: Tensor,
    task: HoleTask,
    coords: Tensor,
    topo: FeatureTopology,
) -> Tensor:
    """The single position the witnesses agree on, in the shared frame.

    Each witness names the hole in its own frame; mapping every answer back and
    combining under a **softmax** of the logits -- not a sigmoid -- is what lets
    a witness abstain, because a softmax is a competition for weight while a
    sigmoid lets every witness assert itself independently.
    """
    rotation, centre = _frames(coords, topo)
    world = centre + torch.einsum("iab,ib->ia", rotation, offset.to(centre.dtype))
    masked = torch.where(
        task.witnesses, logit.to(centre.dtype),
        torch.full_like(logit.to(centre.dtype), -1e9),
    )
    weight = torch.softmax(masked, dim=0)
    return (world * weight.unsqueeze(-1)).sum(0)


def centroid_position(
    task: HoleTask, coords: Tensor, topo: FeatureTopology
) -> Tensor:
    """The plain centroid of the witnesses -- the baseline the fusion must beat."""
    _rotation, centre = _frames(coords, topo)
    return centre[task.witnesses].mean(dim=0)


def consensus_error(
    offset: Tensor, logit: Tensor, task: HoleTask, coords: Tensor, topo: FeatureTopology
) -> Tensor:
    """Angstrom between the fused position and the real hole.

    **The headline number for the whole cloud scheme.** The baseline it has to
    beat is the plain centroid of the witnesses, measured at 2.51 A for a
    surrounded hole and 4.76 A at a frontier -- and the target is about 0.6 A,
    the tolerance within which a newborn's peptide bond stays unambiguous
    (``scripts/peptide_shell_probe.py``). An untrained head emits a zero offset
    and a flat logit, so it reproduces the centroid exactly, which is why the
    centroid is the right zero point rather than "no prediction".
    """
    if not bool(task.witnesses.any()):
        return torch.full((), float("nan"), device=offset.device)
    _rotation, centre = _frames(coords, topo)
    fused = fused_position(offset, logit, task, coords, topo)
    return (fused - centre[task.hole]).norm()


def centroid_error(
    task: HoleTask, coords: Tensor, topo: FeatureTopology
) -> Tensor:
    """The baseline: how far the witnesses' own centroid is from the hole."""
    if not bool(task.witnesses.any()):
        return torch.full((), float("nan"), device=coords.device)
    _rotation, centre = _frames(coords, topo)
    return (centroid_position(task, coords, topo) - centre[task.hole]).norm()


def placement_error(offset: Tensor, task: HoleTask) -> Tensor:
    """Mean Angstrom of the *individual* witnesses' answers, unweighted.

    Reported alongside :func:`consensus_error` and never optimised on its own,
    because the baselines showed what it measures: witness distances are median
    7.6 A, so this number is dominated by how far away the witnesses are rather
    than by how good the prediction is, and emitting zero scores better on it
    than any closed-form guess (``scripts/hole_baseline.py``). It is still worth
    watching -- the gap between it and the fused error is how much the fusion is
    doing, and whether the witnesses' mistakes are independent enough to average
    away.
    """
    witnesses = task.witnesses
    if not bool(witnesses.any()):
        return torch.full((), float("nan"), device=offset.device)
    return (
        (offset[witnesses] - task.target[witnesses].to(offset.dtype))
        .norm(dim=-1)
        .mean()
    )
