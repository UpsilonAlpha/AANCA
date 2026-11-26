"""Train the E(n)-GNCA rule as a DENOISER toward a fixed point.

    .venv/Scripts/python.exe scripts/train_denoise.py configs/denoise.toml --section mvp
    .venv/Scripts/python.exe scripts/train_denoise.py configs/denoise.toml --list-sections

**This is a second trainer, not a flag on ``scripts/train_egnca.py``**, on
``scripts/train_growth.py``'s precedent: growth is its own trainer because it
supervises a different thing, and so does this. ``train_egnca.py`` and
``configs/egnca.toml`` record a published result (Bunya job 33419979, 2.53 A CA
RMSD against the best previous arm's 8.10) and must keep running bit-for-bit, so
nothing here touches them or ``src/aanca/egnca.py``. The rule, the rollout, the
step, the pool entry, the neighbour selection and the loss are all **imported**
from there; what this file adds is the corruption operator, the context mask,
the fixed-point sampler and the training loop around them.

**What the port does and what this removes.** ``egnca_objective`` destroys a
contiguous 12-residue window of a 128-residue crop and holds the other ~116
residues FROZEN at their crystal pose, message-passing over the contact graph
the NATIVE coordinates select. Two things in that sentence are why it works, and
both are the thing to remove:

  * the frozen native context supplies ~116 exact anchors, so each placement is
    close to analytically determined;
  * the native contact graph is the task statement rather than something the
    rule has to find.

Here:

  * **Corruption replaces damage.** Isotropic Gaussian noise is added to ALL
    non-context residue coordinates, with sigma drawn LOG-UNIFORMLY from
    ``[sigma_min, sigma_max]``. **No timestep reaches the rule.** The same
    weights run at sigma 0.5 and at sigma 30, and that absence is the point:
    this is a diffusion model minus the schedule and minus the conditioning,
    which is what keeps it a cellular automaton rather than a denoiser with a
    clock. The target is therefore a FIXED POINT -- the native structure has to
    be where the rule stops from every noise level at once.
  * **``context_fraction`` is the one axis.** That share of residues, chosen as
    contiguous runs, is held at native coordinates and excluded from the noise
    and from the loss. ``0.0`` is pure generation and is the MVP; ``1.0`` pins
    everything and so has nothing to score (see :data:`SKIP_REASONS`).
  * **The graph is rebuilt from where the residues CURRENTLY are**, every step
    (``native_edges = false``, the default here). That is the recorded failure
    mode ``docs/claims/the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is.md``,
    and here it is the research question rather than a bug, so it is the
    default; ``native_edges = true`` stays available as a control arm.

**The loss is not new.** ``growth.pairwise_distance_restraint`` over the full
pair set on plain distances, exactly as the port calls it, with the corrupted
set standing where the damaged window stood. At ``context_fraction = 0.0`` the
window-to-context half is exactly 0.0 because there is no context, and the
corrupted-to-corrupted half carries the whole of it -- which is then literally
Gala's all-pairs ``L_INV``.

**What is kept from the port, unchanged**, so that a difference in a result is
the two removals above and not a sixth thing: the rule and its five published
settings, 15-25 step rollouts with full BPTT, the replay pool at probability
0.5, both halves of the restraint weighted, lr 1e-3, clip 1.0, weight decay
1e-5, the widths, ``k``, ``cutoff``, the crop size, and the per-reason skip
counter.

**Exit code 64 means "checkpointed, not finished".** ``scripts/bunya_train.slurm``
resubmits its chain **only** on 64, and ``train_egnca.py`` returns 0 when it
stops on ``--max-seconds`` -- so a port segment that ran out of walltime ended
the chain silently, which is the ``--max-seconds`` exit HANDOFF.md §0a names.
This trainer returns 64 on that path and 0 only on a real finish or a stop file.

Nothing recorded in this docstring is an expected value. What 300 laptop updates
at ``--section mvp`` actually read is in the report of the run that wrote the
file, and 300 updates against a 20,000-update config says nothing about whether
the rule learns the task.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from aanca import checkpoint as ckpt
from aanca import config as run_config
from aanca.data import RepackingDataset, read_manifest
from aanca.egnca import (
    ACTIVATIONS,
    NORM_TYPES,
    EGNCAPoolEntry,
    EGNCARule,
    EGNCAState,
    ca_positions,
    egnca_neighbours,
    egnca_rollout,
    egnca_step,
    with_ca_positions,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
RUNS = ROOT / "runs"

#: The exit code ``scripts/bunya_train.slurm`` resubmits its chain on. Anything
#: else ends the chain, which is why this is a named constant rather than a
#: literal three screens down: ``train_egnca.py`` returns 0 on this path and a
#: walltime-truncated port segment therefore stopped the chain in silence.
CHECKPOINTED_NOT_FINISHED = 64

#: How long one contiguous run of held context is, in residues, when
#: ``context_fraction`` is strictly between 0 and 1. **Not a config key**:
#: ``context_fraction`` is the one axis of this sweep and a second key would make
#: every point on it two-dimensional. 12 is the port's own damage width, so the
#: corrupted stretches between held runs are the same scale as the hole the port
#: fills -- which is what makes ``ctx100`` the end of one sweep rather than a
#: different experiment.
CONTEXT_RUN_WIDTH = 12

#: A compact globular chain's radius of gyration in Angstrom, ``2.2 * N**0.38``.
#: The Gaussian prior is scaled to exactly this, so a sample starts at a
#: protein's own size rather than at whatever ``randn`` happens to give: a cloud
#: ten times too large would measure the rule's ability to contract and not its
#: ability to fold.
RG_PREFACTOR = 2.2
RG_EXPONENT = 0.38


def radius_of_gyration(n_residues: int) -> float:
    """The natural radius of gyration of an ``n_residues``-residue chain, A."""
    if n_residues < 1:
        raise ValueError(f"n_residues must be positive, got {n_residues}")
    return RG_PREFACTOR * float(n_residues) ** RG_EXPONENT


# -- the corruption operator and the context mask ---------------------------


def draw_sigma(
    sigma_min: float, sigma_max: float, generator: torch.Generator | None = None
) -> float:
    """One noise level, **log-uniform** in ``[sigma_min, sigma_max]``.

    Log-uniform rather than uniform because the quantity that matters is the
    ratio: uniform on [0.5, 30] puts 98 % of the draws above 1 A, so the rule
    would almost never see the near-converged regime it has to be a fixed point
    of. Equal bounds fix the level, which is how a single-sigma control arm is
    stated without a second key.
    """
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError(
            "sigma must be positive to be drawn log-uniformly, got "
            f"[{sigma_min}, {sigma_max}]"
        )
    if sigma_max < sigma_min:
        raise ValueError(f"sigma_max {sigma_max} is below sigma_min {sigma_min}")
    if sigma_max == sigma_min:
        return float(sigma_min)
    u = float(torch.rand((), generator=generator))
    lo, hi = math.log(sigma_min), math.log(sigma_max)
    return float(math.exp(lo + u * (hi - lo)))


def context_mask(
    n_residues: int,
    fraction: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    width: int = CONTEXT_RUN_WIDTH,
) -> Tensor:
    """``(R,)`` bool: the residues held at native coordinates.

    ``fraction`` of the residues, as **contiguous runs** of about ``width``
    rather than a scatter of independent residues. A scatter would hand the rule
    an exact anchor inside every neighbourhood at any fraction, so the sweep
    would measure nothing: what a held residue carries is local information, and
    what the axis has to vary is how much of the structure has *no* anchor
    nearby. The runs are laid out one per equal segment of the chain at a random
    offset inside it, so they never overlap and the count is exact.

    ``0.0`` returns an all-false mask and ``1.0`` an all-true one, both exactly:
    those are the two ends of the sweep and neither may be approximate.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"context_fraction must be in [0, 1], got {fraction}")
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    mask = torch.zeros(max(n_residues, 0), dtype=torch.bool, device=device)
    if n_residues < 1 or fraction <= 0.0:
        return mask
    if fraction >= 1.0:
        mask[:] = True
        return mask
    wanted = min(max(int(round(fraction * n_residues)), 0), n_residues)
    if wanted == 0:
        return mask
    if wanted >= n_residues:
        mask[:] = True
        return mask
    runs = min(max(1, int(round(wanted / width))), wanted)
    bounds = [(index * n_residues) // runs for index in range(runs + 1)]
    base, extra = divmod(wanted, runs)
    for index in range(runs):
        lo, hi = bounds[index], bounds[index + 1]
        length = min(base + (1 if index < extra else 0), hi - lo)
        room = (hi - lo) - length
        offset = (
            int(torch.randint(room + 1, (1,), generator=generator).item())
            if room > 0
            else 0
        )
        mask[lo + offset : lo + offset + length] = True
    return mask


def corrupt(
    ca: Tensor,
    sigma: float,
    corrupted: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``ca`` with isotropic Gaussian noise of scale ``sigma`` on ``corrupted``.

    Isotropic and per-residue: three independent ``N(0, sigma^2)`` draws, so the
    per-axis standard deviation is ``sigma`` exactly and a corrupted residue's
    expected displacement is ``sigma * sqrt(3)``. Held residues are left
    bit-for-bit, which is what makes them anchors rather than
    slightly-wrong anchors.

    Drawn on the CPU from the passed generator, like every other random choice
    in this project, so the corruption is a property of the seed and not of the
    device.
    """
    if ca.dim() != 2 or ca.shape[-1] != 3:
        raise ValueError(f"ca must be (R, 3), got {tuple(ca.shape)}")
    if corrupted.shape != (ca.shape[0],) or corrupted.dtype is not torch.bool:
        raise ValueError(
            f"corrupted must be a ({ca.shape[0]},) bool mask, got "
            f"{tuple(corrupted.shape)} of {corrupted.dtype}"
        )
    if sigma < 0:
        raise ValueError(f"sigma must not be negative, got {sigma}")
    noise = torch.randn(ca.shape, generator=generator, dtype=ca.dtype) * sigma
    return ca + noise.to(ca.device) * corrupted.unsqueeze(-1).to(ca.dtype)


# -- the objective ----------------------------------------------------------


@dataclass
class DenoiseTerms:
    """The loss and the CA-level readouts. ``EGNCATerms``' shape, one key more.

    Not ``EGNCATerms`` itself: that class lives in ``src/aanca/egnca.py``, which
    the port's published result rests on and which this work may not touch, and
    it carries no noise level. ``sigma`` is part of the readout rather than a
    diagnostic -- ``rmsd`` is only ever comparable against ``start_rmsd`` at the
    same draw.
    """

    total: Tensor
    pair_anchor: Tensor
    pair_window: Tensor
    #: The level this sample was corrupted at, A; ``nan`` for a pool resume,
    #: which does not remember one and did not need to.
    sigma: float = float("nan")
    #: CA RMSD over the CORRUPTED residues, superposed on the whole structure.
    rmsd: float | None = None
    #: The same, superposed on the HELD context alone. Absent -- not zero --
    #: when fewer than three residues are held, which is the whole of
    #: ``context_fraction = 0.0``.
    rmsd_fit_context: float | None = None
    #: What the corrupted set scored before a single step. The pool's accept bar
    #: and the only thing that says a number is better than not trying.
    start_rmsd: float | None = None
    #: Mean CA displacement over the rollout, A. Without it a flat ``rmsd``
    #: cannot be told from a rule that emits nothing.
    move: float | None = None
    corrupted_mask: Tensor | None = None


def denoise_objective(
    rule: EGNCARule,
    native_coords: Tensor,
    topo,
    steps: int,
    context_fraction: float = 0.0,
    sigma_min: float = 0.5,
    sigma_max: float = 30.0,
    generator: torch.Generator | None = None,
    pairwise_weight: float = 1.0,
    pairwise_window_weight: float = 1.0,
    k: int = 24,
    cutoff: float = 10.0,
    device_neighbours: bool = False,
    native_edges: bool = False,
    resume_from: EGNCAPoolEntry | None = None,
) -> tuple[DenoiseTerms, EGNCAState]:
    """One denoising sample, priced.

    Draw a context mask, draw a noise level, corrupt everything else, roll out
    with full BPTT, and price the result with
    ``growth.pairwise_distance_restraint`` over the full pair set. The corrupted
    set stands exactly where the port's damaged window stands, so the loss is
    the port's loss rather than a new one.

    ``resume_from`` is the replay pool's path and skips the corruption: a pool
    entry carries the state the rule reached and the mask it was asked about, so
    re-drawing either would score a structure against a question it was never
    asked -- ``EGNCAPoolEntry``'s own reason, reused.

    **A diverged rollout reports ``None``, it does not raise**, for the reason
    ``egnca_objective`` gives: this rule's displacement is unbounded even under
    the authors' tanh, and ``backbone_rmsd_on``'s SVD raises on a non-finite
    input, which would take down a run hours in instead of skipping an update.
    """
    from aanca.fape import backbone_rmsd_on
    from aanca.growth import pairwise_distance_restraint

    native_ca = ca_positions(native_coords, topo).detach().to(rule.dtype)
    if resume_from is not None:
        corrupted = resume_from.damaged
        start_ca = resume_from.ca
        hidden = resume_from.hidden
        sigma = float("nan")
    else:
        held = context_mask(
            topo.n_residues,
            context_fraction,
            generator=generator,
            device=native_coords.device,
        )
        corrupted = ~held
        sigma = draw_sigma(sigma_min, sigma_max, generator=generator)
        with torch.no_grad():
            start_ca = corrupt(native_ca, sigma, corrupted, generator=generator)
        hidden = rule.initial_hidden(
            topo.n_residues, device=native_coords.device, dtype=rule.dtype
        )
    held = ~corrupted

    state = egnca_rollout(
        rule,
        EGNCAState(ca=start_ca, hidden=hidden),
        topo,
        steps,
        native_coords,
        moving=corrupted,
        k=k,
        cutoff=cutoff,
        device_neighbours=device_neighbours,
        native_edges=native_edges,
    )

    final = with_ca_positions(native_coords, state.ca, topo)
    anchor, window = pairwise_distance_restraint(
        final, native_coords, topo, corrupted, held
    )
    total = pairwise_weight * anchor + pairwise_window_weight * window

    terms = DenoiseTerms(
        total=total,
        pair_anchor=anchor.detach(),
        pair_window=window.detach(),
        sigma=sigma,
        corrupted_mask=corrupted,
        move=float((state.ca - start_ca).detach().norm(dim=-1)[corrupted].mean())
        if bool(corrupted.any())
        else None,
    )
    if bool(corrupted.any()) and bool(torch.isfinite(state.ca).all()):
        with torch.no_grad():
            everything = torch.ones_like(corrupted)
            begin = with_ca_positions(native_coords, start_ca, topo)
            terms.rmsd = float(
                backbone_rmsd_on(final, native_coords, topo, everything, corrupted)
            )
            terms.start_rmsd = float(
                backbone_rmsd_on(begin, native_coords, topo, everything, corrupted)
            )
            # nan by ``backbone_rmsd_on``'s own contract when fewer than three
            # residues are held, which is every sample at context_fraction 0.
            terms.rmsd_fit_context = float(
                backbone_rmsd_on(final, native_coords, topo, held, corrupted)
            )
    return terms, state


# -- sampling from the prior, which is the generative question --------------


def prior_coordinates(
    n_residues: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """``(R, 3)`` Gaussian coordinates scaled to exactly ``2.2 * N**0.38`` A.

    Centred and then rescaled so the radius of gyration is the target
    **exactly**, not in expectation: the alternative is a per-sample radius that
    varies by a few percent, and steps-to-convergence would then be partly a
    measurement of how large that draw happened to be.
    """
    x = torch.randn(n_residues, 3, generator=generator, dtype=dtype)
    x = x - x.mean(dim=0, keepdim=True)
    current = x.pow(2).sum(-1).mean().clamp_min(1e-12).sqrt()
    return (x * (radius_of_gyration(n_residues) / current)).to(device)


@dataclass
class Sample:
    """One run of the rule from the prior to a fixed point, or to the cap."""

    ca: Tensor
    steps: int
    converged: bool
    #: The max per-residue displacement of the final step, A -- the quantity the
    #: tolerance is tested on.
    last_move: float
    radius_of_gyration: float


def run_to_fixed_point(
    rule: EGNCARule,
    topo,
    carrier_coords: Tensor,
    start_ca: Tensor,
    tolerance: float = 0.01,
    max_steps: int = 200,
    k: int = 24,
    cutoff: float = 10.0,
    device_neighbours: bool = False,
    native_edges: bool = False,
) -> Sample:
    """Step until the MAX per-residue displacement falls below ``tolerance``.

    The max and not the mean: a mean over 128 residues hides one residue still
    flying, and a fixed point is a statement about every residue.

    ``carrier_coords`` is the padded ``(R, MAX_ATOMS, 3)`` tensor the project's
    neighbour selection reads -- the selection is a minimum heavy-atom distance,
    so a CA-only rule needs a carrier for it (``egnca.with_ca_positions``'s
    note). Here it supplies the **atom layout and the chain length only**: the
    CA are the rule's own and the side chains are translated with them. Nothing
    about a native *conformation* enters unless ``native_edges`` is true, which
    for a sample from the prior would be giving away the answer and is why it is
    false by default.

    Stops early and honestly on a non-finite state: ``converged`` is false and
    the coordinates are whatever the rule produced, so a diverged sample is
    counted rather than silently excluded.
    """
    if max_steps < 0:
        raise ValueError(f"max_steps must not be negative, got {max_steps}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    state = EGNCAState(
        ca=start_ca.to(rule.dtype),
        hidden=rule.initial_hidden(
            topo.n_residues, device=start_ca.device, dtype=rule.dtype
        ),
    )
    fixed = (
        egnca_neighbours(carrier_coords, topo, k, cutoff, device_neighbours)
        if native_edges
        else None
    )
    taken = 0
    last_move = float("inf")
    converged = False
    with torch.no_grad():
        for _ in range(max_steps):
            if fixed is not None:
                selection = fixed
            else:
                selection = egnca_neighbours(
                    with_ca_positions(carrier_coords, state.ca, topo),
                    topo,
                    k,
                    cutoff,
                    device_neighbours,
                )
            before = state.ca
            state = egnca_step(
                rule, state, selection.neighbours, selection.edge_mask
            )
            taken += 1
            if not bool(torch.isfinite(state.ca).all()):
                last_move = float("nan")
                break
            last_move = float((state.ca - before).norm(dim=-1).max())
            if last_move < tolerance:
                converged = True
                break
    centre = state.ca - state.ca.mean(dim=0, keepdim=True)
    return Sample(
        ca=state.ca.detach(),
        steps=taken,
        converged=converged,
        last_move=last_move,
        radius_of_gyration=float(centre.pow(2).sum(-1).mean().sqrt()),
    )


def write_samples(path: Path, samples: list[Sample], update: int) -> None:
    """**One** compressed ``.npz`` of CA coordinates for the whole batch.

    Not one file per sample. The user scratch quota on Bunya has ~313k inodes
    free, so a few thousand small files is a quota failure in the middle of a
    run rather than an untidiness. Ragged lengths are handled by concatenating
    into a flat ``(total, 3)`` array with an ``offsets`` index, so crops of
    different sizes share one file and no padding value has to be invented.

    Rewritten in place at every evaluation: the samples are a readout of the
    current weights, and keeping every generation would be the inode problem
    again by another route.
    """
    if samples:
        flat = torch.cat([s.ca.detach().cpu().to(torch.float32) for s in samples])
        offsets = np.cumsum(
            [0] + [int(s.ca.shape[0]) for s in samples], dtype=np.int64
        )
    else:
        flat = torch.zeros(0, 3, dtype=torch.float32)
        offsets = np.zeros(1, dtype=np.int64)
    np.savez_compressed(
        path,
        update=np.int64(update),
        ca=flat.numpy(),
        offsets=offsets,
        steps=np.asarray([s.steps for s in samples], dtype=np.int64),
        converged=np.asarray([s.converged for s in samples], dtype=bool),
        last_move=np.asarray([s.last_move for s in samples], dtype=np.float32),
        radius_of_gyration=np.asarray(
            [s.radius_of_gyration for s in samples], dtype=np.float32
        ),
    )


def sample_batch(
    rule: EGNCARule, dataset, indices: list[int], args, seed: int = 0
) -> list[Sample]:
    """``sample_count`` samples from the prior, on held-out crops' lengths.

    The lengths and atom layouts come from held-out crops because the selection
    needs a carrier; their coordinates are never the start state and, with
    ``native_edges`` false, never the graph either.
    """
    rule.eval()
    out: list[Sample] = []
    rng = random.Random(seed)
    for position in range(args.sample_count):
        index = indices[position % len(indices)]
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 3:
            continue
        generator = torch.Generator().manual_seed(seed * 7_919 + position)
        start = prior_coordinates(
            topo.n_residues,
            generator=generator,
            dtype=rule.dtype,
            device=native.device,
        )
        out.append(
            run_to_fixed_point(
                rule,
                topo,
                native,
                start,
                tolerance=args.sample_tolerance,
                max_steps=args.sample_max_steps,
                k=args.k,
                cutoff=args.cutoff,
                device_neighbours=args.device_neighbours,
                native_edges=args.native_edges,
            )
        )
    rule.train()
    return out


# -- the trainer ------------------------------------------------------------


#: Every reason a training update can contribute no gradient, counted
#: separately and permanently -- ``train_egnca.py``'s tuple, with ``no_window``
#: carrying one more case: ``context_fraction = 1.0`` holds every residue, so
#: there is nothing corrupted to score and the restraint is a constant zero with
#: no ``grad_fn``. That is the degenerate end of the sweep and it is counted
#: rather than crashed on, for the same reason a crop too short to hold a window
#: is counted in the port.
SKIP_REASONS = ("no_crop", "no_window", "nonfinite_loss", "nonfinite_gradient")


class _SkipUpdate(Exception):
    """This rollout produced nothing the objective can price, and why."""

    def __init__(self, reason: str) -> None:
        if reason not in SKIP_REASONS:
            raise ValueError(f"unknown skip reason {reason!r}")
        super().__init__(reason)
        self.reason = reason


def skip_report(counts: dict[str, int], window: int) -> str:
    """``skip`` as a share of the updates since the last evaluation."""
    total = sum(counts.values())
    detail = " ".join(f"{name} {counts[name]}" for name in SKIP_REASONS)
    share = 100.0 * total / window if window else float("nan")
    return f"skip {total}/{window} ({share:.1f}%: {detail})"


def build_schema() -> argparse.ArgumentParser:
    """Every setting the trainer reads, with its type and its default."""
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument("--split", default="train")
    p.add_argument("--structures", type=int, default=400)
    p.add_argument("--eval-structures", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)

    # The rule. Gala's widths, unchanged from the port: the published parameter
    # count is quoted against them.
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--message-dim", type=int, default=32)

    # The five settings the paper omits and the authors' code carries on by
    # default. Identical to the port's, and not re-argued here.
    p.add_argument("--has-coord-act", action="store_true", default=True)
    p.add_argument("--is-residual", action="store_true", default=True)
    p.add_argument("--activation", default="tanh", choices=tuple(ACTIVATIONS))
    p.add_argument("--norm-type", default="pn", choices=NORM_TYPES)

    # The rollout, as published: "15 to 25 steps, sampled uniformly".
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--steps-min", type=int, default=15)

    # THE TASK. The one axis, and the noise band it is swept against.
    p.add_argument("--context-fraction", type=float, default=0.0)
    p.add_argument("--sigma-min", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=30.0)

    # Perception. ``native-edges`` defaults FALSE here, which is the opposite of
    # the port: rebuilding from the current coordinates is this arm's research
    # question rather than its failure mode.
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--device-neighbours", action="store_true")
    p.add_argument("--native-edges", action="store_true", default=False)

    # The objective. Both halves of the unclamped all-pairs CA restraint, and
    # nothing else.
    p.add_argument("--pairwise-weight", type=float, default=1.0)
    p.add_argument("--pairwise-window-weight", type=float, default=1.0)

    # Training. The published values, as in the port.
    p.add_argument("--updates", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--pool-probability", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=250)

    # Sampling from the prior: the generative readout. ``sample-count`` 0 turns
    # it off entirely.
    p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--sample-tolerance", type=float, default=0.01)
    p.add_argument("--sample-max-steps", type=int, default=200)

    for dest in run_config.RUNTIME_ONLY:
        p.add_argument(f"--{dest.replace('_', '-')}", default=None)
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--section", default=None)
    p.add_argument("--list-sections", action="store_true")
    p.add_argument("--diff", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--stop-file", default=None)
    p.add_argument("--resume", action="store_true")
    return p


def recorded_args(args: argparse.Namespace) -> dict:
    """What the run *is*, for the checkpoint: settings without the resources."""
    return {
        key: value
        for key, value in vars(args).items()
        if key not in run_config.RUNTIME_ONLY
    }


def build_rule(args) -> EGNCARule:
    """The rule, built from the section. One call, so the keys cannot miss it."""
    return EGNCARule(
        hidden_dim=args.hidden_dim,
        message_dim=args.message_dim,
        has_coord_act=args.has_coord_act,
        is_residual=args.is_residual,
        activation=args.activation,
        norm_type=args.norm_type,
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def on_disk(specs) -> list:
    """Specs whose coordinates are present, with the path pointed at the file.

    Lifted from ``scripts/train_egnca.py``, and it has to **redirect** the path
    rather than only filter: the manifest records a ``.cif`` path while what is
    downloaded is ``.bcif``.
    """
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def objective_kwargs(args) -> dict:
    """Every setting that reaches :func:`denoise_objective`.

    One dict, built once and passed to both the training step and the held-out
    evaluation -- ``train_egnca.objective_kwargs``' reason, which is that this
    project's record carries a retraction from weights that reached training and
    not evaluation (job 32405172). There is no second place here for a key to go
    missing from.
    """
    return dict(
        context_fraction=args.context_fraction,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        pairwise_weight=args.pairwise_weight,
        pairwise_window_weight=args.pairwise_window_weight,
        k=args.k,
        cutoff=args.cutoff,
        device_neighbours=args.device_neighbours,
        native_edges=args.native_edges,
    )


KEYS = (
    "total",
    "pair_anchor",
    "pair_window",
    "sigma",
    "rmsd",
    "rmsd_fit_context",
    "start_rmsd",
    "move",
)


def score(rule, dataset, indices, args, seed: int = 0) -> dict[str, float]:
    """The held-out objective and its parts, averaged over crops that scored.

    ``n`` and ``diverged`` are part of the readout and not diagnostics, for
    ``train_egnca.score``'s reason: this rule is unbounded and its authors report
    exploding gradients, so a non-finite rollout is an ordinary outcome and a
    mean over the survivors printed without saying so would be the best-looking
    column in the log. Read ``rmsd`` against ``start_rmsd`` at the same
    ``sigma``, and read ``n`` and ``diverged`` before either.

    The noise level is drawn from a per-index generator, so the evaluation set is
    the same corruption of the same crops at every interval and the series is a
    measurement of the weights.
    """
    rule.eval()
    sums = {key: 0.0 for key in KEYS}
    counted = {key: 0 for key in KEYS}
    n = 0
    diverged = 0
    rng = random.Random(seed)
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 3:
                continue
            generator = torch.Generator().manual_seed(seed + index)
            terms, _state = denoise_objective(
                rule,
                native,
                topo,
                steps=args.steps,
                generator=generator,
                **objective_kwargs(args),
            )
            if terms.rmsd is None or not torch.isfinite(terms.total):
                diverged += 1
                continue
            for key, value in (
                ("total", float(terms.total)),
                ("pair_anchor", float(terms.pair_anchor)),
                ("pair_window", float(terms.pair_window)),
                ("sigma", terms.sigma),
                ("rmsd", terms.rmsd),
                ("rmsd_fit_context", terms.rmsd_fit_context),
                ("start_rmsd", terms.start_rmsd),
                ("move", terms.move),
            ):
                # ``rmsd_fit_context`` is nan whenever fewer than three residues
                # are held, which is every sample of the MVP. Averaging nan in
                # would turn the whole row into nan and hide the columns that
                # *are* measurements, so it is counted out of its own mean only.
                if value is None or value != value:
                    continue
                sums[key] += value
                counted[key] += 1
            n += 1
    rule.train()
    tail = {"n": float(n), "diverged": float(diverged)}
    if not n:
        return {key: float("nan") for key in KEYS} | tail
    return {
        key: (sums[key] / counted[key] if counted[key] else float("nan"))
        for key in KEYS
    } | tail


def report(scores: dict[str, float], samples: list[Sample] | None) -> str:
    out = (
        f"total {scores['total']:9.3f}  anchor {scores['pair_anchor']:9.3f}  "
        f"window {scores['pair_window']:9.3f}  sigma {scores['sigma']:6.3f}  "
        f"rmsd {scores['rmsd']:7.3f}  (fit_ctx {scores['rmsd_fit_context']:7.3f}, "
        f"start {scores['start_rmsd']:7.3f})  move {scores['move']:6.3f}  "
        f"n {int(scores['n'])}"
    )
    if scores["diverged"]:
        out += f"  DIVERGED {int(scores['diverged'])}"
    if samples:
        reached = [s for s in samples if s.converged]
        mean_steps = sum(s.steps for s in samples) / len(samples)
        out += (
            f"  steps {mean_steps:6.1f}  conv {len(reached)}/{len(samples)}  "
            f"rg {sum(s.radius_of_gyration for s in samples) / len(samples):6.2f}"
        )
    return out


def main() -> int:
    runtime = build_parser().parse_args()
    schema = build_schema()

    data = run_config.load(runtime.config)
    if runtime.list_sections:
        for name in run_config.sections(data):
            print(name)
        return 0
    if not runtime.section:
        print(
            "error: --section is required. This config defines: "
            + (", ".join(run_config.sections(data)) or "(none)")
        )
        return 2
    run_config.check_base_is_complete(data, schema)
    args = run_config.to_namespace(
        schema, run_config.resolve(data, runtime.section), runtime.section, runtime
    )

    if runtime.diff:
        other = run_config.to_namespace(
            schema, run_config.resolve(data, runtime.diff), runtime.diff, runtime
        )
        rows = [
            (key, getattr(other, key), getattr(args, key))
            for key in sorted(vars(args))
            if key != "tag" and getattr(args, key) != getattr(other, key)
        ]
        print(f"{runtime.diff} -> {runtime.section}")
        for key, was, now in rows:
            print(f"  {key}: {was!r} -> {now!r}")
        print(f"  {len(rows)} setting(s) differ")
        return 0

    if args.steps_min > args.steps:
        print(
            f"steps_min {args.steps_min} exceeds steps {args.steps}: the "
            "rollout length is drawn uniformly from [steps_min, steps] and an "
            "empty range cannot be drawn from."
        )
        return 2
    if not 0.0 <= args.context_fraction <= 1.0:
        print(f"context_fraction must be in [0, 1], got {args.context_fraction}")
        return 2
    if args.sigma_min <= 0 or args.sigma_max < args.sigma_min:
        print(
            f"sigma band [{args.sigma_min}, {args.sigma_max}] is not a "
            "positive, ordered interval, and the draw is log-uniform."
        )
        return 2

    print(f"config     : {runtime.config} [{runtime.section}]")
    print(
        "E(n)-GNCA DENOISER: the port's rule, trained toward a FIXED POINT from "
        "corrupted coordinates. Isotropic Gaussian noise at sigma drawn "
        f"log-uniformly from [{args.sigma_min:g}, {args.sigma_max:g}] A, and "
        "NO timestep reaches the rule -- the same weights run at every level, "
        "which is a diffusion model minus the schedule and minus the "
        "conditioning.\n  It places CA positions and produces NO residue "
        "orientation, so fape, twist, bond, clash and identity are not "
        "reported. Read rmsd against start_rmsd at the same sigma."
    )
    print(
        f"  CONTEXT {args.context_fraction:.0%}: that fraction of residues is "
        "held at native coordinates in contiguous runs of about "
        f"{CONTEXT_RUN_WIDTH}, and is excluded from the noise AND from the "
        "loss."
        + (
            "\n  PURE GENERATION: no residue is held, so there is no anchor "
            "anywhere and fit_ctx is nan by construction rather than missing."
            if args.context_fraction <= 0.0
            else ""
        )
        + (
            "\n  DEGENERATE: every residue is held, so nothing is corrupted, "
            "the restraint is a constant zero and every update will be skipped "
            "as no_window. This is the end of the sweep, not a run."
            if args.context_fraction >= 1.0
            else ""
        )
    )
    print(
        "  "
        + (
            "NATIVE EDGES: message passing runs over the contact graph the "
            "NATIVE coordinates select. That is the port's task statement and "
            "here it is the CONTROL arm, not the question."
            if args.native_edges
            else "LIVE EDGES: the k-NN is rebuilt from where each residue "
            "CURRENTLY is, every step. That is the research question of this "
            "arm (docs/claims/the-perception-graph-is-rebuilt-from-where-the-"
            "residue-wrongly-is.md)."
        )
    )

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    splits = read_manifest(MANIFEST)
    if args.split not in splits:
        print(f"no split {args.split!r}; available: {sorted(splits)}")
        return 1
    available = on_disk(splits[args.split])
    if not available:
        print(f"no coordinates in {PDB} -- run scripts/fetch_structures.py")
        return 1

    wanted = args.structures + args.eval_structures
    catalogue = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        catalogue,
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=max(len(catalogue), 1),
    )
    indices = list(range(len(catalogue)))
    random.Random(args.seed).shuffle(indices)
    eval_indices = indices[: args.eval_structures]
    train_indices = indices[args.eval_structures :]
    if not train_indices:
        print("no training entries left after holding some out")
        return 1

    rule = build_rule(args).to(device)
    optimiser = torch.optim.Adam(
        rule.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_text = run_config.snapshot(
        args, args.tag, runtime.config, trainer="scripts/train_denoise.py"
    )
    snapshot_path = out_dir / "config.toml"
    if (
        snapshot_path.exists()
        and snapshot_path.read_text(encoding="utf-8") != snapshot_text
    ):
        snapshot_path = out_dir / "config.resumed.toml"
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    samples_path = out_dir / "samples.npz"
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "STOP"

    start_update = 0
    best = float("inf")
    resumed_skips: dict = {}
    if args.resume and last_path.exists():
        state = ckpt.load(last_path, map_location=device)
        drifted = run_config.resume_drift(state.get("args") or {}, args)
        if drifted:
            detail = "\n".join(
                f"    {key}: checkpoint {was!r}, config now {now!r}"
                for key, (was, now) in drifted.items()
            )
            print(
                f"refusing to resume {last_path}: the config no longer "
                f"describes the run in it.\n{detail}\n"
                "Resuming would splice two experiments into one curve."
            )
            return 1
        rule.load_state_dict(state["rule"])
        optimiser.load_state_dict(state["optimiser"])
        start_update = int(state["update"])
        best = float(state.get("best", float("inf")))
        resumed_skips = state.get("skips") or {}
        ckpt.set_rng_state(state["rng"])
        ckpt.restore_python_rng(rng, state["python_rng"])
        if state.get("eval_indices"):
            eval_indices = list(state["eval_indices"])
            train_indices = [i for i in indices if i not in set(eval_indices)]
        print(f"resumed from {last_path} at update {start_update}")

    try:
        eval_pdb_ids = [catalogue[i].pdb_id for i in eval_indices]
    except (IndexError, KeyError, AttributeError) as problem:
        print(f"cannot resolve eval_indices against this catalogue: {problem!r}")
        return 1

    print(
        f"device {device}, split {args.split}: {len(train_indices)} training, "
        f"{len(eval_indices)} held out (disjoint)"
    )
    print(
        f"{rule.parameter_count()} parameters (hidden {rule.hidden_dim}, message "
        f"{rule.message_dim}), crop {args.crop_size}, k {args.k}, "
        + (
            f"{args.steps_min}-{args.steps} steps"
            if args.steps_min != args.steps
            else f"{args.steps} steps"
        )
        + ", full BPTT"
    )
    print(
        "  the authors' own defaults for this task: coord_act "
        + ("tanh" if args.has_coord_act else "NONE (ours, and it diverged)")
        + f", h update {'residual' if args.is_residual else 'REPLACING (ours)'}"
        + f", hidden act {args.activation}, node norm {args.norm_type}"
        + ", phi_x output zero-initialised and unbiased."
        + (
            ""
            if (
                args.has_coord_act
                and args.is_residual
                and args.activation == "tanh"
                and args.norm_type == "pn"
            )
            else "\n  AT LEAST ONE IS OFF, so this is an ablation of the "
            "published rule and not the port's rule."
        )
    )
    print(
        f"weights: pairwise {args.pairwise_weight:g} (corrupted-to-context), "
        f"pairwise-window {args.pairwise_window_weight:g} "
        "(corrupted-to-corrupted) -- unclamped, A^2"
        + (
            "  FULL PAIR SET"
            if args.pairwise_weight and args.pairwise_window_weight
            else "  ONE HALF ONLY, which is not the published loss"
        )
    )
    print(
        f"sampling: {args.sample_count} draw(s) per evaluation from a Gaussian "
        f"prior at rg = {RG_PREFACTOR:g} * N^{RG_EXPONENT:g} A, run to a max "
        f"per-step move below {args.sample_tolerance:g} A or {args.sample_max_steps} "
        f"steps; written to one {samples_path.name}"
        if args.sample_count
        else "sampling: OFF (sample_count 0), so this run measures denoising "
        "and not generation"
    )

    started = time.perf_counter()
    skips = {reason: int(resumed_skips.get(reason, 0)) for reason in SKIP_REASONS}
    interval_skips = {reason: 0 for reason in SKIP_REASONS}
    interval_updates = 0
    history_path = out_dir / "history.json"
    history: list[dict] = []
    if args.resume and history_path.exists():
        try:
            history = [
                row
                for row in json.loads(history_path.read_text(encoding="utf-8"))
                if row.get("update", 0) <= start_update
            ]
        except (OSError, ValueError):
            history = []

    #: Keyed by ``(structure index, crop seed)``, like the port's and for the
    #: same reason: ``load_crop`` re-crops on every visit, so a state kept under
    #: the index alone would be resumed against a topology it was never built
    #: on.
    pool: dict[tuple[int, int | None], EGNCAPoolEntry] = {}

    out_of_time = False
    stopped = False
    update = start_update
    for update in range(start_update + 1, args.updates + 1):
        try:
            resume: EGNCAPoolEntry | None = None
            key: tuple[int, int | None]
            if (
                args.pool_probability > 0
                and pool
                and rng.random() < args.pool_probability
            ):
                key = rng.choice(sorted(pool, key=lambda k: (k[0], k[1] or -1)))
                index, crop_seed = key
                try:
                    topo, native, _crop, _used = dataset.load_crop(
                        index, seed_residue=crop_seed, rng=rng
                    )
                except Exception:
                    pool.pop(key, None)
                else:
                    resume = pool[key]
                    if resume.ca.shape[0] != topo.n_residues:
                        pool.pop(key, None)
                        resume = None
            if resume is None:
                index = rng.choice(train_indices)
                try:
                    topo, native, _crop, crop_seed = dataset.load_crop(index, rng=rng)
                except Exception:
                    raise _SkipUpdate("no_crop")
                key = (index, crop_seed)
            if topo.n_residues < 3:
                raise _SkipUpdate("no_window")

            generator = torch.Generator().manual_seed(
                args.seed * 1_000_003 + update
            )
            steps = (
                rng.randint(args.steps_min, args.steps)
                if args.steps_min != args.steps
                else args.steps
            )
            terms, state = denoise_objective(
                rule,
                native,
                topo,
                steps=steps,
                generator=generator,
                resume_from=resume,
                **objective_kwargs(args),
            )
            # **Nothing corrupted carries a finite loss with no graph behind
            # it.** At ``context_fraction = 1.0`` every residue is held, so
            # ``pairwise_distance_restraint`` is a constant zero with no
            # ``grad_fn`` -- ``isfinite`` passes and ``backward()`` raises "does
            # not require grad". The port has the same statement for a crop too
            # short to hold a window; this is the same skip under the same name.
            if terms.corrupted_mask is None or not bool(terms.corrupted_mask.any()):
                raise _SkipUpdate("no_window")
            if not torch.isfinite(terms.total):
                raise _SkipUpdate("nonfinite_loss")
            if not terms.total.requires_grad:
                raise _SkipUpdate("no_window")
            optimiser.zero_grad()
            terms.total.backward()
            # **A finite loss does not imply a finite gradient**, and
            # ``clip_grad_norm_`` turns one infinity into NaN parameters: the
            # total norm is inf, the scale factor is clip/inf = 0, and
            # 0 * inf = NaN. Skipping is right rather than clamping, because a
            # clamped infinite gradient is a fabricated direction.
            if not all(
                torch.isfinite(p.grad).all()
                for p in rule.parameters()
                if p.grad is not None
            ):
                optimiser.zero_grad(set_to_none=True)
                seen = skips["nonfinite_gradient"] + 1
                if seen in (1, 10, 100) or seen % 500 == 0:
                    print(
                        f"  update {update}: non-finite gradient with a finite "
                        f"loss ({float(terms.total.detach()):.3f}); skipped. "
                        f"{seen} so far.",
                        flush=True,
                    )
                raise _SkipUpdate("nonfinite_gradient")
            torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)
            optimiser.step()

            # **The pool's bar is the sample's own corrupted start**, remembered
            # the first time this crop was seen, so there is no threshold to
            # pick and it self-calibrates per protein and per noise level. A
            # state worse than not trying is not worth resuming, and without the
            # guard the pool fills with whatever the rule diverges into.
            if args.pool_probability > 0:
                bar = (
                    resume.baseline_rmsd
                    if resume is not None
                    else terms.start_rmsd
                )
                reached = terms.rmsd
                if (
                    bar is not None
                    and reached is not None
                    and reached <= bar
                    and terms.corrupted_mask is not None
                ):
                    pool[key] = EGNCAPoolEntry(
                        ca=state.ca.detach().clone(),
                        hidden=state.hidden.detach().clone(),
                        damaged=terms.corrupted_mask.clone(),
                        baseline_rmsd=bar,
                    )
        except _SkipUpdate as skipped:
            skips[skipped.reason] += 1
            interval_skips[skipped.reason] += 1
        interval_updates += 1

        if update % args.eval_every == 0 or update == args.updates:
            scores = score(rule, dataset, eval_indices, args, seed=args.seed)
            samples = (
                sample_batch(rule, dataset, eval_indices, args, seed=args.seed)
                if args.sample_count
                else []
            )
            if samples:
                write_samples(samples_path, samples, update)
            elapsed = time.perf_counter() - started
            print(
                f"update {update:7d}  {report(scores, samples)}  "
                f"{skip_report(interval_skips, interval_updates)}  "
                f"{elapsed / max(update - start_update, 1):.2f}s/update",
                flush=True,
            )
            history.append({
                "update": update,
                **scores,
                "skips": dict(interval_skips),
                "skip_window": interval_updates,
                "sample_steps": [s.steps for s in samples],
                "sample_converged": [bool(s.converged) for s in samples],
            })
            interval_skips = {reason: 0 for reason in SKIP_REASONS}
            interval_updates = 0
            history_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
            payload = {
                "rule": rule.state_dict(),
                "optimiser": optimiser.state_dict(),
                "update": update,
                "best": best,
                "args": recorded_args(args),
                "runtime": {
                    key: getattr(args, key)
                    for key in run_config.RUNTIME_ONLY
                    if hasattr(args, key)
                },
                "eval_indices": eval_indices,
                "eval_pdb_ids": eval_pdb_ids,
                "scores": scores,
                "rng": ckpt.rng_state(),
                "python_rng": rng.getstate(),
                "skipped_updates": sum(skips.values()),
                "skips": dict(skips),
            }
            ckpt.save(last_path, payload)
            # Selection is on the held-out ``total``, the only scalar here that
            # prices the two halves of the restraint against each other.
            if scores["total"] == scores["total"] and scores["total"] < best:
                best = scores["total"]
                payload["best"] = best
                ckpt.save(best_path, payload)

        if stop_file.exists():
            print(f"stopping at update {update}: {stop_file} exists")
            stopped = True
            break
        if args.max_seconds and time.perf_counter() - started > float(
            args.max_seconds
        ):
            print(f"stopping at update {update}: out of time")
            out_of_time = True
            break

    if sum(skips.values()):
        print(
            f"{sum(skips.values())} update(s) contributed no gradient -- this "
            "run trained on fewer updates than its curve's x-axis says. By "
            "reason: "
            + ", ".join(f"{name} {skips[name]}" for name in SKIP_REASONS)
            + ". `nonfinite_loss` is the rule exploding and being excused for "
            "it; `no_window` is a property of the crop or of "
            "context_fraction = 1."
        )
    # **64, not 0.** ``scripts/bunya_train.slurm`` resubmits only on 64, and
    # ``train_egnca.py`` returns 0 here -- so a port segment that hit its
    # walltime ended its chain without saying so. A stop file is a deliberate
    # stop and stays 0.
    if out_of_time and not stopped and update < args.updates:
        print(
            f"exiting {CHECKPOINTED_NOT_FINISHED} (checkpointed, not finished) "
            f"at update {update} of {args.updates}, so the chain resubmits."
        )
        return CHECKPOINTED_NOT_FINISHED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
