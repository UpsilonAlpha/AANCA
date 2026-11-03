"""The rotamer wells, and the ASCII plotting the diagnostics read through.

Two scripts share this vocabulary -- ``scripts/chi1_histogram.py`` asks what
the chi1 error is made of, ``scripts/basin_probe.py`` asks whether that error
is a basin-of-attraction property -- and a second spelling of the well
assignment would make their numbers incomparable for no reason. The benchmark
results path was already duplicated into a fourth, wrong spelling once
(``aanca.casp.RESULTS``); this is the same lesson applied before it costs
anything.

The plotting lives here too because it exists for the same reason: these
diagnostics are read in a SLURM job log that comes back over a git push, so
every figure has to survive as text.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .torch_geometry import wrap_to_pi

#: The three canonical chi1 rotamer wells, in degrees. Lovell et al. (2000),
#: the penultimate rotamer library; every packing paper in this line uses the
#: same three names.
WELLS = {"g-": -60.0, "t": 180.0, "g+": 60.0}

#: Names in the order :func:`nearest_well` indexes them.
WELL_NAMES = tuple(WELLS)


def nearest_well(degrees: Tensor) -> tuple[Tensor, Tensor]:
    """``(index, signed distance)`` to the nearest of :data:`WELLS`.

    Distance is wrapped, so a prediction at 170 is 10 degrees from trans and
    not 350 from it.
    """
    # **On ``degrees``' device**, which is the whole bug of job 31736674: this
    # built the centres on the CPU and subtracted them from a GPU tensor, so
    # every path that assigns a rotamer well died on a card with "found at
    # least two devices". It is the only line in the identity track that does
    # arithmetic against a constant built here rather than one carried on the
    # topology, and so the only one that could.
    #
    # Invisible to 809 CPU tests and to the first two GPU checks alike, because
    # the two earlier checks never reached it -- a CPU generator raised first
    # (``aanca.rng``), and then a CPU topology raised first
    # (``scripts/device_check.py``). Three device faults in the same code path,
    # each hidden behind the one before it.
    centres = torch.tensor(
        list(WELLS.values()), dtype=degrees.dtype, device=degrees.device
    )
    offsets = wrap_to_pi(torch.deg2rad(degrees.unsqueeze(-1) - centres)).rad2deg()
    index = offsets.abs().argmin(dim=-1)
    return index, offsets.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def histogram(values: Tensor, low: float, high: float, bins: int) -> list[int]:
    """Counts per equal-width bin; values outside the range are clamped in."""
    if values.numel() == 0:
        return [0] * bins
    width = (high - low) / bins
    index = ((values - low) / width).floor().to(torch.long).clamp(0, bins - 1)
    counts = torch.zeros(bins, dtype=torch.long)
    return counts.scatter_add_(0, index, torch.ones_like(index)).tolist()


def draw(counts: list[int], low: float, high: float, width: int = 48) -> str:
    """An ASCII bar chart, because this is read in a job log over a git push."""
    peak = max(counts) or 1
    step = (high - low) / len(counts)
    total = sum(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        left = low + i * step
        bar = "#" * round(width * count / peak)
        lines.append(
            f"  {left:7.0f} to {left + step:7.0f} | {bar:<{width}} "
            f"{count:6d}  {100 * count / total:5.1f}%"
        )
    return "\n".join(lines)
