"""Plumbing the CASP diagnostics share: which residues, which angles, which files.

Three scripts now ask questions about the same residues of the same targets --
``chi1_histogram.py`` (what the error is made of), ``basin_probe.py`` (is it a
basin property), ``clamp_test.py`` (does the rule read its neighbours) -- and
their numbers are only comparable if they are measuring the same population.
Proline exclusion and the chi1 mask are exactly the kind of detail that drifts
when it is spelled three times, and a drift there moves every percentage
quietly. So it is spelled once.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .casp import RESULTS
from .structure import load_protein
from .torch_geometry import measure_chis, wrap_to_pi
from .torch_graph import feature_topology


def chi1_degrees(coords: Tensor, torsion) -> Tensor:
    """chi1 of every residue, in degrees, on the CPU."""
    return torch.rad2deg(measure_chis(coords, torsion)[:, 0]).cpu()


def chi1_residues(torsion) -> Tensor:
    """Residues with a real chi1 that the three-well model describes.

    Proline is excluded: its chi1 is pinned by the pyrrolidine ring to about
    +/-25 degrees, so the g-/t/g+ model does not describe it and including it
    would blur every well statistic.
    """
    has_chi1 = torsion.chi_mask[:, 0].cpu()
    is_pro = torch.tensor(
        [name == "PRO" for name in torsion.resnames], dtype=torch.bool
    )
    return has_chi1 & ~is_pro


def circular_error(predicted: Tensor, target: Tensor) -> Tensor:
    """``|predicted - target|`` in degrees, by the shortest arc."""
    return wrap_to_pi(torch.deg2rad(predicted - target)).rad2deg().abs()


def percent(flags: Tensor) -> float:
    """The share of ``flags`` that are true, as a percentage."""
    return 100.0 * float(flags.float().mean())


def each_target(stems, backbone: str, device: str):
    """``(index, stem, topo, native_coords)``, skipping unreadable files.

    The index is the seed offset, so it counts every stem whether or not the
    file loaded -- a target that starts failing to parse must not shift the
    seeds of the targets after it, or two runs of a diagnostic stop being
    comparable.
    """
    for index, stem in enumerate(stems):
        try:
            protein = load_protein(
                RESULTS / "native" / backbone / f"{stem}.pdb", repair_columns=True
            )
            topo, native_coords = feature_topology(protein, device=device)
        except Exception:  # malformed or unreadable; benchmark.py reports these
            continue
        yield index, stem, topo, native_coords
