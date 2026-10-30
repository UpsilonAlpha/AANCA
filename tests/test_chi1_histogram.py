"""The well assignment behind ``scripts/chi1_histogram.py``.

The histograms are eyeballed; the assignment is not. It is the part that
decides between the two hypotheses -- "picks a real rotamer, picks wrong"
against "lands between rotamers" -- and both of its failure modes produce a
clean-looking answer. Wrapping the wrong way puts a prediction at 170 degrees
350 degrees from trans instead of 10, which would report a model that sits on
the wells as one that sits nowhere near them; and the boundary between two
wells has to fall where it falls, or the match rate is biased toward whichever
well is listed first.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "chi1_histogram", ROOT / "scripts" / "chi1_histogram.py"
)
chi1_histogram = importlib.util.module_from_spec(SPEC)
sys.modules["chi1_histogram"] = chi1_histogram
SPEC.loader.exec_module(chi1_histogram)

NAMES = list(chi1_histogram.WELLS)


def assign(degrees):
    index, offset = chi1_histogram.nearest_well(torch.tensor(degrees))
    return [NAMES[i] for i in index.tolist()], offset.tolist()


def test_angles_on_a_well_are_assigned_to_it_at_zero_distance():
    names, offsets = assign([-60.0, 180.0, 60.0])
    assert names == ["g-", "t", "g+"]
    assert offsets == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_the_wrap_is_shortest_arc_not_raw_difference():
    """170 is 10 degrees from trans, not 350. Getting this wrong would make a
    model that sits on the wells look like one that sits between them."""
    names, offsets = assign([170.0, -170.0, -179.0])
    assert names == ["t", "t", "t"]
    for offset in offsets:
        assert abs(offset) <= 11.0


def test_an_angle_between_two_wells_is_far_from_both():
    """The signature the script exists to detect: the circular mean of g- and
    t is 120 degrees from nothing and 60 from each neighbour."""
    _, offsets = assign([-120.0, 120.0, 0.0])
    for offset in offsets:
        assert abs(offset) == pytest.approx(60.0, abs=1e-3)


def test_the_boundary_falls_halfway_between_neighbouring_wells():
    names, _ = assign([-119.0, -121.0, 119.0, 121.0])
    assert names == ["g-", "t", "g+", "t"]


def test_a_histogram_counts_every_value_once():
    values = torch.tensor([-180.0, -90.0, 0.0, 90.0, 179.0])
    counts = chi1_histogram.histogram(values, -180.0, 180.0, 36)
    assert sum(counts) == values.numel()
    assert counts[0] == 1  # -180 lands in the first bin, not outside it
    assert counts[-1] == 1  # and 179 in the last


def test_a_histogram_clamps_rather_than_dropping_out_of_range_values():
    """A wrap bug upstream would put values outside +/-180; losing them
    silently would hide it, so they pile into the end bins instead."""
    values = torch.tensor([-400.0, 400.0])
    counts = chi1_histogram.histogram(values, -180.0, 180.0, 36)
    assert sum(counts) == 2
    assert counts[0] == 1 and counts[-1] == 1


def test_an_empty_input_gives_empty_bins_rather_than_an_error():
    assert chi1_histogram.histogram(torch.tensor([]), -180.0, 180.0, 36) == [0] * 36
