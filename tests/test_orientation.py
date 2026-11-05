"""The term for a peptide bond's rotation, which the objective did not have.

``peptide_bond_restraint`` charges a bond's length and holds omega planar. That
leaves the bond's *rotation* -- how one residue's frame is turned relative to
the next, which is what phi and psi are -- charged only through FAPE, clamped
at 10 A and averaged over every residue that exists.

And that is the quantity measured to be wrong: `segment_placement_probe.py`
puts a four-residue window at 2.28 A on ``runs/g4/best.pt`` where a real
tetrapeptide superposes to a few tenths, and
`lever_arm_probe.py` converts that to about fifteen degrees per birth. These
tests pin the term that charges it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from aanca.growth import growth_links, relative_frame_restraint
from aanca.structure import load_protein
from aanca.backbone import rotation_from_vector
from aanca.torch_graph import feature_topology

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(path))
    return topo, coords, growth_links(topo)


def test_a_crystal_structure_scores_zero(ubiquitin):
    """The property that makes it reportable unweighted.

    Like the bond and clash terms, this is a physical quantity with a known
    value on a real structure, so the number in a training log can be read
    against a crystal's directly rather than against another run's.
    """
    topo, coords, links = ubiquitin
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    loss, degrees = relative_frame_restraint(coords, coords, topo, alive, links)
    assert float(loss) < 1e-9
    assert degrees < 1e-4


def test_a_bodily_rotation_scores_zero(ubiquitin):
    """Invariant to a global rigid motion, exactly as FAPE is.

    This is why the term compares ``R_i^T R_j`` and not the frames themselves.
    A structure that is correct but differently oriented must not be charged,
    or the term would be asking the rule to guess a global frame it has no way
    to know.
    """
    topo, coords, links = ubiquitin
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    rot = rotation_from_vector(torch.tensor([[0.4, -0.9, 0.2]], dtype=coords.dtype))[0]
    moved = coords @ rot.transpose(0, 1) + torch.tensor(
        [12.0, -3.0, 40.0], dtype=coords.dtype
    )
    loss, degrees = relative_frame_restraint(moved, coords, topo, alive, links)
    assert float(loss) < 1e-9
    assert degrees < 1e-4


@pytest.mark.parametrize("theta", [0.05, 0.15, 0.30])
def test_the_reported_angle_is_the_angle(ubiquitin, theta):
    """Rotating one residue should read as that rotation, in degrees.

    The loss itself is ``1 - cos theta``, which is the right shape for a
    gradient and useless for reading a log. The degrees are what the probes
    quote, so they have to be the real geodesic angle and not a proxy.
    """
    topo, coords, links = ubiquitin
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[20:23] = True
    axis = torch.tensor([[0.0, 0.0, theta]], dtype=coords.dtype)
    rot = rotation_from_vector(axis)[0]
    moved = coords.clone()
    centre = coords[21, 1]
    moved[21] = (coords[21] - centre) @ rot.transpose(0, 1) + centre

    _loss, degrees = relative_frame_restraint(moved, coords, topo, alive, links)
    # Two bonded pairs touch residue 21 -- (20, 21) and (21, 22) -- and both
    # are rotated by the same angle, so the mean is that angle.
    assert degrees == pytest.approx(math.degrees(theta), rel=0.02)


def test_the_loss_is_one_minus_cosine(ubiquitin):
    """The trace form and the angle must be the same quantity.

    They are reported side by side and a reader will compare them, so an
    algebraic slip between the two would be invisible and misleading.
    """
    topo, coords, links = ubiquitin
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[20:22] = True
    theta = 0.25
    rot = rotation_from_vector(torch.tensor([[0.0, theta, 0.0]], dtype=coords.dtype))[0]
    moved = coords.clone()
    centre = coords[21, 1]
    moved[21] = (coords[21] - centre) @ rot.transpose(0, 1) + centre
    loss, degrees = relative_frame_restraint(moved, coords, topo, alive, links)
    assert float(loss) == pytest.approx(1.0 - math.cos(math.radians(degrees)), rel=1e-5)


def test_only_bonded_pairs_alive_at_both_ends_count(ubiquitin):
    """The same gate the bond restraint has, for the same reason.

    A dormant residue is parked five hundred Angstrom away and its frame is
    whatever its parked pose left; charging a bond to it would be charging the
    parking.
    """
    topo, coords, links = ubiquitin
    theta = 0.3
    rot = rotation_from_vector(torch.tensor([[0.0, 0.0, theta]], dtype=coords.dtype))[0]
    moved = coords.clone()
    centre = coords[21, 1]
    moved[21] = (coords[21] - centre) @ rot.transpose(0, 1) + centre

    only_neighbour = torch.zeros(topo.n_residues, dtype=torch.bool)
    only_neighbour[40:44] = True
    loss, degrees = relative_frame_restraint(
        moved, coords, topo, only_neighbour, links
    )
    assert float(loss) < 1e-9, "a rotation outside the alive set was charged"

    nothing = torch.zeros(topo.n_residues, dtype=torch.bool)
    loss, degrees = relative_frame_restraint(moved, coords, topo, nothing, links)
    assert float(loss) == 0.0 and degrees == 0.0


def test_gradient_reaches_the_coordinates(ubiquitin):
    """Otherwise it is a diagnostic rather than a term.

    The point of adding it is that the rotation gets a gradient of its own
    instead of a share of FAPE's clamped mean, so the gradient has to arrive at
    the coordinates the heads produce.
    """
    topo, coords, links = ubiquitin
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[20:30] = True
    moved = coords.clone().requires_grad_(True)
    loss, _degrees = relative_frame_restraint(moved, coords, topo, alive, links)
    # Not exactly zero: the native structure's own frames differ from the
    # idealised ones, but a *perfect* match would give no gradient, so perturb.
    (loss + (moved[25] ** 2).sum() * 1e-6).backward()
    assert moved.grad is not None
    assert float(moved.grad[20:30].abs().sum()) > 0.0
