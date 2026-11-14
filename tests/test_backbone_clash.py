"""The steric term ``growth_objective`` has never had.

Nothing in the growth objective says two residues may not occupy the same
space, so a regenerated window can be threaded straight through the protein and
only the RMSD notices -- weakly, and after the fact.
:func:`steric.backbone_clash_loss` is that term at growth's own resolution
(N, CA, C, O, CB), and these tests pin the four things that have to hold for it
to be readable rather than merely nonzero:

  * a crystal structure scores near zero, so the number means overlap and not
    packing density;
  * real interpenetration scores an order of magnitude more;
  * covalently close pairs are exempt, which is the failure mode -- a term that
    charges bonded geometry reads as a constant and drowns the real overlaps;
  * and the gradient pushes the overlapping atoms apart rather than together.

Every number quoted below was measured, not estimated; the measuring code is
the assertion next to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.growth import growth_links
from aanca.steric import backbone_clash_loss
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

NATIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "1UBQ.pdb"

#: What crystal ubiquitin reads, at tolerance zero, every residue alive.
#: Measured: 0.025765 A of mean overlap per scored atom, which is 9.64 A of
#: total overlap spread over 374 atoms and 64 pairs. Every one of those pairs is
#: a backbone N/O hydrogen bond (the worst is N(1)-O(17) at 2.56 A, 0.51 A
#: inside the Bondi sum of 3.07) or an i/i+2 C-N contact at about 2.96 A. So the
#: residue is hydrogen bonding, not clashing, and it is what a flat-bottom
#: penalty at tolerance zero necessarily charges.
CRYSTAL = 0.025765


@pytest.fixture(scope="module")
def ubiquitin():
    if not NATIVE.exists():
        pytest.skip(f"{NATIVE} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords, growth_links(topo)


def _all_alive(topo):
    return torch.ones(topo.n_residues, dtype=torch.bool)


def test_crystal_structure_scores_near_zero(ubiquitin):
    """0.0258 A on 1UBQ -- small enough that the term reports overlap only.

    The bound asserted is the measured value plus a little, not a guess. The
    thing it protects against is an exemption bug, which does not shift this
    number slightly: scoring the peptide-bond block instead of exempting it
    takes 1UBQ to 1.118 A, forty-three times over, because C(i)-N(i+1) is a
    1.33 A bond against a 3.25 A vdW sum and every residue in the chain has
    one. That is the constant the term would read if the exemption were wrong.
    """
    topo, coords, links = ubiquitin
    loss = backbone_clash_loss(coords, topo, _all_alive(topo), links)
    assert loss.shape == ()
    assert 0.0 <= float(loss) < 0.03
    assert float(loss) == pytest.approx(CRYSTAL, abs=1e-4)


def test_hydrogen_bonds_are_the_whole_crystal_residue(ubiquitin):
    """The residue is tolerance-shaped, which is how we know what it is.

    At tolerance 0.25 A -- less than a sixth of AlphaFold2's 1.5 --
    ubiquitin falls to 0.0020 A, and by 0.5 A to 1.5e-05. A structure with a
    real clash in it would not behave like that. Recorded here because it is
    the argument for wiring the term with a small nonzero tolerance.
    """
    topo, coords, links = ubiquitin
    alive = _all_alive(topo)
    at_quarter = float(backbone_clash_loss(coords, topo, alive, links, 0.25))
    at_half = float(backbone_clash_loss(coords, topo, alive, links, 0.5))
    assert at_quarter == pytest.approx(0.00196, abs=1e-4)
    assert at_half < 1e-4
    assert float(backbone_clash_loss(coords, topo, alive, links, 1.5)) == 0.0


def test_the_crystal_floor_is_not_a_property_of_ubiquitin(ubiquitin):
    """The same number on lysozyme and on haemoglobin, which is the point.

    193L (129 residues, one chain) reads 0.0246 A and 4HHB (574 residues, four
    chains and three chain breaks) reads 0.0283 A, against ubiquitin's 0.0258.
    A floor that is the same to a few thousandths across a monomer, a larger
    monomer and a four-chain assembly is the hydrogen-bond density of folded
    protein, not an artefact of one structure or of the per-chain bookkeeping.
    4HHB also exercises the chain breaks: ``links`` is false at three places,
    and the residue pairs either side of them are scored, correctly, as
    non-bonded.
    """
    for name, expected in (("193L.pdb", 0.0246), ("4HHB.pdb", 0.0283)):
        path = NATIVE.parent / name
        if not path.exists():
            pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
        topo, coords = feature_topology(load_protein(path))
        links = growth_links(topo)
        loss = backbone_clash_loss(coords, topo, _all_alive(topo), links)
        assert float(loss) == pytest.approx(expected, abs=1e-3)


def test_superposed_residues_score_large(ubiquitin):
    """Two residues 17.7 A apart, driven onto one point.

    Residue 50 translated so its CA lands on residue 5's: 0.141 A, 5.5 times
    the crystal value, from one residue's five atoms interpenetrating one
    other's. Only five atoms of 374 move, which is why the ratio is 5x and not
    500x -- the per-atom normalisation is doing exactly what it is for.
    """
    topo, coords, links = ubiquitin
    alive = _all_alive(topo)
    ca = topo.frame_idx[:, 1]
    i, j = 5, 50
    separation = (coords[i, ca[i]] - coords[j, ca[j]]).norm()
    assert float(separation) == pytest.approx(17.70, abs=0.05)

    moved = coords.clone()
    moved[j] = coords[j] + (coords[i, ca[i]] - coords[j, ca[j]])
    loss = float(backbone_clash_loss(moved, topo, alive, links))
    assert loss == pytest.approx(0.141, abs=0.005)
    assert loss > 4.0 * CRYSTAL


def test_window_threaded_through_the_core_scores_larger(ubiquitin):
    """The actual failure the term exists to catch: a regenerated loop inside.

    Residues 3-10 translated bodily into the core scores 0.472 A, eighteen
    times the crystal value. That is the signal a regeneration run needs, and
    it is well clear of the hydrogen-bond floor.
    """
    topo, coords, links = ubiquitin
    ca = topo.frame_idx[:, 1]
    shift = coords[40, ca[40]] - coords[5, ca[5]]
    moved = coords.clone()
    moved[3:11] = coords[3:11] + shift
    loss = float(backbone_clash_loss(moved, topo, _all_alive(topo), links))
    assert loss == pytest.approx(0.472, abs=0.01)
    assert loss > 10.0 * CRYSTAL


def test_bonded_neighbours_are_not_penalised(ubiquitin):
    """A peptide-bonded pair at crystal geometry charges exactly nothing.

    Scored on its own -- only residues 10 and 11 alive -- the loss is 0.0, not
    "small": every pair across the bond is exempt, so the term is identically
    zero rather than cancelling. Contrast residues 30 and 32, which are not
    bonded and do have a genuine 0.0082 A tight contact, so the exemption is
    not simply switching the whole term off.
    """
    topo, coords, links = ubiquitin
    n_res = topo.n_residues

    for i in (10, 30):
        assert bool(links[i])
        bonded = torch.zeros(n_res, dtype=torch.bool)
        bonded[i] = True
        bonded[i + 1] = True
        assert float(backbone_clash_loss(coords, topo, bonded, links)) == 0.0

    skip = torch.zeros(n_res, dtype=torch.bool)
    skip[30] = True
    skip[32] = True
    assert float(backbone_clash_loss(coords, topo, skip, links)) > 1e-3


def test_intra_residue_pairs_are_exempt(ubiquitin):
    """One residue alone charges nothing, at crystal geometry or scaled down.

    Every pair among N, CA, C, O, CB is at most three bonds apart, so all of
    them fall under ``CLASH_MIN_SEPARATION`` and none is a steric observation.
    Shrinking the residue by half -- which puts its own atoms far inside each
    other's radii -- still scores zero, so the exemption is structural and not
    an accident of the crystal's spacing.
    """
    topo, coords, links = ubiquitin
    alone = torch.zeros(topo.n_residues, dtype=torch.bool)
    alone[20] = True
    assert float(backbone_clash_loss(coords, topo, alone, links)) == 0.0

    ca = topo.frame_idx[:, 1]
    squashed = coords.clone()
    centre = coords[20, ca[20]]
    squashed[20] = centre + 0.5 * (coords[20] - centre)
    assert float(backbone_clash_loss(squashed, topo, alone, links)) == 0.0


def test_gradient_pushes_overlapping_atoms_apart(ubiquitin):
    """Hand-built two-residue clash: residues 5 and 50, CAs put 2.0 A apart.

    The sign is the whole point. The loss is ``relu(r_i + r_j - d)``, so
    ``dL/dx`` for an atom points *towards* its partner, and a descent step
    ``x -= lr * grad`` therefore moves it away. Asserted as the dot product of
    the gradient with the separation vector being negative, plus a finite
    difference: pulling the pair further apart lowers the loss.
    """
    topo, coords, links = ubiquitin
    i, j = 5, 50
    ca = topo.frame_idx[:, 1]
    pair = torch.zeros(topo.n_residues, dtype=torch.bool)
    pair[i] = True
    pair[j] = True

    offset = coords[j, ca[j]] - coords[i, ca[i]]
    direction = offset / offset.norm()
    built = coords.clone()
    built[j] = coords[j] - offset + 2.0 * direction
    separation = built[j, ca[j]] - built[i, ca[i]]
    assert float(separation.norm()) == pytest.approx(2.0, abs=1e-6)

    leaf = built.detach().clone().requires_grad_(True)
    loss = backbone_clash_loss(leaf, topo, pair, links)
    assert float(loss.detach()) > 0.1
    loss.backward()
    grad = leaf.grad
    assert grad is not None
    assert torch.isfinite(grad).all()

    # Residue j's CA feels a force; it points back towards residue i's CA, so
    # descending it separates them.
    pull = grad[j, ca[j]]
    assert float(pull.norm()) > 0.0
    assert float((pull * separation).sum()) < 0.0
    # And the mirror image on the other atom of the pair.
    assert float((grad[i, ca[i]] * separation).sum()) > 0.0

    # Finite difference along the separation: further apart is cheaper.
    further = built.clone()
    further[j] = built[j] + 0.1 * direction
    assert float(backbone_clash_loss(further, topo, pair, links)) < float(
        loss.detach()
    )


def test_dead_residues_are_ignored(ubiquitin):
    """Killing the clashing residues returns the loss to the crystal value.

    A dead slot's coordinates are a parking position, not a claim about where
    anything is, so a term that scored them would charge the baseline for
    atoms the rule has not placed yet.
    """
    topo, coords, links = ubiquitin
    ca = topo.frame_idx[:, 1]
    moved = coords.clone()
    moved[3:11] = coords[3:11] + (coords[40, ca[40]] - coords[5, ca[5]])

    alive = _all_alive(topo)
    assert float(backbone_clash_loss(moved, topo, alive, links)) > 0.4

    alive[3:11] = False
    quiet = float(backbone_clash_loss(moved, topo, alive, links))
    assert quiet == pytest.approx(0.0235, abs=1e-3)
    # The clash is gone; what is left is the hydrogen-bond floor of the smaller
    # alive set, not a residue of the displacement.
    assert quiet < 0.03

    nobody = torch.zeros(topo.n_residues, dtype=torch.bool)
    none_loss = backbone_clash_loss(moved, topo, nobody, links)
    assert float(none_loss) == 0.0
    assert none_loss.shape == ()


def test_the_depth_tables_count_separation_through_the_peptide_bond():
    """15 of the 25 i/i+1 pairs are steric observations, and these are them.

    Pinned by name rather than by count alone, because the count is what a
    hand tally gets wrong -- an early version of
    ``docs/claims/the-clash-term-does-not-see-the-ramachandran-plot.md`` said
    11. The path from an atom of residue i to one of residue i+1 runs
    ``a -> C(i) -> N(i+1) -> b``, so the separation is
    ``to_c[a] + 1 + to_n[b]``.
    """
    from aanca import residue_constants as rc
    from aanca.steric import GROWTH_DEPTH_TO_C, GROWTH_DEPTH_TO_N

    names = ("N", "CA", "C", "O", "CB")
    scored = {
        f"{a}-{b}"
        for a in names
        for b in names
        if GROWTH_DEPTH_TO_C[a] + 1 + GROWTH_DEPTH_TO_N[b]
        >= rc.CLASH_MIN_SEPARATION
    }
    assert scored == {
        "N-CA", "N-C", "N-O", "N-CB",
        "CA-C", "CA-O", "CA-CB",
        "C-O",
        "O-C", "O-O", "O-CB",
        "CB-CA", "CB-C", "CB-O", "CB-CB",
    }
    # The exempt ten are the ones psi and omega fix, which
    # ``peptide_bond_restraint`` already holds.
    assert len(scored) == 15


def test_bonded_pairs_scores_the_distant_pairs_and_only_those(ubiquitin):
    """``bonded_pairs=True`` really does score the 15 distant pairs -- and on a
    crystal backbone they charge NOTHING, which is the finding.

    The default drops the bonded pair whole, so a bonded pair reads 0.0 however
    it is deformed. Under ``bonded_pairs=True`` it still reads 0.0 at crystal
    geometry, even at tolerance zero: the 15 pairs that are four or more bonds
    apart sit comfortably outside their van der Waals bounds in a real
    structure. Collapse the two residues onto each other and the switch charges
    while the default stays silent, which is what proves the pairs are in the
    sum at all.
    """
    topo, coords, links = ubiquitin
    n_res = topo.n_residues

    bonded = torch.zeros(n_res, dtype=torch.bool)
    bonded[10] = True
    bonded[11] = True
    assert float(backbone_clash_loss(coords, topo, bonded, links)) == 0.0
    assert float(
        backbone_clash_loss(coords, topo, bonded, links, bonded_pairs=True)
    ) == 0.0

    # Residue 11's atoms dropped onto residue 10's CA. The peptide bond is
    # nonsense here; the point is only that the pairs are scored.
    squeezed = coords.clone()
    squeezed[11] = coords[10, topo.frame_idx[10, 1]]
    assert float(backbone_clash_loss(squeezed, topo, bonded, links)) == 0.0
    assert float(
        backbone_clash_loss(squeezed, topo, bonded, links, bonded_pairs=True)
    ) > 0.1


def test_bonded_pairs_leaves_non_bonded_pairs_alone(ubiquitin):
    """The switch touches only pairs ``links`` calls bonded.

    Residues 30 and 32 are not bonded, so every atom pair between them is a
    steric observation under either setting and the two readings must agree
    exactly. Without this, a bug in the separation mask could silently change
    every pair in the protein.
    """
    topo, coords, links = ubiquitin
    skip = torch.zeros(topo.n_residues, dtype=torch.bool)
    skip[30] = True
    skip[32] = True
    assert float(backbone_clash_loss(coords, topo, skip, links)) == pytest.approx(
        float(backbone_clash_loss(coords, topo, skip, links, bonded_pairs=True))
    )


def test_a_whole_crystal_structure_still_scores_near_zero_with_bonded_pairs(
    ubiquitin,
):
    """The floor the term is read against, with the exemption narrowed.

    ``clash_bonded`` changes what a reported `clash` MEANS, so the floor has to
    be on the record beside the default's 0.0258 A. Measured at tolerance 0 and
    at the arm's 0.5.
    """
    topo, coords, links = ubiquitin
    alive = _all_alive(topo)
    strict = float(
        backbone_clash_loss(coords, topo, alive, links, 0.0, bonded_pairs=True)
    )
    forgiving = float(
        backbone_clash_loss(coords, topo, alive, links, 0.5, bonded_pairs=True)
    )
    assert strict < 0.1
    assert forgiving < 1e-3
