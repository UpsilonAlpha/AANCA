"""The steric term: does it read zero where it must, and is the fast path right?

Two of these matter more than the rest.

``test_a_crystal_structure_carries_no_overlap`` is the validity check for the
whole module. If a crystal structure scores above zero, some exemption or some
radius is wrong, and the term would spend training pulling the rule *away* from
the right answer -- silently, because the reconstruction loss would absorb the
blame.

``test_the_fast_path_matches_a_brute_force_reference`` re-derives the same
number from independent logic: explicit Python loops over atom *names*, no
padding, no neighbour list, no broadcasting. Every bug the vectorised version
can have -- a transposed mask, an off-by-one in the triangle, padded atoms
leaking in, the disulfide exemption applied to the wrong pair -- changes one
and not the other.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch

from aanca import residue_constants as rc
from aanca.data import RepackingDataset, StructureSpec, uniform_deltas
from aanca.steric import (
    DEFAULT_CUTOFF,
    DEFAULT_TOLERANCE,
    _EPS,
    clash_loss,
    steric_topology,
)
from aanca.torch_geometry import apply_chi_deltas

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


def _load(pdb_id: str):
    """``(FeatureTopology, native)``. Most tests here only want the torsion
    topology; the objective test needs the whole thing, since that is what
    ``training_objective`` takes."""
    path = DATA / f"{pdb_id}.pdb"
    if not path.exists():
        pytest.skip("run scripts/fetch_test_structures.py")
    specs = [StructureSpec(pdb_id=pdb_id, path=path, group=pdb_id)]
    dataset = RepackingDataset(specs, concentrations=(0.0,))
    topo, native, _indices, _seed = dataset.load_crop(0, seed_residue=0)
    return topo, native


@pytest.fixture(scope="module")
def ubiquitin_topology():
    return _load("1UBQ")


@pytest.fixture(scope="module")
def ubiquitin(ubiquitin_topology):
    topo, native = ubiquitin_topology
    return topo.torsion, native


@pytest.fixture(scope="module")
def lysozyme():
    topo, native = _load("193L")
    return topo.torsion, native


# --- the chemistry tables ----------------------------------------------------


def test_every_standard_atom_gets_an_element_and_a_radius():
    seen = set()
    for resname in rc.AA3:
        names = (
            set(rc.BACKBONE_ATOMS)
            | set(rc.TERMINAL_ATOMS)
            | set(rc.sidechain_atoms(resname))
        )
        for name in names:
            seen.add(rc.element(name))
            assert 1.0 < rc.vdw_radius(name) < 2.5
    assert seen == {"C", "N", "O", "S"}


def test_an_unassignable_atom_name_raises_rather_than_defaulting():
    """A default would hand a metal or a modified residue carbon's radius and
    report no problem, which is the one failure mode a steric term cannot
    survive quietly."""
    for name in ("FE", "ZN", "H", "", "1HB"):
        with pytest.raises(ValueError):
            rc.element(name)


def test_the_radii_are_alphafold2s():
    assert rc.VAN_DER_WAALS_RADIUS == {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


def test_bond_separation_counts_bonds_not_atoms():
    arg = rc.bond_separation("ARG")
    assert arg[("CA", "CB")] == 1
    assert arg[("N", "CB")] == 2
    assert arg[("CA", "CG")] == 2
    assert arg[("CA", "CD")] == 3
    assert arg[("CB", "NH1")] == 5
    assert arg[("CB", "CA")] == arg[("CA", "CB")]


def test_the_c_terminal_oxt_is_bonded_even_though_the_bond_tables_omit_it():
    """Without the C-OXT bond, OXT reads as non-bonded to C at 1.25 A and would
    be charged as a severe clash in every structure with an intact
    C-terminus -- a constant penalty no rotamer could relieve."""
    sep = rc.bond_separation("ALA")
    assert sep[("C", "OXT")] == 1
    assert sep[("O", "OXT")] == 2
    assert ("C", "OXT") in rc.CLASH_EXEMPT_PAIRS["ALA"]
    assert ("O", "OXT") in rc.CLASH_EXEMPT_PAIRS["ALA"]


def test_the_exemptions_cover_bonds_angles_and_torsions_and_stop_there():
    """1-5 and beyond is where a long side chain folding back on itself lives,
    and it is the only separation a torsion term cannot already see."""
    exempt = rc.CLASH_EXEMPT_PAIRS["LYS"]
    sep = rc.bond_separation("LYS")
    assert ("CA", "CD") in exempt and sep[("CA", "CD")] == 3
    assert ("CA", "CE") not in exempt and sep[("CA", "CE")] == 4
    assert ("N", "NZ") not in exempt
    for resname in rc.AA3:
        for (a, b), distance in rc.bond_separation(resname).items():
            assert ((a, b) in rc.CLASH_EXEMPT_PAIRS[resname]) == (
                distance < rc.CLASH_MIN_SEPARATION
            )


def test_the_exemptions_are_symmetric_and_include_each_atom_with_itself():
    for resname in ("ALA", "TRP", "PRO"):
        exempt = rc.CLASH_EXEMPT_PAIRS[resname]
        for a, b in exempt:
            assert (b, a) in exempt
        for name in rc.sidechain_atoms(resname):
            assert (name, name) in exempt


# --- the headline: zero on a crystal structure -------------------------------


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
def test_a_crystal_structure_carries_no_overlap(fixture, request):
    """At AlphaFold2's tolerance a deposited structure must score exactly 0.

    Not approximately: the flat bottom means a correct structure is at the
    penalty's zero, not near it. This is the test that would catch a wrong
    radius, a missing exemption or an unhandled bond.
    """
    torsion, native = request.getfixturevalue(fixture)
    assert float(clash_loss(native, torsion, steric_topology(torsion, native))) == 0.0


def test_a_wrecked_structure_scores_well_above_a_crystal_one(ubiquitin):
    torsion, native = ubiquitin
    steric = steric_topology(torsion, native)
    deltas = uniform_deltas(torsion.chi_mask, torch.Generator().manual_seed(0))
    wrecked = apply_chi_deltas(native, deltas, torsion)
    assert float(clash_loss(wrecked, torsion, steric)) > 0.01


def test_the_term_is_monotone_in_the_tolerance(ubiquitin):
    """Less slack cannot charge less. A failure here means a sign or a mask is
    inverted somewhere the crystal-structure test cannot see, because that one
    is comparing against zero from only one side."""
    torsion, native = ubiquitin
    steric = steric_topology(torsion, native)
    values = [
        float(clash_loss(native, torsion, steric, tolerance=t))
        for t in (1.5, 1.0, 0.5, 0.0, -0.5)
    ]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] > values[-2] > 0.0


# --- the brute-force reference ------------------------------------------------


def _reference(coords, torsion, steric, tolerance):
    """The same quantity from independent logic: names, not indices.

    Deliberately slow and deliberately ignorant of the neighbour list, the
    padding and the broadcasting, so that it shares no bug with the thing it
    checks.

    The one thing it does share is ``steric._EPS``: ``_overlap`` measures
    ``sqrt(d2 + eps)`` on purpose, to keep the derivative at a padded pair's
    zero distance finite. That softening biases every charged pair low by about
    ``eps / 2d``, i.e. ~1.7e-9 A, and the bias accumulates with the number of
    charged pairs -- at tolerance -0.75 on 1UBQ there are 730 of them and the
    two numbers part company at the ninth decimal. Matching the epsilon here
    buys back an exact comparison (agreement to 8e-16) without sharing any
    *logic*: the indices, the masks, the triangle and the padding are still
    derived independently, so every bug this test is for still breaks it.
    """
    atoms = []
    for r, names in enumerate(torsion.atom_names):
        for i, name in enumerate(names):
            atoms.append((r, i, name))
    partner = [int(p) for p in steric.partner]
    total = 0.0
    for (ra, ia, na), (rb, ib, nb) in itertools.combinations(atoms, 2):
        if na in rc.BACKBONE_ATOMS | rc.TERMINAL_ATOMS and (
            nb in rc.BACKBONE_ATOMS | rc.TERMINAL_ATOMS
        ):
            continue
        if ra == rb:
            if (na, nb) in rc.CLASH_EXEMPT_PAIRS[torsion.resnames[ra]]:
                continue
        elif partner[ra] == rb and na in ("CB", "SG") and nb in ("CB", "SG"):
            continue
        bound = rc.vdw_radius(na) + rc.vdw_radius(nb) - tolerance
        offset = coords[ra, ia] - coords[rb, ib]
        distance = float(torch.sqrt((offset * offset).sum() + _EPS))
        if distance < bound:
            total += bound - distance
    return total / int(torsion.atom_mask.sum())


@pytest.mark.parametrize("tolerance", [1.5, 0.0, -0.75])
def test_the_fast_path_matches_a_brute_force_reference(ubiquitin, tolerance):
    torsion, native = ubiquitin
    steric = steric_topology(torsion, native)
    fast = float(
        clash_loss(native, torsion, steric, tolerance=tolerance, cutoff=1e9)
    )
    assert fast == pytest.approx(_reference(native, torsion, steric, tolerance), abs=1e-9)


def test_the_default_cutoff_loses_nothing(lysozyme):
    """``DEFAULT_CUTOFF`` is derived from the longest reachable contact rather
    than tuned, so dropping it must change nothing at all. Checked on a
    deliberately over-slack tolerance, where the bound is largest and the
    cutoff is therefore closest to being too small."""
    torsion, native = lysozyme
    steric = steric_topology(torsion, native)
    deltas = uniform_deltas(torsion.chi_mask, torch.Generator().manual_seed(3))
    wrecked = apply_chi_deltas(native, deltas, torsion)
    for tolerance in (1.5, 0.0):
        near = clash_loss(
            wrecked, torsion, steric, tolerance=tolerance, cutoff=DEFAULT_CUTOFF
        )
        far = clash_loss(wrecked, torsion, steric, tolerance=tolerance, cutoff=1e9)
        assert float(near) == pytest.approx(float(far), abs=1e-12)


# --- disulfides ---------------------------------------------------------------


def test_lysozyme_has_its_four_disulfides(lysozyme):
    """Hen egg-white lysozyme has exactly four, and the matching is found from
    the coordinates alone. A fifth or a third would mean the threshold is
    picking up non-bonded S-S contacts or missing a real bond."""
    torsion, native = lysozyme
    steric = steric_topology(torsion, native)
    linked = (steric.partner >= 0).nonzero(as_tuple=True)[0]
    assert linked.numel() == 8
    for r in linked.tolist():
        assert int(steric.partner[int(steric.partner[r])]) == r


def test_without_the_exemption_the_disulfides_would_be_scored_as_clashes(lysozyme):
    """An S-S bond is 2.04 A against a radius sum of 3.6, so it is 1.56 A
    inside the bound -- the largest single overlap in the structure. The
    exemption is not a refinement."""
    torsion, native = lysozyme
    steric = steric_topology(torsion, native)
    free = steric_topology(torsion, None)
    assert float(clash_loss(native, torsion, steric)) == 0.0
    assert float(clash_loss(native, torsion, free)) > 0.0


def test_cysteines_too_far_apart_are_not_bonded(lysozyme):
    """Passing no coordinates leaves every cysteine free, which is the right
    answer for a synthetic topology and the wrong one for a real protein."""
    torsion, _native = lysozyme
    assert int((steric_topology(torsion, None).partner >= 0).sum()) == 0


# --- gradients and masking ----------------------------------------------------


def test_the_gradient_is_finite_and_reaches_the_side_chains(ubiquitin):
    """Padded atoms sit at the origin, where a naive distance has an undefined
    derivative; this is the test that the epsilon and the flat bottom between
    them keep that out of the graph."""
    torsion, native = ubiquitin
    steric = steric_topology(torsion, native)
    deltas = uniform_deltas(torsion.chi_mask, torch.Generator().manual_seed(1))
    wrecked = apply_chi_deltas(native, deltas, torsion).clone().requires_grad_(True)
    clash_loss(wrecked, torsion, steric).backward()
    grad = wrecked.grad
    assert bool(torch.isfinite(grad).all())
    sidechain = (~steric.backbone) & torsion.atom_mask
    assert int((grad.abs().sum(-1)[sidechain] > 0).sum()) > 0
    assert float(grad[~torsion.atom_mask].abs().max()) == 0.0


def test_an_unsupervised_pair_is_not_scored(ubiquitin):
    """A crop's boundary shell is context. Shell-against-shell overlap belongs
    to whatever produced the shell, and scoring it would train the rule on
    residues whose native torsions were deliberately not supervised."""
    torsion, native = ubiquitin
    steric = steric_topology(torsion, native)
    deltas = uniform_deltas(torsion.chi_mask, torch.Generator().manual_seed(2))
    wrecked = apply_chi_deltas(native, deltas, torsion)
    everything = torch.ones(torsion.n_residues, dtype=torch.bool)
    half = everything.clone()
    half[torsion.n_residues // 2 :] = False
    nothing = torch.zeros_like(everything)
    full = float(clash_loss(wrecked, torsion, steric, supervised=everything))
    part = float(clash_loss(wrecked, torsion, steric, supervised=half))
    assert 0.0 < part
    assert float(clash_loss(wrecked, torsion, steric, supervised=nothing)) == 0.0
    # Not simply half: the normaliser is the supervised atom count, and a pair
    # with one supervised end still counts. Only that it is a real restriction.
    assert part != full


def test_a_wrong_coordinate_shape_is_refused(ubiquitin):
    torsion, native = ubiquitin
    steric = steric_topology(torsion, native)
    with pytest.raises(ValueError, match="coords must be"):
        clash_loss(native[:, :4], torsion, steric)


# --- the objective ------------------------------------------------------------


def test_the_term_is_pinned_against_a_resume():
    """Changing the objective mid-run would continue one curve under a
    different problem. A *warm start* is expected to change it, which is why it
    is absent from ``train.WARM_START_MUST_MATCH``."""
    from aanca import checkpoint as ckpt

    assert "clash_weight" in ckpt.PINNED
    assert "clash_tolerance" in ckpt.PINNED
    with pytest.raises(ValueError, match="clash_weight"):
        ckpt.check_fingerprint(
            {"clash_weight": 0.0}, {"clash_weight": 1.0}, {"clash_weight": 0.0}
        )


def test_an_old_checkpoint_without_the_keys_still_resumes_at_the_default():
    from aanca import checkpoint as ckpt

    ckpt.check_fingerprint(
        {}, {"clash_weight": 0.0, "clash_tolerance": 1.5},
        {"clash_weight": 0.0, "clash_tolerance": 1.5},
    )
    with pytest.raises(ValueError):
        ckpt.check_fingerprint(
            {}, {"clash_weight": 0.5}, {"clash_weight": 0.0}
        )


def test_weight_zero_leaves_the_objective_exactly_as_it_was(ubiquitin_topology):
    """The term is skipped, not multiplied by zero, so a v6 run is reproducible
    and nothing is spent building a pair list that cannot change the loss."""
    import random

    from aanca.model import AANCARule
    from aanca.rollout import RolloutConfig, training_objective

    topo, native = ubiquitin_topology
    torsion = topo.torsion
    deltas = uniform_deltas(torsion.chi_mask, torch.Generator().manual_seed(6))
    damaged = apply_chi_deltas(native, deltas, torsion)
    # Small on purpose: this is about which terms are in ``total``, not about
    # the rule. ``training_objective`` builds the graph itself each step, so
    # there is nothing to hand it beyond the feature topology.
    rule = AANCARule(hidden_dim=8, message_dim=8)
    config = RolloutConfig(steps=2, truncate=2, k=16, loss_window=1.0)

    terms, _result = training_objective(
        rule, damaged, native, topo, config,
        generator=torch.Generator().manual_seed(0),
        rng=random.Random(0), noop_weight=0.0,
    )
    assert float(terms.clash) == 0.0
    # ``total`` is the only term still attached to the graph, hence the detach.
    assert float(terms.total.detach()) == pytest.approx(
        float(terms.reconstruction) + float(terms.restraint), abs=1e-12
    )
