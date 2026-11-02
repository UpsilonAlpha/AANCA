"""The (type, well) library and its selector: is the enumeration a bijection,
and is the gradient the one the module claims to pass?

Four of these carry the file; the rest are guard rails.

``test_every_residue_of_a_crystal_structure_recovers_its_own_resname`` is the
keystone. The library is an index space that the model's logits, its losses and
eventually its atom sets all agree on by position alone, so an off-by-one
anywhere in the enumeration -- a chi-less type given three slots, a type given
one, the wells enumerated in the wrong order -- is silent everywhere except
here, where ``resname -> index -> resname`` on real residues stops being the
identity. Two structures, because an error that happens to miss every type
present in one is unlikely to miss both.

``test_straight_through_gradient_equals_the_plain_softmax_gradient`` is the
only test that shows the straight-through estimator is what its name says. The
forward value is a one-hot, whose own derivative is zero almost everywhere, so
the entire point of the construction is the substituted backward value; an
implementation that detached the wrong side would produce a correct-looking
one-hot and a zero (or wrong) gradient, and nothing in training would say so.
It is asserted against an autograd gradient of the plain softmax rather than
eyeballed.

``test_type_marginal_loss_is_never_larger_than_the_joint_loss`` asserts the
property that makes the two losses worth reporting separately: summing the
correct type's wells can only add probability, so the type-marginal
cross-entropy is bounded above by the (type, well) one. It is a property of
*any* logits, so it is checked on random ones, and a bug in the scatter
(wrong ``TYPE_INDEX``, summing the wrong axis) breaks it.

``test_gumbel_concentrates_at_low_temperature_and_spreads_at_high`` is what
makes the temperature mean something. The two limits are the two failure modes
-- a zero-variance one-hot with no gradient, and a uniform sample carrying no
information about the logits -- and a Gumbel-softmax that ignored its
temperature argument entirely would pass every other test here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca import identity as idn
from aanca import residue_constants as rc
from aanca import rotamers
from aanca.data import RepackingDataset, StructureSpec
from aanca.torch_geometry import measure_chis

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


def _load(pdb_id: str):
    """``(resnames, chi1_degrees)`` for a crystal structure, measured the way
    training measures it: through the dataset's crop loader and
    ``measure_chis``, so a convention mismatch between the library and the
    geometry code would show up here rather than in a hand-built fixture."""
    path = DATA / f"{pdb_id}.pdb"
    if not path.exists():
        pytest.skip("run scripts/fetch_test_structures.py")
    specs = [StructureSpec(pdb_id=pdb_id, path=path, group=pdb_id)]
    dataset = RepackingDataset(specs, concentrations=(0.0,))
    topo, native, _indices, _seed = dataset.load_crop(0, seed_residue=0)
    chis = measure_chis(native, topo.torsion)
    return topo.torsion.resnames, torch.rad2deg(chis[:, 0])


@pytest.fixture(scope="module")
def ubiquitin():
    return _load("1UBQ")


@pytest.fixture(scope="module")
def lysozyme():
    return _load("193L")


# --- the enumeration ---------------------------------------------------------


def test_the_library_is_a_bijection_with_no_duplicates_or_gaps():
    """Every entry maps to an index and back unchanged, and every index in
    ``range(LIBRARY_SIZE)`` is reachable. Together these are what let the rest
    of the codebase treat a library index as a complete, dense description of a
    side chain."""
    assert len(set(idn.LIBRARY)) == idn.LIBRARY_SIZE
    for i, (resname, well) in enumerate(idn.LIBRARY):
        assert idn.index_of(resname, well) == i
        assert idn.entry(i) == (resname, well)
    assert sorted(idn.LIBRARY_INDEX.values()) == list(range(idn.LIBRARY_SIZE))


def test_the_chi_less_types_get_exactly_one_entry_each():
    """GLY and ALA have no chi1, and ``residue_constants`` finds PRO's CA-CB
    bond to lie in a ring and treats it as rigid too. Each is one type with a
    single trivial conformation: giving it three slots would waste two indices
    the head could never populate, and dropping it would shift every index
    above it, which is the failure the round-trip tests exist to catch."""
    chi_less = [name for name in rc.AA3 if rc.num_chi(name) == 0]
    assert set(chi_less) == {"ALA", "GLY", "PRO"}
    for name in chi_less:
        assert idn.WELLS_PER_TYPE[name] == 1
        assert [e for e in idn.LIBRARY if e[0] == name] == [(name, 0)]


def test_every_type_appears_and_the_size_decomposes():
    """All 20 standard residues are present, and the size is the sum of the
    per-type well counts rather than a hardcoded number."""
    assert {name for name, _ in idn.LIBRARY} == set(rc.AA3)
    assert idn.LIBRARY_SIZE == sum(idn.WELLS_PER_TYPE.values())
    assert idn.LIBRARY_SIZE == 17 * len(rotamers.WELL_NAMES) + 3


def test_type_index_and_well_chi1_line_up_with_the_entries():
    """The two ``(LIBRARY_SIZE,)`` tensors the head and the regression half read
    are derived from the same enumeration, so they must agree with it
    element-wise -- a scatter-add type marginal is only correct if
    ``TYPE_INDEX`` is."""
    for i, (resname, well) in enumerate(idn.LIBRARY):
        assert rc.AA3[int(idn.TYPE_INDEX[i])] == resname
        expected = (
            rotamers.WELLS[rotamers.WELL_NAMES[well]] if rc.num_chi(resname) else 0.0
        )
        assert float(idn.WELL_CHI1_DEGREES[i]) == pytest.approx(expected)


def test_library_index_agrees_with_nearest_well_on_synthetic_chi1():
    """``library_index`` must assign the well ``rotamers.nearest_well`` assigns
    and no other, since the chi1 diagnostics and the identity head would
    otherwise be reporting different well assignments for the same structure."""
    names = tuple(rc.AA3)
    chi1 = torch.tensor([-55.0, 175.0, 65.0, -170.0] * 5)
    indices = idn.library_index(names, chi1)
    wells, _ = rotamers.nearest_well(chi1)
    for i, name in enumerate(names):
        expected_well = int(wells[i]) if rc.num_chi(name) else 0
        assert idn.entry(int(indices[i])) == (name, expected_well)


def test_a_chi_less_residue_ignores_whatever_chi1_it_is_handed():
    """``measure_chis`` writes zero into masked entries, so a GLY arrives with
    a chi1 of 0 degrees, which is nearest to no well in particular. That number
    must not select anything."""
    for junk in (0.0, 180.0, -60.0, 60.0, 1e3):
        got = idn.library_index(("GLY", "PRO"), torch.tensor([junk, junk]))
        assert idn.index_resnames(got) == ("GLY", "PRO")
        assert [idn.entry(int(i))[1] for i in got] == [0, 0]


def test_library_index_rejects_a_mismatched_chi1_length():
    """A silent broadcast here would pair residues with other residues' angles,
    which is the kind of error that produces a plausible loss curve and a wrong
    model."""
    with pytest.raises(ValueError):
        idn.library_index(("LEU", "VAL"), torch.tensor([60.0]))


# --- real structures ---------------------------------------------------------


@pytest.mark.parametrize("structure", ["ubiquitin", "lysozyme"])
def test_every_residue_of_a_crystal_structure_recovers_its_own_resname(
    structure, request
):
    """The test that the enumeration is not off by one somewhere. Every residue
    of a real structure gets a valid index, and the resname read back out of
    that index is the resname that went in."""
    resnames, chi1 = request.getfixturevalue(structure)
    indices = idn.library_index(resnames, chi1)
    assert indices.shape == (len(resnames),)
    assert bool(((indices >= 0) & (indices < idn.LIBRARY_SIZE)).all())
    assert idn.index_resnames(indices) == tuple(resnames)


@pytest.mark.parametrize("structure", ["ubiquitin", "lysozyme"])
def test_a_crystal_structure_lands_near_the_well_it_is_assigned(structure, request):
    """Not a property of the library but of the claim the library rests on: a
    three-well chi1 description is only worth having if real chi1 angles sit
    near those wells. The wells are 120 degrees apart, so the worst possible
    distance is 60; a mean far above the ~15 degrees crystal structures give
    would mean the well centres are wrong, not that the structures are."""
    resnames, chi1 = request.getfixturevalue(structure)
    indices = idn.library_index(resnames, chi1)
    has_chi1 = torch.tensor([rc.num_chi(n) > 0 for n in resnames])
    centres = idn.index_chi1_degrees(indices)
    offset = (
        rotamers.nearest_well(chi1)[1].abs()[has_chi1]
    )
    assert float(offset.max()) <= 60.0
    assert float(offset.mean()) < 25.0
    # And the central chi1 handed back is the centre of the well just measured.
    wells, _ = rotamers.nearest_well(chi1)
    expected = torch.tensor(
        [rotamers.WELLS[rotamers.WELL_NAMES[int(w)]] for w in wells]
    )
    assert torch.allclose(centres[has_chi1], expected[has_chi1])


# --- the discretisation ------------------------------------------------------


def test_straight_through_is_exactly_one_hot():
    """Exactly, not approximately: the forward value is what broadcasts a
    steric field, and a 0.97/0.03 side chain is the smeared field the whole
    discrete design is there to avoid."""
    logits = torch.randn(9, idn.LIBRARY_SIZE)
    out = idn.straight_through(logits)
    assert torch.equal(out.sum(dim=-1), torch.ones(9))
    assert bool(((out == 0.0) | (out == 1.0)).all())
    assert torch.equal(out.argmax(dim=-1), logits.argmax(dim=-1))


def test_straight_through_gradient_equals_the_plain_softmax_gradient():
    """Asserted against autograd on the softmax, not eyeballed. Contracted
    against a fixed random weight vector so the comparison is of a full
    Jacobian-vector product and not of one coordinate that might agree by
    accident."""
    torch.manual_seed(0)
    raw = torch.randn(6, idn.LIBRARY_SIZE)
    weights = torch.randn(6, idn.LIBRARY_SIZE)

    hard_logits = raw.clone().requires_grad_(True)
    (idn.straight_through(hard_logits) * weights).sum().backward()

    soft_logits = raw.clone().requires_grad_(True)
    (torch.softmax(soft_logits, dim=-1) * weights).sum().backward()

    assert torch.allclose(hard_logits.grad, soft_logits.grad, atol=1e-6)
    # And the gradient is not trivially zero, which would also "match" a
    # broken reference.
    assert float(hard_logits.grad.abs().max()) > 1e-3


def test_straight_through_sampling_is_reproducible_from_the_generator():
    """Two generators at the same seed must draw the same labels, and a
    different seed must not -- the sampled path is used inside a rollout, so an
    unseeded draw would make a reported convergence curve unrepeatable."""
    logits = torch.randn(64, idn.LIBRARY_SIZE)
    draw = lambda seed: idn.straight_through(
        logits, generator=torch.Generator().manual_seed(seed), sample=True
    ).argmax(dim=-1)
    assert torch.equal(draw(7), draw(7))
    assert not torch.equal(draw(7), draw(8))


def test_straight_through_soft_mode_is_the_plain_softmax():
    """The ablation path has to be the thing it is an ablation of."""
    logits = torch.randn(4, idn.LIBRARY_SIZE)
    assert torch.allclose(
        idn.straight_through(logits, hard=False), torch.softmax(logits, dim=-1)
    )


def test_gumbel_is_reproducible_from_the_generator():
    logits = torch.randn(32, idn.LIBRARY_SIZE)
    first = idn.gumbel_softmax(
        logits, 0.5, generator=torch.Generator().manual_seed(3)
    )
    again = idn.gumbel_softmax(
        logits, 0.5, generator=torch.Generator().manual_seed(3)
    )
    other = idn.gumbel_softmax(
        logits, 0.5, generator=torch.Generator().manual_seed(4)
    )
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_gumbel_concentrates_at_low_temperature_and_spreads_at_high():
    """What the temperature does, and what it provably does *not* do. It
    divides the perturbed logits, so ``argmax((logits + g) / T)`` does not
    depend on ``T`` at all: *which* entry wins is a draw from
    ``softmax(logits)`` at every temperature, and the temperature controls only
    how much of the returned vector's mass sits on that winner. So
    "concentrates at low temperature" has to be asserted on the mass, not on
    the identity of the argmax, and the argmax-invariance is asserted directly
    below as the thing it is.

    The thresholds. With 54 entries, a fully spread sample has every component
    at 1/54 = 0.0185 and entropy log 54 = 3.99. Cold is asserted above 0.99
    mass on one entry (a numerically one-hot vector, the forward value the
    discrete design needs); hot is asserted below 0.1 mass and above 0.9 of
    the uniform entropy, both of which a near-uniform draw clears by an order
    of magnitude while a sample at temperature 1 would fail. The gap of 8 on
    entry 11 keeps ~0.98 of the untempered softmax mass there, so the cold
    sample's winner is that entry nearly always and the concentration being
    measured is not an artefact of a flat distribution.
    """
    logits = torch.zeros(idn.LIBRARY_SIZE)
    logits[11] = 8.0
    batch = logits.expand(4096, idn.LIBRARY_SIZE)

    cold = idn.gumbel_softmax(
        batch, 0.1, generator=torch.Generator().manual_seed(1)
    )
    hot = idn.gumbel_softmax(
        batch, 50.0, generator=torch.Generator().manual_seed(1)
    )

    assert float(cold.max(dim=-1).values.mean()) > 0.99
    assert float(hot.max(dim=-1).values.mean()) < 0.1

    uniform_entropy = torch.tensor(float(idn.LIBRARY_SIZE)).log()
    hot_entropy = -(hot * hot.clamp_min(1e-30).log()).sum(dim=-1).mean()
    cold_entropy = -(cold * cold.clamp_min(1e-30).log()).sum(dim=-1).mean()
    assert float(hot_entropy) > 0.9 * float(uniform_entropy)
    assert float(cold_entropy) < 0.1

    # And the winner is a draw from the untempered softmax, at both ends.
    for sample in (cold, hot):
        assert float((sample.argmax(dim=-1) == 11).to(torch.float32).mean()) > 0.95


def test_the_gumbel_argmax_does_not_depend_on_the_temperature():
    """Recorded because it is a trap in the other direction: annealing the
    temperature does not sharpen *which* label the forward pass commits to, it
    only sharpens the gradient-carrying vector around a draw that was already
    discrete. Anyone reading a falling churn curve as an effect of annealing
    would be reading the wrong mechanism."""
    logits = torch.randn(256, idn.LIBRARY_SIZE)
    labels = [
        idn.gumbel_softmax(
            logits, t, generator=torch.Generator().manual_seed(11)
        ).argmax(dim=-1)
        for t in (0.1, 1.0, 10.0)
    ]
    assert torch.equal(labels[0], labels[1])
    assert torch.equal(labels[1], labels[2])


def test_gumbel_hard_is_one_hot_and_still_carries_a_gradient():
    """The forward pass needs the discrete value; training needs the soft
    sample's gradient to survive the discretisation."""
    logits = torch.randn(5, idn.LIBRARY_SIZE, requires_grad=True)
    out = idn.gumbel_softmax(
        logits, 0.7, generator=torch.Generator().manual_seed(2), hard=True
    )
    assert bool(((out == 0.0) | (out == 1.0)).all())
    assert torch.equal(out.sum(dim=-1), torch.ones(5))
    out.sum(dim=-1).mul(torch.randn(5)).sum().backward()
    assert float(logits.grad.abs().max()) > 0.0


def test_gumbel_rejects_a_non_positive_temperature():
    """Zero would divide by zero into NaNs that only surface as a dead loss
    several hundred steps later."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            idn.gumbel_softmax(torch.zeros(1, idn.LIBRARY_SIZE), bad)


# --- the losses --------------------------------------------------------------


def test_the_joint_loss_matches_cross_entropy_and_respects_the_mask():
    """Shaped and masked like ``losses.chi_nll``: masked-out entries read zero
    in the per-residue tensor and contribute nothing to the mean."""
    logits = torch.randn(8, idn.LIBRARY_SIZE)
    target = torch.randint(0, idn.LIBRARY_SIZE, (8,))
    mask = torch.tensor([True, True, False, True, False, True, True, False])

    per_residue = idn.identity_nll(logits, target, mask)
    reference = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    assert torch.allclose(per_residue[mask], reference[mask])
    assert bool((per_residue[~mask] == 0.0).all())
    assert torch.allclose(idn.identity_loss(logits, target, mask), reference[mask].mean())


def test_an_empty_mask_gives_zero_rather_than_a_nan():
    """``masked_mean``'s convention, which this follows so a crop with no
    designable positions does not poison the epoch's average."""
    logits = torch.randn(3, idn.LIBRARY_SIZE)
    target = torch.zeros(3, dtype=torch.long)
    empty = torch.zeros(3, dtype=torch.bool)
    assert float(idn.identity_loss(logits, target, empty)) == 0.0
    assert float(idn.type_loss(logits, target, empty)) == 0.0


def test_the_type_marginal_loss_is_the_log_of_the_summed_well_probabilities():
    """Computed in log space in the module; checked here against the
    straightforward sum of the correct type's well probabilities, which is the
    definition."""
    logits = torch.randn(10, idn.LIBRARY_SIZE)
    target = torch.randint(0, idn.LIBRARY_SIZE, (10,))
    mask = torch.ones(10, dtype=torch.bool)
    probabilities = torch.softmax(logits, dim=-1)
    marginal = idn.type_marginal(probabilities)
    assert torch.allclose(marginal.sum(dim=-1), torch.ones(10), atol=1e-6)
    wanted = marginal.gather(-1, idn.type_index(target).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(idn.type_nll(logits, target, mask), -wanted.log(), atol=1e-6)


def test_type_marginal_loss_is_never_larger_than_the_joint_loss():
    """A property, not a coincidence: summing over the correct type's wells can
    only add probability to it, so the type cross-entropy is bounded above by
    the (type, well) one. Asserted on random logits, and the strict inequality
    is asserted for a target whose type has more than one well, since equality
    there would mean the sum dropped the siblings."""
    torch.manual_seed(5)
    for _ in range(25):
        logits = torch.randn(16, idn.LIBRARY_SIZE) * 3.0
        target = torch.randint(0, idn.LIBRARY_SIZE, (16,))
        mask = torch.ones(16, dtype=torch.bool)
        assert bool(
            (idn.type_nll(logits, target, mask) <= idn.identity_nll(logits, target, mask)
             + 1e-6).all()
        )

    multi = torch.tensor([idn.index_of("LEU", 0)] * 4)
    logits = torch.randn(4, idn.LIBRARY_SIZE)
    mask = torch.ones(4, dtype=torch.bool)
    assert float(idn.type_loss(logits, multi, mask)) < float(
        idn.identity_loss(logits, multi, mask)
    )


def test_the_two_losses_agree_for_a_chi_less_type():
    """ALA, GLY and PRO have one well each, so their type marginal *is* their
    joint probability and the two numbers must coincide. This is the boundary
    case that would break if the scatter put a chi-less type's mass somewhere
    else."""
    for name in ("ALA", "GLY", "PRO"):
        target = torch.tensor([idn.index_of(name, 0)] * 6)
        logits = torch.randn(6, idn.LIBRARY_SIZE)
        mask = torch.ones(6, dtype=torch.bool)
        assert torch.allclose(
            idn.type_nll(logits, target, mask),
            idn.identity_nll(logits, target, mask),
            atol=1e-6,
        )


# --- the convergence diagnostic ----------------------------------------------


def test_label_churn_is_zero_for_identical_labels_and_one_when_all_differ():
    labels = torch.randint(0, idn.LIBRARY_SIZE, (20,))
    mask = torch.ones(20, dtype=torch.bool)
    assert float(idn.label_churn(labels, labels, mask)) == 0.0
    shifted = (labels + 1) % idn.LIBRARY_SIZE
    assert float(idn.label_churn(labels, shifted, mask)) == 1.0


def test_label_churn_counts_only_the_masked_positions():
    """The diagnostic is read per rollout step against the same mask the loss
    uses, so a flip at a frozen position must not register -- otherwise a
    rollout could appear to be in a limit cycle because of positions it is not
    even updating."""
    previous = torch.zeros(10, dtype=torch.long)
    current = previous.clone()
    current[0] = 1  # masked in, and changed
    current[5] = 2  # masked out, and changed
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:5] = True
    assert float(idn.label_churn(previous, current, mask)) == pytest.approx(0.2)

    unmasked_only = torch.zeros(10, dtype=torch.bool)
    unmasked_only[5:] = True
    frozen = previous.clone()
    frozen[0] = 1
    assert float(idn.label_churn(previous, frozen, unmasked_only)) == 0.0


def test_label_churn_is_zero_when_nothing_is_masked_in():
    """"No positions changed" is the right reading of "no positions", and it
    keeps the logged curve a number rather than a NaN on a fully frozen crop."""
    previous = torch.zeros(4, dtype=torch.long)
    current = torch.arange(4)
    assert float(idn.label_churn(previous, current, torch.zeros(4, dtype=torch.bool))) == 0.0
