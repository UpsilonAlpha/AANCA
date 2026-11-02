"""The update schedule: is the set it hands the rollout actually independent?

``aanca.update_schedule`` exists to make one claim true -- that no two residues whose
side chains could touch ever tick from the same stale neighbourhood -- and
everything else in the module is bookkeeping around it. So the test that
matters is ``test_every_set_is_independent``: it re-derives independence from
the coupling matrix directly, on both structures, across radii and seeds. If
that fails the schedule is not chromatic Gibbs, it is Bernoulli with extra
steps, and the oscillation it was built to prevent is back.

Two others carry nearly as much weight for different reasons.

``test_bernoulli_is_bit_identical_to_not_setting_the_field`` is the
compatibility guarantee. ``update_schedule`` is a new field on
``RolloutConfig`` with a default, and the whole point of that default is that
v1 through v6 still reproduce -- not approximately, and not just in the
coordinates but in the random stream, since a schedule that drew one extra
number would silently reseat every later draw in training.

``test_the_coupling_graph_agrees_with_the_message_graph_at_the_same_radius``
pins the two notions of contact together. ``coupling_graph`` and
``torch_graph``'s neighbour selection both mean "minimum heavy-atom distance",
and the coupling radius is deliberately *smaller* than the perception cutoff
rather than measured differently. Run both at 10 A and they must agree edge for
edge; if they ever stop agreeing, one of them changed its distance measure and
the argument in the module docstring no longer holds.

The one thing no test here asserts is that the two schedules move a *fresh*
rule differently. A newly constructed ``AANCARule`` is a no-op -- its output
layer is zero-initialised -- so both schedules return the input coordinates
bit for bit, and a test phrased that way would pass or fail for reasons having
nothing to do with the schedule. ``test_the_two_schedules_really_diverge``
perturbs the weights first, which is the only state in which the question has
an answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.data import RepackingDataset, StructureSpec
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.update_schedule import (
    COUPLING_RADIUS,
    coupling_graph,
    is_independent,
    random_colouring,
    random_independent_set,
)
from aanca.torch_graph import build_graph, sequence_links

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"

#: Radii the structural tests sweep. 0.1 A is below any real contact, so the
#: graph there is the forced sequence links and nothing else -- which is the
#: only radius at which the forcing is observable on its own.
RADII = (0.1, 2.0, 5.0, 8.0, 10.0)


def _load(pdb_id: str):
    path = DATA / f"{pdb_id}.pdb"
    if not path.exists():
        pytest.skip("run scripts/fetch_test_structures.py")
    specs = [StructureSpec(pdb_id=pdb_id, path=path, group=pdb_id)]
    dataset = RepackingDataset(specs, concentrations=(0.0,))
    topo, native, _indices, _seed = dataset.load_crop(0, seed_residue=0)
    return topo, native


@pytest.fixture(scope="module")
def ubiquitin():
    return _load("1UBQ")


@pytest.fixture(scope="module")
def lysozyme():
    return _load("193L")


def _mean_degree(coupled) -> float:
    return float(coupled.sum(1).to(torch.float64).mean())


def _perturbed_rule(hidden_dim: int = 16, message_dim: int = 16) -> AANCARule:
    """A rule whose output layer is not zero, so the rollout actually moves.

    At initialisation the chi head is zeroed on purpose -- the rule starts as
    the identity -- which makes every schedule indistinguishable. Adding noise
    to every parameter is not training, but it is enough to make the
    coordinates a function of which residues ticked, which is what the
    schedule tests need to observe.
    """
    torch.manual_seed(1)
    rule = AANCARule(hidden_dim=hidden_dim, message_dim=message_dim)
    with torch.no_grad():
        for parameter in rule.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)
    return rule


# --- the headline: the sets are independent ----------------------------------


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
@pytest.mark.parametrize("radius", RADII)
def test_every_set_is_independent(fixture, radius, request):
    """No two chosen residues are coupled, checked against the matrix itself.

    This is the correctness property of the whole module; everything else is
    secondary. Re-derived here rather than trusting ``is_independent``, which
    shares its indexing with nothing but is still part of the module under
    test.
    """
    topo, native = request.getfixturevalue(fixture)
    coupled = coupling_graph(native, topo, radius)
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        chosen = random_independent_set(coupled, generator)
        assert chosen.dtype == torch.bool
        assert chosen.shape == (topo.n_residues,)
        assert bool(chosen.any()), "an empty set would tick nothing"
        selected = chosen.nonzero(as_tuple=True)[0]
        assert not bool(coupled[selected][:, selected].any())
        assert is_independent(chosen, coupled)


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
def test_the_sets_are_maximal(fixture, request):
    """Every unchosen residue is blocked by a chosen one, so nothing could be
    added. Independence alone is satisfied by the empty set and by a single
    residue; maximality is what makes one set a full colour class and so what
    bounds a sweep at the number of colours rather than at ``n_residues``."""
    topo, native = request.getfixturevalue(fixture)
    coupled = coupling_graph(native, topo, COUPLING_RADIUS)
    generator = torch.Generator().manual_seed(4)
    for _ in range(3):
        chosen = random_independent_set(coupled, generator)
        for vertex in (~chosen).nonzero(as_tuple=True)[0].tolist():
            assert bool(coupled[vertex][chosen].any()), (
                f"residue {vertex} could have been added"
            )


def test_the_sets_are_reproducible_from_the_seed_and_vary_between_seeds(ubiquitin):
    """The schedule has to be a property of the seed, like the damage and the
    Bernoulli mask -- otherwise a rerun of a run is a different experiment. It
    also has to actually move: a fresh set per step is the reason there is no
    fixed partition for a cell to learn as a phase."""
    topo, native = ubiquitin
    coupled = coupling_graph(native, topo, COUPLING_RADIUS)
    a = random_independent_set(coupled, torch.Generator().manual_seed(11))
    b = random_independent_set(coupled, torch.Generator().manual_seed(11))
    c = random_independent_set(coupled, torch.Generator().manual_seed(12))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)

    # Consecutive draws from one generator differ too, which is the property
    # the rollout relies on -- it passes the same generator every step.
    generator = torch.Generator().manual_seed(0)
    sets = [random_independent_set(coupled, generator) for _ in range(3)]
    assert [int(s.sum()) for s in sets] == [13, 15, 14]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(sets[i], sets[j])


# --- the colouring, which is what prices a full sweep ------------------------


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
def test_the_colouring_is_proper_and_total(fixture, request):
    topo, native = request.getfixturevalue(fixture)
    coupled = coupling_graph(native, topo, COUPLING_RADIUS)
    generator = torch.Generator().manual_seed(0)
    for _ in range(4):
        colour = random_colouring(coupled, generator)
        assert colour.shape == (topo.n_residues,)
        assert int(colour.min()) >= 0, "every residue must be coloured"
        rows, cols = coupled.nonzero(as_tuple=True)
        assert not bool((colour[rows] == colour[cols]).any())


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
def test_a_sweep_costs_a_handful_of_colours_not_a_residue_each(fixture, request):
    """The number that decides whether an exact schedule is affordable at all.

    If the coupling graph needed one colour per residue -- which is what the
    conservative "could ever be within the perception cutoff" superset gives --
    a sweep would be 76 or 129 steps and the schedule would be unusable.
    Measured at 8-10 on both structures at 5 A, against 38-64 residues ticking
    per Bernoulli step. Fixed seeds, so this is a measurement and not a
    sampling hope.
    """
    topo, native = request.getfixturevalue(fixture)
    coupled = coupling_graph(native, topo, COUPLING_RADIUS)
    counts = [
        int(random_colouring(coupled, torch.Generator().manual_seed(seed)).max()) + 1
        for seed in range(4)
    ]
    assert all(8 <= count <= 11 for count in counts), counts
    assert max(counts) < topo.n_residues // 4


# --- the coupling graph itself -----------------------------------------------


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
@pytest.mark.parametrize("radius", RADII)
def test_the_graph_is_symmetric_with_no_self_edges(fixture, radius, request):
    """A residue coupled to itself would be excluded by its own selection and
    the sets would come back empty; an asymmetric edge would let one of the two
    residues tick while the other thought it was blocked."""
    topo, native = request.getfixturevalue(fixture)
    coupled = coupling_graph(native, topo, radius)
    assert coupled.shape == (topo.n_residues, topo.n_residues)
    assert coupled.dtype == torch.bool
    assert bool((coupled == coupled.T).all())
    assert int(coupled.diagonal().sum()) == 0


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
def test_sequence_neighbours_are_coupled_at_any_radius(fixture, request):
    """Bonded neighbours share backbone atoms, so the distance test is beside
    the point for them -- they are coupled by chemistry. At 0.1 A nothing else
    is in contact, so the graph collapses to exactly the sequence links, which
    is the only radius where the forcing is visible on its own."""
    topo, native = request.getfixturevalue(fixture)
    links = sequence_links(topo, native.device)
    tiny = coupling_graph(native, topo, 0.1)
    index = torch.arange(topo.n_residues - 1)
    assert bool(tiny[index, index + 1][links].all())
    assert int(tiny.sum()) == 2 * int(links.sum())
    # And still forced where the distance test would also have found them.
    for radius in RADII:
        coupled = coupling_graph(native, topo, radius)
        assert bool(coupled[index, index + 1][links].all())


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
def test_the_coupling_graph_agrees_with_the_message_graph_at_the_same_radius(
    fixture, request
):
    """Both mean minimum heavy-atom distance, so at 10 A they must be the same
    graph -- edge for edge, not merely similar in size.

    The coupling radius is smaller than the perception cutoff because contact
    and perception are different questions, not because the distance is
    measured differently. This is the test that keeps that distinction honest:
    if ``torch_graph`` ever switched to CA-CA, or ``coupling_graph`` ever
    measured something else, the two counts would part company here while
    every other test in this file still passed.
    """
    topo, native = request.getfixturevalue(fixture)
    coupled = coupling_graph(native, topo, 10.0)
    graph = build_graph(native.to(torch.float32), topo, k=64, cutoff=10.0)
    assert int(coupled.sum()) == int(graph.edge_mask.sum())


@pytest.mark.parametrize("fixture", ["ubiquitin", "lysozyme"])
def test_wider_coupling_means_more_edges_and_smaller_sets(fixture, request):
    """Monotone in the radius, which is the sanity check on the knob itself.

    A radius that did not tighten the sets would mean the distance was not
    being read, and the 5 A choice would be arbitrary rather than a trade. The
    size is averaged over seeds because a single greedy draw is noisy by a
    residue or two, while the degree is deterministic.
    """
    topo, native = request.getfixturevalue(fixture)
    degrees, sizes = [], []
    for radius in (2.0, 5.0, 8.0, 10.0):
        coupled = coupling_graph(native, topo, radius)
        generator = torch.Generator().manual_seed(0)
        draws = [int(random_independent_set(coupled, generator).sum()) for _ in range(8)]
        degrees.append(_mean_degree(coupled))
        sizes.append(sum(draws) / len(draws))
    assert degrees == sorted(degrees) and len(set(degrees)) == len(degrees)
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)
    # The measured anchors: contact plus a margin keeps about a sixth of the
    # protein ticking, the perception cutoff would keep a twentieth.
    assert 0.12 < sizes[1] / topo.n_residues < 0.22
    assert sizes[3] / topo.n_residues < 0.10


# --- the rollout ------------------------------------------------------------


def _config(steps: int = 4, **kwargs) -> RolloutConfig:
    return RolloutConfig(steps=steps, truncate=2, k=24, cutoff=10.0, **kwargs)


def test_bernoulli_is_bit_identical_to_not_setting_the_field(ubiquitin):
    """A v6 run must reproduce exactly, in the coordinates *and* in the stream.

    Comparing the generator state afterwards is the half that matters: a
    schedule branch that drew one spare number would leave the coordinates
    identical for this rollout and reseat every later draw in training, which
    is the kind of break that shows up as an unexplained curve rather than as
    a failure. Run on a perturbed rule because a fresh one is the identity and
    would compare equal whatever the schedule did.
    """
    topo, native = ubiquitin
    rule = _perturbed_rule()

    def run(config):
        generator = torch.Generator().manual_seed(17)
        result = rollout(rule, initial_state(native, rule), topo, config, generator)
        return result.state.coords.detach(), generator.get_state()

    silent_coords, silent_stream = run(_config())
    named_coords, named_stream = run(_config(update_schedule="bernoulli"))
    assert torch.equal(silent_coords, named_coords)
    assert torch.equal(silent_stream, named_stream)
    assert float((silent_coords - native).abs().max()) > 1e-3, "nothing moved"


def test_the_two_schedules_really_diverge(ubiquitin):
    """With the weights perturbed the schedule is observable, and the two
    disagree. The same comparison on a freshly constructed rule is vacuous --
    its chi head is zeroed, so both schedules return the input coordinates bit
    for bit and the test would pass without testing anything."""
    topo, native = ubiquitin
    rule = _perturbed_rule()

    def run(schedule):
        generator = torch.Generator().manual_seed(17)
        result = rollout(
            rule,
            initial_state(native, rule),
            topo,
            _config(update_schedule=schedule),
            generator,
        )
        return result.state.coords.detach()

    bernoulli = run("bernoulli")
    independent = run("independent")
    assert not torch.equal(bernoulli, independent)
    assert float((independent - native).abs().max()) > 1e-3
    assert len(
        rollout(
            rule,
            initial_state(native, rule),
            topo,
            _config(update_schedule="independent"),
            torch.Generator().manual_seed(2),
        ).mean_abs_delta
    ) == 4


def test_the_independent_schedule_ticks_every_step_within_the_coupling_graph(
    ubiquitin,
):
    """The rollout redraws the graph from the *current* coordinates each step,
    which is the whole reason a fixed colouring was rejected. Checked by
    confirming the set the rollout would draw at the state it reaches is still
    independent there -- a set carried over from step 0 would not be."""
    topo, native = ubiquitin
    rule = _perturbed_rule()
    result = rollout(
        rule,
        initial_state(native, rule),
        topo,
        _config(steps=6, update_schedule="independent"),
        torch.Generator().manual_seed(5),
    )
    moved = result.state.coords.detach()
    assert not torch.equal(moved, native)
    coupled = coupling_graph(moved, topo, COUPLING_RADIUS)
    generator = torch.Generator().manual_seed(9)
    for _ in range(3):
        assert is_independent(random_independent_set(coupled, generator), coupled)


def test_an_unknown_schedule_is_refused(ubiquitin):
    """Silently falling back to Bernoulli would report one experiment and run
    another, which is worse than not running."""
    topo, native = ubiquitin
    rule = AANCARule(hidden_dim=8, message_dim=8)
    with pytest.raises(ValueError, match="unknown update_schedule"):
        rollout(
            rule,
            initial_state(native, rule),
            topo,
            _config(update_schedule="chromatic"),
            torch.Generator().manual_seed(0),
        )


# --- the resume guard -------------------------------------------------------


def test_the_schedule_is_pinned_against_a_resume():
    """Changing which residues tick mid-run would continue one curve under
    different dynamics, so both keys are pinned. A *warm start* is allowed to
    change them -- that is the experiment -- which is why they are absent from
    ``train.WARM_START_MUST_MATCH``."""
    from aanca import checkpoint as ckpt

    assert "update_schedule" in ckpt.PINNED
    assert "coupling_radius" in ckpt.PINNED
    with pytest.raises(ValueError, match="update_schedule"):
        ckpt.check_fingerprint(
            {"update_schedule": "bernoulli"},
            {"update_schedule": "independent"},
            {"update_schedule": "bernoulli"},
        )
    with pytest.raises(ValueError, match="coupling_radius"):
        ckpt.check_fingerprint(
            {"coupling_radius": 5.0},
            {"coupling_radius": 10.0},
            {"coupling_radius": 5.0},
        )


def test_an_old_checkpoint_without_the_keys_still_resumes_at_the_default():
    """Pinning a new setting must not strand a run already chaining through its
    segments: a checkpoint written before the schedule existed was a Bernoulli
    run at the default radius, and resuming it at the defaults is correct."""
    from aanca import checkpoint as ckpt

    defaults = {"update_schedule": "bernoulli", "coupling_radius": 5.0}
    ckpt.check_fingerprint({}, dict(defaults), dict(defaults))
    with pytest.raises(ValueError):
        ckpt.check_fingerprint({}, {"update_schedule": "independent"}, defaults)
    with pytest.raises(ValueError):
        ckpt.check_fingerprint({}, {"coupling_radius": 6.0}, defaults)
