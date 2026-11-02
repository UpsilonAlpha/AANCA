"""The Potts energy and its samplers: is the fast path exact, and is the chain
actually sampling the distribution it claims?

Three of these carry the file; the rest are guard rails.

``test_delta_energy_equals_the_difference_of_two_full_energies`` is the
keystone. :func:`aanca.potts.delta_energy` is an optimisation -- it is the
reason a Potts model was chosen over a learned scalar energy at all -- and the
only thing that makes an optimisation legitimate is that it returns the number
the slow path returns. Every bug the incident-edge arithmetic can have (an
untransposed block for an edge where the moved position is the second
endpoint, a neighbour read at its post-move value, a dropped self-field)
changes ``delta_energy`` and not ``energy``.

``test_gibbs_reproduces_the_exact_boltzmann_distribution`` and
``test_metropolis_reproduces_it_from_a_biased_proposal`` are the only tests
here that show the samplers are *correct* rather than merely plausible. The
system is 4 positions over 3 types, so all 81 states enumerate and the exact
``p(s) ~ exp(-E(s)/kT)`` is available in closed form to compare a long chain
against. The Metropolis one uses a deliberately skewed proposal: recovering the
right stationary distribution from a wrong proposal is exactly what the
Hastings ``q`` ratio is for, and it is the only way to tell a real MH
implementation from Gibbs with extra arithmetic -- a sampler that ignored ``q``
would pass the Gibbs test and fail this one.

``test_annealing_to_zero_finds_the_brute_force_global_minimum`` checks the
other operating point, since the design uses the same machinery for both:
finite temperature for free energies out of the ensemble, and a quench for a
single answer.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from aanca.potts import (
    AnnealResult,
    PottsModel,
    anneal,
    conditional,
    delta_energy,
    edges_from_adjacency,
    energy,
    gibbs_sweep,
    metropolis_ratio,
    metropolis_sweep,
)

#: Small enough that every state enumerates (3**4 = 81), big enough that the
#: five edges below make a loopy graph rather than a chain -- a tree would be
#: exactly solvable by dynamic programming and would not exercise the
#: sequential sweep the way a frustrated loop does.
N_RES = 4
N_TYPES = 3

#: Deliberately not the complete graph: positions 1 and 3 are not coupled, so
#: ``test_delta_energy_only_reads_incident_edges`` has a non-incident edge to
#: perturb.
EDGES = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]], dtype=torch.long)


def small_model(seed: int = 1) -> PottsModel:
    """The enumerable system. The default seed is fixed at 1 because the
    annealing test needs a well-separated global minimum; the sweep that chose
    it is in that test's docstring."""
    generator = torch.Generator().manual_seed(seed)
    return PottsModel(
        h=torch.randn(N_RES, N_TYPES, generator=generator),
        edges=EDGES.clone(),
        J=torch.randn(EDGES.shape[0], N_TYPES, N_TYPES, generator=generator),
    )


def random_model(
    n_res: int, n_types: int, density: float, seed: int
) -> PottsModel:
    """A larger model on a random contact graph, built through
    :func:`edges_from_adjacency` so the construction path the callers will use
    is the one under test."""
    generator = torch.Generator().manual_seed(seed)
    adjacency = torch.rand(n_res, n_res, generator=generator) < density
    edges = edges_from_adjacency(adjacency)
    return PottsModel(
        h=torch.randn(n_res, n_types, generator=generator),
        edges=edges,
        J=torch.randn(edges.shape[0], n_types, n_types, generator=generator),
    )


def all_states() -> torch.Tensor:
    """``(81, 4)``: every sequence of the small system."""
    return torch.tensor(
        list(itertools.product(range(N_TYPES), repeat=N_RES)), dtype=torch.long
    )


def exact_boltzmann(model: PottsModel, temperature: float) -> torch.Tensor:
    """``(81,)`` normalised ``p(s) ~ exp(-E(s)/kT)`` by brute-force
    enumeration -- the partition function summed, not estimated."""
    energies = energy(model, all_states())
    weights = torch.exp(-(energies - energies.min()) / temperature)
    return weights / weights.sum()


def state_index() -> dict[tuple[int, ...], int]:
    return {tuple(int(v) for v in s): k for k, s in enumerate(all_states())}


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * float((p - q).abs().sum())


def chain_histogram(
    model: PottsModel,
    sweeps: int,
    burn_in: int,
    step,
    seed: int,
) -> torch.Tensor:
    """Run ``step(sequence, generator)`` for ``burn_in + sweeps`` sweeps and
    return the empirical distribution over all 81 states.

    One state is recorded per sweep rather than per single-site update, so
    successive samples are a full sweep apart and the autocorrelation that
    inflates the measured TV distance is as small as it cheaply can be.
    """
    generator = torch.Generator().manual_seed(seed)
    sequence = torch.zeros(N_RES, dtype=torch.long)
    index = state_index()
    counts = torch.zeros(len(index))
    for sweep in range(burn_in + sweeps):
        sequence, _changed = step(sequence, generator)
        if sweep >= burn_in:
            counts[index[tuple(int(v) for v in sequence)]] += 1
    return counts / counts.sum()


# --- the keystone: delta_energy is exact -------------------------------------


def test_delta_energy_equals_the_difference_of_two_full_energies():
    """The whole point of the sparse edge list and the incidence tables is that
    a one-position move costs O(degree). This asserts the cheap number is the
    true number, over 400 random (sequence, position, type) triples on a
    30-residue model with 20 types -- enough positions of differing degree,
    including the first and last endpoints of edges, that a transposition bug
    cannot hide."""
    model = random_model(n_res=30, n_types=20, density=0.12, seed=3)
    generator = torch.Generator().manual_seed(11)
    worst = 0.0
    for _ in range(400):
        sequence = torch.randint(
            model.n_types, (model.n_res,), generator=generator, dtype=torch.long
        )
        position = int(torch.randint(model.n_res, (1,), generator=generator))
        new_type = int(torch.randint(model.n_types, (1,), generator=generator))
        mutant = sequence.clone()
        mutant[position] = new_type
        reference = energy(model, mutant) - energy(model, sequence)
        fast = delta_energy(model, sequence, position, new_type)
        worst = max(worst, abs(float(fast) - float(reference)))
    # Tolerance is float32 cancellation in the *reference*: the two total
    # energies are O(100) and their difference is O(1), so the slow path is the
    # imprecise one here.
    assert worst < 1e-4


def test_delta_energy_of_a_no_op_mutation_is_exactly_zero():
    """Not approximately: a Metropolis sweep re-proposes the current type
    often, and a residual of 1e-7 there would bias acceptance by a constant
    factor at low temperature."""
    model = random_model(n_res=12, n_types=20, density=0.3, seed=5)
    generator = torch.Generator().manual_seed(2)
    sequence = torch.randint(
        model.n_types, (model.n_res,), generator=generator, dtype=torch.long
    )
    for position in range(model.n_res):
        same = delta_energy(model, sequence, position, int(sequence[position]))
        assert float(same) == 0.0


def test_delta_energy_only_reads_incident_edges():
    """The locality claim, tested as an independence rather than as a cost.
    Position 1 touches edges (0,1) and (1,2) and not (2,3); changing the
    coupling on (2,3) must leave every delta at position 1 bit-identical. A
    ``delta_energy`` that fell back to two full energies would still be
    numerically right and would fail this."""
    model = small_model()
    sequence = torch.tensor([0, 1, 2, 1])
    assert set(int(v) for v in model.neighbours(1)) == {0, 2}

    before = [float(delta_energy(model, sequence, 1, t)) for t in range(N_TYPES)]
    far = int((model.edges == torch.tensor([2, 3])).all(dim=1).nonzero()[0])
    perturbed = model.J.clone()
    perturbed[far] += 100.0
    moved = PottsModel(h=model.h, edges=model.edges, J=perturbed)
    after = [float(delta_energy(moved, sequence, 1, t)) for t in range(N_TYPES)]
    assert before == after
    # And the control: the same perturbation does move a delta at position 2,
    # which does touch (2,3) -- otherwise the assertion above would pass for a
    # model where J simply had no effect anywhere.
    assert float(delta_energy(model, sequence, 2, 0)) != float(
        delta_energy(moved, sequence, 2, 0)
    )


# --- the samplers against the exact distribution -----------------------------


def test_gibbs_reproduces_the_exact_boltzmann_distribution():
    """20000 sweeps of sequential Gibbs on the 81-state system at kT = 1, against
    the enumerated ``p(s)``.

    The threshold is 0.03, justified against the sampling noise rather than
    picked. For 81 bins and N independent samples the expected
    total-variation error is ``0.5 * sum_k sqrt(2 p_k (1 - p_k) / (pi N))``,
    which for this distribution at N = 20000 is 0.0106. The measured value here
    is 0.0104 -- at the iid floor, so a sweep of four sequential single-site
    updates is effectively decorrelating this system. 0.03 is therefore about
    three sigma of headroom, while still being an order of magnitude below what
    any real bug produces: a sampler that ignores the ``q`` ratio measures 0.35
    on the Metropolis test below, and sign or transposition errors are worse
    still. The seed is fixed, so this is a deterministic assertion rather than
    a flaky one."""
    model = small_model()
    empirical = chain_histogram(
        model,
        sweeps=20000,
        burn_in=200,
        step=lambda s, g: gibbs_sweep(model, s, range(N_RES), 1.0, g),
        seed=1234,
    )
    assert total_variation(empirical, exact_boltzmann(model, 1.0)) < 0.03


def test_metropolis_reproduces_it_from_a_biased_proposal():
    """The same comparison, but proposing from ``(0.7, 0.2, 0.1)`` -- a
    distribution with nothing to do with the energy, skewed almost an order of
    magnitude toward type 0.

    This is the test that MH is MH. The target is unchanged, so the only thing
    that can remove the proposal's bias from the stationary distribution is the
    ``q_old / q_new`` factor in :func:`metropolis_ratio`; an implementation that
    accepted on ``exp(-dE/kT)`` alone -- the Metropolis criterion, correct only
    for a symmetric proposal -- would converge to something visibly
    over-weighted toward type 0 and fail here while passing the Gibbs test
    above.

    More sweeps than the Gibbs test (40000) because the chain is slower: a
    fraction of proposals are rejected, so one sweep moves less. The threshold
    is the same 0.03; the measured value is 0.0192, against an iid floor of
    0.0075 at this N -- the factor of 2.5 above the floor is the rejection
    autocorrelation, and it is the reason the sweep count is doubled rather
    than the threshold loosened. For scale, the same chain with the ``q`` ratio
    deleted (accepting on ``exp(-dE/kT)`` alone) measures 0.35, so this
    threshold separates the two implementations by a factor of eighteen."""
    model = small_model()
    proposal = torch.tensor([0.7, 0.2, 0.1])
    empirical = chain_histogram(
        model,
        sweeps=40000,
        burn_in=500,
        step=lambda s, g: metropolis_sweep(
            model, s, range(N_RES), 1.0, proposal, g
        ),
        seed=1234,
    )
    assert total_variation(empirical, exact_boltzmann(model, 1.0)) < 0.03


def test_annealing_to_zero_finds_the_brute_force_global_minimum():
    """Six starting seeds, geometric schedule from kT = 2 to 0.05 and then five
    sweeps at exactly zero, against the enumerated minimum.

    The quench matters: at ``temperature=0`` the conditional is the argmin, so
    the last sweeps are iterated conditional modes and the result is a genuine
    single-site local minimum rather than a low-temperature sample near one.

    One honest caveat, found while writing this and recorded rather than hidden
    inside a longer schedule. Annealing is a stochastic optimiser, so this
    assertion is about the code only when the global minimum is separated from
    its runner-up. Swept over model seeds with this edge list, the hit rate
    tracks that gap and nothing else: seeds 0, 1 and 5 (gaps 0.83, 1.72, 1.34)
    reach the minimum from 16 of 16 starts, seed 2 (gap 0.62) from 11, and seed
    3 (gap 0.11) from 7 -- correct behaviour for a quench into a near-degenerate
    pair, and a test of luck rather than of this module. Lengthening the
    schedule does not fix seed 3; the two minima are genuinely that close. So
    the fixed model is seed 1, and the assertion on the gap below is what keeps
    this test honest if anyone changes the model."""
    model = small_model()
    states = all_states()
    energies = energy(model, states)
    best = float(energies.min())
    sorted_energies = energies.sort().values
    assert sorted_energies[1] - sorted_energies[0] > 1.0, "degenerate target"

    schedule = [2.0 * (0.05 / 2.0) ** (k / 199) for k in range(200)] + [0.0] * 5
    for seed in range(6):
        generator = torch.Generator().manual_seed(seed)
        start = torch.randint(N_TYPES, (N_RES,), generator=generator, dtype=torch.long)
        result = anneal(model, start, schedule, generator)
        assert isinstance(result, AnnealResult)
        assert float(energy(model, result.sequence)) == pytest.approx(best, abs=1e-5)
        # The convergence diagnostic HANDOFF.md asks for: the change count must
        # decay to zero. A plateau at a nonzero value is a limit cycle.
        assert result.changed[-5:] == [0, 0, 0, 0, 0]
        assert len(result.changed) == len(schedule)


# --- conditional ------------------------------------------------------------


def test_conditional_agrees_with_explicit_delta_energies():
    """``conditional`` is an optimisation of the same locality argument as
    ``delta_energy``, so it gets the same treatment: rebuild it from n_types
    explicit deltas off the current type and compare. The reference is written
    the obvious way on purpose -- a softmax over ``-dE/kT`` with no shifting --
    which only agrees at a temperature where the naive form does not
    overflow."""
    model = random_model(n_res=10, n_types=20, density=0.4, seed=8)
    generator = torch.Generator().manual_seed(4)
    sequence = torch.randint(
        model.n_types, (model.n_res,), generator=generator, dtype=torch.long
    )
    for temperature in (0.5, 1.0, 3.0):
        for position in range(model.n_res):
            deltas = torch.tensor(
                [
                    float(delta_energy(model, sequence, position, t))
                    for t in range(model.n_types)
                ]
            )
            reference = torch.exp(-deltas / temperature)
            reference = reference / reference.sum()
            got = conditional(model, sequence, position, temperature)
            assert torch.allclose(got, reference, atol=1e-5)


def test_conditional_stays_a_probability_vector_at_very_low_temperature():
    """``exp(-E/kT)`` at kT = 1e-6 with E of order 1 is ``exp(1e6)``, which
    overflows float32 for every type at once -- and then subtracting the
    maximum afterwards gives ``inf - inf``. The shift has to happen on the
    energies before the division, and this is the test that it does."""
    model = random_model(n_res=8, n_types=20, density=0.5, seed=9)
    sequence = torch.zeros(model.n_res, dtype=torch.long)
    for temperature in (1e-3, 1e-6, 1e-30, 1e-38, 1e-45, 1e-300, 0.0):
        for position in range(model.n_res):
            probabilities = conditional(model, sequence, position, temperature)
            assert torch.isfinite(probabilities).all()
            assert (probabilities >= 0).all()
            assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-6)
        # And at these temperatures it should have collapsed onto the argmin,
        # not merely stayed finite.
        effective = torch.tensor(
            [float(delta_energy(model, sequence, 0, t)) for t in range(model.n_types)]
        )
        collapsed = conditional(model, sequence, 0, temperature)
        assert int(collapsed.argmax()) == int(effective.argmin())


def test_a_temperature_that_underflows_the_dtype_is_a_quench_not_a_nan():
    """The hole between 1e-6 and exactly 0, which both of those tests miss.

    ``temperature`` arrives as a Python float, so 1e-300 is genuinely positive
    and the ``== 0`` guard does not fire -- but torch casts it to the tensor's
    dtype for the division, where float32 has nothing below about 1.2e-38, so
    it becomes exactly zero and the division is 0/0 at the best type and -inf
    at the rest. Every entry comes back NaN at a temperature that looks fine.

    The fix treats any temperature at or below ``finfo.tiny`` as a quench,
    which is also the physically honest reading: a kT below the dtype's
    resolution is zero here. This test pins the whole range rather than one
    value, because the bug is a gap and a single probe can sit either side of
    it.
    """
    model = random_model(n_res=6, n_types=20, density=0.6, seed=4)
    sequence = torch.arange(model.n_res) % model.n_types
    delta = torch.tensor(-0.5)
    for temperature in (1e-20, 1e-38, 1e-39, 1e-45, 1e-300, 5e-324, 0.0):
        probabilities = conditional(model, sequence, 2, temperature)
        assert torch.isfinite(probabilities).all(), temperature
        assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-6)
        # A downhill move is still accepted, rather than NaN-ing the ratio.
        ratio = metropolis_ratio(
            delta, torch.tensor(0.3), torch.tensor(0.2), temperature
        )
        assert torch.isfinite(ratio), temperature
        assert 0.0 <= float(ratio) <= 1.0


def test_conditional_rejects_a_negative_temperature():
    model = small_model()
    with pytest.raises(ValueError):
        conditional(model, torch.zeros(N_RES, dtype=torch.long), 0, -1.0)


# --- the acceptance ratio ----------------------------------------------------


def test_a_symmetric_proposal_cancels_the_q_ratio():
    """Rosetta's case: the ratio must reduce to ``min(1, exp(-dE/kT))``."""
    for delta in (-2.0, -0.1, 0.0, 0.5, 3.0):
        plain = metropolis_ratio(delta, 0.25, 0.25, 1.5)
        expected = min(1.0, float(torch.exp(torch.tensor(-delta / 1.5))))
        assert float(plain) == pytest.approx(expected, abs=1e-6)


def test_a_conditional_proposal_makes_the_ratio_one():
    """Gibbs's case, taken off the real conditional rather than constructed:
    for any pair of types, ``q_new / q_old`` from
    :func:`aanca.potts.conditional` is exactly ``exp(-dE/kT)``, so the whole
    acceptance is 1 and the chain never rejects. This is the claim the
    docstring of ``metropolis_ratio`` makes, and the reason one function covers
    both samplers."""
    model = small_model()
    sequence = torch.tensor([2, 0, 1, 1])
    temperature = 0.8
    q = conditional(model, sequence, 2, temperature)
    for new_type in range(N_TYPES):
        delta = delta_energy(model, sequence, 2, new_type)
        ratio = metropolis_ratio(delta, q[int(sequence[2])], q[new_type], temperature)
        assert float(ratio) == pytest.approx(1.0, abs=1e-5)


def test_the_ratio_is_a_probability_even_for_an_enormous_uphill_move():
    """``exp(-dE/kT)`` for dE = 500 at kT = 0.01 underflows; for dE = -500 it
    overflows. Both must come out as a usable acceptance probability, because
    an annealing schedule visits exactly those temperatures."""
    assert float(metropolis_ratio(500.0, 0.5, 0.5, 0.01)) == 0.0
    assert float(metropolis_ratio(-500.0, 0.5, 0.5, 0.01)) == 1.0
    assert float(metropolis_ratio(1.0, 0.5, 0.5, 0.0)) == 0.0
    assert float(metropolis_ratio(-1.0, 0.5, 0.5, 0.0)) == 1.0


# --- reproducibility ---------------------------------------------------------


def test_the_same_generator_seed_gives_the_same_chain():
    """Every sampler here takes its randomness from the ``generator``
    argument and nothing else, so a run is replayable from a seed. Without that
    a diverging diagnostic cannot be re-run."""
    model = small_model()
    start = torch.tensor([0, 1, 2, 0])
    schedule = [1.0, 0.5, 0.25, 0.1, 0.0]

    def run(seed: int, sweep_proposal=None):
        return anneal(
            model,
            start,
            schedule,
            torch.Generator().manual_seed(seed),
            proposal=sweep_proposal,
        )

    first, again = run(42), run(42)
    assert torch.equal(first.sequence, again.sequence)
    assert first.changed == again.changed

    proposal = torch.tensor([0.5, 0.3, 0.2])
    mh_first, mh_again = run(42, proposal), run(42, proposal)
    assert torch.equal(mh_first.sequence, mh_again.sequence)
    assert mh_first.changed == mh_again.changed


def test_different_seeds_give_different_chains():
    """The complement: identical output across seeds would mean the generator
    is being ignored and every "independent" run is the same run."""
    model = small_model()
    start = torch.tensor([0, 0, 0, 0])
    schedule = [2.0] * 40
    traces = {
        tuple(
            anneal(model, start, schedule, torch.Generator().manual_seed(seed)).changed
        )
        for seed in range(8)
    }
    assert len(traces) > 1


# --- energy, construction and validation ------------------------------------


def test_energy_sums_the_fields_and_the_edges_it_is_given():
    """A hand-computed case, so that a sign error shared between ``energy`` and
    every test written against it cannot survive."""
    h = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    edges = torch.tensor([[0, 1]])
    J = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    model = PottsModel(h=h, edges=edges, J=J)
    assert float(energy(model, torch.tensor([0, 0]))) == pytest.approx(1.0 + 0.5 + 10.0)
    assert float(energy(model, torch.tensor([0, 1]))) == pytest.approx(
        1.0 + 0.25 + 20.0
    )
    # J[e] is read as J[e, s_i, s_j], so the off-diagonal entries must not be
    # interchangeable -- that asymmetry is what the i < j rule protects.
    assert float(energy(model, torch.tensor([1, 0]))) == pytest.approx(
        -2.0 + 0.5 + 30.0
    )


def test_energy_takes_a_batch_of_sequences():
    model = random_model(n_res=9, n_types=20, density=0.4, seed=6)
    generator = torch.Generator().manual_seed(1)
    batch = torch.randint(
        model.n_types, (7, model.n_res), generator=generator, dtype=torch.long
    )
    stacked = energy(model, batch)
    assert stacked.shape == (7,)
    for k in range(7):
        assert float(stacked[k]) == pytest.approx(float(energy(model, batch[k])), abs=1e-4)
    assert energy(model, batch[0]).shape == ()


def test_edges_from_adjacency_reads_only_the_upper_triangle():
    """A contact matrix may arrive symmetric, one-sided or with a self-contact
    on the diagonal; all three must give the same edge list, because a repeated
    or reversed pair would double-count a coupling and a diagonal entry is a
    field masquerading as a pair term."""
    adjacency = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [False, True, False],
        ]
    )
    edges = edges_from_adjacency(adjacency)
    assert edges.tolist() == [[0, 1], [1, 2]]
    one_sided = torch.triu(adjacency, diagonal=1)
    assert edges_from_adjacency(one_sided).tolist() == edges.tolist()
    assert (edges[:, 0] < edges[:, 1]).all()

    with pytest.raises(ValueError):
        edges_from_adjacency(torch.ones(3, 4, dtype=torch.bool))


def test_from_adjacency_and_an_explicit_edge_list_agree():
    adjacency = torch.zeros(N_RES, N_RES, dtype=torch.bool)
    adjacency[EDGES[:, 0], EDGES[:, 1]] = True
    h = torch.zeros(N_RES, N_TYPES)
    J = torch.arange(EDGES.shape[0] * N_TYPES * N_TYPES, dtype=torch.float32).reshape(
        EDGES.shape[0], N_TYPES, N_TYPES
    )
    built = PottsModel.from_adjacency(h, adjacency, J)
    assert built.edges.tolist() == EDGES.tolist()


def test_construction_rejects_mismatched_shapes():
    h = torch.zeros(5, 4)
    edges = torch.tensor([[0, 1], [1, 2]])
    good_J = torch.zeros(2, 4, 4)
    PottsModel(h=h, edges=edges, J=good_J)

    with pytest.raises(ValueError):  # h not 2-D
        PottsModel(h=torch.zeros(5), edges=edges, J=good_J)
    with pytest.raises(ValueError):  # edges not (n_edges, 2)
        PottsModel(h=h, edges=torch.tensor([0, 1, 2]), J=good_J)
    with pytest.raises(ValueError):  # one J per edge
        PottsModel(h=h, edges=edges, J=torch.zeros(3, 4, 4))
    with pytest.raises(ValueError):  # J types must match h types
        PottsModel(h=h, edges=edges, J=torch.zeros(2, 4, 3))
    with pytest.raises(ValueError):  # endpoint outside the protein
        PottsModel(h=h, edges=torch.tensor([[0, 1], [1, 9]]), J=good_J)


def test_construction_rejects_an_edge_that_is_not_ordered():
    """``J[e]`` is indexed ``J[e, s_i, s_j]``, so a row stored as ``(j, i)``
    transposes that edge's coupling matrix and the energy is wrong by a number
    no shape check would catch. A self-edge is rejected by the same rule: it is
    a field, not a pair term."""
    h = torch.zeros(4, 3)
    J = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError, match="i < j"):
        PottsModel(h=h, edges=torch.tensor([[2, 1]]), J=J)
    with pytest.raises(ValueError, match="i < j"):
        PottsModel(h=h, edges=torch.tensor([[2, 2]]), J=J)


def test_a_model_with_no_edges_is_the_independent_sites_case():
    """Degenerate but reachable -- a contact map with a cutoff too tight, or a
    single-residue crop. Every function must still run, and the energy must be
    the field sum."""
    h = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    model = PottsModel(h=h, edges=torch.zeros(0, 2, dtype=torch.long), J=torch.zeros(0, 2, 2))
    assert float(energy(model, torch.tensor([1, 0]))) == pytest.approx(1.0 + 2.0)
    assert model.neighbours(0).numel() == 0
    assert float(delta_energy(model, torch.tensor([0, 0]), 0, 1)) == pytest.approx(1.0)
    # With no couplings the conditional is the single-site Boltzmann of h.
    reference = torch.softmax(-h[1] / 0.5, dim=0)
    assert torch.allclose(conditional(model, torch.tensor([0, 0]), 1, 0.5), reference)


def test_sweeps_reject_positions_outside_the_protein():
    """``positions`` comes from outside this module (``aanca.schedule`` will
    supply it), so an out-of-range index is an interface error worth raising
    rather than a silent wrap-around."""
    model = small_model()
    sequence = torch.zeros(N_RES, dtype=torch.long)
    with pytest.raises(IndexError):
        gibbs_sweep(model, sequence, [0, N_RES], 1.0, torch.Generator())
    with pytest.raises(ValueError):
        metropolis_sweep(
            model, sequence, [0], 1.0, torch.ones(N_TYPES + 1), torch.Generator()
        )


def test_a_sweep_touches_only_the_positions_it_is_given():
    """The reason positions are an argument: a caller passing a conditionally
    independent set must get exactly that set updated, or the parallel-update
    correctness argument it is making is void."""
    model = small_model()
    sequence = torch.tensor([0, 0, 0, 0])
    updated, _changed = gibbs_sweep(
        model, sequence, [1], 2.0, torch.Generator().manual_seed(0)
    )
    assert updated[0] == sequence[0]
    assert updated[2] == sequence[2]
    assert updated[3] == sequence[3]
    # And the input is not mutated, since callers keep the pre-sweep state to
    # compute the change count against.
    assert torch.equal(sequence, torch.tensor([0, 0, 0, 0]))


def test_a_per_position_proposal_is_accepted_as_well_as_a_shared_one():
    """``(n_res, n_types)`` is the shape a network head emits, and it is the
    whole reason ``metropolis_sweep`` exists rather than just Gibbs."""
    model = small_model()
    sequence = torch.zeros(N_RES, dtype=torch.long)
    shared = torch.tensor([0.4, 0.4, 0.2])
    per_position = shared.unsqueeze(0).expand(N_RES, -1).contiguous()
    one = metropolis_sweep(
        model, sequence, range(N_RES), 1.0, shared, torch.Generator().manual_seed(3)
    )
    other = metropolis_sweep(
        model,
        sequence,
        range(N_RES),
        1.0,
        per_position,
        torch.Generator().manual_seed(3),
    )
    assert torch.equal(one[0], other[0])
