"""The two claims ``scripts/basin_probe.py`` rests on that are not eyeballable.

The probe's output is read as prose, so a wrong number there looks exactly like
a right one. Two things underneath it can be wrong silently.

**The starting structure has to be the benchmark's.** ``descent`` asks whether
the rule stayed in the well it was dropped into, which is only a statement
about the published row if the drop is the published row's drop. That was a
duplicated recipe until ``aanca.casp.randomised_start`` existed, so the first
test pins the refactor against the recipe it replaced.

**Segmenting a rollout has to be free.** ``converge`` measures drift from the
step-100 structure by running one rollout and snapshotting it, which is only
comparable to the benchmark if the segments reconstruct an unsegmented run
exactly. One mask drawn out of order would make every drift number wrong and
nothing would complain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from aanca.casp import randomised_start
from aanca.data import von_mises_deltas
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.structure import load_protein
from aanca.torch_geometry import apply_chi_deltas, measure_chis, wrap_to_pi
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "basin_probe", ROOT / "scripts" / "basin_probe.py"
)
basin_probe = importlib.util.module_from_spec(SPEC)
sys.modules["basin_probe"] = basin_probe
SPEC.loader.exec_module(basin_probe)

DATA = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    protein = load_protein(path)
    topo, coords = feature_topology(protein)
    return topo, coords


def _trained_ish(seed: int = 0) -> AANCARule:
    """Identity init means zero updates, which makes dynamics tests vacuous."""
    rule = AANCARule()
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (rule.delta_head[-1], rule.kappa_head[-1]):
            layer.weight.normal_(0.0, 0.1)
    return rule


def test_randomised_start_is_the_recipe_the_benchmark_used(ubiquitin):
    """Pin the refactor against the inline version it replaced.

    If these ever disagree, every number the probe prints is about a different
    task from the one the CASP13 row reports.
    """
    topo, coords = ubiquitin
    deltas = von_mises_deltas(
        topo.torsion.chi_mask, 0.0, generator=torch.Generator().manual_seed(7)
    )
    expected = apply_chi_deltas(coords, deltas, topo.torsion)
    assert torch.equal(randomised_start(coords, topo, 7), expected)


def test_randomised_start_depends_on_the_seed_and_nothing_else(ubiquitin):
    topo, coords = ubiquitin
    again = randomised_start(coords, topo, 3)
    assert torch.equal(randomised_start(coords, topo, 3), again)
    assert not torch.equal(randomised_start(coords, topo, 4), again)


def test_kappa_controls_how_far_the_start_is_from_native(ubiquitin):
    """Larger kappa is milder damage, and kappa=0 is the uniform draw."""
    topo, coords = ubiquitin
    mask = topo.torsion.chi_mask
    native = measure_chis(coords, topo.torsion)

    def mean_shift(kappa: float) -> float:
        start = randomised_start(coords, topo, 11, kappa)
        error = wrap_to_pi(measure_chis(start, topo.torsion) - native).abs()
        return float(error[mask].rad2deg().mean())

    mild, harsh, uniform = mean_shift(64.0), mean_shift(2.0), mean_shift(0.0)
    assert mild < harsh < uniform
    # A von Mises at kappa 64 has a circular standard deviation near 7 degrees.
    assert mild < 15.0
    # A uniform draw on the circle averages 90 degrees from any fixed angle.
    assert 75.0 < uniform < 105.0


def test_segmenting_a_rollout_with_one_generator_visits_the_same_states(ubiquitin):
    """The claim behind ``converge``'s drift column.

    ``rollout`` draws one update mask per step from the generator it is handed,
    so resuming from the returned state with the *same* generator has to be
    indistinguishable from never having stopped.
    """
    topo, coords = ubiquitin
    rule = _trained_ish()
    whole = RolloutConfig(steps=6, truncate=2, k=32, cutoff=10.0)
    first = RolloutConfig(steps=2, truncate=2, k=32, cutoff=10.0)
    rest = RolloutConfig(steps=4, truncate=2, k=32, cutoff=10.0)

    with torch.no_grad():
        unsegmented = rollout(
            rule,
            initial_state(coords, rule),
            topo,
            whole,
            generator=torch.Generator().manual_seed(5),
            record=True,
        )
        generator = torch.Generator().manual_seed(5)
        state = initial_state(coords, rule)
        head = rollout(rule, state, topo, first, generator=generator, record=True)
        tail = rollout(rule, head.state, topo, rest, generator=generator, record=True)

    assert torch.equal(tail.state.coords, unsegmented.state.coords)
    assert tail.state.step == unsegmented.state.step
    assert (
        head.mean_abs_delta + tail.mean_abs_delta
        == pytest.approx(unsegmented.mean_abs_delta)
    )


def test_a_fresh_generator_does_not_resume_the_same_rollout(ubiquitin):
    """The negative control for the test above: the generator is what carries it."""
    topo, coords = ubiquitin
    rule = _trained_ish()
    first = RolloutConfig(steps=2, truncate=2, k=32, cutoff=10.0)
    rest = RolloutConfig(steps=4, truncate=2, k=32, cutoff=10.0)

    with torch.no_grad():
        kept = torch.Generator().manual_seed(5)
        state = initial_state(coords, rule)
        head = rollout(rule, state, topo, first, generator=kept, record=False)
        continued = rollout(rule, head.state, topo, rest, generator=kept, record=False)
        restarted = rollout(
            rule,
            head.state,
            topo,
            rest,
            generator=torch.Generator().manual_seed(5),
            record=False,
        )

    assert not torch.equal(continued.state.coords, restarted.state.coords)


def test_circular_error_takes_the_shortest_arc():
    predicted = torch.tensor([170.0, -170.0, 0.0, 179.0])
    target = torch.tensor([-170.0, 170.0, 0.0, -179.0])
    error = basin_probe.circular_error(predicted, target)
    assert error.tolist() == pytest.approx([20.0, 20.0, 0.0, 2.0], abs=1e-3)


def test_chi1_residues_drops_proline_and_residues_with_no_chi1(ubiquitin):
    """Proline has a chi1 the three-well model does not describe; GLY has none."""
    topo, _coords = ubiquitin
    torsion = topo.torsion
    keep = basin_probe.chi1_residues(torsion)
    names = list(torsion.resnames)

    assert any(name == "PRO" for name in names), "1UBQ should contain prolines"
    assert any(name == "GLY" for name in names), "1UBQ should contain glycines"
    for index, name in enumerate(names):
        if name in ("PRO", "GLY", "ALA"):
            assert not bool(keep[index]), f"{name} at {index} should be excluded"
    assert bool(keep.any())
    # Everything kept really does have a chi1 to measure.
    assert bool((torsion.chi_mask[:, 0].cpu() | ~keep).all())
