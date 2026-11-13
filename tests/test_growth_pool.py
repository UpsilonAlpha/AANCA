"""The sample pool resumes a state rather than rebuilding the baseline.

Three properties, because "it ran and the loss went down" would pass with the
pool wired to nothing. What has to be true is that a resumed rollout **starts
where the last one ended**, that it keeps the window it was grown against
rather than drawing a fresh one, and that the hidden state travels -- resuming
coordinates alone would hand the rule a structure it had built with no memory
of building it, which is a start state no rollout ever produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca.growth import (  # noqa: E402
    GrowthConfig,
    GrowthPoolEntry,
    growth_objective,
)
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import RolloutConfig  # noqa: E402
from aanca.structure import load_protein  # noqa: E402
from aanca.torch_graph import feature_topology  # noqa: E402

PDB = ROOT / "data" / "raw" / "1UBQ.pdb"


@pytest.fixture(scope="module")
def protein():
    if not PDB.exists():
        pytest.skip(f"no {PDB}")
    return feature_topology(load_protein(str(PDB)))


@pytest.fixture(scope="module")
def rule():
    r = AANCARule(growth=True, birth="none")
    r.eval()
    return r


@pytest.fixture(scope="module")
def moving_rule():
    """A rule that actually moves atoms.

    **An untrained ``AANCARule`` is the identity map**, deliberately: the delta
    head's final layer is zero-initialised so a rule that has learned nothing
    does not move anything, which is the Growing-NCA trick and is why training
    does not diverge in its first thirty steps. It also means a rollout's end
    state equals its start state, so any property of the form "the resumed run
    began where the last one ended" is trivially true and invisible on an
    untrained rule -- which is how the first version of this file passed the
    wiring it meant to test and failed the behaviour.

    So the weights are perturbed here. Small enough to stay finite over a short
    rollout, large enough that coordinates change.
    """
    r = AANCARule(growth=True, birth="none")
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in r.parameters():
            parameter.add_(
                torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                ) * 0.05
            )
    r.eval()
    return r


def run(rule, topo, native, resume=None, steps=6, seed=0):
    config = RolloutConfig(
        steps=steps, truncate=steps, k=16,
        growth=GrowthConfig(birth="none"),
    )
    return growth_objective(
        rule, native, topo, config,
        generator=torch.Generator().manual_seed(seed),
        damage=12,
        fape_weight=1.0,
        resume_from=resume,
    )


def entry_from(terms, result):
    state = result.state
    return GrowthPoolEntry(
        coords=state.coords.detach().clone(),
        hidden=state.hidden.detach().clone(),
        damaged=terms.damaged_mask.clone(),
        baseline_rmsd=terms.start_rmsd_damaged,
    )


def test_the_resumed_rollout_starts_from_the_entrys_coordinates(rule, protein):
    """Tested against coordinates chosen to be unmistakable, not against a
    rollout's output -- so it holds on an identity-map rule too."""
    topo, native = protein
    terms, result = run(rule, topo, native)
    baseline_start = terms.start_rmsd_damaged
    assert baseline_start > 1.0, "the baseline is not displaced; nothing to tell apart"

    # An entry holding the NATIVE coordinates. If the resume path is wired,
    # the rollout starts at zero displacement; if it silently rebuilds the
    # baseline, it starts at `baseline_start` instead. Nothing in between.
    native_entry = GrowthPoolEntry(
        coords=native.clone(),
        hidden=result.state.hidden.detach().clone(),
        damaged=terms.damaged_mask.clone(),
        baseline_rmsd=baseline_start,
    )
    resumed, _ = run(rule, topo, native, resume=native_entry, seed=1)
    assert resumed.start_rmsd_damaged == pytest.approx(0.0, abs=1e-3), (
        f"resumed from native coordinates but started at "
        f"{resumed.start_rmsd_damaged:.3f} A, so the entry was ignored"
    )


def test_a_resumed_rollout_starts_where_the_last_one_ended(moving_rule, protein):
    topo, native = protein
    terms, result = run(moving_rule, topo, native)
    assert terms.rmsd_damaged != pytest.approx(terms.start_rmsd_damaged, abs=1e-6), (
        "the rule did not move the window, so this property is untestable"
    )
    entry = entry_from(terms, result)

    resumed, _ = run(moving_rule, topo, native, resume=entry, seed=1)

    # The first rollout began at the baseline; the second began at the state
    # the first reached. So the resumed run's START must equal the first run's
    # END, and must differ from the baseline it would otherwise have rebuilt.
    assert resumed.start_rmsd_damaged == pytest.approx(
        terms.rmsd_damaged, abs=1e-4
    ), "a resumed rollout did not start where the previous one ended"
    assert abs(resumed.start_rmsd_damaged - terms.start_rmsd_damaged) > 1e-6, (
        "the resumed start equals the baseline, so the entry was ignored"
    )


def test_a_resumed_rollout_keeps_the_window_it_was_grown_against(rule, protein):
    topo, native = protein
    terms, result = run(rule, topo, native)
    entry = entry_from(terms, result)

    # A different generator seed would draw a different damage window if the
    # resume path re-drew one. It must not.
    resumed, _ = run(rule, topo, native, resume=entry, seed=99)
    assert torch.equal(resumed.damaged_mask, terms.damaged_mask), (
        "the resumed rollout drew a fresh window, so the loss scored a hole "
        "the state was never asked to fill"
    )


def test_the_hidden_state_travels_with_the_entry(moving_rule, protein):
    topo, native = protein
    terms, result = run(moving_rule, topo, native)
    entry = entry_from(terms, result)
    rule = moving_rule

    blank = GrowthPoolEntry(
        coords=entry.coords.clone(),
        hidden=torch.zeros_like(entry.hidden),
        damaged=entry.damaged.clone(),
        baseline_rmsd=entry.baseline_rmsd,
    )
    with_memory, _ = run(rule, topo, native, resume=entry, seed=3)
    without_memory, _ = run(rule, topo, native, resume=blank, seed=3)

    # Same coordinates, same window, same seed -- so if the two outcomes agree,
    # the hidden state is not being carried and the pool is storing geometry
    # only.
    assert with_memory.rmsd_damaged != pytest.approx(
        without_memory.rmsd_damaged, abs=1e-9
    ), "zeroing the hidden state changed nothing, so it is not being resumed"


def test_the_baseline_path_is_untouched_when_nothing_is_resumed(rule, protein):
    topo, native = protein
    first, _ = run(rule, topo, native, seed=7)
    again, _ = run(rule, topo, native, seed=7)
    assert first.start_rmsd_damaged == pytest.approx(again.start_rmsd_damaged)
    assert torch.equal(first.damaged_mask, again.damaged_mask)
