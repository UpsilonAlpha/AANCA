"""Pinning has to actually pin, and has to change nothing else.

``scripts/clamp_test.py`` asks whether correct neighbours improve the free
residues. Two silent failures would each produce a clean-looking table that
means nothing.

**A leaking pin.** If a pinned residue's atoms move, the arms stop differing in
what they claim to differ in, and at the limit the test compares a rollout to
itself. The script measures this at runtime too -- the "pinned drift" column --
but by then the compute is spent.

**A disturbed mask stream.** Pinning must not change which residues tick, or
the ``native`` and ``random`` arms are no longer the same rollout with
different held values, and neither is comparable to the unpinned anchor.

The last tests cover the deliberate asymmetry: a pinned residue keeps updating
its hidden state, because every message is built from the sender's hidden state
and a frozen one would be off-distribution for its neighbours. That is easy to
"fix" into a bug later, so it is pinned by a test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout, step_once, update_mask
from aanca.structure import load_protein
from aanca.torch_geometry import measure_chis
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "clamp_test", ROOT / "scripts" / "clamp_test.py"
)
clamp_test = importlib.util.module_from_spec(SPEC)
sys.modules["clamp_test"] = clamp_test
SPEC.loader.exec_module(clamp_test)

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
    """Identity init means zero updates, which makes pinning vacuous."""
    rule = AANCARule()
    torch.manual_seed(seed)
    with torch.no_grad():
        for layer in (rule.delta_head[-1], rule.kappa_head[-1]):
            layer.weight.normal_(0.0, 0.1)
    return rule


def _config(steps: int = 6) -> RolloutConfig:
    return RolloutConfig(steps=steps, truncate=3, k=32, cutoff=10.0)


def test_pinning_everything_leaves_every_atom_exactly_where_it_was(ubiquitin):
    topo, coords = ubiquitin
    rule = _trained_ish()
    frozen = torch.ones(topo.n_residues, dtype=torch.bool)
    with torch.no_grad():
        result = rollout(
            rule,
            initial_state(coords, rule),
            topo,
            _config(),
            generator=torch.Generator().manual_seed(1),
            record=False,
            frozen=frozen,
        )
    assert torch.equal(result.state.coords, coords)


def test_the_rule_really_would_have_moved_them(ubiquitin):
    """The control for the test above -- otherwise it passes on a silent rule."""
    topo, coords = ubiquitin
    rule = _trained_ish()
    with torch.no_grad():
        result = rollout(
            rule,
            initial_state(coords, rule),
            topo,
            _config(),
            generator=torch.Generator().manual_seed(1),
            record=False,
        )
    assert not torch.equal(result.state.coords, coords)


def test_pinning_nothing_is_the_same_rollout_as_not_pinning(ubiquitin):
    """Freezing must not perturb the mask stream."""
    topo, coords = ubiquitin
    rule = _trained_ish()
    empty = torch.zeros(topo.n_residues, dtype=torch.bool)
    with torch.no_grad():
        unpinned = rollout(
            rule,
            initial_state(coords, rule),
            topo,
            _config(),
            generator=torch.Generator().manual_seed(2),
            record=False,
        )
        pinned_none = rollout(
            rule,
            initial_state(coords, rule),
            topo,
            _config(),
            generator=torch.Generator().manual_seed(2),
            record=False,
            frozen=empty,
        )
    assert torch.equal(pinned_none.state.coords, unpinned.state.coords)


def test_only_the_free_residues_move(ubiquitin):
    topo, coords = ubiquitin
    rule = _trained_ish()
    frozen = torch.zeros(topo.n_residues, dtype=torch.bool)
    frozen[::2] = True
    with torch.no_grad():
        result = rollout(
            rule,
            initial_state(coords, rule),
            topo,
            _config(),
            generator=torch.Generator().manual_seed(3),
            record=False,
            frozen=frozen,
        )
    before = measure_chis(coords, topo.torsion)
    after = measure_chis(result.state.coords, topo.torsion)
    moved = ((after - before).abs() > 1e-9) & topo.torsion.chi_mask

    assert not bool(moved[frozen].any()), "a pinned residue moved"
    assert bool(moved[~frozen].any()), "no free residue moved, so this proves nothing"


def test_a_pinned_residue_still_updates_its_hidden_state(ubiquitin):
    """The deliberate asymmetry: the cell ticks, its atoms do not.

    Messages are built from the *sender's* ``cat([node_features, hidden])``, so
    a pinned residue with a frozen hidden state would feed its neighbours an
    all-zero vector that never occurs in training.
    """
    topo, coords = ubiquitin
    rule = _trained_ish()
    frozen = torch.ones(topo.n_residues, dtype=torch.bool)
    state = initial_state(coords, rule)
    mask = update_mask(topo.n_residues, 1.0, coords.device)
    with torch.no_grad():
        after, _mu, _log_kappa = step_once(rule, state, topo, _config(), mask, frozen)

    assert torch.equal(after.coords, coords), "geometry should be pinned"
    assert not torch.equal(after.hidden, state.hidden), "the cell should still tick"


def test_frozen_mask_is_reproducible_and_scales_with_the_fraction():
    a = clamp_test.frozen_mask(2000, 0.25, 11, "cpu")
    assert torch.equal(clamp_test.frozen_mask(2000, 0.25, 11, "cpu"), a)
    assert not torch.equal(clamp_test.frozen_mask(2000, 0.25, 12, "cpu"), a)
    assert 0.22 < float(a.float().mean()) < 0.28
    assert not bool(clamp_test.frozen_mask(2000, 0.0, 11, "cpu").any())


def test_the_native_arm_holds_pinned_residues_at_their_crystal_conformation(ubiquitin):
    topo, coords = ubiquitin
    rule = _trained_ish()
    with torch.no_grad():
        packed, frozen = clamp_test.clamped_pack(
            rule, coords, topo, _config(), 4, 0.5, "native"
        )
    assert bool(frozen.any()) and not bool(frozen.all())
    held = frozen.unsqueeze(-1) & topo.torsion.chi_mask
    native = measure_chis(coords, topo.torsion)
    got = measure_chis(packed, topo.torsion)
    assert torch.allclose(got[held], native[held], atol=1e-9)


def test_the_random_arm_does_not_hold_them_at_native(ubiquitin):
    """Otherwise the control arm is secretly the treatment arm."""
    topo, coords = ubiquitin
    rule = _trained_ish()
    with torch.no_grad():
        packed, frozen = clamp_test.clamped_pack(
            rule, coords, topo, _config(), 4, 0.5, "random"
        )
    held = frozen.unsqueeze(-1) & topo.torsion.chi_mask
    native = measure_chis(coords, topo.torsion)
    got = measure_chis(packed, topo.torsion)
    assert not torch.allclose(got[held], native[held], atol=1e-2)


def test_both_arms_pin_the_same_residues(ubiquitin):
    """So that 'native - random' varies only what they are held at."""
    topo, coords = ubiquitin
    rule = _trained_ish()
    with torch.no_grad():
        _packed_a, first = clamp_test.clamped_pack(
            rule, coords, topo, _config(), 9, 0.5, "native"
        )
        _packed_b, second = clamp_test.clamped_pack(
            rule, coords, topo, _config(), 9, 0.5, "random"
        )
    assert torch.equal(first, second)
