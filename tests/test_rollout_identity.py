"""The identity track: does a label ripple, and does it stay put when it should."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.identity import LIBRARY_SIZE, library_index
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    initial_identity,
    initial_state,
    rollout,
)
from aanca.structure import load_protein
from aanca.torch_geometry import measure_chis
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    return feature_topology(load_protein(path))


def _config(**kw):
    base = dict(steps=5, truncate=3, k=24, blind=True, identity_sampling="argmax")
    base.update(kw)
    return RolloutConfig(**base)


def _native_index(topo, coords):
    return library_index(
        topo.torsion.resnames, torch.rad2deg(measure_chis(coords, topo.torsion)[:, 0])
    )


def test_the_v1_rollout_carries_no_identity_at_all(ubiquitin):
    topo, coords = ubiquitin
    rule = AANCARule()
    out = rollout(rule, initial_state(coords, rule), topo, _config(blind=False),
                  generator=torch.Generator().manual_seed(0), record=False)
    assert out.state.identity is None and out.state.identity_logits is None


def test_an_untrained_rule_holds_a_native_sequence_still(ubiquitin):
    """The discrete fixed point, and the cheapest test that can fail.

    A rule whose identity head is zeroed emits nothing but the keep-current
    term, so a rollout started at the native sequence must end there. If this
    fails, the ripple is overwriting labels rather than revising them.
    """
    topo, coords = ubiquitin
    rule = AANCARule(identity=True)
    start = initial_identity(topo, coords, "native")
    out = rollout(
        rule,
        initial_state(coords, rule, identity=start),
        topo,
        _config(steps=12),
        generator=torch.Generator().manual_seed(0),
        record=False,
    )
    assert torch.equal(out.state.identity.argmax(-1), _native_index(topo, coords))


def test_a_masked_start_commits_a_label_everywhere_it_ticks(ubiquitin):
    topo, coords = ubiquitin
    rule = AANCARule(identity=True)
    out = rollout(
        rule,
        initial_state(coords, rule, identity=initial_identity(topo, coords, "masked")),
        topo,
        _config(steps=8, update_probability=1.0),
        generator=torch.Generator().manual_seed(0),
        record=False,
    )
    committed = out.state.identity.sum(dim=-1)
    assert torch.allclose(committed, torch.ones_like(committed))


def test_a_frozen_residue_keeps_the_label_it_was_given(ubiquitin):
    """Frozen means held, for a discrete label as much as for geometry."""
    topo, coords = ubiquitin
    rule = AANCARule(identity=True)
    torch.manual_seed(0)
    for p in rule.identity_head.parameters():
        p.data.add_(torch.randn_like(p) * 0.4)

    frozen = torch.zeros(topo.n_residues, dtype=torch.bool)
    frozen[::3] = True
    start = initial_identity(topo, coords, "native")
    out = rollout(
        rule,
        initial_state(coords, rule, identity=start.clone()),
        topo,
        _config(steps=6, update_probability=1.0),
        generator=torch.Generator().manual_seed(1),
        record=False,
        frozen=frozen,
    )
    assert torch.equal(
        out.state.identity[frozen].argmax(-1), start[frozen].argmax(-1)
    )
    moved = out.state.identity[~frozen].argmax(-1) != start[~frozen].argmax(-1)
    assert bool(moved.any()), "a perturbed head should move some free label"


def test_the_committed_label_is_an_input_at_the_next_step(ubiquitin):
    """The ripple, stated as a gradient: the label channel is differentiable.

    A loss on the final step's logits must reach the labels the rollout was
    started from, several steps and several truncation boundaries earlier.
    That is only true because the committed label goes forward as a
    straight-through one-hot; as a hard index the chain would stop at every
    commitment and the ripple would be observable but not trainable.

    Note what this does *not* test. ``identity_keep`` gets gradient even in a
    one-step rollout, because the start's own labels are an input to that one
    call -- so step count is not the discriminator here, and an earlier
    version of this test asserted that it was.
    """
    # ``truncate`` bounds it: with truncate < steps the chain is detached at
    # the boundary and the start receives nothing, which is correct BPTT and
    # also the ripple's real gradient horizon -- 12 steps by default.
    topo, coords = ubiquitin
    rule = AANCARule(identity=True)
    torch.manual_seed(0)
    for p in rule.identity_head.parameters():
        p.data.add_(torch.randn_like(p) * 0.2)

    start = initial_identity(topo, coords, "native").requires_grad_(True)
    out = rollout(
        rule,
        initial_state(coords, rule, identity=start),
        topo,
        _config(steps=6, truncate=99, update_probability=1.0),
        generator=torch.Generator().manual_seed(2),
        record=False,
    )
    out.state.identity_logits.square().mean().backward()
    assert start.grad is not None and float(start.grad.abs().sum()) > 0.0


def test_an_unknown_sampling_mode_is_refused(ubiquitin):
    topo, coords = ubiquitin
    rule = AANCARule(identity=True)
    with pytest.raises(ValueError, match="identity_sampling"):
        rollout(
            rule,
            initial_state(
                coords, rule, identity=initial_identity(topo, coords, "masked")
            ),
            topo,
            _config(identity_sampling="gumbel"),
            generator=torch.Generator().manual_seed(0),
            record=False,
        )


def test_an_identity_rollout_without_an_identity_head_is_refused(ubiquitin):
    topo, coords = ubiquitin
    rule = AANCARule()
    with pytest.raises(ValueError, match="identity head"):
        rollout(
            rule,
            initial_state(
                coords, rule, identity=initial_identity(topo, coords, "masked")
            ),
            topo,
            _config(),
            generator=torch.Generator().manual_seed(0),
            record=False,
        )


def test_initial_identity_knows_its_three_starts(ubiquitin):
    topo, coords = ubiquitin
    masked = initial_identity(topo, coords, "masked")
    assert float(masked.abs().max()) == 0.0
    native = initial_identity(topo, coords, "native")
    assert torch.equal(native.argmax(-1), _native_index(topo, coords))
    drawn = initial_identity(
        topo, coords, "random", torch.Generator().manual_seed(4)
    )
    assert drawn.shape == (topo.n_residues, LIBRARY_SIZE)
    assert not torch.equal(drawn.argmax(-1), native.argmax(-1))
    with pytest.raises(ValueError, match="identity_start"):
        initial_identity(topo, coords, "nonsense")
    with pytest.raises(ValueError, match="native coordinates"):
        initial_identity(topo, None, "native")
