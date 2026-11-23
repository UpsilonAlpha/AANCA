"""``native_edge_selection``'s blinding is decided by a VALUE, not by a presence.

**The defect this file pins, which was live.** The parameter was
``identity: Tensor | None``, the branch was ``identity is not None``, and the
value was never read. So ``identity=False`` -- which every call site reads as
"no identity, do not blind" -- turned blinding ON, and the selection was
computed over coordinates with every atom beyond CB collapsed onto it. Three
tests in ``tests/test_native_edges_readout.py`` were passing exactly that and
getting blinded selection.

That is the same class of defect as a setting that appears to be used and is
not, and that class has already cost this project retractions: a plausible
number under the wrong description. It is now ``carries_identity: bool``, read
by value, and these tests are the guard.

Two assertions, and the second is the one that fails if the fix is reverted:

  1. blinding happens when it should -- a rollout that carries an identity
     state, under ``config.blind``, selects over blinded coordinates, matching
     ``rollout.step_once``, which blinds iff ``state.identity is not None and
     config.blind``;
  2. **a falsy argument does not blind.** ``False`` and the default both give
     the unblinded selection. Under the old signature ``carries_identity=False``
     would be truthy-by-presence and this test would fail with the blinded
     neighbour list.

1UBQ is used because the blinded and unblinded selections have to actually
differ for either assertion to mean anything, and the test asserts that
difference before asserting anything about the parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca.blind import blind_coords  # noqa: E402
from aanca.growth import (  # noqa: E402
    GrowthConfig,
    alive_topology,
    native_edge_selection,
)
from aanca.rollout import RolloutConfig  # noqa: E402
from aanca.structure import load_protein  # noqa: E402
from aanca.torch_graph import feature_topology, select_neighbours  # noqa: E402

PDB = ROOT / "data" / "raw" / "1UBQ.pdb"
K = 16


@pytest.fixture(scope="module")
def protein():
    if not PDB.exists():
        pytest.skip(f"no {PDB}")
    return feature_topology(load_protein(str(PDB)))


def _config(blind: bool) -> RolloutConfig:
    return RolloutConfig(
        steps=4, truncate=4, k=K, blind=blind,
        growth=GrowthConfig(birth="none"),
    )


def _expected(coords, topo, config, blind: bool):
    """The selection by hand: ``alive_topology`` all-true, then select.

    Spelled out rather than taken from the function under test, so the two
    sides of the assertion are not the same code.
    """
    topo = alive_topology(
        topo, torch.ones(topo.n_residues, dtype=torch.bool, device=coords.device)
    )
    seen = blind_coords(coords, topo) if blind else coords
    neighbours, _mask = select_neighbours(
        seen.to(config.feature_dtype), topo, config.k, config.cutoff
    )
    return neighbours


def test_the_blinded_and_unblinded_selections_differ_at_all(protein):
    """Otherwise every assertion below is vacuous on this structure."""
    topo, coords = protein
    config = _config(blind=True)
    assert not torch.equal(
        _expected(coords, topo, config, blind=True),
        _expected(coords, topo, config, blind=False),
    ), "1UBQ's blinded and unblinded neighbour lists agree, so this file proves nothing"


def test_a_rollout_that_carries_an_identity_state_selects_blinded(protein):
    """The behaviour being preserved: ``step_once``'s own two-part condition."""
    topo, coords = protein
    config = _config(blind=True)
    fixed = native_edge_selection(coords, topo, config, carries_identity=True)
    assert torch.equal(
        fixed.neighbours, _expected(coords, topo, config, blind=True)
    )


@pytest.mark.parametrize("falsy", [False, None, 0])
def test_a_falsy_argument_does_not_silently_enable_blinding(protein, falsy):
    """**The regression test.** Presence must not stand in for truth.

    ``False`` is the case that was wrong: under ``identity: Tensor | None``
    tested with ``is not None`` it blinded the selection, and this assertion
    fails with the blinded list. ``None`` and ``0`` are here because a
    presence test and a truth test disagree on exactly the falsy values, so
    pinning one of them is pinning half the fault.
    """
    topo, coords = protein
    config = _config(blind=True)
    fixed = native_edge_selection(coords, topo, config, carries_identity=falsy)
    assert torch.equal(
        fixed.neighbours, _expected(coords, topo, config, blind=False)
    ), (
        f"carries_identity={falsy!r} blinded the selection, so the parameter "
        "is being read for its presence rather than its value"
    )


def test_the_default_does_not_blind(protein):
    """A caller that says nothing gets no blinding, which is what the name says."""
    topo, coords = protein
    config = _config(blind=True)
    fixed = native_edge_selection(coords, topo, config)
    assert torch.equal(
        fixed.neighbours, _expected(coords, topo, config, blind=False)
    )


def test_config_blind_off_never_blinds(protein):
    """``config.blind`` stays the other half of the condition, inside the function.

    So an identity-carrying rollout with blinding off selects unblinded, and
    the decision is not duplicated at the call sites -- which is why the
    parameter is ``carries_identity`` and not ``blind``.
    """
    topo, coords = protein
    config = _config(blind=False)
    fixed = native_edge_selection(coords, topo, config, carries_identity=True)
    assert torch.equal(
        fixed.neighbours, _expected(coords, topo, config, blind=False)
    )


def test_no_caller_passes_the_old_argument_name():
    """``identity=`` would now be a ``TypeError`` at run time, in a readout.

    Cheap, and it catches the one thing a signature change leaves behind: a
    call site in a script nobody imports from a test. Searched over the source
    rather than by importing, because ``scripts/`` is not a package.
    """
    hits = []
    for path in sorted((ROOT / "src" / "aanca").glob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        if "native_edge_selection" not in text and "native_selection" not in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if ("native_edge_selection(" in line or "native_selection(" in line) and (
                "identity=" in line
            ):
                hits.append(f"{path.name}:{number}")
    assert not hits, f"these call sites still pass the removed parameter: {hits}"
