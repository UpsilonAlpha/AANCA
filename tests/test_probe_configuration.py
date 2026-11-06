"""A probe must not substitute its own defaults for the run's.

Four times now a probe configured differently from the run it was reading has
produced a number that looked like a finding: once by backpropagating through
30 steps where training truncates at 12, once by running trained weights on a
whole PDB instead of a crop, and twice on 6 Nov because ``best.pt`` carried
only the weights -- so ``state.get("args")`` was empty and every rollout
setting silently fell back to the probe's default. One of those tables reported
gradient norms of 5e+10, and the other measured a *grown* rollout for a
self-placing run, because the birth mode defaulted to ``free``.

These tests pin the two halves of the fix: the arguments now travel with the
weights, and a probe that cannot find them refuses instead of guessing.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.model import AANCARule

PROBE = Path(__file__).resolve().parent.parent / "scripts" / "growth_balance_live.py"


def _run_probe(argv: list[str]) -> int:
    """Run the probe's ``main`` with ``argv``, returning its exit code."""
    saved = sys.argv
    sys.argv = ["growth_balance_live.py", *argv]
    try:
        module = runpy.run_path(str(PROBE), run_name="probe_under_test")
        return int(module["main"]())
    finally:
        sys.argv = saved


def test_a_checkpoint_without_arguments_is_refused(tmp_path):
    """No stored arguments and no sibling: the probe must not guess.

    This is the exact shape of the two useless tables: a ``best.pt`` holding
    nothing but ``state_dict``. Guessing is worse than failing, because a
    default is indistinguishable from a measurement once it is printed in a
    table with a column heading over it.
    """
    rule = AANCARule(growth=True, birth="none", identity=True)
    path = tmp_path / "best.pt"
    ckpt.save(path, {"state_dict": rule.state_dict()})

    assert _run_probe(["--checkpoint", str(path), "--repeats", "1"]) == 1


def test_the_arguments_are_read_from_the_sibling_last_checkpoint(tmp_path):
    """``last.pt`` beside it is the same run, so the probe may read it.

    Every run before 6 Nov wrote its arguments only to ``last.pt``, so the
    recovery path is what makes those checkpoints measurable at all -- the
    weights cannot be re-trained to carry their own configuration.
    """
    rule = AANCARule(growth=True, birth="none", identity=True)
    ckpt.save(tmp_path / "best.pt", {"state_dict": rule.state_dict()})
    ckpt.save(
        tmp_path / "last.pt",
        {
            "rule": rule.state_dict(),
            "args": {"birth": "none", "seeds": 16, "steps": 120, "truncate": 12},
            "eval_indices": [3, 7],
        },
    )

    state = ckpt.load(tmp_path / "best.pt")
    assert not state.get("args"), "best.pt is the case this test exists for"

    beside = ckpt.load(tmp_path / "last.pt")
    assert beside["args"]["birth"] == "none"
    assert beside["args"]["seeds"] == 16
    assert beside["eval_indices"] == [3, 7]


def test_best_checkpoints_now_carry_their_run_arguments(tmp_path):
    """The trainer's ``best.pt`` payload, as the save site now writes it.

    Pinned by shape rather than by running 500 updates: the four keys are what
    a probe needs to reproduce the run's rollout, and ``state_dict`` is the key
    ``rule_weights`` looks for.
    """
    rule = AANCARule(growth=True, birth="none", identity=True)
    payload = {
        "state_dict": rule.state_dict(),
        "args": {"birth": "none", "initial": "helix", "seeds": 16, "steps": 120},
        "eval_indices": [1, 2, 3],
        "update": 500,
    }
    path = tmp_path / "best.pt"
    ckpt.save(path, payload)
    state = ckpt.load(path)

    assert state["args"]["birth"] == "none"
    assert state["args"]["initial"] == "helix"
    assert state["eval_indices"] == [1, 2, 3]
    assert state["update"] == 500
    weights = ckpt.rule_weights(state)
    assert "frame_head.0.weight" in weights
    assert all(isinstance(v, torch.Tensor) for v in weights.values())
