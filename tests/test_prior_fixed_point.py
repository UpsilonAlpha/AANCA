"""The probe's defaults ARE the registered numbers, and it refuses rather than
falling through to its own.

The one thing worth pinning with a test: `--tolerance` and `--max-steps` default
to 0.1 A and 500 steps because
`docs/claims/a-local-rule-with-an-attractor-is-a-diffusion-model-without-a-schedule.md`
registers those, and `scripts/train_denoise.py` samples at 0.01 A and 200 steps
instead. If someone "tidies" these defaults to match the trainer, the probe stops
measuring the registered claim while still printing a verdict under its name --
which is the exact failure
`docs/claims/a-probe-pointed-at-best-pt-silently-used-its-own-defaults.md`
records.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


probe = _module("prior_fixed_point")


def test_defaults_are_the_registered_numbers():
    options = probe.build_parser().parse_args([])
    assert options.tolerance == 0.1, "the registered threshold is 0.1 A per step"
    assert options.max_steps == 500, "the registered cap is 500 steps"


def test_the_trainer_samples_at_a_stricter_setting():
    """So `conv 0/7` in a training log is not the registered measurement."""
    trainer = _module("train_denoise")
    args = trainer.build_schema().parse_args([])
    registered = probe.build_parser().parse_args([])
    assert args.sample_tolerance < registered.tolerance
    assert args.sample_max_steps < registered.max_steps


def test_a_checkpoint_without_args_is_refused(tmp_path):
    """Not scored with the probe's own defaults."""
    readout = _module("persistence_readout")
    trainer = _module("train_denoise")
    with pytest.raises(readout.ProbeRefusal):
        readout.settings_from_checkpoint({}, trainer.build_schema())


def test_a_missing_checkpoint_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["p", "--checkpoint", str(tmp_path / "nope.pt")]
    )
    assert probe.main() == 1


def test_native_edges_is_not_a_flag():
    """It comes from the checkpoint: the native graph would give the answer away."""
    dests = {a.dest for a in probe.build_parser()._actions}
    assert "native_edges" not in dests


def test_the_tolerance_is_tested_on_the_max_not_the_mean():
    """One residue still flying must not be hidden by 127 that have stopped."""
    trainer = _module("train_denoise")
    with pytest.raises(ValueError):
        trainer.run_to_fixed_point(
            rule=None, topo=None, carrier_coords=None, start_ca=torch.zeros(3, 3),
            tolerance=0.0,
        )
