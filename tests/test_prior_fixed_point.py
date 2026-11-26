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


def _recorded(schema) -> dict:
    """What a trainer's `recorded_args` would write, at the schema's defaults.

    The checkpoints this exercises are synthetic on purpose: `runs/cond/best.pt`
    lives on Bunya, and the thing under test is the agreement between a recorded
    key set and a declared schema, which needs no weights at all.
    """
    from aanca import config as run_config

    args = schema.parse_args([])
    return {
        key: value
        for key, value in vars(args).items()
        if key not in run_config.RUNTIME_ONLY
    }


def test_the_trainer_is_declared_and_defaults_to_the_denoiser():
    """A closed set of two, and the MVP's trainer stays the default.

    `runs/mvp/best.pt` is the checkpoint this probe was written for, so the
    command line that measures it must not grow a required flag.
    """
    options = probe.build_parser().parse_args([])
    assert options.trainer == "train_denoise"
    assert options.label is None, "the unconditional row is the default"
    choices = {
        action.choices
        for action in probe.build_parser()._actions
        if action.dest == "trainer"
    }
    assert choices == {("train_denoise", "train_conditioned")}


def test_a_conditioned_checkpoint_declared_as_the_denoiser_still_refuses():
    """THE REFUSAL THIS WHOLE OPTION EXISTS TO KEEP.

    `label_dim` and `label_dropout` are settings `scripts/train_denoise.py` has
    no name for, so a checkpoint carrying them cannot be honoured against its
    schema. Naming the trainer is a declaration; if a wrong declaration were
    tolerated the check would be vacuous, which is the failure
    `docs/claims/a-probe-pointed-at-best-pt-silently-used-its-own-defaults.md`
    records.
    """
    readout = _module("persistence_readout")
    denoiser = _module("train_denoise")
    conditioned = _module("train_conditioned")
    with pytest.raises(readout.ProbeRefusal) as refusal:
        readout.settings_from_checkpoint(
            {"args": _recorded(conditioned.build_schema())},
            denoiser.build_schema(),
        )
    assert "label_dim" in str(refusal.value)
    assert "label_dropout" in str(refusal.value)


def test_the_denoiser_s_checkpoint_declared_as_conditioned_also_refuses():
    """The wrong declaration refuses in BOTH directions.

    An MVP checkpoint read against the conditioned schema is missing two keys,
    and a missing key is where a silent default would enter.
    """
    readout = _module("persistence_readout")
    denoiser = _module("train_denoise")
    conditioned = _module("train_conditioned")
    with pytest.raises(readout.ProbeRefusal) as refusal:
        readout.settings_from_checkpoint(
            {"args": _recorded(denoiser.build_schema())},
            conditioned.build_schema(),
        )
    assert "missing" in str(refusal.value)


def test_the_right_declaration_is_honoured():
    """And each schema reads its own trainer's checkpoint without complaint."""
    readout = _module("persistence_readout")
    for name in ("train_denoise", "train_conditioned"):
        trainer = _module(name)
        args = readout.settings_from_checkpoint(
            {"args": _recorded(trainer.build_schema())}, trainer.build_schema()
        )
        assert args.k == 24
    conditioned = _module("train_conditioned")
    args = readout.settings_from_checkpoint(
        {"args": _recorded(conditioned.build_schema())},
        conditioned.build_schema(),
    )
    assert args.label_dim == 4


def test_the_trainer_is_not_sniffed_out_of_the_checkpoint():
    """No auto-detection, which would make every checkpoint readable."""
    source = (ROOT / "scripts" / "prior_fixed_point.py").read_text(
        encoding="utf-8"
    )
    assert '_module(options.trainer)' in source
    assert "label_dim" not in source.split('"""', 2)[2], (
        "the trainer must not be inferred from the recorded keys"
    )


def test_the_label_is_inert_under_the_denoiser(monkeypatch, tmp_path):
    """Passing one is an error, not a silently ignored flag.

    `scripts/train_denoise.py`'s rule has no label to set, so a probe that
    accepted the option would print a class in the header of a rollout that
    never read one.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        ["p", "--checkpoint", str(tmp_path / "x.pt"), "--label", "mainly-alpha"],
    )
    assert probe.main() == 2


def test_the_label_choices_are_the_classes_plus_the_unconditional_row():
    from aanca.conditioned import FOLD_CLASSES

    choices = {
        action.choices
        for action in probe.build_parser()._actions
        if action.dest == "label"
    }
    assert choices == {FOLD_CLASSES + ("unconditional",)}


def test_the_rollout_is_wrapped_in_conditioned_on():
    """The label reaches the rule the way the arm's own trainer sets it.

    `ConditionedEGNCARule` carries the label as state and restores it on exit,
    so the rollout must run inside `conditioned_on` rather than assigning the
    attribute -- otherwise one sample's condition can leak into the next and
    nothing in the printed table would say so.
    """
    source = (ROOT / "scripts" / "prior_fixed_point.py").read_text(
        encoding="utf-8"
    )
    assert "rule.conditioned_on(row)" in source
    assert "nullcontext()" in source, "the denoiser's path must stay unchanged"


def test_the_tolerance_is_tested_on_the_max_not_the_mean():
    """One residue still flying must not be hidden by 127 that have stopped."""
    trainer = _module("train_denoise")
    with pytest.raises(ValueError):
        trainer.run_to_fixed_point(
            rule=None, topo=None, carrier_coords=None, start_ca=torch.zeros(3, 3),
            tolerance=0.0,
        )
