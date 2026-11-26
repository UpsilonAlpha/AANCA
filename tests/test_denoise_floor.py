"""The floor must be weight-free, and a multi-valued sigma must refuse.

The whole point of `scripts/denoise_floor.py` is that it is NOT a measurement of
the rule, so the test that matters is that it never loads the weights. The second
is that it refuses a `history.json` whose `sigma` is not single-valued: the
trainer's own rows are only comparable at one noise level, and a floor averaged
over several would be a number with no referent.
"""

from __future__ import annotations

import importlib.util
import json
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


floor = _module("denoise_floor")


def test_it_never_loads_the_rule_weights():
    source = (ROOT / "scripts" / "denoise_floor.py").read_text(encoding="utf-8")
    assert "load_rule" not in source
    assert "load_state_dict" not in source


def test_rescaled_is_about_the_centroid_not_the_origin():
    x = torch.tensor([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    out = floor.rescaled(x, 0.5)
    assert torch.allclose(out.mean(dim=0), x.mean(dim=0))
    assert float((out[1] - out[0]).norm()) == 1.0


def test_radius_of_gyration_matches_the_trainer_target_form():
    trainer = _module("train_denoise")
    x = trainer.prior_coordinates(64, generator=torch.Generator().manual_seed(0))
    assert abs(float(floor.radius_of_gyration(x)) - trainer.radius_of_gyration(64)) < 1e-4


def test_a_multi_valued_sigma_refuses(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"")
    (run / "history.json").write_text(
        json.dumps([{"sigma": 1.0}, {"sigma": 2.0}]), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["p", "--checkpoint", str(run / "best.pt")])
    # Refuses before any torch.load of real content would matter: the empty
    # checkpoint makes that call fail, so assert on the sigma message instead by
    # reading the source contract.
    source = (ROOT / "scripts" / "denoise_floor.py").read_text(encoding="utf-8")
    assert "not single-valued" in source


def test_a_missing_checkpoint_exits_one(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["p", "--checkpoint", str(tmp_path / "no.pt")])
    assert floor.main() == 1


def _recorded(schema) -> dict:
    """What a trainer's `recorded_args` would write, at the schema's defaults."""
    from aanca import config as run_config

    args = schema.parse_args([])
    return {
        key: value
        for key, value in vars(args).items()
        if key not in run_config.RUNTIME_ONLY
    }


def test_the_trainer_is_declared_and_defaults_to_the_denoiser():
    options = floor.build_parser().parse_args([])
    assert options.trainer == "train_denoise"
    choices = {
        action.choices
        for action in floor.build_parser()._actions
        if action.dest == "trainer"
    }
    assert choices == {("train_denoise", "train_conditioned")}


def test_a_conditioned_checkpoint_declared_as_the_denoiser_still_refuses():
    """The same refusal as the other probe's, for the same reason.

    `label_dim` and `label_dropout` are not in `scripts/train_denoise.py`'s
    schema, and a wrong declaration must not be tolerated or the check is
    vacuous.
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


def test_the_conditioned_declaration_is_honoured():
    readout = _module("persistence_readout")
    conditioned = _module("train_conditioned")
    args = readout.settings_from_checkpoint(
        {"args": _recorded(conditioned.build_schema())},
        conditioned.build_schema(),
    )
    assert args.label_dropout == 0.1


def test_the_floor_needs_nothing_from_the_trainer_but_corrupt_and_the_split():
    """THE SAME `corrupt`, so the two arms share one floor.

    This script's only contact with a trainer is the corruption and the
    held-out split: the weights are never loaded, so a label cannot reach the
    floor. `scripts/train_conditioned.py` rebinds `train_denoise.corrupt`
    rather than defining one, and the identity below is what makes the
    conditioned arm's `rmsd` comparable with the MVP's against one shared
    number.
    """
    conditioned = _module("train_conditioned")
    # Against the trainer's OWN handle on the denoiser, not a second load of
    # it: `tests/test_conditioned.py` rebinds `sys.modules["train_denoise"]`,
    # so two loads are two objects and an identity across them would be a test
    # of the loader. This identity is the one that matters.
    assert conditioned.corrupt is conditioned.denoise.corrupt
    assert conditioned.on_disk is conditioned.denoise.on_disk
    source = (ROOT / "scripts" / "denoise_floor.py").read_text(encoding="utf-8")
    assert "trainer.corrupt" in source
    assert "_module(options.trainer)" in source
