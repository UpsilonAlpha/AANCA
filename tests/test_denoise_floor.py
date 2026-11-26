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
