"""The fold-class readout script, which is the only place readout 0a exists."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


readouts = _module("cond_readouts")


def test_it_repairs_bare_nan_and_counts_it_as_a_regression(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('[{"update": 1, "rmsd": NaN}]')
    rows, hits = readouts.read_history(path)
    assert hits == 1
    assert rows[0]["rmsd"] is None


def test_null_is_not_counted_as_a_regression(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([{"update": 1, "rmsd": None}]))
    rows, hits = readouts.read_history(path)
    assert hits == 0
    assert rows[0]["rmsd"] is None


def test_single_valued_ignores_nan_and_rounds():
    rows = [{"sigma": 8.8990001}, {"sigma": float("nan")}, {"sigma": 8.899}]
    assert readouts.single_valued(rows, "sigma") == [8.899]


def test_block_means_put_the_last_update_of_a_block_in_that_block():
    rows = [{"update": u, "rmsd": float(u)} for u in (1, 2000, 2001, 4000)]
    blocks = readouts.block_means(rows, "rmsd", 2000)
    assert [b for b, _m, _n in blocks] == [2000, 4000]
    assert blocks[0][2] == 2 and blocks[1][2] == 2


def test_offsets_that_do_not_span_the_coordinates_are_refused():
    flat = np.zeros((6, 3))
    with pytest.raises(ValueError):
        readouts.split_by_offsets(flat, np.array([0, 3]))


def test_identical_paired_samples_report_an_inert_label():
    trace = np.arange(9, dtype=np.float64).reshape(3, 3)
    traces = [trace.copy() for _ in range(4)]
    out = readouts.paired_displacement(traces, np.arange(4), 4)
    assert out["mean"] == 0.0
    assert out["n_pairs"] == 6.0


def test_a_live_label_is_the_mean_per_residue_distance():
    base = np.zeros((4, 3))
    moved = np.zeros((4, 3))
    moved[:, 0] = 2.0
    traces = [base, moved, base.copy(), moved.copy()]
    out = readouts.paired_displacement(traces, np.array([0, 1, 2, 3]), 4)
    assert out["alpha_beta"] == pytest.approx(2.0)
    assert out["n_alpha_beta"] == 1.0
    assert out["max"] == pytest.approx(2.0)
