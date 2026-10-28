"""End-to-end: stopping the real training script and resuming it is exact.

``tests/test_checkpoint.py`` proves the pieces on a surrogate loop. This runs
the actual driver twice -- once straight through, once in two halves -- and
compares the weights, which is the only version of the claim that covers the
script's own bookkeeping (what order it restores things in, what it consumes
between restoring the RNGs and the first update).

**Single-threaded on purpose.** Measured on this machine: two *identical*
invocations of the training script differ by ~6e-8 in the weights with torch's
default thread count, because multithreaded reductions sum in whatever order
the threads finish; at one thread both the reruns and the resume are
bit-identical. So resume is compared under the one condition where an inequality
can only mean a bookkeeping bug.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
TRAIN = ROOT / "scripts" / "train.py"

SMALL = [
    "--eval-structures", "2",
    "--structures", "4",
    "--steps", "2",
    "--eval-steps", "2",
    "--truncate", "1",
    "--crop-size", "96",
    "--eval-every", "100",
    "--checkpoint-every", "2",
    "--threads", "1",
]


@pytest.fixture(autouse=True)
def clean_runs():
    """Clear the tags first: a ``last.pt`` left by an earlier session would
    turn the "straight through" run into a resume and the test would pass for
    the wrong reason."""
    for tag in ("pytest_straight", "pytest_halves", "pytest_pinned"):
        shutil.rmtree(ROOT / "runs" / tag, ignore_errors=True)
    yield


def run(tag: str, updates: int, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRAIN), "--tag", tag, "--updates", str(updates),
         *SMALL, *extra],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )


def weights(tag: str) -> dict:
    state = torch.load(
        ROOT / "runs" / tag / "last.pt", map_location="cpu", weights_only=False
    )
    return state["rule"], state["update"]


@pytest.mark.needs_data
@pytest.mark.slow
def test_two_halves_match_one_straight_run():
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    straight = run("pytest_straight", 4)
    assert straight.returncode == 0, straight.stdout[-3000:] + straight.stderr[-3000:]

    first = run("pytest_halves", 2)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    second = run("pytest_halves", 4)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "resumed" in second.stdout

    straight_weights, straight_update = weights("pytest_straight")
    resumed_weights, resumed_update = weights("pytest_halves")
    assert straight_update == resumed_update == 4
    for key, value in straight_weights.items():
        assert torch.equal(value, resumed_weights[key]), f"{key} diverged across a resume"


@pytest.mark.needs_data
@pytest.mark.slow
def test_a_resume_with_changed_rollout_length_is_refused():
    """The pinned-settings check, through the script rather than the module:
    silently joining a 2-step run to a 5-step one would make the progress curve
    two curves pretending to be one."""
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    first = run("pytest_pinned", 2)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    # ``--steps`` appears twice; argparse takes the last, so this overrides it.
    changed = run("pytest_pinned", 4, "--steps", "5")
    assert changed.returncode != 0
    assert "different settings" in (changed.stdout + changed.stderr)
