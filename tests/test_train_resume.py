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
    for tag in (
        "pytest_straight", "pytest_halves", "pytest_pinned",
        "pytest_sched", "pytest_sched_halves", "pytest_warm", "pytest_warm_bad",
        "pytest_native", "pytest_native_halves",
    ):
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
def test_a_schedule_and_a_variable_rollout_survive_a_resume():
    """The two settings added for v4, under the one test that can catch them.

    Both are new ways for a segment boundary to go wrong: the learning rate
    now depends on the update number, and the rollout length now consumes a
    draw from the run's RNG. Either would show up here as weights that differ
    after a stop and restart, and nowhere else until a two-day run had already
    produced a curve with a seam in it.
    """
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    varying = ("--lr-drop", "3", "--lr-gamma", "0.5", "--steps-max", "4")
    straight = run("pytest_sched", 4, *varying)
    assert straight.returncode == 0, straight.stdout[-3000:] + straight.stderr[-3000:]

    first = run("pytest_sched_halves", 2, *varying)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    # Stopping at 2 and resuming crosses the drop at update 3.
    second = run("pytest_sched_halves", 4, *varying)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]

    straight_weights, _ = weights("pytest_sched")
    resumed_weights, _ = weights("pytest_sched_halves")
    for key, value in straight_weights.items():
        assert torch.equal(value, resumed_weights[key]), f"{key} diverged across a resume"


@pytest.mark.needs_data
@pytest.mark.slow
def test_native_starts_are_counted_and_survive_a_resume():
    """``--native-probability`` is how the fixed point is supervised, and it
    consumes a draw from the run's RNG on every update. A segment boundary
    that lost the draw, or the tally, would show up as a seam in the curve
    and nowhere else."""
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    native = ("--native-probability", "1.0", "--noop-weight", "0",
              "--pool-probability", "0")
    straight = run("pytest_native", 4, *native)
    assert straight.returncode == 0, straight.stdout[-3000:] + straight.stderr[-3000:]

    first = run("pytest_native_halves", 2, *native)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    second = run("pytest_native_halves", 4, *native)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]

    whole = torch.load(ROOT / "runs" / "pytest_native" / "last.pt",
                       map_location="cpu", weights_only=False)
    halves = torch.load(ROOT / "runs" / "pytest_native_halves" / "last.pt",
                        map_location="cpu", weights_only=False)
    # Every update takes a native start, and the tally is the run's, not the
    # segment's.
    assert whole["native_starts"] == 4
    assert halves["native_starts"] == 4
    for key, value in whole["rule"].items():
        assert torch.equal(value, halves["rule"][key]), f"{key} diverged across a resume"


@pytest.mark.needs_data
@pytest.mark.slow
def test_a_run_without_native_starts_draws_no_extra_randomness():
    """At probability 0 the draw is skipped entirely, so a run started before
    the setting existed resumes bit-identically. Without the short circuit the
    two runs below would differ from the first update."""
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    plain = run("pytest_native", 2)
    assert plain.returncode == 0, plain.stdout[-3000:] + plain.stderr[-3000:]
    weights_plain, _ = weights("pytest_native")

    shutil.rmtree(ROOT / "runs" / "pytest_native", ignore_errors=True)
    explicit = run("pytest_native", 2, "--native-probability", "0")
    assert explicit.returncode == 0, explicit.stdout[-3000:] + explicit.stderr[-3000:]
    weights_explicit, _ = weights("pytest_native")

    for key, value in weights_plain.items():
        assert torch.equal(value, weights_explicit[key]), f"{key} moved"


@pytest.mark.needs_data
@pytest.mark.slow
def test_a_warm_start_keeps_the_training_and_begins_a_new_curve():
    """``--init-from`` is how a pinned setting changes without discarding work.

    What it must do: carry the weights and the pool over, and still start at
    update 0 under the new settings. What it must not do is look like a
    resume, which would hide a changed rollout length inside one curve.
    """
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    source = run("pytest_pinned", 2)
    assert source.returncode == 0, source.stdout[-3000:] + source.stderr[-3000:]
    before, before_update = weights("pytest_pinned")
    assert before_update == 2

    checkpoint = ROOT / "runs" / "pytest_pinned" / "last.pt"
    # A rollout length the fingerprint would refuse on a resume.
    warm = run("pytest_warm", 2, "--steps", "5", "--init-from", str(checkpoint))
    assert warm.returncode == 0, warm.stdout[-3000:] + warm.stderr[-3000:]
    assert "warm start" in warm.stdout

    state = torch.load(
        ROOT / "runs" / "pytest_warm" / "last.pt", map_location="cpu", weights_only=False
    )
    assert state["update"] == 2, "a warm start counts its own updates from zero"
    assert state["init_from"]["update"] == 2, "provenance is not recorded"
    assert state["fingerprint"]["steps"] == 5, "the new setting did not take"
    assert len(state["pool"]["entries"]) > 0, "the pool was not carried over"


@pytest.mark.needs_data
@pytest.mark.slow
def test_a_warm_start_over_different_structures_is_refused():
    """The pool and the split come with the weights, so the structures must
    match. A bigger ``--structures`` would restore the old split and train on
    it while the log claimed otherwise -- the crop/pool failure again, one
    level up."""
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    source = run("pytest_pinned", 2)
    assert source.returncode == 0, source.stdout[-3000:] + source.stderr[-3000:]
    checkpoint = ROOT / "runs" / "pytest_pinned" / "last.pt"

    bad = run(
        "pytest_warm_bad", 2, "--structures", "6", "--init-from", str(checkpoint)
    )
    assert bad.returncode != 0
    assert "same structures" in (bad.stdout + bad.stderr)


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
