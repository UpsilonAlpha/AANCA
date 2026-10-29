"""Stopping a run at any point and continuing it as though nothing happened.

A run on a shared cluster is interrupted by default: the walltime ends, the job
is pre-empted, the queue is drained, the user cancels it. So the only question
is whether an interruption costs the run's *history* or merely its last few
minutes. Two things make the difference, and both are easy to get wrong.

**Every source of randomness has to be in the checkpoint, not just the
weights.** The structure drawn each update, the damage applied to it, the
asynchronous update mask, the crop, and the pool's eviction and re-seeding
choices are all random. Saving the weights alone gives a run that continues
from the right parameters along a different trajectory -- which is defensible
for a long run and indefensible for a diagnostic one, because the fixed-point
diagnostics (does ``|delta|`` decay, does a native start stay put) are read as
a curve over updates. A curve assembled from differently-seeded segments is not
a curve. So the Python RNG, the torch generator, the global torch RNG and the
pool's RNG all go in.

**The sample pool is state, not cache.** Dropping it would silently restart the
"does it stay" half of the objective: the rule would train on fresh damage only
until the pool refilled.

**A checkpoint must never be half-written.** The run is killed at an arbitrary
instant, including during the save, and a truncated file read back at resume
would lose everything rather than one interval. Writes go to a temporary file
beside the target and are renamed over it, which is atomic on POSIX, so the
file at the target path is always either the previous checkpoint or the new one.

What is *not* in here: the dataset, which is reconstructed from the manifest,
and the held-out cases, which are deterministic functions of their structure
indices (``evaluate.build_eval_set``) and so are rebuilt rather than stored.
The indices themselves are stored, because they come from a shuffle and a
load-success filter that a different machine need not reproduce.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch

#: Settings a resume may not silently differ on. The rest (``--updates``,
#: ``--eval-every``, ``--max-seconds``) are free to change: extending a run or
#: evaluating it more often does not invalidate what came before, while a
#: different crop size or rollout length makes the curve two curves.
#:
#: ``split``, ``structures`` and ``eval_structures`` are pinned for a sharper
#: reason than the others: the structure indices are *in* the checkpoint, so a
#: resume that asked for a different split or a bigger pool would restore the
#: old indices and carry on training on them while the log and the run
#: directory claimed otherwise. That is a wrong answer with no symptom, which
#: is worse than a refusal. ``cache`` is deliberately absent: it changes how
#: often an entry is re-parsed and nothing else.
#: ``lr``, ``lr_drop`` and ``lr_gamma`` are pinned because the schedule is a
#: function of the update number: a resume that changed any of them would
#: continue the same curve at a rate the earlier half never saw, and nothing
#: in the log would say so. ``steps_max`` joins ``steps`` for the same reason
#: the rollout length is pinned at all.
PINNED = (
    "crop_size", "steps", "steps_max", "truncate", "seed", "pool_probability", "k",
    "noop_weight", "lr", "lr_drop", "lr_gamma",
    "split", "structures", "eval_structures",
)

#: Exit code for "checkpointed and stopped early, not finished". The batch
#: script keys its resubmission off this, so a genuine crash (any other
#: non-zero code) stops the chain instead of resubmitting forever.
UNFINISHED = 64


def fingerprint(args: Any) -> dict:
    """The pinned subset of the command line, for comparison at resume."""
    values = vars(args) if not isinstance(args, dict) else args
    return {key: values[key] for key in PINNED if key in values}


def check_fingerprint(stored: dict, current: dict) -> None:
    """Refuse a resume that would splice two different experiments together."""
    changed = {
        key: (stored.get(key), current.get(key))
        for key in set(stored) | set(current)
        if stored.get(key) != current.get(key)
    }
    if changed:
        detail = ", ".join(
            f"{key}: checkpoint {was!r}, now {now!r}" for key, (was, now) in sorted(changed.items())
        )
        raise ValueError(
            "this checkpoint was written with different settings, so resuming "
            f"would join two different runs ({detail}). Use a new --tag, or "
            "pass --force-resume if the change really is immaterial."
        )


def rng_state() -> dict:
    """Every global random source this project draws from."""
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }


def set_rng_state(state: dict) -> None:
    random.setstate(_tuples(state["python"]))
    torch.set_rng_state(_as_byte_tensor(state["torch"]))


def _tuples(state):
    if isinstance(state, (list, tuple)):
        return tuple(_tuples(item) for item in state)
    return state


def _as_byte_tensor(state) -> torch.Tensor:
    tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state)
    return tensor.to(dtype=torch.uint8, device="cpu")


def restore_python_rng(rng: random.Random, state) -> None:
    """Put a ``random.Random`` back exactly where it was."""
    rng.setstate(_tuples(state))


def restore_generator(generator: torch.Generator, state) -> None:
    """Put a ``torch.Generator`` back exactly where it was."""
    generator.set_state(_as_byte_tensor(state))


def save(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically.

    The temporary file sits in the same directory so the rename stays within
    one filesystem; ``os.replace`` is then atomic, and a kill at any instant
    leaves either the old checkpoint or the new one at ``path``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(scratch, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, path)
    finally:
        if scratch.exists():
            scratch.unlink()


def load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    """Read a checkpoint. ``weights_only=False`` because RNG states are tuples."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)
