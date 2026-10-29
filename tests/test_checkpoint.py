"""Resume has to be bit-for-bit, or the progress curve is several curves.

The test that matters here is the last one: a run stopped and resumed must end
in the same weights as a run that was never interrupted. Everything else is a
component of that claim, written separately so a failure says which piece
broke.
"""

from __future__ import annotations

import random

import pytest
import torch

from aanca import checkpoint as ckpt
from aanca.model import AANCARule
from aanca.rollout import RolloutState, SamplePool


def test_save_is_atomic_and_round_trips(tmp_path):
    path = tmp_path / "last.pt"
    ckpt.save(path, {"update": 7, "x": torch.arange(3)})
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp*")), "temporary file left behind"
    state = ckpt.load(path)
    assert state["update"] == 7
    assert torch.equal(state["x"], torch.arange(3))


def test_save_leaves_the_previous_checkpoint_intact_on_failure(tmp_path):
    """A kill during the write must cost the new checkpoint, never the old one."""
    path = tmp_path / "last.pt"
    ckpt.save(path, {"update": 1})

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("killed mid-save")

    with pytest.raises(RuntimeError):
        ckpt.save(path, {"update": 2, "bad": Unpicklable()})
    assert ckpt.load(path)["update"] == 1
    assert not list(tmp_path.glob("*.tmp*"))


def test_fingerprint_refuses_a_changed_rollout_length():
    stored = {"crop_size": 384, "steps": 20, "truncate": 8, "seed": 0}
    with pytest.raises(ValueError, match="different settings"):
        ckpt.check_fingerprint(stored, {**stored, "steps": 30})


def test_fingerprint_refuses_a_changed_split():
    """The indices are in the checkpoint, so this change would be invisible."""
    import argparse

    args = argparse.Namespace(crop_size=384, steps=20, seed=0, split="validation")
    moved = argparse.Namespace(**{**vars(args), "split": "train"})
    with pytest.raises(ValueError, match="different settings"):
        ckpt.check_fingerprint(ckpt.fingerprint(args), ckpt.fingerprint(moved))


def test_fingerprint_allows_extending_the_run():
    """``--updates`` is deliberately not pinned: a longer run is the same run."""
    import argparse

    args = argparse.Namespace(
        crop_size=384, steps=20, truncate=8, seed=0, pool_probability=0.5, k=64,
        updates=100,
    )
    longer = argparse.Namespace(**{**vars(args), "updates": 5000})
    ckpt.check_fingerprint(ckpt.fingerprint(args), ckpt.fingerprint(longer))


def test_a_newly_pinned_setting_does_not_strand_a_running_chain():
    """Pinning a setting must not break the checkpoints already written.

    A key missing from the stored fingerprint describes no choice the old run
    made. At the new setting's default the experiment is unchanged, so the
    resume stands; the alternative is a chain that fails at a segment
    boundary hours after the commit that caused it.
    """
    stored = {"crop_size": 384, "steps": 20, "seed": 0}
    defaults = {**stored, "native_probability": 0.0}
    ckpt.check_fingerprint(stored, {**stored, "native_probability": 0.0}, defaults)


def test_a_newly_pinned_setting_that_was_actually_set_is_still_refused():
    """The leniency is for the setting's absence, not for a changed run."""
    stored = {"crop_size": 384, "steps": 20, "seed": 0}
    defaults = {**stored, "native_probability": 0.0}
    with pytest.raises(ValueError, match="native_probability"):
        ckpt.check_fingerprint(
            stored, {**stored, "native_probability": 0.2}, defaults
        )


def test_python_rng_restores_exactly():
    rng = random.Random(3)
    [rng.random() for _ in range(5)]
    state = rng.getstate()
    expected = [rng.random() for _ in range(4)]

    other = random.Random(99)
    ckpt.restore_python_rng(other, _through_torch(state))
    assert [other.random() for _ in range(4)] == expected


def test_torch_generator_restores_exactly():
    generator = torch.Generator().manual_seed(11)
    torch.rand(5, generator=generator)
    state = generator.get_state()
    expected = torch.rand(4, generator=generator)

    other = torch.Generator().manual_seed(1)
    ckpt.restore_generator(other, _through_torch(state))
    assert torch.equal(torch.rand(4, generator=other), expected)


def test_global_rng_restores_exactly():
    torch.manual_seed(5)
    torch.rand(3)
    state = ckpt.rng_state()
    expected = torch.rand(3)
    torch.manual_seed(12345)
    ckpt.set_rng_state(_through_torch(state))
    assert torch.equal(torch.rand(3), expected)


def test_pool_round_trips_with_its_rng(tmp_path):
    """Including the pool's RNG: without it a resumed run re-seeds from
    different stored states, which is the 'does it stay' half of the objective
    quietly changing."""
    pool = SamplePool(capacity=4, seed=7)
    for i in range(6):
        pool.insert(
            i % 2,
            RolloutState(coords=torch.full((2, 3), float(i)), hidden=torch.zeros(2, 4)),
            crop_seed=i,
        )
    # Saved *before* the draws, since the draws advance the pool's own RNG and
    # the question is whether a resume continues the same sequence.
    ckpt.save(tmp_path / "p.pt", pool.state_dict())
    expected = [pool.sample(0)[1].coords[0, 0].item() for _ in range(5)]
    restored = SamplePool(capacity=4, seed=0)
    restored.load_state_dict(ckpt.load(tmp_path / "p.pt"))
    assert len(restored) == len(pool)
    assert [restored.sample(0)[1].coords[0, 0].item() for _ in range(5)] == expected
    assert restored.sample(0)[0] in (0, 2, 4), "the crop seed survives the round trip"


def test_pool_refuses_a_changed_capacity():
    pool = SamplePool(capacity=4, seed=0)
    with pytest.raises(ValueError, match="capacity changed"):
        SamplePool(capacity=8, seed=0).load_state_dict(pool.state_dict())


def test_unfinished_code_is_distinct_from_success_and_from_one():
    """The batch script resubmits on this code alone, so it must not collide
    with a crash (1) or with success (0)."""
    assert ckpt.UNFINISHED not in (0, 1)


def test_a_stopped_and_resumed_run_matches_an_uninterrupted_one(tmp_path):
    """The whole point, exercised on a surrogate training loop.

    The loop is deliberately not the real one -- that needs structures on disk
    and minutes per step -- but it draws from every random source the real loop
    does, in the same order, and it is the *bookkeeping* that resume gets wrong.
    """

    def steps(rule, optimiser, pool, rng, generator, count, start=0):
        for i in range(start, start + count):
            index = rng.choice([0, 1, 2])
            noise = torch.rand(4, generator=generator)
            pooled = pool.sample(index)
            seed = pooled[1].coords.sum() if pooled is not None else torch.zeros(())
            weight = next(iter(rule.parameters()))
            loss = ((weight * noise.sum()) ** 2).sum() + seed * 0.0
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            pool.insert(
                index,
                RolloutState(coords=torch.rand(2, 3), hidden=torch.zeros(2, 2)),
                crop_seed=index * 7,
            )

    def fresh():
        torch.manual_seed(0)
        rule = AANCARule()
        return (
            rule,
            torch.optim.Adam(rule.parameters(), lr=1e-3),
            SamplePool(capacity=8, seed=0),
            random.Random(0),
            torch.Generator().manual_seed(0),
        )

    straight = fresh()
    steps(*straight, count=12)

    halted = fresh()
    steps(*halted, count=5)
    rule, optimiser, pool, rng, generator = halted
    ckpt.save(
        tmp_path / "last.pt",
        {
            "update": 5,
            "rule": rule.state_dict(),
            "optimiser": optimiser.state_dict(),
            "pool": pool.state_dict(),
            "rng_python": rng.getstate(),
            "rng_generator": generator.get_state(),
            "rng_global": ckpt.rng_state(),
        },
    )

    state = ckpt.load(tmp_path / "last.pt")
    torch.manual_seed(99999)  # a fresh process would start somewhere else
    resumed = (
        AANCARule(),
        None,
        SamplePool(capacity=8, seed=123),
        random.Random(123),
        torch.Generator().manual_seed(123),
    )
    rule2, _, pool2, rng2, generator2 = resumed
    optimiser2 = torch.optim.Adam(rule2.parameters(), lr=1e-3)
    rule2.load_state_dict(state["rule"])
    optimiser2.load_state_dict(state["optimiser"])
    pool2.load_state_dict(state["pool"])
    ckpt.restore_python_rng(rng2, state["rng_python"])
    ckpt.restore_generator(generator2, state["rng_generator"])
    ckpt.set_rng_state(state["rng_global"])
    steps(rule2, optimiser2, pool2, rng2, generator2, count=7, start=5)

    for a, b in zip(straight[0].parameters(), rule2.parameters()):
        assert torch.equal(a, b), "resumed weights diverged from the uninterrupted run"


def _through_torch(state):
    """Round-trip through ``torch.save`` so the test sees what a resume sees."""
    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)
