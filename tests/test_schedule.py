"""The learning-rate schedule.

Small enough to look correct and still worth testing, because every one of
these properties is relied on somewhere: the resume lands mid-schedule, the
chain extends a run past its last drop, and the fingerprint refuses a changed
schedule only if the schedule is a function of settings rather than of how
many times a scheduler object has been stepped.
"""

from __future__ import annotations

from aanca.schedule import learning_rate


def test_no_drops_is_a_constant_rate():
    assert all(learning_rate(3e-4, u, [], 0.3) == 3e-4 for u in (1, 10, 10**6))


def test_a_drop_takes_effect_on_its_own_update():
    """Off-by-one here would be invisible in a log and wrong in a comparison."""
    assert learning_rate(1.0, 99, [100], 0.3) == 1.0
    assert learning_rate(1.0, 100, [100], 0.3) == 0.3
    assert learning_rate(1.0, 101, [100], 0.3) == 0.3


def test_drops_compound():
    rate = learning_rate(2e-3, 8000, [4000, 6000], 0.3)
    assert abs(rate - 2e-3 * 0.09) < 1e-15


def test_the_rate_depends_only_on_the_update_number():
    """The property that makes a resume safe without restoring a scheduler.

    A run that reaches update 500 in one go and one that is stopped at 300 and
    resumed must be at the same rate, which is trivially true of a function
    and is exactly what a stepped scheduler gets wrong when its state is not
    checkpointed.
    """
    drops, base, gamma = [200, 400], 1.0, 0.5
    straight = [learning_rate(base, u, drops, gamma) for u in range(1, 501)]
    halves = [learning_rate(base, u, drops, gamma) for u in range(1, 301)]
    halves += [learning_rate(base, u, drops, gamma) for u in range(301, 501)]
    assert straight == halves


def test_order_and_duplicates_do_not_surprise():
    assert learning_rate(1.0, 500, [400, 100], 0.5) == learning_rate(
        1.0, 500, [100, 400], 0.5
    )
    assert learning_rate(1.0, 500, [100, 100], 0.5) == 0.25
