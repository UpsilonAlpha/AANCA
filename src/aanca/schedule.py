"""The learning-rate schedule, as a pure function of the update number.

Deliberately stateless. A scheduler object is one more thing to checkpoint and
restore, and one more thing to get wrong at a segment boundary; a function of
the update number is right by construction on every resume, including one that
lands in the middle of the schedule.

The drops are **absolute update numbers**, not fractions of ``--updates``, for
the same reason ``--updates`` is not pinned: extending a run must not
retroactively move the schedule the run already trained under.

Why there is a schedule at all: Ulban's Neural Cubes NCA reaches ~3e-3 loss at
4,000 iterations and 5e-5 to 1e-4 at 8,000, and the two orders of magnitude are
bought in the half of the run after its first decay. A flat curve under a
constant learning rate is the commonest false ceiling there is, and this
project read one at update 9,000 before noticing it had no schedule.
"""

from __future__ import annotations

from collections.abc import Sequence


def learning_rate(
    base: float, update: int, drops: Sequence[int], gamma: float
) -> float:
    """``base`` scaled by ``gamma`` once for every drop already reached.

    ``update`` is 1-based, matching the training loop, and a drop at ``n`` takes
    effect *on* update ``n``, so the first ``n - 1`` updates run at the old rate.
    Order and duplicates in ``drops`` do not matter: two drops at the same
    update apply twice, which is what a caller asking for that deserves.
    """
    return base * gamma ** sum(1 for drop in drops if update >= drop)
