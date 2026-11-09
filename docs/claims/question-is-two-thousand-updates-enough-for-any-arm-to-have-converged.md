---
type: question
status: open
---

# Is 2,000 updates enough for any growth arm to have converged, or is the whole lever table a budget artefact?

Every growth arm in the record ran 2,000 updates at a **constant learning rate,
with no sample pool and a fixed 120-step rollout**. None of those three is an
oversight in the packing trainer -- `scripts/train.py` applies a learning-rate
decay per update and samples from a pool -- they are simply absent from
`scripts/train_growth.py`, whose own docstring says so: *"the cheapest thing
that can: one structure, one rollout, one backward pass per update, no sample
pool."*

The closest prior art available is Ulban's own 3D NCA project, and its recorded
lesson (`docs/PLAN_V1.md`, the neural-cubes section) is that **"it will not stay
put" is answered by longer rollouts, a working pool and a learning-rate decay,
not by a stronger penalty term.** The same document already warned once, in
different words, that *"nothing here has yet been trained under conditions the
prior art says are necessary, so a plateau is not evidence about the
objective."*

Three facts make this gating rather than theoretical:

  * g13 reaches its best at update **250** and gives 0.84 A back; g14 reaches
    its best at 1250 and gives 1.25 A back. "It finds a position it cannot
    keep" is exactly the failure the prior art answers with budget and pool.
  * g15's best reading is at update **1750 -- its second-to-last evaluation**,
    i.e. still improving when the budget ended.
  * g15 used 37.4 min of a 24 h slot.

Until an arm has been run long enough to flatten, every bucket boundary in the
lever table is a comparison of three unconverged curves, and "superadditive"
could be a statement about which arm converges fastest rather than which
converges lowest.

## Pre-registration: `g17_long`, submitted as 32505354

`g15_both` with `updates = 10000`, one setting, `eval_every` held at 250 so
`settled` is read with the same instrument the lever table was read with.
Running on `bun159` from 2026-11-09 08:20 AEST, 8 h wall.

**Written at 08:24, four minutes in.** The early evaluations had almost
certainly landed by then -- g15 did 2,000 updates in 37 min, so update 250 is
about five minutes -- and they have not been read. The quantity being judged is
`settled` = the mean of `rmsd_dmg` at updates **9750 and 10000**, roughly three
hours out, so the prediction is blind with respect to the thing it predicts.
Said plainly rather than claimed as a clean pre-registration.

Judged against **g15's settled 10.24** over four exclusive buckets:

  * **`settled <= 9.57`** -- better than g15 by at least the g14-to-g15 gap
    (0.67 A). The 2,000-update table was read before any arm had converged, so
    "superadditive" is a statement about which arm converges *fastest*, and
    every bucket boundary in the record needs re-reading at this budget.
    **Budget, not levers, is then the project's live variable.**
  * **`9.57 < settled <= 9.90`** -- better than g15's best *single* reading
    (9.94), but by less than one lever's worth. The lever table's ordering
    stands; its absolute numbers do not, and future arms must be run at 10,000.
  * **`9.90 < settled <= 10.58`** -- inside g15's +-0.34 A band. **2,000 updates
    was already enough.** The table stands exactly as read, this question
    closes, and `g16_localrot` becomes the right next arm with a comparator
    that has been shown to be converged.
  * **`settled > 10.58`** -- longer training makes it *worse*. With 400
    structures, no sample pool and a fixed rollout, that reads as overfitting
    the pool rather than a budget ceiling, and the indicated next change is the
    **pool**, not more updates.

**Void condition.** If the run does not complete 10,000 updates -- Bunya goes
into maintenance today and no SLURM reservation exists, so a drain may kill it
mid-flight -- or if more than 5 % of updates are skipped for a non-finite
gradient, `settled` is not read. The curve is reported and the arm is resubmitted
with `--resume` instead. Without this the metric would otherwise be chosen from
whatever portion of the run survived.

**What it cannot settle.** `g17_long` is still a double-ceiling arm
(`--native-edges` on top of `--move-weight 1`), so none of the four buckets
bears on whether a *learned* rule can do this. It bounds the architecture at a
budget, which is the only thing the lever table ever did.

## Required by
- [[the-two-levers-together-beat-either-alone-and-never-give-the-gain-back]]
- [[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]]

## Answered, and reopened one order of magnitude up

`g17_long` settled at **6.69** against g15's 10.24 -- the first bucket, by five
times its boundary ([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]]).
So **no number in the lever table is a converged number**: g13, g14 and g15
were three unconverged curves compared at update 2,000, and "superadditive"
describes which arm fell fastest in its first 2,000 updates.

This does not retract the lever comparison -- g15 really is the best of the
three at 2,000 updates on the same 12 crops -- but it retires it as a guide to
what to do next. The question stays `open` rather than closing, because g17 is
not converged either: still drifting down from 8.44 at update 4250 and reading
6.62 / 6.75 at its last two evaluations. Budget is the project's live variable
and its ceiling has not been found.
