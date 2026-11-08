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

**What settles it:** `g17_long` -- `g15_both` with `updates = 10000` and nothing
else changed. If it settles near 10.2, 2,000 updates was enough and the table
stands as read. If it reaches 8, the three-arm comparison was taken at a budget
where none of them had converged.

## Required by
- [[the-two-levers-together-beat-either-alone-and-never-give-the-gain-back]]
- [[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]]
