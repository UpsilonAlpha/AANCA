---
type: observation
status: supported
evidence: ours
sources: [32604037, regrowth_dispersion.py]
---

# Rolled past 120 steps the rule destroys the window it just built: 7.78 angstrom at the horizon, 16.5 at 600 steps, worse than doing nothing

`scripts/regrowth_dispersion.py` on `g20_orient`'s `best.pt`, 6 held-out crops
(`3B82 4CQE 1JNP 2FAZ 8QTF 6HJ6`), 8 members per amplitude, 600-step cap, CPU.
The measurement was built to find conformational freedom. It found something
else first.

    amp deg  start RMSF  settled RMSF  ratio  steps  converged  rmsd_dmg  start bond
          0       0.000         5.098    n/a    600     0/48        16.93        7.49
         15       0.601         5.139   8.56    600     0/48        16.35        7.53
         45       1.701         6.253   3.68    600     0/48        16.41        7.88

## Three readings, and the first one is the result

**1. The rule does not hold what it built.** `g20_orient`'s reported `settled`
is **7.78 A at step 120**, the step its loss was applied to
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]).
Rolled to 600 steps the same checkpoint on the same crops reads **16.4-16.9 A**
-- against a **12.23 A untrained baseline**. So past its training horizon the
rule does not merely drift, it drives the window **further from native than
leaving it alone would**.

**2. Nothing converges. 0 of 48**, at every amplitude, at a 0.02 A-per-chunk
tolerance over 30-step chunks. There is no fixed point and there is no bounded
basin either: the structure is still moving at 600 steps, away from where it
was.

**3. The initial condition is irrelevant and the mask is not.** Amplitude 0 --
members differing by update mask alone -- gives **5.098 A** of settled spread,
and amplitude 15 gives **5.139 A**, indistinguishable. Only 45 deg lifts it, to
6.25. So the ensemble spread this measurement was built to read is dominated by
the update schedule and by divergence, **not** by conformational multiplicity.

## What this does to the flexibility programme

**It blocks it.** `-log` of a distribution requires a distribution, and what is
here is a diverging trajectory: the spread at 600 steps measures how fast
members come apart, not how many ways the loop can sit
([[question-does-regrowth-dispersion-measure-conformational-freedom]],
[[destruction-and-regrowth-yields-a-local-statistical-potential]]). A Potts
model fitted to this would be a confident model of drift.

So the missing Growing-NCA ingredients are no longer the cheapest *improvement*
available -- they are the **precondition**. A sample pool and a supervision step
the rule cannot anticipate are what make persistence trainable, and persistence
is what the whole programme assumes
([[the-growth-objective-supervises-a-fixed-final-step-while-the-track-that-converged-samples-one]]).
E(n)-GNCA reports its attractor maintained **1,000 steps past** the training
horizon, and it trains with a replay pool; this project has never had one, and
this is what that costs.

## And it qualifies the result it was run on

g20's 7.78 A is a reading **at exactly the step the loss was applied to**, of a
trajectory that does not stay there. That makes it a waypoint rather than a
settled state, and the honest phrasing is *"the rule passes through 7.78 A at
step 120"* rather than *"the rule reaches 7.78 A"*. The number is not retracted
-- it is the same quantity every arm in the record reports, measured the same
way -- but every `settled` value in this project now has to be read as a
snapshot of a transient, because **nothing in the record was ever read anywhere
but at the horizon**.

This is the clearest vindication the "bounded basin, not a fixed point"
correction could have got, and it lands in the uncomfortable direction: there
is no basin to be bounded yet.

## Scope

  * **One checkpoint**, `g20_orient`'s `best.pt`, 6 of its 12 crops. `g17_long`
    has not been run through this, so whether a double-ceiling arm holds its
    structure better is unmeasured and is the obvious next comparison -- it
    costs nothing but CPU.
  * **`best.pt`, not `last.pt`.** Best-on-held-out-total, so a slightly
    different rule from the one that produced the final `history.json` row.
  * The perturbation is per-residue rigid about the window centroid, which
    strains the start: `bond` reads 7.5-7.9 before the rollout starts against a
    crystal's ~0. Amplitude 0 is unperturbed, and it shows the same divergence,
    so the strain is not what causes this.
  * **Not a statement about training.** It says the trained rule does not
    maintain its output past 120 steps. Whether that is fixable by a pool, by a
    sampled supervision step, or only by an architectural change is open.

## What this is NOT in tension with

A `refutes` edge to [[the-rule-keeps-whatever-conformation-it-is-given]] was
written here first and then removed, because it was a miscategorisation rather
than a tension the validator should carry. That claim was narrowed on 8 Nov to
**three whole-structure arms under clamped FAPE, over 8,000 updates, which did
not move at all.** This observation is about a *regeneration* arm whose trained
rule moves a great deal, at *inference*, past a horizon it was never trained
beyond. "Did training produce movement" and "does a rule that learned to move
hold still afterwards" are different questions, and both can be answered yes.

Recorded rather than left to the edge list, so the reasoning is visible instead
of an edge having quietly disappeared.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]

## Requires
- [[the-growth-objective-supervises-a-fixed-final-step-while-the-track-that-converged-samples-one]]
