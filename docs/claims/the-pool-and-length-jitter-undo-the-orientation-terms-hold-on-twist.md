---
type: observation
status: supported
evidence: ours
sources: [32904055, 32604034, 32604037]
---

# g21_pool's twist settles at 13.25 against g20's 11.24, so the pool and the length jitter give back what the orientation term had just won

`g21_pool` (32904055) read on 20 Nov, the first reading since the OnDemand
session came back. Completed, exit 0, 10,000 updates, 444.6 min on bun144.

## Its placement number is VOID, by its own condition and by exact precedent

The condition: *"VOID if the run does not complete, or if `twist` leaves the
9-12 band -- g20's orientation term is inherited and this arm is not the place
to retest it."* The run completed. The twist did not hold:

    settled twist (mean of the last two evaluations)      13.25
      held out at  9750 updates                           14.2
      held out at 10000 updates                           12.3
    held-out evaluations outside the band                 10 of 39
      all of them from update 5500 onward; max 14.2
    training-line twist                                   9.0 - 13.7

**So `settled` rmsd_dmg of 8.41 A is void and is not this arm's result.**

The precedent is exact rather than analogous. `g19_nomove` carried the same
clause word for word, settled at twist 14.26, and the record voided its
rmsd_dmg of 8.39 while keeping its twist reading
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]).
The same scope applies here: the **placement** claim is void, because that is
what the rationale was about, and the twist reading is a direct measurement of
a different quantity and stands.

For the record and not as a reading: 8.41 would have been 0.63 A worse than
g20's 7.78, inside the second bucket's 1.0 A tolerance. It is not available.

## The reason the void fired is the result

g21 **inherits `orientation_weight = 2.5`** from g20 and changes two things:
`pool_probability = 0.5` and `steps_min = 40`. Against the arms either side of
that term, settled twist:

    arm           what it had                      twist   settled rmsd_dmg
    baseline      --                                 9.6             12.23
    g17_long      no orientation term, 10k          14.3              6.69
    g19_nomove    no orientation term, 10k         14.26         8.39 (void)
    g20_orient    orientation 2.5, 10k             11.24              7.78
    g21_pool      orientation 2.5 + pool + jitter  13.25         8.41 (void)

g20's headline was that the term **holds orientation in the band that every
identity-free arm held before budget pushed g17 out of it**, with g19 as the
control that made it causal. g21 keeps the term at the same weight and lands
most of the way back at the no-term value. So the persistence protocol does not
merely fail to help twist: **it gives back what the orientation term was the
first thing in the project to win.**

This is attributable, because g21 differs from g20 in exactly those two
settings. It is not attributable to *which* of them, and that was a recorded
departure from this file's one-variable rule rather than an accident: *"a pool
without length jitter still only ever supervises step 120, and jitter without a
pool still only ever starts from the baseline."* Separating them is two slots.

A mechanism worth stating as a guess and not more: the pool's guard keeps an
end-state only if it is no worse than that crop's fresh baseline, and it ranks
by **accuracy**. Twist is not in that ranking, so a pool entry can be better on
placement and worse on orientation and still be kept, which selects for exactly
the trade the table shows. If that is right, the fix is the one the arm's own
third bucket already anticipated for a different reason -- the guard needs
replacing with something that does not rank states by accuracy alone.

## What is still not answered

**The primary bucket has not been read.** It is the 600-step reading from
`scripts/regrowth_dispersion.py` on this checkpoint, against g20's 16.4-16.9 A
and the 12.23 A untrained baseline, and nothing above is a substitute for it:
twist drifting says the checkpoint is in a different conformational regime from
g20's, which makes a persistence comparison **confounded, not void**. Worth
taking anyway, in both directions -- if persistence improved despite the drift
there is a trade to characterise, and if it did not, the arm's fourth bucket
fires and persistence is not a training-protocol problem.

`runs/g21_pool/best.pt` exists on scratch and the script defaults to
`--device cpu`, so this costs a CPU job and not a GPU slot.

So the second of the three questions that decide the central claim -- *does the
rule hold what it builds?* -- is **still open after this arm**, and the arm has
cost its placement number in the asking.

## Scope

  * Twist here is a **regime marker, not an error bar.** The 9-12 band is where
    identity-free arms sat; no arm has ever been replicated
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]), so
    13.25 against 11.24 is a difference in regime and not a significance test.
  * `move 0.0000` at every evaluation is expected, not a dead head: this chain
    runs `move_weight = 0.0` from `g19_nomove`, so the teacher-forcing term is
    off and reads zero by construction.
  * The arm's own two-variable departure is the limit on attribution, stated
    above.

## Supports
- [[the-rule-has-no-attractor-past-its-training-horizon]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Requires
- [[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]
