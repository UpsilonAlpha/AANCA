---
type: question
status: open
---

# Both previous capacity tests were confounded, and the cell that settles whether the architecture can be taught to aim has never been run

**Pre-registration, written before `g28_ceil_local` was submitted.** The whole
value of this arm is that its thresholds exist before its number does. The
project has twice read a capacity result off a confounded arm and twice drawn
an architectural conclusion from it, so the buckets below are binding.

## The question this settles

[[the-architecture-can-place-residues-under-some-supervision]] is the open
question the thesis `requires`, and it gates everything: a failure at any lower
rung is uninterpretable until it is answered. Its top rung is teacher forcing --
`move_weight = 1.0` supervises every emitted move against the exactly computable
ideal one, so the rule is **handed the answer at every step**. If it cannot
learn to aim under that, no objective will teach it.

It is a CEILING. No number from this arm is ever a result about regeneration.

## Why the two existing attempts do not answer it

    arm             actuator                          identity   twist 250 -> 2000
    g8_ceiling      lab translation,  lab rotation    on         9.1 -> 16.6
    g10_equiv_ceil  LOCAL translation, LOCAL rotation on         9.9 -> 50.2
    g9_ceil_noid    (no equivariance fix at all)      OFF        9.1 -> 9.2
    g28_ceil_local  LOCAL translation, LOCAL rotation OFF        <- never run

  * **`g8_ceiling` had neither half of the fix.** The move was applied in the
    lab frame while perception is invariant by construction, so the direction
    was never representable
    ([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]).
    Its failure measures that, not the architecture.

  * **`g10_equiv_ceil` had a sound actuator**, and this note exists partly to
    correct the record: it is tabulated as *"local translation, local rotation"*
    in [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]],
    so **both** halves were local. HANDOFF said for part of 23 Nov that no
    ceiling arm had ever had both fixes. Wrong, and corrected there.

    What confounds `g10` is **when its bearing was read**. Twist runs 9.9 at
    update 250, **52.3 at 1000**, 50.2 at 2000, and the align -0.096 against a
    +0.206 null ([[the-equivariance-fix-did-not-release-a-bearing]]) was taken
    off the **final** checkpoint -- a rule whose geometry had already broken by
    a factor of five. That reading is sound as a reading; it is just not a
    reading of a working rule.

  * **The break belongs to identity, not to equivariance.** `g8_ceiling` has
    neither fix and breaks in the same row, which kills the two-arm attribution
    that blamed the conjugated rotation. `g9_ceil_noid` at `identity_weight = 0`
    holds twist 9.1-9.2 for the full 2,000 updates. So identity off is what
    keeps a ceiling arm alive long enough to be read.

**`g10` had the actuator without the quiet, `g9_ceil_noid` had the quiet without
the actuator, and nothing has had both.** That is the gap. `base` already
carries `move_weight = 1.0` and `identity_weight = 0.0`, so the arm is one
variable from it: `lab_rotation = false`.

## Pre-registered buckets, on `align` at the LAST READABLE checkpoint

`align` is the cosine between the emitted move and the ideal one, from
`scripts/ceiling_readout.py`, over the twelve held-out crops. Reference points
already in the record:

    a handed gradient, chemotaxis task      0.96   <- the architecture aiming
    convention-matched null / untrained     +0.206
    a fixed random vector                   -0.4 to +0.25
    g10_equiv_ceil, final (broken) ckpt     -0.096

  * **align >= 0.50 -- CAPACITY EXISTS, decisively.** The architecture can be
    taught to aim when the actuator works and nothing is breaking it. Every
    rung-1 failure in the record is then about the **objective**, the project's
    negative result is a result about this loss and not about neural cellular
    automata, and [[the-architecture-can-place-residues-under-some-supervision]]
    closes YES. The next question becomes how little supervision suffices, which
    is rungs 2-4, and the morphogen bearing hypothesis
    ([[a-residue-can-follow-the-morphogen-gradient]]) becomes the thing to test.

  * **align 0.25-0.50 -- partial, direction only.** Clear of the +0.206 null but
    far short of the 0.96 the same architecture reaches on a handed gradient.
    Report the direction and **say** that the magnitude is not a result: one
    seed, no arm in this project has been run twice
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).
    Does not close the question either way.

  * **align inside the null band, -0.10 to +0.25 -- THE LIMITATION IS
    ARCHITECTURAL.** Teacher forcing with a working actuator, identity off and
    no break, for 2,000 updates, and the rule still emits a bearing
    indistinguishable from an untrained head. Then invariant perception plus a
    local move head **cannot be taught to aim even when handed the answer**,
    this closes the capacity question NO, and the project's negative result
    becomes a statement about this class of model. **This is the most
    informative outcome and the one that should end the arm chain rather than
    extend it.** Do not then add terms and re-run; that clause is inherited from
    bucket 0 and applies here.

  * **align < -0.10 -- SUSPECT THE INSTRUMENT, NOT THE RULE.** Systematically
    anti-aimed is not a thing a teacher-forced head should be. Re-check the
    readout's frame convention before reading anything into it: that exact class
    of error has already cost this project a retraction
    ([[a-probe-pointed-at-best-pt-silently-used-its-own-defaults]], and the
    32423296 retraction of an intermediate checkpoint read as final). Publish
    nothing from this bucket until the convention is re-derived.

## The guard bucket, which decides WHICH checkpoint is readable

This is the whole lesson of `g10` and it is pre-registered so it cannot be
forgotten under a result:

  * **Twist must hold in the project's 9-12 degree band through update 2,000**,
    as `g9_ceil_noid` did at 9.1-9.2. **Oscillation is not a breach**:
    `g23_localrot` oscillated 10.2-12.8 and was read as inside its band, and
    that precedent is adopted here rather than invented after the fact. What
    this guard is for is the `g10` failure mode, which was not an oscillation.
  * **If twist jumps by more than 2x at any evaluation, the final checkpoint is
    NOT readable.** Read `align` off the last checkpoint before the jump, say so
    explicitly and in the same breath as the number, and treat the arm as
    truncated at that update rather than as a 2,000-update arm.
  * **A twist regime marker does not void this arm's `align`.** Four
    consecutive arms have had a *placement* number voided on twist, because a
    regime marker invalidates a cross-regime placement comparison. `align` is
    not a placement comparison: it is the cosine between what this rule emitted
    and the ideal move for this rule's own state. The void clause does not
    reach it, and this is stated now so it is not stretched either way later.
  * **If it jumps with `identity_weight = 0`, the identity attribution is
    wrong** and that is a finding of its own, against
    [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]].
    It would mean something else breaks these arms at update 1000 and the
    mechanism is unknown. Do not read capacity off such an arm at all.

## What this arm cannot settle, stated now so it is not claimed later

  * **It says nothing about regeneration.** Teacher forcing hands over the
    answer, so `rmsd_dmg` here is a ceiling and is comparable only to `g8`'s
    and `g10`'s, never to the rung-1 arms or to the 4.0 A bar.
  * **One seed.** Every caveat in
    [[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]
    applies, which is why the buckets are wide and sign-based rather than
    margins.
  * **The context is FREE in this arm**, because it is one variable from `base`
    and `base` has `freeze_context = false`. So it is not comparable with
    `g24`/`g25` on any whole-structure quantity, and the context drift that
    [[freezing-the-context-improves-placement-twist-and-whole-structure-rmsd-together]]
    measured is present here. That does not touch `align`, which is the cosine
    between the emitted move and the ideal move for this rule's own state, but
    it does mean `rmsd_dmg` and `rmsd` from this arm are comparable only to
    `g8`'s and `g10`'s, which also ran the context free.

  * **A NO does not establish that no local rule could aim**, only that this
    one cannot be taught to under the strongest supervision available. The
    bearing may have to be **constructed** rather than perceived -- by
    diffusion, more hops, or iterative registration -- and that question stays
    open either way. See
    [[in-regeneration-the-surviving-context-is-the-grid]].

  * **"One variable" is against `base`, NOT against either prior ceiling arm.**
    Verified by resolving both through `config.load` -> `check_base_is_complete`
    -> `resolve` and diffing key by key: exactly one setting differs,
    `lab_rotation: True -> False`, and `tests/test_config.py` passes 25. But
    `g10_equiv_ceil` also carried `identity_weight` above zero and sat on an
    older generation of the base, so **g28 is not one variable from g10**. The
    table above is a map of an unoccupied cell, not a two-arm comparison, and a
    difference between g28 and g10 may not be attributed to `lab_rotation`
    alone. (`inherits = "base"` is deliberately absent: `resolve` refuses
    `base` as a section name and stops its walk there, so a base-rooted arm
    states no parent -- the same spelling `g14_pairwise` uses.)

  * **`truncate = 12` bounds a NO, and the record already says by how much.**
    This arm inherits the 27 % gradient window
    ([[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]),
    so a null here is strictly a null for *"cannot be taught to aim with BPTT
    spanning 12 of 120 steps"*. That node made the relevant prediction before
    this arm existed and it is adopted rather than re-derived: teacher forcing
    *"teaches a policy -- move toward where you belong -- which then applies at
    all 120 steps at inference. Truncation should therefore not limit
    `g8_ceiling` the way it limits a run learning from the outcome."* Each
    step's move is supervised directly at that step, so no gradient needs to
    cross the window. **Raising `truncate` is not the escape hatch** if this
    arm reads NO: `g26_fullbptt` measured `truncate = 120` as *harmful*, 1.68 A
    worse and below the geometric floor. Appealing to the window after a NO
    would be contradicted by both nodes at once.

## Supports
- [[the-architecture-can-place-residues-under-some-supervision]]
