---
type: observation
status: supported
evidence: ours
sources: [33308293, 33313562, 33313576]
---

# Handed the exact answer at every step for 2,000 updates, the rule emits a bearing of zero: align +0.036, CI [-0.043, +0.114]

**The pre-registered capacity test, read against buckets written and pushed
before the number existed**
([[the-capacity-test-has-never-been-run-with-a-working-actuator-and-no-identity-break]]).
`g28_ceil_local`, job 33308293, the unoccupied cell: teacher forcing
(`move_weight = 1`), a local rotation (`lab_rotation = false`), identity off
(`identity 0`, read off the run's own weights line rather than inferred from
the config), 2,000 updates, `train exit : 0`, 72.2 min.

## The guard first, because it decides which checkpoint may be read

    update     250     500     750    1000    1250    1500    1750    2000
    twist     13.9    22.1    13.2    15.3    12.3    17.2    13.3    18.9
    ratio             1.59    0.60    1.16    0.80    1.40    0.77    1.42

  * **No evaluation jumped by 2x or more.** The largest step is 1.59x. By the
    pre-registered rule the final checkpoint is therefore **readable**, and
    both readout jobs confirm they read `update: 2000`, written 15:10:04 --
    the same "is this actually the final checkpoint" check whose failure
    caused the 32423296 retraction.
  * **But twist never entered the 9-12 degree band, at any evaluation.** The
    minimum is 12.3. It oscillates 12.3-22.1, a far wider excursion than the
    `g23_localrot` precedent (10.2-12.8) the pre-registration adopted for
    "oscillation is not a breach". **So the band condition the
    pre-registration required was not met**, and that case was not one the
    buckets assigned an action to. It is recorded as unmet rather than
    reinterpreted as met.
  * **It does not run away, which is what distinguishes it from `g10`.** After
    every excursion it returns to 12.3-13.3. `g10_equiv_ceil` went 9.9 to 52.3
    and **stayed** at 50.2; this oscillates around a level. The guard existed
    for the `g10` failure mode and that mode did not occur here.
  * **Identity off did not keep this arm's geometry quiet, and `g9_ceil_noid`
    is the comparison that shows it.** `g9` held 9.1-9.2 for 2,000 updates at
    `identity_weight = 0`; `g28` oscillates 12.3-22.1 with identity equally
    off. The difference between them is the local rotation. That does **not**
    refute [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]],
    whose claim is about a jump of more than 2x and no jump occurred -- but it
    does mean identity off is not sufficient for a quiet ceiling arm, and the
    pre-registration's reading that "identity off is what keeps a ceiling arm
    alive long enough to be read" is too strong. Alive, yes. Quiet, no.

## The number

`align` is the cosine between the emitted move and the ideal one, over the
twelve held-out crops. Per crop, trained:

    0.034  -0.022   0.030   0.014  -0.005  -0.107
   -0.040   0.348   0.141   0.224  -0.071  -0.118

    mean           +0.036   sd 0.138   SE 0.040   95% CI [-0.043, +0.114]
    untrained, frame head perturbed 0.05, SAME crops and convention:
                   -0.141   sd 0.266   SE 0.077   95% CI [-0.292, +0.009]

**The entire confidence interval sits inside the pre-registered null band of
-0.10 to +0.25**, not merely the point estimate, and `mean / SE = 0.89` --
indistinguishable from zero on its own terms.

## The bucket, applied as written

**`align` 0.036 falls in: -0.10 to +0.25, THE LIMITATION IS ARCHITECTURAL.**

Teacher forcing with a working actuator, identity off, no break and 2,000
updates, and the rule still emits a bearing indistinguishable from zero. So
invariant perception plus a local move head **cannot be taught to aim even
when handed the answer at every step**. This answers
[[the-architecture-can-place-residues-under-some-supervision]] **NO**, and by
the clause written into the pre-registration, **it ends the arm chain.** No
terms get added and nothing gets re-run. Three corroborating details, none of
which was needed for the bucket:

  * **`gained` is -0.02 A.** Mean `need` 12.34, mean achieved `rmsd_dmg`
    12.37. Over 120 steps the window ends marginally further from native than
    it started. The rule is not slowly approaching anything.
  * **`net/path` = 0.996.** The motion is almost perfectly straight, so this
    is not a rule wandering or oscillating in place. It moves coherently, in a
    direction uncorrelated with the right one.
  * **No "right then lost".** The per-step bearing on the first crop runs
    `[0.064, 0.012, 0.233, 0.019, -0.156, 0.267, 0.037, -0.131, 0.12, 0.099,
    -0.152, -0.004]` -- noise from the first step, not a bearing that decays.
    That was the alternative a mean could have hidden, and it is excluded.

**What training changed is the variance, not the mean.** Trained sd 0.138
against the perturbed-untrained control's 0.266, while the mean moved from
-0.141 to +0.036. Two thousand updates of exact supervision taught the rule to
emit a *consistent* small move that points nowhere in particular. The step
magnitude agrees: it asks for 0.04 A per step, 12.8 % of the 0.3 A cap, with
the tanh at 0.98 of its gradient near the origin -- so it is not saturated and
not asking for nothing. It asks confidently, in the wrong direction.

## What this does NOT establish

  * **The control is not a clean null, and the pre-registered one was a
    different number.** The within-arm untrained-perturbed control reads
    -0.141, where the pre-registration cited a convention-matched null of
    +0.206 from another arm. At 12 crops the control is 1.84 SE from zero and
    its interval also crosses zero, so both are nulls within noise -- but the
    phrase "indistinguishable from an untrained head" should not be used. The
    claim is **indistinguishable from zero**, which is what the bucket was
    defined on.
  * **The arrival caveat attaches to `rmsd_dmg`, not to `align`.** The readout
    flags that the gradient window is `truncate 12 x 0.3 A = 3.6 A` while the
    damaged window starts a median 10.95 A from native, 10 of 12 crops beyond
    it, so those crops *cannot be taught to arrive*. True, and it is why no
    `rmsd_dmg` here means anything. It does not touch `align`, which is a
    per-step bearing and needs no arrival -- which is precisely why the
    pre-registration made `align` the primary readout. And **raising
    `truncate` is not the escape hatch**, pre-registered on two grounds:
    [[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]
    predicted in advance that truncation should not limit a teacher-forced arm
    because each step's move is supervised at that step, and `g26_fullbptt`
    measured `truncate = 120` as *harmful*.
  * **One seed, no replicate.** Every caveat in
    [[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]
    applies. The buckets were made wide and sign-based for exactly this
    reason, and a CI wholly inside a band is the strongest form this evidence
    can take at k = 12.
  * **It says nothing about regeneration**, and `rmsd_dmg` 11.43 A is a
    teacher-forced ceiling comparable only to `g8`'s and `g10`'s, never to the
    rung-1 arms or the 4.0 A bar. The context was free here.
  * **One variable is against `base`, not against `g10`.** And `base` is not
    where the recent arms sit: resolved key by key, this arm trained with
    `pairwise_weight 0.0` and `orientation_weight 0.0`, while `g19` through
    `g27` all inherit `g14_pairwise` and carry `pairwise 0.01`, and `g22`
    onward carry `orientation 2.5`. So `g28` is three settings from `g25`, not
    one. For `align` under teacher forcing that is the clean choice rather than
    a confound -- `move_weight = 1` supervises the emitted move against the
    exactly computable ideal one, so the bearing was trained against the answer
    itself with no competing placement term pulling on it. But it does mean
    this arm is not `g25` with teacher forcing added, and no column of it may
    be set beside `g25`'s.
  * **The NO names an architecture, and the implicated feature is the EDGE
    SET, not the message passing.** Teacher forcing supervised the emitted
    move; it did nothing to the *perception*, which is still a k-NN rebuilt
    every step from the displaced coordinates
    ([[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]).
    So this null is what that node predicted a correct implementation of a
    mis-specified perception would produce, and the verdict reads: invariant
    perception **over an error-derived neighbourhood** plus a local move head
    cannot be taught to aim. It is not a verdict on message passing, which
    `Gala 2024` shows succeeding at this task on 8 targets with 5K parameters
    and no anchors at all, message-passing over the **target** graph
    ([[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]).
  * **A NO does not establish that no local rule could aim**, only that this
    one cannot be taught to under the strongest supervision available. The
    bearing may have to be **constructed** rather than perceived -- by
    diffusion, more hops, or iterative registration -- and
    [[in-regeneration-the-surviving-context-is-the-grid]] is still the right
    frame for why.
  * **The band condition was not met**, as above. A reader who weighs that
    more heavily than I do should know the arm's relative-rotation error ran
    1.5-2x the project's band throughout, while never breaking.

## Refutes
- [[the-architecture-can-place-residues-under-some-supervision]]

## Supports
- [[the-capacity-test-has-never-been-run-with-a-working-actuator-and-no-identity-break]]
