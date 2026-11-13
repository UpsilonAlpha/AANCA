---
type: observation
status: supported
evidence: ours
sources: [32604034, 32604037]
---

# With no ceiling inside the rollout, the rule places a damaged window at 7.78 angstrom, and the orientation term is what makes it its best

The first two arms since g7 with **no native information inside the rollout**:
no teacher forcing (`move_weight = 0`), no target graph (`native_edges`
removed). Both complete, 10,000 updates, exit 0, on bun144 (mi300x).

**The same 12 held-out crops as every comparator, verified from the
checkpoints rather than assumed** -- the `g18_seed1` lesson:

    g14_pairwise  [180, 167, 393, 264, 330, 193, 348, 297, 6, 319, 225, 129]
    g17_long      [180, 167, 393, 264, 330, 193, 348, 297, 6, 319, 225, 129]
    g19_nomove    [180, 167, 393, 264, 330, 193, 348, 297, 6, 319, 225, 129]
    g20_orient    [180, 167, 393, 264, 330, 193, 348, 297, 6, 319, 225, 129]

## The numbers

`settled` is the mean of the last two available evaluations.

    arm           ceilings            settled   twist   bond   anchor
    baseline      --                    12.23     9.6     --       --
    g14_pairwise  move 1, 2k updates    10.91    ~10    3.0+       29
    g17_long      move 1 + edges, 10k    6.69    14.3   1.55     19.8
    g19_nomove    NONE, 10k              8.39   14.26   1.53    27.72
    g20_orient    NONE + orient 2.5      7.78   11.24   0.84    23.58

Curves, first and last:

    g19   update  250: rmsd_dmg 11.34, twist  9.90
          update 10000: rmsd_dmg  8.37, twist 13.67
    g20   update  250: rmsd_dmg 11.09, twist 10.10
          update 10000: rmsd_dmg  7.43, twist 11.14

## g20's pre-registered bucket: the second one, and the term works

The primary quantity was **`twist`**, not `rmsd_dmg`, against g17's 14.3 and
the 9-12 band every identity-free arm held before. `settled` twist **11.24**
lands in `9.0 < twist <= 12.0`: *the term holds orientation in the band that
arms held before budget pushed g17 out of it.*

**And g19 is the control that makes it causal.** Same arm, same crops, same
budget, `orientation_weight = 0`: twist drifts to **14.26**, which reproduces
g17's 14.3 under a completely different objective. So the drift is not a
property of the ceilings, and turning on `relative_frame_restraint` at the
measured parity weight removes it. The term built on 5 Nov and never given a
weight controls the quantity it was built for
([[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]).

**The secondary bucket was wrong-footed in the good direction.** It asked
whether orientation was bought by spending placement: within 0.5 A of g19
meant free, worse by more than 0.5 A meant a trade. g20 is **better by
0.61 A** (7.78 against 8.39), which neither bucket covers. Orientation did not
cost placement; it improved it, and `bond` 0.84 is the lowest this project has
recorded against g17's 1.55.

## g19's placement number is VOID by its own pre-registration

Stated rather than quietly dropped. g19's void condition was *"if `twist`
leaves the 9-12 band -- that would be g17's rotation drift, not this arm's
variable, and the two cannot be read apart on one curve."* g19's twist settled
at **14.26**, outside the band. **So g19's `rmsd_dmg` of 8.39 is void**, and
the thesis-bearing number is g20's 7.78.

Scope of the void, because the rule did not say: it voids the **placement**
claim, which is what the rationale was about. g19's *twist* reading is a direct
measurement of a different quantity and stands -- it is the control above, and
the void condition firing is what makes that control legible rather than
awkward.

## And the pre-registration itself contained the error it was written to avoid

The buckets were stated against **g14's 10.91**, and g14 ran **2,000** updates
where these ran 10,000. So "beats g14 outright" conflates removing two ceilings
with five times the budget -- which is exactly the mistake the lever table was
retired for ([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]]),
committed again four days later by the same author in the pre-registration
meant to prevent it.

The budget-matched comparison is the only sound one, and both arms are in it:

    g17_long   both ceilings, 10k   6.69    45 % of the way to native
    g20_orient no ceilings,   10k   7.78    36 %
    g19_nomove no ceilings,   10k   8.39    31 %  (void)

**Removing teacher forcing and the target graph costs 1.09 A at matched
budget.** That is the number this pair was run for.

## Why this bears on the thesis when nothing before it did

Nine arms without teacher forcing sat at 12.18-12.85 A -- their own baselines,
to two decimal places
([[the-rule-keeps-whatever-conformation-it-is-given]]). g20 moves the window
4.45 A from the same baseline with no native information in the rollout at all.
**This is the first measurement in the project that reports a capability rather
than a bound.**

It does not isolate *what* changed, and three things did against those nine
arms: 5x the budget, the equivariant translation fix, and the unclamped
window-to-anchor restraint. What it establishes is that the combination makes
placement learnable from restraints alone.

## Scope

  * **One seed, one split, no replicate.** `seed` still chooses the held-out
    set, so a second seed is not a replicate and `--eval-indices` is still
    unbuilt ([[question-has-any-arm-been-replicated-at-a-second-seed]]). This
    is why the regeneration claim is not being marked `supported` on it.
  * **Read at step 120 and nowhere else.** Whether the window *stays* is
    unmeasured, in this arm as in every other
    ([[the-growth-objective-supervises-a-fixed-final-step-while-the-track-that-converged-samples-one]]).
  * **Still structure-only.** `identity_weight = 0`, so neither arm touches
    cofolding, which is the problem the project says it is solving
    ([[a-local-per-residue-rule-can-build-a-protein-backbone]]).
  * **Neither is converged.** g20 is still falling at 10,000 (8.27 -> 8.12 ->
    7.43 over the last three readings), and its best reading is its last.
  * **The pre-registered `settled` pair did not exist, for the third time.**
    Updates **5750 and 9750 are missing from `history.json` in both runs**, 38
    rows of 40, identical to g17. Three runs, three different objectives, the
    same two updates: this is now reproducible and therefore diagnosable, and
    it is no longer excusable as a one-off. `settled` is the mean at 9500 and
    10000 in all three.
  * **A different card.** 3.2 s/update on mi300x against g17's 0.85 on mi355x,
    which is the forced card change showing up at roughly 3.8x. It makes the
    chain longer, not the result wrong, and it is the fourth undiagnosed rate
    on one configuration.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]

## Requires
- [[question-has-any-arm-been-replicated-at-a-second-seed]]
