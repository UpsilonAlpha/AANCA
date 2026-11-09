---
type: observation
status: supported
evidence: ours
sources: [32505354]
---

# Five times the budget takes the same arm from 10.24 to 6.69 angstrom, so every number in the lever table was read before the arm had converged

`g17_long` is `g15_both` with `updates = 10000` and nothing else changed --
same `eval_every 250`, and **the identical 12 held-out crops**, verified from
the checkpoints rather than assumed:

    g15_both   eval_indices [6, 129, 167, 180, 193, 225, 264, 297, 319, 330, 348, 393]
    g17_long   eval_indices [6, 129, 167, 180, 193, 225, 264, 297, 319, 330, 348, 393]

Complete at 10,000 updates, 147.6 min, exit 0, `bun159`, 0.85 s/update.

    update   dmg   anchor   window   twist   bond
       250 10.43     40.3     29.9    10.4    3.82
      1000 10.43     41.5     31.3    10.2    3.81
      2000 10.54     44.7     31.1    10.3    3.43   <- g15's whole run ends here
      3250  9.54     35.3     33.3    11.2    3.07
      4250  8.44     30.0     32.7    11.7    2.70
      5500  8.86     30.4     34.0    11.9    2.42
      6500  7.87     27.2     27.0    13.2    2.50
      7750  7.59     26.4     27.4    13.2    2.11
      8500  6.94     20.9     32.6    12.6    1.55
      9000  6.69     20.7     28.1    13.3    1.73
      9500  6.62     20.0     24.3    15.6    1.94
     10000  6.75     19.8     25.3    14.3    1.55

## The pre-registered bucket: the first one, by a wide margin

`settled` against g15's **10.24** (commit `88a975b`, written four minutes into
the run):

    settled = (6.62 + 6.75) / 2 = 6.69

The first bucket is `settled <= 9.57` -- better than g15 by at least the
g14-to-g15 gap. The measured difference is **3.55 A**, five times that
boundary. In the pre-registration's own words: *the 2,000-update table was read
before any arm had converged, so "superadditive" is a statement about which arm
converges fastest, and every bucket boundary in the record needs re-reading at
this budget. Budget, not levers, is the project's live variable.*

**The pre-registered pair was not available and this is the substitute.** The
rule said the mean at updates **9750 and 10000**. `history.json` holds 38 rows,
not 40: the evaluations at **5750 and 9750 are absent**, with no "skipped for a
non-finite gradient" line in the log and no error. So `settled` is the mean at
**9500 and 10000** instead. Recorded as a departure from the rule rather than
folded in silently -- it cannot have been chosen to flatter the result, since
every reading from 9000 on lies between 6.62 and 6.75, but the rule was still
not the one that was applied, and **why two evaluations are missing is not
diagnosed**.

## What moved, and the one thing that got worse

  * **Placement**: `anchor` 40.3 -> 19.8, halved. `rmsd_dmg` 10.43 -> 6.69
    against a 12.23 baseline, so the arm now covers **45 %** of the distance to
    native where g15 covered 16 %.
  * **Bond**: 3.82 -> 1.55, the lowest this project has recorded. The window's
    C-ward join is closing, which is the task.
  * **`twist` is the cost**: 10.4 -> 14.3, outside the 9-12 band every
    `--identity-weight 0` arm has held, with a 17.0 excursion at 8750. This is
    not the update-1000 identity signature
    ([[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]])
    -- there is no jump, it drifts up from about update 6000 as placement
    improves. **A rule that is getting the window into place is getting its
    relative orientations worse**, and the rotation is the half of the move
    that is still inexpressible
    ([[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]).
  * **`window` does not deteriorate**: 29.9 -> 25.3. Against the untrained
    16.88 that is 1.50x, where g15 ended at 1.85x -- i.e. the internal half
    *improved* while placement improved, which is the opposite of the trade this
    project briefly believed in
    ([[placement-and-internal-shape-are-traded-against-each-other]]) and is
    consistent with the transit reading that replaced it
    ([[a-straight-line-between-two-poses-passes-through-a-contracted-intermediate]]):
    on the straight-line walk, `rmsd_dmg` 6.37 carries `window` at 2.27x its
    baseline, so g17's 1.50x at 6.75 is still below transit.

## Scope

  * **Still a double-ceiling arm, and the setting is now gone.** `native_edges`
    read the target graph and `move_weight = 1` supervises every emitted move
    against the ideal one. The honest statement is *the architecture can place a
    damaged window to 6.7 A given the target graph and the ideal move, if
    trained for 10,000 updates*. It bounds what a learned rule could reach and
    demonstrates nothing about one.

    **`native_edges` was removed on 9 Nov**, so this arm cannot be re-run from
    the current tree (commit `bd0c7f8` can still run it). The reason is not that
    it is a ceiling but that the thing it hands over -- who contacts whom -- is
    the structural-interaction topology the project's interpretability payoff
    needs the rule to learn for itself. The replacement proposal is
    [[question-can-the-rule-learn-its-own-perception-graph]], and **this number
    is its sharpest counter-evidence**: given the native selection and live
    distances, the rule still only reached 6.69 A, so recovering the graph may
    not be sufficient.
  * **It is not converged either.** 6.62 at 9500, 6.69 at 9000, 6.75 at 10000 --
    flattening, but the run ends while still drifting down from 8.4 at 4250.
    The budget question is reopened one order of magnitude up, not closed.
  * Constant learning rate, no sample pool, fixed 120-step rollout: all three
    neural-cubes levers are still absent
    ([[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]).
    This result is what budget alone bought.
  * 0.85 s/update, against g15's 1.12 and the single-lever arms' 2.0-2.4 on the
    same configuration. Three rates, no recorded card, thread count or torch
    build. Not diagnosed.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[the-two-levers-together-beat-either-alone-and-never-give-the-gain-back]]
- [[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]

## Requires
- [[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]
