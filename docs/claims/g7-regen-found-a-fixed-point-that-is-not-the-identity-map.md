---
type: observation
status: open
evidence: ours
sources: [32389742]
---

# The first regeneration arm moved the window 1.5 Angstrom the wrong way and held it there

`--damage 12`, otherwise the g6 objective: FAPE 1.0, no clash term. At 1,500 of
8,000 updates:

    updates   rmsd_dmg   twist   recovery
       500     12.23      9.0      5.7%
      1000     12.27      ...      ...
      1500     12.27     14.7     18.0%

The baseline is 10.77 A and 9.6 deg
([[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]]).

**This is a different failure from the four reconstruction arms, and a more
informative one.** They sat at exactly their starting conformation -- the
identity map, a perfect fixed point, which is what the stability half of the
objective rewards once the accuracy half has no gradient
([[the-rule-keeps-whatever-conformation-it-is-given]]). This arm found a fixed
point that is **not** the identity map: it moved, settled, and has held for
three evaluations. The degeneracy that regeneration was predicted to break
([[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]])
did break.

It settled on the wrong one, and the shape of the error is what the reach
probes predicted. Given an unambiguous magnitude signal -- the window starts
~11 A out with 90 % of FAPE's pairs live -- it travels about that far in a
direction that does not help, degrading local conformation on the way (twist
9.6 to 14.7). A rule that moves decisively and wrongly is what a loss with a
magnitude and no bearing produces
([[the-reach-probes-split-the-three-frozen-arms]]).

`status: open` because 6,500 updates remain. It is being left to run as the
control at FAPE 1.0 with no clash term, which is what separates the *task*
change from the *objective* change -- killing it would cost that comparison for
nothing.

## Supports
- [[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]
- [[the-objective-should-restrain-the-output-not-teach-the-placement]]
