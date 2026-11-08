---
type: observation
status: supported
evidence: ours
sources: [growth.py, baseline_probe.py]
---

# FAPE spends 7.5 per cent of its fall on the first half of the window's journey; an unclamped pairwise restraint spends 82

Measured by linearly interpolating a damaged 12-residue window from the ideal
helix baseline toward its native pose and reading every geometric term at each
step. The rule is not involved: this is a property of the objective over the
path the window has to take.

    t   rmsd_dmg   anchor   window    fape   fape_inf
    0.0    21.24   101.49     8.72  0.2621    0.6164
    0.2    16.99    54.50    25.61  0.2585    0.5316
    0.4    12.75    27.18    44.70  0.2509    0.4354
    0.5    10.62    18.62    43.16  0.2423    0.3640
    0.6     8.50    12.17    32.43  0.2234    0.2690
    0.8     4.25     3.39     9.22  0.1210    0.1210
    1.0     0.00     0.00     0.00  0.0000    0.0000

Share of each term's total fall spent on the first half of the journey:

    pairwise, window-to-anchor only     81.7 %
    pairwise, both halves lumped        43.9 %
    FAPE, clamped at 10 A                7.5 %
    FAPE, unclamped                     41.0 %

**Clamped FAPE moves 0.2621 to 0.2423 across the first half -- 1.5 per cent of
its value -- while the window closes 21.24 A to 10.62 A.** It only begins to
move near t = 0.6, and from t = 0.8 the clamped and unclamped columns are
identical, which is the clamp releasing. So the gradient is flat over exactly
the stretch the window must cross, which is the mechanical form of
[[the-rule-keeps-whatever-conformation-it-is-given]]: a rule reading this
objective is told almost nothing until it has already arrived.

This is what
[[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
predicted from E(n)-GNCA's own argument -- that a clamped local loss does not
identify the target uniquely -- arriving as a number on our own objective.

## The internal half cancels the placement signal, so the two must stay apart

Unanticipated and the reason the term reports two quantities. **Window-to-window
distances are not monotone along the path**: they rise 8.72 to 44.70 before
falling, because a straight line between two poses passes through a contracted
intermediate. Lumped into one number the term is still monotone overall, but its
first-half share drops from 81.7 to 43.9 per cent -- the internal half actively
cancels the placement signal in the far regime the term exists for.

Window-to-window pairs only fix the window's internal shape; window-to-anchor
pairs are what place it in the structure. Only the anchor half enters the loss.

## Scope, and what it does not say

  * **The ratios are the claim, not the absolute values.** The A^2 figures come
    from the laptop pool, where this probe reads helix 10.77 A against Bunya's
    12.23 A ([[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]).
    The clamp's flatness is a property of the clamp and the geometry and travels;
    pick a weight against a run's own first evaluation, not against 101.49.
  * **No training run has been made.** This says the gradient exists, not that a
    rule follows it. At `--pairwise-weight 1.0` the term is about 100x the rest
    of the objective, so parity with FAPE is near 0.002 to 0.01.
  * It does not refute
    [[the-objective-should-restrain-the-output-not-teach-the-placement]]. An
    unclamped distance restraint is more supervision than a clamped one, and
    that tension is a real choice rather than a measurement.

## Supports
- [[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
- [[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]
- [[the-rule-keeps-whatever-conformation-it-is-given]]
