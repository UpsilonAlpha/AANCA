---
type: observation
status: supported
evidence: ours
sources: [32494313]
---

# Both levers together settle below either arm's best, and unlike either arm they keep what they find

`g15_both` ran `--native-edges` **and** `--pairwise-weight 0.01` together, with
everything else identical to
[[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]]'s two
single-lever arms: `--identity-weight 0`, `--equivariant-move --lab-rotation`,
`--birth none --initial helix --damage 12`, 2000 updates, eval every 250,
crop 128, 120 steps, same 12 held-out Bunya crops, same 12.23 A baseline.
Complete at 2000 updates, 37.4 min, exit 0.

    update   dmg   anchor   window   twist   bond
       250 10.43    40.33    29.92    10.4   3.815
       500 10.53    40.68    32.34    10.4   3.612
       750 10.54    41.05    28.73     9.7   3.704
      1000 10.43    41.54    31.33    10.2   3.806
      1250 10.53    43.72    26.84    10.4   3.833
      1500 10.28    41.94    32.26    12.1   3.398
      1750  9.94    36.98    32.27    10.1   3.145
      2000 10.54    44.69    31.15    10.3   3.434

## The pre-registered bucket: superadditive

The prediction at commit `ec0ce82`, written before the job was submitted:
**`settled` = the mean of `rmsd_dmg` at updates 1750 and 2000**, judged against
`g13` settled 11.42, `g14` settled 10.91, baseline 12.23.

    settled = (9.94 + 10.54) / 2 = 10.24

That falls in the first of the four exclusive buckets: **`settled < 10.70`
-- superadditive; below the best either arm ever reached. The next arm is a
weight sweep, not another lever.** g13's best single reading was 10.48 and
g14's was 10.70; g15's *settled* value is below both, and its best single
reading, **9.94 A at update 1750, is the lowest `rmsd_dmg` this project has
produced**.

**The bucket does not depend on the averaging.** The two points it averages are
0.60 A apart, which is four times the ~0.15 A adjacent-evaluation wobble the
two-point rule was written to smooth, so the rule was applied to a wider gap
than it was designed for. It survives anyway: the *worse* of the two points,
10.54, is still below the 10.70 boundary and below g14's settled 10.91. Said
here because a two-point mean over a 0.6 A gap would otherwise be the kind of
instrument fault that produced the correction in
[[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]].

**The void condition did not trigger.** It would have, had g15 repeated g14's
shape -- a minimum near 1250 drifting back by 2000. g15's minimum is at 1750,
its second-to-last point, and its first 1250 updates are flat at 10.43-10.54.
So `settled` is reading the arm and not a drift, and the comparison stays on
`settled` as pre-registered rather than moving to each arm's best.

## The result that was not predicted: it never gives the gain back

Both single arms found a position better than they kept. g13 reached 10.48 at
update 250 and gave up 0.9 A of it; g14 reached 10.70 at 1250 and gave up
0.28 A. **g15 gives up nothing**: its whole curve lies within 10.43-10.54
until it improves, and its worst reading over 2000 updates (10.54) is better
than either single arm's settled value.

    arm             best   settled   worst   held at 2000
    g13_natedge    10.48     11.42   11.63      0.84 A
    g14_pairwise   10.70     10.91   11.70      1.25 A
    g15_both        9.94     10.24   10.54      1.69 A

So the combination is not a 0.7 A improvement on g14's number; it is a change
in the shape of the curve. The reading the pre-registration offered for this
bucket -- *"the edge set finds the position early and the restraint defends
it"* -- is exactly what the numbers show, and it is visible in the first
evaluation: g15 is already at 10.43 at update 250, where g14 was at 11.70.
Neither lever alone produced a run with no retreat in it.

## The second prediction: it came true and it proved nothing

Pre-registered independently: *"g15's window-to-window half ends above the
baseline's 16.88, because nothing in either flag charges for internal shape."*
It ends at **31.15**, against g13's 24.81 and g14's 29.14 -- the worst of the
three. It is above 16.88 at every evaluation and never falls below 26.84. So
the stated outcome held.

**The causal half is withdrawn, 9 Nov.** 16.88 is the window's value at *zero*
displacement, and a window in transit contracts: on a straight-line walk from
baseline to native the internal half rises to 4.95x its baseline before it
falls
([[a-straight-line-between-two-poses-passes-through-a-contracted-intermediate]]).
Ending above 16.88 was therefore guaranteed by the geometry of being partway
there, whether or not the objective had anything to do with it -- and read at
matched displacement the three arms sit at 1.47-1.85x, i.e. **below** the
straight line rather than above it. A prediction that the mechanism's absence
would also produce is not a test of the mechanism, and it should not have been
written as one.

What this node claimed on the strength of it --- that placement and internal
shape are being traded, and that the window-to-window weight is the indicated
next arm --- is retracted in
[[placement-and-internal-shape-are-traded-against-each-other]]. The `window`
column is an unsigned MSE, so it cannot distinguish collapse from stretch
either. **Nothing is currently known about the sign or the cause of the arms'
internal-shape number.**

A caveat on the anchor column: it does not move monotonically with `rmsd_dmg`
here (41.54 at 1000 against 36.98 at 1750 against 44.69 at 2000), so on this
arm the unclamped anchor term tracks the metric less tightly than it did on the
two single-lever arms. Not diagnosed; recorded so a later arm is not read as if
it did.

## The bucket's decision rule, and what is left of it

The bucket g15 landed in reads *"the next arm is a weight sweep, not another
lever"*, and the body of this node then named the window-to-window weight as
the indicated arm. That reading rested on the second prediction, which is
withdrawn above, so **the rule no longer points at a specific weight** -- and
with [[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]
and [[question-has-any-arm-been-replicated-at-a-second-seed]] both open, a
sweep over a boundary with no error bar on either side would be measuring the
wrong thing first.

Recorded here rather than acted on silently: the decision rule is released
because its premise was withdrawn, not because a different arm looked more
interesting after the fact. The next arm is budget
(`g17_long`), which is what both open questions gate on, and it is a new
section rather than an edit to this one.

## Identity and bond

`twist` holds 9.7-12.1 with **no jump at update 1000**, which is the third
further arm at `--identity-weight 0` to run flat through the onset that breaks
every identity-bearing arm ([[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]).
The 12.1 at update 1500 is the highest single reading across the three arms and
sits just above the 9-11 band; it falls back to 10.1 and 10.3 and is not at the
onset, so it is noise rather than the signature. `bond` falls 3.815 to 3.434
and does not climb.

## Scope

  * **`--native-edges` reads the native structure's edge set, which no de novo
    design run has.** So g15, like g13, reports a **ceiling** and is not a
    method. `--move-weight 1` is teacher forcing on top of that. The honest
    statement of this arm is *the architecture can place a damaged window to
    9.9 A given the target graph and the ideal move*, which bounds what the
    learned version could reach and does not demonstrate it.
  * The placement term alone (g14) is ordinary supervision and is available to
    a real run; the combination is not.
  * 37.4 min at 1.12 s/update, roughly half the 2.0-2.4 s/update the two
    single-lever arms took on the same configuration. Not investigated.
  * 12 held-out crops. Per-crop spread is in the log, not in this node.

## Requires
- [[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]
- [[question-has-any-arm-been-replicated-at-a-second-seed]]

## Refutes
- [[placement-and-internal-shape-are-traded-against-each-other]]

## Supports
- [[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
- [[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]
- [[clamped-fape-is-flat-over-the-half-of-the-journey-the-window-must-travel]]
- [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]
