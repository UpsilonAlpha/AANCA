---
type: observation
status: supported
evidence: ours
sources: [33277377, 33249861, 33277250]
---

# Pinning the surviving context improves placement, orientation and whole-structure RMSD at once, so context drift was a cause and not a symptom -- and the absolute bar still fails by a factor of two

`g24_frozen` (33277377) completed 21 Nov, exit 0, 10,000 updates, 393.5 min on
bun161. One variable against `g22_random`: `freeze_context = true`, which pins
the surviving residues' geometry while they keep ticking, perceiving and sending
messages.

    settled, 12 held-out crops      fit_all  fit_intact   twist   rmsd    bond
    g22_random   context FREE          8.41        9.02   12.8d   3.73   0.629
    g24_frozen   context PINNED        7.98        8.49   10.3d   2.52   1.355

    bucket 0 PASS line                    --        4.00

## Three quantities moved the right way together, which is the finding

**Placement improved by 0.53 A** `fit_intact`, **twist came back inside the 9-12
band** from 12.8 to 10.3, and **whole-structure `rmsd` fell 1.21 A** -- the last
being the context staying where it was put, which is the mechanism working
rather than an independent result.

One quantity improving by 0.53 A on a single unreplicated seed would be nothing
([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]). Three
moving together in the direction a single mechanism predicts is harder to get
from sampling noise, and that is the whole weight this arm carries. **The
direction is trustworthy; the magnitude is not.**

**So the context drift measured at 2.34 A was a CAUSE, not a symptom**
([[the-rule-displaces-the-surviving-context-it-is-supposed-to-read]]). That was
the open question when the drift was confirmed: the rule might have been moving
the context *because* it could not place the window, in which case freezing
would change nothing or hurt. It helped, on the quantity the project cares about
and on two others.

**And twist falling tells us something the placement number does not.** The
orientation damage that voided four consecutive arms -- g19 at 14.26, g21 at
13.25, g22 at 12.8, g23 at 12.9 -- was partly the context moving. A bonded
pair's relative rotation is a statement about *two* residues, and when one of
them is a context residue that has drifted, the pair reads as twisted whether or
not the rule mis-oriented anything.

## The predicted cost showed up, in the bond term

`bond` rose from 0.629 to 1.355. This was anticipated: with the context pinned,
the rule can no longer relieve strain at the window's covalent joins by dragging
the anchors toward it, so closing the chain is genuinely harder. The arm pays
0.73 on the bond term to gain 0.53 A of placement, which is the trade the
registration named and is worth stating as a real cost rather than a footnote.

## Bucket 0 still FAILS, by a factor of two

**8.49 A against a 4.0 A line.** The bar was set from the 3.8 A CA-CA spacing --
the scale at which every residue is within one position of where it belongs --
and 8.49 A is a different conformation in a different place. The registration's
clause stands: *a local rule with this information and this objective does not
place a backbone.*

Against the native-blind floor, provisionally: 8.49 against the random floor's
9.48 is a **0.99 A margin**, up from `g22_random`'s 0.46. But **both figures are
pending**, because `geometric_floor.py` drew a different spatial crop of each
protein than the arms it is quoted against, and the corrected floor is expected
~0.3 A higher (jobs 33282053 / 33282054 are the re-measurement). On that
estimate the margin is ~1.2-1.3 A. Still small, still one seed.

## What this arm makes available that no recent arm did

**`twist` is INSIDE the band, so buckets 1 and 2 are readable here.** g19, g21,
g22 and g23 all had placement numbers or comparisons voided on that guard. In
particular **bucket 2 is now live**: the fraction of natively-helical window
residues the rule places in the helical Ramachandran basin, against the ~5-10 %
chance rate of a uniform start. That is the direct test of learned secondary
structure, it is the measurement to lead with rather than any `rmsd_dmg`, and it
is a CPU readout on an existing checkpoint rather than a GPU run. It has never
been computed for any arm.

## Scope

  * **One seed, one run, no replication.** 0.53 A has no error bar; see above
    for why three correlated movements are the argument and the magnitude is
    not.
  * **Not comparable with g1-g22 on `fape`, `rmsd` or `twist` as absolute
    values** in the sense those columns had before: 116 of 128 residues cannot
    move, so a mean over residues or bonded pairs is a different quantity. The
    comparison above is deliberately against `g22_random` alone, which differs
    in exactly this one setting, and `rmsd_dmg`/`fit_intact` are the columns the
    bar is set on.
  * **The floor comparison is provisional** until 33282053 lands.
  * Freezing is a change to the TASK as much as to the rule -- the rule is no
    longer asked to leave the context alone, it is prevented from moving it. A
    rule that would have learned context stability on its own is no longer
    tested on that, and nothing here says whether it could have.

## Supports
- [[the-rule-displaces-the-surviving-context-it-is-supposed-to-read]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Requires
- [[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]
