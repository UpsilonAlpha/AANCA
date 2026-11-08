---
type: observation
status: supported
evidence: ours
sources: [32489939, 32489940]
---

# Both arms left the 12.23 Angstrom baseline and stayed off it, and the arm with a placement term holds more

The first movement off the regeneration baseline in this project. Nine earlier
arms sat at 12.18-12.85 against a baseline of 12.23
([[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]).
Both arms here are below it at every evaluation.

Bunya pool, 12 held-out crops, crop 128, `--damage 12`, `--identity-weight 0`,
`--equivariant-move --lab-rotation`. Baseline: `rmsd_dmg` **12.23**, anchor
**64.49**, window **16.88**.

**32489939 g13_natedge** -- `--native-edges`, no placement term. Complete.

    update   dmg   anchor   window   twist
       250  10.48    40.89    33.29    10.1
       500  11.06    49.10    26.49    10.2
       750  11.15    51.30    24.12    10.3
      1000  11.21    51.65    24.39    10.4
      1250  11.63    56.77    20.39    11.0
      1500  11.43    54.02    20.55    10.9
      1750  11.44    54.87    21.93    10.6
      2000  11.39    53.94    24.81    10.6

**32489940 g14_pairwise** -- `--pairwise-weight 0.01`, live edges. At 1500 of
2000 when this was written.

    update   dmg   anchor   window   twist
       250  11.70    55.46    22.03     9.8
       500  10.97    43.51    31.27    10.1
       750  10.97    43.35    27.15     9.7
      1000  11.18    47.88    26.94    10.5
      1250  10.70    41.63    29.00    10.1
      1500  10.77    41.77    27.14    10.2

## The pre-registered test took two branches that were written as alternatives

The prediction recorded at commit `b447f7b`, before g14's second evaluation
existed: *"if the pairwise term does what it was added for, g14's rmsd_dmg
holds or falls across 500, 750 and 1000 while g13's keeps rising."*

**That is what happened** -- g14 fell to 10.97 at 500, held it at 750, and is
at 10.70-10.77 by 1250-1500, while g13 rose 11.06, 11.15, 11.21 over the same
three. The named branch is *"g14 holds or falls while g13 rises -> the levers
are complementary: the edge set finds the position, the restraint keeps it."*

**But the fourth branch is also true.** *"g13 flattens ABOVE the baseline
instead of reaching it -> the edge set holds some of the gain with no placement
term at all, which is the stronger of the two results."* g13 flattened at
11.39-11.44 and held there for a thousand updates: **0.8 A below baseline,
permanently, with nothing in its objective charging for placement.**

The pre-registration listed those four outcomes as if exactly one could occur,
and two did. They are not mutually exclusive and I should have seen that when
writing them: one is about g14's direction, the other about where g13 settles.
Recorded as a fault in the prediction rather than quietly reported as a clean
hit, because a pre-registration that can be satisfied two ways at once is
weaker than it looked.

One wobble against the prediction: g14's 1000 reading rose to 11.18 before
falling again to 10.70. The brief said not to let a reading fit itself to a
branch afterwards, so: that point is against the "holds or falls" wording
taken strictly, and the curve as a whole is not.

## What the two levers are each worth

  * **native edge set alone: 0.8 A**, held to 2000 updates with no placement
    term (11.39 against 12.23).
  * **placement term alone: 1.5 A**, and still falling at 1500 (10.70 against
    12.23).

So the restraint holds about twice what the edge set does, and the edge set's
best reading -- 10.48 at update 250 -- is better than anything g14 has reached,
then decays. That is consistent with
[[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]] and
[[clamped-fape-is-flat-over-the-half-of-the-journey-the-window-must-travel]]
being two different problems: the edge set lets the rule *find* the position
quickly, and with nothing defending it the optimiser spends it again.

## Each arm trades the window's internal shape, in opposite directions

The window-to-window column is not decoration. In **g13** it falls 33.29 to
20.39 while anchor rises 40.89 to 56.77 -- buying internal shape with
placement. In **g14** it rises 22.03 to 29.00 while anchor falls 55.46 to 41.63
-- buying placement with internal shape. Both end worse than the baseline's
16.88 on the internal half.

So neither arm improves both at once, and the term that charges for one pays in
the other. That is the contracted-intermediate geometry showing up in training
rather than in an interpolation, and it is the argument for the next arm
carrying a weight on the window-to-window half too, or a stronger internal
restraint beside it. It is also why the two halves are reported separately:
lumped, this would be invisible.

## Identity's attribution survives two more arms

`twist` holds 9.7 to 11.0 across all fourteen evaluations in both arms, with
**no jump at update 1000**, where every identity-bearing arm breaks.
[[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]] was
supported on g9 alone; this is two further arms at `--identity-weight 0`
holding flat through the onset, which is the same prediction confirmed again
rather than a new one. `bond` does not climb: g13 wobbles 3.96-4.95, g14 falls
5.195 to 3.879.

## Scope

  * **g14 is incomplete** at 1500 of 2000 here. Its last two evaluations do not
    change the direction but the final figure may move by a tenth.
  * Neither arm is a design method. `--native-edges` reads the native graph and
    no de novo task has one, so g13 reports a **ceiling**. The placement term
    is ordinary supervision and is available to a real run.
  * 12 held-out crops. The spread across crops is not in this node and the
    per-crop numbers are in the logs.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
- [[in-regeneration-the-surviving-context-is-the-grid]]
- [[clamped-fape-is-flat-over-the-half-of-the-journey-the-window-must-travel]]
- [[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]
- [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]
