---
type: observation
status: supported
evidence: ours
sources: [growth.py, test_pairwise_restraint.py]
---

# A window walking straight from baseline to native contracts on the way, so a high internal-shape number is what transit looks like

Walk the damaged window linearly from the baseline pose to its native pose on
1UBQ, evaluating `pairwise_distance_restraint` at each tenth. The placement
half falls monotonically. **The internal half does not**: it rises from 8.72 to
44.70 A^2 over the first four tenths and only then falls, because a straight
line between two poses passes through a contracted intermediate -- the window's
internal CA-CA distances get worse before they get better.

      t  rmsd_dmg    anchor   window    fape  fape_inf
    0.0     21.24    101.49     8.72  0.2621    0.6164
    0.1     19.12     75.14    15.51  0.2607    0.5741
    0.2     16.99     54.50    25.61  0.2585    0.5316
    0.3     14.87     38.80    37.02  0.2558    0.4871
    0.4     12.75     27.18    44.70  0.2509    0.4354
    0.5     10.62     18.62    43.16  0.2423    0.3640
    0.6      8.50     12.17    32.43  0.2234    0.2690
    0.7      6.37      7.14    19.79  0.1798    0.1896
    0.8      4.25      3.39     9.22  0.1210    0.1210
    0.9      2.12      0.92     2.38  0.0587    0.0587
    1.0      0.00      0.00     0.00  0.0000    0.0000

This was measured and pinned in `tests/test_pairwise_restraint.py` on 3 Nov, as
the justification for reporting the two halves separately rather than summed.
It was never entered in the graph, and that omission is what let the inference
in [[placement-and-internal-shape-are-traded-against-each-other]] stand for a
day: **the column that inference reads has a second mechanism in it, already
measured in this project.**

## The comparator the arms were never read against

A window a sixth of the way along its journey *should* show a large internal
number. Read at matched displacement rather than against the baseline, the
three arms do not look traded -- they look better than transit:

    read at                    rmsd_dmg   window   window / baseline
    straight-line walk, t=0.5     10.62    43.16               4.95x
    g15_both settled              10.24    31.15               1.85x
    g14_pairwise settled          10.91    29.14               1.73x
    g13_natedge settled           11.42    24.81               1.47x

So at the same placement quality the arms' internal half is **below** the
straight line, not above it. Whatever the arms are doing to internal shape, it
is less damaging than the shortest path between the two poses.

## Scope

  * **The walk is one crop of 1UBQ, baseline `window` 8.72 and `rmsd_dmg`
    21.24; the arms are 12 Bunya crops, baseline `window` 16.88 and `rmsd_dmg`
    12.23.** Different windows, so the absolute A^2 values are not commensurable
    and the `window / baseline` ratio is doing the work in that table. The
    comparison is indicative of the mechanism's sign and rough scale, and is
    **not** a measurement on the pool the arms ran on
    ([[two-instruments-disagree-by-2-5-angstrom-about-where-the-window-starts]]
    is the standing warning about reading one pool's number on another).
  * The matched-displacement walk on the actual held-out crops has not been run.
    It is CPU-only and cheap, and it is what would make this quantitative.
  * An interpolated window is not a physical backbone at the midpoints. The
    question here is the shape of the loss surface along the line, not whether
    every point on it is a protein.

## Refutes
- [[placement-and-internal-shape-are-traded-against-each-other]]

## Supports
- [[clamped-fape-is-flat-over-the-half-of-the-journey-the-window-must-travel]]
