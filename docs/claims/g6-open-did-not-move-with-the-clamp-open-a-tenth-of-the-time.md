---
type: observation
status: supported
evidence: ours
sources: [32359791]
---

# Leaving FAPE unclamped on a tenth of updates changed nothing, and made the twist worse

`--fape-clamp-prob 0.9`, AF2's own scheme, otherwise `g6_self`'s command
exactly. 8,000 updates, exit 0, 3 skipped for a non-finite gradient.

    updates    rmsd   twist    bond
       500   17.45    64.4   0.011
      3000   17.49    65.7   0.701
      5500   17.49    66.1   0.856
      6500   17.49    71.0   4.738
      8000   17.53    73.4   1.832

The helix baseline is 17.45 A and 64.4 deg. So the arm spent its whole run
inside 0.25 A of where it started and ended **nine degrees worse** on twist,
while identity trained normally over the same updates (NLL 2.555, recovery
21.5 %).

The held-out evaluation is clamped at 10 A unconditionally in every arm, so
these numbers are directly comparable with `g6_self`'s -- which was the point
of keeping the evaluation fixed while the training clamp varied.

## Refutes
- [[the-fape-clamp-is-what-gates-the-geometry]]

## Supports
- [[the-rule-keeps-whatever-conformation-it-is-given]]
