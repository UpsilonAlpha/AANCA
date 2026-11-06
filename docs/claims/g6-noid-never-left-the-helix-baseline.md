---
type: observation
status: supported
evidence: ours
sources: [32241722]
---

# With geometry as the entire objective, g6_noid did not move by one printed digit in 8,000 updates

`--birth none --initial helix --identity-weight 0 --identity-track on`:
identical rule and identical perception to `g6_self`, identity head built and
blind perception kept, one term removed. All sixteen held-out evaluations:

    updates   total    fape     rmsd    alive    bond   twist
       500    0.837   0.837   17.45 A  100.0%   0.003    64.4
      ...     0.837   0.837   17.45 A  100.0%   0.002    64.4
      8000    0.837   0.837   17.45 A  100.0%   0.002    64.4

Not a single digit changed. `total` equals `fape` throughout because identity is
zero, completion and reach are zero on a fully-alive rollout, and the bond term
times 0.122 is 0.0002 -- so **geometry was literally the whole loss and the
whole loss was motionless**, at the helix baseline's own 17.45 A and 64.4 deg.

The `--identity-track` flag exists because this arm would otherwise have been
confounded: `--identity-weight 0` was also switching off blind perception and
removing the identity head, so it would have differed from `g6_self` in three
ways and isolated none.

## Refutes
- [[the-identity-gradient-starves-the-geometry]]

## Supports
- [[the-rule-keeps-whatever-conformation-it-is-given]]
- [[identity-supplies-the-trunk-that-geometry-reads]]
- [[the-identity-term-perturbs-geometry-rather-than-helping-it]]
