---
type: observation
status: open
evidence: ours
sources: [32359791]
---

# At 5,625 of 8,000 updates, the partially-unclamped arm sits in the same band as the reference

`--fape-clamp-prob 0.9`, otherwise `g6_self`'s command exactly. Held-out:

    updates   rmsd   twist    bond
       500   17.45    64.4   0.011
      1500   17.44    67.0   0.231
      3000   17.49    65.7   0.701
      4000   17.50    66.6   1.757
      5500   17.49    66.1   0.856

17.44 to 17.51 against the helix baseline's 17.45 -- indistinguishable from
`g6_self`'s 17.44 to 17.53. Identity trains normally (NLL 2.555, recovery
21.5 %) and the bond term drifts as in the other identity-weighted arms.

`status: open` because the run has 2,375 updates left. Recording it as a
finding now would repeat this morning's mistake of reading a conclusion off a
partial log.

## Supports
- [[the-rule-keeps-whatever-conformation-it-is-given]]
