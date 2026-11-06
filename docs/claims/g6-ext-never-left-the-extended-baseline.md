---
type: observation
status: supported
evidence: ours
sources: [32314557]
---

# From 19 Angstrom away, g6_ext froze on the extended baseline rather than the helix one

`--birth none --initial extended`, otherwise `g6_self`'s command exactly.
8,000 updates, exit 0, all sixteen held-out evaluations between **36.72 and
36.75 A** against the extended baseline's own 36.73, and twist 84.1 to 85.4
against its 84.1. Identity trained normally alongside (NLL 2.654 against the
ln-20 floor of 2.996, recovery 15.4 %).

This was a prediction recorded before the run and it held, which makes it the
only one of three today that did. Its force is comparative: `g6_self` and
`g6_noid` freezing at 17.45 A was consistent with 17.45 being a genuine optimum
the rule had discovered, whereas freezing at 36.73 A -- nineteen Angstrom away,
at a conformation no fold resembles -- is not. Two independent baselines, two
freezes, each at its own starting value.

It also disposes of the last innocent reading, that a 17.45 A helix is near
enough to a real fold for FAPE not to care. 36.73 A is near nothing.

## Supports
- [[the-rule-keeps-whatever-conformation-it-is-given]]
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
- [[the-identity-term-perturbs-geometry-rather-than-helping-it]]
