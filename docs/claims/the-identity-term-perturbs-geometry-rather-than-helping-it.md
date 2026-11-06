---
type: claim
status: supported
sources: [32215014, 32241722, 32314557]
---

# Both identity-weighted arms broke their bond geometry by three orders of magnitude

The bond term over 8,000 updates, which is the only column that moves in any
arm:

    arm       identity   bond at 500   bond peak   rmsd change
    g6_self   93.7         0.006         4.757     none
    g6_ext    93.7         0.004         5.415     none
    g6_noid   0            0.003         0.004     none

Both identity-weighted arms break the exact peptide geometry their baseline
handed them, by roughly three orders of magnitude, across two different
baselines -- and the arm with identity removed holds bond flat. So the drift is
a property of the identity term, not of the helix and not of the rollout.

No arm converts the movement into a better fold. The rule is therefore capable
of changing the structure; what it cannot do is change it in the direction FAPE
wants, which is a sharper statement than "the geometry is frozen" and is why
[[fape-can-teach-the-frame-head-which-way-to-move]] is the premise under
suspicion rather than the rule's capacity to act.

**The heading used to assert a mechanism** -- "wherever identity is supervised,
the geometry degrades" -- and carried a check-1 rebuttal saying our own runs
were the subject rather than a proxy. The pMHC session's review was right that
the rebuttal was sound for a measurement and covering for a mechanism: a claim
about what a term does to optimisation *in general* is not a claim about our
calculation, and there `ours` is a proxy again. So the heading is narrowed to
what was measured, which is strong, is ours, and needs no defence.

**On the tension with [[identity-supplies-the-trunk-that-geometry-reads]]**,
which the same review raised: both claims rested on
[[g6-noid-never-left-the-helix-baseline]] and read it in opposite directions --
identity is what geometry needs, against identity is what damages it. They are
reconcilable as "supply the representation, damage the output", but the map
asserted both without saying so, and check 2 could not fire because neither
refutes the other. One observation read two ways is the cheapest thing for a
second reader to catch and the hardest for an author. The other node is now
`open`, which resolves the tension by demoting the stronger reading rather than
by asserting a reconciliation neither of us measured.

## Supported by
- [[g6-self-wobbled-around-the-helix-baseline-and-broke-its-bonds]]
- [[g6-ext-never-left-the-extended-baseline]]
- [[g6-noid-never-left-the-helix-baseline]]
