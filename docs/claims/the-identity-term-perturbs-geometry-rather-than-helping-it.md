---
type: claim
status: supported
sources: [32215014, 32241722, 32314557]
---

# Wherever identity is supervised, the peptide geometry degrades and the fold does not improve

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

**On check 1's finding that every observation beneath this is `ours`:** it is,
and it stands, for the same reason as
[[identity-supplies-the-trunk-that-geometry-reads]] -- the claim is about what
our own training run does to its own bond term, so our own runs are the
subject, not a proxy for one.

## Supported by
- [[g6-self-wobbled-around-the-helix-baseline-and-broke-its-bonds]]
- [[g6-ext-never-left-the-extended-baseline]]
- [[g6-noid-never-left-the-helix-baseline]]
