---
type: observation
status: supported
evidence: ours
sources: [32383664, 32383695, 32383733]
---

# g6_self moved the right distance coherently and gained nothing; g6_noid moved 0.12 Angstrom

`scripts/reach_ceiling_probe.py` on each arm's `last.pt`, held-out crop 180 of
412, 120 steps against the 36.0 A ceiling:

    arm       path   net   net/path   % of ceiling   needs (mean)
    g6_self   7.69   7.67    0.998       21.4%          8.64 A
    g6_ext    5.78   5.74    0.993       16.0%         19.48 A
    g6_noid   0.12   0.07    0.575        0.3%          8.64 A

**Three different failures, and only one of them is the one that was
suspected.**

`g6_noid` asks for nothing. 0.3 % of its movement budget over 120 steps, which
is the probe's own third branch -- "the rule has learned to ask for nothing at
all, and neither the ceiling nor the window is what is stopping it". With the
identity term removed the frame head emits approximately zero, which is exactly
what [[identity-supplies-the-trunk-that-geometry-reads]] predicts: no identity
gradient, no trunk, nothing for the frame head to read.

`g6_ext` is genuinely short. It moves 5.74 A coherently and its residues are
19.48 A from native on average, 69.50 A at worst, so the ceiling binds for it
as the arithmetic said.

**`g6_self` is the one that matters, and it is a fourth case the probe did not
anticipate.** It moves 7.67 A, in an almost perfectly straight line (0.998),
against a mean need of 8.64 A. So it is not out of budget -- it uses a fifth of
it -- and it is not thrashing. It moves very nearly the right *distance*, and
its held-out RMSD did not improve by more than noise over 8,000 updates.

The only reading left is that the **direction** is wrong. Neither the reach
ceiling nor credit assignment over steps is the binding constraint for a helix
arm; the rule is being told how far to go and not where.

## Supports
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]

## Refutes
- [[fape-can-teach-the-frame-head-which-way-to-move]]
