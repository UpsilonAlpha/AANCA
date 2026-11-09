---
type: claim
status: void
sources: [32359791]
---

# FAPE's 10 A clamp is what gates the geometry, and opening it will let the fold move

Stated, under test, **and currently failing**. Recorded here rather than as a
finding because the arm has not finished.

The mechanism is real and tested in isolation: `clamp_max(10.0)` is exactly
flat above the clamp, so 40 A and 80 A out of place score identically and the
gradient on the moved atoms is exactly zero (`tests/test_fape_clamp.py`). About
26 % of pairs are under the clamp on these crops, and those are near-diagonal
sequence neighbours **which every baseline already satisfies exactly**, having
been built by walking exact peptide geometry. So the surviving gradient sits
where there is nothing to fix.

`g6_open` tests it with AF2's own scheme (supplementary 1.9.5): clamp 90 % of
updates, leave a tenth open. The held-out evaluation stays clamped at 10 A
unconditionally, so the reported numbers remain comparable across arms.

**`g6_open` finished all 8,000 updates and the prediction failed.** Held-out
RMSD 17.53 A against the helix baseline's 17.45, having spent the whole run
between 17.44 and 17.69 -- the same band as `g6_self`. Twist ended at 73.4 deg
against the baseline's 64.4, so nine degrees *worse*, and the bond term drifted
to 4.7 as in every identity-weighted arm. Exit 0, 3 of 8,000 updates skipped.

The recorded prediction was "this is the arm that can move". It did not, so the
clamp was not the gate on whole-structure reconstruction. That is the fourth
failed prediction in two days and, like the others, it narrows rather than
wastes: with reach, credit assignment, identity balance and now the clamp all
eliminated for a helix arm, what is left is that FAPE's direction is wrong
([[the-reach-probes-split-the-three-frozen-arms]]).

**The mechanism was not wrong, only its importance on that task.** The clamp
really is flat above 10 A and really does leave only near-diagonal pairs, which
`tests/test_fape_clamp.py` pins. It gates little on *regeneration*, where 90.1 %
of pairs are live rather than 26.5 %
([[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]]) -- so the
flag stays, and the arm that needed it never needed it.

## Reopened 9 Nov on the reasoning, not the result

The result stands: `g6_open` ran AF2's 90 % scheme and did not move
([[g6-open-did-not-move-with-the-clamp-open-a-tenth-of-the-time]]), so opening
the clamp is not the key to whole-structure growth and this node stays `void`.

**What does not stand is the reason given for leaving the flag at 1.0 in
regeneration.** That reason was the live fraction: 90.1 % of scored pairs are
inside the clamp on a damaged window against 26.5 % on whole-structure growth,
so "the arm that needed it never needed it". But a regeneration crop is ~116
intact residues at their crystal poses and a 12-residue window, so the pair
counts are roughly 13,000 intact-intact against 1,400 window-intact. **The 90 %
is almost entirely pairs that are already correct and contribute no gradient.**

So the live fraction and the clamp problem are not in tension, they are the same
observation counted two ways: the pairs inside the clamp are the pairs with
nothing to fix, which is what this node said in the first place. A statistic
over all pairs cannot answer a question about the informative ones.

The quantity that would settle it is the live fraction **restricted to
window-to-intact pairs**, which nothing has measured. At a 13.3 A displacement
it should be near zero, and if it is, `fape_clamp_prob` has never been tested on
a task where FAPE's gradient reaches the residues that need to move --
`g6_open` tested it where 73.5 % of all pairs were dead, which is a different
regime from where 90 % are live but the wrong 10 % are dead.

Cheap: `scripts/baseline_probe.py` already computes the live fraction, and
splitting it by pair class is a few lines. CPU-only.

## Superseded by
- [[the-reach-probes-split-the-three-frozen-arms]]


## Refuted by
- [[g6-open-did-not-move-with-the-clamp-open-a-tenth-of-the-time]]
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
