---
type: claim
status: open
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

At 5,625 of 8,000 updates the held-out RMSD is 17.45 to 17.51 against the
baseline's 17.45 -- the same band as `g6_self` -- while identity trains
normally (recovery 21.5 %). **The recorded prediction was that this is the arm
that can move; so far it has not.** If it finishes there, the clamp was not the
gate either.

The reason this node is `open` and not already `void` is that 2,375 updates
remain and the prediction deserves its full run before being written off, which
is the mistake made this morning on a partial log
([[g6-self-wobbled-around-the-helix-baseline-and-broke-its-bonds]]).

## Refuted by
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
