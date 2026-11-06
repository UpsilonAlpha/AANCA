---
type: claim
status: unsupported
---

# FAPE gives the frame head a gradient that points toward the native fold

A premise of the thesis. Four arms have now trained against FAPE and none of
them improved its backbone RMSD by more than measurement noise, so the premise
is recorded as unsupported rather than assumed.

The mechanism that makes it doubtful: `error.clamp_max(10.0)` is exactly flat
above the clamp, so a residue 40 A out of place contributes a constant and no
direction -- 40 A and 80 A score identically, and the gradient on the moved
atoms is exactly zero (`tests/test_fape_clamp.py`). Only about 26 % of pairs
are under the clamp on these crops, and those are the near-diagonal sequence
neighbours, **which every baseline already satisfies exactly** because it is
built by walking exact peptide geometry. So the surviving gradient is
concentrated where there is nothing to fix.

Two candidate explanations were tested and both failed, which is why this node
is the premise rather than either of them: see the `## Refuted by` edges on
[[the-identity-gradient-starves-the-geometry]] and the open status of
[[the-fape-clamp-is-what-gates-the-geometry]].

**As of 7 Nov this is no longer merely the premise left standing after others
failed; there is direct evidence against it.** `g6_self` moves 7.67 A per
residue against a mean need of 8.64 A, coherently, and ends no closer to its
native than it began. The magnitude is right and the result is not, so the
gradient is supplying a distance and not a direction
([[the-reach-probes-split-the-three-frozen-arms]]).

## Refuted by
- [[the-rule-keeps-whatever-conformation-it-is-given]]
- [[the-reach-probes-split-the-three-frozen-arms]]
