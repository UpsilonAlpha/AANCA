---
type: observation
status: supported
evidence: ours
sources: [train_growth.py]
---

# FAPE was a hardcoded 1.0 in every growth arm, and no run has ever been de novo

Checked on 7 Nov because the premise came up and turned out to be wrong, which
is the kind of thing this map exists to stop being wrong twice.

**`--fape-weight` did not exist until 7 Nov.** `growth_objective` took
`fape_weight` as a parameter and the trainer never passed it, so it sat at its
default of 1.0 in g1 through g7. The comment beside it read "nothing should
ever set it to 0 in training", so the absence was deliberate rather than an
oversight -- and its reasoning assumed FAPE is what teaches placement, which
[[the-reach-probes-split-the-three-frozen-arms]] contradicts.

**The FAPE-free runs are the other track.** `rollout.py`'s repacking objective
contains zero references to FAPE: v1-v9 optimised von Mises on chis, harmonic
bond and angle restraints, clash and identity. There was no backbone to place,
because those runs repack side chains on a *fixed native* one. The only other
`fape_weight: 0.0` in the tree is in `growth_balance_probe.py`, which measures
gradients and does not train.

**And nothing has run de novo.** `growth_objective`'s own docstring says so:
"De novo growth is the goal and this is not it -- a native target is available
here, so this trains the machinery without needing the distribution growth does
not have."

So the project has occupied exactly one rung of the information ladder, which
is what [[question-can-the-architecture-place-residues-under-any-supervision]]
is about.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
