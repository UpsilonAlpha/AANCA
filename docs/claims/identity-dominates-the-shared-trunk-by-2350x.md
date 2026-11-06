---
type: observation
status: supported
evidence: ours
sources: [32315224, 32315225]
---

# Identity's gradient on the shared trunk is 2,350x FAPE's on g6_self and 10,400x on g5_ctrl

`scripts/growth_balance_live.py --crop`, pointed at each run's `last.pt` so the
arguments travel with the weights: 16 seeds, 120 steps, truncate 12, held-out
crop 180 of 412, correct birth mode read from the weights.

    trunk gradient        FAPE      identity    ratio
    g5_ctrl  (grown)    3.57e-03    3.72e+01    10,400x
    g6_self  (self)     3.96e-03    9.31e+00     2,350x

The trunk column is the only one where a ratio means anything. Adam normalises
each parameter's step by that parameter's own gradient RMS, so a uniformly
smaller gradient on a *separate* head still takes full-size steps and the ratio
cancels; it bites only where parameters are **shared**, because there the two
gradients are summed before Adam sees them.

Caveat: one rollout, one crop, `--repeats 1`. Order of magnitude, not
precision. The `activation` and `birth` columns read exactly zero on both runs
because at 120 steps with 16 seeds the crop is finished and `completion` is
0.0000 with `draws 0`, so this table says nothing about the growth heads.

Three earlier attempts at this measurement were invalid, all for the same
structural reason; see
[[a-probe-pointed-at-best-pt-silently-used-its-own-defaults]].

## Supports
- [[identity-supplies-the-trunk-that-geometry-reads]]
