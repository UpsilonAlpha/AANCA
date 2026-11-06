---
type: observation
status: supported
evidence: ours
sources: [32420519, 32420520, 32420521]
---

# g6_self walked half the distance in a straight line and gained nothing; g6_noid barely moved

`scripts/reach_ceiling_probe.py` on each arm's `last.pt`, **k = 12 held-out
crops**, 120 steps against the 36.0 A ceiling:

    arm       path   net   needed   net/path   path spread   net spread
    g6_self   7.71   7.54   13.94     0.982    7.39-9.74     7.37-7.80
    g6_ext    5.95   5.75   28.77     0.973    5.72-7.69     5.67-5.85
    g6_noid   0.29   0.07   13.94     0.530    0.12-2.13     0.07-0.07

**Re-measured at k=12 on 7 Nov, and the denominator moved materially.** The
first version of this node read a single crop -- `indices[0]`, crop 180 -- and
reported `g6_self` travelling 7.67 A against **8.64 A needed**, which is 89 %
of the distance and licensed the reading "the right magnitude, the wrong
direction". Crop 180 turns out to have an unusually short journey: over twelve
crops the requirement is **13.94 A**, so the arm travels **54 %** of it.

What survives and what does not:

  * **Robust**: the split between arms, and the straight lines. `path` and
    `net` reproduce to within noise (7.69 -> 7.71, 7.67 -> 7.54) and their
    spreads are tight, so the three arms really do behave differently and all
    three move coherently or not at all.
  * **Not robust**: "very nearly the right distance". That was a property of
    one crop's requirement, and the honest statement is that `g6_self` covers
    about half the distance in a straight line and gains nothing -- which is
    what a near-constant vector of roughly half the typical magnitude would
    do, and fits
    [[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]
    better than the original reading did.

So the k=1 version overstated how well-calibrated the magnitude was while
getting the direction conclusion right. Both halves needed the re-run to
separate.

**Three different failures, and only one of them is the one that was
suspected.**

`g6_noid` asks for nothing: a net 0.07 A over 120 steps, 0.2 % of the 36 A
budget, and the only arm whose little movement is incoherent (net/path 0.530
against 0.98 for the others). That is the probe's own third branch -- "the rule
has learned to ask for nothing at all, and neither the ceiling nor the window
is what is stopping it". With the identity term removed the frame head emits
approximately zero.

`g6_ext` is genuinely short: 5.75 A coherently against a 28.77 A requirement,
so a fifth of the way, and the ceiling binds for it as the arithmetic said.

**`g6_self` is the one that matters.** It moves 7.54 A in an almost perfectly
straight line (0.982) against a mean need of 13.94 A. So it is not out of
budget -- 36 A is available and it uses a fifth -- and it is not thrashing.

The reading that survives is that the **direction** is wrong. Neither the reach
ceiling nor credit assignment over steps is the binding constraint for a helix
arm, and a straight walk of half the required distance that arrives no closer
is a magnitude without a bearing. What the k=12 numbers remove is the stronger
claim that the magnitude itself was right.

## Supports
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]

## Refutes
- [[fape-can-teach-the-frame-head-which-way-to-move]]
