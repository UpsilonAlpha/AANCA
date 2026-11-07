---
type: observation
status: supported
evidence: experimental
sources: [32462158]
---

# The baseline is 12.23 Angstrom on the pool the runs train on, which is what every arm scores

Run on **Bunya**, the machine every arm trained on, 412 structures, the same
twelve held-out crops, crop 128, `--damage 12`:

    baseline     fit_all  fit_intact    diff   crops
    helix          12.23       13.26    1.02      12
    extended       23.55       25.71    2.16      12
    random         12.45       13.51    1.06      12

`fit_all` is the mask `growth_objective` uses, so it is the column comparable
with a training log
([[rmsd-dmg-superposes-on-the-damaged-window-it-is-measuring]]).

**The helix baseline is 12.23 A, and every arm this project has run scores
12.18 to 12.85.** Nine arms, a tenfold range of step size, four frame
conventions, an identity term present or absent — all of them sitting on their
own baseline to three significant figures. An untrained rule with a perturbed
head reads 12.26.

## This retires 10.77 A, and the retirement explains nine arms at once

[[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]] reported
10.77 A from the same script, and **that figure was measured on a different
structure pool** — a laptop checkout holding 1,903 files against Bunya's
24,709. Both take `available[:412]`, so a different set of files on disk means a
different 412 structures and different crops. Re-running the probe locally today
reproduces 10.77 A exactly; re-running it on Bunya gives 12.23 A. Same code,
same seed, same crop indices, different proteins behind those indices.

So the "learned nothing" line every arm has been judged against was computed on
structures no arm has ever seen, and the comparison was never like for like.
**The arms are not 1.5 A worse than doing nothing. They are doing nothing**,
which the rest of the evidence has been saying by other routes:

  * [[both-arms-park-at-12-23-angstrom-by-update-250]] — two objectives sharing
    no placement term both land on 12.23 and hold it. Not an attractor of
    training, and not the measurement fault it was suspected of being: **12.23
    is the baseline**, so the arms never left their start state.
  * [[the-rollout-gains-one-angstrom-and-it-is-not-the-rules]] — on a matched
    fit mask the rollout closes about zero, and an untrained rule matches a
    trained one.
  * [[the-equivariance-fix-did-not-release-a-bearing]] — `align` says no arm
    aims, by the one instrument anchored on scale and convention.
  * [[the-rule-keeps-whatever-conformation-it-is-given]] — stated before any of
    this, and the number now says it outright.

Four independent routes to one statement, which is the strongest thing in this
map: **the rule does not move the window.** The previous framing — "worse than
doing nothing" — gave the arms a spurious property they do not have, and read a
pool mismatch as a result.

## The two instruments never actually disagreed

[[two-instruments-disagree-by-2-5-angstrom-about-where-the-window-starts]] asked
whether the window starts 10.77 or 13.28 A out. On one pool the probe's
intact-fit figure is **13.26 A** and the readout's `need` is **13.28 A** —
agreement to 0.02 A. The 2.5 A split 1.06 A of fit mask and 1.46 A of pool, and
neither was a disagreement about the quantity. Both halves were mine.

**Why this took three steps to see.** The pool is invisible in every output
either script produces: both print "412 structures on disk", which is true on
both machines and identical, while the 412 differ. A diagnostic that reports its
sample size and not its sample cannot show this, and `baseline_probe` now prints
the first and last structure id of the pool it used.

## Supersedes
- [[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]]

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]
- [[the-rule-keeps-whatever-conformation-it-is-given]]
- [[two-instruments-disagree-by-2-5-angstrom-about-where-the-window-starts]]
