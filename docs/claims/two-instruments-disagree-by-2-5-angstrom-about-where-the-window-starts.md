---
type: question
status: open
---

# Does the damaged window start 10.77 or 13.28 Angstrom from native?

**The "learned nothing" line every arm has been judged against is in doubt by
2.5 Angstrom, and this is written as a question because it is not yet settled.**

    baseline_probe.py --damage 12, helix        10.77 A
    ceiling_readout.py `need`, measured on
      the run's own start state                 13.28 A  (mean of 12 crops,
                                                          median 13.13)

Two instruments, nominally the same quantity, on nominally the same twelve
held-out crops, with the same `manual_seed(seed + index)` damage draw.
[[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]] says so in
as many words: "on the same twelve held-out crops a run sees".

**Why it matters more than its size.** 10.77 A is the line above which an arm
"has made the window worse than leaving it where the walk put it", and every
arm in the project reads 12.18-12.85. Against 10.77 they are all 1.5 A worse
than doing nothing; against 13.28 they are 1.0 A better than their own start
state. The two readings point in opposite directions, and the project's
headline summary of nine arms rests on which one is the start.

**The likely answer, which is not an excuse for either.** The two construct
different start states rather than computing different arithmetic: the baseline
probe builds its own helix placement for the damaged window, while the run's
comes out of `--initial helix` inside the rollout. If so, `need` is the right
comparator for a run and 10.77 A has been the wrong one -- a cross-sample
comparison of the kind already retracted once in this project, where a twist
figure was compared against a different crop sample's and the arm turned out to
be the worst of three.

**The check is cheap and decides it.** One crop, both code paths, print the
damaged window's start coordinates and the RMSD each path computes from them.
If the coordinates differ the question is answered and
[[the-rollout-gains-one-angstrom-and-it-is-not-the-rules]] supplies the
replacement comparator; if they match, one of the two superpositions is wrong
and that is worse.

Nothing is being restated or withdrawn on the strength of this yet. It is here
so that the next summary of "every arm is worse than doing nothing" has to go
past it.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
