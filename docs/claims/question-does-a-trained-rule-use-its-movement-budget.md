---
type: question
status: open
---

# Does a trained rule spend its 36 A of movement budget, and does it spend it coherently?

The ceiling in [[the-per-step-translation-cap-is-0-3-angstrom]] is arithmetic
and settles `g6_ext` outright. What it cannot say is whether the 17.45 A arms
were stopped by it, because two very different failures are indistinguishable
in a held-out RMSD:

  * **path length near the ceiling, net displacement near zero** -- the rule
    moves hard and incoherently, and the fix is credit assignment over steps: a
    longer truncation window, or a loss on intermediate steps.
  * **path length near zero** -- the rule has learned to ask for nothing, and
    neither the ceiling nor the window is what stops it.

**Answered on 7 Nov, and the answer was a fourth case neither branch
anticipated.** `g6_self` spends 21 % of its budget in an almost perfectly
straight line (net/path 0.998) and travels 7.67 A against a mean need of
8.64 A -- so it is neither out of budget nor thrashing. It moves very nearly
the right distance and gets no closer, which leaves only the direction.
`g6_noid` is the second branch, spending 0.3 % of its budget: it asks for
nothing. `g6_ext` is the first, coherent and genuinely short.

Full numbers in [[the-reach-probes-split-the-three-frozen-arms]].

Written as a question rather than a claim so the gap is visible in the map
rather than absent from it.
