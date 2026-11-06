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

`scripts/reach_ceiling_probe.py` reports both per residue against the ceiling,
on the CPU, from a `last.pt`. Until it runs on the three finished arms this
question gates
[[the-rule-can-move-a-residue-far-enough-to-reach-the-native]], and through it
the thesis.

Written as a question rather than a claim so the gap is visible in the map
rather than absent from it.
