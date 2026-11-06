---
type: thesis
status: unsupported
---

# One local update rule, applied per residue, can build a protein backbone

The project's assertion. A single rule reads a residue's neighbourhood and
emits a small change to that residue's pose; applying it everywhere, for many
steps, is supposed to assemble a fold. Nothing about the fold is global: there
is no superposition step, no template, no per-residue parameter.

**`unsupported`, as of 6 Nov, and that is the honest status rather than a
placeholder.** No run has yet produced a backbone measurably closer to its
native than the conformation it started from. Four arms of 8,000 updates have
each ended at their own starting baseline
([[the-rule-keeps-whatever-conformation-it-is-given]]), so there is no
observation anywhere in this map that is evidence *for* the thesis -- only
observations about why it has not been reached.

The parts that do work are real and should not be read as support for the
whole: identity prediction trains to 22 % recovery with NLL below the ln-20
floor, the self-placing architecture does what it was built to do
([[a-residue-can-be-placed-without-a-parent-placing-it]]), and the rollout,
data pipeline and optimiser are all demonstrably functioning. The geometry is
the part that has never moved.

This node exists so the premises below can be listed and checked. Two of the
three are not holding.

## Requires
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
- [[fape-can-teach-the-frame-head-which-way-to-move]]
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
