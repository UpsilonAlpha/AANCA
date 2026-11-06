---
type: claim
status: unsupported
---

# One local update rule, applied per residue, can build a whole protein backbone from a baseline

**This was the map's thesis for about an hour on 6 Nov, and it was the wrong
root.** It is a claim, and a strictly stronger one than the project's actual
thesis: a rule that can build an entire backbone from a constructed baseline
could certainly fill a damaged window into intact context, so if this held it
would support
[[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]].
The reverse does not follow, which is the whole point -- and it is why four
arms failing at *this* is not evidence against the thesis
([[the-four-arms-measured-the-architectures-ceiling]]).

How the error happened is worth keeping. I inferred the thesis from the last
three weeks of work -- the growth arms -- instead of reading the README and
HANDOFF §1, both of which state the objective plainly. argmap's own guide calls
getting the thesis as an assertion "the single most important step and the
easiest to skip", and skipping it meant every node's relevance was being judged
against a question the project is not asking.

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

Its own premises are kept below, because they are what the four arms actually
tested and both of the hard ones failed.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Supported by
- [[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]

## Requires
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
- [[fape-can-teach-the-frame-head-which-way-to-move]]
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
