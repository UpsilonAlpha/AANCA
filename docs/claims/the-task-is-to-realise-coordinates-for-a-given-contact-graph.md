---
type: claim
status: open
sources: [Gala 2024, Ingraham 2023]
---

# The task is to realise coordinates for a given contact graph, not to discover the graph -- so the native edge set is the task statement and not a leak

**Ulban's decision, 23 Nov 2026**, taken when the two payoffs the project had
been holding at once were put to him as a choice. It settles a question that
had been answered implicitly, differently, in different places.

## What it decides

In `Gala 2024`'s pattern-formation task the target graph is an **input**: the
problem is *"realise coordinates for this graph"*, and reading the graph is the
task statement rather than a shortcut through it
([[an-equivariant-graph-nca-morphs-random-coordinates-into-a-target-point-cloud]]).
That is the task this project is mapping to proteins. The protein instance is
contact-graph-to-coordinates, which has a protein-scale precedent in Chroma's
predict-then-solve: a GNN predicts inter-residue geometry, then a separate step
solves for a global consensus structure
([[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]).

**The alternative, now explicitly NOT the task, was for the rule to discover
the interaction topology**, so that an ablation or an autoencoder on the
weights could recover it. That is a strictly *harder* problem than the
published one, and the published result licenses no expectation about it.
Trying to satisfy both is a large part of why the arm chain produced
twenty-eight arms that each ruled out something small.

## What follows immediately

  * **The 9 Nov retirement of `native_edges` was protecting the discarded
    payoff.** Its reasoning -- hand over the contact graph and "the rule never
    learns it, so there is nothing in the weights to read back" -- is sound
    *only* under discover-topology.
    `tests/test_config.py::test_the_retired_arms_can_no_longer_be_expressed`
    says so in its own docstring: the removal exists to protect "the
    structural-interaction topology the whole interpretability argument needs".
    Under this decision that test enforces a constraint from a task the project
    is not mapping, and the setting should come back as the **task statement**
    rather than as a ceiling.
  * **`g13`/`g15`/`g17` stop being leaky ceilings and become the nearest thing
    to a baseline the project has** -- with one ceiling removed from them and
    one still in place. Their given graph is now legitimate. Their
    `--move-weight 1.0` is not: teacher forcing supplies the answer and is a
    diagnostic in any framing.
  * **[[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]
    changes meaning without changing a single fact.** That the edge set is a
    k-NN of the live coordinates remains exactly true and exactly as measured.
    It stops being a defect to be diagnosed and becomes a **departure from the
    task statement** -- the project was solving a harder problem than the one
    it meant to.
  * **The grid node's "two of the three, not three" is reversed.** Under
    realise-coordinates the intact context supplies all three of the lattice's
    gifts -- where you are, a frame, and now a determinate neighbourhood --
    because the third comes from the task definition rather than having to be
    inferred ([[in-regeneration-the-surviving-context-is-the-grid]]).

## The one arm this specifies, and it has never been run

Every native-edge arm was teacher-forced, so **the combination that the task
actually calls for does not exist in the record**:

    native edge set        the task statement, not a ceiling
    move_weight = 0        NO teacher; the rule is not handed the answer
    all-pairs distance     unclamped, the loss Gala's authors call NECESSARY
    short rollout, FULL BPTT   ~15-25 steps differentiated end to end
    no step cap
    small network          5K parameters succeeded; ours has 229,588

**Two of those invert what the project has been doing**, and both inversions
come from reading the published method rather than from a result here. The
rollout should get *shorter and fully differentiated*, not longer: Gala trains
over 15 to 25 steps total with a replay pool, where we run 120 steps with
`truncate 12`, and `g26_fullbptt` raised truncation to 120 over 120 steps while
leaving the 0.3 A cap in place -- so "full BPTT made it worse" tested neither
the short rollout nor the removal of the cap
([[fixing-the-actuator-helped-a-little-and-fixing-the-gradient-reach-made-it-worse-than-geometry]]).
And the network should get *smaller*.

**It changes several settings at once on purpose.** That breaks the
one-variable discipline the chain used, and the justification is that this is a
**replication target rather than an ablation**: one-variable arms are correct
when ablating *from* something that works, which this project has never had.
Replicate, then ablate from a working baseline.

## The live tension this does not resolve

Weighting the full pair set cuts against
[[the-objective-should-restrain-the-output-not-teach-the-placement]], and that
node is not thereby refuted -- an under-determined target is a different
target, not weaker supervision.

**Decided 23 Nov: the port weights the FULL pair set.** The objection below is
recorded as the thing to watch rather than as a reason not to.

**That measured objection is weaker than it looks under this decision, and the
reason is worth stating.** The window-to-window half is
unweighted because walking the window *linearly* from the baseline to its
native pose makes that half rise from 8.7 to 44.7 A^2 before falling, since a
straight line between two poses passes through a contracted intermediate. That
measurement is sound and it is about **linear interpolation**, which is not the
path gradient descent takes. Gala's loss is all pairs and their rule succeeds,
so the measurement shows the straight line is a bad path rather than that the
loss is a bad objective. It is a reason to watch the internal half, not a
settled reason to leave it unweighted.

**Status `open`.** This is a decision about what the project is for, and what
would make it a finding is the arm above moving off the baseline. Nothing has
been run under it.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[in-regeneration-the-surviving-context-is-the-grid]]
