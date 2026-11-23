---
type: claim
status: open
sources: [Mordvintsev 2020]
---

# A Growing NCA cell is told where it is by the lattice, and in regeneration the intact context is that lattice

**Why the whole-structure arms failing does not refute the thesis, and why
regeneration is the nearer target for a mechanical reason rather than because it
is easier.**

In Growing Neural Cellular Automata a cell sits at a fixed site in a grid. The
lattice hands it three things for free:

  * **where it is** -- the site *is* the position, so no cell ever has to infer
    its own coordinates;
  * **a shared frame** -- every cell's up is the same up, so "move left" means
    one thing across the whole automaton;
  * **a determinate neighbourhood** -- who is adjacent is a property of the
    substrate, not something the cell works out.

A residue in this project has **none** of that. Perception is invariant by
construction, there is no ambient lattice, and the only frame available is the
residue's own N-CA-C triad, which is exactly the frame that moves when the
residue does. So
[[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]
was never only a bug: it was the symptom of an architecture asked to do a
Growing-NCA task **without the information a Growing NCA is given**. The four
whole-structure arms therefore measured a rule starved of positional
information, not the impossibility of the thesis -- and
[[the-four-arms-measured-the-architectures-ceiling]] is already `void` for a
neighbouring reason.

**The claim, then, is that the thesis is short of information rather than
false**, and the missing piece is a positional-information channel: a morphogen
gradient, which is the protein-shaped analogue of the grid, not more
supervision. That is consistent with
[[the-objective-should-restrain-the-output-not-teach-the-placement]]: the fix
for too little information is to supply information, and the fix for too much
supervision is to remove it, and the project has been reaching for the second
lever while the first was the one stuck.

## And this is exactly what regeneration already has

**In the regeneration task the surviving residues are the lattice.** A damaged
window sits inside an intact context that is still in its crystal pose, so the
context supplies, for free, the same three things the grid supplies a cell:
where the window is, a frame to read a direction in, and who its neighbours
are. That is a mechanistic reason -- not merely a difficulty argument -- for
[[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
being the near target and the one to aim at, while the thesis stays at the root
because it stays possible.

It also sharpens where the current failure sits. The positional information is
*present* in the regeneration arms and the rule still does not use it: the
latest reading is that `rmsd_dmg` cannot distinguish a trained rule from an
untrained one
([[the-rollout-gains-one-angstrom-and-it-is-not-the-rules]]), and `align` says
no arm aims. **So on this task the grid is already there and the rule is not
reading it**, which is a different and more tractable problem than the
whole-structure arms had, and it is the one worth working on.

## Two of the three, not three

Sharpened since: the context supplies *where it is* and *a frame*, and it does
**not** supply the determinate neighbourhood. The perception graph is a k-NN
selection rebuilt from the live coordinates every step
([[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]), so a
residue 13 A out reads the neighbours of the wrong place -- the one thing this
node said a cell gets from the substrate rather than working out. The published
automaton that does succeed at this task message-passes over the *target* graph
([[an-equivariant-graph-nca-morphs-random-coordinates-into-a-target-point-cloud]]),
and the solvability condition for placing a node from local distances at all is
in
[[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]].

**Status `open`.** The Growing-NCA half is literature and the
protein half is an argument, not a measurement. What would make it a finding: a
morphogen or positional-encoding channel on the whole-structure task moving an
arm off its baseline, or a regeneration arm shown to improve as the intact
context is made more informative.

**CORRECTED 23 Nov: the native-edge half HAS been run, three times, and this
note said it had not.** `native_edges` held the perception graph at the native
edge set and was a real setting until it was retired on 9 Nov as a leak (the
retirement banner is in `configs/regeneration.toml`, and
`tests/test_config.py::test_the_retired_arms_can_no_longer_be_expressed` keeps
it unexpressible). Against the 12.23 A baseline of its own era:

    g13_natedge   32489939   native edges                    settled 11.42
    g15_both      32494313   native edges + pairwise          settled 10.24
    g17_long      32505354   g15 + updates = 10000            settled  6.69

**CORRECTED WITHIN THE DAY: these are not comparable to `g25`'s 8.10 A, and
saying so was a category error.** The recorded command lines in
`tests/test_config.py::RETIRED` show `g13_natedge` and `g15_both` both ran
**`--move-weight 1.0`**, and `g17_long` is `g15` with more updates, so **every
native-edge arm was TEACHER-FORCED** -- the exact ideal move supplied at every
step. So 6.69 A is a ceiling number with *two* ceilings stacked, the given
graph and the supplied move, and it belongs beside `g8`/`g10`/`g28`'s ceilings,
never beside `g25`'s 8.10 A, which ran `move_weight 0`. Three further reasons
no number here is quotable as it stands: they are **`fit_all`**, from before the
superposition correction, so they compare only to the 12.23 A baseline of their
own era and not to the `fit_intact` bar; **`align` was never read on any of
them**, so the one instrument the capacity question turned on has no value for
the arms bearing most directly on it; and the config banner says the test holds
all four retired command lines when it holds two.

**What survives, and it is the useful part:** within their own era and with
both ceilings held fixed across them, closing the edge-set difference moved the
number 12.23 -> 11.42, and adding the pairwise half moved it to 10.24. Those
are within-era, like-for-like steps. The run directories all survive on the
cluster, so the `align` readout costs no GPU.

What the retirement was right about is separate and still stands: training on
the native edge set hands over the structural interaction topology, which is
the thing an ablation on the rule would be trying to recover, so those weights
cannot carry the payoff even when they carry the claim.


## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
