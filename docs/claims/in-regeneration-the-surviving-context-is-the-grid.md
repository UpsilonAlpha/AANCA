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

**Status `open`.** The Growing-NCA half is literature and the
protein half is an argument, not a measurement. What would make it a finding: a
morphogen or positional-encoding channel on the whole-structure task moving an
arm off its baseline, or a regeneration arm shown to improve as the intact
context is made more informative. Neither has been run.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
