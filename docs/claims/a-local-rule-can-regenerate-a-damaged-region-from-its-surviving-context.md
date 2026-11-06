---
type: thesis
status: unsupported
---

# A local per-residue rule can regenerate a damaged region of a protein from the context that survived

The project's assertion, corrected on 6 Nov after the map was first rooted in
the wrong one. **Regeneration from damage is the goal**, not reconstruction:
delete a part of a protein, apply the rule, and have it fill the region back in
using only local information from the intact remainder.

This is the Growing-NCA objective -- regeneration from damage, stability past
the trained horizon -- applied to protein geometry, and a prior-art search on
2026-10-27 found it unpublished. The novelty is the **objective**, not the
architecture: existing packers predict in a fixed number of passes or along a
fixed diffusion schedule, and none trains for a fixed point that must be
reached from many starting conditions over a step count it was never told.

`unsupported`, and the gap is larger than it looks. **No run in this project
has yet trained on regeneration.** Every geometry arm to date starts the whole
structure from a constructed baseline with one native anchor per covalent run,
which is a different and strictly harder task
([[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]).

## Supports
- [[regrowing-a-designed-interface-can-produce-a-better-binder]]
- [[a-peptide-motif-can-be-grown-a-scaffold-around-it]]

## Requires
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
- [[fape-can-teach-the-frame-head-which-way-to-move]]
- [[the-damaged-region-must-start-within-reach-of-where-it-belongs]]

## Supported by
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
