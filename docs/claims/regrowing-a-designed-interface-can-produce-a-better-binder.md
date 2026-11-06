---
type: claim
status: unsupported
---

# Destroying a designed binder's interface and regrowing it yields something that binds better

The application the project exists for, and the reason the objective is
regeneration rather than prediction. Take an interface between a designed
protein and its target, destroy the designed side of it, and let the rule
regrow it from local context -- with the target still present, so the context
the rule reads *includes* the thing the result has to bind. The hope is that
accurate side-chain placement against a real partner beats what the original
design process produced.

**`unsupported` and entirely untested.** Nothing in this project has been run
on a complex, an interface or a binder. It is recorded here so the map shows
the distance between what is measured and what is wanted: every observation in
this map is backbone geometry on single-chain crops.

Two things it needs first. Regeneration training that works at all, which is
the thesis. And a perception graph that spans two chains -- which may already
hold, since the rule reads a spatial neighbourhood rather than a chain, but has
never been checked, and `growth_links` is explicitly covalent so a target chain
would be a separate covalent run.

The measurement that would support it is also not yet defined here: "binds
better" is not FAPE to a crystal structure, and every loss in this project
supervises against a native answer. That gap is worth naming before the
capability arrives rather than after.
