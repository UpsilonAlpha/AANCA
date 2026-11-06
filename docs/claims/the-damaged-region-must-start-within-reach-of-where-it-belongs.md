---
type: claim
status: unsupported
sources: [growth.py]
---

# A regeneration start state has to park the damaged region within the rollout's reach

A premise of the thesis, and currently **not** satisfied by the code -- though
it is closer than it looks.

`baseline_coords` already takes an arbitrary bool mask of which residues keep
their crystal pose, so handing it `~damaged` instead of one-anchor-per-run
preserves the context exactly. Measured on 1UBQ, window 40:52, with the other
64 residues native:

    intact residues moved      0.00e+00 A     context exactly preserved
    CA-CA inside the window    3.804 A        ideal geometry, as designed
    damaged residues moved     73.96 A mean   parked past the 36 A ceiling
    CA-CA at the N-ward join   64.78 A        not hinged off its neighbour

**The walk order is wrong for this use.** `baseline_coords` runs a walk out of
every anchor and later walks overwrite earlier ones, so a window surrounded by
anchors is last written by a walk from the *furthest* anchor in its run rather
than the adjacent one. With one anchor per run that is harmless -- there is
only ever one walk -- which is why it has never shown up.

So a regeneration start state needs the walk to begin at the intact residue
adjacent to the window. That is a change to anchor ordering, not to geometry.
Both joins should stay open afterwards so the rule has to close them: loop
closure is part of the task rather than something to hand it.

Until this holds, a regeneration arm would start its damaged residues 74 A out
and hit the very ceiling that
[[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]
argues regeneration escapes.

## Supported by
- [[the-per-step-translation-cap-is-0-3-angstrom]]
