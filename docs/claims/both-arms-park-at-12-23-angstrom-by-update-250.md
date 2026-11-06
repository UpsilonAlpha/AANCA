---
type: observation
status: open
evidence: ours
sources: [32389742, 32405172]
checked: [32405172]
---

# Two unrelated objectives both park the window at 12.23 Angstrom by update 250 and never move it

`g7_regen` optimises FAPE at 1.0. `g8_ceiling` teacher-forces every emitted
move against the exact ideal pose. They share no placement term, and:

    updates   g7_regen  g8_ceiling     bond (g8)
       250       --       12.23          6.035
       500      12.23     12.23          6.033
       750       --       12.23          6.029
      1000      12.27     12.24          7.596
      1250       --       12.22          6.044
      1500      12.27     12.25            --
      3500      12.30       --             --

**12.23 is reached by update 250 and then held to three significant figures
across 1,250 and 3,500 updates of completely different gradients.** That is not
convergence; it is an attractor of the rollout rather than of either objective.

**The artefact reading was checked and refuted.** Two different objectives
landing on the same number looks like a measurement fault, and this project has
produced four of those
([[a-probe-pointed-at-best-pt-silently-used-its-own-defaults]]). So the two
`rmsd_dmg` paths were compared on the start state, crop by crop, on the same
twelve held-out crops: the probe's and the trainer's agree **exactly** at 10.77
A mean and to two decimals on every individual crop. The baseline is right and
the 12.23 is real.

What the numbers do say is where it comes from. The start state scores 10.77 at
one step and 12.23 after 120, so the rule applies a small systematic drift --
about 0.012 A per step -- a hundred and twenty times, and parks. Both arms do
the same thing, so whatever sets that drift is not the placement term.

**And the mechanism is currently unreadable, which is a fault of my own
instrumentation rather than of the runs.** `growth_objective` computes the
ceiling term only `if move_weight:`, and `evaluate()` never passed the weight,
so the held-out `move` was structurally absent for g8_ceiling's entire run. The
distinction it would settle needs opposite fixes:

  * `move` falling while `rmsd_dmg` sits still -- the rule learns the right
    moves and they do not accumulate into placement, which points at the
    rollout's composition: the bernoulli update schedule, moving a residue
    about its own CA, or moves being applied and undone.
  * `move` flat -- the placement gradient is not reaching the frame head, and
    [[question-can-the-architecture-place-residues-under-any-supervision]]
    answers no.

Fixed, and the next arm is the same experiment made readable rather than a new
one. `status: open` until it is.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
