---
type: observation
status: supported
evidence: experimental
sources: [model.py]
---

# A damaged window must travel 10.1 Angstrom and the gradient can only ever see 3.6 of it

Model-free, measured on the twelve held-out crops a run sees, 12-residue
window, helix baseline. Classed `experimental`: the distance is to **deposited
structures**, and the two bounds are arithmetic on top of it.

    per-step translation cap        0.3 A
    budget over 120 steps          36.0 A
    inside the 12-step window       3.6 A   <- all the gradient ever sees

    the window must travel         10.10 A mean, 36.92 A worst

    budget / need                   3.6x    not binding
    gradient window / need          0.36x   BINDING

**So on the regeneration task the truncation window binds and the per-step cap
does not**, which is the opposite of the whole-structure case where
[[the-per-step-translation-cap-is-0-3-angstrom]] disqualified `g6_ext`
outright. A rule trained here is being taught about a third of the journey it
has to make.

This is pre-positioning rather than a result: it says which rollout
hyperparameter to test first if
[[question-can-the-architecture-place-residues-under-any-supervision]] comes
back saying the architecture is the constraint. `--truncate` is already a
trainer flag, so the ablation costs nothing to express.

**And it makes a sharp prediction about the pair of runs now in flight.** The
ceiling check teacher-forces a target at every step in the live window, so it
teaches a *policy* -- "move toward where you belong" -- which then applies at
all 120 steps at inference. Truncation should therefore not limit `g8_ceiling`
the way it limits a run learning from the outcome. If `g8_ceiling` succeeds and
rung 2 then fails, the 12-step window is the difference between them, and this
measurement is why.

One case worth keeping: the worst window must travel 36.92 A against a 36.0 A
budget, so a small share of them are unreachable in 120 steps however good the
rule is. That is a property of the sampling, not of the rule, and it puts a
floor under `rmsd_dmg` that no objective can lift.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[question-should-a-secondary-model-calibrate-the-rollout-parameters]]
