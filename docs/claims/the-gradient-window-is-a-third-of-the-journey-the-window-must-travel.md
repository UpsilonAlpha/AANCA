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

**The 10.10 A need is one of the two numbers in dispute.** Measuring the start
state directly inside a rollout gives **13.28 A** on the same crops, which
would make the ratio 0.27x rather than 0.36x --
[[two-instruments-disagree-by-2-5-angstrom-about-where-the-window-starts]] is
the open question. The conclusion below is unaffected: the window binds on
either figure, and the cap binds on neither.

**So on the regeneration task the truncation window binds and the per-step cap
does not**, which is the opposite of the whole-structure case where
[[the-per-step-translation-cap-is-0-3-angstrom]] disqualified `g6_ext`
outright. A rule trained here is being taught about a third of the journey it
has to make.

This is pre-positioning rather than a result: it says which rollout
hyperparameter to test first if
[[the-architecture-can-place-residues-under-some-supervision]] comes
back saying the architecture is the constraint. `--truncate` is already a
trainer flag, so the ablation costs nothing to express.

## REFUTED IN PART, 22 Nov: the arithmetic stands, "the binding constraint" does not

`g26_fullbptt` raised `truncate` from 12 to 120 -- the exact test this note
named -- and placement got **1.68 A WORSE**, from `g24_frozen`'s 8.49 A to
10.17 A, which is *above* the measured geometric floor of 10.05 A and so no
signal at all
([[fixing-the-actuator-helped-a-little-and-fixing-the-gradient-reach-made-it-worse-than-geometry]]).

**What survives is the measurement, which is what this note is titled on.** The
window does have to travel ~10-13 A and BPTT does only see 3.6 A of it. That
arithmetic is untouched.

**What does not survive is the inference.** Calling the truncation window *the
binding constraint* assumed that a constraint which binds is one whose removal
helps. Removing it hurt. So the ratio is a true description of what the rule is
taught and **not** an explanation of why it fails, and this note must not be
cited as grounds to expect a gain from more gradient reach. That is why the
refutation does not reach the title: it bites the conclusion drawn here, not the
quantity measured here, and the status stays `supported` on the measurement.

Two things the refutation does **not** establish, since conflating them would be
the same error mirrored: an intermediate `truncate` was never tried, so nothing
says 24 or 40 would also hurt; and full BPTT through a 120-step recurrent rule
may simply be badly conditioned, which would be a fact about the optimisation
rather than evidence that gradient reach is irrelevant.

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
- [[the-architecture-can-place-residues-under-some-supervision]]
- [[a-secondary-model-should-calibrate-the-rollout-parameters]]
