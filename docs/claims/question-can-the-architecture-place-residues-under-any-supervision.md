---
type: question
status: open
---

# Can this architecture place residues correctly under *any* amount of supervision?

**The gap the map was hiding, written as a question so it is visible at check 3
rather than absent.** Seven arms have failed and not one of them establishes
that the rule *can* place residues. Every failure so far has been diagnosed as
a property of the objective -- the clamp, the identity balance, the reach
ceiling -- and each diagnosis assumed the capacity was there to be unlocked.
None of them checked.

What varied across g1-g7 was the **architecture's** information, downward:

    torsion    inherits parent pose x reference x spin   most
    cartesian  inherits position, learns orientation
    free       no birth head, reference pose
    none       inherits nothing, every slot alive         least

while the supervision held at FAPE 1.0 the whole way
([[no-growth-arm-has-ever-trained-without-fape]]). That is the reverse of
establishing a ceiling and then removing information to find what is necessary.

    rung  information given                      status
    0     exact per-step move (teacher forced)   g8_ceiling, running
    1     FAPE 1.0, native target                g1-g7, all failed
    2     FAPE loose + clash + bond + reach      built, not yet run
    3     physical restraints only, no native    not built
    4     de novo, no native at all              the goal

**This question gates the whole map**, which is why the thesis `requires` it: a
rung 2 result is uninterpretable until it is answered. A failure there could not
be told from the 0.3 A per-step cap
([[the-per-step-translation-cap-is-0-3-angstrom]]), the 12-step truncation
window, or one-hop perception.

`g8_ceiling` (32405172) is the measurement: `--move-weight 1.0` teacher-forces
every emitted move against the exactly computable ideal, 2,000 updates, read as
`rmsd_dmg` against the model-free 10.77 A
([[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]]).

Both answers are useful and only one is comfortable. If it drops hard, the
capacity exists and every rung above measures how little supervision suffices.
If it barely moves, **the constraint was never the objective** and three days of
loss design were aimed at the wrong layer -- which would be the most valuable
thing the project has learned, and the reason to spend 70 minutes of GPU on a
term that must never ship
([[the-objective-should-restrain-the-output-not-teach-the-placement]]).
