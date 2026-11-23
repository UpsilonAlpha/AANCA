---
type: question
status: supported
sources: [33321241, 33321243, 33320678]
---

# The architecture can place residues correctly under *some* amount of supervision

**ANSWERED YES, 23 Nov, after a day that passed through NO and back.** Under
teacher forcing the architecture reaches `align` **+0.367** (`g14_pairwise`,
33321241) with the rotation applied in the **lab frame**, and six arms with
that convention read 0.159 to 0.384. The two arms with the conjugated local
rotation -- `g28` 0.036 and `g25` 0.122 -- are the only ones that do not, and
the readout's frame convention was re-derived from the code and **cleared**, so
that split is a property of the rule and not of the instrument. **It is the
conjugated rotation that leaves the bearing at zero, not the architecture.**
[[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]] carries the
re-derivation and the mechanism. The arm chain is not ended by a null, because
the null belonged to one setting.

**The reading below is kept as the record of the NO this passed through**, with
its reasoning, because the scope it was given before the number existed is what
made the flip detectable.

**ANSWERED NO, 23 Nov 2026, by `g28_ceil_local` (33308293), against buckets
pre-registered and pushed before the number existed.** The top rung was run in
the one configuration that was never confounded -- teacher forcing, a sound
actuator, identity off, no break -- and `align` came back **+0.036, 95% CI
[-0.043, +0.114]**, the whole interval inside the pre-registered null band. The
rule is handed the exact move at every one of 2,000 updates and still emits a
bearing indistinguishable from zero.
[[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]] carries the
guard reading, the control, the corroboration and the scope.

**REOPENED, 23 Nov, hours later.** `align` read on six older arms sorts
cleanly by `lab_rotation` and by nothing else -- every lab-frame-rotation arm
0.159 to 0.384, both local-rotation arms 0.036 and 0.122 -- and `g14_pairwise`
is `g28` with that convention flipped and reads +0.367 against `g28`'s 0.036.
The readout rotates the emitted move into the lab frame for comparison, so the
instrument's convention may match one group and not the other, in which case
`g28` read near zero by construction.
[[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]] carries the
full account and is withdrawn to provisional. **So this question is open again
and the arm chain is not ended**, until the frame convention is re-derived --
which costs no GPU and must come before anything else.

**The NO as it stood:** The clause was written
into the pre-registration before submission: a null here is the most
informative outcome and the one that should end the chain rather than extend
it, so no terms get added and nothing gets re-run. Everything below is the
question as it stood, kept because the reasoning that made it the gate is what
made the answer worth having.


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

**Answered on 7 Nov: no.** After 2,000 updates of being handed the exact ideal
move at every step, the emitted bearing is statistically indistinguishable from
a randomly initialised head -- trained -0.259, random head -0.355, paired
difference +0.096 with 95 % CI [-0.235, +0.427] over twelve crops
([[the-teacher-forced-rule-emits-moves-pointing-away-from-the-target]]). And
the reason is architectural rather than a training failure: the perception is
invariant and the move was applied in the lab frame, so the direction was never
representable
([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]).

So the ladder's top was never reachable with the architecture as it stood, and
every rung-1 result is a measurement of that rather than of an objective.
`--equivariant-move` is the repair and `g10_equiv_ceil` is its first arm; at
250 updates it reads `rmsd_dmg` 12.18 against `g8_ceiling`'s 12.23, so the fix
alone has not yet moved the outcome and the question is now whether it is
necessary-but-not-sufficient.

Both answers were useful and only one was comfortable. If it drops hard, the
capacity exists and every rung above measures how little supervision suffices.
If it barely moves, **the constraint was never the objective** and three days of
loss design were aimed at the wrong layer -- which would be the most valuable
thing the project has learned, and the reason to spend 70 minutes of GPU on a
term that must never ship
([[the-objective-should-restrain-the-output-not-teach-the-placement]]).
