---
type: observation
status: supported
evidence: ours
sources: [32405172, 32418879, 32420910]
checked: [32405172]
---

# After 2,000 updates of exact supervision, the bearing is indistinguishable from a randomly initialised head

`g8_ceiling` ran 2,000 updates with `--move-weight 1.0`, teacher-forcing every
emitted move against the exactly computable ideal pose, and its `rmsd_dmg` went
12.23 to 12.25. `scripts/ceiling_readout.py` then read the *bearing* off its
checkpoint, on **k = 12 held-out crops**:

    align   mean -0.259    spread -0.942 to +0.624
    step    mean 0.0309 A  spread 0.0302 to 0.0315   (cap 0.30)
    move    mean 0.4894
    rmsd_dmg mean 12.25

`align` is the cosine between each emitted translation and the one that would
reach the native CA, over the damaged window and the live segment.

**Corrected twice, and the second correction is the one that matters.**

First: "anti-correlated" overstated it. At k = 12 the spread puts zero inside
the interval -- se 0.134, t = -1.93 on 11 df, 95 % CI [-0.555, +0.037].

Then the control landed and showed **zero was never the right reference
point.** A randomly perturbed, untrained head scores `align` **-0.355**
(se 0.119) on the same twelve crops, because a near-constant vector over 120
steps overshoots: once a residue is past its target, "toward the target"
reverses and the late-step cosines go negative. **-0.36 is what no information
looks like here**, and the trained rule sits on it:

    trained     -0.259   se 0.134    2000 updates, ideal move handed over
                                     at every step
    random head -0.355   se 0.119    no training at all
    paired diff +0.096   se 0.150    t +0.64, 95 % CI [-0.235, +0.427]

**Training left the bearing statistically indistinguishable from a randomly
initialised head.** That is the answer to
[[question-can-the-architecture-place-residues-under-any-supervision]], and it
is `supported` rather than `open` because the null is now measured instead of
assumed.

Both of my earlier readings used a null of zero and both were wrong about it.
The pMHC binder session insisted the control was load-bearing and gave a
reason; the real reason is better than either of us stated -- **the statistic's
null is not zero, and only the control could say what it was.** Nothing about
the mechanism would have told me.

**The per-step series says how, and it is not what was guessed.** The pMHC
session asked whether a correct bearing was being *re-aimed* each step from a
drifting frame -- right everywhere, arriving nowhere, the two-frames question
pointed at time rather than space. Measured, it is not: the per-step cosine
runs 0.498, 0.351, 0.435, 0.093, 0.088, -0.113, ... -0.375 while net
displacement over path length is **0.995**. A walk re-aimed from a rotating
frame would show a ratio far below 1. This is a straight line whose bearing
decays through zero -- aimed once, never corrected, which is what a constant
vector does when it overshoots. The mean hid that shape entirely.

What is *not* ambiguous is the step length: 0.0309 A against a 0.30 A cap,
spread 0.0302 to 0.0315, a constant to three digits across twelve different
structures. The cap is therefore not throttling a correct bearing -- the
trained rule uses a tenth of it, and the random head used two fifths.

The first control (32419198) was **vacuous and I reported it as evidence.** A
fresh frame head is zero-initialised, so it emits the zero vector and the
cosine divides by a `clamp_min(1e-8)` norm: `align` exactly 0.000 is the
arithmetic of dividing by nothing, not a measurement that the bearing is
random. It establishes only that the 0.031 A step is learned. `--perturb`
replaced it, and a step emitting nothing is now skipped rather than scored
zero.

**A loss value could not have shown this.** `move` tracks `rmsd_dmg` at a near
constant ratio, about 50x, because its position half is the same per-residue
distance the RMSD is, averaged over live steps and diluted by the intact
residues. "The loss fell but the RMSD did not" is close to impossible by
construction, so the instrument had to be directional.

**The term's sign was checked before this was written down.** A sign error
between the emitted translation and the scalar produces exactly this number and
is indistinguishable from a finding about the architecture. Differentiating the
loss with respect to the emitted translation, on a purely translated structure,
gives `cos(descent step, ideal) = +1.0000` on every residue: descent moves the
translation toward the native CA. The pre-existing test asserted only that the
loss *value* is lower along the ideal bearing, which is weaker than the
gradient being right and can come apart from it. One limit: that structure is
translated and not rotated, so the rotation half's gradient is legitimately
zero there (4.1e-18) and the check says nothing about it.

`status: open` because the explanation is not settled. The frame head's own
last layer receives this gradient directly; what it *reads* is the trunk, and
[[identity-dominates-the-shared-trunk-by-2350x]] measured the trunk to be
identity's by three orders of magnitude. A correctly-signed gradient on an
output layer, applied to a representation another term is shaping, can compose
into moves that point elsewhere -- which would also explain
[[both-arms-park-at-12-23-angstrom-by-update-250]], where two arms with no
placement term in common settle on the same number.

`g9_ceil_noid` (32419399) is the test, one variable away: `--identity-weight 0
--identity-track on`, so the head and blind perception stay and only the
identity gradient goes. If `align` turns positive, the trunk is the mechanism
and the fix is architectural rather than a reweighting -- which
[[g6-noid-never-left-the-helix-baseline]] already shows cannot work alone,
since with identity gone and FAPE the only driver the rule asked for 0.3 % of
its budget. If `align` stays negative,
[[question-can-the-architecture-place-residues-under-any-supervision]] answers
no.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[both-arms-park-at-12-23-angstrom-by-update-250]]

## Refutes
- [[fape-can-teach-the-frame-head-which-way-to-move]]
