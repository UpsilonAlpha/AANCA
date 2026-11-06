---
type: observation
status: open
evidence: ours
sources: [32405172, 32418879]
---

# Handed the exact ideal move at every step, the rule's emitted bearing is indistinguishable from random

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

**Corrected within the hour: "anti-correlated" overstated it.** At k = 12 the
spread is wide enough that zero is inside the interval -- sd 0.465, se 0.134,
t = -1.93 on 11 df, 95 % CI **[-0.555, +0.037]**. So the honest reading is that
the emitted bearing is **uncorrelated** with the one the residue needs, not
opposed to it. I wrote "anti-correlated" in this node and in a commit message
before doing the arithmetic, hours after taking the pMHC session's point that
the reportable quantity is the unit count. The discipline is easy to state and
easy to skip on one's own headline number.

**The correction strengthens the diagnosis rather than weakening it.**
Uncorrelated is exactly what
[[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]
predicts: a near-constant lab-frame vector, scored against target directions
that vary by crop, gives cosines scattered about zero. Anti-correlation would
have needed its own explanation. What is *not* ambiguous is the step length --
0.0309 A against a 0.30 A cap, spread 0.0302 to 0.0315, which is a constant to
three digits across twelve different structures.

The untrained control (32419198) completes the picture: `align` exactly 0.000
and `step` exactly 0.0000 on every crop, because the frame head is
zero-initialised and emits nothing at all. So training did change the rule --
it learned to emit a fixed 0.031 A step -- and that step carries no information
about where the residue belongs.

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
