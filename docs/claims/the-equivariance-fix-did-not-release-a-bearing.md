---
type: observation
status: supported
evidence: ours
sources: [32423296, 32420154]
---

# With the frame bug fixed and the readout corrected, the bearing is 0.008 against a 0.206 null

The decisive number, read off `g10_equiv_ceil`'s checkpoint with
`--equivariant-move` in the run and the readout's frame convention corrected
(job **32423296**, k = 12 held-out crops):

    align    mean +0.008    per crop -0.534 to +0.457
    step          0.3000    100 % of the cap, every crop
    net/path      1.000     every crop
    rmsd_dmg      12.52     per crop 1.75 to 23.92

**Judged against a threshold set before the number was read**, which is the
only reason this reading can be trusted at all:

    pure per-step noise          0.00
    a fixed random vector       -0.4 to +0.25   <- an untrained HEAD
    an untrained EQUIVARIANT     +0.206         <- the convention-matched null
      head, same corrected readout
    half-aimed                   0.67
    perfectly aimed              1.00
    threshold to claim aiming   ~0.5

**+0.008 is below its own null and far below half-aimed, so the equivariance
fix did not release a bearing.** The comparison that counts is against +0.206,
not against the -0.355 null measured on `g8_ceiling`: that one was taken with
the lab-frame convention on a non-equivariant run, and a null is only a null
for the convention and architecture it was measured in.

**The pre-registered distinction resolves to "inside the band", not "flat".**
The judgement written down in advance was that a trained head sitting at one
value on every crop regardless of that crop's motion is *flat*, which is a
different finding from being inside the band. Per-crop values span -0.534 to
+0.457 and the per-step series on the first crop runs 0.131, 0.181, 0.02,
-0.088, 0.335, 0.299, 0.306, 0.426, 0.235, 0.041, 0.017, 0.282. The head
responds to geometry; it responds uninformatively.

One crop reads **+0.457**, above the random-head band and below half-aimed's
0.644. It is one crop in twelve and is recorded as such.
[[the-four-arms-measured-the-architectures-ceiling]] is void partly because a
single favourable point was read as a trend once already, in the 11.82 A dip in
this very arm's curve.

## What did change, and it is not nothing

`step` went from 0.0309 A on `g8_ceiling` to 0.3000 A here -- a tenfold
increase to full saturation of the cap. Unlike `align`, a norm is
convention-free, since rotating a vector into another frame preserves its
length, so the two arms are directly comparable on this column. The fix changed
the rule's behaviour substantially and
[[the-move-heads-magnitude-gradient-is-attenuated-three-thousandfold]] is what
that change cost.

## The agreement check is underpowered, which is not the same as negative

`align` is computed from emitted moves against ideal directions and `rmsd_dmg`
from final coordinates against the native -- two independently derived
quantities that must agree if both read the same reality, which is the only
question that tests an instrument's *convention* as against its sensitivity.
Better aim should mean lower RMSD, so the correlation should be clearly
negative:

    Pearson  r = -0.223   95% CI [-0.707, +0.402]
    Spearman rho = -0.399

The sign is right and **the interval crosses zero**, so k = 12 cannot settle
it. The held-out set is twelve by the split, so the check needs crops the split
cannot supply; `--extra-crops` adds training crops, which is legitimate because
agreement is a property of the instruments rather than of the split, and the
readout labels any `rmsd_dmg` printed that way as not held out on every run.
Job 32449795 runs it at k = 100.

Until that lands, **`align` and `rmsd_dmg` have not been shown to agree**, and
that is a stated limitation of the bearing instrument rather than a reason to
discount this reading, which rests on a calibrated scale with a measured
ceiling.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[the-move-heads-magnitude-gradient-is-attenuated-three-thousandfold]]

## Requires
- [[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]
