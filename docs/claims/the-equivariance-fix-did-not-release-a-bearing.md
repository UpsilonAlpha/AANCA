---
type: observation
status: supported
evidence: ours
sources: [32449992, 32420154]
checked: [32423296]
---

# With the frame bug fixed and the readout corrected, the bearing is -0.096 against a +0.206 null

The decisive number, read off `g10_equiv_ceil`'s **final** 2000-update
checkpoint with `--equivariant-move` in the run and the readout's frame
convention corrected (job **32449992**, k = 12 held-out crops):

    align    mean -0.096    per crop -0.742 to +0.717
    step          0.3000    100 % of the cap, every crop
    net/path      1.000     every crop
    rmsd_dmg      12.85     per crop 2.52 to 23.96

**The first reading of this was mislabelled and is retracted.** Job 32423296
ran 01:05:19-01:10:55 while the training run still owned `last.pt`, which was
overwritten at 01:10:13 — so it measured a complete but *intermediate*
checkpoint around update 1860-1922 and reported it as the arm's result
(`align` +0.008, `rmsd_dmg` 12.52). For as long as it took to read the SLURM
timestamps that looked like instrument non-determinism, which would have been
far worse. It is in `docs/RETRACTIONS.md`, and
[[a-probe-pointed-at-best-pt-silently-used-its-own-defaults]] is the same
failure in its first costume: a probe pointed at a file whose identity nothing
in the output recorded. The readout now prints the checkpoint's update and
mtime.

**The conclusion is unchanged and slightly strengthened**, since -0.096 is
further below the null than +0.008 was. But the published precision was wrong
for an hour and the per-crop table was wrong throughout.

**Judged against a threshold set before the number was read**, which is the
only reason this reading can be trusted at all:

    pure per-step noise          0.00
    a fixed random vector       -0.4 to +0.25   <- an untrained HEAD
    an untrained EQUIVARIANT     +0.206         <- the convention-matched null
      head, same corrected readout
    half-aimed                   0.67
    perfectly aimed              1.00
    threshold to claim aiming   ~0.5

**-0.096 is below its own null and far below half-aimed, so the equivariance
fix did not release a bearing.** The comparison that counts is against +0.206,
not against the -0.355 null measured on `g8_ceiling`: that one was taken with
the lab-frame convention on a non-equivariant run, and a null is only a null
for the convention and architecture it was measured in.

**The pre-registered distinction resolves to "inside the band", not "flat".**
The judgement written down in advance was that a trained head sitting at one
value on every crop regardless of that crop's motion is *flat*, which is a
different finding from being inside the band. Per-crop values span -0.742 to
+0.717 and the per-step series on the first crop runs 0.169, 0.043, 0.004,
-0.077, 0.217, 0.16, 0.132, 0.259, 0.101, -0.047, -0.021, 0.092. The head
responds to geometry; it responds uninformatively, and eight of twelve crops
are negative.

## The one crop that aims is the only one the gradient can follow to arrival

Worth writing down because it is what `g12_trunc` was submitted to test, and
because it would be easy to find afterwards and call a prediction:

    crop   needs    inside the 3.6 A gradient window?   align
     297    2.52 A  yes                                 +0.717
     225    3.87 A  no, just                            -0.742
     the other ten  no, by 6 to 20 A                    -0.568 to +0.467

`+0.717` is **above half-aimed's 0.670**, and it belongs to the only crop whose
damaged window sits close enough to native that the truncated segment covers
the whole journey. 11 of 12 crops need more than the window can see.

**n = 2 on either side of that boundary, so this is a hypothesis and not a
finding.** It is exactly the shape of the error already indexed against this
arm -- the 11.82 A dip read as a trend from one point in seven -- and crop 225
reads the *worst* value in the table from just outside the boundary, which a
clean story would not predict. `--truncate 60` (job 32449661) moves the
boundary out to 18 A and settles it: if the bearing is a function of whether
the gradient reaches arrival, nine more crops should cross over with it.

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

    Pearson  r = -0.349   95% CI [-0.769, +0.281]

(the retracted intermediate checkpoint gave r = -0.223, CI [-0.707, +0.402], so
the statistic is not stable at this k either -- which is the same conclusion.)

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
