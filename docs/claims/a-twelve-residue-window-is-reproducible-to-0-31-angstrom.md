---
type: observation
status: supported
evidence: experimental
sources: [deposition_anchor.py]
---

# Between two depositions of one protein, a 12-residue window reads 0.31 Angstrom

Built because `rmsd_dmg` had a floor and a ceiling and **no scale**. Measured on
13 X-ray pairs drawn from RCSB's 100 %-identity clusters, 825 sliding windows,
each superposed on every matched residue outside it -- which is what `rmsd_dmg`
does with a damaged window against its surviving context, using the *same*
code (`fape.ca_rmsd_on`, extracted from `backbone_rmsd_on` for this purpose):

    median 0.31 A    10th 0.095    90th 0.98    max 7.04

So **0.3 A is the noise floor of this metric and ~1 A is where real
disagreement starts.** Two crystallographers, given the same sequence, place
the same twelve residues within about a third of an Angstrom of each other.

Against [[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]],
the usable dynamic range of `rmsd_dmg` is **0.31 to 10.77 A**. Every arm this
project has run scores 12.18 to 12.85, which is outside that range entirely --
worse than doing nothing, as already recorded.

## The gate is the whole measurement, and the first run did not have it

A window RMSD is read *after* a fit to the context, so **a pair whose global
fit fails makes every window read large for reasons that have nothing to do
with the window.** Without that gate the pooled figure was **0.620 A with a
90th percentile of 14.5 A** -- two populations wearing one number:

    13 pairs, global fit <= 2 A    window median 0.31 A   <- the anchor
    12 pairs, global fit 2.5-17.7  window median 7.76 A   <- something else
     4 pairs, mixed methods        window median 14.5 A   <- excluded

Four of the pairs had a median window of 17-19 A **with their minimum also
above 10 A**, which is not one flexible loop but the entire correspondence
failing. `global`, the CA RMSD over all matched residues, is derived
independently of the window values and has to agree with them before any of
them mean anything -- the convention check this project has been applying to
everything else, applied to a new instrument on the day it was built.

## Three traps avoided, two of them only because they were looked for

  * **AlphaFold entries in the cluster file.** `clusters-by-entity-100.txt`
    includes `AF_*` and `MA_*` computed models, and splitting them on `_`
    yields the token `AF`, which the first run reported as the partner for
    nineteen of its twenty best candidates. Fetching those would have measured
    **how far a prediction sits from a crystal** and reported it as
    crystallographic reproducibility: a plausible number of roughly the right
    size answering a different question, which is the same shape as the probe
    that reported `birth free` for a `--birth none` run.
  * **Matching residues by position.** The two entries have different
    missing-residue sets, so a positional match compares residue *i* to *i+3*
    wherever a loop is unmodelled. Matched on author numbering within a chain
    instead; only residues present in both are used.
  * **Method mixing.** An X-ray/NMR pair measures the difference between two
    techniques. Reported separately, and their median is 14.5 A, so pooling
    them would have moved the anchor by a factor of forty.

## This does NOT close the gap it was built for

[[the-success-criterion-has-no-positive-anchor]] is about the absence of an
anchor for a **good non-native** answer -- an interface regrown into something
that binds better would score badly against the crystal and be the result
wanted. What is measured here is the metric's *reproducibility*, which tells us
what "as good as the crystal" means and nothing about what "better than the
crystal" scores. That claim stands open.

**The 12 gated-out pairs are the route to it and are not yet an answer.** Same
sequence, global fit 2.5-17.7 A, window median 7.76 A: on the face of it, the
scale at which "a different valid conformation" lives. But that population is a
**mixture** -- 1VC3/2EEO reads global 3.72 A with windows in a 0.6 A band,
which looks like a rigid-body or domain difference, while 4IT4/4IT6 reads
global 12.44 A with windows from 3.1 to 38.5 A. Separating real alternative
states from broken correspondences needs each pair inspected, and until that is
done **no number from this population should be quoted.**

## Supports
- [[the-success-criterion-has-no-positive-anchor]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
