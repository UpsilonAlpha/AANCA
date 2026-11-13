---
type: question
status: open
---

# Does the spread of repeated regrowths measure a loop's conformational freedom, and does it agree with anything outside this project?

Ulban's hypothesis, 13 Nov, and it reorders the queue. If low entropic cost is
what makes a binder both good and *predictable*, the quantity worth measuring
is **local conformational freedom** -- and a rule that can destroy and regrow a
window repeatedly can be asked for it directly: regrow one loop a hundred times
and the spread of the outcomes is that freedom. A pairformer can be asked for
an affinity; it cannot cheaply be asked how many ways a loop can sit.

## The measurement I had queued is the wrong one

Two different quantities were both being called "dispersion", and only one of
them is this:

**Temporal dispersion.** One rollout, continued past the 120-step training
horizon to 1,000, measuring how far the settled coordinates wander. Answers
*does the attractor hold* -- a persistence check, and the thing E(n)-GNCA
reports when it maintains an attractor 1,000 steps past training. This is what
was on the queue
([[the-growth-objective-supervises-a-fixed-final-step-while-the-track-that-converged-samples-one]]).

**Ensemble dispersion.** **N** rollouts from the *same* checkpoint and the
*same* damaged window, differing only in the rollout generator, each run to
step 120, measuring the spread of the N settled poses. Answers *how many ways
can this loop sit* -- conformational freedom.

The hypothesis needs the second. They are not the same number and a rule could
easily be tight in time and broad in ensemble, which would be exactly the
physical picture: one basin, visited differently.

**It is available now and costs nothing.** `update_schedule = "bernoulli"` makes
per-cell updates stochastic, so two rollouts from one checkpoint already
diverge without any change to the rule. Inference only, no training, no GPU
queue, and `g20_orient`'s checkpoint is a trained rule with no ceiling in the
rollout
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]).
The readout is a per-residue RMSF over the N regrowths.

## The part that matters most: it has an external anchor

Every observation in this map is `evidence: ours`, and the validator has
carried a standing finding about exactly that all week -- a calculation
performed on an input is evidence about the calculation.

**Crystallographic B-factors are an outside quantity for this one.** Deposited
structures carry per-atom B-factors, and a high B-factor is a poorly ordered,
mobile atom. If per-residue regrowth dispersion correlates with B-factor across
the 12 held-out crops, then the flexibility readout is agreeing with something
measured in a diffraction experiment by someone else. Alternate conformations
(`altloc`) and multi-model NMR entries are two further anchors in the same
deposited files.

That would be the first `experimental` observation under the thesis, and it is
the cheapest route to one the project has.

## The tension this exposes in the objective

**If dispersion is the product, then driving `rmsd_dmg` down on its own can
destroy it.** A rule that collapses to a single tight answer has zero dispersion
and says nothing about entropy -- a perfect regressor to the crystal pose is the
*worst* possible flexibility instrument. So the two quantities are in tension
and the objective currently rewards only the first.

This is what Ulban's 9 Nov correction was actually about. "Must converge to a
fixed point" was not a terminological slip: an exact fixed point is a zero-width
basin, and a **bounded basin** is both the physical object and the one that
carries the measurement
([[a-local-per-residue-rule-can-build-a-protein-backbone]]).

**And it predicts something about the ceiling arms, testable on checkpoints
that already exist.** `g17_long` was told the ideal move at every step; a rule
trained that way has been pushed toward one answer. If g17's ensemble
dispersion is *lower* than g20's at a *better* `rmsd_dmg`, then teacher forcing
suppressed the quantity the project is for -- a second and independent reason
the ceilings were bad, arrived at from the opposite direction to the
interpretability argument. That is a paired comparison over two existing
checkpoints and needs no new training at all.

## What would answer this

  1. Per-residue RMSF over N regrowths of the same window, on `g20_orient`'s
     checkpoint. Pick N from where the RMSF estimate stabilises rather than by
     taste.
  2. The same on `g17_long`'s checkpoint, for the paired ceiling comparison.
  3. Spearman of per-residue dispersion against per-residue mean backbone
     B-factor on the same 12 crops.
  4. Temporal dispersion as a separate column, because the persistence question
     is still open and this is the cheapest time to answer both.

Open questions before building it: whether the stochastic update schedule gives
enough spread to be a physical ensemble or merely sampling noise -- which
(1) answers by itself, since noise would show no structure across residues --
and whether B-factor is the right anchor given it also absorbs crystal contacts
and refinement choices.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]

## Requires
- [[question-has-any-arm-been-replicated-at-a-second-seed]]
