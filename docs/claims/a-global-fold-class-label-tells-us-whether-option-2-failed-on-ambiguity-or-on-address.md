---
type: question
status: open
---

# Pre-registered: a global fold-class label separates "too many targets" from "no address" as the reason option 2 failed

**Written before the arm exists.** Ulban, 26 Nov, chose this over the
`context_fraction` sweep and over a frame head, after
[[a-local-rule-with-an-attractor-is-a-diffusion-model-without-a-schedule]] was
refuted. Nothing here is a result.

## This is a DIAGNOSTIC arm and not a rescue attempt

**It cannot make option 2 true.** 5,328 parameters and 400 structures do not
become a protein generator because a label is added, and Proteina's >400M
parameters on a synthetic set ~35x its curated one is the scale at which this
problem is currently solved. What this arm can do is say *which of two
diagnoses* is right, and the two imply different next moves. Registering it as
diagnostic is the point: an arm whose only good outcome is "it worked" is the
arm this project has run thirty times.

## IT IS NOT `morphogen.py`'s MORPHOGEN, and the distinction is load-bearing

Ulban asked for the "fold-class morphogen", and this node is that arm -- but it
must not be confused with `src/aanca/morphogen.py`, which is a different
mechanism with a standing position against part of it.

  * **This is SPECIFIED global conditioning**, in Proteina's sense: one label per
    structure, identical for every residue, broadcast. Nothing diffuses, nothing
    is secreted, nothing is transported.
  * **`morphogen.py` is a FIELD.** Its chain-transported group was rejected
    because with nothing dormant it is exactly `decay ** index`, a positional
    encoding by the module's own criterion
    ([[the-chain-transported-morphogen-is-a-positional-encoding]]), and the
    project's standing position is that **no cell may read a global
    coordinate**.
  * **This arm does not violate that position, and here is why.** A fold class
    is the same number for every residue in the structure, so it is a global
    *scalar*, not a global *coordinate*: it cannot tell a residue where it is.
    That is exactly the property that makes it a discriminator below -- and it
    is also why it is permitted where `decay ** index` is not.

**A better-aligned arm already exists and has never been run**, and it is
recorded here so this one is not mistaken for it.
[[the-emitted-chemical-diffuses-over-the-perception-graph]] is emergent -- the
rule chooses what to secrete, it spreads over the edges the rule already
computes, neighbours read it -- and `morphogen = false` in every arm ever run,
so there is no result to reproduce. **That arm could supply address information
where this one provably cannot**, because a field over the perception graph is a
function of the structure as it currently is. It is the natural successor if
this arm confirms H-address, and it is a cheaper successor than the frame head.

## The two hypotheses, and why a constant label discriminates them

**H-ambiguity.** One rule, with no timestep and no conditioning, was asked to
map one Gaussian prior onto the *whole distribution of folds*. Nothing told it
which protein to build, so the best available map is the one that satisfies the
average pair constraint -- which is a cloud of about the right size and nothing
else. Narrow the target and the constraints become mutually satisfiable.

**H-address.** The deficit is positional. A residue's only positional
information is its sequence index: a 1-D address for a 3-D target
([[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]). Gala's all-pairs loss works
on a point cloud because there the target *is* the constraint set; a protein puts
a 3-D target behind a 1-D address, and no amount of conditioning supplies an
address.

**A fold-class label is identical for every residue in the structure, so it adds
exactly zero positional information.** That is what makes it a clean
discriminator rather than just another lever:

    H-ambiguity true   ->  conditioning moves the pair loss and the geometry
    H-address true     ->  conditioning changes the output distribution but not
                           its quality, because the ambiguity was never the
                           binding constraint

**Either outcome is informative, and the second is the one the record currently
predicts.** `scripts/denoise_floor.py` measured the MVP at 9.1 A against a
weight-free radial rescaling's 10.202 A, and the pair loss at 87 A^2 -- an RMS
distance error of 9.33 A per pair where the bonded pairs' own error is 7.76 A.
**The rule has essentially no pair information at any separation, not a
misallocation of it**, which is H-address's prediction and not H-ambiguity's. So
**the registered expectation is that this arm does NOT help**, and its value is
that it says so cheaply and sends the next arm to the frame head for a measured
reason instead of a hopeful one.

## The label, and it needs no external data

**CATH's C level is defined by secondary-structure content, and this project
already computes it.** `generation_metrics.structure_class` maps P-SEA helix and
strand fractions onto four classes deliberately aligned with Orengo et al. 1997:
`mainly-alpha`, `mainly-beta`, `alpha/beta`, `irregular`. So the morphogen is
derived from the structure by our own validated code rather than fetched:

  * **No network, no new dependency, no CATH download.** The arm cannot be
    blocked by a data problem.
  * **P-SEA is validated, not assumed** -- 1.000 per-residue agreement with
    biotite's `annotate_sse` on all four crystal structures in `data/raw/`
    ([[a-local-rule-with-an-attractor-is-a-diffusion-model-without-a-schedule]]).
  * **It is OURS and must be named as ours.** This is a P-SEA-derived C-level
    *analogue*, not a CATH assignment, and no claim here may say "CATH class".

**Two honest costs, registered now.**

  * **The label is computed on the CROP, not the chain.** A 128-residue crop of
    a larger protein can carry a different class from its parent, and the crop
    is what is supervised, so the crop is what is labelled. A number from this
    arm is about crops.
  * **`irregular` is rare.** The 60-entry natural reference reads 35
    `alpha/beta`, 14 `mainly-beta`, 10 `mainly-alpha`, 1 `irregular`. The class
    counts over the 400 training structures must be printed, and a class with
    too few examples to train is a reported fact rather than a silent
    imbalance.

## Where the morphogen enters, and why not the seed

**Into `phi_m`'s input, at every step.** A small learned embedding of the class,
broadcast to every residue and concatenated to the message function's invariant
input alongside the squared distance and the two hidden states. Three reasons,
each ruling out an alternative:

  * **Not the initial hidden state.** A seed relies on the rule learning to
    *preserve* the signal across 15-25 steps, which is a second question
    confounded with the first. A morphogen is a field that is present
    throughout, not an initial condition.
  * **Not added to the hidden state.** `norm_type = pn` is PairNorm, which would
    normalise a constant offset away. Concatenating at `phi_m` bypasses it.
  * **Equivariance is preserved by construction**, because the embedding is an
    E(n)-invariant scalar vector entering the invariant branch -- the same place
    the squared distance enters. **It must still be tested**, to the 1e-4 the
    existing arms are tested at, with a live k-NN graph.

**Parameter cost stays small**, which is load-bearing for the whole framing: a
4-wide embedding over 4 classes plus the widened `phi_m` input is on the order of
a hundred weights against 5,328. If this arm needs to grow the rule to work, the
"one small local rule" claim is what is being traded away and that trade gets
argued before it is made, not after.

**Label dropout** at a small probability, so an unconditional pass exists and
classifier-free guidance -- Proteina's mechanism -- is available later without
retraining. **CFG is NOT part of this arm**: it needs two forward passes and a
scale, and a guidance scale is a second axis.

## Readouts, in this order, and the first is an instrument check

**0. Does the label reach the output at all?** P-SEA helix fraction of samples
drawn under `mainly-alpha` against samples drawn under `mainly-beta`, **from the
same noise seeds**. If those two distributions are indistinguishable the
conditioning is inert -- a bug, or washed out -- and **nothing below may be
read** until it is fixed. This is the `start_rmsd`-single-valued check of this
arm: the thing that says the instrument works before any result is quoted.

**1. Does it narrow the target?** The `window` pair loss against the MVP's
measured block means, 8.96-9.25 A for `rmsd` and 79.9-98.5 A^2 for `total`. Same
pool, same sigma, same crops, so these are comparable -- and the laptop and
Bunya pools name different proteins at the same index, so the comparison is Bunya-to-Bunya only.

**2. Does it beat the floor?** `scripts/denoise_floor.py`'s `oracle_scale`,
**10.202 A on Bunya's pool**, which the MVP beat by 1.1 A. A conditioned rule
that does not widen that margin has not used the label for anything geometric.

**3. Is there a fixed point?** `scripts/prior_fixed_point.py` at the registered
0.1 A per step within 500 steps. The MVP reached 2 of 15.

**4. Geometry.** The bucket-2 axes against the measured natural reference:
CA-CA 3.8027 A, outlier fraction 0.0087, clash rate 0.0001,
Rg / 2.2 N^0.38 1.101. **The Rg axis is uninformative here as it was there** --
`prior_coordinates` builds the start state at the target Rg, so it passes for a
rule that does nothing, and it is reported rather than counted.

**5. Does the class actually come out?** `structure_class` of each sample against
the class it was conditioned on, as a confusion table. This is readout 0's
quantitative form and the only one that could be a positive result on its own:
**a rule that reliably produces more helix when asked for helix has learned
something about secondary structure even if its bonds are wrong.**

## The conditions that refute this arm

**Readout 0 shows an inert label after the implementation is checked** -- that is
a bug and not a result, and the arm does not report until it is fixed. Then,
with a live label:

  * the `window` pair loss sits inside the MVP's own block-mean spread, OR
  * the margin over `oracle_scale` does not improve on the MVP's 1.1 A, OR
  * fixed points stay below 3 of 15 AND the bond outlier fraction stays above
    0.5.

**Any of those confirms H-address and sends the next arm to the frame head.**
The arm costs one GPU job of about 1 h 55 m at the MVP's measured 0.34 s/update,
one `.npz` of samples, and no new dependency.

## What this does NOT claim

  * **It does not reopen option 2's headline claim.**
    [[a-local-rule-with-an-attractor-is-a-diffusion-model-without-a-schedule]]
    is `unsupported` and this arm cannot restore it; a fixed point from the prior
    was refuted at the registered numbers and a label does not make a rule
    converge.
  * **It is not Proteina and no number here may be set beside one.** Proteina
    conditions a >400M-parameter flow-matching model with classifier-free
    guidance over curated CATH labels and reports designable backbones to 800
    residues. This is a ~5.4k-parameter local rule with a P-SEA-derived label
    and no guidance. **The mechanism is borrowed; the comparison is not
    available**, for the same reason recorded against the diffusion comparison
    generally -- every published metric is gated on ProteinMPNN + ESMFold.
  * **One seed.**
    [[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]
    applies here as everywhere, and this arm is not replicated.
  * **It does not touch the port.** `src/aanca/egnca.py`,
    `scripts/train_egnca.py` and `configs/egnca.toml` stay bit-for-bit, and so
    does `scripts/train_denoise.py` -- the refuted MVP is a quoted result and
    its checkpoint must stay readable by both existing probes.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
