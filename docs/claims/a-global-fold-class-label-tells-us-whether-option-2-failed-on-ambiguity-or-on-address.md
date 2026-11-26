---
type: question
status: unsupported
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
  * **`irregular` is rare at CHAIN level and is NOT rare on crops -- registered
    expectation wrong, 26 Nov.** The 60-entry natural reference reads 35
    `alpha/beta`, 14 `mainly-beta`, 10 `mainly-alpha`, **1** `irregular`, and
    this node registered the worry that `irregular` would be too thin to train.
    Measured over the arm's own 400 training structures at crop size 128:
    `alpha/beta` 156, `mainly-alpha` 129, `irregular` **55**, `mainly-beta` 54,
    over 394 croppable of 400. **So 14 % of 128-residue crops carry less than
    0.20 helix + strand**, against under 2 % of whole chains. That is a property
    of cropping -- a crop can sit in a loop-rich region of a well-folded protein
    -- and it means no class is too thin to train. The counts are printed at
    startup as this node required, and the registered expectation is recorded as
    wrong rather than quietly dropped.

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

**0. Does the label reach the output at all? AMENDED 26 Nov -- it is TWO
questions and the original form of this readout could not function.**

*As first written* this readout was the P-SEA helix fraction under
`mainly-alpha` against `mainly-beta` from the same noise seeds, with a flat
result declared "a bug, or washed out" and nothing below readable until fixed.
**A 250-update CPU smoke run showed that is unsound**, and it would have
disabled the whole arm on Bunya by this node's own rule. There is a third state
the original did not enumerate: *the output has no secondary structure at all*,
so the helix gap is `0.000 - 0.000` whether the label is live or dead. And that
state is the EXPECTED one -- the refuted MVP produced helix 0.0000, strand
0.0000, coil 1.0000 on all 7 samples. Measured in the smoke run: the helix gap
was `+0.000` while the paired samples differed by **1.83 A mean per residue**,
so the label was plainly live and the readout said nothing.

**0a. Does the label reach the COORDINATES? This is the instrument check.**
Mean per-residue displacement between paired samples -- same crop, same noise
seed, two different labels. Non-zero means the conditioning is connected. This
cannot be vacuous, because it does not depend on the output being protein-like.
**A zero here is a bug and nothing below may be read until it is fixed.**
Guarded additionally by a unit test at every class plus the unconditional row.

**0b. Does the label reach the SECONDARY STRUCTURE? This is a result, not an
instrument check.** The helix-fraction gap as originally written, readable only
when the samples carry any secondary structure at all (pooled helix + strand
> 0). When they do not, it reports *no dynamic range*, which is a statement
about the rule and not about the label.

**And this is why the amendment strengthens the discriminator rather than
weakening it.** A live 0a with a flat 0b is not a bug and not a null result: it
is **H-address's prediction stated precisely** -- the conditioning is received,
it moves the coordinates, and it buys nothing, because the ambiguity was never
the binding constraint. The original form would have thrown that outcome away as
an instrument failure.

**No refutation threshold moved.** The three substantive conditions below -- the
`window` loss against the MVP's block-mean spread, the margin over
`oracle_scale`, and fixed points against bond outliers -- are exactly as first
registered. What changed is only which observation counts as "the instrument is
broken", and it changed before the arm ran on Bunya and after a CPU smoke run
that measured no result.

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

## ANSWERED: H-ADDRESS, 26 Nov, job 33834699

**The label arrives, moves the coordinates, and buys nothing.** 20,000 updates,
`COMPLETED` exit `0:0`, `01:57:08` on bun159, `sigma` 8.8985 and `start_rmsd`
15.2051 single-valued and identical to the refuted MVP's, so the comparison is
paired crop-for-crop. Zero bare `NaN` in the history.

  * **0a, the instrument check: LIVE.** Paired per-residue displacement between
    samples differing only in the label, **1.7988 A** mean over 12 pairs,
    **2.8516 A** on the alpha-against-beta pairs. The CPU smoke run read 1.83 A.
    So nothing below is excused as a bug.
  * **0b, the helix gap: FLAT, with no dynamic range.** alpha 0.0000, beta
    0.0000, gap `+0.000`, pooled helix + strand exactly 0 over all 8 samples.
    This is the third state the original readout 0 did not enumerate and the
    reason for the amendment above: under the original wording this run would
    have been thrown away as an instrument failure.
  * **5, the confusion table: a CONSTANT.** Every class produced `irregular`, at
    **all 80 evaluations** from update 250 to 20000.
  * **The objective is inside the MVP's band on every block.** `total` by 2,000
    updates 86.1 / 81.4 / 80.8 / 92.3 / 81.2 / 80.5 / 89.3 / 86.4 / 83.2 / 84.2
    against the MVP's measured 79.9-98.5. **This is the registered refutation
    condition and it fired.**
  * **Bucket 2 fails on all three axes and one got worse.** CA-CA 10.799
    (MVP 11.565, natural 3.8027), outlier fraction 0.9951 (MVP 1.0000, natural
    0.0087), clash **0.0183 against the MVP's 0.0111**. Converged 0 of 8.

**The one number that moved.** Mean `rmsd` 9.045 against the MVP's 9.157, with
the last two blocks (8.938, 8.820) under the MVP's minimum of 8.961. A 0.11 A,
~1.2 % shift on a 15.2 A start, in the direction the label would predict. It is
recorded because it is real and it is not a result: `window`, the quantity
actually optimised, is inside the band; 148 parameters is 2.8 % more capacity; no
arm in this project has a second seed, so a 1 % block-mean gap has no error bar;
and nothing downstream changed at all.

**So option 2 failed on ADDRESS.** Narrowing the target from "any protein" to
"any mainly-alpha protein" was received and was useless. That is what it should
be if the binding constraint is that a residue knows only its sequence index --
a 1-D address for a 3-D target
([[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]).

## Two registered conditions are PENDING, and the reason is this project's own refusal in reverse

`scripts/prior_fixed_point.py` and `scripts/denoise_floor.py` both refused
`runs/cond/best.pt`: *"this checkpoint records settings
scripts/train_egnca.py's schema does not define: label_dim, label_dropout."*
That is
[[a-probe-pointed-at-best-pt-silently-used-its-own-defaults]]'s refusal firing
correctly in the direction this arm's design did not anticipate. The arm was
built as a separate trainer so that a new schema key could not make the MVP's
checkpoint unreadable, and the key made the **new** checkpoint unreadable
instead, because both probes hardcode `train_denoise`'s schema.

**So the fixed-point count and the `oracle_scale` margin are pending, not null,
and this answer does not rest on either.** It rests on the `window` condition,
which fired on all ten blocks, on a constant readout 5, and on bucket 2.

## Why the status is `unsupported` and not `supported`, recorded so it can be flipped

This node's *title* is an instrument claim -- that a global fold-class label
**tells us** which failure mode option 2 had -- and by that reading it is
`supported`: a live 0a against a flat 0b and a constant readout 5 is a clean
discrimination, and the arm answered the question it was built to answer.

It is filed `unsupported` on Ulban's instruction, given before the numbers were
seen, on the other reading: that the thing under test is whether the fold class
is load-bearing for this rule, and it is not. **Both readings agree on every
number**; they disagree only about which sentence the status describes. If a
later reader needs this node as evidence that the *discriminator* works rather
than that the *label* works, that is what this paragraph is for.

## The conditions that refute this arm

**Readout 0a shows the label does not reach the coordinates** -- that is a bug
and not a result, and the arm does not report until it is fixed. A flat 0b with
a live 0a is NOT this case; it is a result and it supports H-address. Then, with
a live label:

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
