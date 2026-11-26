---
type: question
status: unsupported
---

# Option 2, pre-registered: a local rule whose fixed point is a protein is a diffusion model with no schedule and no timestep

## ANSWERED NO, 26 Nov, by the run this note was written to interpret

**Bunya job 33794584, all 20,000 updates, `COMPLETED` exit `0:0` in 01:55:11.
The first registered refutation condition is met: no fixed point within 500
steps, 13 of 15 samples.** Measured by `scripts/prior_fixed_point.py` at the
numbers registered below -- 0.1 A per step, 500-step cap -- and not at the
trainer's own `conv` column, which samples at 0.01 A and 200 steps and is
stricter on both axes. Median `last_move` 0.5379 A/step, 5x outside the
tolerance; the two that converge are 6SHR at 212 steps and 3V69 at 287.

**And the held-out series is flat over the whole run.** Ten block means of eight
evaluations each: 9.159, 9.212, 9.227, 9.135, 9.231, 9.162, 9.101, 9.163, 8.961,
9.221 A, against a `start_rmsd` of 15.205 that is single-valued to 15 decimal
places at every one of the 80 evaluations. The fall happens inside the first 250
updates and nothing happens in the remaining 19,750.

**The fall is also mostly not learning**, which only became visible once a
baseline existed. `scripts/denoise_floor.py` loads no weights and scores the
same crops at the same sigma: rescaling the corrupted coordinates to match the
native radius of gyration -- one parameter, one number about the target, no
model -- reaches **11.194 A**, and the best radial rescaling that exists reaches
**10.202 A**. The rule reads 9.1. **So 5.1 of the 6.1 A fall is a one-parameter
geometric correction and 1.1 A is the angular content of the whole run.** This
note registered `start` as the comparison and that was the wrong comparison;
`oracle_scale` is the right one and it did not exist until the result did.

**The generated structures are not chains.** Every virtual bond in every sample
is outside the 3.80 +/- 0.10 A window (outlier fraction 1.0000 against the
natural 0.0087), mean CA-CA 11.565 A, clash rate 0.0111 against 0.0001, and
P-SEA reads helix 0.0000 / strand 0.0000 / coil 1.0000 with all 7 samples
classed `irregular`. Bucket 4's registered failure mode was a helix-biased
generator; the actual outcome has no secondary structure at all, which is worse
and is a different thing.

**One registered condition was badly drawn and should not be reused.** "The
geometry fails all three of bucket 2" cannot be met, because `prior_coordinates`
constructs the start state at exactly `2.2 N^0.38` and the Rg axis therefore
passes for a rule that does nothing. Two of three axes fail decisively; the
third is uninformative by construction. The refutation stands on condition one
regardless.

**What survives, and the separation is the result.** The port's attractor is
real and is unaffected: a damaged 12-residue window with ~116 residues pinned at
their crystal pose returns to the same endpoint within 0.0105 A at 10,000 steps
([[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]). That is a
fixed point of a *conditional* map with the answer pinned at its boundary.
**This note conflated it with a fixed point of the unconditional map, and the
two are now separated by measurement**: remove the pin and the native graph and
there is no fixed point and no protein. The port demonstrates realisability, not
generation.

**The comparison with diffusion was never reached and now will not be.** The
samples fail the geometry gate by three orders of magnitude on bond outliers, so
designability via ProteinMPNN + ESMFold -- the phase 2 priced below -- would be
measuring nothing. Space was never the blocker and is still not; the rule is.

---

**Written before the MVP runs, and before the generation metrics exist.** Ulban,
26 Nov, chose option 2 over option 1 and asked for a minimum viable product
comparable with diffusion. This note is the design, the buckets and the
refutation conditions. Nothing here is a result.

## The claim that would be worth making, stated so it can fail

Every published backbone generator -- RFdiffusion, FrameDiff, Chroma, Genie 1
and 2 -- is **a large network run on a fixed schedule with the timestep as an
input**. Genie 2 uses 1,000 steps at 15.7 M parameters; RFdiffusion 50 steps at
~59.8 M; FrameDiff 500 at 17.4 M.

**The object proposed here is different in kind, not in degree: ONE rule of
5,328 parameters, no timestep input, no schedule, applied to its own output
until it stops moving.** If a protein backbone is a *fixed point* of a local
per-residue rule, then generation is "run until convergence" rather than
"integrate a reverse SDE", and the published E(n)-GNCA headline property --
holding an attractor 1,000 steps past a 15-25 step training horizon -- is the
mechanism rather than a curiosity.

That is the thesis [[a-local-per-residue-rule-can-build-a-protein-backbone]]
in its generative form, and it is the first version of it this project has been
able to state as a measurement.

## Why option 1 was not chosen, recorded so it is not relitigated

Option 1 is sequential growth: add a residue, relax, add another.
[[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
makes its elementary step **ill-posed, not merely hard**: four already-placed
non-coplanar anchors fix a residue exactly, and a residue at a growing terminus
has **one** bonded parent, leaving a 2-sphere of answers that no amount of
training selects among. Consistently,
[[the-backbone-should-be-grown-sequentially]] is `unsupported` and
[[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]] is
`supported`.

**The deeper statement, and it is the sharpest form of the grid objection.**
Growing NCA works because a lattice gives every cell a persistent **2-D address
for a 2-D target**. A protein gives every residue a persistent **1-D** address
-- sequence index -- for a **3-D** target. The chain-transported morphogen is a
real positional encoding, but it encodes position *along the chain*, which is
the dimension we already had. The deficit is two dimensions, and the obvious
patch of supplying identity as the extra address has been tried and hurt:
[[identity-dominates-the-shared-trunk-by-2350x]] and
[[the-identity-term-perturbs-geometry-rather-than-helping-it]].

**Option 1 is sequenced after, not binned.** Option 2 with a growing node set
*is* option 1, except the new residue joins a system that already has a fixed
point and a populated neighbourhood instead of dangling off a terminus.

## The design, and it is ONE axis removed from something that works

The port reads 2.53 A because of two things that are now exactly what must go:
the ~116 surviving residues **frozen at crystal pose**, which supply the
anchors that make each placement determined, and the **native** contact graph.
[[the-divergence-is-a-missing-published-tanh-not-a-property-of-the-rule]] and
[[the-port-must-reproduce-the-published-result-on-point-clouds-before-it-is-run-on-proteins]]
carry that result.

`scripts/train_denoise.py`, a second trainer rather than a flag on the first:

    corruption        all coordinates, isotropic Gaussian, sigma log-uniform
                      in [0.5, 30] A -- NOT a 12-residue damage window
    timestep input    NONE. The same rule at every noise level. This absence
                      is the scientific content.
    graph             k-NN rebuilt from where residues CURRENTLY are, which is
                      [[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]
                      promoted from failure mode to research question
    context_fraction  THE axis. 0.0 is pure generation and IS the MVP;
                      1.0 is approximately the port
    loss              `pairwise_distance_restraint`, full pair set, plain
                      distances -- unchanged, not reinvented
    kept verbatim     15-25 steps, full BPTT, pool at 0.5, the five published
                      settings, lr 1e-3, clip 1.0, wd 1e-5
    sampling          Gaussian at natural Rg (2.2 * N^0.38 A), run until max
                      per-step displacement < 0.1 A or a step cap

**`context_fraction = 0.0` runs FIRST and alone.** The intermediate sweep
(`ctx25`, `ctx50`, `ctx75`, `ctx100`) exists only as the localiser if 0.0 fails
-- so this is one experiment with a contingency, not five arms. The instruction
is fewer arms.

## THE COMPARISON IS NOT AVAILABLE, and that is a finding about the field

**Every published metric is gated on two neural networks we do not have.**
Designability is ProteinMPNN (8 sequences, sometimes 100) plus ESMFold or
OmegaFold, scored as scRMSD <= 2 A with pLDDT >= 70, or the looser scTM > 0.5.
Worse for our purposes, **Genie 2 and FrameDiff compute diversity and novelty
only over the samples that already passed designability**, so even those,
which are deterministic on CA coordinates, are not comparable to a number
computed over all samples.

Read 26 Nov from Genie 1 (arXiv 2301.12485), FrameDiff (2302.02277) and
Genie 2 (2405.15489). Genie 2's Table 1, in-distribution, lengths 50-256:

    method        designability  diversity  PDB novelty
    Chroma               0.70       0.51         0.13
    RFdiffusion          0.96       0.63         0.26
    Genie 2              0.96       0.91         0.41
    PDB reference       0.872          -            -

**None of the four papers reports a single metric computable from CA
coordinates with no learned model and no database.** Not CA-CA bond length, not
clash rate, not radius of gyration. So the MVP cannot produce a number that
goes in that table, and **any claim of being competitive with diffusion is
unlicensed until ProteinMPNN and ESMFold run here.** That is phase 2 and it is
not a space problem -- see below.

### The one axis that IS weight-free, and it is the interesting one

**Secondary-structure composition by P-SEA**, which assigns helix/strand/coil
from CA distances and angles alone. Genie 1 uses it. Both Genie papers report
the same qualitative failure: **strands and loops underrepresented, generated
structures helix-biased.**

This project has independently recorded
[[the-rule-learned-helical-torsions-and-learned-nothing-about-sheet]] and
[[the-start-state-hands-the-rule-its-secondary-structure]]. **If the MVP is
also helix-biased, the finding is a shared failure mode of the field rather
than a defect of ours**, and that is a better result than a leaderboard row.
Registered now so it cannot be claimed after the fact.

## Buckets, registered now

`scripts/generation_metrics.py` computes all of these, and **computes them on
natural proteins through the same code path**, because without an in-house
reference none of the numbers mean anything.

**1. Is there a fixed point at all? -- ANSWERED YES, 26 Nov, before any
generative training.** Measured on the port's existing checkpoint by
`scripts/persistence_readout.py`:

    steps    n  diverged   rmsd_dmg   fit_intact  max_step
       25   15         0      1.878        1.967    0.1533
     1000   15         0      2.154        2.295    0.0703
    10000   15         0      2.154        2.295    0.0703

`n` 15 and `diverged` 0 at every step count in both protocols, at **400x the
training horizon**; the 1,000 and 10,000 step rows agree to 6-7 significant
figures on every individual crop, so it is a fixed point and not a slow drift;
9 of 15 reach an exact numerical fixed point. **And in the strong form**: re-
damaged mid-rollout at step 24 and continued, the rule returns to the *same*
endpoint as the undamaged rollout, mean difference **0.0105 A**, 12 of 15 crops
within 0.01 A -- the same basin from two different states. Price of persistence
0.28 A, still inside the `<= 4.0 A` bucket at 10,000 steps.

**This was the bucket the existing record predicted we would FAIL**, since
[[the-rule-has-no-attractor-past-its-training-horizon]] is `supported` and
[[the-pool-and-length-jitter-stop-the-divergence-past-the-training-horizon]]
stopped the *divergence* without demonstrating an attractor. It does not
overturn that node: that one is `g20_orient`, an `AANCARule` with frames and
the 0.3 A cap over a 120-step rollout, which read 16.4-16.9 A at 600 steps with
0 of 48 converging. **Both stand, about different rules, and the contrast is
the result.**

**Steps to the fixed point: 12 of 15 converge, median 25, min 18, max 104**,
the three that never doing so being the three with residual motion. So the rule
arrives at its fixed point at essentially the horizon it was trained on, and
the paper's 15-25 step sampling range straddles that median. **But drift from
the convergence step to step 10,000 is mean +0.355 A and up to 1.533 A**, so
the displacement threshold fires early on some crops and the fixed-point claim
rests on the 1,000-vs-10,000 agreement, not on `converged_at`.

**Two things this answer does not cover.** `native_edges = true` holds the
contact graph fixed for all 10,000 steps, so it is persistence under the task
statement rather than under a live k-NN graph -- which is precisely what the
MVP varies. And the checkpoint is update 11,500, not 20,000.

**2. Is the fixed point protein-shaped?** Anchored, 26 Nov, to the measured
in-house reference rather than to round numbers -- 60 natural entries, 50-400
residues, scored by `scripts/generation_metrics.py` through the same code path
a generated set will use:

    quantity                     natural reference     generated must reach
    CA-CA mean                   3.8027 A              3.80 +/- 0.10
    CA-CA outlier fraction       0.0087                < 0.05
    clash rate (|i-j|>=3, <4 A)  0.0001                < 0.01
    Rg / 2.2 N^0.38              1.101 (sd 0.131)      within 2x

**So the bar for being geometrically indistinguishable from crystal structures
is 0.9 % bond outliers and one clash in 10,000.** All three axes or it is a
blob.

**3. Is it diverse, or one attractor?** Mean pairwise TM-score over >= 100
samples, against the same statistic on natural proteins.

    < 0.30     diverse -- the noise selects among many attractors
    0.30-0.50  partially collapsed
    0.50-0.90  collapsed toward one fold
    > 0.90     ONE attractor. Not a generator.

**The natural reference is mean pairwise TM 0.2068, max 0.5142, with 59 of 60
clusters at TM 0.5 and 60 of 60 at TM 0.6.** So `< 0.30` was the right line
drawn in advance, and 59/60 clusters is a CEILING -- a sample set that beat it
would be more diverse than real proteins, which is a warning sign and not a
win. Novelty against 50 disjoint training entries reads mean max TM 0.3293,
with 3.3 % above 0.5.

**This is the make-or-break metric.** With no conditioning, a shared local rule
has only the initial noise to distinguish one sample from another, and the
honest prior is that it collapses -- plausibly onto the helix, given
[[the-helix-prior-is-responsible-for-some-of-the-7-78-angstrom]].

**4. Helix-biased?** The natural reference, pooled over the 60 entries, is
**helix 0.2804, strand 0.1978, coil 0.5218** (classes: 35 alpha/beta, 14
mainly-beta, 10 mainly-alpha, 1 irregular). `> 0.90` helix is collapse onto the
prior, not generation, and is read with bucket 3.

**The P-SEA implementation is validated, not assumed.** Checked against
biotite's `annotate_sse` -- an independent implementation of the same 1997
paper, working on the AtomArray rather than our extracted CA trace -- on four
crystal structures: **1.000 per-residue agreement on all four.** 1UBQ comes out
as the five-strand beta-grasp sheet plus the single helix at residues 23-33
against DSSP's 23-34, one residue short at the C-terminal end, which is the
known cost of assigning with no hydrogen bonds to see.

**One caution on the reference itself.** 4HHB chain B reads 34.5 % bond
outliers against 0.008-0.035 for the modern entries -- a 1974 structure at
1.74 A with genuinely loose CA geometry. The bond-outlier axis partly reads the
reference set's resolution, so the 60-entry pool is the reference and no single
entry is.

**And TM-score here is a documented lower bound**: correct d0, TM-score's own
iterative seed-fragment superposition, but the correspondence search restricted
to gap-free threadings where TM-align runs dynamic programming with gaps. That
is the safe direction for diversity and novelty, which it can only overstate,
and the wrong direction for claiming two structures match.

**Refuted if**: no fixed point within 500 steps, OR mean pairwise TM > 0.90, OR
the geometry fails all three of bucket 2. Any one of those ends option 2 in its
present form, cheaply, which is the best property this experiment has.

**FIRED, 26 Nov, on the first condition.** 13 of 15 samples never settle inside
500 steps at 0.1 A per step. The second did not fire and fired the other way --
mean pairwise TM 0.1379, below the natural reference's 0.2068. The third is
unmeasurable as written: the Rg axis passes for a rule that does nothing,
because the prior is constructed at the target Rg. **It cost one 1h55m GPU job
and three read-only probes, which is the property this note was written for.**

## Bunya's space, measured 26 Nov, because it decides where phase 2 can live

    /scratch/user/uquadhik   244 GB of 300    687,120 files of 1,000,000
    project fileset S0253    491 GB of 10,240 234,551 files of 1,000,000

**The user scratch quota has 56 GB and ~313 k inodes free; the project fileset
has 9.7 TB.** So ESMFold's ~5 GB is affordable in either place and **space is
not what blocks the diffusion comparison** -- the blocker is the ROCm
dependency install and GPU memory, which is a different risk and should be
priced separately. Two consequences, binding on anything written for this:

  * **Phase-2 model weights go in the project fileset**, not the user scratch.
  * **Samples go in one compact `.npz`, never one file per sample.** 313 k free
    inodes is a real limit and this project has hit an inode quota before.

## What this does NOT claim

  * **It does not reopen the capacity question.**
    [[the-architecture-can-place-residues-under-some-supervision]] was answered
    NO for invariant perception over an error-derived neighbourhood, and the
    rule here is a different architecture on a different task statement.
  * **It is not a replication of anything.** Unlike the port, there is no
    published result to port: Gala's autoencoder trains a latent per target,
    and this trains no latent at all. Where the port's virtue was faithfulness,
    this arm's risk is that it is **ours**, so it gets buckets registered in
    advance instead.
  * **One seed.** [[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]
    applies to this arm as to every other, and nothing here is replicated.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]

## 26 Nov, LATER: the attractor EXISTS and the claim is still refuted, which is the cleanest form of the refutation

The fold-class arm
([[a-global-fold-class-label-tells-us-whether-option-2-failed-on-ambiguity-or-on-address]])
was not built to test this, and it did. Its checkpoint, rolled out on the
**unconditional** row at this node's own registered numbers -- 0.1 A/step,
500-step cap -- settles **14 of 15 crops**, median `last_move` 0.0969 A/step,
against this node's measured **2 of 15** at median 0.5379.

**So bucket 1 passes, and the conclusion does not change.** The thing it settles
to is all coil, `irregular` on every sample, CA-CA 10.799 A against the natural
3.8027, outlier fraction 0.9951, and a clash rate of 0.0183 -- **worse** than
the 0.0111 recorded here.

**That is why this refutation is now stronger than when it was written.** It no
longer rests on the rule failing to converge, which is a property anyone could
argue away with a longer cap or a looser tolerance. It rests on the rule
converging and the fixed point not being a protein. The two halves of "a local
rule with an attractor **is** a diffusion model without a schedule" are
separated by measurement: the attractor is necessary and it is nowhere near
sufficient.

The caveat travels with it: the arm that exhibited the attractor also added a
constant learned edge feature and 148 parameters, and no measurement separates
those from each other, so *what* bought the fixed point is open.

