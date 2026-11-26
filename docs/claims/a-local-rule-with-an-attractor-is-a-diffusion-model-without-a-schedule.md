---
type: question
status: open
---

# Option 2, pre-registered: a local rule whose fixed point is a protein is a diffusion model with no schedule and no timestep

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

**1. Is there a fixed point at all?** Max per-step displacement < 0.1 A within
500 steps.

  * **YES** -- the object exists and the rest of the questions are worth asking.
  * **NO** -- the MVP is refuted on its first measurement. Note that
    [[the-rule-has-no-attractor-past-its-training-horizon]] is `supported` and
    [[the-pool-and-length-jitter-stop-the-divergence-past-the-training-horizon]]
    stopped the *divergence* without demonstrating an attractor, so **NO is the
    outcome the existing record predicts.**

**2. Is the fixed point protein-shaped?** CA-CA virtual bond 3.80 +/- 0.10 A,
clash rate (|i-j| >= 3 closer than 4.0 A) < 1 %, Rg within 2x of 2.2 N^0.38.
All three or it is a blob.

**3. Is it diverse, or one attractor?** Mean pairwise TM-score over >= 100
samples, against the same statistic on natural proteins.

    < 0.30     diverse -- the noise selects among many attractors
    0.30-0.50  partially collapsed
    0.50-0.90  collapsed toward one fold
    > 0.90     ONE attractor. Not a generator.

**This is the make-or-break metric.** With no conditioning, a shared local rule
has only the initial noise to distinguish one sample from another, and the
honest prior is that it collapses -- plausibly onto the helix, given
[[the-helix-prior-is-responsible-for-some-of-the-7-78-angstrom]].

**4. Helix-biased?** P-SEA fractions against natural. `> 0.90` helix is
collapse onto the prior, not generation, and should be read with bucket 3.

**Refuted if**: no fixed point within 500 steps, OR mean pairwise TM > 0.90, OR
the geometry fails all three of bucket 2. Any one of those ends option 2 in its
present form, cheaply, which is the best property this experiment has.

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
