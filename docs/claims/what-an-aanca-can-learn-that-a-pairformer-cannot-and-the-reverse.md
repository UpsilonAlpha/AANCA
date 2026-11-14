---
type: claim
status: open
sources: [model.py, rollout.py, 32604037]
---

# What this architecture can learn that a pairformer cannot, what it cannot learn that a pairformer can, and why side chains decide whether the potential is worth anything

Asked 13 Nov. Written as a ledger rather than an argument, because the project
has twice mistaken an inherited design choice for a result.

## What RFD4-Proteina-class models can learn and this cannot

  * **Global topology in one shot.** A pair representation with triangle
    updates can enforce a tertiary arrangement because it sees every pair at
    every step. This rule reaches globally only through iteration -- degree
    28-33, two hops to cross a small protein -- so global consistency is an
    *emergent fixed point* rather than an architectural guarantee. For
    proposing a novel fold topology from nothing, that is a real
    disadvantage and not a framing problem.
  * **Scale.** Proteina-class training is tens of millions of structures
    against hundreds of millions of parameters. 229,588 parameters cannot
    absorb that distribution, and no amount of iteration substitutes for it.
  * **A calibrated sample.** A diffusion model's outputs approximate draws
    from a learned distribution with a temperature. This rule's outputs are
    **optimiser outcomes**, which is the limitation that bounds the whole
    statistical-potential programme
    ([[destruction-and-regrowth-yields-a-local-statistical-potential]]).

## What this can do that they cannot

  * **Dense local conditionals.** Not that a diffusion model cannot produce
    one -- inpainting does -- but that each sample costs a full global
    denoising trajectory, so a *sweep* of every window at N samples each is
    out of reach. The AANCA's sampling cost is local, so the map is the
    deliverable.
  * **An exact conditional.** Diffusion inpainting re-noises and re-denoises,
    so the "fixed" context is only approximately fixed. Here the intact
    context is held at its crystal pose and never perturbed, so
    `p(window | context)` is conditioned on the context *exactly*.
  * **Settling time as an observable.** A diffusion model's step count is a
    hyperparameter, so "how long did this region take to settle" is not a
    property of the protein there. With no schedule and no step index it is
    one here, and it should itself track flexibility.
  * **One small function to interpret.** The point of the whole constraint.

**Two of these four no longer hold, as of reading the RFD4-Proteina preprint on
14 Nov; see the section below.** *Dense local conditionals* is weakened, not
refuted: PLACER is a local ensemble model by construction, and RFD4-Proteina
scored 16 conformers for each of 406 retroaldolase designs at 42-fold the
throughput of the next-best binder method, so "out of reach" was wrong about
cost. *An exact conditional* is refuted: the model takes per-atom conditioning
masks and supports "arbitrary combinations of fixed and flexible coordinates",
so conditioned atoms are held exactly, not approximately. *Settling time* and
*one small function* stand, and settling time arguably gains -- their recurrent
pass count is chosen by schedule, never emergent.

## Side chains are what decide whether any of this is worth having

**Correction to this record, 13 Nov: ProteinEBM is backbone-only in its
implementation.** An earlier note here inferred the opposite from its being
built on AlphaFold3 and Boltz-1 diffusion modules, which are all-atom
([[question-can-proteinebm-be-reused-and-in-which-of-three-ways]]). The
architecture being all-atom-capable and the model as implemented being
backbone-only are compatible, and the implementation is what one can actually
score with.

That matters more than it sounds, because **binding is side-chain packing.** A
backbone-only energy cannot score an interface at all -- it can rank backbone
plausibility and nothing else. So:

  * The Boltzmann-reweighting use of ProteinEBM survives, but only for
    **backbone** conformational entropy of a regrown window. It is not and
    cannot be the interface energy.
  * The AANCA's claimed advantage over it is therefore **explicit side
    chains**, and that is the real content of "cofolding with explicit side
    chains" as the goal.

**And the uncomfortable half: the growth track does not have them either.**
`identity_weight = 0.0` in every arm, and the growth objective scores N, CA and
C. The project has side chains in the *packing* track -- chi angles, rotamer
libraries, 38.0 % held-out sequence recovery with chi accuracy -- and not in the
track that does destruction and regrowth. The two have never been joined.

So the advantage over a backbone-only EBM is real and **not yet realised**, and
the thing that realises it is exactly the cofolding arm that has been fourth in
the queue. An all-atom local statistical potential is reachable here and
unreachable from a backbone-only EBM; a backbone-only local potential is a
better-localised version of what ProteinEBM already does, which is a much
weaker claim.

**This is the argument for moving cofolding up**, and it is an argument from the
deliverable rather than from curiosity: without side chains the potential
inherits the limitation it was supposed to escape.

## First measurement, and it already says two things

`scripts/regrowth_dispersion.py` on `g20_orient`'s `best.pt`, one crop (3B82),
2 members per amplitude, cap 120 -- a mechanical shakedown, not a result:

    amp deg  start RMSF  settled RMSF  ratio  steps  converged  rmsd_dmg
          0       0.000         0.458    n/a    120      0/2         5.44
         15       0.399         0.821   2.06    120      0/2         6.00

  * **The noise floor is not zero.** Mask alone produces **0.458 A** of
    settled spread, so the update schedule is a large fraction of any spread
    this measurement will see, exactly as feared
    ([[update-probability-is-a-result-affecting-setting-that-no-config-records]]).
    Every flexibility signal has to be read above that, and the synchronous
    arm becomes more valuable, not less.
  * **Nothing converged.** 0 of 2 at a 120-step cap and a 0.02 A/chunk
    tolerance: the window is **still moving at the training horizon**, which is
    the first direct evidence for the "bounded basin, not a fixed point"
    reading and says the cap has to rise before any of this is read.

Per-residue spread was not flat -- 0.10 to 0.82 A across the window at the
noise floor, higher in the middle -- which is the shape a real signal would
have and is worth nothing at n=2. A larger run is in flight.

## RFD4-Proteina has no per-region noise, and the ensemble-spread measurement is already published twice

Read 14 Nov from the preprint itself -- Corley, Didi, Kazman, Brent *et al.*,
"Biomolecular design through latent reasoning with RFD4-Proteina", 2026, at
`https://rfd4-proteina.org/assets/paper/RFD4-Proteina.pdf` (not peer reviewed;
no DOI or arXiv id as of reading). The model is **flow matching in a learned
latent space**, not coordinate-space denoising diffusion: an autoencoder gives
each residue an anchor-atom coordinate plus a **32-dimensional latent**, and the
flow model generates both. ~435M stored parameters across flow model and
autoencoder.

### The premise of "differential denoising" does not hold, and the mechanism is the shape of ours

Table S7, the flow model's complete input inventory, settles it. Every
per-residue feature carries an `[N, ...]` shape. The noise levels do not:

    Coordinate time embedding   [256]   adaLN   "Coordinate noise level"
    Latent time embedding       [256]   adaLN   "Latent noise level"
    Recycle indicator           [1]     adaLN

Two **global scalars**, entering by adaptive layer normalisation, and the text
confirms "Time embeddings condition both the single track and initial pair
representation". There is no per-residue or per-region noise level in the model.

**The differential axis is coordinates versus latents -- structure versus
sequence -- not region versus region.** Inverse folding runs "partial diffusion
from the input backbone (t = 0.99) with the sequence latents at t = 0", which is
the two knobs used against each other, and the docs' `partial_t_backbone` /
`partial_t_latent` are exactly those two.

Spatial control is a **binary mask plus a radius**, not a noise profile: the
model supports "arbitrary combinations of fixed and flexible coordinates"
through per-atom conditioning masks (`Absolute coordinates and masks
[N, 37 x (3 + 1)]`, `Hotspots [N, 37]`), and the interface protocol reads
"Sequence changes were allowed within 8 A of the interface for fixed
RFD4-Proteina and 12 A for flexible RFD4-Proteina; backbone flexibility was
restricted to 8 A."

So the real mechanism is **one global corruption magnitude times a binary
spatial mask** -- which is the shape of our damage protocol, not a contrast with
it. The earlier version of this section claimed theirs was a continuous
per-residue sigma against our binary mask, and drew a difference from it. That
difference does not exist and the claim was wrong.

### What of the operator comparison survives

  * **The corruption is still a different operator.** They noise toward the
    prior at `partial_t`; we discard the window and rebuild it as an ideal
    helix in torsion space. Informative prior against uninformative one, and
    local geometry ideal at t=0 because a NeRF walk cannot break a bond. Both
    stand.
  * **Conditioning on the corruption still differs.** Theirs reads the noise
    level, globally, through adaLN. Ours reads nothing. Still the architectural
    fork.
  * **The pair representation went the other way.** They "strengthened the pair
    representation through triangle multiplicative updates every other
    transformer block, more expressive communication from single to pair
    features, and a wider hidden dimension". The no-pairformer thesis is now
    contrarian against an actively strengthening trend, which raises both its
    value if it works and the burden of showing that it does.
  * **And they generate in a latent space.** Theirs is a learned 32-d code per
    residue; ours is explicit coordinates. This is a real interpretability
    advantage of ours and was under-weighted before.

### The hard part: local-ensemble spread as a preorganisation readout is prior art

Not adjacent work -- the same measurement, benchmarked, twice.

**PLACER** (Anishchenko *et al.*, *PNAS* 122, e2427161122, 2025, "Modeling
protein-small molecule conformational ensembles with PLACER") is the direct
precedent, and this preprint calls it "the current state-of-the-art".
RFD4-Proteina then reproduces it: "Following PLACER, we sample local
conformational ensembles for designed enzyme active sites, **using the
consistency of independent outputs as a proxy for catalytic motif
preorganization**." Seven highly active retroaldolases against 399 inactive, 16
conformers per design, and the score is "**negative intermediate sidechain-atom
spread after backbone alignment**".

That is the quantity `scripts/regrowth_dispersion.py` computes, applied to the
question the flexibility hypothesis asks, against a functional label. AUROCs,
RFD4 against PLACER: 0.84 [0.74, 0.93] vs 0.88 [0.82, 0.95]; 0.62 [0.56, 0.67]
vs 0.60 [0.55, 0.66]; 0.90 [0.81, 0.98] vs 0.82 [0.52, 0.98]. Their own caveats
are worth keeping: "paired intervals do not establish a method advantage" and
"the main-panel intermediate was selected retrospectively".

**So the ensemble-dispersion programme is not novel as a measurement.** It is
validated, which is worth something -- the hypothesis that spread tracks
preorganisation and preorganisation tracks function is now supported by two
published models and a functional assay, so the project is not betting on an
unproven premise. But a result of the form "our spread correlates with
B-factors" is a replication, and has to be presented as one.

### The confound I raised is real and is handled by convention

Confirmed: spread **is** a dial. "Diversity was tuned by the scale of the
sampling noise: 12 values from 0 to 5.0"; "Greater sampling noise broadened mean
pairwise side-chain variation from 0.31 to 1.20 A, while the mean MolProbity
score rose only from 1.70 to 1.77". Fig. 3a is a quality-diversity Pareto
frontier over "eight sampling-noise settings".

And the practice is to **fix the dial and only rank** -- the retroaldolase run
used "sidechain noise 1.0" throughout. So "confounded, not impossible" was
right, and the confound does not bite in the application, because ranking
designs against each other at one setting is all the use case needs. This is not
an opening for us.

### What is actually left, and it is narrow

  * **Potts appears nowhere.** Zero occurrences of `potts`, `statistical
    potential` or `free energy` in the whole preprint; one `interpret`, about
    the sampler's `k` parameter as a temperature-like control. Their readout is
    a **number** -- a spread -- where ours would be a **model**, explicit `h_i`
    and `J_ij` over discrete basins, ablatable term by term. On this axis the
    ground is still open, and it is now the *only* open ground in the
    flexibility programme.
  * **The sweep argument is weakened and must be measured, not asserted.** They
    report 42-fold throughput over BindCraft at 10.6% clustered pass rate on
    protein binding, and binder rescue "in a fraction of the time". A dense
    sweep is cheaper for them than the cost-scaling argument assumed.
  * **Explicit coordinates against a learned latent**, as above.

### The convergence nobody planned

"We enabled intrinsic reasoning within trajectories by performing **recycling at
each diffusion time step** rather than as an upstream feature-processing stage.
The number of recurrent passes can **vary across timesteps**, allowing greater
refinement at critical stages" -- and the abstract's "recurrent refinement to
**adaptively allocate computation**".

Repeated application of the same update until the state settles, with a variable
number of passes. That is an NCA-shaped inner loop, reached from the diffusion
side, and it is the strongest evidence yet that iterative local refinement is the
right instinct. It also means the architectural gap is closing from their end,
and their inner loop terminates by schedule where ours does not
([[the-rule-has-no-attractor-past-its-training-horizon]]).

### Where this leaves the positioning

Dead: the damage protocol as a contribution, regrowth spread as a novel
measurement, and any claim that a pairformer cannot resample a region locally.
Alive: no pairwise module at all, explicit coordinates rather than a learned
code, and the fields-and-couplings form of the potential. The bar is 10.6%
clustered pass rate with wet-lab validation against our 7.78 A with no
attractor, so the case is never "better" -- it is "readable, and cheap enough to
run everywhere" -- and it cannot be made at all until the rule stops drifting.

## Requires
- [[question-does-regrowth-dispersion-measure-conformational-freedom]]

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
