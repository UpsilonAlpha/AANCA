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

## Our corruption operator is not noise, which is the whole comparison with differential denoising

Asked 14 Nov: how does this compare to re-noising one region -- an interface --
after a good backbone is already in hand? That is **partial diffusion with a
per-residue noise level**: hold `sigma_i` near zero on the scaffold, raise it on
the target region, denoise, and the region resamples while the rest stays put.
Our damage mask looks like the same operator with a binary schedule. Read the
code and it is not the same operator at all.

`growth.damage_window` picks the window and `growth_objective` then sets
`anchors = ~damaged` and calls `baseline_coords(...growth.initial...)`. So the
window's native pose is **discarded** and the window is **regrown in torsion
space off the intact flanks** at ideal `(phi, psi, omega)`. `BASELINE_MODES` is
`("helix", "extended", "random")`, every arm has run `initial = "helix"`, and
**no mode perturbs a pose** -- there is no path in the training code that keeps
a structure and displaces it.

Four consequences, and the first two were invisible until the operator was read
as an operator:

  1. **The prior is informative, not maximum-entropy.** Gaussian noise at high
     sigma is the least committal corruption there is. An ideal helix is one
     conformation class, handed over identically every time. Given the window,
     the start state has near-zero entropy, so each window is trained on
     essentially one trajectory -- the same deficiency the pool and the length
     jitter were added for ([[update-probability-is-a-result-affecting-setting-that-no-config-records]] is the fix from
     the other end), and it is a reason to expect memorisation of a correction
     rather than a learned basin.
  2. **Local geometry is ideal at t=0 and wrong only globally.** Cartesian noise
     breaks bonds and the denoiser learns to repair them; a NeRF walk cannot.
     So `bond` and `clash` only ever see damage the *rule itself* produced, and
     the rule has never been asked to repair bad local geometry presented as
     input. Against a partial-diffusion model that is a capability we do not
     train.
  3. **No `sigma` input, by design, so no temperature and no reliability
     label.** A diffusion model is *told* which residues to trust. This rule
     infers it from geometry. That is the liability and the feature at once: it
     is why there is no calibrated `p_sigma` to take a log of
     ([[destruction-and-regrowth-yields-a-local-statistical-potential]] on the
     honest limit), and why an external scorer at a matched sigma is the only
     route to a temperature ([[question-can-proteinebm-be-reused-and-in-which-of-three-ways]]).
  4. **The region control is binary, and the context is fully mobile.** A
     per-residue sigma lets the scaffold relax *a little*. Ours is on/off at a
     hard seam, and intact residues are not frozen during the rollout -- they
     are correct at t=0 and free to drift, which is a third regime neither
     model has, and they do drift.

**And for an interface specifically the operator cannot express the task.**
`damage_window` draws a contiguous run *inside a single covalent run and never
spanning a chain break*, for a good reason of its own -- a window across a break
is two holes and a failure would not say which. But an interface is defined by
**spatial** proximity to another chain, not by sequence contiguity, so the
current damage can only ever hit one side of it. A binder window would need a
mask drawn from the inter-chain contact set.

### Read as prior art, not as a to-do list

Ulban's reason for asking, stated immediately after: **this is prior art, and
the rule stays simple.** The operator difference above is descriptive -- it says
our conditional is a different object from theirs -- and is explicitly *not* a
list of features to add. A noise mode, a soft seam and a spatial mask would each
make this rule more like the model it is supposed to be an alternative to, which
is the wrong direction.

**What it pre-empts.** "Locally resample a region conditioned on surviving
structure" is now first-class in a production model, not a trick. Any claim of
the form "only a local rule can regrow a chosen region" is dead -- it was
already dead via motif scaffolding and inpainting, and this closes the door
louder. So the damage protocol is an **instrument**, never a contribution, and
the interpretability pitch cannot rest on it. It has to be carried entirely by
the shared local rule and the explicit fields and couplings of the Potts
summary.

**What it does not pre-empt**, and this is where the novelty actually sits:

  * **The sweep.** Per-region noise still runs all-pairs attention over the
    whole complex at every step, however few residues were noised. One window's
    conditional is affordable; a dense map over every window at N samples each
    is not. Prior art on the operation, not on the scaling.
  * **Spread as a measurement rather than a setpoint.** `sigma` is *chosen*, and
    training calibrates the output variance to it, so the local diversity of a
    partial-diffusion sample is a hyperparameter. Our damage sets the initial
    condition only and the rule is never told how bad it was, so the spread is a
    **response**. That difference is the entire reason "regrow a loop N times
    and the spread is its conformational freedom" is a measurement at all. The
    weaker honest form: rigidity read off a diffusion model at fixed `sigma` is
    *confounded* -- it mixes the model's noise calibration with the structure's
    freedom, and there is no principled `sigma` -- not impossible.
  * **Refusing the reliability label.** They condition on `sigma`; this rule has
    no way to be told which residues to trust. Opposite designs, so their work
    leaves the architectural question untouched.

**Where it hurts.** The comparison on quality is unfavourable and will stay that
way: a flagship model does this operation well, and ours does it at 7.78 A with
no attractor. The case can never be "we do this better". It is "we do it in a
form that can be read, at a cost that permits doing it everywhere" -- and that
case is only worth making once the rule stops drifting.

**Where it helps.** It establishes that region-local resampling against a fixed
context is an operation worth having, so the framing is de-risked rather than
speculative. And it supplies a **fair baseline**: differential denoising at
matched displacement is the comparison for any claim about what regrowth spread
measures, which is sharper than "versus pairformers".

## Requires
- [[question-does-regrowth-dispersion-measure-conformational-freedom]]

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
