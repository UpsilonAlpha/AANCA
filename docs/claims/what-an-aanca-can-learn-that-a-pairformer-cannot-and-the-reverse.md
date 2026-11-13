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

## Requires
- [[question-does-regrowth-dispersion-measure-conformational-freedom]]

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
