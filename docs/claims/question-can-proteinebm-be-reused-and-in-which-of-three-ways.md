---
type: question
status: open
sources: [Du 2020, ProteinEBM 2025]
---

# Can ProteinEBM be reused here, and is it as weights, as a scorer, or as a term in the loss?

Asked 13 Nov. Three modes of reuse get conflated by the word "reuse", and they
have different answers: **no, yes, and probably-not-if-the-interpretability-claim-matters.**

## First, two premises to correct

**PottsMPNN's reuse here was never weights.** `HANDOFF.md` §7 records it as the
**head-swap trick**: keep a ProteinMPNN-style encoder, replace the output head
-- here emitting torsion deltas instead of sequence logits. An architectural
pattern, not a checkpoint. So "we reused some of PottsMPNN's architecture" is
exactly right and does not generalise to weight transfer.

The limitation Ulban states for it does hold in the record: *"PottsMPNN has no
chi number to compare, which is exactly the point"* -- it does not place its own
residue's side-chain atoms.

**ProteinEBM is very likely not CA-only.** The preprint introducing it
describes an energy-parameterised, **sequence-conditioned** protein diffusion
model built on **the diffusion modules from AlphaFold3 and Boltz-1**. Those are
all-atom and non-equivariant-with-augmentation, which also matches what this
project already recorded about it (`src/aanca/graph.py` copied "invariant
features plus rotation augmentation" from ProteinEBM's choice, and `HANDOFF.md`
notes it is "deliberately *not* equivariant").

Not fully confirmed: the search snippets did not state the representation
explicitly, and the methods section has not been read. Flagged rather than
asserted. **If it is all-atom, the stated objection disappears and it is a
better interface scorer than a CA-only model would be** -- but it makes weight
reuse worse, not better.

## Weights: no, and for a principled reason

ProteinEBM is a diffusion model repurposed as a potential, so its network is
**noise-conditioned** and carries **global attention** from the AF3/Boltz-1
modules. Those are precisely the two things this architecture is *defined* by
not having -- `src/aanca/model.py` states it as the claim: *"No step index, no
noise-level conditioning, no global pooling, no attention over the whole
protein. That absence is the scientific claim -- do not let a time embedding
creep in."*

So transferring weights would mean importing the conditioning the thesis is
about refusing. Add the ordinary mismatches -- 229,588 parameters against a
diffusion trunk, one message-passing hop, 32 + 24 feature dims, two heads -- and
there is nothing to initialise *from*. This is the same shape of answer as
`native_edges`: the problem is not that it would be hard, it is that it would
dissolve the claim.

## As a scorer: yes, it is already the plan, and it upgrades the flexibility measurement

`HANDOFF.md` §8 has had three tiers since 27 Oct, and `docs/PLAN_V1.md` lists
ProteinEBM scoring as *"post-hoc, addable to a trained model at any time"*:

    (a)  score the converged structure with ProteinEBM-x, the low-noise variant
    (b)  energy difference as a ddG readout -- but take the bound/unbound
         decomposition from EBM-DDG, because ProteinEBM's own protocol measures
         folding stability against a Ramachandran-random unfolded ensemble
    (c)  auxiliary energy gradient on the last few rollout steps, small lambda

**(b) is the direct answer to the interface question**, and it was already
settled by the prior-art search: for an interface, do not use ProteinEBM's own
protocol, use EBM-DDG's bound/unbound decomposition. Asking a
folding-stability energy about a binding interface is the same category error
the pmhc work records about SKEMPI being out of regime.

**And (a) is worth more than it looks, because of the dispersion programme.**
The ensemble measurement gives N regrown poses of one window
([[question-does-regrowth-dispersion-measure-conformational-freedom]]). An
energy per pose turns that set into a **Boltzmann-weighted ensemble**, which is
the step from a geometric spread to a conformational *entropy* -- the quantity
Ulban's binder hypothesis is actually about. Geometric RMSF says how far the
poses are apart; a weighted ensemble says how many of them a protein would
actually occupy.

That is the highest-value use of ProteinEBM available to this project, and it
needs no training and no architectural contact at all.

## The regime caveat, and the one place noise conditioning helps us

At `settled` 7.78 A the regrown window is **far** from native
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]),
so ProteinEBM-x, the *low-noise* variant, is out of domain on our current
outputs -- scoring a 7.8 A-wrong window with a near-native energy is asking a
calibrated instrument about something it never saw.

But the model is noise-conditioned, so **the σ can be chosen to match the
displacement we are actually at.** The property that makes it useless for
weight transfer is the property that makes it usable as a scorer at a known
error -- and it gives a principled way to pick the variant instead of defaulting
to the low-noise one.

## (c) is the one to refuse, or at least to refuse while the payoff is interpretability

Putting the energy gradient in the loss means the rule learns to satisfy
**another model's** energy landscape. Two costs, and the second is the serious
one:

  * It imports that model's biases, which §8 already anticipates by requiring a
    check that it "does not degrade repacking".
  * **It contaminates the flexibility measurement.** If the rule is trained
    against an EBM's landscape, the ensemble spread reflects the EBM's basins,
    and correlating that spread with B-factors stops being a test of the rule
    and becomes a test of the EBM. The external anchor is the most valuable
    thing in the dispersion plan and this would spend it.

Not a ceiling in the native-information sense -- no target leaks -- but a
provenance problem for the interpretability claim. A rule whose learned
"physics" is a distillation of AF3-module diffusion weights is not a rule whose
ablations tell you about proteins.

## What would answer this

  1. Read the ProteinEBM methods for representation, noise conditioning, and
     whether weights are released at all -- the last is unconfirmed and gates
     everything above.
  2. Score the existing `g20_orient` regrowth ensemble at a σ matched to its
     displacement, and check whether the energy ordering of the poses carries
     any signal beyond their geometric spread.
  3. Only then consider (b), and only with EBM-DDG's decomposition.

## Supports
- [[question-does-regrowth-dispersion-measure-conformational-freedom]]

## Requires
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
