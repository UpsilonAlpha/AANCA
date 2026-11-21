---
type: observation
status: supported
evidence: ours
sources: [33277250, 33277251, 33249861, context_drift.py]
---

# The trained rule displaces the surviving context by 2.34 angstrom while placing the window, so it corrupts the reference frame its own placement is measured against

Measured directly on `runs/g22_random/best.pt`, 12 held-out crops, 120-step
regeneration rollout, `scripts/context_drift.py` (jobs 33277250 free and
33277251 frozen, both exit 0):

    g22_random best.pt      context N/CA/C drift   worst atom   O (reconstructed)
    context FREE (as run)               2.3402 A      16.020 A            2.6545
    context FROZEN                      0.0000 A       0.000 A            0.0392

N, CA and C are the atoms that say whether a residue moved. O is recomputed from
the backbone every step by `correct_oxygens`, so it moves whether or not the rule
touches the residue and is reported apart -- note it tracks the drift when the
context is free (2.65) and collapses to 0.039 when the context is frozen, which
is what an O following a moving backbone looks like.

**So the surviving context does not survive.** The task is defined as a window
destroyed inside an otherwise intact protein, and by step 120 the intact part has
moved 2.34 A on average with at least one atom flung 16 A. The central claim is
*regeneration from surviving context*
([[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]):
the rule is reading, and being scored against, a frame it is itself destroying.

## This claim was inferred badly, retracted, then confirmed. All three steps matter

Recorded because the sequence is the useful part.

**Inferred (21 Nov, morning).** By decomposing `g22_random`'s logged `rmsd`
against its `rmsd_dmg`: 1.20 A untrained, 2.83 A settled. The note said in its
own scope that this was an estimate from a shared superposition.

**Retracted (21 Nov, afternoon).** `scripts/context_drift.py` measured the drift
directly and an *untrained* rule read **0.0000 A** while the decomposition read
**1.70 A** on the same rollout. So the method manufactures drift from none, by
the same mechanism corrected one level up that morning: fit the superposition on
every residue and the badly-placed window pulls the frame, so a correct context
reads as displaced. The claim was voided.

**Confirmed (21 Nov, evening) by the measurement that should have been made
first.** On the *trained* checkpoint the direct figure is **2.34 A**. The
retraction's control was the flaw -- an untrained rule's move heads emit ~0, so
it *cannot* drift and free-versus-frozen is vacuously identical for it. **A null
that cannot exhibit the effect is not a null**, and treating it as one voided a
true claim for half a day.

**The method remains invalid and must not be quoted.** On the frozen run, where
the true drift is exactly 0.0000, the decomposition still reads **0.96 A**. It
was right about the direction this morning by luck. Any number in this project
that comes from a fit over all residues is suspect until checked against a direct
measurement -- the same lesson that moved `g22_random`'s bar onto `fit_intact`
([[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]).

## Why it happens, and why nothing was charging for it

`growth_objective` called `rollout` with no `frozen` argument, so the moving set
is `mask & alive` and a regeneration rollout is 100 % alive -- every residue
moves. **That was deliberate**, and the reasoning is at the call site: scoring
FAPE over the whole structure was meant to teach *"fill the window"* and *"leave
the context alone"* together, the second half being what stops "emit nothing"
from being a free solution.

The measurement says that enforcement fails. A reason offered as a guess and not
more: FAPE is a mean over every residue, so 12 badly-placed residues are diluted
by 116 correct ones, and the cheapest way to reduce window strain may be to move
the anchors toward the window rather than the window toward the anchors. Nothing
in the objective distinguishes those two.

## `freeze_context` works, which was the second pre-registered check

`[g24_frozen]`'s second bucket required that the context's drift read ~0 under
freezing, because a flag silently dropped is this project's recurring failure
mode. It reads **exactly 0.0000 with a worst atom of 0.000** over 12 crops, not
merely small. **The gate on that arm is open**: freezing changes a real 2.34 A
rather than a rounding error.

## Scope

  * **One checkpoint, one seed.** `g22_random`'s `best.pt` only, and no arm here
    has been replicated
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).
  * **It does not follow that freezing improves placement.** The rule currently
    gets to relieve window strain by moving anchors; removing that could make
    `rmsd_dmg_fit_intact` worse. That would itself be the finding -- it would
    mean the 9.02 A depended on being allowed to corrupt the frame.
  * **Cumulative drift at step 120 against the crystal**, not a per-step rate,
    and not separated into drift that helps the window and drift that is simply
    wrong.
  * The 16.0 A worst atom is a single atom in one crop and is uncharacterised;
    the mean is the number to use.

## Supports
- [[the-gradient-reaches-a-quarter-of-the-journey-and-half-the-move-operator-is-unsteerable]]

## Requires
- [[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]
