---
type: observation
status: void
evidence: ours
sources: [33249861, growth.py, rollout.py, context_drift.py]
---

# VOID: the context drift this node was built on was an artefact of a shared superposition, and the direct measurement is exactly zero

**Retracted 21 Nov, the same day it was written, by `scripts/context_drift.py`.**

The headline claimed the rule displaces the surviving context by 1.20 A barely
trained and 2.83 A settled. Both figures came from decomposing `g22_random`'s
logged `rmsd` against its `rmsd_dmg`, and the note said in its own scope that
this was an estimate from a shared superposition rather than a measurement. It
was worse than an estimate -- it was measuring the superposition.

**The control that kills it.** An UNTRAINED rule over a full 120-step
regeneration rollout, where the move heads emit ~0 and nothing can have drifted:

    direct context N/CA/C displacement     0.0000 A   exactly
    rmsd decomposition on the same rollout   1.70 A   <- the method, on zero drift

So the decomposition manufactures ~1.7 A of apparent drift from none. The cause
is the same one this project had just finished correcting one level up: fitting
the superposition on every residue lets the badly-placed window pull the frame,
and the correctly-placed context then reads as displaced
([[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]
sets the arm's bar on `fit_intact` for exactly this reason). My 1.20 A
"untrained" figure and this control's 1.70 A are the same artefact.

A second error inside the first: a first version of the probe measured all
atoms together and found 2.1 A of context motion, which was **one oxygen**.
`correct_oxygens` recomputes every O from its backbone each step, so O moves by
the crystal-to-ideal difference whether or not the rule touches the residue --
identically with the context frozen and unfrozen. N, CA and C are the atoms
that say whether a residue moved, and they read 0.0000.

**What is still true, and it is the narrow version.** `growth_objective` did
call `rollout` with no `frozen` argument, so the context is *permitted* to move,
and `freeze_context` now exists and demonstrably works -- the moving mask drops
from ~33 residues of 64 to ~5. Whether the TRAINED rule uses that permission is
**unmeasured**: the untrained control cannot test it, because a rule whose move
heads emit nothing cannot drift, so free and frozen are vacuously identical.
`scripts/context_drift.py` on `runs/g22_random/best.pt` is the measurement and
it is a CPU job.

**`[g24_frozen]` must not be read -- or run -- until that number exists.** Its
whole rationale was this node. If the trained rule's context drift is also near
zero then freezing changes nothing, the arm is pointless, and the GPU slot
should go to `truncate` instead.

The two mechanisms below this one are unaffected: they were read off the config
and a previously measured number, not off this decomposition.

## The original claim, kept for the record

Asked on 21 Nov, after `g22_random` failed its absolute bar: **why** does the
rule not place a backbone? Three mechanisms are confirmed in the code and the
logs, they compound, and the first is a discrepancy between what this record
says the task is and what the code does.

## 1. Context stability is enforced by the loss rather than by construction, and that enforcement is failing

`growth_objective` calls `rollout(...)` at `growth.py:1894` **with no `frozen`
argument**. The parameter exists and gates geometry exactly as wanted --
*"the cell ticks under `mask`; its atoms move under `mask & ~frozen`"*
(`rollout.py:557`) -- but with `frozen=None` the moving set is `mask & alive`,
and a regeneration rollout is **100 % alive by construction**. So all 128
residues of the crop are moved by the rule, every step, for 120 steps.

**This is deliberate, and the reasoning is at the call site** -- an earlier
version of this note called it a discrepancy, which was wrong.
`growth.py:1820`: *"Scoring FAPE over the whole structure now teaches two things
at once: fill the window, and leave the context alone. That second half is the
stability the fixed-point objective asks for, and it is what makes 'emit
nothing' stop being a free solution -- doing nothing leaves the window wrong
while the context is right, which is a difference FAPE can see."*

So context stability was designed to be **taught by the loss** rather than
imposed by construction, for a stated reason: freezing it hands the rule a free
pass on the half of the objective that was meant to stop the degenerate
identity solution of the four arms of 6 Nov.

**The finding is therefore not that nobody noticed. It is that the mechanism
chosen to enforce context stability does not work.** The aspiration is in the
docstrings -- `GrowthOutcome.rmsd`: *"the intact context, which starts correct
**and should stay correct**"*; `baseline_probe.py`: *"Everything outside the
window keeps its crystal pose"* -- and the measurement below says it does not
stay correct, and gets worse the longer it trains.

**How much it moves, from the log's own two timepoints.** `rmsd` is over all
residues, `rmsd_dmg` over the 12-residue window. If the context held its
crystal pose, `rmsd` would be the window's contribution alone; the excess is
context displacement:

    g22_random            rmsd   window-only predicts   implied context
    update 250 (untrained) 3.96                  3.79             1.20 A
    update 10000 (settled) 3.73                  2.58             2.83 A

**The drift more than doubles over training.** The rule does not merely fail to
leave the context alone -- ten thousand updates teach it to displace the context
further, because nothing in the objective charges it for that while FAPE and the
pairwise anchor term both reward relieving window strain by moving the anchors
toward the window instead of the reverse.

This is the mechanism that matters most, because the central claim is
*regeneration from surviving context*
([[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]).
A rule that corrupts the context is destroying its own reference frame, and the
placement error is then measured against residues that have themselves moved.

**Caveat on the arithmetic**: `backbone_rmsd` fits its own superposition over
all residues, so this decomposition is an estimate rather than an exact split.
The direction and the doubling are robust because both rows are the same
instrument on the same crops; the absolute 2.83 A is not, and should be measured
directly on the checkpoint before it is quoted as a number.

## 2. The gradient sees about a quarter of the journey

`truncate = 12` in `[base.inputs]` (`configs/regeneration.toml:74`) against
`steps = 120`, and **nothing in g22's lineage overrides it** -- not
`g14_pairwise`, `g19_nomove`, `g20_orient` nor `g22_random`. So BPTT carries
gradient through the last 12 steps only. At the 0.3 A per-step cap that is 3.6 A
of visible travel against a start state 13.3 A out of place: **27 %**
([[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]], which
measured this model-free and classed the ratio as the binding constraint while
the per-step cap is not binding at 3.6x headroom).

The rule is trained on the end of a journey whose first three quarters it must
also make with no gradient reaching them. That is the recorded prediction for
which hyperparameter to test first, and `--truncate` is already a trainer key.

## 3. Half the move operator is still a direction the rule cannot choose

g22's own log header: *"EQUIVARIANT MOVE: the emitted move is read in each
residue's own N-CA-C frame. **TRANSLATION ONLY: the rotation is left in the lab
frame.**"* The perception is invariant by construction, so a lab-frame rotation
is a direction an invariant encoder cannot represent. Measured, the rotation
half is **0.68 A of error on a 1.50 A move -- 45 %**, which is *larger* than the
27 % translation half that was already fixed
(`tests/test_a_lab_frame_rotation_is_not_equivariant_either`). `g23_localrot`
tested switching it off and is void on the twist guard, so this is still open.

## The synthesis, and it is consistent with the margin

The rule learned the terms with dense local in-window gradients and did not
learn placement. `bond` fell **4.278 -> 0.629** and `clash` to 0.0148, while
placement went 13.44 -> 9.02 `fit_intact`, which is **0.46 A past what bond
closure and sterics alone force**
([[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]).

That is exactly what 1, 2 and 3 predict together: the placement signal reaches a
quarter of the journey (2), the frame it is measured against is being corrupted
and increasingly so (1), and nearly half of the rotational actuator is noise the
rule cannot steer (3). **So "the architecture cannot place a backbone" is not
yet established.** What is established is that this *training setup* cannot, and
three specific reasons why.

## The one test, and the real argument against it

`g22_random`'s registration says *"if it fails, we do not add terms and
re-run."* Freezing the context adds **no loss term and changes no weight**: it
passes `frozen = ~damaged` to a parameter that already exists. So the clause
survives it -- but the honest objection is not that clause, it is the design
reason quoted in 1, and it deserves answering rather than waving past.

**The objection**: freezing the context makes "emit nothing" free for 116 of 128
residues, removing the half of the FAPE signal that was meant to stop the
degenerate identity solution.

**Why it does not transfer to this regime.** The four arms of 6 Nov that froze
at their starting conformation were running *reconstruction*: the whole
structure started from a baseline, so "do nothing" was simultaneously the stable
answer and close to the scored answer, and the rule found it
([[the-four-arms-measured-the-architectures-ceiling]]). Under **regeneration**
with a frozen context, doing nothing leaves the window **13.3 A out of place**
while the context is already exactly right, and both FAPE on the window and the
unclamped window-to-anchor pairwise term see that directly. The free lunch the
comment is guarding against is a property of the reconstruction task, not of
this one.

**And it is cheap to falsify.** If freezing removes real signal, the arm's
placement gets worse, which is itself informative: it would say the rule was
using context motion to relieve strain and that the 9.02 A depended on being
allowed to corrupt its own reference frame. Either direction is worth a slot.

It is still a re-run, and the clause's spirit is that the bar does not move to
meet the result. So: **the 4.0 A bar on `fit_intact` stays exactly where it is**,
and the failed 9.02 A stands as the number for the setup that produced it.

Cheapest ordering, one variable each: freeze the context (1), then raise
`truncate` (2), then `lab_rotation = false` on a non-void arm (3). One is a
one-argument change and tests the mechanism most specific to this project's
actual claim.

## Scope

  * **Nothing here is a new measurement of the rule.** 1 and 2 are readings of
    the code plus arithmetic on an existing log; 3 is a previously measured
    number restated. No GPU time was spent to produce this note.
  * The 2.83 A context drift is an estimate from a shared-superposition
    decomposition, not a direct measurement -- see the caveat in 1.
  * Whether freezing the context *helps* is unmeasured and this note does not
    predict it will. It could also make placement harder, since the rule
    currently gets to relieve strain by moving anchors and would lose that.
    That would be a cleaner failure, not a worse one.
  * One seed throughout
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).

## Superseded by
- [[the-gradient-reaches-a-quarter-of-the-journey-and-half-the-move-operator-is-unsteerable]]

## Requires
- [[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]
