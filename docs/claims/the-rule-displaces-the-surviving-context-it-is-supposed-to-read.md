---
type: observation
status: supported
evidence: ours
sources: [33249861, growth.py, rollout.py]
---

# The intact context is never frozen, so the rule spends training learning to displace the information the task says it should read: 1.20 angstrom of context drift untrained, 2.83 settled

Asked on 21 Nov, after `g22_random` failed its absolute bar: **why** does the
rule not place a backbone? Three mechanisms are confirmed in the code and the
logs, they compound, and the first is a discrepancy between what this record
says the task is and what the code does.

## 1. The surviving context is not held fixed, and training makes that worse

`growth_objective` calls `rollout(...)` at `growth.py:1894` **with no `frozen`
argument**. The parameter exists and gates geometry exactly as wanted --
*"the cell ticks under `mask`; its atoms move under `mask & ~frozen`"*
(`rollout.py:557`) -- but with `frozen=None` the moving set is `mask & alive`,
and a regeneration rollout is **100 % alive by construction**. So all 128
residues of the crop are moved by the rule, every step, for 120 steps.

Every description of the task in this repository says otherwise.
`baseline_probe.py`: *"Everything outside the window keeps its crystal pose."*
The trainer's own banner: *"a 12-residue window is destroyed and every other
residue keeps its crystal pose."* `GrowthOutcome.rmsd`'s docstring: *"the intact
context, which starts correct **and should stay correct**."* It does not stay
correct.

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

## The one test, and why it is not "adding terms and re-running"

`g22_random`'s registration says *"if it fails, we do not add terms and
re-run."* Freezing the context adds **no loss term and changes no weight**: it
passes `frozen = ~damaged` to a parameter that already exists, to make the code
do what every description of the task in this repository already claims it does.
That is a correctness fix, not a tuning knob, and the distinction is the whole
reason the clause is survivable here.

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

## Supports
- [[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]

## Requires
- [[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]
