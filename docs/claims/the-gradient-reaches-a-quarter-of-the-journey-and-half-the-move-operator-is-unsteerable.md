---
type: observation
status: supported
evidence: ours
sources: [33249861, regeneration.toml, model.py]
---

# Two reasons the rule does not place a backbone that survive scrutiny: the gradient reaches 27 % of the journey, and 45 % of every rotation is a direction an invariant encoder cannot choose

Asked on 21 Nov after `g22_random` failed its absolute bar
([[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]):
**why** does it not place a backbone? Three mechanisms were proposed and **all
three now stand**. The third --
[[the-rule-displaces-the-surviving-context-it-is-supposed-to-read]] -- was
inferred from a bad method, retracted on a null that could not exhibit the
effect, and then confirmed by direct measurement at 2.34 A. These two were read
off the config and off a previously measured number rather than off that
decomposition, so they were never affected either way.

Neither cost a GPU hour to establish.

## 1. BPTT reaches about a quarter of the journey

`truncate = 12` against `steps = 120` in `[base.inputs]`
(`configs/regeneration.toml`), and **nothing in g22's lineage overrides it** --
not `g14_pairwise`, `g19_nomove`, `g20_orient` nor `g22_random`. Gradient
therefore flows through the last 12 steps only. At the 0.3 A per-step
translation cap that is **3.6 A of visible travel against a start state 13.3 A
out of place: 27 %**.

This was already measured model-free and already named as the binding
constraint, with the per-step cap having 3.6x headroom and so binding on
nothing
([[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]).
That note called itself *"pre-positioning rather than a result -- it says which
rollout hyperparameter to test first"*. `g22_random`'s failure is the condition
it was waiting for, and `truncate` is already a config key, so the test costs a
slot and no code.

The rule is trained on the end of a journey whose first three quarters it must
also make with nothing supervising them.

## 2. Half the move operator is a direction the rule cannot choose

`g22_random`'s own log header: *"EQUIVARIANT MOVE: the emitted move is read in
each residue's own N-CA-C frame. **TRANSLATION ONLY: the rotation is left in the
lab frame.**"*

The perception is invariant by construction -- measured at 1.0e-08 on the
rotation half -- so a lab-frame rotation is a direction an invariant encoder
cannot represent, and no objective can teach it
([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]).
Measured, the rotation half is **0.68 A of error on a 1.50 A move, 45 %**,
which is *larger* than the 27 % translation half that was already fixed, and
`equivariant_move` leaves it untouched. Pinned in
`tests/test_move_equivariance.py::test_a_lab_frame_rotation_is_not_equivariant_either`.

`g23_localrot` tested switching it off and settled at `rmsd_dmg` 8.33 `fit_all`
with twist 12.9, **void on its own guard**, so this is still open after a
400-minute run.

## Why these are consistent with the 0.46 angstrom margin

The rule learned the terms with dense local in-window gradients and did not
learn placement: `bond` fell **4.278 -> 0.629** and `clash` to 0.0148, while
placement went 13.44 -> 9.02 `fit_intact`, which is 0.46 A past what bond
closure and steric overlap alone force. Those two terms are local, in-window and
scored every step; placement is the one that has to survive truncation and be
actuated through a half-broken rotation.

**So "this architecture cannot place a backbone" is not established.** What is
established is that this *training setup* cannot, and two specific reasons why
that are cheap to test one variable at a time.

## Scope

  * **No new measurement of the rule.** 1 is a reading of the config plus
    arithmetic already in the record; 2 is a previously measured number
    restated with its test named.
  * **Neither is shown to be THE cause.** They are candidate causes with
    magnitudes attached. Raising `truncate` or fixing the rotation could each
    leave the result where it is, which would itself bound how much they
    mattered.
  * One seed throughout
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]), so
    a difference smaller than ~1 A from either test is not readable.
  * The third mechanism is now measured and holds at 2.34 A of context drift
    on the trained checkpoint, so "two reasons" in this node's title is a
    statement about what THIS node carries, not a count of the causes.

## Supports
- [[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]

## Requires
- [[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]
