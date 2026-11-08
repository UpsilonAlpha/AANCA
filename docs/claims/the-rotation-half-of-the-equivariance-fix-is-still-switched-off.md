---
type: claim
status: open
sources: [growth.py, 32405172, 32489939, 32489940, 32494313]
checked: [32405172]
---

# Every arm in the record applies the emitted rotation in the lab frame, and the reason it was kept has expired

`--equivariant-move --lab-rotation` is what g13, g14 and g15 all ran, and the
second flag means `rigid_relax` skips the conjugation:

    if local:
        shift = (frame @ shift.unsqueeze(-1)).squeeze(-1)
        if not lab_rotation:
            rot = frame @ rot @ frame.transpose(-1, -2)

So the **translation** is read in the residue's own N-CA-C frame and the
**rotation** is left exactly as g1-g9 had it: predicted from invariant features
and applied in the lab frame. By the argument that settled the translation half
([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]) that is the same category
error, unfixed: an invariant encoder cannot represent a lab-frame direction, and
it cannot represent a lab-frame *axis* either. The rule's rotation output cannot
depend on the structure's orientation, and a lab-frame axis-angle vector must.

## Why it was kept, and why that reason no longer applies

`--lab-rotation` exists to isolate the two halves. What it established is that
the twist transition's **amplitude** trebles when the rotation is local:
16.6 deg (`g8_ceiling`) and 17.6 (`g11`) against 50.2 (`g10_equiv_ceil`) and
53.9 (`g12`), two arms either side of one variable. On that basis the lab-frame
rotation has been kept in every arm since.

**But those four arms all have an identity gradient, and the transition belongs
to identity** ([[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]).
`g9_ceil_noid` at `--identity-weight 0` holds twist 9.1-9.2 for 2,000 updates,
and g13, g14 and g15 hold 9.7-12.1 with no jump at update 1000. So what
`--lab-rotation` was measured to suppress is *the amplitude of a break that no
longer occurs in this regime*. There is nothing for the conjugation to amplify.

The claim is therefore narrow and testable: **at `--identity-weight 0` the
reason for `lab_rotation = true` is gone, and dropping it should cost nothing on
twist while making the rotation half representable for the first time.**

## Pre-registration, if run

One arm, `g16_localrot` = `g15_both` with `lab_rotation = false` and nothing
else changed. Written here before submission so the arm can fail.

**The metric is `settled` = the mean of `rmsd_dmg` at updates 1750 and 2000**,
the same rule g13/g14/g15 were judged by, against **g15's settled 10.24**.
Four exclusive buckets:

  * **`settled < 9.90`** -- the rotation was load-bearing and the lab-frame
    axis was costing about as much as either lever gained. The frame half of
    the equivariance fix is then the live direction and `lab_rotation = false`
    becomes the base.
  * **`9.90 <= settled <= 10.58`** -- indistinguishable from g15 at the
    precision available. **This is a null, not a confirmation**: the
    representation argument stands unmeasured, the flag flips to `false` on the
    argument alone because it costs nothing, and the next arm is elsewhere.
    The window is +-0.34 A, which is 0.5x the g14-to-g15 gap and the smallest
    difference this instrument has any claim to resolve.
  * **`settled > 10.58`** -- dropping the lab-frame rotation *hurts*, which the
    representation argument says should be impossible. Then either the
    invariance argument is wrong or the local conjugation has a cost of its own,
    and this node is refuted rather than merely unsupported.
  * **`twist` trebles out of the 9-12 band at any evaluation** -- the amplitude
    effect does not need the identity break after all, the four-arm table
    (16.6 / 17.6 against 50.2 / 53.9) attributed it to the wrong cause, and
    **this claim is wrong whatever `rmsd_dmg` does.** This bucket overrides the
    other three.

**Void condition.** If the run does not reach 2000 updates, or `bond` climbs
above 4.5 (geometry breaking rather than rotating), `settled` is not read and
the arm is reported as void. Without this the metric would be chosen after
seeing which half of the curve looked better.

**Why `+-0.34`:** it is deliberately wider than the 0.15 A adjacent-evaluation
wobble and narrower than the 0.67 A g14-to-g15 gap. It is **not** an error bar
-- no arm has ever been replicated
([[question-has-any-arm-been-replicated-at-a-second-seed]]) -- so a reading
inside it must be reported as "no difference resolvable", never as "holds".

## Why it is not the next arm anyway

It removes an inexpressibility rather than adding a term, which is why it was
argued for first. Two things outrank it:

  * **It would be a third double-ceiling arm.** `g15_both` runs `--native-edges`
    (the target graph) on top of `--move-weight 1` (teacher forcing), so
    whatever `g16` returns bounds the architecture and does not move the
    method.
  * **Its comparator is not yet known to be converged or reproducible**
    ([[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]],
    [[question-has-any-arm-been-replicated-at-a-second-seed]]). Judging a
    +-0.34 A window against a point estimate taken at a budget where the arm
    was still improving at its second-to-last evaluation is the instrument
    fault, not the arm.

Both are cheap to remove and neither makes this node less true.

## Scope

  * This is an argument from `rigid_relax` plus the identity attribution, not a
    measurement. No arm at `--identity-weight 0` with a local rotation exists.
  * **The premise it rests on is weakly held.** The attribution of the twist
    transition to the identity head is `supported` on our own arms only --
    check 1 of the validator flags it for exactly that -- and this node's whole
    case is built on top of it. If that attribution is wrong, the amplitude
    table's cause is unexplained again and the fourth bucket above is the one
    that fires.
  * `32405172` is cited for its twist number. `RETRACTIONS.md` voids that job's
    `move` column only, which this node does not use; acknowledged in
    `checked`.
  * It does not touch the translation half, which is fixed and is what produced
    the movement off the 12.23 A baseline.
  * The amplitude table (16.6 / 17.6 against 50.2 / 53.9) is not disputed here.
    What is disputed is that it generalises to a regime with no transition in
    it.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]

## Requires
- [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]
