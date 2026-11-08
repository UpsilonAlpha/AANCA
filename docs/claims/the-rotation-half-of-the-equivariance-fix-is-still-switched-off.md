---
type: claim
status: open
sources: [growth.py, 32405172, 32489939, 32489940, 32494313]
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

## What it predicts, if run

One arm, `g15_both` with `lab_rotation = false` and nothing else changed:

  * `twist` stays in the 9-12 band, because the amplitude effect needs the
    identity break. **If twist trebles with identity at zero, this claim is
    wrong** and the conjugation has a cost of its own that the five-arm table
    attributed to the wrong cause.
  * `rmsd_dmg` improves or holds. The window has to rotate as well as travel,
    and nothing has yet been able to express the rotation.

That is a cheaper and more fundamental arm than weighting the window's internal
half ([[the-two-levers-together-beat-either-alone-and-never-give-the-gain-back]]),
because it does not add a term -- it removes an inexpressibility.

## Scope

  * This is an argument from `rigid_relax` plus the identity attribution, not a
    measurement. No arm at `--identity-weight 0` with a local rotation exists.
  * It does not touch the translation half, which is fixed and is what produced
    the movement off the 12.23 A baseline.
  * The amplitude table (16.6 / 17.6 against 50.2 / 53.9) is not disputed here.
    What is disputed is that it generalises to a regime with no transition in
    it.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]
- [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]
