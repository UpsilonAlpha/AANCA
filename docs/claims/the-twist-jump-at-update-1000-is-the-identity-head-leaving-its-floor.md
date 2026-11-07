---
type: claim
status: open
sources: [32405172, 32420154, 32449660, 32449661]
checked: [32405172]
---

# Every geometry column degrades at update 1000, in the arm with no equivariance too, and that is when identity leaves its floor

**This node exists because the pre-registered rule it was meant to settle had a
false premise, and the control said so.** Four arms, differing only in how the
emitted move is framed, all break at the *same* evaluation:

    update    g8      g11     g10     g12        <- twist, degrees
      250     9.1     9.1     9.9     9.8
      500     9.1     9.0    10.0     9.5
      750     9.0     9.1    10.2    10.4
     1000    16.1    18.4    52.3    43.3        <- all four, same row
     2000    16.6    17.6    50.2    53.9

    g8_ceiling      lab translation,   lab rotation
    g11_split       local translation, lab rotation
    g10_equiv_ceil  local translation, local rotation
    g12_trunc       local translation, local rotation, --truncate 60

`g8_ceiling` has **neither half of the equivariance fix** and jumps 9.0 -> 16.1
at update 1000, with `fape` 0.146 -> 0.843 and `bond` 6.03 -> 7.60 in the same
row. So the phase transition is **not caused by the equivariance fix**, and the
attribution written into `growth.py`, `train_growth.py` and
[[the-equivariance-fix-did-not-release-a-bearing]] -- that the conjugated
rotation was "the suspect" -- was reasoning from a two-arm comparison with no
control in it. The control was available the whole time: g8's log was already on
disk when that comment was written.

## What identity does in the same row, which is the mechanism

`g8_ceiling` is the only one of the four that prints the identity NLL, and it
is the arm that makes this readable:

    update   type NLL   recovery   twist
      250     3.005        --       9.1     <- ln 20 = 2.996, i.e. uniform
      500     2.992        --       9.1
      750     2.977        --       9.0
     1000     2.799      11.0 %    16.1     <- breaks through, and so does twist
     1250     2.749      11.0 %    13.9
     2000     2.678      13.2 %    16.6

For 750 updates the identity head **sits on the uniform floor**: NLL at or
barely under ln 20, recovery not yet reported. At update 1000 it leaves the
floor for the first time and begins recovering sequence -- and that is the row
where every geometry column gets worse.

Identity's weight is **93.7**, the largest in the objective by two orders of
magnitude. While its NLL sat at the floor it contributed a near-constant and
little gradient; the moment it starts learning it starts consuming the shared
trunk, and geometry pays for it in the same evaluation. That is
[[identity-dominates-the-shared-trunk-by-2350x]] and
[[identity-supplies-the-trunk-that-geometry-reads]] observed *as it happens*
rather than as a static ratio, and it is the same phenomenon as
[[the-identity-term-perturbs-geometry-rather-than-helping-it]] seen from the
time axis.

**Status `open`, and the checker is right to insist.** The only evidence under
this is our own calculation, which is evidence about the calculation and not
about the rule, and the mechanism rests on one arm's NLL column. Marked `open`
rather than argued around.

**Why this is a claim and not an observation.** The coincidence is measured and
it is in four arms, but only one of the four printed the NLL, so "identity
leaving the floor" is a one-arm reading of a four-arm event. The direct test is
`g9_ceil_noid`'s own trajectory -- `--identity-weight 0` with the head and
blind perception kept -- which should show **no jump at 1000** if this is right,
and the same jump if the cause is elsewhere in the trainer.

**That test has not run, and this node said it had.** `g9_ceil_noid` (32419399)
was `scancel`led at 23:39 on 6 Nov with `train exit : 143`, having reached
update 310 -- it was killed to free one of the two GPU slots for
`g10_equiv_ceil`, and "it has already run" was written from the submission
rather than from the log. Its single evaluation at update 250 reads twist 9.1
and `rmsd_dmg` 12.23, indistinguishable from every other arm there and 750
updates short of the question. Resubmitted as **32457571**, resuming from the
update-310 checkpoint, so only the untested 1,690 updates cost anything. Nothing in the code
is keyed to update 1000: there is no learning-rate schedule in the growth
trainer, `--fape-clamp-prob` is at its default of 1.0, and the only `% 1000` in
the tree is a logging line.

## What survives from the frame comparison, and it is not nothing

The **amplitude** splits cleanly on the rotation half, and consistently across
two independent pairs:

    local rotation absent   g8  16.6 deg    g11  17.6 deg
    local rotation present  g10 50.2 deg    g12  53.9 deg

A threefold difference, two arms on each side, one variable apart. So the
conjugated rotation does not *cause* the transition but it makes the rule about
three times more sensitive to it. That is a weaker and better-supported claim
than the one it replaces, and it is the one the split arm was built to produce.

## The pre-registered rule is void, not reinterpreted

What was written down before the arms ran:

> If the twist explosion follows the rotation, the conjugation is wrong and the
> translation half survives; if it follows the translation, the fix released a
> gradient the restraints cannot contain and the lever is their weights.

Both branches assume the fix caused the transition. The control shows it did
not, so **the rule does not get read -- it gets voided.** Picking whichever
branch the numbers nearest fit would be fitting a story to a premise already
known to be false, and on the face of it the onset "follows the translation"
(present in g10, g11, g12) which is the branch that would have been reported.
Recorded because a pre-registered rule is only worth having if a failed premise
is allowed to retire it.

No `## Refutes` edge on
[[the-equivariance-fix-did-not-release-a-bearing]], deliberately: that node's
own conclusion -- `align` -0.096 against a +0.206 null -- is untouched by any
of this, and marking the whole node contradicted would overstate what the
control shows. The twist attribution inside it has been struck out in place
instead, which is the narrower and truer edit.

## Supported by
- [[four-arms-break-at-update-1000-and-only-the-amplitude-follows-the-frame]]

## Supports
- [[identity-supplies-the-trunk-that-geometry-reads]]
- [[question-can-the-architecture-place-residues-under-any-supervision]]
