---
type: question
status: open
---

# The regrowth starts from an ideal helix, so nobody has checked how much of 7.78 A the rule is responsible for

Raised 14 Nov as part of the rescope, and it is the first question a reviewer
would ask of the project's one capability result.

## The start state is a prior, not noise

`src/aanca/growth.py`, the start-state branch:

    anchors = ~damaged
    start_coords = baseline_coords(
        native_coords, topo, links, anchors, growth.initial, generator
    )

`growth.initial` is `"helix"` in every arm ever run. So the damaged window's
native pose is discarded and the window is **regrown in torsion space off the
intact flanks at ideal (phi, psi, omega)**. That is a stereochemically valid
polypeptide anchored at one end to the surviving structure -- an informative
guess about where the window goes, not a corruption of where it was. The
distinction was only stated clearly on 14 Nov
([[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]).

## Why that makes the headline number ambiguous

`g20_orient` reports **7.78 A** against a **12.23 A** baseline, so 4.45 A of
progress ([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]).
But 12.23 A is the *un-relaxed* prior as drawn. The missing comparison is the
same prior given any cheap non-learned relaxation against the fixed flanks --
closing the far-end gap, or a short minimisation under the project's own bond
and clash terms with no rule at all. If that reaches 8 A, the rule has
demonstrated nothing yet; if it stalls near 11 A, the 4.45 A is the rule's.

Two reasons to expect the prior is stronger than 12.23 A suggests:

  * **The flanks pin one end exactly.** A 12-residue window inside one covalent
    run has both ends constrained by residues at their crystal poses, so the
    space of valid placements is already small.
  * **The untrained rule is the identity map by construction** -- the delta
    head is zero-initialised -- so 12.23 A is literally "the prior, untouched",
    which is the weakest possible control rather than a competitive one.

## What would answer it

A non-learned baseline on the same 12 held-out crops: draw the helix prior,
relax it against the fixed flanks under `bond` and `clash` only, report
`rmsd_dmg` through the same superposition
([[rmsd-dmg-superposes-on-the-damaged-window-it-is-measuring]]). Inference only,
no training, no queue. Until it exists, 7.78 A is a number against a null that
nobody has defended.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
