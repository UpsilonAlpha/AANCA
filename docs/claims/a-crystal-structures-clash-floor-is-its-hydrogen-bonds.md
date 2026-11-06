---
type: observation
status: supported
evidence: experimental
sources: [steric.py]
---

# A crystal structure reads 0.0258 Angstrom of backbone overlap per atom, and all of it is hydrogen bonds

Measured while building `backbone_clash_loss`, and it decided the term's one
free parameter. Classed `experimental` because the quantity is van der Waals
overlap in **deposited structures**: our script takes the measurement, the
thing measured is the PDB.

At tolerance 0 on 1UBQ the loss is 0.025765 A/atom, and the whole of it is
9.64 A of overlap over exactly 64 pairs. Enumerated, they are backbone N/O
pairs at 2.56-2.9 A against a Bondi radius sum of 3.07 -- the worst being
N(1)-O(17) at 2.56 A, 0.51 A "inside" -- plus 19 i/i+2 C-N contacts at ~2.96 A.
Those are hydrogen bonds. **A flat-bottom van der Waals penalty at zero
tolerance necessarily charges a hydrogen bond, because an H-bond is sub-van der
Waals by construction.**

It is a property of folded protein rather than of ubiquitin: 0.0246 on 193L
lysozyme, 0.0283 on 4HHB haemoglobin across 574 residues and three chain
breaks.

**So the tolerance is a correctness choice, not a tuning knob**: at zero, the
term teaches the rule to break hydrogen bonds. Crystal against the damaged
regeneration window against a window shoved into the core:

    tol    crystal    window    shoved    window/crystal
    0.00   0.02576   0.03431   0.20850       1.3
    0.25   0.00196   0.00918   0.13410       4.7
    0.50   0.00002   0.00311   0.09200     150.0
    1.50   0.00000   0.00000   0.01232        --

**0.5 A**, which is neither AF2's published 1.5 nor the 0.25 first proposed.
At 1.5 the damaged window reads *exactly zero*, so the term would be blind to
the structure being trained on. At 0.25 the window is discriminated worse than
at 0.5, because the floor left behind is the same depth as the shallow contacts
a near-miss loop makes. 0.5 zeroes the H-bond floor and still sees the window
at 150x it.

One number worth keeping for its diagnostic value: removing the across-a-
peptide-bond exemption gives 1.118, forty-three times the real floor. That is
the constant-valued failure this kind of term dies of, and it would read as a
large loss that never moves.

## Supports
- [[the-objective-should-restrain-the-output-not-teach-the-placement]]
