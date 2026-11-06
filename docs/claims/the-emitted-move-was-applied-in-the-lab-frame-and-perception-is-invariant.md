---
type: observation
status: supported
evidence: ours
sources: [growth.py, rollout.py]
---

# The network cannot see the global orientation, and its move was applied in the lab frame

Measured 7 Nov, and it explains every arm from g1 to g9.

Rotate the whole structure by 40 degrees and the rule's output does not change:

    |axis_angle diff|    1.03e-08    (|axis_angle| = 0.039)
    |translation diff|   1.24e-08    (|translation| = 0.114)

That is numerical noise, and it is correct behaviour for the encoder --
perception is invariant by construction. The defect is what `rigid_relax` then
did with the output:

    moved = (coords - ca) @ rot.T + ca + translation

The translation is added to **lab** coordinates and the rotation composes on
the left. So the same local neighbourhood, rotated, produces a move in a
different local direction:

    |step(Qx) - Q step(x)|   0.36 A mean, 1.66 A max
    |step(x) - x|            1.32 A mean  (the move itself)

A third of the move, every step, compounding over the 120 a rollout takes.

**A lab-frame direction is exactly what an invariant encoder cannot represent.**
The rule could not express "move toward my target" however it was trained, and
no objective could have taught it. The sign of the ceiling term was verified
correct first (`cos(descent step, ideal) = +1.0000`), so this is not that.

It accounts for the whole record, which is the reason to believe it rather than
the argument for it:

  * teacher-forced, the emitted translation is anti-correlated with the ideal
    one -- cosine -0.259 over 12 crops, spread straddling zero
    ([[the-teacher-forced-rule-emits-moves-pointing-away-from-the-target]]);
  * two arms with no placement term in common park at the same 12.23 A,
    because a near-constant vector applied 120 times arrives somewhere fixed
    ([[both-arms-park-at-12-23-angstrom-by-update-250]]);
  * the emitted step length is identical to three digits on every crop
    (0.0302-0.0315 A against a 0.30 cap), which is what a constant looks like;
  * four reconstruction arms froze at their own baselines
    ([[the-rule-keeps-whatever-conformation-it-is-given]]);
  * `g6_noid` asked for 0.3 % of its budget with identity gone, the best
    constant being zero ([[g6-noid-never-left-the-helix-baseline]]);
  * and FAPE supplied "a magnitude and not a bearing"
    ([[the-reach-probes-split-the-three-frozen-arms]]) -- now mechanical rather
    than mysterious, because a distance is invariant and learnable while a
    lab-frame direction is not representable.

`--equivariant-move` interprets the output in the residue's own N-CA-C frame:
translation becomes `R_i t_i`, the rotation conjugates into that basis, and a
rotation preserves norm so the 0.3 A cap still bounds the step. Off by default,
because g1-g9 are the evidence and must stay reproducible.
`tests/test_move_equivariance.py` pins the premise, the defect and the fix;
`g10_equiv_ceil` (32420154) is the arm, one variable from `g8_ceiling`.

**Found because the pMHC binder session warned that a sign error and a frame
convention mismatch are indistinguishable in a single number.** I was holding
the -0.259 pending an untrained control, which would not have distinguished
them either.

## Refutes
- [[fape-can-teach-the-frame-head-which-way-to-move]]
- [[identity-supplies-the-trunk-that-geometry-reads]]

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
