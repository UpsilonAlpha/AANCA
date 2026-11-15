---
type: question
status: open
evidence: ours
sources: [backbone.py, growth.py, a-residue-can-be-placed-without-a-parent-placing-it]
---

# Why not place the frames in sequence from the N terminus, with the rest frozen?

Ulban, 15 Nov: *"why can't we do sequential growth with the backbone again?
Each residue's frame placed in sequence with N terminal constrained. All other
residues keep their frame fixed, but sidechains mobile."*

The honest answer is that **two of the three things this asks for are already
true, the third is incompatible with the project's task, and the reason is loop
closure rather than taste.**

## The machinery exists and a version of this was the architecture

`backbone.py` is the module for it. `place_atom` is NeRF -- three placed atoms
plus an internal-coordinate triple (bond, angle, torsion) puts the fourth atom
down -- and it is **differentiable in the coordinates and in all three internal
coordinates**, which is what makes a predicted phi/psi trainable rather than
sampled. Sequential C-ward building is three calls per residue.

And the parent-constrained version was the architecture until it was replaced.
The birth-mode progression `torsion` / `cartesian` / `free` had a child inherit
`rot_parent @ rot_reference @ rot_spin` from the residue before it; `birth =
none` replaced all three
([[a-residue-can-be-placed-without-a-parent-placing-it]]). So this is not an
untried idea, it is the superseded one.

## The prize is already in hand: the baseline's peptide geometry is exact

The reason to build sequentially is that bond lengths come out right by
construction instead of by penalty. **That is already the case.** Under `birth
= none` every slot starts at a constructed baseline whose **bond term is 0.0021
before any update** -- exact peptide geometry, no training required.

Which relocates the problem. The 0.95 A bonded-position error is not a failure
to *construct* a chain; it is what the rule *does to* an exact chain once it
starts emitting free frame moves. Sequential construction would hand the rule a
perfect backbone to break in precisely the same way.

## What would actually help is the parameterisation, and it is separable from the order

The useful half of the proposal is not *sequential* but *internal-coordinate*:
have the rule emit `delta phi / delta psi / delta omega` instead of a free
6-DoF frame move. Then the peptide bond cannot break, ever, and the `bond`
loss term becomes unnecessary rather than violated. `place_atom` already
supports it.

The cost is that a torsion change is **local in cause and global in effect**. A
one-degree change at residue 5 swings every atom C-ward of it, with a lever arm
that grows along the chain. `backbone.py` records this as measured rather than
assumed -- regrowing a real protein from its own measured torsions and ideal
Engh & Huber bonds does not reproduce the crystal backbone, and *the error
accumulates along the chain*, which is a property of torsion-space building and
the reason `scripts/growth_probe.py` measures it. It is also why frame
prediction displaced torsion-space folding generally.

## The incompatibility, which is the real answer

**Exact bonds and frozen context cannot both hold.** Internal coordinates
propagate by construction; frozen context forbids propagation. If residues 1-30
and 43-76 keep their frames and the rule rebuilds 31-42 by torsions from the N
side, the chain arrives at residue 43 at whatever position the torsions
produced, and nothing makes that the frozen residue 43. That is the **loop
closure** problem, and it is why fragment assembly needs an analytic closure
step (CCD, kinematic closure) rather than just a good move set.

Regeneration is a frozen-context task -- the surviving remainder is most of the
structure and the damaged region is a window inside it
([[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]).
So the project's own task is the one that forbids sequential internal-coordinate
building. The current design picks frozen context and pays for it with a soft
bond penalty the rule beats by 1.6x.

## Two further costs of the order itself

  * **An order is a clock.** The rule has no step index on purpose, and all
    residues update in parallel every step. "Place residue i once residues
    < i are placed" reintroduces exactly the positional information the
    architecture was stripped of.
  * **It is N times the rollout.** One residue per pass over a 128-residue crop
    is 128 sequential passes where there is now one parallel pass, and the
    gradient has to traverse all of them.

## Where this leaves it

Three options, and the choice is a real one:

  1. **Keep free frame moves and fix the bond by relaxation** -- a few
     closure/relaxation steps per birth. This is already where the three-hop
     ablation is pointing.
  2. **Internal-coordinate moves, whole-structure task only.** Exact bonds for
     free, no `bond` term at all, but it gives up regeneration, which is the
     root claim.
  3. **Internal-coordinate moves plus a closure term**, which is the hard and
     interesting version: the rule proposes torsions and something has to make
     the window meet its C-side anchor.

Unresolved. Recorded because the premise "sequential building would fix the
chain" is the natural reading and is wrong for a specific reason.

## Supports
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]
