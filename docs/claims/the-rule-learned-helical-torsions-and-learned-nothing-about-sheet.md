---
type: observation
status: supported
evidence: ours
sources: [33291294, 33291295, 33291297, 33283263, 33249861]
---

# The rule learned helical torsions and learned NOTHING about sheet: the one secondary structure a one-hop rule can see is the local one

Bucket 2, the one cheap measurement never made, made at last.
`scripts/ramachandran_basins.py` on the three readable arms, exit 0 each,
on **the twelve held-out crops each checkpoint records in its own
`eval_indices`**, with `eval_pdb_ids` checked so a pool mismatch refuses
rather than measures.

Four rows, same residues, same boxes: `native` (a wiring check, must read
1.000), `start` (the damaged window before the rule touches it), `floor`
(`geometric_floor.py`, bond closure and sterics only, no network and no
native information), `model`.

    HELIX basin, natively-helical window residues    -10d      +0d     +10d
    native                                          1.000    1.000    1.000
    start                                           0.036    0.034    0.065
    GEOMETRIC FLOOR                                 0.055    0.068    0.065
    g22_random    context free                      0.073    0.102    0.194
    g24_frozen    context pinned                    0.018    0.034    0.097
    g25_equivariant  + rotation fix                 0.218    0.237    0.306

    SHEET basin, natively-sheet window residues      -10d      +0d     +10d
    native                                          1.000    1.000    1.000
    start                                           0.043    0.107    0.119
    GEOMETRIC FLOOR                                 0.085    0.143    0.186
    g22_random                                      0.021    0.125    0.136
    g24_frozen                                      0.021    0.036    0.051
    g25_equivariant                                 0.000    0.089    0.085

Denominators are 55-62 helical and 47-59 sheet residues, so the fractions
are measurements rather than arithmetic on a handful -- though residues
within one window are correlated and these are not 60 independent draws.

## What is read, and why only this

**`model` against `floor`, never against `start`.** The boxes approximate
the Lovell favoured regions and are rectangles where the real basins are
contours: the helix box clips the bridge region, the sheet box swallows
polyproline II, and the `native` row reads 1.000 for *any* box, so it cannot
catch a wrong shape. What licenses the reading anyway is that it is a
**controlled** comparison -- all four rows share the same boxes and the same
residues, so a box that is too wide or the wrong shape biases every row
alike and largely cancels in the difference. **No absolute fraction here is
"the share of residues in the helical basin"**; it belongs to these
rectangles and not to the Ramachandran plot.

## Helix: a real learned signal, in the best arm only

`g25_equivariant` -- the project's best placement at 8.10 A -- sits **3.4x
to 4.7x above the geometric floor at all three widenings** (0.218/0.055,
0.237/0.068, 0.306/0.065), and the sign does not depend on where the box
edge was drawn. That is the first direct evidence in the record that the
rule learned *local backbone conformation* rather than only pushing atoms
toward a centroid. A placement RMSD cannot show this: 8.10 A is compatible
with correct local torsions badly placed, and with no local structure at all.

`g22_random` is above the floor too but weakly (0.073/0.102/0.194). And
`g24_frozen` **crosses** the floor across the widenings (0.018, 0.034,
0.097 against 0.055, 0.068, 0.065), which is not a conclusion at all -- it
is a statement about where the box edge was drawn, and nothing is quoted
from it.

**So the ordering is not monotone with placement.** Placement runs
g22 9.02 > g24 8.49 > g25 8.10 A, but basin fraction runs g24 (no reading)
< g22 < g25. The honest form is: **only the arm with the equivariant
actuator has a clear helical signal**, and the chain of three does not
establish that basin occupancy tracks placement.

## Sheet: at or below the floor in every arm, at every widening

Every arm is **below** the geometric floor on sheet residues, stably across
widenings -- `g25` reads 0.000, 0.089, 0.085 against a floor of 0.085,
0.143, 0.186. A network-free optimiser that only closes the chain and avoids
clashes puts *more* residues in the sheet basin than any trained rule does.
The rule is not merely failing to learn sheet; **it is actively displacing
residues out of a basin that bond closure and sterics alone would have
found.**

**This is the sharpest mechanistic result the project has.** It splits the
two secondary structures along exactly the line the architecture predicts.
A helix is *local*: its hydrogen bond is i to i+4, inside one message-passing
hop of a k = 32 spatial neighbourhood, so a one-hop per-residue rule can in
principle see everything that defines it -- and the measurement says it did.
A sheet is *not local*: its partner strand is arbitrarily far away in
sequence and arrives through a neighbourhood the damaged window does not
contain, since the partner may itself be destroyed or may only become a
neighbour once the window is already roughly in place. One hop cannot carry
it, and the measurement says it did not.

This is the same boundary [[in-regeneration-the-surviving-context-is-the-grid]]
draws from the other side, and it is consistent with
[[the-clash-term-does-not-see-the-ramachandran-plot]]: nothing in the
objective rewards a torsion directly, so whatever basin occupancy exists is
inferred from placement supervision alone.

## What it does and does not do to the central claim

It does **not** rescue it. Bucket 0's bar is 4.0 A on `fit_intact` and the
best arm reads 8.10 A
([[fixing-the-actuator-helped-a-little-and-fixing-the-gradient-reach-made-it-worse-than-geometry]]).
A rule that puts a quarter of helical residues in the right basin and
*fewer* sheet residues there than geometry does has not regenerated a
backbone.

What it changes is the **diagnosis**. The failure is no longer
undifferentiated -- "it does not place a backbone" -- it is now localised:
the rule learns the secondary structure whose defining interaction fits
inside its receptive field, and loses to plain geometry on the one that does
not. That is a statement about the **one-hop local rule**, not about neural
cellular automata in general, and it names the change that would test it:
reach, not another loss term.

## Scope

  * One seed per arm; no arm has been run twice
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).
    The helix gap in `g25` is 3-5x and survives the widenings, which is why
    it is reported as a signal; the g22-vs-g25 difference is not.
  * The floor is one optimiser configuration (Adam, lr 0.03, 1500 iters,
    bond 0.0188 clash 0.0053 achieved), so it is an **upper** bound on the
    true geometric floor. A better optimiser raises every `floor` row and
    shrinks the helix gap.
  * Residues inside one 12-residue window are correlated, so the effective
    sample is smaller than the denominator.
  * `native` reading 1.000 is a tautology up to a wiring error. It tests the
    torsion measurement, the radian conversion and the masking -- nothing
    about the rule.

## Supports
- [[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]

## Requires
- [[fixing-the-actuator-helped-a-little-and-fixing-the-gradient-reach-made-it-worse-than-geometry]]
