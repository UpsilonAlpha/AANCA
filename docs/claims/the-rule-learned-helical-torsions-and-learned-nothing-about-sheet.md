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

## RETRACTED SAME DAY: the sequence-distance explanation of the split

**The split above is measured and stands. The mechanism first written here
for it was wrong, and is withdrawn in full.** It said a helix is local
because its hydrogen bond is i to i+4 and so fits inside one hop, while a
sheet is non-local because its partner strand is *arbitrarily far away in
sequence* -- and concluded that the next thing to test is **reach**.

**Sequence distance is not what a residue perceives.** `graph.py:327`
selects neighbours by distance over *all heavy atoms*, sequence neighbours
merely forced in as well; `rollout.py:497` rebuilds the graph **live at every
step** from the current coordinates, and `fixed_neighbours` -- the only way
to hold a stale edge set -- is used by one diagnostic and never in training.
Two strands in a sheet sit about 4.8 A apart. A native sheet partner is
therefore comfortably inside the k = 32 neighbourhood whenever the residue is
anywhere near where it belongs, no matter how far apart the two are in
sequence. One hop already reaches it.

So **"more hops" does not follow from this measurement and must not be
cited as if it did.** The split needs a different explanation. Three
candidates survived the correction, and the cross-tabulation below
(33303677, 33303679, 33303681, exit 0 each) has now separated two of them:

  1. **One conformational mode, over-applied.** If the rule puts
     natively-sheet residues in the *helix* box, it learned a single local
     mode and applies it everywhere. Decided by cross-tabulating native
     basin against achieved basin -- the torsions are already computed.
  2. **The floor is trivially good at beta.** The beta region is the open,
     extended part of the plot, and an optimiser that only closes peptide
     bonds and avoids clashes will tend to extended conformations by
     default. If the floor dumps natively-*helical* residues into the sheet
     box too, then "the model is below the floor on sheet" says more about
     the floor than about the rule. Same cross-tabulation decides it.
  3. **Registration rather than reach.** A helix is determined by residues
     *inside the window*, so the rule can assemble one wherever the window
     happens to be. A sheet has to be in register with a partner strand at
     a specific offset, which an 8.10 A placement error destroys even with
     the partner plainly in view. Being a neighbour is necessary and not
     sufficient. This one is untouched by the correction, but it predicts
     sheet recovers only when placement does, so it cannot be tested apart
     from the placement failure itself.

## The cross-tabulation, 23 Nov: candidate 1 SUPPORTED, candidate 2 weakened

Where did the natively-SHEET residues go that a row did *not* put in the
sheet box? Four destinations, exclusive and exhaustive: the helix box, the
sheet box, **another favoured region** (alpha-L, taken by difference from
`ramachandran_clash_probe.FAVOURED` so a region added there cannot be
silently counted disallowed) and **no favoured region at all**. The first
version of this probe had three destinations and counted alpha-L with the
disallowed, which let it assert "nothing local was learned" from evidence
that could not distinguish it from "a real conformation, wrong one".

    natively-SHEET residues -> HELIX box       -10d     +0d    +10d
    start                                     0.021   0.036   0.068
    GEOMETRIC FLOOR                           0.021   0.054   0.085
    g24_frozen                                0.043   0.036   0.068
    g22_random                                0.106   0.107   0.119
    g25_equivariant                           0.149   0.125   0.169

**The rule sends natively-sheet residues into the HELIX box at 2-7x the
floor's rate, and the arm that does it most is the arm that learned helix
best.** Conditional on landing in *any* favoured region -- which removes
the shared "off the plot" bulk and is the cleanest form of the comparison --
the helix share of g25's favoured landings runs **88 % / 47 % / 52 %**
across the sweep against the floor's **17 % / 25 % / 30 %**.

**The ordering across arms is the same ordering as helical success.** Helix
over-application runs g24 (at or below floor, no effect) < g22 < g25;
helix-basin hit rate runs g24 (crosses, no reading) < g22 0.102 < g25 0.237.
Two quantities, same order, three arms. That is the same form of
corroboration the rotation-fix arm rested on -- twist and placement moving
together -- and it carries the same weight: suggestive, three points, one
seed each, no error bar anywhere.

So **candidate 1 is supported: the rule learned ONE local conformation and
applies it without regard to what the native conformation is.** That is a
better answer to "why does it not place a backbone" than anything a
placement number gave, and it is a statement about a single shared
per-residue rule under this objective.

**Candidate 2 is weakened but not dead.** The refusal did not fire, so the
floor converged (bond under 0.5) and reading (b) is readable. The floor does
discriminate by native conformation -- natively-sheet to the sheet box at
0.085 / 0.143 / 0.186 against natively-helical at 0.036 / 0.119 / 0.129,
sign stable across the sweep -- so its sheet number is not purely chain
closure landing on the extended region. **But the margin narrows sharply as
the boxes widen (2.4x, 1.2x, 1.4x), and at the two wider settings the floor
puts natively-HELICAL residues in the SHEET box more often than in the helix
box** (0.119 vs 0.068, 0.129 vs 0.065). Chain closure does have a real
extended tendency. The floor's sheet advantage is genuine and thin, and
"every arm is below the floor on sheet" should be quoted with that attached.

**The destination that dominates every row is still DISALLOWED** -- 0.68 to
0.83 for the model, 0.71 to 0.87 for the floor, and the model beats the floor
there by almost nothing (0.678 vs 0.712 at +10). **Most window residues, in
every row, end at a phi/psi in no favoured region at all.** Neither the
learned rule nor plain geometry produces protein-like local conformation for
the bulk of the window, and the helix signal above lives inside the minority
that lands anywhere real.

Candidate 3 remains untested and untestable apart from the placement failure.

The boundary [[in-regeneration-the-surviving-context-is-the-grid]] draws is
still the right frame, and
[[the-clash-term-does-not-see-the-ramachandran-plot]] still applies: nothing
in the objective rewards a torsion directly, so whatever basin occupancy
exists at all is inferred from placement supervision alone.

## What it does and does not do to the central claim

It does **not** rescue it. Bucket 0's bar is 4.0 A on `fit_intact` and the
best arm reads 8.10 A
([[fixing-the-actuator-helped-a-little-and-fixing-the-gradient-reach-made-it-worse-than-geometry]]).
A rule that puts a quarter of helical residues in the right basin and
*fewer* sheet residues there than geometry does has not regenerated a
backbone.

What it changes is the **diagnosis**, though less than the first version of
this note claimed. The failure is no longer undifferentiated -- "it does not
place a backbone" -- because it now has a shape: **helical conformation is
learned, sheet conformation is worse than plain chain closure.** What that
shape is caused by is open, and the three candidates above are the open
question, not a conclusion. **It does not name reach**, and the paragraph
that did has been withdrawn above.

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
