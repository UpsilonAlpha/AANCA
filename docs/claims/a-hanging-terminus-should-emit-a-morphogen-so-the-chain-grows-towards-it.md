---
type: claim
status: open
sources: [growth.py, morphogen.py, a-residue-can-follow-the-morphogen-gradient]
---

# A hanging terminus should emit a morphogen, so the chain grows towards it

Ulban's premise, 15 Nov: *"a hanging N terminus should probably emit a
morphogen so that the chain can 'grow towards' it."*

Open, and it is the first proposal in this project that attacks loop closure
with the machinery that already exists rather than with a new restraint.

## It lands on a problem that was already named and deferred

`growth.py`'s docstring lists what the module does not claim to have solved:
*"two segments that meet are not covalently closed, which is the loop-closure
problem and is left to the bond restraint."* Left to the bond restraint means
left to a penalty that acts only once the two ends are already within reach of
each other -- a restraint on a bond that does not yet exist cannot say which
way to walk to make it exist.

A gradient can. This is **chemotactic closure** rather than analytic closure
(CCD, kinematic closure): the growing end is not constructed to arrive at its
partner, it is attracted to it.

## The measurement that bears on it already exists, and was almost this experiment

[[a-residue-can-follow-the-morphogen-gradient]] put a persistent source on one
residue of 1UBQ and asked every other residue for a translation pointing at it,
with the structure identical in every example so the field was the only input
carrying the answer. Held-out cosine on unseen sources:

    concentration only   +0.709
    + the gradient block +0.960

The source there was arbitrary. **Make it a hanging terminus and the probe
becomes this premise.** So the rule's ability to follow such a source is not
the open part; what it would be following is.

## It also fixes the initialisation problem that gates the whole field

Gap 3 of that node: an untrained sigmoid emit head secretes about 0.5
*everywhere*, so after 20 steps the concentration on 1UBQ is 0.8935-0.9038 and
the gradient block reads |value| 0.0032 mean. The field is saturated and flat,
the rule must learn to secrete **differentially** before the block carries
anything, and nothing in the objective rewards that.

A terminus source is differential **from step 0 with no learning at all**,
because it is placed by the topology rather than predicted. That is the
difference between a field the rule has to bootstrap and a field it can start
reading immediately.

## The design decision this forces: given or learned

  * **Given** -- the rollout injects a source at every unsatisfied covalent
    valence. The field carries real signal from update 0. It is a hand-placed
    input, but it is a **boundary condition rather than a potential**, and the
    precedent is exact: a Growing NCA cell is handed its seed state, and the
    seed is given while the pattern is learned. Nothing is tabulated, so the
    condition from [[the-clash-term-does-not-see-the-ramachandran-plot]] holds.
  * **Learned** -- the rule must discover that a hanging end should secrete.
    Purer, and it walks straight into the chicken-and-egg above: the field is
    flat until the rule secretes differentially, and the rule has no gradient
    towards differential secretion until the field is informative.

**Given is the recommendation**, on the Growing NCA precedent. Stated as a
choice because it is one.

## Hanging ends are plentiful here, not an edge case

Crops are not single chains. `growth.py` measures *"a 128-residue crop at
measured link density of 0.92-1.00 contains about nine covalent runs"*, while
[[a-residue-can-be-placed-without-a-parent-placing-it]] reports *"15.7 anchors
in a 128-residue crop"* at one anchor per run. **Those two figures disagree and
should be reconciled before either is used**, but either way it is ten to
sixteen runs and therefore twenty to thirty-two hanging ends per crop. The
premise applies to most of the structure, not to two residues.

## Four objections to settle before building it

  1. **A scalar source says "towards a terminus", not "towards *your*
     terminus".** Every run has two ends and a crop has dozens. A growing
     C-ward end must find the N-ward end of the next run and not the nearest
     end of any run. This needs either N-ward and C-ward ends in separate
     channels -- now affordable, `EMIT_CHANNELS` is 4 -- or a partner identity
     the field cannot carry, in which case the field gets the window close and
     something else pairs it.
  2. **The reach may be shorter than the gap.** `DEGRADATION = 0.10` was set
     from the perception graph's three-hop diameter and gives a screening
     length near 9 A; the three-hop shell reads 0.0335. Whether that is enough
     signal across a window that must travel several Angstrom is unmeasured,
     and lambda was tuned on ubiquitin only.
  3. **A source attracts everything, not only its partner.** Every residue
     reads the same gradient, so the naive version pulls the whole structure
     towards the cut ends. The rule would have to learn to ignore the field
     unless it is itself at a frontier -- which the closure block (24 features,
     `frontier` is already gated on `links`) is the natural thing to gate on,
     and that gating is the part that is genuinely new work.
  4. **A gradient is a bearing, not a bond.** It gets the ends near each other
     and cannot close them. So this is **complementary to a relaxation step
     per birth, not a replacement for it** -- it solves the half of loop
     closure that the bond restraint cannot see, and the bond restraint still
     has to finish.

## Status

Nothing has run. `morphogen = false` in every arm, so the field has never been
on during training at all, and this premise additionally needs a source
injector that does not exist yet.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Requires
- [[a-residue-can-follow-the-morphogen-gradient]]
