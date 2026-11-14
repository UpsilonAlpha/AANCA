---
type: observation
status: supported
evidence: ours
sources: [growth.py, baseline_probe.py]
---

# Every arm ever run started its damaged window as an ideal helix, and dropping that costs 3.35 A of distance and nothing else

Ulban, 14 Nov: *"I don't think we should have a prior, the model should learn
to form helices and sheets."* Measured the same day with
`scripts/baseline_probe.py --damage 12`, and the objection is right and cheap to
act on.

## Three things are handed over and only one of them is secondary structure

`baseline_coords` builds the window by a reference walk off the intact flanks,
and `growth.initial` only chooses **which torsions the walk uses**. Separating
what that does and does not give away matters, because "no prior" is not
available -- a start state exists whatever we do.

  1. **Ideal covalent geometry.** `rot_child = rot_parent @ rot_reference` makes
     CA-CA exactly 3.804 A at every link, in **every** mode. Removing this means
     starting from arbitrary Cartesian coordinates, which is the
     whole-structure reconstruction task the four frozen arms already failed
     ([[the-four-arms-measured-the-architectures-ceiling]]). Not the thing to
     drop, and arguably not a prior about the answer at all: it is a hard
     stereochemical constraint, not a hypothesis about the fold.
  2. **Each residue's internal atom geometry**, which rides rigidly from the
     crystal pose in every mode. Same argument.
  3. **The torsions, which is the secondary structure.** This is the only thing
     `mode` controls, and this is the one Ulban is objecting to. `initial =
     "helix"` is the default and **has been set in every arm from g1 to g21**.

So the gift is specific: the rule has never been asked to produce a helix. It
has been handed one and asked to move it.

## Why that makes the headline result weaker than it reads

About a third of PDB residues are helical, so a helix start is already roughly
right for a third of windows and the rule inherits that for free. Worse for the
argument the project now wants to make: **"the rule learns to form helices and
sheets" is untestable from a helix start.** There is no way to tell a learned
helix from the one that was in the start state.

From `random` there is. A rule that puts helical torsions where the native is
helical, starting from uniform ones, has demonstrated learned secondary
structure directly -- which is a stronger and more legible result than any
`rmsd_dmg`.

## What dropping it actually costs, measured

12 held-out crops, 12-residue window, laptop pool (954 structures, so these are
**not** comparable with Bunya's numbers --
[[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]] and the
pool warning in `baseline_probe.py`):

    start state   rmsd_dmg   fit_intact    bond    live
    helix           10.77       11.83     4.5043   90.1%
    random          14.12       15.84     5.4872   87.1%
    extended        21.30       23.74    17.9454   86.6%

`random` is **3.35 A further out and nothing else**:

  * **Well inside the rollout's reach.** The ceiling is 120 steps x 0.3 A =
    36 A ([[the-per-step-translation-cap-is-0-3-angstrom]],
    [[the-damaged-region-must-start-within-reach-of-where-it-belongs]]), and
    14.12 is not close to it.
  * **FAPE stays alive.** 87.1 % of scored pairs under the 10 A clamp against
    helix's 90.1 %, so the gradient is not being thrown away
    ([[fapes-ninety-percent-live-is-eighty-two-percent-pairs-with-zero-error]]).
  * **Bond barely moves**, 5.49 against 4.50, because the walk's geometry is
    ideal whatever the torsions are.

`extended` is the mode that would have been a problem -- 21.30 A and a bond term
four times higher -- which is probably why `helix` stuck. It was the safe
choice, and `random` was never measured against it until today.

## And it replaces the control this needed rather than defending it

[[question-is-7-8-angstrom-better-than-relaxing-the-helix-prior]] asked how much
of g20's 4.45 A of progress was the rule and how much was the helix. From a
`random` start that question does not arise: there is no secondary structure in
the start state to inherit, so the null is defensible without a separate
non-learned relaxation baseline.

## The one choice left open

`random` draws phi/psi uniformly on [-180, 180], so it includes Ramachandran
regions no residue occupies. That is a clean null for *secondary structure* and
a poor model of a coil, and the alternative -- sampling a coil background -- is
more physical but reintroduces a prior, a weaker one. Uniform is the choice that
cannot be accused of helping.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
