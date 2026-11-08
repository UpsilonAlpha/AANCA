---
type: claim
status: supported
sources: [32215014, 32241722, 32314557]
---

# Training does not change the conformation the rollout starts from

Three arms, three full runs of 8,000 updates, three freezes -- each at its own
starting value, to two decimal places:

    arm       baseline      final rmsd   baseline rmsd
    g6_noid   helix          17.45 A        17.45 A
    g6_self   helix          17.44 A        17.45 A
    g6_ext    extended       36.75 A        36.73 A

The arms differ in whether identity is supervised and in which baseline they
start from, and neither difference matters to the outcome. Because the two
baselines are nineteen Angstrom apart, this is not a basin the optimisation
falls into: **it is the optimisation never leaving.**

What rules this out as a measurement artefact is that identity trains normally
in every arm over the same updates -- NLL below the ln-20 floor, recovery to
22 % -- so the rollout, the data pipeline and the optimiser are all working.
Only the geometry is static.

## Narrowed 8 Nov: this is about whole-structure growth under clamped FAPE

Two regeneration arms have now moved off their baseline and stayed off it
([[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]]):
0.8 A with the perception graph held at the native edge set, 1.5 A with an
unclamped window-to-anchor distance restraint. So the title of this node, read
as a general statement, is **false**.

What it still says, and what its evidence actually measured, is narrower and
unchanged: **three whole-structure arms, 8,000 updates each, under clamped
FAPE and a perception graph rebuilt from the rule's own wrong coordinates, did
not move at all.** Nothing about those three runs has been re-measured or
retracted, and the 17.45 / 36.73 figures stand
([[the-baselines-are-17-45-and-36-73-angstrom-from-native]]).

**Narrowed rather than voided, deliberately.** The freeze was real and thirteen
nodes rest on it. What the new arms show is that it was a property of the
objective and the perception rather than of the architecture -- which is the
reading [[in-regeneration-the-surviving-context-is-the-grid]] argued for before
either arm ran, and the reason this node was never evidence that the thesis is
impossible. Anyone citing it for "the rule cannot move a residue" is citing it
past its scope.

## Refutes
- [[fape-can-teach-the-frame-head-which-way-to-move]]

## Supported by
- [[g6-noid-never-left-the-helix-baseline]]
- [[g6-self-wobbled-around-the-helix-baseline-and-broke-its-bonds]]
- [[g6-ext-never-left-the-extended-baseline]]
- [[the-baselines-are-17-45-and-36-73-angstrom-from-native]]
