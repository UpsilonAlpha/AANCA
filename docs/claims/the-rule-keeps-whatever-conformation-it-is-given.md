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

## Refutes
- [[fape-can-teach-the-frame-head-which-way-to-move]]

## Supported by
- [[g6-noid-never-left-the-helix-baseline]]
- [[g6-self-wobbled-around-the-helix-baseline-and-broke-its-bonds]]
- [[g6-ext-never-left-the-extended-baseline]]
- [[the-baselines-are-17-45-and-36-73-angstrom-from-native]]
