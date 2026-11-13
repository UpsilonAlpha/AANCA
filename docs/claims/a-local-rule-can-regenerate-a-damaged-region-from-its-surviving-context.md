---
type: claim
status: open
sources: [growth.py]
---

# A local per-residue rule can regenerate a damaged region of a protein from the context that survived

**The near-term goal, and the most useful thing this project could produce.**
Delete a part of a protein and have the rule fill it back in using only local
information from the intact remainder: that is the Growing-NCA objective --
regeneration from damage, stability past the trained horizon -- applied to
protein geometry, and a prior-art search on 2026-10-27 found it unpublished.
The novelty is the **objective**, not the architecture; existing packers
predict in a fixed number of passes or along a fixed diffusion schedule, and
none trains for a fixed point reached from many starting conditions over a step
count it was never told.

It supports the thesis rather than being it: a rule that fills a window from
intact context is evidence that local information suffices to place a residue,
and the thesis asks for the whole backbone. The two applications the project is
for hang off this node, not off the thesis, because both are regeneration
problems.

`status: open` as of 13 Nov, and now for a different reason: **trained, and it
worked.** `g20_orient` moved a 12-residue window from the 12.23 A baseline to
**7.78 A** on 12 held-out crops with **no native information inside the
rollout** -- no teacher forcing, no target graph
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]).
Nine earlier arms without teacher forcing had sat at their own baselines to two
decimal places.

What keeps this `open` rather than `supported` is one specific gap, not
caution: **no arm in this project has been replicated at a second seed**,
because `seed` also chooses the held-out set, so the one replicate attempted
measured different proteins
([[question-has-any-arm-been-replicated-at-a-second-seed]]). The claim is one
`--eval-indices` flag and one run away from supported, and that run is the
cheapest thing in the queue.
`--damage N` destroys a contiguous window inside one covalent run and keeps
every other residue at its crystal pose. Measured on an untrained rule over
three held-out crops with a 12-residue window, against the reconstruction arms'
own held-out numbers:

    quantity        regeneration   reconstruction
    fape                0.186           0.836
    twist              10.7 deg        64.4 deg
    rmsd_dmg          10.76 A          --

fape and twist are means over pairs, and with the context intact most pairs are
already correct -- which is why those two numbers are **not** comparable
between the tasks, and why a regeneration run reports `rmsd_dmg`: the window
alone, superposed on the context so it measures placement rather than plausible
shape elsewhere.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[regrowing-a-designed-interface-can-produce-a-better-binder]]
- [[a-peptide-motif-can-be-grown-a-scaffold-around-it]]

## Requires
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
- [[the-damaged-region-must-start-within-reach-of-where-it-belongs]]
