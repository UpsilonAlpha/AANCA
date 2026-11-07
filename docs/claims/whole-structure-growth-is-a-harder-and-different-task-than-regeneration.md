---
type: claim
status: supported
sources: [32215014, 32241722, 32314557, baseline_probe.py]
---

# Every geometry arm so far trained on whole-structure reconstruction, which is harder than and different from the goal

`--birth none` starts **every** residue at a constructed baseline and keeps one
native anchor per covalent run -- 15.7 anchors in a 128-residue crop. So the
rule is asked to build an entire structure from almost nothing, with native
context at roughly one residue in eight.

Regeneration is the opposite ratio: the intact remainder is most of the
structure and the damaged region is a window in it. Measured on 1UBQ with a
12-residue window damaged and the other 64 residues left native:

    start state                       fape    error outside the window
    whole structure from helix        0.836   17.45 A  (12 held-out crops)
    12-residue window damaged        0.283    exactly zero

**The same loss is three times smaller on the task the project is actually
for**, because FAPE is a mean over residue pairs and most pairs are now between
two intact residues. More to the point, the pairs that are *wrong* have correct
neighbours inside the 10 A clamp, so they carry a direction instead of a
constant.

Three of today's blockers are therefore properties of the task that was
trained, not of the architecture:

  * the clamp gates little when a damaged residue's neighbours are intact;
  * the reach ceiling is not binding when a residue must travel a few Angstrom
    rather than 17 or 36;
  * and **"emit nothing" stops being a free fixed point**, because doing
    nothing leaves the window wrong while the context is right, which is
    exactly the comparison FAPE can make.

That last one matters most. The four arms converged on the identity map, which
is a perfectly stable fixed point and therefore *rewarded* by the half of the
objective that asks for stability past the trained horizon. On a regeneration
task it is not available: the stable answer and the correct answer stop being
the same thing.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]

## A note on pools, since one of this node's premises went void

[[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]] is void:
10.77 A was a laptop-pool figure and the training pool gives **12.23 A**
([[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]).
**The argument above does not use it.** It rests on the FAPE ratio (0.836
against 0.283) and on the error outside the window being exactly zero by
construction, neither of which is a pool statement.

What does need saying: the 17.45 A whole-structure figure has **not** been
re-measured on Bunya's pool, so a regeneration-against-reconstruction
comparison in Angstrom is now mixed-pool and should not be quoted. The live
fraction — 90.1 % against 26.5 % — is a ratio of scored pairs inside the clamp
and is the robust half of the comparison. Re-running `baseline_probe --damage 0`
on Bunya closes this and costs one CPU job.

## Supported by
- [[the-rule-keeps-whatever-conformation-it-is-given]]
- [[the-baselines-are-17-45-and-36-73-angstrom-from-native]]
- [[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]

## Refutes
- [[the-four-arms-measured-the-architectures-ceiling]]
