---
type: observation
status: supported
evidence: ours
sources: [baseline_probe.py, growth.py]
---

# The regeneration metric fits on every residue, damaged ones included, which its own docstring says not to do

`backbone_rmsd_on`'s docstring states the reason the function exists:

> A damaged window superposed **onto its own native copy** scores well for any
> plausible shape placed anywhere, because the superposition absorbs exactly
> the error being asked about [...] So the rotation is fitted on the intact
> structure and the deviation is then read off the window.

**Both callers hand it the mask that does the thing it warns against.**
`growth_objective` passes `geometric`, which is `final.alive` (line 1695), and a
`--damage` run reports `alive 100.0%` at every evaluation -- so the fit includes
the twelve damaged residues. `baseline_probe` passes `alive`, all ones, while
its own banner read "superposed on the intact context".

Measured by running the baseline probe with both masks on one start state, so
nothing varies but the mask:

    baseline     fit_all  fit_intact    diff   crops
    helix          10.77       11.83    1.06      12
    extended       21.30       23.74    2.44      12
    random         14.12       15.84    1.73      12

**So every number this project has reported is about 1 A flattering**, and the
10.77 A "learned nothing" line in
[[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]] is on the
lenient convention. The saving grace, and the reason nothing has to be
withdrawn: arms and baseline are flattered *equally*, so every arm-to-baseline
comparison made so far survives unchanged.

## It is also the whole of this morning's "gain"

[[the-rollout-gains-one-angstrom-and-it-is-not-the-rules]] reported that three
trained arms and an untrained one all closed 1.01-1.02 A, and concluded from
the agreement that the gain was not the rule's. **The agreement was mine:**
`gained` was `need - rmsd_dmg`, and the new `need` fitted on the intact context
while `rmsd_dmg` fitted on everything. The column subtracted two different
superpositions, and 1.06 A is what that difference is worth on the helix
baseline.

Computed on one mask the gain is about zero, which is the same conclusion
arrived at honestly: **the rule closes none of the gap.** The readout now
returns the start distance on *both* masks, uses the matched one for `gained`,
and prints the other beside it, so the convention cannot be invisible in the
column again.

Why this kept happening: a quantity whose name says what it means
(`rmsd_dmg`, "the window superposed on the intact context") and whose code does
something else is unfalsifiable by reading either one alone. The three faults of
this kind in two days were all caught by computing the same thing two ways,
which is the only method that has worked.

## This settles about 1 A of a 2.5 A disagreement, not all of it

[[two-instruments-disagree-by-2-5-angstrom-about-where-the-window-starts]] asked
whether the window starts 10.77 or 13.28 A out. The mask accounts for 1.06 of
it: the probe's own intact-fit figure is **11.83 A**, still 1.45 A short of the
readout's 13.28. The remainder has to be a different *start state* -- the probe
builds its own helix placement, the readout reads the first frame of a real
rollout -- and that part is still open.

## Supports
- [[two-instruments-disagree-by-2-5-angstrom-about-where-the-window-starts]]
- [[the-rollout-gains-one-angstrom-and-it-is-not-the-rules]]
