---
type: observation
status: supported
evidence: experimental
sources: [baseline_probe.py]
---

# A damaged window starts 10.77 Angstrom out with 90 % of FAPE's pairs live, against 17.45 and 26 % for whole-structure growth

`scripts/baseline_probe.py --damage 12`, no model at all, on the same twelve
held-out crops a run sees. Classed `experimental`: the quantity is the distance
between a constructed start state and **deposited crystal structures**.

    baseline    rmsd_dmg   rmsd   fape    live    bond   twist
    helix         10.77    3.45   0.128   90.1%   4.504    9.6
    extended      21.30    6.87   0.157   86.6%  17.945    9.5
    random        14.12    4.57   0.155   87.1%   5.487   10.8

Against the whole-structure baselines on the same crops: helix 17.45 A rmsd,
0.836 fape, **26.5 % live**, 64.4 deg twist.

**10.77 A is the "learned nothing" line for regeneration**, the way 17.45 A was
for reconstruction, and it is the only figure comparable with a `--damage 12`
run's `rmsd_dmg`. An arm above it has made the window worse than leaving it
where the walk put it.

**The live fraction is the number that settles the clamp argument.** 90.1 % of
scored pairs are inside FAPE's 10 A clamp here against 26.5 % on
whole-structure reconstruction, because a damaged residue's neighbours are
intact and nearby. So the clamp gates almost nothing on this task -- without
needing `--fape-clamp-prob` at all, which is the flag that
[[the-fape-clamp-is-what-gates-the-geometry]] was built to test and that
`g6_open` then falsified on the *other* task.

`bond` is 4.5 rather than 0.002 because the window's C-ward join starts as a
broken peptide bond by design, and `twist` is 9.6 deg rather than 64.4 because
most bonded pairs are two intact residues. Neither is comparable with a
reconstruction run's, which is why the banner on a damaged run says so.

## Supports
- [[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
