---
type: observation
status: supported
evidence: experimental
sources: [baseline_probe.py]
---

# The helix baseline scores 17.45 A and the extended 36.73 A against native, with no model at all

`scripts/baseline_probe.py` runs **no model**. It builds each baseline on the
twelve held-out crops a run would see and scores them, so these are the numbers
an arm that learns nothing must report:

    baseline       rmsd    fape    live%    bond   twist
    helix         17.45   0.836   26.5%   0.0021    64.4
    extended      36.73   0.923   13.5%   0.0033    84.1
    random        14.40   0.916   15.8%   0.0020    90.6

Classed `experimental` rather than `ours` because the quantity is the distance
between a constructed reference and **deposited crystal structures** -- our
script takes the measurement, but the thing measured is the PDB.

Two consequences used throughout this map. First, any arm reporting its own
baseline's value has learned no geometry, and the test is exact to two decimals
rather than a judgement. Second, the 64.4 deg figure independently confirmed
the collapse claim: a trained rule scoring ~64 deg of twist is scoring the
natives' own distance from the helical reference, which is arithmetic checkable
with no weights.

## Supports
- [[the-rule-keeps-whatever-conformation-it-is-given]]
- [[a-collapsed-rules-twist-cannot-be-compared-across-crop-samples]]
