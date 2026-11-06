---
type: claim
status: supported
sources: [baseline_probe.py]
---

# A collapsed rule's twist score measures the crop, not the rule, so it cannot be read across samples

A rule that emits approximately one conformation scores, per bonded pair, **the
natives' own distance from that conformation**. That distance is a property of
the structures in the crop sample, not of the weights: 64.4 deg on the g5 crops
and about 95 deg on g4's.

So any twist comparison between runs evaluated on different crop samples is
measuring the sample. This is what invalidated
[[the-orientation-term-buys-22-degrees-of-twist]], and the model-free
confirmation is that a pure helix baseline scores 64.4 deg on these crops with
no weights at all -- matching the trained figure to a tenth of a degree.

The operational consequence, now enforced in two probes: a diagnostic must run
at the checkpoint's own crop size and split, not on a convenient PDB.

## Supported by
- [[the-baselines-are-17-45-and-36-73-angstrom-from-native]]
