---
type: observation
status: supported
evidence: ours
sources: [rollout.py, graph.py]
---

# Every step rebuilds the neighbour list from the current coordinates, so a displaced residue reads the neighbourhood of the wrong place

`build_graph` is called **inside** `step_once` (`rollout.py:458`, in the
function defined at `rollout.py:388`), on `coords_seen`, which is the live
state. Neighbours are a k-NN selection over current heavy-atom positions
(`graph.py`), and selection is a hard argsort -- not differentiable, so the
gradient flows through the distance envelope and never through *which*
neighbour was chosen.

In regeneration the damaged window starts about 13.3 A from where it belongs
([[two-instruments-disagree-by-2-5-angstrom-about-where-the-window-starts]]).
So on step 0 a damaged residue's neighbour list is **the neighbours of the
wrong location**. The rule is asked "where do you belong?" while being shown
the environment of the place it does not belong. Its 64 nearest neighbours are
whichever intact residues happen to surround the ideal-helix filler, which is a
fact about the filler and not about the native pose.

## This is the third thing the lattice gives, and the one regeneration does not supply

[[in-regeneration-the-surviving-context-is-the-grid]] argues that the intact
context hands a damaged residue the three things a Growing-NCA grid hands a
cell: where it is, a shared frame, and a determinate neighbourhood -- and that
node's own words for the third are *"who is adjacent is a property of the
substrate, not something the cell works out."*

Ours is worked out, from the current positions, every step. **The context
supplies the first two and not the third.** That is a narrower and more
mechanical reading of why "the grid is already there and the rule is not
reading it": two-thirds of it is there, and the missing third is the one that
would tell a residue which native neighbours to measure itself against.

## Why this matters more than the step size

It stands behind the four routes that all say the rule does not move the window
([[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]).
A rule that reads the wrong neighbourhood has nothing to aim at, so `align`
finding no bearing
([[the-equivariance-fix-did-not-release-a-bearing]]) is what a correct
implementation of a mis-specified perception *should* produce. It also makes
[[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]] the
second constraint rather than the first: widening the gradient window lets a
rule follow a bearing further, and there is no bearing to follow while the
edges describe the mistake.

Not a claim about the fix. Three different edge sets are defensible and
[[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
sets out what the literature does instead; this node only records that the edge
set is currently a function of the error being corrected.

## Supports
- [[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
- [[in-regeneration-the-surviving-context-is-the-grid]]
- [[the-rule-keeps-whatever-conformation-it-is-given]]
