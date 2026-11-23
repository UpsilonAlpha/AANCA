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

## MEASURED 23 Nov: handing the rule the native edge set releases no bearing

Jobs 33319941/42/43/62, exit 0 each, both perceptions on each of two
checkpoints, with every log stating its own perception so neither can be
mistaken for the other. `align`, mean over the twelve held-out crops:

    checkpoint            live k-NN edges    NATIVE edges
    g28_ceil_local             0.036            0.038
    g25_equivariant            0.141            0.208

**`g28`'s live reading reproduces the recorded +0.036 exactly, per crop as well
as in the mean**, which is what licenses reading the rest: the readout is
deterministic and the comparison is of the thing it claims.

**Neither arm's bearing moves out of the null band** (-0.10 to +0.25, from
[[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]]), and `g25`'s
0.208 lands on the convention-matched null of +0.206 almost exactly. So **the
strongest form of this node's hypothesis is dead**: the bearing is NOT sitting
in the native neighbourhood waiting to be read. Whatever the edge set withholds,
it is not something these weights could use if handed it.

**This was pre-registered as the ambiguous branch and it must not be read as
closing anything.** A jump would have been conclusive; a null is not, because
neither set of weights ever trained on edges it did not see, so "cannot use
them" and "never learned to use them" are indistinguishable here. The
unambiguous version needs training on the native edge set, which under
[[the-task-is-to-realise-coordinates-for-a-given-contact-graph]] is now the
task statement rather than a leak.

**A new number falls out, and it is not a small one: `g25_equivariant` -- the
project's best arm at 8.10 A -- has `align` 0.141, inside the null band.** Its
bearing had never been read. So the best free-running arm has no more of a
bearing than the teacher-forced one, which is independent corroboration of the
capacity NO rather than a separate finding.

**One defect in the instrument, found by reading `need` rather than `align`.**
The comparison is nearly paired, not exactly paired: the window's starting
distance moves between the two runs of the same checkpoint -- 15.90 -> 16.14 A
on `g25`, 12.34 -> 12.28 A on `g28` -- although only the edge set was meant to
change. The likely mechanism is that building the native selection draws on the
same generator, shifting the RNG stream and so the crop `load_crop` returns, as
that function re-crops on every visit by design. `g28`'s null survives this
easily. **`g25`'s 0.141 -> 0.208 does not and must not be reported as a rise**,
being the same order as the noise a crop difference can induce. Fix the pairing
before this flag is used again.

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
