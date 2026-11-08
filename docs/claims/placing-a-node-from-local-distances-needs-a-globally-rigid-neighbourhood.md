---
type: claim
status: open
sources: [Aspnes 2006, Dong 2003, Gala 2024, Ingraham 2023]
---

# Asking a node where it belongs from local distances is the network-localization problem, and it is solvable exactly when its neighbourhood is globally rigid

**The literature answer to "can we look at the graph and place these residues
from local information?" is yes, with a precise condition on which
neighbourhood -- and the condition is what picks interfaces and interior
windows over terminal growth.**

## The problem has a name and a solvability criterion

Placing unknown nodes from distances to known ones is **network localization**
(Aspnes et al. 2006, *A theory of network localization*). Their result, in our
terms:

  * Model the problem as a *grounded graph*: anchors are the residues whose
    positions are known, tied to a common ground.
  * A node's position is **uniquely localizable** iff that grounded graph is
    *generically globally rigid*. Otherwise the distances admit a family of
    solutions -- a reflection, or a continuous manifold -- and no amount of
    local information picks one.
  * The general case is NP-hard; the efficiently solvable subclass is
    **trilateration graphs**, where nodes can be placed one at a time off
    already-placed ones.
  * Random geometric graphs become trilateration graphs with high probability
    once node density passes a threshold.

The classical protein instance is the **geometric build-up algorithm** (Dong &
Wu 2003, J. Global Optimization 26:321): given **four already-placed,
non-coplanar** atoms and their distances to an undetermined atom, that atom's
position is fixed, in O(n) operations. Sit, Wu & Yuan (2009) added a
least-squares form for noisy distances.

**This is the formal version of the project's own intuition about growth
direction.** A residue being extended off a terminus has anchors on one side
only -- a short, nearly-collinear chain of them -- so its grounded graph is not
globally rigid and its position is genuinely under-determined by local
information. A residue in the middle of an interior window, or across an
interface, has intact anchors on several sides and is close to the trilateration
case. The asymmetry is not a heuristic about information richness; it is a
rigidity condition, and it says the richest local environment is wherever the
anchors surround the node. See
[[the-damaged-region-must-start-within-reach-of-where-it-belongs]], which is the
reach half of the same question.

## Two architectures already do this, and they differ from ours in known ways

**E(n)-equivariant Graph Neural Cellular Automata** (Gala, Grattarola &
Quaeghebeur, TMLR 2024) is this project's architecture, on point clouds. One
E(n)-equivariant graph convolution per CA step; the coordinate update is

    x_i <- x_i + (1/|N(i)|) sum_{j in N(i)} (x_i - x_j) phi_x(m_ij)
    m_ij = phi_m(||x_i - x_j||^2, h_i, h_j, e_ij)

5K parameters. Its *pattern formation* task is ours: morph randomly initialised
coordinates into a target point cloud, 8 geometric graphs, 256 to 2503 nodes,
"we succeed with all of them", converging to a stable attractor maintained for
1000 steps past the training horizon and regenerating after local and global
noise. It is trained by BPTT over 15 to 25 steps with a replay pool, the same
persistence trick as Growing NCA.

Three differences from our runs, all measurable:

  1. **The loss is all-pairs distances.** `L_INV` is the MSE of
     `||x_i - x_j|| - ||xhat_i - xhat_j||` over *every* node pair, and the
     authors say it is *"necessary to uniquely identify the target state"*,
     noting that nodes "are not aware of their global location, and have
     |V|-1 constraints to satisfy". Ours is FAPE, clamped at 10 A, so a residue
     13 A out contributes a constant and no signal -- the rigidity condition
     stated as a gradient.
  2. **The computation graph comes from the target, not from current
     positions.** Pattern formation message-passes over the fixed target graph
     G; a node at a random Gaussian start still talks to its *true* neighbours.
     Only their Boids task recomputes a radius graph each step, and Boids has
     no target shape to reach. Ours recomputes every step from the live
     coordinates
     ([[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]).
  3. **No step cap and no truncated window.** phi_x is an unbounded MLP output;
     stability came from weight decay and gradient clipping. Our 0.3 A cap and
     `truncate 12` are both ours, and the second cuts the gradient through the
     iteration that the authors identify as the whole source of global reach:
     *"the effective receptive field ... grows larger with each state update"*.
     See [[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]].

They use no anchors at all -- every node starts random. We have an intact native
context, which is strictly more information than their hardest task had.

**Chroma** (Ingraham et al., *Nature* 623:1070, 2023) is the protein-scale
instance of the decoupling Ulban asked for -- "connecting them into a backbone
should happen naturally or come later". Chroma predicts **inter-residue
geometry** with a random-graph GNN of sub-quadratic cost, then **solves for a
global consensus structure** from those predicted pairwise geometries with
equivariant layers. Local prediction, global solve: the node says what its
distances should be, and a separate step turns a set of satisfied distances
into coordinates. That is the geometric build-up algorithm with a learned
distance predictor in front of it.

## What this says to do here

The proposal is sound and has precedent at both scales. The literature's
conditions on it are:

  * **Supervision must pin the target uniquely.** Clamped local FAPE does not,
    by E(n)-GNCA's own argument; a pairwise-distance term over the window and
    its anchors would. This cuts against
    [[the-objective-should-restrain-the-output-not-teach-the-placement]], and
    the tension is real rather than resolvable by preference: an under-determined
    target is not a weaker form of supervision, it is a different target.
  * **The edge set must be the one that holds in the answer**, or be predicted
    and solved (Chroma) rather than read off the error.
  * **Place where the anchors surround the node**, not from a terminus. Note
    that RFdiffusion's authors report inpainting failing "when insufficient
    starting information is provided", which is the same condition arriving as
    an empirical limit.

**Status `open`.** All literature; nothing here is measured on our system. What
would make it a finding: an arm with a pairwise-distance term over window-anchor
pairs, or with the perception graph held at the native edge set, moving off
12.23 A. Neither has been run, and
[[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]] is the
cheaper of the two to test.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[in-regeneration-the-surviving-context-is-the-grid]]

## Requires
- [[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]
