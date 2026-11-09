---
type: question
status: open
---

# Can the rule learn its own perception graph, by pruning an over-complete one rather than being handed the native one?

Ulban's proposal, 9 Nov, in the same message that retired `native_edges`.
Written down before anything is built, because it is the first idea in this
project that attacks the perception problem without a ceiling.

## The problem it addresses

`build_graph` runs inside `step_once` on the **live** coordinates, so a residue
13 A from where it belongs selects the neighbourhood of the wrong location and
then tries to place itself from it. Two things have been tried:

  * **Recompute from live coordinates** (the default): nine arms sat at
    12.18-12.85 A, their own baselines
    ([[the-rule-keeps-whatever-conformation-it-is-given]]).
  * **Freeze at the native edge set** (`native_edges`, retired): the arms moved,
    reaching 6.69 A at 10,000 updates
    ([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]]) --
    but the selection *was* the answer's topology, so the result bounds the
    architecture and reports nothing about a rule.

Neither is a method. The first starves perception, the second buys it.

## The proposal: start over-constrained and prune

Give every residue far more candidate edges than it can keep -- in the limit all
N-1 -- and make the rule **score and discard** them until what remains supports
a geometrically realistic placement. Rather than *finding* its neighbours from
wrong coordinates, the rule *eliminates* non-neighbours.

**Why this is the principled replacement for `native_edges` and not a variant of
it.** Both arrangements guarantee the true edges are reachable. The difference is
who supplies them:

    native_edges              "here are your true contacts"
    over-complete + prune     "your true contacts are in this set; find them"

The first is a leak. The second is a search problem whose solution is the thing
we want, and it removes the failure that froze nine arms -- a residue can never
again have a neighbour list that fails to contain its actual neighbours -- while
leaving the work undone.

**And the learned edge set is the interpretable object.** The payoff the whole
local-rule constraint exists for is that a single shared rule can be ablated or
autoencoded where a pairformer cannot
([[a-local-per-residue-rule-can-build-a-protein-backbone]]). If the rule
converges on an edge set, the structural interactions it learned are *the
output*, readable directly, rather than something to be probed for in a hidden
state. This is a strictly better interpretation target than anything the current
design offers.

## Three objections to settle before building it

**1. A complete graph is global information, which is the thesis.** One
message-passing hop over all pairs is what a pairformer does. If the rule starts
complete, then during training it has global access, and "no global coordinator"
survives only as an *outcome* -- the converged edge set happens to be sparse and
local -- rather than an architectural guarantee. That is a weaker and arguably
more interesting claim, but it must be made deliberately and not discovered
later.

The mitigation that keeps both properties: a **hard per-residue edge budget**.
The rule scores N-1 candidates with a cheap scalar and must prune to exactly
k, so message passing is never all-pairs; only the *selection* sees everything,
and it sees it through one number per pair. Locality stays structural. Whether
a scalar-scored selection is enough to find the true contacts is the open
empirical question.

**2. Self-certification -- the rule can satisfy the objective by editing the
graph instead of the coordinates.** "Geometrically realistic" scored against the
rule's own learned edges is a free lunch: prune away whatever contradicts the
current wrong geometry and the structure is consistent by construction. This is
the mirror of the cofolding co-adaptation failure, and it is the serious one.
It needs a term that does not depend on the learned edges at all -- clash, bond
and reach are exactly that, and they are built
([[question-can-the-architecture-place-residues-under-any-supervision]], rung 2).
A pruning arm therefore cannot be run under FAPE alone; it requires the physical
restraints that have never been switched on.

**3. The degenerate optimum is to prune everything.** No edges, no forces,
nothing moves -- the identity-map fixed point that froze four arms. The budget
in objection 1 doubles as the fix: prune to exactly k, not at most k.

## What it would cost to try

Less than it sounds. `FixedNeighbours` in `torch_graph.py` already accepts a
caller-supplied selection, so the seam exists and was kept when `native_edges`
was removed. Edges already carry a **continuous envelope weight** recomputed
every step; making that weight *learned* rather than distance-derived is the
cheapest first version of this idea and needs no discrete machinery at all.
A hard selection afterwards can reuse the straight-through one-hot the identity
head already uses, so the taste and the code for discrete commitment both exist.

## How it compares to E(n)-equivariant Graph NCA (TMLR 2024)

**It is a departure, not a variant.** E(n)-GNCA message-passes over the target
graph and learns the embedding. That is sound in its own setting, where a
geometric graph is the problem's *input* and reproducing its embedding is the
task, so reading the edges is the task statement rather than a leak. Protein
design has no target graph at inference, which is what turned the same wiring
into `native_edges` here.

The proposal inverts their fixed point: **learn the graph and the embedding
jointly**, with the graph as a pruning process. Not aware of E(n)-GNCA doing
edge editing, and not confident about the wider literature -- learned graph
structure, differentiable rewiring and latent graph inference all exist in
graph-ML generally, and a search is owed before claiming novelty.

## One measurement that bears on it already, and cuts against the intuition

The proposal's motivating intuition is that given a graph, residues should place
themselves approximately correctly. **They did not.** `g17_long` had the native
selection, live distances, teacher-forced moves and 10,000 updates, and settled
at **6.69 A** -- not an approximately correct backbone. A contact graph with
live distances ought to nearly determine coordinates, so either the architecture
is weaker than the information available to it, or a 10 A selection
underdetermines geometry more than expected. That distinction is worth resolving
before a pruning arm, because pruning can only ever recover the graph -- and the
graph was not sufficient.

## Requires
- [[question-can-the-architecture-place-residues-under-any-supervision]]

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
