---
type: observation
status: supported
evidence: ours
sources: [morphogen.py, rollout.py, model.py]
---

# The morphogen is now one secreted chemical spreading over the graph the rule already perceives, and it cost 16 of 20 node features

Ulban, 14 Nov: *"make the emitted channels diffuse over the perception graph --
we want to have growth similar to growing NCA, but for proteins."* Built the
same day, and the change is a removal.

## What it is

In Growing NCA a cell's channels beyond the visible ones are hidden chemicals
that spread by a **fixed local kernel over the grid**, while what each cell
secretes is **learned**. The grid is the thing a protein does not have
([[growing-nca-grid-is-the-missing-information]]), so the neighbourhood the
chemical spreads over is the spatial graph the rule already perceives --
selected from the model's own coordinates, rebuilt every step as the structure
moves.

    field_new = (1 - lambda) * (field + D * (mean over perceived neighbours - field))
                 soft-or  secretion

`diffuse` now takes `graph.neighbours` and `edge_mask` instead of `links`, so
the chemical spreads over exactly the pairs the rule can see and nothing has to
be kept in agreement with anything.

## What went, and the arithmetic

Both chain-transported groups: `NWARD_DECAYS`, `CWARD_DECAYS`, `_live_bonds`,
the terminus clamping. They converged to `decay ** index` and were a positional
encoding ([[the-chain-transported-morphogen-is-a-positional-encoding]]).

    FIELD_CHANNELS       10 -> 2
    FIELD_FEATURE_DIM    20 -> 4
    FIELD_NODE_EXTRA     20 -> 4

So a field-aware rule now carries **4** extra node features rather than 20, and
one mechanism rather than three. Two further simplifications fell out:

  * **No sources.** The field is zero until the rule secretes, which the old
    module could not express -- its termini were clamped to one whatever the
    emit head did, so a rule with a dead head still received a gradient in
    chain distance. `tests/test_morphogen.py` pins the zero.
  * **No alive mask required.** `initial_field` is no longer gated on `alive`
    in `rollout.initial_state`, because a secreted chemical has no sources to
    place. A packing rollout can carry it too.

**And it is not free.** On a growth-plus-identity rule the field costs
**12,482** parameters -- the emit head plus the wider input layer -- against
242,891 without it, so about 5 % of the model for a channel nothing has yet
shown a use for. That is the argument for `morphogen = false` staying the
default until an arm needs it, and it is a better argument at 4 node features
than it was at 20.

## Two choices inside it, stated because neither is forced

  * **The receiver's own neighbour list, so transport is not symmetric.** A kNN
    graph is not symmetric and this does not symmetrise it: residue i reads what
    the residues *it* perceives are carrying, which is what a cell sensing its
    surroundings does and is the same relation perception uses. Mass is
    therefore not conserved -- a sensing rule, not a continuity equation, and it
    was never conservative anyway because secretion is free.
  * **A masked mean, not a distance weighting.** The neighbour list already
    says "within the cutoff". Weighting by distance would put a coordinate
    gradient on the secretion head, which is a second pathway by which the rule
    could move atoms -- exactly the kind of extra mechanism this change removes.
    The spreading stays a fixed local operator, as Growing NCA's kernel is.

## Verified end to end

20 steps on 1UBQ, `birth="free"`, one seed, 30 residues alive at the start:

    alive at end              33
    carrying a concentration  33
    carrying but not alive     0
    finite                   yes
    concentration range      0.90 - 0.92

The range is what an untrained sigmoid emit head gives -- about 0.5 everywhere,
plus inflow, through `a + e - a*e`. So the wiring is live rather than a field
that stayed at zero and passed a weaker check. `tests/test_free_placement.py`
pins all four rows; its old assertion was *"one residue per hop from the run's
N-terminus, so twenty steps carries it twenty residues"*, which was the
positional encoding written down as a requirement.

## It has memory, which was the open question and is now closed

**Ulban, 15 Nov: real diffusion.** The first spatial version recomputed each
residue's concentration from its neighbourhood every step, which is a relay --
one hop around a *persistent* source, and nothing carried through a region that
is not itself secreting. It is now the explicit Euler step of the diffusion
equation on a graph, with a degradation term:

    DIFFUSION_RATE  D       = 0.5    fraction of the gap to the mean closed per step
    DEGRADATION     lambda  = 0.10   fraction degraded per step

Two constants instead of one, which is the price. What they buy is that a pulse
**travels**: one secretion at step 0, nothing after, and on a 40-residue chain
the concentration twelve hops out is 8.5e-06 at step 20 and 3.6e-05 at step 40 --
a front moving through residues that are not secreting. The relay's reading at
every one of those residues was exactly zero, and
`test_a_pulse_travels_because_the_field_has_memory` is the assertion.

Degradation is not decoration. Diffusion alone conserves, so a persistent source
fills a finite graph and saturates the channel to a constant; `1 - lambda` is
what makes the steady state a gradient with a finite reach. And the reach is
analytic on a chain -- the steady state decays by `r` per hop where
`1 / (1 - lambda) = (1 - D) + D(1/r + r)/2` -- which predicts **0.519494** at
these constants against **0.51949** measured, constant to five figures over ten
residues. The test asserts the formula rather than the number, so retuning
`lambda` keeps it checking the physics.

## lambda is set by the graph's diameter, and that is the finding

A kNN graph over a compact protein is **small-world**, which the first choice of
`lambda = 0.02` ran straight into. 1UBQ at `k = 16` and a 10 A cutoff has a mean
degree of **15.5** and a **diameter of three hops** -- every residue is within
three of every other, 19 A away. A reach much longer than three hops therefore
leaves a field that is nearly constant and carries nothing. Steady state around
one secreting residue, as the mean concentration of each hop shell:

    lambda    hop1     hop2     hop3    hop1/hop3   max/min
      0.02   0.3826   0.2883   0.2516      1.52       4.1
      0.10   0.1448   0.0590   0.0335      4.32      34.1
      0.30   0.0558   0.0102   0.0027     20.52      516

0.02 is flat; 0.30 decays to a one-hop indicator, which the rule's own message
passing already supplies and so is a channel that duplicates perception. 0.10
keeps a gradient across the whole molecule and still delivers 0.03 to the far
side, well above the log feature's 1e-6 floor. **So the one free constant in
this module is fixed by the perception graph's topology rather than by taste**,
and on a larger protein, whose diameter is larger, the same `lambda` reaches
proportionally less far -- which is the right direction and is not yet measured
on anything but ubiquitin.

## Mass is not conserved, and on a real graph the total briefly rises

Transport is a gather, so a residue that appears in fifteen neighbour lists is
read fifteen times and never written. On 1UBQ a single pulse's **total**
concentration rises from 1.116 to 1.124 before it falls. That is not a bug and
it is not hidden: what is bounded whatever the degree is each residue's own
concentration, because `c + D(mean - c)` is a convex combination of two numbers
in `[0, 1]`; the sum is not a quantity anything reads. A conservative diffusion
would need the graph symmetrised, which would stop it being the receiver's rule.

**What a travelling field still cannot do** is cross a gap that has no edge.
Two fronts that have drifted apart are not neighbours, so nothing diffuses
between them whatever the memory. That is the bootstrap problem the old chain
transport existed for, it is unsolved, and it is the right kind of unsolved: a
field over the geometry as it is, becoming right as the structure does.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
