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

    field_new = EMIT_DECAY * mean over perceived neighbours  +  secretion

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

## The open question, and it is a real one

**The field has no memory.** Each step recomputes the concentration from the
neighbourhood rather than accumulating, so a residue that secretes once produces
one hop and then nothing -- the chemical is a steady state around *persistent*
sources, not a travelling wave. The old module had the same structure, so this
is not a regression, but it is also not diffusion.

Checked rather than assumed: the perception graph carries **no self-edge** --
`((neighbours == row) & edge_mask).any()` is `False` on 1UBQ at k=16 -- so a
residue reads none of its own concentration back and the no-memory property
holds on the real graph and not only in the tests' hand-built one.

Real diffusion is `c_new = c + D * (mean - c)`, which keeps memory, and would
need a degradation term to stay bounded: `c_new = (1 - lambda) * (c + D * (mean
- c)) + secretion`. That is three fixed constants instead of one and is closer
to a physical concentration field, which is the stated direction -- *"we want a
simple AANCA model that can hopefully learn real physics"*. Whether a pulse
should be able to travel is the choice, and it is Ulban's: a relay reaches a
steady state in one hop and cannot carry a signal through a region that is not
secreting, while a diffusing field can.

Deliberately not changed in the same commit as the topology, so the one variable
is the topology. `morphogen = false` in every arm, so nothing is waiting on it.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
