"""Which residues tick on a given step, and why a colouring is not a partition.

The Bernoulli mask in :func:`rollout.update_mask` updates each residue
independently with probability ``update_probability``, so half the protein moves
at once. For a *reconstruction* rule that is fine and is what v1 through v6 were
trained with. For local updates read as conditional distributions -- Gibbs
sampling, which is what an identity decoder would be -- it is not: Gibbs
requires that the cells updated together be conditionally independent given the
rest, and two residues in contact are the opposite of that. Updating both from
the same stale neighbourhood is synchronous mean field, and strongly coupled
neighbours under synchronous updates oscillate rather than converge. Chromatic
Gibbs is the standard fix: partition the interaction graph into colour classes
and update one class at a time.

The obvious implementation does not work here, and the reason is worth keeping.
A fixed colouring has to be proper for the graph that message passing actually
uses, and ``torch_graph`` selects neighbours by **minimum heavy-atom distance**
(``torch_graph.py:380-395``) -- which moves when a side chain moves. So a
colouring computed once stops being proper during the rollout. Colouring a
conservative superset instead is worse: a side-chain atom reaches about 6.6 A
from its own CA, so "could ever be within the 10 A cutoff" means CA-CA within
23 A, which on a 128-residue crop is every pair. The superset graph is complete
and wants 128 colours, one residue per step.

What this module does instead is draw a fresh random **maximal independent
set** from the current graph on every step. That is better than a fixed
colouring on all three axes that matter:

* it is exactly proper, because it is computed from the geometry in hand rather
  than from the geometry at step 0;
* it is freshly random every step, so there is no fixed partition and no phase
  for a cell to learn -- a deterministic schedule is a clock, and a hidden state
  that persists across steps can count; and
* it needs no colour bookkeeping: one maximal independent set *is* one colour
  class, and drawing a new one each step is recolouring and taking class zero.

The coupling graph is deliberately **not** the message-passing graph. Gibbs
independence is a property of the energy, not of the perception radius: two
residues 9 A apart exchange messages but their side chains cannot touch, so
they are conditionally independent for packing and may tick together. Coupling
at ``COUPLING_RADIUS`` of minimum heavy-atom distance is that distinction, and
it is what keeps the independent sets from collapsing to a handful of residues.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .torch_graph import FeatureTopology, sequence_links

#: Minimum heavy-atom distance, in angstroms, within which two residues are
#: treated as coupled and so may not tick together.
#:
#: Two heavy atoms are in van der Waals contact at about 3.6 A at most (two
#: sulfurs), so 5 A is contact plus a margin for the fact that the set is drawn
#: before the step rather than after it, and a residue that moves during the
#: step could come into contact with one that was just outside. It is
#: deliberately far below the 10 A perception cutoff: coupling is about whether
#: side chains can touch, not about whether messages are exchanged.
COUPLING_RADIUS = 5.0


def coupling_graph(
    coords: Tensor,
    topo: FeatureTopology,
    radius: float = COUPLING_RADIUS,
) -> Tensor:
    """``(n_res, n_res)`` bool: which residues are too close to tick together.

    Minimum heavy-atom distance, the same measure ``torch_graph`` selects
    neighbours by, so the two notions of "in contact" cannot drift apart.
    Sequence neighbours are always coupled whatever the distance says, because
    they share backbone atoms and one's CB sits inside the other's reach.
    """
    n_res = topo.n_residues
    device = coords.device
    atoms = coords.detach()[topo.atom_mask]
    # ``owner`` is NumPy, not a tensor -- it is the index the NumPy graph path
    # scatters by, and it stays on the host.
    owner = torch.as_tensor(topo.owner, dtype=torch.long, device=device)
    n_atoms = atoms.shape[0]
    distances = torch.cdist(atoms, atoms)

    by_residue = torch.full(
        (n_res, n_atoms), float("inf"), dtype=distances.dtype, device=device
    ).scatter_reduce(0, owner.unsqueeze(-1).expand(-1, n_atoms), distances, "amin")
    pair = torch.full(
        (n_res, n_res), float("inf"), dtype=distances.dtype, device=device
    ).scatter_reduce(1, owner.unsqueeze(0).expand(n_res, -1), by_residue, "amin")
    pair.fill_diagonal_(float("inf"))

    coupled = pair <= radius
    if n_res > 1:
        links = sequence_links(topo, device)
        index = torch.arange(n_res - 1, device=device)
        coupled[index, index + 1] |= links
        coupled[index + 1, index] |= links
    return coupled


def random_independent_set(
    coupled: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """A maximal independent set of ``coupled``, from a random vertex order.

    Greedy: walk a random permutation and take every residue not already
    excluded by one taken earlier. Maximal rather than maximum -- the maximum
    independent set is NP-hard and buying it would gain nothing, since any
    independent set is valid and the randomness is the point.

    The permutation is drawn on the CPU and the result moved, for the same
    reason the damage and the Bernoulli mask are: the schedule has to be a
    property of the seed and not of the device.
    """
    n_res = coupled.shape[0]
    order = torch.randperm(n_res, generator=generator)
    adjacency = coupled.detach().cpu()
    chosen = torch.zeros(n_res, dtype=torch.bool)
    blocked = torch.zeros(n_res, dtype=torch.bool)
    for vertex in order.tolist():
        if not bool(blocked[vertex]):
            chosen[vertex] = True
            blocked |= adjacency[vertex]
    return chosen.to(coupled.device)


def random_colouring(
    coupled: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """A proper colouring of ``coupled`` from a random vertex order.

    Not used by the rollout -- :func:`random_independent_set` is -- but it is
    what answers "how many steps would one sweep take", which is the number
    that decides whether an exact schedule is affordable at all. Random greedy,
    the randomised Welsh-Powell: each residue takes the lowest colour none of
    its already-coloured neighbours holds.
    """
    n_res = coupled.shape[0]
    adjacency = coupled.detach().cpu()
    colour = torch.full((n_res,), -1, dtype=torch.long)
    for vertex in torch.randperm(n_res, generator=generator).tolist():
        taken = {
            int(c) for c in colour[adjacency[vertex]].tolist() if c >= 0
        }
        candidate = 0
        while candidate in taken:
            candidate += 1
        colour[vertex] = candidate
    return colour.to(coupled.device)


def is_independent(chosen: Tensor, coupled: Tensor) -> bool:
    """Whether no two selected residues are coupled. For tests and probes."""
    selected = chosen.nonzero(as_tuple=True)[0]
    if selected.numel() < 2:
        return True
    return not bool(coupled[selected][:, selected].any())
