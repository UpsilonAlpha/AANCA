"""A chemical the rule secretes, diffusing over the perception graph.

This is the Growing-NCA mechanism, kept and nothing else. In Growing NCA a
cell's channels beyond the visible ones are hidden chemicals that spread by a
fixed local kernel over the grid, and what each cell secretes is learned while
the spreading is not. The grid is what a protein does not have
([[growing-nca-grid-is-the-missing-information]] in the claim map), so the
neighbourhood the chemical spreads over is the **spatial graph the rule already
perceives** -- within the cutoff, selected from the model's own coordinates,
rebuilt every step as the structure moves.

Four properties, each a constraint rather than a nicety:

* **Local.** A residue reads its own value and its current spatial neighbours'.
  Nothing sees the molecule as a whole and no step costs more than the
  perception already does, because it reuses the same neighbour list.
* **Not a coordinate.** There is no source at a terminus, no decay indexed by
  chain distance, and nothing a residue could integrate into "how far along the
  chain am I". The field is zero everywhere until something secretes, so it
  carries only what the rule puts in it.
* **Emergent.** The rule chooses what to secrete; the spreading is fixed. So a
  cell can manufacture a gradient nobody specified, and the chain can develop
  coordinates of its own -- or not, and then this costs four node features.
* **A concentration, so it has memory.** The step is
  ``c <- (1 - lambda) * (c + D * (mean - c)) + secretion``: a residue retains
  what it is carrying and exchanges a fraction of it with its neighbourhood,
  which is the explicit Euler step of the diffusion equation on a graph. The
  consequence is that a pulse *travels* -- a residue that secretes once sends a
  front outward over many steps -- where a rule that recomputed the
  concentration from the neighbourhood each step could only hold a steady state
  around a *persistent* source and could carry nothing through a region that was
  not itself secreting. Ulban, 15 Nov, asked for the diffusion rather than the
  relay.

**What was here before, and why it went.** Two groups transported along the
covalent chain, sourced at the termini of each alive run and decaying per
residue. Under ``birth="none"`` -- what every arm since g9 runs -- nothing is
dormant, every bond conducts, and those groups converge to ``decay ** index`` to
1.2e-07: a deterministic function of the residue index, which is the positional
encoding the old module's own ``_live_bonds`` docstring named as the thing to
avoid. Ulban, 14 Nov: *"they shouldn't transport along the chain -- the residue
graph is spatial."* Measured and recorded in
``docs/claims/the-chain-transported-morphogen-is-a-positional-encoding.md``.
``morphogen = false`` in every arm ever run, so there was no result to
reproduce and the module was replaced rather than deprecated.

The bootstrap problem those groups existed for is real and is not solved here:
two fronts that have drifted apart have no edge, so no chemical reaches across
either. That is the right kind of wrong -- the field is over the geometry as it
currently is, and becomes right as the structure does -- where a chain field was
robust to the geometry being wrong precisely because it carried no geometry.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Channels the rule secretes into. Two, so a cell can make a gradient and a
#: second signal that is not a function of the first.
EMIT_CHANNELS = 2
#: ``D``: the fraction of the gap to the neighbourhood mean that is closed in
#: one step. This is the explicit Euler step of the diffusion equation on a
#: graph -- ``c + D * (mean - c)``, a discrete Laplacian relaxation -- and it is
#: below one so that a residue keeps most of what it is carrying. That retained
#: term is the memory, and it is the whole difference from what was here before.
DIFFUSION_RATE = 0.5
#: ``lambda``: the fraction degraded per step. Diffusion alone conserves, so a
#: persistent source would fill the graph and the field would saturate to a
#: constant, which carries no information. Degradation is what makes the steady
#: state a *gradient* with a finite reach, and the reach is what this constant
#: sets: on a chain the steady state decays by ``r`` per hop, where
#: ``1 / (1 - lambda) = (1 - D) + D * (1 / r + r) / 2``.
#:
#: **Set from the graph's diameter rather than from taste**, which is the one
#: place the perception graph's topology enters this module. A kNN graph over a
#: compact protein is small-world: 1UBQ at ``k=16`` and a 10 A cutoff has a mean
#: degree of 15.5 and a diameter of **three hops**, so every residue is within
#: three of every other. A reach much longer than that leaves a field that is
#: nearly constant and says nothing. Measured there, as the mean concentration
#: of the one-, two- and three-hop shells around a single secreting residue:
#:
#:     lambda    hop1     hop2     hop3    hop1/hop3
#:       0.02   0.3826   0.2883   0.2516      1.52
#:       0.10   0.1448   0.0590   0.0335      4.32
#:       0.30   0.0558   0.0102   0.0027     20.52
#:
#: 0.02 is flat and 0.30 decays to a one-hop indicator, which the rule's own
#: message passing already supplies. 0.10 keeps a gradient across the molecule
#: and still delivers 0.03 to the far side, well above the log feature's floor.
DEGRADATION = 0.10

FIELD_CHANNELS = EMIT_CHANNELS
#: Raw amplitude and log-amplitude per channel. The log is kept because a
#: concentration that has decayed over several hops is small, and the rule has
#: more resolution in the log than in the raw value near zero.
FIELD_FEATURE_DIM = 2 * FIELD_CHANNELS

#: Floor inside the logarithm, and the scale the log is divided by so the
#: feature lands in roughly ``[-1, 0]``. An unreached residue reads ``-1``
#: rather than minus infinity.
_LOG_FLOOR = 1e-6
_LOG_SCALE = -float(torch.log(torch.tensor(_LOG_FLOOR)))


def initial_field(n_residues: int, device=None, dtype=torch.float32) -> Tensor:
    """``(R, FIELD_CHANNELS)`` of zeros.

    Zero and not source-clamped, because there are no sources: nothing is
    secreting until the rule does. A rollout whose emit head never fires
    carries a field of zeros from start to finish, which is the correct null --
    the old module could not express it, since its termini were clamped to one
    whatever the rule did.
    """
    return torch.zeros(n_residues, FIELD_CHANNELS, device=device, dtype=dtype)


def diffuse(
    field: Tensor,
    neighbours: Tensor,
    edge_mask: Tensor,
    emission: Tensor | None = None,
    alive: Tensor | None = None,
) -> Tensor:
    """One diffusion step over the perception graph. Returns ``(R, C)``.

    ``neighbours`` ``(R, k)`` and ``edge_mask`` ``(R, k)`` are the graph's own,
    so the chemical spreads over exactly the pairs the rule can perceive and
    nothing has to agree with anything. ``emission`` is ``(R, EMIT_CHANNELS)``
    in ``[0, 1]`` -- what each residue secretes this step -- or ``None`` for no
    secretion. ``alive`` zeroes dormant slots; ``None`` means every slot is
    alive, which is the ``birth="none"`` case and needs no mask.

    **The receiver's own neighbour list, so transport is not symmetric.** A
    kNN graph is not symmetric and this does not symmetrise it: residue i reads
    the concentration of the residues *it* perceives, which is what a cell
    sensing its surroundings does and is the same relation perception itself
    uses. The consequence to be aware of is that mass is not conserved -- this
    is a sensing rule, not a continuity equation, and it was never conservative
    anyway because secretion is free.

    **It has memory, and that costs a second constant.** The concentration is
    retained and relaxed toward the neighbourhood mean rather than recomputed
    from it, so a front propagates and the field is a state of the structure
    rather than an instantaneous readout of it. Diffusion alone conserves,
    though, so a persistent source would fill the graph and saturate the channel
    to a constant: ``DEGRADATION`` is what keeps the steady state a gradient
    with a finite reach. Two fixed constants rather than one, which is the price
    of a travelling signal.

    **A masked mean, not a distance weighting.** The neighbour list already
    says "within the cutoff", and weighting by distance would put a coordinate
    gradient on the secretion head -- a second pathway by which the rule could
    move atoms, which is exactly the kind of extra mechanism this module just
    lost. The spreading stays a fixed local operator, as Growing NCA's kernel
    is.

    ``emission`` carries gradient: every residue the chemical reaches reads it
    as a node feature, so a loss on their geometry trains the head that
    secreted.
    """
    if field.dim() != 2 or field.shape[1] != FIELD_CHANNELS:
        raise ValueError(
            f"field must be (R, {FIELD_CHANNELS}), got {tuple(field.shape)}"
        )
    n_res = field.shape[0]
    if neighbours.shape[:1] != (n_res,) or edge_mask.shape != neighbours.shape:
        raise ValueError(
            f"neighbours and edge_mask must both be (R, k) with R={n_res}, got "
            f"{tuple(neighbours.shape)} and {tuple(edge_mask.shape)}"
        )

    mask = edge_mask.unsqueeze(-1).to(field.dtype)
    # ``field[neighbours]`` is (R, k, C): each residue's view of what its
    # neighbours are carrying. Gathered rather than scattered, so a residue
    # that appears in many neighbour lists is read many times and never
    # written -- which is what makes this the receiver's rule.
    neighbour_field = field[neighbours] * mask
    count = mask.sum(dim=1).clamp(min=1.0)
    mean = neighbour_field.sum(dim=1) / count

    # ``c + D * (mean - c)`` is ``(1 - D) * c + D * mean``: a convex
    # combination, so the channel cannot leave [0, 1] whatever the graph does,
    # and the ``c`` term is the memory. A residue keeps what it holds and
    # exchanges a fraction of it with its neighbourhood, rather than
    # recomputing its concentration from the neighbourhood from scratch -- so a
    # pulse travels, and a residue that stops secreting does not instantly
    # forget. Degradation then bounds it: without the ``1 - lambda`` a
    # persistent source saturates the graph to a constant.
    arrived = (1.0 - DEGRADATION) * (field + DIFFUSION_RATE * (mean - field))

    if emission is None:
        new_field = arrived
    else:
        if emission.shape != (n_res, EMIT_CHANNELS):
            raise ValueError(
                f"emission must be (R, {EMIT_CHANNELS}), got "
                f"{tuple(emission.shape)}"
            )
        # ``a + e - a*e`` rather than ``(a + e).clamp(max=1)``. Both keep the
        # channel in ``[0, 1]``, but a clamp has exactly zero gradient once it
        # binds, so a residue secreting near its maximum would train with no
        # signal at all -- the state a saturating sigmoid head arrives at. This
        # is smooth everywhere, monotone in both arguments, and saturates only
        # in the limit.
        secreted = emission.to(field.dtype)
        new_field = arrived + secreted - arrived * secreted

    if alive is not None:
        new_field = new_field * alive.unsqueeze(-1).to(field.dtype)
    return new_field


def field_features(field: Tensor) -> Tensor:
    """``(R, FIELD_FEATURE_DIM)``: each channel's amplitude and its logarithm.

    Both, because they have resolution in different places: the amplitudes of
    1.0 and 0.99 differ by 0.01 on a range the rule can read, while their logs
    differ by 0.01 of the scale; several hops out the amplitudes are both near
    zero and only the logs separate them.
    """
    amplitude = field.clamp(min=0.0, max=1.0)
    logged = torch.log(amplitude + _LOG_FLOOR) / _LOG_SCALE
    return torch.cat([amplitude, logged], dim=-1)
