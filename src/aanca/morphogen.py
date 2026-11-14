"""A chemical the rule secretes, diffusing over the perception graph.

This is the Growing-NCA mechanism, kept and nothing else. In Growing NCA a
cell's channels beyond the visible ones are hidden chemicals that spread by a
fixed local kernel over the grid, and what each cell secretes is learned while
the spreading is not. The grid is what a protein does not have
([[growing-nca-grid-is-the-missing-information]] in the claim map), so the
neighbourhood the chemical spreads over is the **spatial graph the rule already
perceives** -- within the cutoff, selected from the model's own coordinates,
rebuilt every step as the structure moves.

Three properties, each a constraint rather than a nicety:

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
#: Fraction of a neighbourhood's mean concentration that arrives in one hop.
#: Not a tuple, unlike the length scales the old transported groups carried: the
#: rule modulates the amplitude it secretes, not the distance the chemical
#: travels, which keeps this one mechanism rather than four.
EMIT_DECAY = 0.90

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
    """One hop of diffusion over the perception graph. Returns ``(R, C)``.

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
    arrived = EMIT_DECAY * (neighbour_field.sum(dim=1) / count)

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
