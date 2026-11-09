"""Fields that travel along the chain, not through space.

The deficit this exists for is recorded in docs/JOURNAL.md under "Rethink": every
edge a growth rollout perceives is selected from ``coords.detach()`` within a
10 A cutoff, so two residues that ought to touch and currently do not have no
edge, no message and no gradient path. The rotational lever arm drifts two
independently grown fronts apart before they come within the cutoff, so they
never acquire the edge that would correct them.

A field propagating along the **covalent chain** is a communication channel
that does not depend on the spatial graph being right. Two residues forty
Angstrom apart in a wrong structure have no spatial edge, but they are still
connected through the backbone, and a quantity transported along that path
reaches one from the other regardless of where the model currently thinks
either of them is.

Three properties were required of it, and each is a constraint on the design
rather than a nicety:

* **Local.** A residue reads its own field value and its immediate covalent
  neighbours'. Nothing here sees the chain as a whole, and no step costs more
  than O(R).
* **Length-free.** The sources are the termini of each *alive* covalent run,
  and a terminus is detectable from the absence of a live bond on one side. No
  residue is told the length of the protein or its index in it. A normalised
  sequence index would supply the same gradient and would be neither -- it is
  a global coordinate handed to every cell, and it is exactly what this
  deliberately is not.
* **Emergent where it can be.** The two transported groups below are fixed
  physics -- a conserved quantity emitted at a terminus and decaying as it
  moves. The emitted group is not: the rule chooses what to secrete, into a
  symmetrically diffusing channel, so a cell can manufacture a gradient of its
  own and the chain can develop coordinates nobody specified.

The transported groups use *directed* transport rather than symmetric
diffusion, which is worth stating because "morphogen" usually implies the
latter. Directed transport advances one residue per step and converges to
``decay ** (i - start)`` exactly, so its logarithm is linear in chain distance
from the terminus -- a positional coordinate, at a resolution set by the decay.
Symmetric diffusion from a single source spreads as the square root of time and
would still be propagating when the rollout ends. The emitted group, which has
sources anywhere, is symmetric.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Decay per residue for the group sourced at each alive run's N-terminus and
#: transported C-ward. Four scales so the field resolves position both near a
#: terminus and far from it: ``0.99`` is still 0.28 at a hundred residues while
#: ``0.60`` has fallen to 0.01 by residue nine, and the logarithm of each is a
#: linear coordinate over the range where it has not underflowed.
NWARD_DECAYS: tuple[float, ...] = (0.99, 0.95, 0.85, 0.60)
#: The mirror group: sourced at each run's C-terminus, transported N-ward. Both
#: directions because which end a residue is near is not recoverable from one.
CWARD_DECAYS: tuple[float, ...] = (0.99, 0.95, 0.85, 0.60)
#: Channels the rule secretes into. Symmetric diffusion, so a secreting cell
#: raises its neighbourhood in both directions.
EMIT_CHANNELS = 2
#: Decay per hop for the emitted channels. Not a tuple: the rule can modulate
#: the amplitude it secretes but not the length scale, which keeps the emitted
#: group one mechanism rather than four.
EMIT_DECAY = 0.90

FIELD_CHANNELS = len(NWARD_DECAYS) + len(CWARD_DECAYS) + EMIT_CHANNELS
#: Raw amplitude and log-amplitude per channel. The log is what carries
#: position: ``log(decay ** d) = -d * |log decay|`` is linear in chain distance,
#: while the raw amplitude saturates at one end and underflows at the other.
FIELD_FEATURE_DIM = 2 * FIELD_CHANNELS

#: Floor inside the logarithm, and the scale the log is divided by so the
#: feature lands in roughly ``[-1, 0]``. A dormant or unreached residue reads
#: ``-1`` rather than minus infinity.
_LOG_FLOOR = 1e-6
_LOG_SCALE = -float(torch.log(torch.tensor(_LOG_FLOOR)))


def initial_field(n_residues: int, device=None, dtype=torch.float32) -> Tensor:
    """``(R, FIELD_CHANNELS)`` of zeros.

    Zero rather than source-clamped: the sources depend on ``alive``, which the
    caller has and this does not, and :func:`diffuse` clamps them on its first
    call anyway. So the first step a rollout takes already sees the terminus
    markers, and nothing has to be seeded twice.
    """
    return torch.zeros(n_residues, FIELD_CHANNELS, device=device, dtype=dtype)


def _live_bonds(alive: Tensor, links: Tensor) -> Tensor:
    """``(R-1,)``: bonds whose *both* ends are alive, so they can conduct.

    A dormant slot does not conduct. That is the choice that makes the field a
    property of the structure grown so far rather than of the crop's sequence:
    if dormant slots conducted, the converged field would be a deterministic
    function of the residue index and the whole thing would be a positional
    encoding handed to the rule rather than a quantity the growing chain
    carries.
    """
    if links.numel() == 0:
        return links
    return links & alive[:-1] & alive[1:]


def diffuse(
    field: Tensor,
    alive: Tensor,
    links: Tensor,
    emission: Tensor | None = None,
) -> Tensor:
    """One hop of transport. Returns the new ``(R, FIELD_CHANNELS)`` field.

    ``emission`` is ``(R, EMIT_CHANNELS)`` in ``[0, 1]`` -- what each residue
    secretes this step -- or ``None`` for no secretion. It carries gradient:
    the emitted channels are read as node features by every residue the field
    reaches, so the head that secretes is trained by their losses, which is the
    whole point of a channel that does not need the spatial graph.

    A residue that is alive and has no conducting bond on the inflow side is a
    **source**, clamped to one. That is the same test as "is this the terminus
    of its alive run", written as the absence of inflow rather than as a
    separate predicate, so the two can never disagree.
    """
    bonds = _live_bonds(alive, links)
    live = alive.unsqueeze(-1).to(field.dtype)
    pad_bool = torch.zeros(1, dtype=torch.bool, device=field.device)
    out = []
    at = 0

    # Built by padding and concatenation rather than slice assignment. The
    # emitted channels carry gradient from the secretion head, and an in-place
    # write into a tensor an earlier line already wrote is the shape of bug
    # autograd reports a hundred lines away from its cause.
    for decays, nward in ((NWARD_DECAYS, True), (CWARD_DECAYS, False)):
        group = field[:, at : at + len(decays)]
        at += len(decays)
        decay = torch.tensor(decays, device=field.device, dtype=field.dtype)
        pad = torch.zeros_like(group[:1])
        if bonds.numel() == 0:
            inflow, conducts = torch.zeros_like(group), torch.zeros_like(alive)
        elif nward:
            mask = bonds.unsqueeze(-1).to(field.dtype)
            inflow = torch.cat([pad, group[:-1] * mask], dim=0)
            conducts = torch.cat([pad_bool, bonds], dim=0)
        else:
            mask = bonds.unsqueeze(-1).to(field.dtype)
            inflow = torch.cat([group[1:] * mask, pad], dim=0)
            conducts = torch.cat([bonds, pad_bool], dim=0)
        source = (alive & ~conducts).unsqueeze(-1).to(field.dtype)
        # ``maximum`` rather than ``where``: a source is clamped to one, and a
        # non-source keeps what flowed in. Written as a maximum so a residue
        # that is both -- a run of length one -- is one rather than zero.
        out.append(torch.maximum(decay * inflow, source) * live)

    emitted = field[:, at:]
    pad = torch.zeros_like(emitted[:1])
    if bonds.numel() == 0:
        neighbour_sum = torch.zeros_like(emitted)
        neighbour_count = torch.zeros_like(alive, dtype=field.dtype)
    else:
        mask = bonds.unsqueeze(-1).to(field.dtype)
        from_nward = torch.cat([pad, emitted[:-1] * mask], dim=0)
        from_cward = torch.cat([emitted[1:] * mask, pad], dim=0)
        neighbour_sum = from_nward + from_cward
        count = bonds.to(field.dtype)
        zero = torch.zeros(1, dtype=field.dtype, device=field.device)
        neighbour_count = torch.cat([zero, count]) + torch.cat([count, zero])
    averaged = neighbour_sum / neighbour_count.clamp(min=1.0).unsqueeze(-1)
    arrived = EMIT_DECAY * averaged
    if emission is None:
        new_emitted = arrived
    else:
        # ``a + e - a*e`` rather than ``(a + e).clamp(max=1)``. Both keep the
        # channel in ``[0, 1]``, but a clamp has exactly zero gradient once it
        # binds -- so a residue secreting near its maximum would train with no
        # signal at all, which is the state a saturating sigmoid head arrives
        # at. This is smooth everywhere, monotone in both arguments, and
        # saturates only in the limit.
        secreted = emission.to(field.dtype)
        new_emitted = arrived + secreted - arrived * secreted
    out.append(new_emitted * live)

    return torch.cat(out, dim=-1)


def field_features(field: Tensor) -> Tensor:
    """``(R, FIELD_FEATURE_DIM)``: each channel's amplitude and its logarithm.

    The log is the feature that matters and the raw amplitude is kept because
    it is the only one that distinguishes *at* a source from *near* one: the
    log of 1.0 and of 0.99 differ by 0.01 of the scale, while the amplitudes
    differ by the same 0.01 on a range where the rule has resolution.
    """
    amplitude = field.clamp(min=0.0, max=1.0)
    logged = torch.log(amplitude + _LOG_FLOOR) / _LOG_SCALE
    return torch.cat([amplitude, logged], dim=-1)
