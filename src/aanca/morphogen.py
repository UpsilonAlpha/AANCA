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
  coordinates of its own -- or not, and then this costs ten node features.
* **Readable as a direction, not only as an amount.** A residue perceives each
  channel's concentration *and* its gradient as a vector in the residue's own
  frame (:func:`field_gradient`), which is the other half of Growing NCA's
  perception vector: a cell there reads ``[identity, grad_x, grad_y]`` from
  fixed Sobel filters. Its translation is read in that same frame, so
  "move up the gradient" is one linear map from what it perceives. Measured:
  supplying the gradient takes a learned chemotaxis from cosine 0.71 to 0.96.
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

#: Channels the rule secretes into. **Four, and the number is the design
#: choice.** Two was chosen as a signed pair -- one channel positive, one
#: negative, their difference a signed potential -- which makes the diffused
#: field electrostatics and nothing else
#: (``docs/claims/a-residue-can-follow-the-morphogen-gradient.md``). Four says
#: instead that the field is *whatever the rule finds useful to secrete*: a
#: signed pair still fits inside it if that is what the rule wants, and so does
#: a hydrophobic signal, which has no sign and so cannot share the pair. This
#: is the Growing NCA position, where a cell's chemicals carry no assigned
#: meaning at all and the only thing fixed is how many there are.
#:
#: The cost is ``5 * EMIT_CHANNELS`` node features (two amplitude, three
#: gradient per channel) and about 3.5k parameters. Nothing here asserts the
#: rule will use all four; an unused channel costs its features and decays to
#: nothing, which is a cheaper failure than a channel the rule needed and
#: did not have.
#: Channels reserved for the **polarity source**, which no head predicts:
#: channel 0 is secreted by every residue whose N side has no covalent
#: neighbour and channel 1 by every residue whose C side has none. See
#: :func:`terminus_emission`. Two, because **two gradients fix an orientation
#: where one fixes only a bearing** -- a single source gives each residue a
#: direction to walk, and a frame needs two independent directions before
#: "which way am I turned" is a question the field can answer at all.
TERMINUS_CHANNELS = 2
#: Channels the emit head predicts. Four, which was Ulban's call on 15 Nov and
#: is preserved here: the polarity pair is an addition to that budget and not a
#: tax on it, so the rule still has four channels of its own to put a signed
#: charge and an unsigned hydrophobicity in.
LEARNED_EMIT_CHANNELS = 4
EMIT_CHANNELS = TERMINUS_CHANNELS + LEARNED_EMIT_CHANNELS
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
#: Each channel's spatial gradient, as a vector in the residue's own frame.
#: This is the half of the Growing-NCA perception that is NOT the concentration
#: itself: a cell there is handed ``[identity, grad_x, grad_y]``, the gradients
#: coming from fixed Sobel filters rather than from anything the cell learns.
#: Measured here, supplying it is worth a great deal -- see
#: :func:`field_gradient`.
FIELD_GRADIENT_DIM = 3 * FIELD_CHANNELS

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


def field_gradient(
    field: Tensor,
    neighbours: Tensor,
    direction: Tensor,
    weight: Tensor,
) -> Tensor:
    """``(R, FIELD_GRADIENT_DIM)``: each channel's gradient in the own frame.

    ``sum_j w_ij (c_j - c_i) u_ij``, where ``u_ij`` is the unit vector to
    neighbour j **in residue i's own N-CA-C frame** -- the ``local_direction``
    block the edge features already carry, so this costs one einsum and no new
    geometry. A vector in a local frame is invariant under a global rotation
    for the same reason that block is, and it is read by a head whose
    translation is interpreted in that same frame, so "move up the gradient" is
    one linear map away.

    **Why it is supplied rather than learned.** In Growing NCA a cell perceives
    ``[identity, grad_x, grad_y]``: the gradients come from fixed Sobel
    filters, not from anything the cell works out. The analogous question here
    has an answer, because the message function *can* infer a gradient -- it
    sees ``(c_j, u_ij)`` for every neighbour and the messages are summed, which
    is the right shape. Trained on a synthetic chemotaxis task on 1UBQ (one
    residue secretes, every other should emit a translation pointing at it; the
    structure is identical in every example, so the field is the only input
    that says where the source is), held-out cosine against the true direction,
    two seeds, mean over three unseen sources:

        update   concentration only   + this block
           100         +0.652            +0.947
           300         +0.715            +0.960
           600         +0.709            +0.960

    So it can be inferred -- 0.71 is far above the 0.00 of a rule that cannot
    read direction at all -- and inferring it is much worse than being told,
    for 3,456 parameters. Supplying the derivative is Growing NCA's choice and
    it carries over to a graph.
    ``docs/claims/a-residue-can-follow-the-morphogen-gradient.md``.

    ``weight`` is ``(R, k)``, the envelope times the edge mask, so a padded or
    dormant slot contributes nothing and a neighbour at the cutoff fades out
    rather than dropping. **Unnormalised**, as Growing NCA's Sobel is: dividing
    by the weight sum would make the block a mean difference per neighbour
    rather than a sum over the neighbourhood. Which is better is untested; the
    numbers above are the unnormalised form, and the density block already
    tells a rule its degree.
    """
    if field.dim() != 2 or field.shape[1] != FIELD_CHANNELS:
        raise ValueError(
            f"field must be (R, {FIELD_CHANNELS}), got {tuple(field.shape)}"
        )
    n_res, k = neighbours.shape
    # Checked because it broadcasts rather than failing: ``field[neighbours]``
    # is (R', k, C) and ``field.unsqueeze(1)`` is (R, 1, C), so a mismatch
    # produces an (R, k, C) difference between rows that are not the same
    # residue, which trains.
    if field.shape[0] != n_res:
        raise ValueError(
            f"field has {field.shape[0]} residues and neighbours has {n_res}"
        )
    if direction.shape != (n_res, k, 3):
        raise ValueError(
            f"direction must be (R, k, 3) = {(n_res, k, 3)}, got "
            f"{tuple(direction.shape)}"
        )
    if weight.shape != (n_res, k):
        raise ValueError(
            f"weight must be (R, k) = {(n_res, k)}, got {tuple(weight.shape)}"
        )
    difference = field[neighbours] - field.unsqueeze(1)
    gradient = torch.einsum(
        "rkc,rka,rk->rca",
        difference,
        direction.to(field.dtype),
        weight.to(field.dtype),
    )
    return gradient.reshape(n_res, FIELD_GRADIENT_DIM)


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


def terminus_emission(
    links: Tensor,
    n_residues: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``(R, TERMINUS_CHANNELS)``: 1.0 at each hanging covalent end.

    Channel 0 is secreted by every residue with **no covalent neighbour on its
    N side**, channel 1 by every residue with none on its C side. ``links`` is
    the ``(R - 1,)`` bool the rest of the code uses -- ``links[i]`` means
    residue ``i`` is bonded to ``i + 1`` -- so a run boundary is a ``False``
    and the two real chain ends are the first and last residue.

    **This is a given input, not a prediction, and that is the design
    decision.** It is a *boundary condition* rather than a potential: the
    precedent is a Growing NCA cell being handed its seed state, where the seed
    is given and the pattern is learned. Two things follow and both are the
    point. The field is **differential from step 0**, where a learned source
    starts flat and saturated -- an untrained sigmoid secretes about 0.5
    everywhere -- and has no gradient towards secreting differentially until
    the field is already informative. And loop closure acquires a **bearing**:
    ``growth.py`` deferred closure to the bond restraint, which cannot act
    until the two ends are already within reach of each other, so nothing in
    the project has ever told a growing end which way its partner lies.

    **Two opposing sources, because their ratio is the readout.** The French
    flag model is one coordinate along one body axis read differently by every
    cell, and the biology is a pair of opposing gradients whose ratio gives the
    position. So the quantity is ``log(c_nward) - log(c_cward)``, and the two
    gradients being anti-aligned -- measured at cos -0.80 on 1UBQ -- is the
    signal rather than a degeneracy.

    **It is a spatial coordinate and not a positional encoding, and that was
    checked rather than argued**, because a field with sources at the termini
    is what the retired chain-transported design had and that one was exactly
    ``decay ** index``
    (``docs/claims/the-chain-transported-morphogen-is-a-positional-encoding.md``).
    Against the residue index, and against the CA projected on the axis joining
    the two termini, the ratio correlates -0.92 to -1.00 with the **spatial**
    coordinate in every start state tried and -0.19 with the index on the
    native fold. The field reports where a residue is in the body, faithfully,
    whatever the geometry.

    What does vary is whether the body axis and the sequence axis are the same
    thing. On a folded structure they are not; on any torsion-walk start state
    they are, so there the field reads as an index because position *is* the
    index. **That degeneracy belongs to an unfolded chain, not to this
    function**, and no choice of baseline torsions removes it -- the random
    mode only moves the control from 1.00 to 0.87. Regeneration does remove it,
    by keeping the intact remainder folded.
    """
    if links.numel() != max(n_residues - 1, 0):
        raise ValueError(
            f"links must be (R - 1,) for R = {n_residues}, got "
            f"{tuple(links.shape)}"
        )
    if device is None:
        device = links.device
    emission = torch.zeros(n_residues, TERMINUS_CHANNELS, device=device, dtype=dtype)
    if n_residues == 0:
        return emission
    bonded = links.to(device=device, dtype=torch.bool)
    # No neighbour N-ward: residue 0 always, plus any residue whose link back
    # is absent. Mirror for C-ward.
    nward_free = torch.ones(n_residues, dtype=torch.bool, device=device)
    cward_free = torch.ones(n_residues, dtype=torch.bool, device=device)
    if n_residues > 1:
        nward_free[1:] = ~bonded
        cward_free[:-1] = ~bonded
    emission[nward_free, 0] = 1.0
    emission[cward_free, 1] = 1.0
    return emission


def compose_emission(learned: Tensor | None, terminus: Tensor | None) -> Tensor | None:
    """``(R, EMIT_CHANNELS)``: the polarity channels first, then the rule's.

    The channel order is fixed rather than configurable, so a field tensor
    means the same thing in every arm: ``[:TERMINUS_CHANNELS]`` is structural
    and the rest is secreted. Either side may be ``None``, which contributes
    zeros, and both ``None`` gives ``None`` so the caller can hand the result
    straight to :func:`diffuse`.
    """
    if learned is None and terminus is None:
        return None
    if learned is not None and learned.shape[1] != LEARNED_EMIT_CHANNELS:
        raise ValueError(
            f"learned emission must be (R, {LEARNED_EMIT_CHANNELS}), got "
            f"{tuple(learned.shape)}"
        )
    if terminus is not None and terminus.shape[1] != TERMINUS_CHANNELS:
        raise ValueError(
            f"terminus emission must be (R, {TERMINUS_CHANNELS}), got "
            f"{tuple(terminus.shape)}"
        )
    reference = learned if learned is not None else terminus
    n_res = reference.shape[0]
    if terminus is None:
        terminus = torch.zeros(
            n_res, TERMINUS_CHANNELS, device=reference.device, dtype=reference.dtype
        )
    if learned is None:
        learned = torch.zeros(
            n_res, LEARNED_EMIT_CHANNELS, device=reference.device, dtype=reference.dtype
        )
    if learned.shape[0] != terminus.shape[0]:
        raise ValueError(
            f"learned emission has {learned.shape[0]} residues and terminus "
            f"emission has {terminus.shape[0]}"
        )
    return torch.cat([terminus.to(learned.dtype), learned], dim=-1)
