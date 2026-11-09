"""The discrete (type, well) library, its straight-through selector, and the
losses and diagnostic that go with unfreezing amino-acid identity.

The architecture is already fixed-backbone, so inverse folding is this setting
with identity unfrozen -- see ``docs/JOURNAL.md``, "Inverse folding: the design
settled on 2026-11-02". Two decisions recorded there are what this module
exists to implement, and both are decisions *against* the differentiable thing:

**Identity is discrete in the forward pass.** A soft 20-way simplex is the
natural output of an NCA neuron and it is wrong here. The closest prior art,
SCMF (Koehl & Delarue 1994), reaches a fixed point by exactly our local field
updates and fails exactly this way: averaging a hard pairwise steric constraint
over a neighbour's *distribution* turns "forbidden with one of j's rotamers"
into "half-forbidden", and the converged field has minima that correspond to no
realisable structure. A cell that is 40 % TRP and 30 % GLY broadcasts a field
no residue exerts, and inter-type steric differences are far larger than the
inter-rotamer ones SCMF already fails at. So the forward value is a one-hot and
only the gradient is soft -- :func:`straight_through`, :func:`gumbel_softmax`.

**The output is one joint categorical over (type, well), not two heads.** That
is Rosetta's rotamer set, where one move is "position i becomes rotamer r of
type t". Joint, so type and conformation cannot disagree; and the atom-set
problem vanishes, because each entry is a complete side chain with known atoms
-- no union-of-atoms mask (which would name the residue and leak the answer)
and no truncation to CB.

**The well is chi1 only, not the full chi tuple.** This is a deliberate first
rung and it bounds what a head over this library can express: it can say "this
position is a trans leucine" but not which of LEU's chi2 sub-wells, so
everything past chi1 has to come from the continuous von Mises head that sits
below the selector (classification plus regression, as the design note has it).
The reason to start here is that the alternative is not a free choice of
granularity. The full chi tuple is ``3**num_chi`` per type, i.e. 81 for ARG and
LYS alone, so a complete enumeration is ~400 entries of which the great
majority are sterically impossible and essentially unpopulated in any real
training set -- the rare ones would be learned from a handful of examples each,
and the discrete head's job (pick a *type*, and the gross chi1 branch that
decides where the side chain points) does not need them. chi1 also carries the
rotamer signal that matters most: it is the one the diagnostics in
``aanca.rotamers`` are already built around, and the one whose error dominates
reported chi MAE. If the library is later deepened, it should be deepened
per-type (chi1 x chi2 for the aromatics and the branched residues, chi1 only
for the long flexible ones) rather than uniformly.

**PRO is a third chi-less type**, not just GLY and ALA. ``residue_constants``
finds PRO's CA-CB bond to lie in a ring and puts it in ``RIGID_RESIDUES``, so
``num_chi("PRO") == 0`` here and PRO gets a single trivial entry exactly as GLY
and ALA do. Dropping any of the three would misalign every index above it.

:func:`label_churn` is the convergence diagnostic the design note requires.
``mean_abs_delta`` decaying stops meaning anything once a discrete label can
flip between NCA steps, because the continuous part can be settling while the
labels churn; the replacement is the fraction of positions whose label changed,
which must decay to zero. A plateau at a nonzero value is a limit cycle.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import residue_constants as rc
from .rng import draw_multinomial, draw_uniform
from . import rotamers

#: Entries of the library, in the canonical order: residue types in
#: :data:`aanca.residue_constants.AA3` order (which is sorted, so the
#: enumeration is deterministic and does not depend on dict insertion order),
#: and within a type the chi1 wells in :data:`aanca.rotamers.WELL_NAMES` order.
#: A chi-less type (GLY, ALA, PRO) contributes exactly one entry, with well 0.
LIBRARY: tuple[tuple[str, int], ...] = tuple(
    (resname, well)
    for resname in rc.AA3
    for well in range(len(rotamers.WELL_NAMES) if rc.num_chi(resname) else 1)
)

#: Number of (type, well) entries the joint head is a categorical over.
LIBRARY_SIZE = len(LIBRARY)

#: ``(resname, well) -> index``, the inverse of :data:`LIBRARY`.
LIBRARY_INDEX: dict[tuple[str, int], int] = {
    entry: i for i, entry in enumerate(LIBRARY)
}

#: Number of entries contributed by each residue type: 3 for a type with a
#: chi1, 1 for a chi-less one.
WELLS_PER_TYPE: dict[str, int] = {
    resname: (len(rotamers.WELL_NAMES) if rc.num_chi(resname) else 1)
    for resname in rc.AA3
}

#: ``(LIBRARY_SIZE,)`` long tensor: the index into :data:`AA3` of each entry's
#: type. A type marginal is ``zeros(20).scatter_add_(0, TYPE_INDEX, probs)``.
TYPE_INDEX: Tensor = torch.tensor(
    [rc.AA3.index(resname) for resname, _ in LIBRARY], dtype=torch.long
)

#: ``(LIBRARY_SIZE,)`` the central chi1 of each entry in degrees. A chi-less
#: entry reads zero, which is a placeholder and not an angle -- guard it with
#: ``num_chi``, the same way ``measure_chis`` output is guarded by ``chi_mask``.
WELL_CHI1_DEGREES: Tensor = torch.tensor(
    [
        rotamers.WELLS[rotamers.WELL_NAMES[well]] if rc.num_chi(resname) else 0.0
        for resname, well in LIBRARY
    ],
    dtype=torch.float32,
)

#: ``(20, len(WELL_NAMES))`` lookup used to vectorise :func:`library_index`;
#: entries for a well a type does not have hold the type's only entry, since a
#: chi-less residue's measured chi1 is meaningless and must not select by it.
_TYPE_WELL_TO_INDEX: Tensor = torch.tensor(
    [
        [
            LIBRARY_INDEX[(resname, well if rc.num_chi(resname) else 0)]
            for well in range(len(rotamers.WELL_NAMES))
        ]
        for resname in rc.AA3
    ],
    dtype=torch.long,
)

#: Number of residue types, i.e. the width of a type marginal.
N_TYPES = len(rc.AA3)


def entry(index: int) -> tuple[str, int]:
    """``(resname, well)`` of one library index."""
    return LIBRARY[index]


def index_of(resname: str, well: int) -> int:
    """Library index of one ``(resname, well)`` pair."""
    return LIBRARY_INDEX[(resname, well)]


def library_index(resnames: tuple[str, ...], chi1_degrees: Tensor) -> Tensor:
    """``(n_res,)`` library indices for real residues, from their measured chi1.

    ``chi1_degrees`` is ``(n_res,)`` in degrees; the well is assigned by
    :func:`aanca.rotamers.nearest_well`, so this and the chi1 diagnostics agree
    by construction rather than by two spellings happening to match. The chi1
    of a chi-less residue is ignored, so it may be anything, including the zero
    ``measure_chis`` writes into masked entries.
    """
    if chi1_degrees.ndim != 1 or chi1_degrees.shape[0] != len(resnames):
        raise ValueError(
            f"chi1_degrees {tuple(chi1_degrees.shape)} does not match "
            f"{len(resnames)} residues"
        )
    types = torch.tensor(
        [rc.AA3.index(name) for name in resnames],
        dtype=torch.long,
        device=chi1_degrees.device,
    )
    wells, _distance = rotamers.nearest_well(chi1_degrees)
    table = _TYPE_WELL_TO_INDEX.to(chi1_degrees.device)
    return table[types, wells]


def index_resnames(indices: Tensor) -> tuple[str, ...]:
    """The residue names of ``(n_res,)`` library indices.

    Strings, not a tensor, to match ``TorsionTopology.resnames``: this is the
    output a sequence-recovery score is computed against.
    """
    return tuple(LIBRARY[int(i)][0] for i in indices.reshape(-1))


def index_chi1_degrees(indices: Tensor) -> Tensor:
    """Central chi1 in degrees of ``(n_res,)`` library indices; 0 if chi-less."""
    return WELL_CHI1_DEGREES.to(indices.device)[indices]


def type_index(indices: Tensor) -> Tensor:
    """``AA3`` index of each of ``(n_res,)`` library indices."""
    return TYPE_INDEX.to(indices.device)[indices]


def type_marginal(probabilities: Tensor) -> Tensor:
    """``(..., N_TYPES)`` type probabilities, summing each type's wells.

    Summing is the right operation and not an approximation: the wells of one
    type are disjoint events whose union is "this position is that type".
    """
    flat = probabilities.reshape(-1, LIBRARY_SIZE)
    out = torch.zeros(
        flat.shape[0], N_TYPES, dtype=flat.dtype, device=flat.device
    )
    scatter = TYPE_INDEX.to(flat.device).expand(flat.shape[0], LIBRARY_SIZE)
    out.scatter_add_(1, scatter, flat)
    return out.reshape(*probabilities.shape[:-1], N_TYPES)


# --- the discretisation ------------------------------------------------------


def _gumbel_noise(
    logits: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Standard Gumbel noise shaped like ``logits``, reproducible from ``generator``."""
    # Drawn wherever the generator lives and moved, not drawn on the logits'
    # device: a CPU generator with ``device="cuda"`` raises, and this project
    # draws every random choice on the CPU on purpose. See ``aanca.rng``.
    uniform = draw_uniform(
        logits.shape,
        generator=generator,
        dtype=logits.dtype,
        device=logits.device,
    )
    # Clamped away from 0 so the double log is finite; 1 is never returned by
    # ``torch.rand``, so only the lower end needs guarding.
    return -torch.log(-torch.log(uniform.clamp_min(torch.finfo(logits.dtype).tiny)))


def _straight_through_from(one_hot: Tensor, soft: Tensor) -> Tensor:
    """``one_hot`` forward, ``soft``'s gradient backward."""
    return (one_hot - soft).detach() + soft


def straight_through(
    logits: Tensor,
    generator: torch.Generator | None = None,
    hard: bool = True,
    sample: bool = False,
) -> Tensor:
    """A one-hot over the library whose gradient is the plain softmax's.

    ``sample=False`` takes the argmax, ``sample=True`` draws from the softmax
    (reproducibly from ``generator``). ``hard=False`` returns the softmax
    itself, which is the soft simplex this module exists to avoid in the
    forward pass -- it is here only for ablations and for checking that the
    gradient path is the one it claims to be.

    There is no temperature: the backward value is the softmax at temperature
    1, so the gradient is exactly ``softmax(logits)``'s and nothing anneals.
    When a temperature is wanted, use :func:`gumbel_softmax`, whose estimator
    is biased but whose bias is what the temperature controls.
    """
    soft = torch.softmax(logits, dim=-1)
    if not hard:
        return soft
    if sample:
        # ``draw_multinomial`` and not ``torch.multinomial``: the bare call
        # requires the generator's device to match the probabilities', and a
        # CPU generator against GPU logits is what killed job 31516313 at
        # update 0. See ``aanca.rng``.
        choice = draw_multinomial(soft, generator)
    else:
        choice = soft.argmax(dim=-1)
    one_hot = torch.zeros_like(soft).scatter_(
        -1, choice.unsqueeze(-1), 1.0
    )
    return _straight_through_from(one_hot, soft)


def gumbel_softmax(
    logits: Tensor,
    temperature: float,
    generator: torch.Generator | None = None,
    hard: bool = False,
) -> Tensor:
    """Gumbel-softmax over the library, reproducible from ``generator``.

    ``hard=True`` discretises the result to the one-hot of its argmax and keeps
    the soft sample's gradient, which is the form the forward pass needs.

    On the temperature, and on what it does not do. It divides the *perturbed*
    logits, so ``argmax((logits + g) / temperature)`` is independent of the
    temperature: the label the forward pass commits to is a draw from
    ``softmax(logits)`` at every temperature, and all the temperature controls
    is how much of the returned vector's mass sits on that label. Annealing
    therefore sharpens the gradient estimator, not the choice -- reading a
    falling :func:`label_churn` as an effect of annealing would be reading the
    wrong mechanism.

    The two limits are the two failure modes and neither end is usable. As
    ``temperature -> 0`` the vector becomes a true one-hot and the gradient
    variance diverges (and below roughly 0.1 the ``exp`` underflows for
    well-separated logits in float32, so the gradient is numerically zero --
    anneal to a floor, not to zero). As ``temperature -> inf`` the vector tends
    to uniform, carrying no information about the logits at all: a smooth,
    low-variance gradient for a soft value that says nothing. ``temperature``
    must be strictly positive; around 1 is the usual starting point, annealed
    downward.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    soft = torch.softmax(
        (logits + _gumbel_noise(logits, generator)) / temperature, dim=-1
    )
    if not hard:
        return soft
    one_hot = torch.zeros_like(soft).scatter_(
        -1, soft.argmax(dim=-1, keepdim=True), 1.0
    )
    return _straight_through_from(one_hot, soft)


# --- the losses --------------------------------------------------------------


def identity_nll(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-residue cross-entropy over the library, ``(n_res,)``, zero where masked.

    Shaped like :func:`aanca.losses.von_mises_nll`: the per-element term with
    masked entries zeroed, so the caller chooses how to reduce it. ``logits``
    is ``(n_res, LIBRARY_SIZE)`` and ``target`` is ``(n_res,)`` library
    indices.
    """
    log_probabilities = torch.log_softmax(logits, dim=-1)
    nll = -log_probabilities.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, nll, torch.zeros_like(nll))


def identity_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Scalar mean cross-entropy over the library, over the masked residues."""
    from .losses import masked_mean

    return masked_mean(identity_nll(logits, target, mask), mask)


def type_nll(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-residue cross-entropy of the *type marginal*, ``(n_res,)``.

    The probability of the correct amino acid summed over its wells. This is a
    different number from :func:`identity_nll` and a smaller one, since summing
    over wells can only add probability to the correct type -- and it is the one
    that corresponds to sequence recovery, which is scored on type and not on
    (type, well). Both are worth logging: the gap between them is how much of
    the model's uncertainty is about conformation rather than identity.
    """
    log_probabilities = torch.log_softmax(logits, dim=-1)
    # Summed in log space rather than by exponentiating and scatter-adding: a
    # well whose probability underflows to zero in float32 would make the
    # type's log read -inf, and a correct type whose wells are all small is
    # exactly the case this loss is meant to score.
    membership = torch.nn.functional.one_hot(
        TYPE_INDEX.to(logits.device), num_classes=N_TYPES
    ).to(logits.dtype)
    masked = log_probabilities.unsqueeze(-1) + torch.log(membership)
    per_type = torch.logsumexp(masked, dim=-2)
    type_target = TYPE_INDEX.to(logits.device)[target]
    nll = -per_type.gather(-1, type_target.unsqueeze(-1)).squeeze(-1)
    return torch.where(mask, nll, torch.zeros_like(nll))


def type_loss(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Scalar mean type-marginal cross-entropy over the masked residues."""
    from .losses import masked_mean

    return masked_mean(type_nll(logits, target, mask), mask)


# --- the convergence diagnostic ----------------------------------------------


def label_churn(previous: Tensor, current: Tensor, mask: Tensor) -> Tensor:
    """Fraction of masked positions whose discrete label changed.

    The replacement for ``mean_abs_delta`` once a label can flip between steps:
    the continuous part can be converging while the labels churn, so a decaying
    magnitude no longer means the system is settling. This must decay to zero,
    and a plateau at a nonzero value is a limit cycle, not slow convergence.

    Zero when nothing is masked in, matching
    :func:`aanca.losses.masked_mean`'s convention -- "no positions changed" is
    the right reading of "no positions".
    """
    from .losses import masked_mean

    changed = (previous != current).to(torch.get_default_dtype())
    return masked_mean(changed, mask)
