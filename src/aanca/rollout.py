"""Unrolling the rule, and the training objective built on top of it.

This is where the NCA machinery lives, and where the fixed-point objective is
actually expressed. Four pieces, each of which `docs/PLAN_V1.md` §3 argues for:

**Asynchronous updates.** Each residue updates with probability ~0.5 per step.
This does *not* cost the fixed point, provided something makes zero the output
at the native structure: any subset of zeros is still zero, so the mask
randomises the path and not the attractor. That premise used to be credited to
the no-op term, wrongly -- the term is evaluated at step 0, where the hidden
state is empty and the rule outputs ~0.04 deg whatever it is shown, so it
cannot distinguish a native structure from a wrecked one (scripts/noop_probe.py).
What carries the premise is ``--native-probability``: rollouts started from
native, scored by the same reconstruction term at the same random late step.
Asynchrony also removes the shared clock -- a synchronous update is a weak
global channel, since every cell implicitly knows it is step *t* at the same
instant -- which strengthens the locality claim rather than merely helping
convergence.

**Loss at a random late step, not the last.** This is the mechanism that
produces an attractor rather than a schedule. A rule supervised only at step N
learns a trajectory of length N; a rule supervised at an unpredictable late step
has to be correct *and stay* correct.

**Truncated BPTT.** Measured, not optional: at ~124 MB of retained activations
per step for a 300-residue target, a 30-step rollout is 3.7 GB per sample. The
graph is detached every ``truncate`` steps, so memory is bounded by the window
rather than the rollout length.

**A sample pool.** Final states are persisted and used as future starting
points, so the rule is trained from partially-converged states it produced
itself and has to stay stable rather than merely arrive once.

Diagnostics come out of the rollout for free: mean ``|mu|`` per step is the
tier-2 convergence curve.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from . import residue_constants as rc
from .blind import blind_coords
from .losses import chi_nll, harmonic_restraint, masked_mean, noop_nll
from .identity import (
    LIBRARY_SIZE,
    identity_loss,
    library_index,
    straight_through,
    type_marginal,
)
from .model import AANCARule
from .update_schedule import coupling_graph, random_independent_set
from .steric import StericTopology, clash_loss, steric_topology
from .torch_geometry import (
    TorsionTopology,
    apply_internal_deltas,
    measure_angles,
    measure_bonds,
    measure_chis,
    wrap_to_pi,
)
from .torch_graph import FeatureTopology, build_graph


@dataclass(frozen=True)
class RolloutConfig:
    steps: int = 30
    update_probability: float = 0.5
    #: ``"bernoulli"`` -- each residue ticks independently with
    #: ``update_probability``, which is what v1 through v6 trained with and
    #: what this still defaults to, so an existing run reproduces exactly.
    #:
    #: ``"independent"`` -- a fresh random maximal independent set of the
    #: coupling graph ticks, so no two residues that could touch ever update
    #: from the same stale neighbourhood. That is chromatic Gibbs, and it is
    #: the schedule an identity decoder needs; see ``aanca.update_schedule``
    #: for why
    #: it is drawn per step rather than from a colouring fixed once.
    #:
    #: Measured on native 1UBQ and 193L: at ``coupling_radius`` 5 A the sets
    #: hold 17 % of the protein and a full sweep takes 8-10 colours, against
    #: 20-24 colours if the perception radius were used as the coupling graph.
    #: The Bernoulli mask at 0.5 ticks about three times more residues than
    #: independence allows, which is the quantity the oscillation worry is
    #: about.
    update_schedule: str = "bernoulli"
    #: Minimum heavy-atom distance within which two residues may not tick
    #: together. Contact, not perception -- see ``aanca.update_schedule``.
    coupling_radius: float = 5.0
    truncate: int = 12
    k: int = 64
    cutoff: float = 10.0
    feature_dtype: torch.dtype = torch.float32
    #: Fraction of the rollout the loss step is drawn from, measured from the
    #: end. 0.5 means "the back half", which is what the plan specifies.
    loss_window: float = 0.5
    bond_stiffness: float = 100.0
    angle_stiffness: float = 10.0
    #: Weight on the steric term in ``aanca.steric``. 0 reproduces every run
    #: before 2026-11-02 exactly, and costs nothing when off -- the term is
    #: skipped rather than multiplied by zero, so neither the pair list nor the
    #: distances are built.
    #:
    #: It needs no interaction with ``bond_geometry``, which is worth recording
    #: because the comment in ``losses.harmonic_restraint`` reads as though it
    #: does. The cheat that comment warns about -- relieving a clash by
    #: stretching a bond, which is cheaper than finding the right rotamer -- is
    #: only available if the rule can drive bonds, and the restraint that
    #: forbids it is already wired to fire exactly when ``bond_geometry`` is on.
    #: With bonds frozen, as in v1, the side chain is rigid and the only way to
    #: lower this term is to rotate a chi, which is the point.
    clash_weight: float = 0.0
    #: Angstroms of overlap allowed before the term charges anything. See
    #: ``aanca.steric.DEFAULT_TOLERANCE`` for why slack is correct.
    clash_tolerance: float = 1.5
    #: Inverse folding. ``blind`` truncates perception to backbone-and-CB (see
    #: ``aanca.blind``, which closes the two leaks that are not features), and
    #: the rule's identity head then has nothing to read but the backbone and
    #: its neighbours' *currently committed* types. Off reproduces every run
    #: before 2026-11-02 exactly.
    blind: bool = False
    #: How a committed label is drawn from the head's logits each step.
    #: ``"sample"`` draws from the head's softmax, ``"argmax"`` takes the mode.
    #: Both commit a hard label and pass a straight-through gradient. Named for
    #: what it does rather than "gumbel", because a Gumbel-argmax draw *is* a
    #: draw from the softmax and its temperature does not sharpen the choice --
    #: see ``aanca.identity``. ``identity_temperature`` is the one that does.
    identity_sampling: str = "sample"
    #: Divides the logits before sampling. 1.0 samples from the head's own
    #: distribution; towards 0 it quenches to the mode.
    identity_temperature: float = 1.0
    #: What the labels start as. ``"masked"`` starts every residue at the
    #: all-zero one-hot, which is an unknown token and leaves the keep-current
    #: term contributing nothing, so step 1 is a draw from the head's prior.
    #: ``"native"`` starts from the true sequence, which is the fixed-point
    #: case: the rule must *stay*.
    identity_start: str = "masked"
    #: Select neighbours on the device instead of in NumPy on the host.
    #:
    #: **Not a scientific setting, and deliberately not pinned.** The two
    #: selections differ on about one graph in twenty, always on a pair within
    #: a float ulp of the cutoff -- and the envelope is exactly zero there, so
    #: the rule's output is bit-identical either way. Measured over 60 real
    #: training crops: 3 disagreements, worst ``|delta mu|`` 0.0 degrees
    #: (``scripts/check_neighbours.py``). What it changes is the clock: the
    #: NumPy path is 80% of a rollout step on a card, 29.7 ms against 5.6.
    device_neighbours: bool = False


@dataclass
class RolloutState:
    """Everything carried between steps. ``coords`` is float64, hidden float32."""

    coords: Tensor
    hidden: Tensor
    step: int = 0
    #: ``(R, 54)`` straight-through one-hot over the (type, chi1 well) library:
    #: the identity each residue is currently committing to. ``None`` when the
    #: identity head is off, which is the whole v1 path.
    #:
    #: It is *soft* on the backward pass on purpose. The label committed here is
    #: an input at the next step, so this tensor is the only route by which a
    #: later step's loss reaches the head that chose the label -- which is what
    #: makes a ripple trainable rather than merely observable.
    identity: Tensor | None = None
    #: The logits the last step produced, kept so the loss can see them without
    #: a second forward pass.
    identity_logits: Tensor | None = None

    def detached(self) -> "RolloutState":
        return RolloutState(
            coords=self.coords.detach(),
            hidden=self.hidden.detach(),
            step=self.step,
            identity=None if self.identity is None else self.identity.detach(),
            identity_logits=(
                None if self.identity_logits is None else self.identity_logits.detach()
            ),
        )


@dataclass
class RolloutResult:
    state: RolloutState
    #: Mean |mu| over real chis at each step, in radians. The tier-2 convergence
    #: curve: does it decay, or oscillate as some Graph NCA rules do?
    mean_abs_delta: list[float] = field(default_factory=list)
    #: Mean kappa at each step, as a confidence trace.
    mean_kappa: list[float] = field(default_factory=list)
    #: ``(mu, log_kappa, coords_seen)`` at the supervised step, if one was asked
    #: for. ``coords_seen`` is the rule's *input* at that step, which is what the
    #: target delta has to be measured from -- scoring against the final
    #: coordinates would ask the rule for a delta it was never in a position to
    #: predict.
    supervised: tuple[Tensor, Tensor, Tensor] | None = None
    supervised_step: int | None = None


def initial_state(
    coords: Tensor,
    rule: AANCARule,
    dtype: torch.dtype = torch.float32,
    identity: Tensor | None = None,
) -> RolloutState:
    return RolloutState(
        coords=coords,
        hidden=rule.initial_hidden(coords.shape[0], coords.device, dtype),
        step=0,
        identity=identity,
    )


def initial_identity(
    topo: FeatureTopology,
    coords: Tensor | None = None,
    start: str = "masked",
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``(R, 54)`` one-hot over the library to begin a rollout from.

    - ``"masked"`` -- all zeros. Not a library entry at all but an *unknown*
      token: the keep-current term has nothing to hold, so step 1 is a draw
      from the head's own prior. This is the inverse-folding start.
    - ``"native"`` -- the true sequence and its true chi1 wells, which needs
      ``coords``. The fixed-point case: the rule must stay where it is, and a
      run that drifts off the native sequence has failed a test it should pass
      trivially.
    - ``"random"`` -- a uniform draw over all 54 entries. The analogue of the
      randomised chi start, and the hardest of the three.
    """
    n = topo.n_residues
    device = topo.atom_mask.device
    if start == "masked":
        return torch.zeros(n, LIBRARY_SIZE, device=device, dtype=dtype)
    if start == "native":
        if coords is None:
            raise ValueError("identity_start 'native' needs the native coordinates")
        chi1 = torch.rad2deg(measure_chis(coords, topo.torsion)[:, 0])
        index = library_index(topo.torsion.resnames, chi1)
    elif start == "random":
        draws = torch.randint(
            LIBRARY_SIZE, (n,), generator=generator, dtype=torch.long
        )
        index = draws.to(device)
    else:
        raise ValueError(
            f"unknown identity_start {start!r}; expected 'masked', 'native' "
            "or 'random'"
        )
    one_hot = torch.zeros(n, LIBRARY_SIZE, device=device, dtype=dtype)
    return one_hot.scatter_(-1, index.unsqueeze(-1), 1.0)


def update_mask(
    n_residues: int,
    probability: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Bernoulli mask over residues. ``probability >= 1`` gives a synchronous step."""
    if probability >= 1.0:
        return torch.ones(n_residues, dtype=torch.bool, device=device)
    # Drawn on the CPU and moved, for the same reason the damage is: the mask
    # sequence has to be a property of the seed, not of the device.
    draws = torch.rand(n_residues, generator=generator)
    return (draws < probability).to(device)


def step_once(
    rule: AANCARule,
    state: RolloutState,
    topo: FeatureTopology,
    config: RolloutConfig,
    mask: Tensor,
    frozen: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[RolloutState, Tensor, Tensor]:
    """One application of the rule. Returns ``(new_state, mu, log_kappa)``.

    The mask gates the hidden state as well as the coordinates. Gating only the
    coordinates would let a frozen residue still advance its own memory, which
    is not what asynchrony means -- the cell either ticks or it does not.

    ``frozen`` is a different thing and gates **geometry only**, which is the
    one place that distinction is deliberately broken. It exists for the clamp
    test (``scripts/clamp_test.py``): to ask whether the rule reads its
    neighbours' side-chain conformations, some residues are pinned at a known
    conformation while the rest solve around them. A pinned residue has to keep
    ticking, because ``AANCARule.forward`` builds every message from
    ``cat([node_features, hidden])`` of the *sender* -- so freezing its hidden
    state as well would both remove a channel and feed its neighbours an
    all-zero hidden vector at step 50, which never occurs in training. Pinning
    the coordinates alone leaves every input on-distribution except the one
    under test.
    """
    coords_seen = state.coords
    aa_onehot = None
    if state.identity is not None:
        # The ripple's one and only pathway: each residue perceives its
        # neighbours' *committed* types, not the native ones, and the gradient
        # of this step's loss runs back through the labels they committed last
        # step to the head that chose them.
        aa_onehot = type_marginal(state.identity)
        if config.blind:
            coords_seen = blind_coords(state.coords, topo)

    graph = build_graph(
        coords_seen.to(config.feature_dtype),
        topo,
        k=config.k,
        cutoff=config.cutoff,
        device_neighbours=config.device_neighbours,
        aa_onehot=aa_onehot,
        blind=config.blind,
    )
    node = graph.node_features.to(torch.float32)

    mu, log_kappa, new_hidden = rule(
        node,
        state.hidden,
        graph.edge_features.to(torch.float32),
        graph.neighbours,
        graph.edge_mask,
        graph.envelope.to(torch.float32),
    )

    gate = mask.unsqueeze(-1)
    hidden = torch.where(gate, new_hidden, state.hidden)

    # The cell ticks under ``mask``; its atoms move under ``mask & ~frozen``.
    moving = mask if frozen is None else (mask & ~frozen)
    move = moving.unsqueeze(-1)

    chi = torch.where(move, mu, torch.zeros_like(mu)).to(state.coords.dtype)
    bond, angle = rule.internal_deltas(node, hidden)
    if bond is not None:
        bond = torch.where(move, bond, torch.zeros_like(bond)).to(state.coords.dtype)
        angle = torch.where(move, angle, torch.zeros_like(angle)).to(state.coords.dtype)

    coords = apply_internal_deltas(
        state.coords, topo.torsion, chi=chi, angle=angle, bond=bond
    )

    identity, logits = state.identity, None
    if state.identity is not None:
        logits = rule.identity_logits(node, hidden, state.identity)
        if logits is None:
            raise ValueError(
                "the rollout carries an identity state but the rule has no "
                "identity head; build it with AANCARule(identity=True)"
            )
        if config.identity_sampling not in ("sample", "argmax"):
            raise ValueError(
                f"unknown identity_sampling {config.identity_sampling!r}; "
                "expected 'sample' or 'argmax'"
            )
        scaled = logits / max(config.identity_temperature, 1e-6)
        drawn = straight_through(
            scaled, generator=generator, sample=config.identity_sampling == "sample"
        )
        # Gated like the coordinates, for the same reason: a residue that is
        # not ticking has not decided anything, and a frozen one is being held
        # at a known label on purpose.
        identity = torch.where(move, drawn, state.identity)

    return (
        RolloutState(
            coords=coords,
            hidden=hidden,
            step=state.step + 1,
            identity=identity,
            identity_logits=logits,
        ),
        mu,
        log_kappa,
    )


def rollout(
    rule: AANCARule,
    state: RolloutState,
    topo: FeatureTopology,
    config: RolloutConfig,
    generator: torch.Generator | None = None,
    supervise_step: int | None = None,
    record: bool = True,
    frozen: Tensor | None = None,
) -> RolloutResult:
    """Unroll ``config.steps`` times, detaching every ``config.truncate`` steps.

    ``supervise_step`` captures ``(mu, log_kappa)`` at that step for the loss.
    The graph is never detached at or after that step, so the captured outputs
    keep a gradient path back at least to the preceding truncation boundary.

    ``frozen`` pins those residues' geometry for the whole rollout; see
    :func:`step_once`. It does not touch the mask stream, so a frozen and an
    unfrozen rollout from the same seed differ only in which atoms moved.
    """
    result = RolloutResult(state=state)
    chi_mask = topo.torsion.chi_mask

    for step in range(config.steps):
        if config.update_schedule == "independent":
            # Recomputed each step on purpose: the coupling graph is a function
            # of the coordinates, so a set drawn once stops being independent
            # as the side chains move. Drawing it fresh also means there is no
            # fixed partition for a cell to learn as a phase.
            mask = random_independent_set(
                coupling_graph(state.coords, topo, config.coupling_radius),
                generator,
            )
        elif config.update_schedule == "bernoulli":
            mask = update_mask(
                topo.n_residues,
                config.update_probability,
                state.coords.device,
                generator,
            )
        else:
            raise ValueError(
                f"unknown update_schedule {config.update_schedule!r}; "
                "expected 'bernoulli' or 'independent'"
            )
        seen = state.coords
        state, mu, log_kappa = step_once(
            rule, state, topo, config, mask, frozen, generator
        )

        if record:
            with torch.no_grad():
                result.mean_abs_delta.append(
                    float(masked_mean(wrap_to_pi(mu).abs(), chi_mask))
                )
                result.mean_kappa.append(
                    float(masked_mean(torch.exp(log_kappa), chi_mask))
                )

        if supervise_step is not None and step == supervise_step:
            result.supervised = (mu, log_kappa, seen)
            result.supervised_step = step

        boundary = (step + 1) % config.truncate == 0
        past_supervision = supervise_step is None or step < supervise_step
        if boundary and past_supervision and step + 1 < config.steps:
            state = state.detached()

    result.state = state
    return result


def sample_loss_step(
    config: RolloutConfig, rng: random.Random | None = None
) -> int:
    """Draw the supervised step uniformly from the back ``loss_window`` fraction.

    Supervising the *last* step teaches a trajectory of that length. Supervising
    an unpredictable late step is what forces a fixed point.
    """
    rng = rng or random
    first = int(config.steps * (1.0 - config.loss_window))
    return rng.randrange(max(first, 0), config.steps)


@dataclass
class ObjectiveTerms:
    total: Tensor
    reconstruction: Tensor
    noop: Tensor
    restraint: Tensor
    #: Mean angstroms of van der Waals overlap per supervised atom, *before*
    #: ``clash_weight``. Reported unweighted on purpose: it is a physical
    #: quantity that can be compared against a crystal structure's own value,
    #: which is zero, and multiplying it by the weight would destroy that.
    clash: Tensor
    #: Mean negative log likelihood of the native (type, chi1 well) label at
    #: the final state, before ``identity_weight``. Zero when the head is off.
    identity: Tensor
    supervised_step: int


def training_objective(
    rule: AANCARule,
    damaged: Tensor,
    native_coords: Tensor,
    topo: FeatureTopology,
    config: RolloutConfig,
    generator: torch.Generator | None = None,
    rng: random.Random | None = None,
    noop_weight: float = 1.0,
    steric: StericTopology | None = None,
    identity_weight: float = 0.0,
) -> tuple[ObjectiveTerms, RolloutResult]:
    """The v1 objective: reconstruct from damage, do nothing when already native.

    The reconstruction term asks, at a random late step, how surprised the rule
    was by the delta that would actually have taken it to the native torsions.
    The no-op term runs *one* step on the undamaged structure and requires a
    zero update with high confidence.
    """
    torsion: TorsionTopology = topo.torsion
    native_chis = measure_chis(native_coords, torsion)

    # Every loss term is restricted to residues with complete perception. On an
    # intact structure that is all of them; on a spatial crop the boundary shell
    # is excluded, because its native torsions depend on contacts that were
    # cropped away and asking for them is unlearnable rather than merely hard.
    # The shell stays in the graph as context for the interior.
    supervised = topo.supervised_mask
    if not bool(supervised.any()):
        raise ValueError(
            "no residue has complete perception, so there is nothing to "
            "supervise -- the crop is smaller than one interaction cutoff"
        )
    chi_loss_mask = torsion.chi_mask & supervised.unsqueeze(-1)

    supervise_step = sample_loss_step(config, rng)
    result = rollout(
        rule,
        initial_state(
            damaged,
            rule,
            identity=(
                initial_identity(
                    topo, native_coords, config.identity_start, generator
                )
                if identity_weight
                else None
            ),
        ),
        topo,
        config,
        generator=generator,
        supervise_step=supervise_step,
    )
    assert result.supervised is not None
    mu, log_kappa, seen_coords = result.supervised

    # The delta that would have been correct from the coordinates the rule saw
    # at that step -- not from the final ones.
    target = wrap_to_pi(native_chis - measure_chis(seen_coords, torsion))
    reconstruction = chi_nll(
        mu, log_kappa, target, chi_loss_mask, torsion.periodic
    )

    # **Skipped entirely at weight 0**, which is now the recommended setting:
    # the term is evaluated at step 0, where the hidden state is all zeros and
    # the rule proposes ~0.04 deg whatever it is shown, so it cannot tell a
    # native structure from one 56 deg away (scripts/noop_probe.py). The
    # fixed point is supervised by starting rollouts from native instead --
    # ``--native-probability``. Skipping costs nothing in the random stream,
    # since this step draws no mask, so a run is unaffected either way.
    if noop_weight:
        native_state = initial_state(native_coords, rule)
        native_mask = torch.ones(
            topo.n_residues, dtype=torch.bool, device=native_coords.device
        )
        _after, native_mu, native_log_kappa = step_once(
            rule, native_state, topo, config, native_mask
        )
        noop = noop_nll(
            native_mu, native_log_kappa, chi_loss_mask, torsion.periodic
        )
    else:
        noop = torch.zeros((), dtype=reconstruction.dtype, device=mu.device)

    restraint = torch.zeros((), dtype=reconstruction.dtype, device=mu.device)
    if rule.bond_geometry:
        restraint = harmonic_restraint(
            measure_bonds(result.state.coords, torsion),
            measure_bonds(native_coords, torsion),
            torsion.bond_mask & supervised.unsqueeze(-1),
            config.bond_stiffness,
        ) + harmonic_restraint(
            measure_angles(result.state.coords, torsion),
            measure_angles(native_coords, torsion),
            torsion.angle_mask & supervised.unsqueeze(-1),
            config.angle_stiffness,
        )

    # Scored on the final state, like the restraint above and for the same
    # reason: the rollout's answer is where it ends up, and charging an
    # intermediate state for overlap would penalise passing *through* a clash
    # on the way to resolving one, which is a legitimate trajectory.
    clash = torch.zeros((), dtype=reconstruction.dtype, device=mu.device)
    if config.clash_weight:
        if steric is None:
            steric = steric_topology(torsion, native_coords)
        clash = clash_loss(
            result.state.coords,
            torsion,
            steric,
            supervised=supervised,
            tolerance=config.clash_tolerance,
        )

    # Inverse folding. Supervised at the *final* state only, which is the
    # fixed-point claim in its discrete form: whatever order the labels were
    # decided in and however they rippled, the sequence the rollout settles on
    # has to be the native one. Supervising every step instead would reward the
    # rule for guessing the answer before its neighbourhood could have told it,
    # which is the prefix trap one representation up.
    identity = torch.zeros((), dtype=reconstruction.dtype, device=mu.device)
    if identity_weight:
        logits = result.state.identity_logits
        if logits is None:
            raise ValueError(
                "identity_weight is set but the rollout produced no identity "
                "logits -- build the rule with AANCARule(identity=True)"
            )
        native_chi1 = torch.rad2deg(native_chis[:, 0])
        target = library_index(torsion.resnames, native_chi1)
        identity = identity_loss(logits, target, supervised)

    total = (
        reconstruction
        + noop_weight * noop
        + restraint
        + config.clash_weight * clash
        + identity_weight * identity
    )
    return (
        ObjectiveTerms(
            total=total,
            reconstruction=reconstruction.detach(),
            noop=noop.detach(),
            restraint=restraint.detach(),
            clash=clash.detach(),
            identity=identity.detach(),
            supervised_step=supervise_step,
        ),
        result,
    )


class SamplePool:
    """Growing-NCA sample pool: persist intermediate states as future seeds.

    Training only from freshly damaged structures teaches the rule to *arrive*.
    Re-seeding from states it produced itself, which are partially converged and
    somewhat off-distribution, is what teaches it to *stay* -- and is the cheap
    half of the fixed-point objective.
    """

    def __init__(self, capacity: int = 256, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self.capacity = capacity
        #: ``(structure_id, crop_seed, state)``. The crop seed is part of the
        #: key, not decoration: see :meth:`sample`.
        self._entries: list[tuple[int, int | None, RolloutState]] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._entries)

    def insert(
        self, structure_id: int, state: RolloutState, crop_seed: int | None = None
    ) -> bool:
        """Store a detached copy, evicting at random once full.

        **A non-finite state is refused**, and that is the whole reason this
        returns a bool. The pool is the one place in a run where a single bad
        update stops being transient: a NaN stored here is drawn as a *seed*
        by some later update, every rollout from it is NaN, and the run dies
        at a neighbour search ("data must be finite") an unknown number of
        updates after the thing that actually went wrong. Keeping the
        invariant in the container rather than at the call site means it holds
        for every caller, and it costs one reduction over the coordinates.

        Refusing is not the same as repairing. The caller is told so it can
        count and report; a pool that refuses often is a run in trouble, and
        that should be visible rather than quietly papered over.
        """
        if not (torch.isfinite(state.coords).all() and torch.isfinite(state.hidden).all()):
            return False
        entry = (structure_id, crop_seed, state.detached())
        if len(self._entries) < self.capacity:
            self._entries.append(entry)
        else:
            self._entries[self._rng.randrange(self.capacity)] = entry
        return True

    def sample(self, structure_id: int) -> tuple[int | None, RolloutState] | None:
        """``(crop_seed, state)`` for this structure, or ``None`` for fresh damage.

        **The crop seed is returned, not just the state, and the caller must
        rebuild the sample from it.** Keying by structure alone is not enough
        and was wrong: successive samples of a large entry are deliberately
        *different regions* of it, so a state stored under the bare structure
        id pairs coordinates for one set of residues with a topology and a
        native target for another. The shapes agree -- both crops are
        ``crop_size`` residues -- so nothing catches it, and the chi atom
        indices then address atoms that are not bonded. Measured on the
        validation split, 52% of entries are larger than the 384-residue crop,
        so this was half the pool re-seeds.

        The seed is enough because a crop is a deterministic function of it;
        storing it costs an integer, where storing the topology would put
        hundreds of megabytes into every checkpoint.
        """
        matching = [
            (seed, state) for sid, seed, state in self._entries if sid == structure_id
        ]
        if not matching:
            return None
        return self._rng.choice(matching)

    def clear(self) -> None:
        self._entries.clear()

    def state_dict(self) -> dict:
        """Everything needed to resume, including this pool's own RNG.

        The pool is part of the training state, not a cache: a resumed run that
        started with an empty pool would train for a while on fresh damage
        only, which is exactly the half of the objective the pool exists to
        supply. Tensors go in as they are and are moved on load.
        """
        return {
            "capacity": self.capacity,
            "rng": self._rng.getstate(),
            "entries": [
                {
                    "structure_id": sid,
                    "crop_seed": seed,
                    "coords": state.coords,
                    "hidden": state.hidden,
                    "step": state.step,
                }
                for sid, seed, state in self._entries
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"pool capacity changed: checkpoint has {state['capacity']}, "
                f"this pool has {self.capacity}"
            )
        self._rng.setstate(_tuple_states(state["rng"]))
        # A checkpoint written before the crop seed was stored cannot say
        # which crop each state belongs to, and "no seed" is indistinguishable
        # from "the entry was taken whole". Those states are exactly the ones
        # that caused the mismatch, so they are dropped rather than guessed
        # at; the pool refills within a few hundred updates.
        self._entries = [
            (
                int(entry["structure_id"]),
                (None if entry["crop_seed"] is None else int(entry["crop_seed"])),
                RolloutState(
                    coords=entry["coords"],
                    hidden=entry["hidden"],
                    step=int(entry["step"]),
                ),
            )
            for entry in state["entries"]
            if "crop_seed" in entry
        ]


def _tuple_states(state):
    """``random.setstate`` demands tuples; a round trip through torch.save
    leaves the inner state as a list."""
    if isinstance(state, (list, tuple)):
        return tuple(_tuple_states(item) for item in state)
    return state
