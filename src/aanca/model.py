"""The rule: one message-passing hop, two heads, weights shared across steps.

Architecture decisions and their reasons are in `docs/PLAN_V1.md` §2. The two
that shape this file:

**One hop per step.** Depth comes from time, not from stacked layers. At a 10 A
all-heavy-atom cutoff the measured mean degree is 28-33, so one hop already
reaches a quarter to a third of a small protein and two reach all of it. Three
stacked layers per step would make a single application globally connected on
anything CASP-sized, and nothing in the loss curve would reveal it.

**Identity initialisation.** The final layer of the delta head is zeroed with a
bias that makes the predicted mean exactly zero, so an untrained rule is the
identity map. This is the Growing-NCA trick and it matters more here than there:
a rule that starts by moving atoms randomly, applied thirty times, diverges
before it learns anything.

No step index, no noise-level conditioning, no global pooling, no attention over
the whole protein. That absence is the scientific claim -- do not let a time
embedding creep in.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import residue_constants as rc
from .identity import LIBRARY_SIZE

#: Dimensions of the NumPy featuriser's output, from ``graph.build_graph``.
NODE_FEATURE_DIM = 32

#: Extra node dimensions a growth rule carries: the closure-separation block
#: from ``aanca.growth.closure_features``, which says how many residues away
#: the nearest alive slot is in each direction. Kept here rather than imported
#: from ``growth`` to avoid a cycle, and asserted equal to
#: ``growth.CLOSURE_DIM`` in ``tests/test_closure.py`` so the two cannot drift.
GROWTH_NODE_EXTRA = 24
EDGE_FEATURE_DIM = 575

#: Ceiling on ``log kappa``. A likelihood loss can be reduced by claiming
#: enormous confidence rather than by getting the mean right, so the head's
#: confidence is bounded. exp(6) ~ 403, whose circular standard deviation is
#: ~3 degrees -- already tighter than crystallographic precision warrants.
MAX_LOG_KAPPA = 6.0

#: Hard bounds on the bond and angle heads, when enabled. Chosen from the
#: physics rather than tuned: bond lengths vary by ~0.01-0.02 A and bond angles
#: by ~2-3 degrees at the energies side-chain packing cares about, so these caps
#: are a few times the real spread -- loose enough to be useful, tight enough
#: that the rule cannot resolve a clash by deforming chemistry.
MAX_BOND_DELTA = 0.05
MAX_ANGLE_DELTA = 0.15

#: Initial strength of the identity head's keep-current term. A zeroed head
#: alone would make the untrained rule emit *uniform* logits over the library,
#: which is the opposite of the identity initialisation above: applied thirty
#: times it would resample the whole sequence from noise before the rule has
#: learned anything. The neutral element for a discrete label is "keep what is
#: there", so the head starts as a copy-through: logit 8 on the current entry
#: against 53 others at 0 is exp(8)/(exp(8)+53) ~ 98 % on holding still. It is
#: a learnable parameter, so training can weaken it; the point is only where it
#: starts.
#:
#: **It is right for a native start and a liability for a masked one**, which
#: is why the constructor takes it. From the native sequence a strong keep is
#: exactly the stability wanted: the rule holds a sequence that is already
#: correct. From a masked start there is nothing to keep at step 1, so the
#: first label is a draw from the head's prior -- and from step 2 a keep of 8
#: pins 98 % of the probability on that guess, so the rule has to learn to let
#: go before it can learn to predict. Measured over 40 updates on 8 validation
#: structures the term fell 6.59 -> 6.47 against ln 54 = 3.99, i.e. it was
#: still unlearning the prior. Start a masked run nearer 2.
IDENTITY_KEEP_INIT = 8.0

#: Where an untrained rule puts a newborn, degrees. The delta head's identity
#: initialisation has no analogue in growth -- a newborn has to go *somewhere*,
#: and there is no "do nothing" option -- so the neutral choice is the most
#: populated region of the Ramachandran plot instead: an untrained rule grows
#: an alpha helix. That is a physically valid default rather than a random one,
#: which is the property the identity initialisation was actually buying.
BACKBONE_INIT_PHI = -63.0
BACKBONE_INIT_PSI = -43.0
BACKBONE_INIT_OMEGA = 180.0

#: How far a Cartesian birth may swing the newborn's CA direction, radians.
#:
#: Measured rather than chosen (``scripts/growth_probe.py`` and the numbers in
#: HANDOFF): the CA-CA distance of a trans peptide is **3.804 A for every
#: (phi, psi)** -- it is fixed by the peptide's own geometry, not by the
#: torsions -- so the only thing a birth chooses about *where* the child's CA
#: goes is a direction. Rotating the helical reference direction by 0.72 rad
#: reaches beta sheet, polyproline II and left-handed helix alike, so 1.0 rad
#: covers the Ramachandran plot with room to spare while still forbidding a
#: newborn from appearing behind its parent.
#:
#: The child's *orientation* is deliberately NOT bounded. A rotation is
#: intrinsically bounded, polyproline II sits 3.07 rad from helix -- nearly
#: antipodal -- and any cap tight enough to matter would exclude real
#: secondary structure.
MAX_BIRTH_DIRECTION = 1.0

#: Initial activation logit, so sigmoid is about 0.12 and growth starts slow.
#: Zero would wake every frontier residue of every segment on the first step,
#: before the rule has any hidden state to decide with.
ACTIVATION_INIT = -2.0


class AANCARule(nn.Module):
    """One update step. Apply repeatedly with the same weights.

    ``forward`` is deliberately stateless about the graph: it takes the edge
    tensors as arguments rather than owning a structure, so the caller is
    responsible for rebuilding the graph each step. That keeps the
    non-differentiable part (neighbour selection) outside the module.
    """

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 96,
        message_dim: int = 128,
        max_log_kappa: float = MAX_LOG_KAPPA,
        bond_geometry: bool = False,
        max_bond_delta: float = MAX_BOND_DELTA,
        max_angle_delta: float = MAX_ANGLE_DELTA,
        identity: bool = False,
        identity_keep: float = IDENTITY_KEEP_INIT,
        growth: bool = False,
        birth: str = "torsion",
        activation_init: float = ACTIVATION_INIT,
        max_rotation: float = 0.1,
        max_translation: float = 0.3,
        max_birth_direction: float = MAX_BIRTH_DIRECTION,
    ) -> None:
        super().__init__()
        # A growth rule perceives the closure-separation block as well, so its
        # node dimension is wider. Applied only when ``node_dim`` is the
        # default, so an explicit value still wins -- and the failure mode of
        # getting it wrong is a loud shape mismatch at the first forward pass,
        # not a silent one.
        if growth and node_dim == NODE_FEATURE_DIM:
            node_dim = NODE_FEATURE_DIM + GROWTH_NODE_EXTRA
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.max_log_kappa = max_log_kappa

        # Per-residue state is the observable part (recomputed from geometry
        # every step) concatenated with free hidden channels (carried).
        state_dim = node_dim + hidden_dim
        self.state_dim = state_dim

        self.edge_embed = nn.Linear(edge_dim, message_dim)
        self.message = nn.Sequential(
            nn.Linear(2 * state_dim + message_dim, message_dim),
            nn.GELU(),
            nn.Linear(message_dim, message_dim),
        )
        # Normalising the aggregate keeps the scale stable as degree varies
        # between 24 and 34 across structures. It is over feature channels
        # within one residue, so it introduces no cross-residue pathway.
        self.aggregate_norm = nn.LayerNorm(message_dim)

        # Gated residual update, GRU-style. The gate is what lets a residue hold
        # its state still, which a plain residual MLP has to learn to do.
        self.gate = nn.Linear(hidden_dim + message_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim + message_dim, hidden_dim)

        self.delta_head = nn.Sequential(
            nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, 2 * rc.MAX_CHI)
        )
        self.kappa_head = nn.Sequential(
            nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, rc.MAX_CHI)
        )

        # Bond and angle control. A classical forcefield has stretching and
        # bending terms, so a learned one needs the scope for them; these heads
        # exist so the degrees of freedom do not have to be retro-fitted later.
        #
        # Off in v1. When on, the outputs are hard-bounded here AND harmonically
        # restrained to the crystal values in the loss. That pairing is the
        # point: unrestrained bond geometry is a physically wrong way to relieve
        # a steric clash -- squash a bond instead of finding the right rotamer --
        # which lowers the clash term, raises chi MAE, and is invisible in
        # aggregate metrics.
        self.bond_geometry = bond_geometry
        self.max_bond_delta = max_bond_delta
        self.max_angle_delta = max_angle_delta
        if bond_geometry:
            self.bond_head = nn.Sequential(
                nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, rc.MAX_BONDS)
            )
            self.angle_head = nn.Sequential(
                nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, rc.MAX_ANGLES)
            )

        # Inverse folding. One head over the joint (amino-acid type, chi1 well)
        # library rather than separate type and rotamer heads: the two are
        # conditional on each other, a joint categorical cannot smear across
        # them the way a mean field does, and a committed entry has a definite
        # atom set, so the "which atoms does this residue have" problem never
        # arises. Off in v1 for the same reason as the bond heads -- the v1
        # path must not touch these tensors at all.
        self.identity = identity
        if identity:
            self.identity_head = nn.Sequential(
                nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, LIBRARY_SIZE)
            )
            self.identity_keep = nn.Parameter(torch.tensor(float(identity_keep)))

        # Growth. Three heads, and the split is the design: *whether* to wake a
        # dormant neighbour, *where* to put it if woken, and how the residue's
        # own frame should move while it relaxes. The third is why growth needs
        # heads at all beyond the second -- a backbone built from torsions
        # cannot be corrected from torsions, because the lever arm reaches
        # 29 A over 75 residues (see ``aanca.growth``).
        self.growth = growth
        self.birth = birth
        self.activation_init = activation_init
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.max_birth_direction = max_birth_direction
        if birth not in ("torsion", "cartesian"):
            raise ValueError(
                f"unknown birth {birth!r}; expected 'torsion' or 'cartesian'"
            )
        if growth:
            # Two logits: N-ward and C-ward. Separate rather than one, because
            # a segment's two ends are different situations and a residue in
            # the middle of a segment is on neither frontier.
            self.activation_head = nn.Sequential(
                nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, 2)
            )
            # Where a newborn goes, in one of two parameterisations.
            #
            # ``torsion``: three torsions per direction, each as a (cos, sin)
            # pair for the same reason the delta head uses one -- a raw angle
            # has a discontinuity in the middle of its range. The child's
            # backbone is then *constructed* by NeRF, which reads the child's
            # own measured bond geometry and so is not type-independent.
            #
            # ``cartesian``: six numbers per direction, two axis-angle vectors.
            # The first rotates the reference CA direction, the second composes
            # onto the reference orientation, both relative to the parent's own
            # frame -- so a birth is equivariant and no torsion, and nothing
            # about the child's amino-acid type, enters its placement. A
            # multiplicative residual on the rotation rather than an additive
            # one, because axis-angle is not a vector space: measured, the
            # additive difference between helix and sheet is 4.6 rad while the
            # geodesic between them is 1.86.
            width = 2 * 6 if birth == "cartesian" else 2 * 3 * 2
            self.birth_head = nn.Sequential(
                nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, width)
            )
            # Axis-angle rotation and translation of the residue's own atoms.
            self.frame_head = nn.Sequential(
                nn.Linear(state_dim, 64), nn.GELU(), nn.Linear(64, 6)
            )
        self._init_heads()

    def _init_heads(self, initial_log_kappa: float = 1.0) -> None:
        """Make the untrained rule the identity, with moderate confidence."""
        final_delta = self.delta_head[-1]
        nn.init.zeros_(final_delta.weight)
        with torch.no_grad():
            # (cos, sin) = (1, 0) per chi, so atan2(0, 1) = 0 exactly. Zeroing
            # the bias too would give atan2(0, 0), whose gradient is undefined.
            bias = torch.zeros(2 * rc.MAX_CHI)
            bias[0::2] = 1.0
            final_delta.bias.copy_(bias)

        final_kappa = self.kappa_head[-1]
        nn.init.zeros_(final_kappa.weight)
        with torch.no_grad():
            # Inverted through the tanh cap so the realised value is the one asked for.
            raw = torch.atanh(
                torch.tensor(initial_log_kappa / self.max_log_kappa)
            ) * self.max_log_kappa
            final_kappa.bias.fill_(float(raw))

        if self.bond_geometry:
            for head in (self.bond_head, self.angle_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

        if self.identity:
            # Zeroed, so at initialisation the only thing in the logits is the
            # keep-current term and the rule holds every label still.
            nn.init.zeros_(self.identity_head[-1].weight)
            nn.init.zeros_(self.identity_head[-1].bias)

        if self.growth:
            final = self.activation_head[-1]
            nn.init.zeros_(final.weight)
            with torch.no_grad():
                final.bias.fill_(float(self.activation_init))

            final = self.birth_head[-1]
            nn.init.zeros_(final.weight)
            if self.birth == "cartesian":
                # Zero deviation from the helical reference, which
                # ``growth.give_birth`` holds. So an untrained Cartesian rule
                # puts a newborn *exactly* where an untrained torsion rule
                # does, and the two modes differ in what is learnable rather
                # than in where they start -- which is the only way the probe's
                # comparison between them means anything.
                nn.init.zeros_(final.bias)
            else:
                # The torsion order matches ``backbone.extend_nward`` and
                # ``extend_cward``: N-ward takes (phi parent, omega child, psi
                # child) and C-ward (psi parent, omega parent, phi child).
                with torch.no_grad():
                    degrees = torch.tensor(
                        [
                            [BACKBONE_INIT_PHI, BACKBONE_INIT_OMEGA, BACKBONE_INIT_PSI],
                            [BACKBONE_INIT_PSI, BACKBONE_INIT_OMEGA, BACKBONE_INIT_PHI],
                        ]
                    )
                    radians = torch.deg2rad(degrees)
                    bias = torch.stack(
                        [torch.cos(radians), torch.sin(radians)], dim=-1
                    ).reshape(-1)
                    final.bias.copy_(bias)

            # Zeroed, so an untrained rule relaxes by not moving -- the one
            # place growth does have an identity initialisation.
            nn.init.zeros_(self.frame_head[-1].weight)
            nn.init.zeros_(self.frame_head[-1].bias)

    def initial_hidden(
        self, n_residues: int, device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        return torch.zeros(n_residues, self.hidden_dim, device=device, dtype=dtype)

    def forward(
        self,
        node_features: Tensor,
        hidden: Tensor,
        edge_features: Tensor,
        neighbours: Tensor,
        edge_mask: Tensor,
        envelope: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One step.

        Args:
            node_features: ``(R, node_dim)`` recomputed from current geometry.
            hidden: ``(R, hidden_dim)`` carried from the previous step.
            edge_features: ``(R, K, edge_dim)``.
            neighbours: ``(R, K)`` long, residue indices.
            edge_mask: ``(R, K)`` bool, real edges.
            envelope: ``(R, K)`` smooth cutoff weight.

        Returns:
            ``(mu, log_kappa, new_hidden)`` -- mean delta-chi in radians
            ``(R, MAX_CHI)``, bounded log concentration ``(R, MAX_CHI)``, and the
            next hidden state.
        """
        state = torch.cat([node_features, hidden], dim=-1)

        sender = state[neighbours]
        receiver = state.unsqueeze(1).expand(-1, neighbours.shape[1], -1)
        edges = self.edge_embed(edge_features)

        messages = self.message(torch.cat([receiver, sender, edges], dim=-1))

        # Gradients reach the geometry through this weight, not through the
        # choice of neighbour, which is a hard argsort.
        weight = (envelope * edge_mask).unsqueeze(-1).to(messages.dtype)
        aggregated = self.aggregate_norm((messages * weight).sum(dim=1))

        gate_in = torch.cat([hidden, aggregated], dim=-1)
        z = torch.sigmoid(self.gate(gate_in))
        candidate = torch.tanh(self.candidate(gate_in))
        new_hidden = (1.0 - z) * hidden + z * candidate

        new_state = torch.cat([node_features, new_hidden], dim=-1)

        raw_delta = self.delta_head(new_state).view(-1, rc.MAX_CHI, 2)
        mu = torch.atan2(raw_delta[..., 1], raw_delta[..., 0])

        raw_kappa = self.kappa_head(new_state)
        log_kappa = self.max_log_kappa * torch.tanh(raw_kappa / self.max_log_kappa)

        return mu, log_kappa, new_hidden

    def internal_deltas(
        self, node_features: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor] | tuple[None, None]:
        """Bond and angle deltas from an already-updated state, or ``(None, None)``.

        Separate from ``forward`` so the v1 path never touches these tensors:
        ``None`` means :func:`torch_geometry.apply_internal_deltas` skips the
        coordinate entirely rather than applying zeros.
        """
        if not self.bond_geometry:
            return None, None
        state = torch.cat([node_features, hidden], dim=-1)
        bond = self.max_bond_delta * torch.tanh(
            self.bond_head(state) / self.max_bond_delta
        )
        angle = self.max_angle_delta * torch.tanh(
            self.angle_head(state) / self.max_angle_delta
        )
        return bond, angle

    def identity_logits(
        self, node_features: Tensor, hidden: Tensor, current: Tensor | None = None
    ) -> Tensor | None:
        """``(R, LIBRARY_SIZE)`` logits over (type, chi1 well), or ``None``.

        Separate from ``forward`` for the same reason as ``internal_deltas``:
        every existing caller unpacks three values from ``forward`` and the v1
        path must not pay for a head it does not use.

        ``current`` is each residue's present library entry, and it enters as
        the keep-current term described at :data:`IDENTITY_KEEP_INIT`. Pass it
        as indices ``(R,)`` or as a one-hot / straight-through ``(R, 54)``; the
        soft form is the one to use inside a rollout, because it is how the
        gradient of a *later* step's loss reaches the head that chose the label
        at this step. Omitting it gives the raw head, which is what a one-shot
        prediction from a masked input wants.
        """
        if not self.identity:
            return None
        state = torch.cat([node_features, hidden], dim=-1)
        logits = self.identity_head(state)
        if current is None:
            return logits
        if current.dim() == 1:
            keep = torch.zeros_like(logits)
            keep.scatter_(-1, current.unsqueeze(-1), 1.0)
        else:
            keep = current.to(logits.dtype)
        return logits + self.identity_keep * keep

    def growth_outputs(
        self, node_features: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(activation, torsions, axis_angle, translation)``.

        - ``activation`` ``(R, 2)`` raw logits, N-ward then C-ward.
        - ``birth`` -- what the newborn's placement is made of, and its shape
          says which parameterisation. Under ``birth="torsion"`` it is
          ``(R, 2, 3)`` radians, in the order the two ``extend_`` functions
          take them. Under ``birth="cartesian"`` it is ``(R, 2, 6)``: an
          axis-angle rotating the reference CA direction, bounded by
          ``max_birth_direction``, followed by an unbounded axis-angle
          composing onto the reference orientation.
        - ``axis_angle`` ``(R, 3)`` and ``translation`` ``(R, 3)``, both
          hard-bounded here so one relaxation step stays local whatever the
          head asks for. Bounding by scaling the whole vector rather than each
          component keeps the rotation's *axis* untouched, which a per-component
          tanh would not.

        Raises if the rule was not built with ``growth=True``, rather than
        returning ``None`` like the other optional heads: there is no sensible
        no-growth value for a tuple of four, and a silent zero would look like
        a rule that had decided not to grow.
        """
        if not self.growth:
            raise ValueError(
                "this rule has no growth heads; build it with "
                "AANCARule(growth=True)"
            )
        state = torch.cat([node_features, hidden], dim=-1)
        activation = self.activation_head(state)

        if self.birth == "cartesian":
            raw = self.birth_head(state).view(-1, 2, 6)
            birth = torch.cat(
                [_bounded(raw[..., :3], self.max_birth_direction), raw[..., 3:]],
                dim=-1,
            )
        else:
            raw = self.birth_head(state).view(-1, 2, 3, 2)
            birth = torch.atan2(raw[..., 1], raw[..., 0])

        move = self.frame_head(state)
        axis_angle = _bounded(move[:, :3], self.max_rotation)
        translation = _bounded(move[:, 3:], self.max_translation)
        return activation, birth, axis_angle, translation

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _bounded(vector: Tensor, limit: float) -> Tensor:
    """Scale ``vector`` so its norm never exceeds ``limit``, keeping direction.

    ``tanh`` on the norm rather than a hard clamp, so the bound is smooth and
    the gradient does not vanish for a head that is asking for too much.

    **The branch at the origin is load-bearing, not defensive.** The frame head
    is zero-initialised, so ``vector`` is exactly zero on the first update and
    ``limit * tanh(0) / 0`` evaluates to ``0 / eps = 0`` -- which multiplies
    the whole output by zero and leaves the head with no gradient at all, for
    every update, forever. The factor's true limit as the norm goes to zero is
    1, so that is what it is told to be. Measured: without this the frame
    head's total gradient on a growth objective is exactly 0.000e+00, which is
    the same failure as zeroing the delta head's bias and getting
    ``atan2(0, 0)``.
    """
    norm = vector.norm(dim=-1, keepdim=True)
    safe = norm.clamp_min(1e-8)
    factor = torch.where(
        norm < 1e-6,
        torch.ones_like(norm),
        limit * torch.tanh(norm / limit) / safe,
    )
    return vector * factor
