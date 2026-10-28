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

#: Dimensions of the NumPy featuriser's output, from ``graph.build_graph``.
NODE_FEATURE_DIM = 32
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
    ) -> None:
        super().__init__()
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

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
