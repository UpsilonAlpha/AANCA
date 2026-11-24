"""Gala's E(n)-equivariant graph NCA, verbatim, on the regeneration task.

The published rule (Gala, Grattarola & Quaeghebeur, *E(n)-equivariant Graph
Neural Cellular Automata*, TMLR 2024, section 4.1), transcribed from
``docs/claims/an-equivariant-graph-nca-morphs-random-coordinates-into-a-target-point-cloud.md``::

    m_ij = phi_m(||x_i - x_j||^2, h_i, h_j, e_ij)
    m_i  = sum_{j in N(i)} m_ij
    x_i' = x_i + (1/|N(i)|) sum_{j in N(i)} (x_i - x_j) phi_x(m_ij)
    h_i' = phi_h(h_i, m_i)

Four lines, one hop per step, no step index. ``phi_x`` maps to a **scalar**, so
the coordinate update is a mean of neighbour-difference vectors with learned
scalar weights -- which is the whole of its equivariance: the weights are
functions of invariants only, and the vectors they scale rotate with the
structure.

**This module exists because the rule is not a setting of ``AANCARule``.**
``AANCARule`` emits an axis-angle and a translation per residue and
``rollout.step_once`` moves coordinates only through ``growth.rigid_relax``,
which is a frame update in each residue's own N-CA-C basis. Gala's rule has no
frames at all: no rotation convention, no ``frame_head``, no lab-versus-local
question, and nothing to drift. Narrowing ``AANCARule``'s two width keys bottoms
out at 17,552 parameters because its four heads are ``Linear(state_dim, 64)``
with the 64 a literal (``tests/test_network_size.py``), so the published ~5K is
not reachable from there at all. This is a separate, small, frameless module
rather than a configuration of one that cannot express it.

**What this rule therefore cannot answer, and it is the method's property
rather than an omission here.** It places CA positions and produces **no
residue orientation**. So it says nothing about:

  * backbone conformation -- phi/psi, the Ramachandran basins, ``twist`` (a
    bonded pair's relative rotation, which does not exist for a frameless
    rule);
  * peptide geometry -- the N and C atoms are never placed, so the bond and
    angle restraints have no output to charge;
  * side chains, packing, chis, steric overlap, or sequence identity;
  * FAPE, which is defined in per-residue frames.

A CA trace is what it produces and a CA trace is what may be read off it. The
reporting in :func:`egnca_objective` is restricted to exactly that, rather than
feeding a frame-shaped metric something meaningless and printing the result.

**The padded-coordinate carrier.** The project's loss, RMSD and neighbour
selection all take the ``(R, MAX_ATOMS, 3)`` tensor and index CA through
``topo.frame_idx[:, 1]``. :func:`with_ca_positions` writes this rule's CA back
into such a tensor by translating each residue's whole atom cloud so its CA
lands on the new position. The other atoms therefore carry the *native*
orientation, unchanged, as a carrier -- nothing in the objective or the report
reads them, and the one place they are read is named at
:func:`egnca_neighbours`.

**The rule is the authors' code's rule, not only the paper's four lines.** The
paper prints the four equations above and omits five settings that the authors'
own code for *this* task -- ``trainers/geometric_graph.py`` with
``FixedTargetGAE``, ``EncoderEGNCA`` and ``EGC`` in ``gengala/egnca`` -- carries
on by default: a tanh on ``phi_x``'s output, that layer zero-initialised and
unbiased, a residual ``h`` update, a tanh hidden activation, and PairNorm on the
node features. All five are on here, with the authors' defaults, and
:class:`EGNCARule` names each one against the default it carries there. Written
from the equations alone the port diverged, and
``docs/claims/the-divergence-is-a-missing-published-tanh-not-a-property-of-the-rule.md``
is the account of why.

**Where this port departs from the published setup, stated so a result from it
is not read as a replication of theirs.** The *rule* is theirs. The *task* is
this project's, and three things differ:

  * **The start state is anchored.** Gala draw every node from an isotropic
    Gaussian and record "no anchors. Every node starts random." Regeneration
    anchors the whole surviving context at its crystal pose and displaces a
    12-residue window. That is the protein instance Ulban's 23 Nov decision
    maps, not the published one, and it is a strictly easier start.
  * **The loss is all-pairs over a restricted pair set.** Gala's ``L_INV`` runs
    over *all* ``(i, j)``; ``growth.pairwise_distance_restraint`` covers
    window-to-context and window-to-window and leaves out context-to-context,
    which the frozen context satisfies exactly for free. The gradient content
    is the same; the A^2 values are a mean over a different denominator and are
    not comparable with a true all-pairs figure.
  * **Perturbation is the damage window, not Gaussian noise on half the batch.**
    The published regeneration training perturbs half of each batch with
    Gaussian noise, partly globally and partly locally. Here every sample
    carries the project's own damage operator instead. The replay pool is
    ported; this half of their recipe is not.

Nothing here tests persistence past the training horizon, which is the
published rule's headline claim -- damage at t=24, then 1,000 further steps.
The pool trains towards it and no readout measures it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .torch_graph import (
    FeatureTopology,
    FixedNeighbours,
    select_neighbours,
    select_neighbours_device,
)

#: Gala's widths. Not defaults to be tuned: the published count is quoted
#: against them (``docs/claims/an-equivariant-graph-nca-morphs-random-
#: coordinates-into-a-target-point-cloud.md``: "Hidden 16, message 32, 5K
#: parameters total"), and :data:`PUBLISHED_COUNT` is what they cost here.
HIDDEN_DIM = 16
MESSAGE_DIM = 32

#: What :class:`EGNCARule` holds at :data:`HIDDEN_DIM` / :data:`MESSAGE_DIM`
#: with no edge attributes: 5,328 parameters against the paper's "5K".
#: ``tests/test_egnca.py`` pins it and derives it block by block, so a change to
#: the architecture has to come through that file rather than leaving the
#: comparison with the published figure stale in prose.
#:
#: It was 5,329 while ``phi_x``'s output layer carried a bias. The authors
#: build that layer as ``nn.Linear(message_dim, 1, bias=False)``, so the one
#: parameter is gone and the count is theirs rather than ours.
PUBLISHED_COUNT = 5_328

#: The hidden activation, selectable because the authors select it: ``-act``
#: in ``trainers/geometric_graph.py`` defaults to ``'tanh'`` and sets the same
#: activation in all three MLPs. ``silu`` is kept because it is what this port
#: ran before the authors' code was read, so the departure can be reproduced.
ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU}

#: ``-nt``/``--norm_type`` in the same trainer, whose default is ``'pn'``.
NORM_TYPES = ("pn", "none")


def pair_norm(h: Tensor, scale: float = 1.0, eps: float = 1e-6) -> Tensor:
    """PyG's ``PairNorm(scale=1.0)``, written out rather than imported.

    The authors' ``-nt`` default is ``'pn'``, which builds
    ``torch_geometric.nn.PairNorm(scale=1.0)``. ``torch_geometric`` is **not** a
    dependency of this package (``pyproject.toml`` declares ``numpy`` and
    ``biotite``), and PairNorm in its default mode is two lines: centre the rows,
    then rescale so the mean squared row norm is ``scale ** 2``. Written here so
    the behaviour is readable at the call site and testable without a new
    dependency that would have to hold on the laptop and on the cluster
    mid-experiment.

    Applied to the **node features only, never to the coordinates** -- a
    normaliser on ``x`` would not be equivariant under translation and is not
    what their encoder does.
    """
    centred = h - h.mean(dim=0, keepdim=True)
    return scale * centred / (centred.pow(2).sum(-1).mean() + eps).sqrt()


class EGNCARule(nn.Module):
    """The four lines above, and nothing else.

    Three MLPs, two layers each, in the shape the published rule uses:

    ====== ============================== ==================
    net    in                             out
    ====== ============================== ==================
    phi_m  1 + 2*hidden + edge            message
    phi_x  message                        **1** (a scalar)
    phi_h  hidden + message               hidden
    ====== ============================== ==================

    ``edge_dim`` is ``e_ij``'s width and defaults to 0, which is the
    pattern-formation task: node features all 1 and no edge attributes. The
    protein port passes none either -- a residue type or an RBF expansion would
    be a second change on top of the port.

    **No step-size cap.** ``has_coord_act`` bounds ``phi_x``'s scalar *output*
    in (-1, 1), which is the authors' own default -- but that is a bound on the
    **coefficient**, not on the displacement: the vector it multiplies is
    ``x_i - x_j``, so one edge's contribution is up to ``||x_i - x_j||`` and the
    step this rule may take grows without limit with the neighbourhood's own
    scale -- measured over two decades of scale in ``tests/test_egnca.py``,
    under the test named for that distinction. A cap on the displacement would
    be the project's own
    0.3 A actuator bound smuggled back in, and that bound is a third of the
    journey the window has to travel.

    **The five settings below are the authors' defaults, not options this port
    invented.** Each is a *removal of a departure of ours*: the port was written
    from the paper's four printed equations, and the paper omits five things
    that the authors' own code for this task
    (``trainers/geometric_graph.py``, ``FixedTargetGAE``, ``EncoderEGNCA``,
    ``EGC`` in ``gengala/egnca``) switches on by default. They are named here
    with the default they carry there, and
    ``docs/claims/the-divergence-is-a-missing-published-tanh-not-a-property-of-the-rule.md``
    records why the port diverged without them.

    ``has_coord_act``
        ``nn.Tanh() if has_coord_act else nn.Identity()`` as ``phi_x``'s last
        element. Their ``parser.set_defaults(has_coord_act=True)``, with
        ``-nca`` to disable it, so **on** is the published setting.
    ``is_residual``
        ``h' = h + phi_h(h, m)``. ``is_residual`` defaults **True** in that
        trainer, which settles a question this module's note had recorded as
        unanswerable from the paper: the published update is residual.
    ``activation``
        ``-act`` defaults ``'tanh'`` there, and sets the hidden activation in
        all three MLPs.
    ``norm_type``
        ``-nt`` defaults ``'pn'``: PairNorm on the node features after each
        step. See :func:`pair_norm`.

    The fifth is ``phi_x``'s output layer, which is not an argument because the
    authors do not make it one: see ``__init__``.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        message_dim: int = MESSAGE_DIM,
        edge_dim: int = 0,
        has_coord_act: bool = True,
        is_residual: bool = True,
        activation: str = "tanh",
        norm_type: str = "pn",
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or message_dim < 1:
            raise ValueError(
                f"hidden_dim and message_dim must be positive, got "
                f"{hidden_dim} and {message_dim}"
            )
        if edge_dim < 0:
            raise ValueError(f"edge_dim must not be negative, got {edge_dim}")
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {', '.join(sorted(ACTIVATIONS))}, "
                f"got {activation!r}"
            )
        if norm_type not in NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {', '.join(NORM_TYPES)}, got "
                f"{norm_type!r}"
            )
        if not isinstance(has_coord_act, bool) or not isinstance(is_residual, bool):
            raise ValueError(
                "has_coord_act and is_residual are switches and must be bools, "
                f"got {has_coord_act!r} and {is_residual!r}"
            )
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.edge_dim = edge_dim
        self.has_coord_act = has_coord_act
        self.is_residual = is_residual
        self.activation = activation
        self.norm_type = norm_type
        act = ACTIVATIONS[activation]
        self.phi_m = nn.Sequential(
            nn.Linear(1 + 2 * hidden_dim + edge_dim, message_dim),
            act(),
            nn.Linear(message_dim, message_dim),
            act(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            act(),
            # **No bias, and the weight starts at exactly zero.** The authors'
            # ``EGC`` builds this layer as ``nn.Linear(message_dim, 1,
            # bias=False)`` and then calls
            # ``last_coord_layer.weight.data.zero_()``; the
            # ``xavier_uniform_(..., gain=0.001)`` line sits **commented out**
            # beside it. So the rule emits no displacement at all until
            # training moves this weight, which is a stronger start than the
            # small-gain initialisation this port used to carry -- that one was
            # described here as "the reference E(n)-GNN initialisation" and was
            # in fact a workaround for the missing ``has_coord_act`` tanh,
            # chosen so an untrained rollout would not overflow.
            nn.Linear(message_dim, 1, bias=False),
            nn.Tanh() if has_coord_act else nn.Identity(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, message_dim),
            act(),
            nn.Linear(message_dim, hidden_dim),
        )
        self.last_coord_layer.weight.data.zero_()

    @property
    def last_coord_layer(self) -> nn.Linear:
        """``phi_x``'s output ``Linear``, on the authors' own name for it.

        A property rather than an attribute: assigning the same module twice
        would list it a second time in ``named_children``, and
        ``tests/test_egnca.py::test_the_rule_has_no_frame_machinery`` asserts
        this rule's children are exactly the three MLPs.
        """
        return self.phi_x[2]

    def parameter_count(self) -> int:
        """Total trainable parameters, on ``AANCARule.parameter_count``'s name."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def dtype(self) -> torch.dtype:
        """The weights' dtype, which is the dtype the state has to arrive in.

        ``feature_topology`` hands back **float64** coordinates, because the
        torsion arithmetic's exactness guarantee lives there, and the project
        featurises from ``coords.float()`` for training
        (``torch_graph.build_graph``). The state this rule carries *is*
        coordinates, so the cast happens once at the start of the rollout rather
        than inside every ``Linear``, and the carrier casts back in
        :func:`with_ca_positions`.
        """
        return self.phi_m[0].weight.dtype

    def initial_hidden(self, n_residues: int, device=None, dtype=None) -> Tensor:
        """``h`` at step 0: **all ones**, as published ("node features all 1").

        Not zeros and not learned. A constant start is what makes the state
        one-step Markovian with no step index -- every node is identical until
        geometry distinguishes it.
        """
        return torch.ones(
            n_residues,
            self.hidden_dim,
            device=device,
            dtype=dtype or torch.get_default_dtype(),
        )

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        neighbours: Tensor,
        edge_mask: Tensor,
        edge_attr: Tensor | None = None,
        moving: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One CA step. ``x`` is ``(R, 3)`` and ``h`` is ``(R, hidden)``.

        ``neighbours``/``edge_mask`` are a selection in ``torch_graph``'s shape:
        ``(R, K)`` indices with a bool companion, where a padded slot carries
        index 0 and a false mask. The two only mean anything together.

        ``moving`` is ``(R,)`` bool and holds the rest of the structure still:
        the regeneration task's surviving context keeps its crystal pose. A
        held residue still **sends** messages -- it is part of the
        neighbourhood the window reads -- and both its coordinate and its
        hidden state are left exactly as they came in, so nothing about it
        drifts over a rollout. That restore happens **after** PairNorm, so a
        held residue's ``h`` is the one it arrived with even though the
        normaliser read it.
        """
        if x.dim() != 2 or x.shape[-1] != 3:
            raise ValueError(f"x must be (R, 3), got {tuple(x.shape)}")
        if h.shape != (x.shape[0], self.hidden_dim):
            raise ValueError(
                f"h must be ({x.shape[0]}, {self.hidden_dim}), got "
                f"{tuple(h.shape)}"
            )
        if neighbours.shape != edge_mask.shape:
            raise ValueError(
                f"neighbours {tuple(neighbours.shape)} and edge_mask "
                f"{tuple(edge_mask.shape)} must agree"
            )
        if self.edge_dim and edge_attr is None:
            raise ValueError(
                f"this rule was built for {self.edge_dim}-channel e_ij and "
                "none was passed -- a rule that silently dropped its edge "
                "attributes would train on a different input than its config "
                "claims"
            )
        if edge_attr is not None and edge_attr.shape[-1] != self.edge_dim:
            raise ValueError(
                f"edge_attr must be (..., {self.edge_dim}), got "
                f"{tuple(edge_attr.shape)}"
            )

        weight = edge_mask.to(x.dtype).unsqueeze(-1)  # (R, K, 1)
        # (x_i - x_j), the only place a direction enters. Everything the
        # networks see is built from its squared length.
        difference = x.unsqueeze(1) - x[neighbours]  # (R, K, 3)
        squared = (difference**2).sum(-1, keepdim=True)  # (R, K, 1)

        parts = [
            squared,
            h.unsqueeze(1).expand(-1, neighbours.shape[1], -1),  # h_i
            h[neighbours],  # h_j
        ]
        if edge_attr is not None:
            parts.append(edge_attr)
        m_ij = self.phi_m(torch.cat(parts, dim=-1)) * weight  # (R, K, message)

        # phi_x is scalar-valued: one learned weight per edge, not a vector.
        scale = self.phi_x(m_ij) * weight  # (R, K, 1)
        degree = weight.sum(dim=1).clamp_min(1.0)  # (R, 1); |N(i)|
        shift = (difference * scale).sum(dim=1) / degree  # (R, 3)

        m_i = m_ij.sum(dim=1)  # (R, message)
        x_new = x + shift
        h_new = self.phi_h(torch.cat([h, m_i], dim=-1))
        # ``is_residual`` is the authors' default (True): the node update adds
        # to the state rather than replacing it.
        if self.is_residual:
            h_new = h + h_new
        # PairNorm, on the node features only and after the step, which is
        # where their encoder applies it. Note that it reads **all** rows: the
        # centring is over the whole structure, held residues included, which
        # is what makes it a normaliser over the graph rather than over the
        # window.
        if self.norm_type == "pn":
            h_new = pair_norm(h_new)

        if moving is not None:
            if moving.shape != (x.shape[0],) or moving.dtype is not torch.bool:
                raise ValueError(
                    f"moving must be a ({x.shape[0]},) bool mask, got "
                    f"{tuple(moving.shape)} of {moving.dtype}"
                )
            keep = moving.unsqueeze(-1)
            x_new = torch.where(keep, x_new, x)
            h_new = torch.where(keep, h_new, h)
        return x_new, h_new


# -- the coordinates this rule moves, and the ones it borrows ---------------


def ca_positions(coords: Tensor, topo: FeatureTopology) -> Tensor:
    """``(R, 3)`` CA, on ``fape.ca_rmsd_on``'s convention (``frame_idx[:, 1]``).

    The same index every CA-level quantity in this project reads, so the rule's
    state and the loss are talking about the same atom.
    """
    index = torch.arange(topo.n_residues, device=coords.device)
    return coords[index.unsqueeze(-1), topo.frame_idx[index, 1:2]][:, 0]


def with_ca_positions(coords: Tensor, ca: Tensor, topo: FeatureTopology) -> Tensor:
    """``coords`` with each residue translated so its CA sits at ``ca``.

    The carrier described in this module's docstring. A rigid **translation**
    per residue, not a reorientation: this rule produces no orientation, so the
    other atoms keep the one they came in with. That is why nothing in
    :func:`egnca_objective` reads them, and why no orientation-dependent metric
    is reported -- fed the native orientation, FAPE and ``twist`` would read as
    near-perfect and would be measuring the carrier.
    """
    offset = (ca.to(coords.dtype) - ca_positions(coords, topo)).unsqueeze(1)
    return coords + offset * topo.atom_mask.unsqueeze(-1).to(coords.dtype)


def egnca_neighbours(
    coords: Tensor,
    topo: FeatureTopology,
    k: int = 64,
    cutoff: float = 10.0,
    device_neighbours: bool = False,
    feature_dtype: torch.dtype = torch.float32,
) -> FixedNeighbours:
    """The project's own selection, run on ``coords``.

    **Reused, not reimplemented.** These are the two selectors
    ``torch_graph.build_graph`` itself dispatches between, picked by the same
    flag, with the same ``k`` and ``cutoff`` and
    ``include_sequence_neighbours`` left at the default -- so this rule
    perceives through the project's perception and a difference in a result
    cannot be a difference in the graph. It is the same construction
    ``growth.native_edge_selection`` makes for ``AANCARule``, minus the
    blinding branch, which is about side chains this rule has none of.

    This is the one place the carrier's non-CA atoms are read: the selection is
    a minimum heavy-atom distance. Under the native-edge path -- the task
    statement, per
    ``docs/claims/the-task-is-to-realise-coordinates-for-a-given-contact-graph.md``
    -- that read is on the native coordinates themselves and the carrier never
    enters. Under a live rebuild it is on side-chain clouds translated with
    their CA, which is stated rather than hidden, and is why the native path is
    the default in :func:`egnca_rollout`.
    """
    select = select_neighbours_device if device_neighbours else select_neighbours
    neighbours, edge_mask = select(coords.to(feature_dtype), topo, k, cutoff)
    return FixedNeighbours(neighbours=neighbours, edge_mask=edge_mask)


@dataclass
class EGNCAState:
    """The whole state: CA positions and hidden channels. Nothing else.

    No alive mask, no committed identity, no frames -- the state is one-step
    Markovian and carries no step index, as published.
    """

    ca: Tensor
    hidden: Tensor


def egnca_step(
    rule: EGNCARule,
    state: EGNCAState,
    neighbours: Tensor,
    edge_mask: Tensor,
    moving: Tensor | None = None,
) -> EGNCAState:
    """One step. This rule's ``rollout.step_once``, and it has no actuator.

    ``rollout.step_once`` moves coordinates only through
    ``growth.rigid_relax``, which needs an N-CA-C frame, an axis-angle and a
    step cap. Here the rule's own output *is* the displacement: there is no
    actuator between the network and the coordinates, which is the structural
    difference the port is testing.
    """
    ca, hidden = rule(state.ca, state.hidden, neighbours, edge_mask, moving=moving)
    return EGNCAState(ca=ca, hidden=hidden)


def egnca_rollout(
    rule: EGNCARule,
    state: EGNCAState,
    topo: FeatureTopology,
    steps: int,
    native_coords: Tensor,
    moving: Tensor | None = None,
    k: int = 64,
    cutoff: float = 10.0,
    device_neighbours: bool = False,
    native_edges: bool = True,
    trajectory: list[Tensor] | None = None,
) -> EGNCAState:
    """``steps`` steps, **fully differentiated**: no truncation, no detach.

    Gala trains by BPTT over 15 to 25 steps sampled uniformly. That is short
    enough to differentiate end to end, which is the point: this project's own
    runs took 120 steps with ``truncate 12``, so the gradient never reached
    past a tenth of the rollout, and the one arm that raised truncation left the
    0.3 A cap in place and so tested neither change
    (``docs/claims/the-task-is-to-realise-coordinates-for-a-given-contact-graph.md``).

    ``native_edges`` selects the task statement: the edge set the **native**
    coordinates pick, computed once and held for the rollout, which is what the
    published rule message-passes over ("the fixed target graph, not a graph
    recomputed from current positions"). False recomputes the selection every
    step from the live CA, which is the harder problem and is kept available
    because it is the project's own earlier behaviour.
    """
    if steps < 0:
        raise ValueError(f"steps must not be negative, got {steps}")
    fixed = (
        egnca_neighbours(native_coords, topo, k, cutoff, device_neighbours)
        if native_edges
        else None
    )
    for _ in range(steps):
        if fixed is not None:
            selection = fixed
        else:
            selection = egnca_neighbours(
                with_ca_positions(native_coords, state.ca.detach(), topo),
                topo,
                k,
                cutoff,
                device_neighbours,
            )
        state = egnca_step(
            rule, state, selection.neighbours, selection.edge_mask, moving
        )
        if trajectory is not None:
            trajectory.append(state.ca)
    return state


# -- the task, the loss and what may be read off it -------------------------


@dataclass
class EGNCATerms:
    """The loss and the CA-level readouts. Deliberately short.

    Every frame-shaped quantity ``growth.GrowthTerms`` carries -- ``fape``,
    ``twist``, ``bond``, ``orientation``, ``clash``, ``identity``,
    ``completion``, ``reach`` -- is absent rather than zero, for the reason in
    this module's docstring: this rule places no orientation and no atom other
    than CA, so those are not numbers it could produce.
    """

    total: Tensor
    pair_anchor: Tensor
    pair_window: Tensor
    #: Window CA RMSD, superposed on the whole structure.
    rmsd_damaged: float | None = None
    #: The same, superposed on the surviving context alone -- the convention
    #: that does not let the displaced window absorb part of its own error.
    rmsd_damaged_fit_intact: float | None = None
    #: What the window scored before a single step. The pool's accept bar and
    #: the only thing that says whether a number is better than not trying.
    start_rmsd_damaged: float | None = None
    #: Mean CA displacement over the rollout, in Angstrom. Without it a flat
    #: RMSD cannot be told from a rule that emits nothing.
    move: float | None = None
    damaged_mask: Tensor | None = None


@dataclass
class EGNCAPoolEntry:
    """One remembered end-state, for the replay pool.

    Not ``growth.GrowthPoolEntry``: that one's ``coords`` is the padded
    ``(R, MAX_ATOMS, 3)`` tensor and it carries a committed ``identity``, and
    resuming this rule from it would mean agreeing on a shape and a field that
    do not apply. The state here is the whole state -- CA and hidden -- which
    is the point of a pool: coordinates alone would hand the rule a structure
    it had no memory of building.

    ``damaged`` travels with the state and is never re-drawn, for
    ``GrowthPoolEntry``'s reason: the loss is scored on the window, so a freshly
    drawn window over resumed coordinates would score a hole this state was
    never asked to fill.
    """

    ca: Tensor
    hidden: Tensor
    damaged: Tensor
    baseline_rmsd: float | None = None


def egnca_objective(
    rule: EGNCARule,
    native_coords: Tensor,
    topo: FeatureTopology,
    steps: int,
    damage: int = 12,
    generator: torch.Generator | None = None,
    pairwise_weight: float = 1.0,
    pairwise_window_weight: float = 1.0,
    k: int = 64,
    cutoff: float = 10.0,
    device_neighbours: bool = False,
    native_edges: bool = True,
    initial: str = "helix",
    resume_from: EGNCAPoolEntry | None = None,
) -> tuple[EGNCATerms, EGNCAState]:
    """Regeneration, as this project defines it, under the published rule.

    A contiguous ``damage``-residue window inside an intact context: everything
    outside it starts and stays at its crystal pose, and the window starts where
    ``growth.baseline_coords``' reference walk put it -- ~13.3 A out on the
    project's own 384-residue crops, 4.80 A on whole 1UBQ under the
    whole-structure superposition. Both operators are the project's own, reused:
    the task is unchanged and only the rule is new.

    **The loss is the unclamped all-pairs CA distance restraint over the full
    pair set**, which is ``growth.pairwise_distance_restraint`` with both halves
    weighted: ``pairwise_weight`` on window-to-context and
    ``pairwise_window_weight`` on window-to-window. Gala's authors call the
    all-pairs form *necessary to uniquely identify the target state*, and
    weighting the internal half is Ulban's decision of 23 Nov. Unclamped is the
    whole point: a squared distance error grows with the displacement, so the
    gradient does not fade over the 13 A the window has to travel, where FAPE's
    10 A clamp is exactly flat.

    Nothing else is in ``total``. No bond term, no clash term, no FAPE: the rule
    places no atom a bond or a clash could be computed between.
    """
    from .fape import backbone_rmsd_on
    from .growth import (
        baseline_coords,
        damage_window,
        growth_links,
        pairwise_distance_restraint,
    )

    links = growth_links(topo)
    if resume_from is not None:
        damaged = resume_from.damaged
        start_ca = resume_from.ca
        hidden = resume_from.hidden
    else:
        damaged = damage_window(
            topo.n_residues, links, damage, generator, native_coords.device
        )
        with torch.no_grad():
            start = baseline_coords(
                native_coords, topo, links, ~damaged, initial, generator
            )
        start_ca = ca_positions(start, topo).detach().to(rule.dtype)
        hidden = rule.initial_hidden(
            topo.n_residues, device=native_coords.device, dtype=rule.dtype
        )
    intact = ~damaged

    state = egnca_rollout(
        rule,
        EGNCAState(ca=start_ca, hidden=hidden),
        topo,
        steps,
        native_coords,
        moving=damaged,
        k=k,
        cutoff=cutoff,
        device_neighbours=device_neighbours,
        native_edges=native_edges,
    )

    final = with_ca_positions(native_coords, state.ca, topo)
    anchor, window = pairwise_distance_restraint(
        final, native_coords, topo, damaged, intact
    )
    total = pairwise_weight * anchor + pairwise_window_weight * window

    terms = EGNCATerms(
        total=total,
        pair_anchor=anchor.detach(),
        pair_window=window.detach(),
        damaged_mask=damaged,
        move=float((state.ca - start_ca).detach().norm(dim=-1)[damaged].mean())
        if bool(damaged.any())
        else None,
    )
    # **A diverged rollout reports nan, it does not raise.** The displacement
    # is unbounded even with the authors' tanh on ``phi_x`` -- the tanh bounds
    # the coefficient and the vector it scales is ``x_i - x_j`` -- and the
    # authors report exploding gradients, so a non-finite end state is an
    # ordinary outcome of this rule rather than a bug -- and ``ca_rmsd_on``'s SVD raises ``_LinAlgError`` on one, which
    # would take down a training run several hours in instead of skipping an
    # update. The trainer's own non-finite guard is what handles it.
    if (
        bool(damaged.any())
        and int(intact.sum()) >= 3
        and bool(torch.isfinite(state.ca).all())
    ):
        with torch.no_grad():
            everything = torch.ones_like(damaged)
            begin = with_ca_positions(native_coords, start_ca, topo)
            terms.rmsd_damaged = float(
                backbone_rmsd_on(final, native_coords, topo, everything, damaged)
            )
            terms.rmsd_damaged_fit_intact = float(
                backbone_rmsd_on(final, native_coords, topo, intact, damaged)
            )
            terms.start_rmsd_damaged = float(
                backbone_rmsd_on(begin, native_coords, topo, everything, damaged)
            )
    return terms, state
