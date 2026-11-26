"""A GLOBAL FOLD-CLASS LABEL on Gala's rule, entering ``phi_m``'s invariant input.

Registered in
``docs/claims/a-global-fold-class-label-tells-us-whether-option-2-failed-on-ambiguity-or-on-address.md``
as a **diagnostic** arm. It cannot make option 2 true; what it can do is say
which of two diagnoses is right. H-ambiguity says one rule with no timestep and
no conditioning was asked to map one Gaussian prior onto the whole distribution
of folds, so narrowing the target makes the constraints satisfiable.
H-address says the deficit is positional and a label cannot supply an address.

**A fold class is a global SCALAR, not a global coordinate.** It is the same
number for every residue in the structure, so it cannot tell a residue where it
is. That is what makes it a clean discriminator between those two hypotheses --
and it is also why it is permitted where ``morphogen.py``'s chain-transported
``decay ** index`` is not: the project's standing position is that no cell may
read a global *coordinate*, and this is not one.

**Why a subclass and not an edit.** ``src/aanca/egnca.py`` carries the published
port's result (Bunya job 33419979) and must stay bit-for-bit, and
``scripts/train_denoise.py``'s checkpoints must stay readable by
``scripts/prior_fixed_point.py`` and ``scripts/denoise_floor.py``, which refuse
a checkpoint missing any key their trainer's schema defines. So the conditioning
is a subclass here and a new trainer there.

**Where the label enters, and the three alternatives it rules out.**
:class:`EGNCARule` already accepts an ``edge_dim``-wide ``e_ij`` and
concatenates it into ``phi_m``'s input beside ``||x_i - x_j||^2``, ``h_i`` and
``h_j``. This class sets ``edge_dim`` to the label width and *fills* that slot,
at every step, with a learned embedding of the class broadcast to every residue
and every edge. So the published mechanism carries it rather than a new one.

  * **Not the initial hidden state.** A seed relies on the rule learning to
    *preserve* the signal across 15-25 steps, which is a second question
    confounded with the first.
  * **Not added to the hidden state.** ``norm_type = pn`` is PairNorm, which
    centres the rows over the whole structure and would normalise a constant
    offset away exactly. Concatenating at ``phi_m`` bypasses it.
  * **Not a per-residue feature.** One label per structure, identical
    everywhere, is the whole point: a per-residue channel would be address
    information and would confound the two hypotheses it exists to separate.

**Equivariance is preserved by construction**, because the embedding is an
E(n)-invariant scalar vector entering the invariant branch -- the same place the
squared distance enters, and ``phi_x``'s output still only scales
``x_i - x_j``. It is nevertheless tested, in ``tests/test_conditioned.py``, to
the 1e-4 the existing arms are tested at and with the live k-NN graph.

**The label is current state on the module, set through
:meth:`ConditionedEGNCARule.conditioned_on`.** That is deliberate: it is what
lets ``egnca_rollout``, ``egnca_step`` and ``denoise_objective`` be *imported
and called unchanged* rather than forked to thread one more argument through
four layers. The context manager restores the previous label on exit, so a
forgotten reset cannot leak one sample's condition into the next.

**Label dropout** replaces the class with a learned "unconditional" row at a
configured probability, so an unconditional pass exists and classifier-free
guidance is available later without retraining. **CFG is not implemented
anywhere in this arm**: it needs two forward passes and a guidance scale, and a
guidance scale is a second axis.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor, nn

from .egnca import PUBLISHED_COUNT, EGNCARule

#: The four classes ``generation_metrics.structure_class`` returns, in the order
#: their embedding rows are indexed. **This tuple must match that function's
#: return values exactly** -- it is not a second definition of the classes, and
#: ``tests/test_conditioned.py`` asserts the agreement by calling
#: ``structure_class`` rather than by restating its thresholds.
#:
#: It is a P-SEA-derived C-level *analogue*, deliberately aligned with CATH's
#: four classes (Orengo et al. 1997) and computed by this project's own
#: validated code. **It is ours and must be named as ours**: no claim from this
#: arm may say "CATH class".
FOLD_CLASSES = ("mainly-alpha", "mainly-beta", "alpha/beta", "irregular")

#: The embedding row that means "no class was given". One row past the four, so
#: label dropout has somewhere to point and an unconditional forward pass is a
#: real pass through the same weights rather than a zeroed input.
UNCONDITIONAL = len(FOLD_CLASSES)

#: The embedding's width. 4 channels over 5 rows is 20 weights, and widening
#: ``phi_m``'s input by 4 costs ``4 * message_dim`` more: ~148 against the
#: port's 5,328. **The smallness is load-bearing for the whole framing** -- if
#: this arm needs to grow the rule to work, the "one small local rule" claim is
#: what is being traded away, and that trade gets argued before it is made.
LABEL_DIM = 4


def class_index(name: str) -> int:
    """The embedding row for a ``structure_class`` string.

    An unknown name is an error rather than a fallback to ``UNCONDITIONAL``: a
    silently unconditional sample would look exactly like a live label in every
    column of the log, which is the failure readout 0 exists to catch.
    """
    try:
        return FOLD_CLASSES.index(name)
    except ValueError:
        raise ValueError(
            f"unknown fold class {name!r}; generation_metrics.structure_class "
            f"returns one of {', '.join(FOLD_CLASSES)}"
        ) from None


def class_name(index: int) -> str:
    """The inverse, with the unconditional row named rather than out of range."""
    if index == UNCONDITIONAL:
        return "unconditional"
    if not 0 <= index < len(FOLD_CLASSES):
        raise ValueError(
            f"label index {index} is outside 0..{UNCONDITIONAL} "
            f"({len(FOLD_CLASSES)} classes plus the unconditional row)"
        )
    return FOLD_CLASSES[index]


def effective_label(label: int, label_dropout: float, draw: float | None) -> int:
    """Which row this forward pass conditions on, after dropout.

    ``draw`` is a uniform ``[0, 1)`` sample at training time and **None** at
    evaluation and in the fixed-point sampler, where the condition is the
    question being asked and must not be perturbed. ``label_dropout >= 1.0`` is
    unconditional either way, which is what makes the ``nolabel`` control a
    control rather than a differently-seeded arm: the class never reaches the
    rule, in training or in evaluation.
    """
    if not 0.0 <= label_dropout <= 1.0:
        raise ValueError(f"label_dropout must be in [0, 1], got {label_dropout}")
    if label_dropout >= 1.0:
        return UNCONDITIONAL
    if draw is not None and draw < label_dropout:
        return UNCONDITIONAL
    return label


class ConditionedEGNCARule(EGNCARule):
    """Gala's rule with a global fold-class label in ``phi_m``'s input.

    Everything about the rule -- the four equations, the five published
    settings, ``phi_x``'s zero-initialised unbiased output layer, PairNorm --
    is :class:`EGNCARule`'s, inherited and not restated. The two additions are
    an ``nn.Embedding`` of ``len(FOLD_CLASSES) + 1`` rows and the override that
    fills the inherited ``edge_attr`` slot with the current label's row.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        message_dim: int = 32,
        has_coord_act: bool = True,
        is_residual: bool = True,
        activation: str = "tanh",
        norm_type: str = "pn",
        label_dim: int = LABEL_DIM,
    ) -> None:
        if label_dim < 1:
            raise ValueError(f"label_dim must be positive, got {label_dim}")
        super().__init__(
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            edge_dim=label_dim,
            has_coord_act=has_coord_act,
            is_residual=is_residual,
            activation=activation,
            norm_type=norm_type,
        )
        self.label_dim = label_dim
        self.n_classes = len(FOLD_CLASSES)
        self.label_embedding = nn.Embedding(len(FOLD_CLASSES) + 1, label_dim)
        #: The row the next forward pass conditions on. Starts unconditional so
        #: a caller that forgets to set one gets the honest answer rather than
        #: ``mainly-alpha`` by accident of its being index 0.
        self._label = UNCONDITIONAL

    @property
    def label(self) -> int:
        return self._label

    @label.setter
    def label(self, index: int) -> None:
        class_name(index)  # validates, and raises with the row count
        self._label = int(index)

    @contextmanager
    def conditioned_on(self, index: int):
        """Run a block with the label set, and restore it afterwards.

        The restore is the point: a rollout left conditioned on the previous
        sample's class would be a measurement of the wrong question, and nothing
        in the printed table would say so.
        """
        previous = self._label
        self.label = index
        try:
            yield self
        finally:
            self._label = previous

    def label_row(self, index: int | None = None) -> Tensor:
        """``(label_dim,)``: the embedding of ``index``, or of the current label."""
        row = self._label if index is None else index
        class_name(row)
        return self.label_embedding(
            torch.tensor(row, device=self.label_embedding.weight.device)
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
        """One CA step, with the label broadcast into every edge's ``e_ij``.

        An explicitly passed ``edge_attr`` wins, so the inherited contract still
        holds and a test can feed a chosen vector straight in. When none is
        passed -- which is every call ``egnca_step`` makes -- the current
        label's embedding row is expanded to ``(R, K, label_dim)``. The expand
        is a view, so the label costs one row read per step and no memory.
        """
        if edge_attr is None:
            row = self.label_row().to(x.dtype)
            edge_attr = row.expand(x.shape[0], neighbours.shape[1], self.label_dim)
        return super().forward(x, h, neighbours, edge_mask, edge_attr, moving)


def conditioning_cost(message_dim: int = 32, label_dim: int = LABEL_DIM) -> int:
    """What the label costs in parameters, derived rather than quoted.

    ``label_dim * message_dim`` for ``phi_m``'s widened first layer plus
    ``(len(FOLD_CLASSES) + 1) * label_dim`` for the embedding. Added to
    :data:`egnca.PUBLISHED_COUNT` this is the whole parameter count, and
    ``tests/test_conditioned.py`` checks that identity against the module's own
    ``parameter_count()`` so the two cannot drift.
    """
    return label_dim * message_dim + (len(FOLD_CLASSES) + 1) * label_dim


#: The conditioned rule's count at the published widths: 5,328 + 148 = 5,476.
#: Quoted so a change to the architecture has to come through this constant and
#: the test that pins it, rather than leaving the comparison with the published
#: ~5K stale in prose.
CONDITIONED_COUNT = PUBLISHED_COUNT + conditioning_cost()
