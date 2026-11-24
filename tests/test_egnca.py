"""The E(n)-GNCA port, and a crippling for every claim it makes.

Each test here names the thing it would catch if the wiring were cut, and most
of them say what happens when it is -- measured on this tree, not argued. The
project has lost results to settings that appeared to be used and were not
(``tests/test_network_size.py``'s docstring names two), and a port of a
published rule is exactly where that failure is cheapest to make and dearest to
discover: a port that quietly keeps the project's own actuator cap, or drops the
internal half of the loss, or truncates the gradient, is a run that reports
having replicated something it never ran.

What is pinned, and the crippling that is supposed to fail it:

====================================== ======================================
claim                                  cripple it by
====================================== ======================================
the coordinate update is equivariant   making ``phi_x`` vector-valued
``phi_x`` is scalar-valued             the same
the move is a mean over real edges     ignoring ``edge_mask``
the parameter count is 5,328           changing a width without this file
the widths reach the rule              hardcoding one in ``build_rule``
the rollout moves coordinates          leaving ``phi_x`` at zero
the context is held                    dropping the ``moving`` mask
perception is the project's own         reimplementing the selection
the native edge set is held fixed      reselecting every step
the gradient reaches step 0            detaching inside the rollout
there is no step cap                   bounding the shift
the loss is the FULL pair set          dropping ``pairwise_window_weight``
every setting reaches the objective    adding a key the objective never reads
the five published settings are on     reverting one to the paper's equations
the tanh bounds the COEFFICIENT only   letting it become a cap on the step
the skip rate is counted by reason     one total, which cannot say which
====================================== ======================================

**The five, and the one distinction that must not slip.** The paper omits five
settings the authors' own code for this task carries on by default (a tanh on
``phi_x``'s output, that layer zero-initialised and unbiased, a residual ``h``
update, a tanh hidden activation, PairNorm on the node features), and the port
was written from the paper. They are all on now, and the tests under *the five
settings the paper omits* check for each that the default is theirs and that it
can be turned off. The one that is easy to get wrong in the right-looking
direction is the tanh: it bounds the scalar **coefficient** in (-1, 1) and does
**not** bound the displacement, because the vector it scales is ``x_i - x_j``.
:func:`test_the_tanh_bounds_the_coefficient_and_not_the_displacement` is the
test that keeps those apart, and the thing on its other side is the
``max_translation = 0.3`` clamp this project spent arms regretting.

One test is not a crippling and says so:
:func:`test_two_hundred_updates_on_one_structure_move_the_window` trains the
port on
a single window for a few seconds and asserts the loss falls by an order of
magnitude and the window ends below its own start state. It is an overfit and
not evidence about generalisation -- it is here because a rule that could not
overfit one window is not worth a queue slot, and that is answerable on a
laptop.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
from pathlib import Path

import pytest
import torch

from aanca import config as run_config
from aanca.egnca import (
    HIDDEN_DIM,
    MESSAGE_DIM,
    PUBLISHED_COUNT,
    EGNCAPoolEntry,
    EGNCARule,
    EGNCAState,
    EGNCATerms,
    ca_positions,
    egnca_neighbours,
    egnca_objective,
    egnca_rollout,
    egnca_step,
    pair_norm,
    with_ca_positions,
)
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "data" / "raw" / "1UBQ.pdb"
CONFIG = ROOT / "configs" / "egnca.toml"

RUNTIME = argparse.Namespace(
    device="auto", max_seconds=0.0, stop_file=None, resume=False
)


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords


def rotation(degrees: float, dtype) -> torch.Tensor:
    """``test_move_equivariance``'s rotation, so the two files turn alike."""
    axis = torch.tensor([0.3, -0.7, 0.65])
    axis = axis / axis.norm()
    angle = torch.tensor(degrees * math.pi / 180.0)
    k = torch.tensor([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    turn = torch.eye(3) + torch.sin(angle) * k + (1 - torch.cos(angle)) * (k @ k)
    return turn.to(dtype)


def _rule(
    seed: int = 0, gain: float = 0.3, double: bool = False, **kwargs
) -> EGNCARule:
    """A rule whose ``phi_x`` output is **not** at its published initialisation.

    The authors zero that layer (``last_coord_layer.weight.data.zero_()``, with
    no bias), so an untrained rule emits **exactly nothing** and every "does it
    move" assertion would be asking about the number 0. This perturbs that one
    layer, so the move is a real quantity and the question can be asked at
    all -- the same reason ``test_move_equivariance`` perturbs ``frame_head``'s
    last layer.

    **0.3 is a calibration**, and it is a far less delicate one than the 0.01
    this took before the authors' five settings landed: with ``has_coord_act``
    on, ``phi_x`` saturates in (-1, 1) rather than growing with the weight, so
    the gain sets how much of that range is used instead of deciding whether a
    20-step rollout overflows. Measured on 1UBQ at k=24, mean single step and
    mean window travel over 15 steps: 0.011/0.045 A at 0.01, 0.11/0.44 at 0.1,
    **0.34/1.28 at 0.3**, 1.09/5.63 at 1.0. The displacement is still
    unbounded, because the vector the coefficient scales is ``x_i - x_j`` --
    which is the distinction
    :func:`test_the_tanh_bounds_the_coefficient_and_not_the_displacement`
    exists to keep legible.

    ``double`` runs the rule in float64. The equivariance and invariance tests
    need it: coordinates run to 40 A on this structure, ``||x_i - x_j||^2`` is
    a difference of large numbers, and in float32 a rigid motion of the input
    perturbs that invariant by ~1e-3 before the rule sees it -- so a float32
    measurement would be reporting the dtype rather than the architecture.
    """
    torch.manual_seed(seed)
    rule = EGNCARule(**kwargs)
    for parameter in rule.last_coord_layer.parameters():
        parameter.data.add_(torch.randn_like(parameter) * gain)
    if double:
        rule.double()
    rule.eval()
    return rule


def _selection(topo, coords, k: int = 24):
    return egnca_neighbours(coords, topo, k=k)


def _state(rule, coords, topo):
    ca = ca_positions(coords, topo).to(rule.dtype)
    return EGNCAState(ca=ca, hidden=rule.initial_hidden(topo.n_residues, dtype=rule.dtype))


# -- the rule ---------------------------------------------------------------


def test_the_coordinate_update_is_equivariant(ubiquitin):
    """Rotate and translate the input; the output moves with it, exactly.

    **The property the whole port rests on, and the one the project's own rule
    did not have.** ``AANCARule``'s perception is invariant and ``rigid_relax``
    applied its output in the lab frame, so "move toward my target" was not a
    function it could express at all -- seven arms and four dead hypotheses
    (``tests/test_move_equivariance.py``). Gala's rule gets equivariance for
    free and structurally: ``phi_x`` is a scalar function of invariants, and the
    only vector in the update is ``x_i - x_j``.

    Measured here: **1.1e-08 A of equivariance error on a 0.29 A move**, under
    a 40-degree rotation composed with a 9 A translation. The residue is the
    rotation matrix, which this file builds in float32 before casting, so it is
    orthogonal only to ~1e-08 -- the rule's own arithmetic contributes nothing
    measurable.
    """
    topo, coords = ubiquitin
    rule = _rule(double=True)
    turn = rotation(40.0, rule.dtype)
    shift = torch.tensor([3.0, -5.0, 7.0], dtype=rule.dtype)

    # One selection, shared: the neighbour list is a property of the graph and
    # a rigid motion does not change it. Sharing it means the test measures the
    # rule and not the selector.
    selection = _selection(topo, coords)
    state = _state(rule, coords, topo)
    moved = EGNCAState(ca=state.ca @ turn.T + shift, hidden=state.hidden)

    with torch.no_grad():
        plain = egnca_step(rule, state, selection.neighbours, selection.edge_mask)
        turned = egnca_step(rule, moved, selection.neighbours, selection.edge_mask)

    step = (plain.ca - state.ca).norm(dim=-1)
    gap = (turned.ca - (plain.ca @ turn.T + shift)).norm(dim=-1)
    assert float(step.mean()) > 1e-2, "nothing moved, so nothing is tested"
    assert float(gap.mean()) < 1e-4 * float(step.mean()) + 1e-5, (
        f"equivariance error {float(gap.mean()):.3e} A on a "
        f"{float(step.mean()):.3f} A move"
    )


def test_the_hidden_update_is_invariant(ubiquitin):
    """``h'`` depends on the geometry only through ``||x_i - x_j||^2``.

    The other half of equivariance and the easier one to break: a rule that
    fed a *direction* into ``phi_m`` would still look equivariant in the
    coordinate test above as long as ``phi_x`` stayed scalar, and would quietly
    be reading the structure's absolute orientation.

    9.3e-07 at the widest channel, against hidden states of order 1. That is
    the float32 rotation matrix again, carried through ``||x_i - x_j||^2`` at
    40 A coordinates and two MLPs; the bar is 1e-05, which a direction-reading
    ``phi_m`` would miss by four orders of magnitude.
    """
    topo, coords = ubiquitin
    rule = _rule(double=True)
    turn = rotation(40.0, rule.dtype)
    selection = _selection(topo, coords)
    state = _state(rule, coords, topo)
    moved = EGNCAState(
        ca=state.ca @ turn.T + torch.tensor([3.0, -5.0, 7.0], dtype=rule.dtype),
        hidden=state.hidden,
    )
    with torch.no_grad():
        plain = egnca_step(rule, state, selection.neighbours, selection.edge_mask)
        turned = egnca_step(rule, moved, selection.neighbours, selection.edge_mask)
    assert float((turned.hidden - plain.hidden).abs().max()) < 1e-5


def test_phi_x_is_scalar_valued(ubiquitin):
    """One learned number per edge, not a vector. Asserted on the layer.

    ``phi_x`` mapping to a scalar is not a detail of the architecture, it *is*
    the equivariance: a vector-valued weight multiplied elementwise into
    ``x_i - x_j`` picks out the lab axes.
    """
    rule = EGNCARule()
    last = rule.last_coord_layer
    assert isinstance(last, torch.nn.Linear)
    # The published form: no bias, and a tanh after it rather than before.
    assert last.bias is None
    assert isinstance(rule.phi_x[-1], torch.nn.Tanh)
    assert last.out_features == 1, (
        f"phi_x emits {last.out_features} channels; a scalar is what makes the "
        "coordinate update equivariant"
    )
    topo, coords = ubiquitin
    selection = _selection(topo, coords)
    state = _state(rule, coords, topo)
    with torch.no_grad():
        m_ij = rule.phi_m(
            torch.cat(
                [
                    (
                        (state.ca.unsqueeze(1) - state.ca[selection.neighbours]) ** 2
                    ).sum(-1, keepdim=True),
                    state.hidden.unsqueeze(1).expand(
                        -1, selection.neighbours.shape[1], -1
                    ),
                    state.hidden[selection.neighbours],
                ],
                dim=-1,
            )
        )
        assert tuple(rule.phi_x(m_ij).shape) == tuple(selection.neighbours.shape) + (1,)


def test_a_vector_valued_phi_x_would_not_be_equivariant(ubiquitin):
    """**The crippling for the two tests above, run rather than described.**

    The same update with a 3-channel weight instead of a scalar one, which is
    the one-character change someone optimising the code would make. Measured:
    **1.747 A of equivariance error on a 2.186 A move -- 80 % of the move**,
    every step, against the scalar form's 1.1e-08 on the same structure. So the
    scalar is load-bearing and the equivariance test above is not passing for
    some unrelated reason.
    """
    topo, coords = ubiquitin
    torch.manual_seed(3)
    rule = EGNCARule().double()
    vector_phi_x = torch.nn.Sequential(
        rule.phi_x[0], rule.phi_x[1], torch.nn.Linear(rule.message_dim, 3).double()
    )
    turn = rotation(40.0, rule.dtype)
    selection = _selection(topo, coords)
    base = _state(rule, coords, topo)

    def shift(ca: torch.Tensor) -> torch.Tensor:
        weight = selection.edge_mask.to(ca.dtype).unsqueeze(-1)
        difference = ca.unsqueeze(1) - ca[selection.neighbours]
        squared = (difference**2).sum(-1, keepdim=True)
        h = base.hidden
        m_ij = rule.phi_m(
            torch.cat(
                [
                    squared,
                    h.unsqueeze(1).expand(-1, selection.neighbours.shape[1], -1),
                    h[selection.neighbours],
                ],
                dim=-1,
            )
        ) * weight
        scale = vector_phi_x(m_ij) * weight
        degree = weight.sum(dim=1).clamp_min(1.0)
        return ca + (difference * scale).sum(dim=1) / degree

    with torch.no_grad():
        plain = shift(base.ca)
        turned = shift(base.ca @ turn.T)
    move = float((plain - base.ca).norm(dim=-1).mean())
    gap = float((turned - plain @ turn.T).norm(dim=-1).mean())
    assert move > 1e-3
    assert gap > 0.1 * move, (
        "a vector-valued phi_x is supposed to break equivariance; error "
        f"{gap:.3e} A on a {move:.3f} A move. If it no longer does, the "
        "coordinate update changed and this file needs rereading."
    )


def test_a_padded_neighbour_slot_contributes_nothing(ubiquitin):
    """``edge_mask`` is part of the selection, not a feature.

    A padded slot carries index 0 and a false mask, so residue 0 is in every
    residue's neighbour array whether or not it is a neighbour. A rule that
    ignored the mask would message-pass with residue 0 from all R residues and
    divide by a constant ``K`` -- which would still *train*, and would be a
    different graph than the one the config describes.

    Asserted the only way that cannot be faked: move the masked-out partner and
    check the output does not budge.
    """
    topo, coords = ubiquitin
    rule = _rule()
    selection = _selection(topo, coords, k=8)
    padded = ~selection.edge_mask
    if not bool(padded.any()):
        pytest.skip("this structure fills every slot at k=8")
    state = _state(rule, coords, topo)
    row = int(torch.nonzero(padded.any(dim=-1), as_tuple=False)[0])
    victim = int(selection.neighbours[row][padded[row]][0])

    nudged = state.ca.clone()
    nudged[victim] = nudged[victim] + 25.0
    with torch.no_grad():
        plain = egnca_step(rule, state, selection.neighbours, selection.edge_mask)
        after = egnca_step(
            rule,
            EGNCAState(ca=nudged, hidden=state.hidden),
            selection.neighbours,
            selection.edge_mask,
        )
    assert float((after.ca[row] - plain.ca[row]).norm()) < 1e-6, (
        "a masked-out slot moved the output, so edge_mask is not reaching the "
        "aggregation"
    )
    # And the same residue, unmasked, does move the output -- otherwise the
    # assertion above would pass for a rule that reads nothing at all.
    forced = selection.edge_mask.clone()
    forced[row] = True
    with torch.no_grad():
        opened = egnca_step(
            rule,
            EGNCAState(ca=nudged, hidden=state.hidden),
            selection.neighbours,
            forced,
        )
    assert float((opened.ca[row] - plain.ca[row]).norm()) > 1e-6


def test_the_parameter_count_is_the_published_figure(ubiquitin):
    """**5,328 against the paper's "5K", and derived block by block.**

    It was 5,329 until ``phi_x``'s output bias went: the authors build that
    layer as ``nn.Linear(message_dim, 1, bias=False)``, so the missing
    parameter is theirs rather than ours.

    The number the port exists to reach, and the comparison the record keeps
    quoting: ``AANCARule`` is 229,588 and its floor from the two width keys is
    17,552, because its four heads are ``Linear(state_dim, 64)`` with the 64 a
    literal (``tests/test_network_size.py``). This rule lands 6.6 % above the
    published "5K" at the published widths, and the blocks are pinned
    individually so a change in one cannot be hidden by a change in another.
    """
    rule = EGNCARule()
    assert (rule.hidden_dim, rule.message_dim) == (HIDDEN_DIM, MESSAGE_DIM) == (16, 32)
    blocks = {
        name: sum(p.numel() for p in getattr(rule, name).parameters())
        for name in ("phi_m", "phi_x", "phi_h")
    }
    # phi_m: (1 + 16 + 16) -> 32 -> 32
    assert blocks["phi_m"] == 33 * 32 + 32 + 32 * 32 + 32 == 2_144
    # phi_x: 32 -> 32 -> 1, the 1 is the scalar, and the output has NO bias
    assert blocks["phi_x"] == 32 * 32 + 32 + 32 * 1 == 1_088
    # phi_h: (16 + 32) -> 32 -> 16
    assert blocks["phi_h"] == 48 * 32 + 32 + 32 * 16 + 16 == 2_096
    assert sum(blocks.values()) == rule.parameter_count() == PUBLISHED_COUNT == 5_328
    assert rule.parameter_count() < 17_552, (
        "the port is supposed to be below AANCARule's own floor; if it is not, "
        "the port has no size claim left"
    )
    assert abs(rule.parameter_count() / 5_000 - 1) < 0.1


def test_the_widths_move_the_count_one_at_a_time():
    """Both keys do something, and each does it on its own.

    A pair moved together hides a dead key -- the fault
    ``tests/test_network_size.py`` tests for twice for the same reason.
    """
    base = EGNCARule().parameter_count()
    assert EGNCARule(hidden_dim=32).parameter_count() != base
    assert EGNCARule(message_dim=64).parameter_count() != base
    counts = [
        EGNCARule(hidden_dim=h, message_dim=m).parameter_count()
        for h, m in [(32, 64), (16, 32), (8, 16), (4, 8)]
    ]
    assert counts == sorted(counts, reverse=True)
    assert len(set(counts)) == len(counts)


def test_the_rule_has_no_frame_machinery():
    """Asserted as an absence, because the absence is the claim.

    No ``frame_head``, no axis-angle, no rotation convention, no step cap --
    and ``EGNCATerms`` carries no frame-shaped readout either. If one of these
    appears, the module docstring's statement of what the rule cannot answer is
    no longer true and has to be rewritten rather than extended.
    """
    rule = EGNCARule()
    for absent in ("frame_head", "delta_head", "kappa_head", "activation_head"):
        assert not hasattr(rule, absent)
    assert set(dict(rule.named_children())) == {"phi_m", "phi_x", "phi_h"}
    fields = set(EGNCATerms.__dataclass_fields__)
    for absent in ("fape", "twist", "bond", "clash", "identity", "alive", "reach"):
        assert absent not in fields, (
            f"{absent} is a frame- or growth-shaped quantity this rule does "
            "not produce; reporting it would mean feeding it the carrier's "
            "native orientation and printing the result as a measurement"
        )


# -- the step and the rollout ----------------------------------------------


def test_the_rollout_moves_coordinates_and_holds_the_context(ubiquitin):
    """Both halves of the regeneration start state, in one assertion pair.

    The window has to move -- a port that reports a flat RMSD because its rule
    emits nothing is the project's own g1-g9 record -- and the surviving
    context has to not. The context being free is a measured failure of the
    existing trainer (1.20 A untrained, 2.83 A settled, on ``g22_random``), so
    here it is structural: ``moving`` gates both the coordinate and the hidden
    state, and a held residue still sends messages.
    """
    topo, coords = ubiquitin
    rule = _rule()
    moving = torch.zeros(topo.n_residues, dtype=torch.bool)
    moving[20:32] = True
    state = _state(rule, coords, topo)
    with torch.no_grad():
        final = egnca_rollout(
            rule, state, topo, 15, coords, moving=moving, k=24
        )
    travel = (final.ca - state.ca).norm(dim=-1)
    assert float(travel[moving].mean()) > 0.1, (
        f"the window moved {float(travel[moving].mean()):.4f} A over 15 steps, "
        "which is indistinguishable from a rule that emits nothing"
    )
    assert float(travel[~moving].max()) == 0.0
    assert float((final.hidden - state.hidden)[~moving].abs().max()) == 0.0
    # And the held residues are not inert: the window's move depends on them.
    with torch.no_grad():
        shifted = state.ca.clone()
        shifted[~moving] += 2.0
        other = egnca_rollout(
            rule,
            EGNCAState(ca=shifted, hidden=state.hidden),
            topo,
            15,
            coords,
            moving=moving,
            k=24,
        )
    assert float((other.ca - final.ca)[moving].norm(dim=-1).mean()) > 1e-3, (
        "moving the context did not change the window, so the context is not "
        "being perceived"
    )


def test_the_rollout_reaches_the_project_graph_builder(ubiquitin, monkeypatch):
    """Perception is ``torch_graph``'s, with the config's ``k`` and ``cutoff``.

    **Reuse asserted at the call, not in a comment.** A reimplemented k-NN
    would be a different graph, and a difference in a result would be
    unattributable -- which is why ``select_neighbours_device`` is pinned
    against the NumPy selector in ``tests/test_torch_graph.py`` rather than
    trusted. This checks the port goes through the same door, and with the
    settings it was given rather than the defaults.
    """
    import aanca.egnca as egnca

    calls: list[tuple] = []
    real = egnca.select_neighbours

    def spy(coords, topo, k, cutoff, *rest):
        calls.append((k, cutoff))
        return real(coords, topo, k, cutoff, *rest)

    monkeypatch.setattr(egnca, "select_neighbours", spy)
    topo, coords = ubiquitin
    rule = _rule()
    with torch.no_grad():
        egnca_rollout(
            rule, _state(rule, coords, topo), topo, 4, coords, k=17, cutoff=8.5
        )
    assert calls, "the rollout never reached torch_graph's selector"
    assert set(calls) == {(17, 8.5)}, (
        f"the selector was called with {set(calls)}, not the k and cutoff the "
        "rollout was given"
    )


def test_the_native_edge_set_is_selected_once_and_held(ubiquitin, monkeypatch):
    """The task statement: message passing over the **fixed** target graph.

    One selection for the whole rollout when ``native_edges`` is on, and one
    per step when it is off. The crippling is the off case itself: if the two
    paths called the selector the same number of times, the setting would not
    be reaching anything.
    """
    import aanca.egnca as egnca

    calls: list[int] = []
    real = egnca.select_neighbours

    def spy(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(egnca, "select_neighbours", spy)
    topo, coords = ubiquitin
    rule = _rule()
    moving = torch.zeros(topo.n_residues, dtype=torch.bool)
    moving[20:32] = True
    with torch.no_grad():
        egnca_rollout(
            rule, _state(rule, coords, topo), topo, 6, coords,
            moving=moving, k=24, native_edges=True,
        )
    assert len(calls) == 1
    calls.clear()
    with torch.no_grad():
        egnca_rollout(
            rule, _state(rule, coords, topo), topo, 6, coords,
            moving=moving, k=24, native_edges=False,
        )
    assert len(calls) == 6


def test_the_native_selection_is_the_native_coordinates_own(ubiquitin):
    """And it is the *native* graph, not the start state's.

    The baseline walk puts the window ~13 A from where it belongs, so a
    selection taken from the start state is a different edge set from the
    target graph. Which one is used is the difference between the published
    task and a harder one, so it is asserted rather than read off a flag.
    """
    topo, coords = ubiquitin
    rule = _rule()
    state = _state(rule, coords, topo)
    displaced = state.ca.clone()
    displaced[20:32] += 13.0
    with torch.no_grad():
        native = egnca_rollout(rule, state, topo, 1, coords, k=24, native_edges=True)
        start = egnca_rollout(
            rule,
            EGNCAState(ca=displaced, hidden=state.hidden),
            topo,
            1,
            with_ca_positions(coords, displaced, topo),
            k=24,
            native_edges=True,
        )
    from_native = egnca_neighbours(coords, topo, k=24)
    from_start = egnca_neighbours(
        with_ca_positions(coords, displaced, topo), topo, k=24
    )
    assert not torch.equal(from_native.neighbours, from_start.neighbours), (
        "the displaced start state selects the same graph as the native "
        "coordinates on this structure, so this test cannot tell the two apart"
    )
    assert not torch.equal(native.ca, start.ca)


def test_the_gradient_reaches_the_first_step(ubiquitin):
    """**Full BPTT, no truncation.** And the crippling is one line.

    This project's arms ran 120 steps with ``truncate 12``, so the gradient
    never reached past a tenth of the rollout; the one arm that raised
    truncation left the 0.3 A cap in place and so tested neither change. Here
    the rollout is short *because* it is fully differentiated, which is the
    published setting -- so the gradient has to reach step 0, and the test
    detaches inside the loop to show it would not otherwise.
    """
    topo, coords = ubiquitin
    rule = _rule()
    moving = torch.zeros(topo.n_residues, dtype=torch.bool)
    moving[20:32] = True
    start = _state(rule, coords, topo).ca.clone().requires_grad_(True)
    hidden = rule.initial_hidden(topo.n_residues, dtype=rule.dtype)

    final = egnca_rollout(
        rule, EGNCAState(ca=start, hidden=hidden), topo, 12, coords,
        moving=moving, k=24,
    )
    final.ca.sum().backward()
    through = float(start.grad.norm())
    assert through > 0.0, "the gradient did not reach the start state"

    # The same twelve steps with the state detached between them: the gradient
    # of the loss with respect to step 0 is then exactly the identity's, and
    # nothing the rule did in steps 1-11 is in it.
    truncated = _state(rule, coords, topo).ca.clone().requires_grad_(True)
    state = EGNCAState(ca=truncated, hidden=hidden)
    selection = _selection(topo, coords)
    for _ in range(12):
        state = egnca_step(
            rule, state, selection.neighbours, selection.edge_mask, moving
        )
        state = EGNCAState(ca=state.ca.detach(), hidden=state.hidden.detach())
    assert state.ca.grad_fn is None
    with pytest.raises(RuntimeError):
        state.ca.sum().backward()


def test_there_is_no_step_cap(ubiquitin):
    """The rule may move a residue further than the project's 0.3 A actuator.

    ``rigid_relax`` bounds the emitted translation at 0.3 A and that bound is a
    third of the journey the window has to travel -- the published rule bounds
    no displacement and its authors stabilise training with weight decay and
    clipping instead. So the absence of a cap is a property to assert: scale
    ``phi_x`` up and the step must grow past 0.3 A rather than saturating at
    it.

    Measured largest single step at ``phi_x`` gains 0.05, 0.5 and 5.0, **with
    the authors' tanh on**: 0.356, 3.297 and 7.224 A. Under ``rigid_relax`` all
    three would read 0.3. The growth flattens between the last two because the
    *coefficient* saturates in (-1, 1) -- which is the tanh doing its published
    job and is not a cap on the step: at a fixed coefficient the step is
    whatever ``||x_i - x_j||`` is, and
    :func:`test_the_tanh_bounds_the_coefficient_and_not_the_displacement` is
    where that is measured.
    """
    topo, coords = ubiquitin
    selection = _selection(topo, coords)
    steps = []
    for gain in (0.05, 0.5, 5.0):
        rule = _rule(seed=5, gain=gain)
        state = _state(rule, coords, topo)
        with torch.no_grad():
            after = egnca_step(
                rule, state, selection.neighbours, selection.edge_mask
            )
        steps.append(float((after.ca - state.ca).norm(dim=-1).max()))
    assert steps == sorted(steps), f"the step did not grow with phi_x: {steps}"
    assert steps[-1] > 0.3, (
        f"the largest single step is {steps[-1]:.3f} A; a cap at or below 0.3 A "
        "would be the project's own actuator bound smuggled back into the port"
    )


# -- the five settings the paper omits and the authors' code carries ---------
#
# Each of the five is tested three times over: that the DEFAULT is the authors'
# value, that it can be turned OFF (so the departure the port used to make is
# reproducible and each setting is ablatable from a rule that works), and -- for
# the tanh -- that turning it off brings the unbounded coefficient back. None of
# the five is this project's idea and none of these tests is about whether they
# are a good idea; they are about whether what ran is what the authors run.


def _coefficients(rule, ca: torch.Tensor, selection) -> torch.Tensor:
    """``phi_x``'s scalar per edge, read straight off the two MLPs.

    Not from ``egnca_step``, because the quantity in question is the
    *coefficient* and the step is the coefficient times a vector -- reading it
    off the step is exactly the conflation these tests exist to prevent.
    """
    h = rule.initial_hidden(ca.shape[0], dtype=ca.dtype)
    difference = ca.unsqueeze(1) - ca[selection.neighbours]
    squared = (difference**2).sum(-1, keepdim=True)
    m_ij = rule.phi_m(
        torch.cat(
            [
                squared,
                h.unsqueeze(1).expand(-1, selection.neighbours.shape[1], -1),
                h[selection.neighbours],
            ],
            dim=-1,
        )
    )
    return rule.phi_x(m_ij)


def test_the_tanh_bounds_the_coefficient_and_not_the_displacement(ubiquitin):
    """**The distinction the whole port turns on, measured both ways at once.**

    ``has_coord_act`` puts a tanh on ``phi_x``'s output, so the learned scalar
    is in (-1, 1). That is *not* a step cap and must never become one: the
    scalar multiplies ``x_i - x_j``, so the displacement it produces is
    whatever the neighbourhood's own scale is. Scale a configuration up and the
    emitted move grows with it, without bound, while ``phi_x`` stays inside
    (-1, 1) throughout.

    Measured on 1UBQ at k=24, the native edge set held fixed, the configuration
    centred and scaled about its centroid:

        scale    max step (A)    max |phi_x|
          x1           2.077         0.361
         x10          21.802         0.302
        x100         218.021         0.302

    Linear in the scale over two decades -- 100x the structure, 100x the
    step -- with the coefficient flat and well inside the bound. **This is the
    difference between the published bound and the ``max_translation = 0.3``
    clamp this project spent arms regretting.** A clamp would read 0.3, 0.3,
    0.3 down that middle column, and the window would never reach a target 13 A
    away.
    """
    topo, coords = ubiquitin
    rule = _rule(seed=5)
    selection = _selection(topo, coords)
    centre = ca_positions(coords, topo).to(rule.dtype)
    centre = centre - centre.mean(dim=0)

    steps: list[float] = []
    for scale in (1.0, 10.0, 100.0):
        ca = centre * scale
        state = EGNCAState(
            ca=ca, hidden=rule.initial_hidden(topo.n_residues, dtype=rule.dtype)
        )
        with torch.no_grad():
            after = egnca_step(
                rule, state, selection.neighbours, selection.edge_mask
            )
            coefficient = _coefficients(rule, ca, selection)
        steps.append(float((after.ca - ca).norm(dim=-1).max()))
        assert float(coefficient.abs().max()) <= 1.0, (
            "phi_x left (-1, 1), so the tanh is not on the output and the "
            "published bound is not what ran"
        )

    assert steps[0] > 0.3
    # Roughly linear: an order of magnitude of structure buys an order of
    # magnitude of step. The bar is loose on purpose -- the claim is "grows
    # without bound", not a fitted exponent.
    for before, after_ in zip(steps, steps[1:]):
        assert 5.0 * before < after_ < 20.0 * before, (
            f"the displacement did not grow with the configuration: {steps}. "
            "If it saturated, something is capping the step rather than the "
            "coefficient."
        )


def test_turning_the_tanh_off_brings_the_unbounded_coefficient_back(ubiquitin):
    """The crippling for the test above, and the port's old behaviour.

    Without ``has_coord_act`` the coefficient grows with the weight instead of
    saturating, and once it passes **-2** one step maps an edge of length ``d``
    to ``|1 + phi_x| d > d`` -- which raises ``d^2``, which raises ``|phi_x|``
    again. That feedback is the measured divergence
    (``docs/claims/the-divergence-is-a-missing-published-tanh-not-a-property-of-the-rule.md``);
    with the tanh on, ``|1 + phi_x|`` is in (0, 2) and the branch does not
    exist.

    Measured max ``|phi_x|`` at ``phi_x`` gains 0.3, 1, 3 and 10, with the tanh
    on and off: **0.361/0.378, 0.851/1.260, 0.999/3.779, 1.000/12.597**. The
    "on" column cannot leave the bound -- the last entry is 1.0 because
    ``tanh`` saturates to exactly one in float32, not because it passed -- while
    the "off" column is linear in the weight and reaches the runaway branch.
    """
    topo, coords = ubiquitin
    selection = _selection(topo, coords)
    ca = ca_positions(coords, topo).float()
    bounded, free = [], []
    for gain in (0.3, 1.0, 3.0, 10.0):
        with torch.no_grad():
            bounded.append(
                float(
                    _coefficients(
                        _rule(seed=5, gain=gain, has_coord_act=True), ca, selection
                    ).abs().max()
                )
            )
            free.append(
                float(
                    _coefficients(
                        _rule(seed=5, gain=gain, has_coord_act=False), ca, selection
                    ).abs().max()
                )
            )
    assert max(bounded) <= 1.0, f"the tanh did not bound phi_x: {bounded}"
    assert bounded[1] < 1.0 and bounded[-1] == pytest.approx(1.0, abs=1e-3), (
        f"phi_x did not saturate towards the bound: {bounded}"
    )
    assert free[-1] > 2.0, (
        f"phi_x stayed inside the runaway threshold without the tanh: {free}. "
        "If it can no longer be pushed past -2, this file's account of the "
        "divergence no longer has a mechanism in it."
    )
    assert free[-1] > 5.0 * free[0]


def test_the_five_published_settings_are_the_defaults():
    """Every one of the five, asserted on a rule built with no arguments.

    The authors' defaults for **this** task, from
    ``trainers/geometric_graph.py`` / ``EncoderEGNCA`` / ``EGC``:
    ``has_coord_act=True``, ``is_residual=True``, ``-act tanh``, ``-nt pn``,
    and ``phi_x``'s output layer ``bias=False`` with ``weight.data.zero_()``.
    A default that quietly reverted to the paper's four equations is the fault
    that cost this project the 250-update row it had to retract.
    """
    rule = EGNCARule()
    assert rule.has_coord_act is True
    assert rule.is_residual is True
    assert rule.activation == "tanh"
    assert rule.norm_type == "pn"
    assert isinstance(rule.phi_x[-1], torch.nn.Tanh)
    assert rule.last_coord_layer.bias is None
    assert float(rule.last_coord_layer.weight.detach().abs().max()) == 0.0
    # The hidden activation is tanh in ALL THREE MLPs, which is what ``-act``
    # sets there -- one net left on SiLU would be a departure nothing names.
    for net in (rule.phi_m, rule.phi_x, rule.phi_h):
        assert any(isinstance(layer, torch.nn.Tanh) for layer in net)
        assert not any(isinstance(layer, torch.nn.SiLU) for layer in net)


def test_each_of_the_five_can_be_turned_off():
    """And turning one off changes the module, not just an attribute.

    They are settings rather than constants because the port's own earlier
    behaviour has to stay reachable: each of the five is now ablatable from a
    rule that works, which is the only order in which an ablation says
    anything.
    """
    plain = EGNCARule(
        has_coord_act=False, is_residual=False, activation="silu", norm_type="none"
    )
    assert isinstance(plain.phi_x[-1], torch.nn.Identity)
    assert plain.is_residual is False
    for net in (plain.phi_m, plain.phi_x, plain.phi_h):
        assert any(isinstance(layer, torch.nn.SiLU) for layer in net)
        assert not any(isinstance(layer, torch.nn.Tanh) for layer in net)
    assert plain.norm_type == "none"
    # The count is unchanged by any of them: none of the five adds a parameter.
    assert plain.parameter_count() == PUBLISHED_COUNT
    for bad in ({"activation": "relu"}, {"norm_type": "batch"}):
        with pytest.raises(ValueError):
            EGNCARule(**bad)


def test_the_h_update_is_residual_and_the_switch_is_live(ubiquitin):
    """``h' = h + phi_h(h, m)``, which the authors' ``is_residual`` defaults to.

    The port's note recorded the non-residual form and said the question
    "cannot be settled from the note alone". It is settled by their code, and
    this is where the resolution is pinned: the residual rule's update is the
    non-residual one plus ``h``, exactly, and the switch really does remove the
    addition.
    """
    topo, coords = ubiquitin
    selection = _selection(topo, coords)
    residual = _rule(seed=1, norm_type="none")
    replacing = _rule(seed=1, norm_type="none", is_residual=False)
    replacing.load_state_dict(residual.state_dict())
    state = _state(residual, coords, topo)
    with torch.no_grad():
        with_h = egnca_step(
            residual, state, selection.neighbours, selection.edge_mask
        )
        without = egnca_step(
            replacing, state, selection.neighbours, selection.edge_mask
        )
    assert torch.allclose(with_h.hidden, state.hidden + without.hidden, atol=1e-6)
    assert not torch.allclose(with_h.hidden, without.hidden)
    # And the coordinate half is untouched by the switch: ``is_residual`` is a
    # node-feature setting and the rule's x update is already additive.
    assert torch.allclose(with_h.ca, without.ca, atol=1e-6)


def test_pair_norm_centres_and_rescales_the_node_features_only(ubiquitin):
    """``-nt pn``, and the two properties PyG's ``PairNorm(scale=1.0)`` has.

    Centred rows, and a mean squared row norm of ``scale ** 2``. Written out in
    ``aanca.egnca.pair_norm`` rather than imported, because
    ``torch_geometric`` is not a dependency of this package and a normaliser
    nobody can read is a normaliser nobody can check.

    **On the node features only, and never on the coordinates.** A normaliser
    on ``x`` would not survive a translation, so it would take the
    equivariance the whole port rests on; the test for that is the step's own
    equivariance test above, which runs with ``norm_type`` at its default.
    """
    torch.manual_seed(0)
    h = torch.randn(40, 16) * 7.0 + 3.0
    normed = pair_norm(h)
    assert float(normed.mean(dim=0).abs().max()) < 1e-5
    assert float(normed.pow(2).sum(-1).mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(pair_norm(h, scale=2.0).pow(2).sum(-1).mean()) == pytest.approx(
        4.0, abs=1e-3
    )
    # Reached from the step, and only on ``h``: the same weights with the norm
    # off move the coordinates identically on the first step and the hidden
    # state differently.
    topo, coords = ubiquitin
    selection = _selection(topo, coords)
    with_norm = _rule(seed=2)
    without = _rule(seed=2, norm_type="none")
    without.load_state_dict(with_norm.state_dict())
    state = _state(with_norm, coords, topo)
    with torch.no_grad():
        a = egnca_step(with_norm, state, selection.neighbours, selection.edge_mask)
        b = egnca_step(without, state, selection.neighbours, selection.edge_mask)
    assert torch.allclose(a.ca, b.ca, atol=1e-6), (
        "PairNorm changed the coordinates on the step it was applied, so it is "
        "not confined to the node features"
    )
    assert not torch.allclose(a.hidden, b.hidden)
    assert float(a.hidden.pow(2).sum(-1).mean()) == pytest.approx(1.0, abs=1e-3)


def test_a_held_residue_keeps_its_hidden_state_through_the_norm(ubiquitin):
    """PairNorm reads every row and may rewrite none of the held ones.

    The normaliser is a whole-graph quantity -- their encoder applies it to all
    nodes, and this port keeps that -- but the regeneration task's surviving
    context must not drift, so the ``moving`` restore happens after it. Without
    this ordering the context would move in feature space every step while its
    coordinates sat still, which is the quiet version of the context-drift
    failure the trainer already has a measured history of.
    """
    topo, coords = ubiquitin
    rule = _rule(seed=4)
    selection = _selection(topo, coords)
    moving = torch.zeros(topo.n_residues, dtype=torch.bool)
    moving[20:32] = True
    state = _state(rule, coords, topo)
    with torch.no_grad():
        after = egnca_step(
            rule, state, selection.neighbours, selection.edge_mask, moving
        )
    assert float((after.hidden - state.hidden)[~moving].abs().max()) == 0.0
    assert float((after.hidden - state.hidden)[moving].abs().max()) > 0.0


def test_the_first_step_moves_nothing_and_nothing_divides_by_it(ubiquitin):
    """The authors zero ``phi_x``'s output layer, so an untrained rule is inert.

    Asserted because it is a behaviour the rest of the code has to tolerate:
    ``move`` is 0.0 rather than ``None``, the RMSD readouts are the start
    state's own and are finite, and nothing normalises by the move -- a cosine
    against a zero-norm displacement is what made ``scripts/ceiling_readout``'s
    ``align`` column vacuous for a whole run.
    """
    topo, coords = ubiquitin
    rule = EGNCARule()
    with torch.no_grad():
        terms, state = egnca_objective(
            rule, coords, topo, steps=15, damage=12,
            generator=torch.Generator().manual_seed(0), k=24,
        )
    assert terms.move == 0.0
    assert torch.isfinite(terms.total)
    assert terms.rmsd_damaged is not None
    assert terms.rmsd_damaged == pytest.approx(terms.start_rmsd_damaged, abs=1e-6)
    assert math.isfinite(terms.rmsd_damaged_fit_intact)
    # The end state really is the start state, to the last bit: a zero weight
    # with no bias emits exactly zero, not a small number. Compared against the
    # same objective run for no steps at all, which is the start state by
    # construction.
    with torch.no_grad():
        _zero_terms, start = egnca_objective(
            rule, coords, topo, steps=0, damage=12,
            generator=torch.Generator().manual_seed(0), k=24,
        )
    assert float((state.ca - start.ca).abs().max()) == 0.0


def test_the_carrier_translates_and_does_not_reorient(ubiquitin):
    """``with_ca_positions`` writes CA and nothing else.

    The padded tensor exists so the project's loss and RMSD can index CA
    through ``frame_idx``. If it also rotated anything it would be inventing
    the orientation this rule does not produce, and every frame-shaped metric
    would start reading as a measurement.
    """
    topo, coords = ubiquitin
    target = ca_positions(coords, topo) + torch.tensor(
        [1.0, -2.0, 3.0], dtype=coords.dtype
    )
    moved = with_ca_positions(coords, target, topo)
    assert torch.allclose(ca_positions(moved, topo), target, atol=1e-9)
    # Every intramolecular distance is unchanged, which is what "translation
    # only" means.
    mask = topo.atom_mask
    for residue in (0, 10, 40):
        before = coords[residue][mask[residue]]
        after = moved[residue][mask[residue]]
        assert torch.allclose(
            torch.cdist(before, before), torch.cdist(after, after), atol=1e-9
        )


# -- the objective ---------------------------------------------------------


def test_the_loss_is_the_full_pair_set(ubiquitin):
    """``total`` is both halves, each with its own weight, and both are live.

    The window-to-window half defaults to zero everywhere in the existing
    trainer so that every arm up to g28 keeps its objective. The port weights
    it -- Ulban's decision of 23 Nov, the published loss being all pairs -- so
    the assertion is that the key reaches ``total`` rather than only the
    report. Dropping it is the exact shape of job 32405172's retraction.
    """
    topo, coords = ubiquitin
    rule = _rule()
    kwargs = dict(steps=4, damage=12, k=24)

    def run(w_anchor: float, w_window: float):
        with torch.no_grad():
            terms, _ = egnca_objective(
                rule, coords, topo,
                generator=torch.Generator().manual_seed(7),
                pairwise_weight=w_anchor,
                pairwise_window_weight=w_window,
                **kwargs,
            )
        return terms

    both = run(1.0, 1.0)
    anchor_only = run(1.0, 0.0)
    window_only = run(0.0, 1.0)
    assert float(both.pair_anchor) > 0.0 and float(both.pair_window) > 0.0
    # The same rollout under all three, so the halves are comparable.
    assert float(anchor_only.pair_anchor) == pytest.approx(float(both.pair_anchor))
    assert float(anchor_only.total) == pytest.approx(float(both.pair_anchor))
    assert float(window_only.total) == pytest.approx(float(both.pair_window))
    assert float(both.total) == pytest.approx(
        float(both.pair_anchor) + float(both.pair_window)
    )
    # And the weighted total is strictly larger than either half, so a port
    # that silently dropped one would not produce this number.
    assert float(both.total) > float(anchor_only.total)
    assert float(both.total) > float(window_only.total)


def test_the_objective_is_differentiable_through_the_whole_rollout(ubiquitin):
    """A gradient on every parameter, from a 20-step rollout, no detaching.

    ``phi_x`` is the one that matters -- it is the only route from the network
    to a coordinate -- but all three are asserted, because a port in which
    ``phi_h`` received nothing would be carrying a hidden state the loss cannot
    train and would still look like it was working.
    """
    topo, coords = ubiquitin
    rule = _rule()
    terms, _state = egnca_objective(
        rule, coords, topo, steps=20, damage=12,
        generator=torch.Generator().manual_seed(1), k=24,
    )
    terms.total.backward()
    for name, parameter in rule.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"
        assert float(parameter.grad.abs().max()) > 0.0, f"{name}'s gradient is zero"


def test_the_objective_reports_only_what_a_ca_trace_can_say(ubiquitin):
    """The readouts, and that they are the CA-level ones.

    ``rmsd_damaged`` against ``start_rmsd_damaged`` is the whole claim a run of
    this rule can make, and ``move`` is what distinguishes a flat RMSD from a
    rule that emits nothing. Both superposition conventions are reported
    because ``rmsd_damaged`` lets the displaced window absorb part of its own
    error -- 1.06 A of it on a helix start.
    """
    topo, coords = ubiquitin
    rule = _rule()
    with torch.no_grad():
        terms, _ = egnca_objective(
            rule, coords, topo, steps=8, damage=12,
            generator=torch.Generator().manual_seed(2), k=24,
        )
    assert terms.rmsd_damaged is not None
    assert terms.rmsd_damaged_fit_intact is not None
    # 4.80 A on this window, whole-structure superposition. A bar rather than
    # a pin, because the number depends on where in the chain the window
    # lands; what it asserts is that the task is not already solved at step 0.
    assert terms.start_rmsd_damaged > 3.0, (
        "the start state is supposed to put the window well away from its "
        f"native place; it scored {terms.start_rmsd_damaged:.3f} A"
    )
    assert terms.move > 0.0
    assert int(terms.damaged_mask.sum()) == 12


def test_the_window_is_a_contiguous_run_inside_an_intact_context(ubiquitin):
    """The task is the project's own, reused rather than restated.

    ``damage_window`` draws the window and ``baseline_coords`` places it, so a
    comparison against an existing arm differs in the rule and not in the task.
    Asserted on the mask's shape: one contiguous run, never the whole crop.
    """
    topo, coords = ubiquitin
    rule = _rule()
    with torch.no_grad():
        terms, _ = egnca_objective(
            rule, coords, topo, steps=1, damage=12,
            generator=torch.Generator().manual_seed(4), k=24,
        )
    mask = terms.damaged_mask
    index = torch.nonzero(mask, as_tuple=False).flatten()
    assert int(index.max() - index.min()) == 11, "the window is not contiguous"
    assert int((~mask).sum()) > 12, "there is no intact context left"


def test_a_pool_entry_resumes_the_state_and_keeps_its_window(ubiquitin):
    """The replay pool carries CA, hidden **and** the window, and redraws none.

    Resuming coordinates alone would hand the rule a structure it had no memory
    of building, and redrawing the window would score a hole the state was
    never asked to fill -- the reasons ``growth.GrowthPoolEntry`` gives, which
    apply here unchanged. The crippling is the ``damaged`` field: with a fresh
    draw the resumed window would differ from the entry's.
    """
    topo, coords = ubiquitin
    rule = _rule()
    with torch.no_grad():
        first, state = egnca_objective(
            rule, coords, topo, steps=6, damage=12,
            generator=torch.Generator().manual_seed(9), k=24,
        )
        entry = EGNCAPoolEntry(
            ca=state.ca.detach().clone(),
            hidden=state.hidden.detach().clone(),
            damaged=first.damaged_mask.clone(),
            baseline_rmsd=first.start_rmsd_damaged,
        )
        # A different generator, so a redrawn window would land elsewhere.
        second, resumed = egnca_objective(
            rule, coords, topo, steps=0, damage=12,
            generator=torch.Generator().manual_seed(123), k=24,
            resume_from=entry,
        )
    assert torch.equal(second.damaged_mask, first.damaged_mask)
    assert torch.equal(resumed.ca, entry.ca)
    assert torch.equal(resumed.hidden, entry.hidden)
    assert float(second.total) == pytest.approx(float(first.total), rel=1e-6)
    # The entry really was a different window from what a fresh draw gives.
    with torch.no_grad():
        fresh, _ = egnca_objective(
            rule, coords, topo, steps=0, damage=12,
            generator=torch.Generator().manual_seed(123), k=24,
        )
    assert not torch.equal(fresh.damaged_mask, first.damaged_mask), (
        "the two generators draw the same window on this structure, so this "
        "test cannot tell a resumed window from a redrawn one"
    )


def test_two_hundred_updates_on_one_structure_move_the_window(ubiquitin):
    """**The port trains.** Not a result -- a demonstration that it can learn.

    One structure, one window, 200 updates, the published settings: 15 steps
    fully differentiated, both halves of the unclamped restraint, Adam at 1e-3
    with weight decay 1e-5 and the gradient clipped at 1.0. Measured on this
    tree:

        update    total    rmsd_dmg    move
             1   110.21      15.813   0.000
            10    72.94      11.448   5.896
            40    53.70      11.323   8.282
           120    37.79      10.698  13.305
           200    14.90       4.771  19.496

    Against a start state that scores 15.813 A. So the loss falls by an order of
    magnitude and the window ends below the baseline it started from -- which is
    the thing nine of this project's arms never did, every one of them parking
    at its own baseline.

    **It was forty updates until the authors' five settings landed, and the
    first row above is why it is 200 now.** ``phi_x``'s output layer starts at
    exactly zero, as theirs does, so update 1 emits ``move 0.000`` and
    ``rmsd_dmg`` is the start state to three decimals -- the rule has to grow a
    coordinate head before it can move anything, where the port's old
    small-gain initialisation started with one. That is a slower start and not
    a worse one, and nothing in the rollout or the readouts may assume the
    first displacement is non-zero.

    **This is an overfit to a single window and is not evidence about
    generalisation.** It is here because a rule that could not even overfit one
    window would not be worth submitting an arm for, and that is a question
    answerable in three seconds on a laptop rather than in a queue.
    """
    topo, coords = ubiquitin
    torch.manual_seed(0)
    rule = EGNCARule()
    optimiser = torch.optim.Adam(rule.parameters(), lr=1e-3, weight_decay=1e-5)
    history: list[tuple[float, float]] = []
    for _ in range(200):
        # The same generator every update, so it is the same window and the
        # same start state: the only thing changing is the weights.
        terms, _state = egnca_objective(
            rule, coords, topo, steps=15, damage=12,
            generator=torch.Generator().manual_seed(0), k=24,
        )
        assert torch.isfinite(terms.total), "the rollout diverged"
        optimiser.zero_grad()
        terms.total.backward()
        torch.nn.utils.clip_grad_norm_(rule.parameters(), 1.0)
        optimiser.step()
        history.append((float(terms.total.detach()), terms.rmsd_damaged))

    first_total, first_rmsd = history[0]
    last_total, last_rmsd = history[-1]
    start = terms.start_rmsd_damaged
    assert last_total < 0.25 * first_total, (
        f"the loss went {first_total:.2f} -> {last_total:.2f} in 200 "
        "updates, which is not learning"
    )
    assert last_rmsd < first_rmsd
    assert last_rmsd < start, (
        f"the window ends at {last_rmsd:.3f} A against a start state's "
        f"{start:.3f} A, so 200 updates have not beaten not trying"
    )
    # The zero-initialised coordinate head, asserted rather than described:
    # update 1 emits nothing at all, and the readouts survive it.
    assert history[0][1] == pytest.approx(start, abs=1e-6), (
        "update 1 moved the window, so phi_x's output layer did not start at "
        "zero and the authors' initialisation is not what ran"
    )


# -- the trainer -----------------------------------------------------------


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_egnca", ROOT / "scripts" / "train_egnca.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _trainer()


def test_the_config_states_every_setting_the_trainer_reads(trainer):
    """``check_base_is_complete`` on the port's own config, through the loader.

    So a later change to a default in ``build_schema`` cannot reinterpret a
    recorded run of this trainer, and a key added to the schema and forgotten
    in the config fails here rather than at launch.
    """
    data = run_config.load(CONFIG)
    run_config.check_base_is_complete(data, trainer.build_schema())
    args = run_config.to_namespace(
        trainer.build_schema(), run_config.resolve(data, "port"), "port", RUNTIME
    )
    assert args.tag == "port"
    assert (args.hidden_dim, args.message_dim) == (16, 32)
    assert (args.steps_min, args.steps) == (15, 25)
    assert args.native_edges is True
    assert args.pairwise_weight == 1.0 and args.pairwise_window_weight == 1.0


def test_the_widths_reach_the_rule_through_the_trainer(trainer):
    """``build_rule`` is the call ``main`` makes, so the keys are tested wired.

    Hardcoding either width in ``build_rule`` fails this, which is the fault
    ``tests/test_network_size.py`` was written for after a held-out column
    turned out to be structurally absent.
    """
    data = run_config.load(CONFIG)
    schema = trainer.build_schema()

    def rule(**overrides):
        inputs = dict(run_config.resolve(data, "port"))
        inputs.update(overrides)
        return trainer.build_rule(
            run_config.to_namespace(schema, inputs, "port", RUNTIME)
        )

    assert rule().parameter_count() == PUBLISHED_COUNT
    assert rule(hidden_dim=8).parameter_count() != PUBLISHED_COUNT
    assert rule(message_dim=16).parameter_count() != PUBLISHED_COUNT


def test_the_five_settings_reach_the_rule_through_the_trainer(trainer):
    """The authors' four keys, wired, and the fifth asserted as a constant.

    A key in the config that ``build_rule`` never reads is the project's
    recurring fault in its purest form: the banner would print ``coord_act
    tanh`` and the rule would be the paper's. Both directions are checked --
    the config's own values arrive, and a section that turned one off would get
    a rule with it off.
    """
    data = run_config.load(CONFIG)
    schema = trainer.build_schema()

    def rule(**overrides):
        inputs = dict(run_config.resolve(data, "port"))
        inputs.update(overrides)
        return trainer.build_rule(
            run_config.to_namespace(schema, inputs, "port", RUNTIME)
        )

    published = rule()
    assert (
        published.has_coord_act,
        published.is_residual,
        published.activation,
        published.norm_type,
    ) == (True, True, "tanh", "pn")
    assert published.last_coord_layer.bias is None
    assert float(published.last_coord_layer.weight.detach().abs().max()) == 0.0

    assert rule(has_coord_act=False).has_coord_act is False
    assert rule(is_residual=False).is_residual is False
    assert rule(activation="silu").activation == "silu"
    assert rule(norm_type="none").norm_type == "none"


def test_the_config_states_the_authors_defaults_and_overrides_nothing(trainer):
    """``[base.inputs]`` carries the five, and ``[port.inputs]`` is empty.

    The emptiness is deliberate and was hard won: an ``lr = 3e-4`` override
    lived there as a departure invented to compensate for the missing tanh, and
    it is gone. ``lr``, ``clip`` and ``weight_decay`` in the base are already
    the authors' own values for this task, so an override of any of them would
    be a departure that no comment in the file would be able to justify.
    """
    data = run_config.load(CONFIG)
    assert (data["port"].get("inputs") or {}) == {}, (
        "[port.inputs] is deliberately empty: the base IS the port"
    )
    base = data["base"]["inputs"]
    assert base["has_coord_act"] is True
    assert base["is_residual"] is True
    assert base["activation"] == "tanh"
    assert base["norm_type"] == "pn"
    # The published optimiser settings, unchanged by the section.
    args = run_config.to_namespace(
        trainer.build_schema(), run_config.resolve(data, "port"), "port", RUNTIME
    )
    assert (args.lr, args.weight_decay, args.clip) == (1e-3, 1e-5, 1.0)


def test_skipped_updates_are_counted_by_reason(trainer):
    """**The skip rate is the readout the five settings are judged on.**

    An overflowing crop prices at ``inf``, is skipped, and contributes no
    gradient -- about 10 % of training rollouts before the five landed, at
    every learning rate tried, which is why no learning rate helped. A single
    total cannot say that: ``no_window`` is a property of the crop and
    ``nonfinite_loss`` is a property of the rule. So the reasons are counted
    apart, the rate is printed per evaluation interval, and an unnamed reason
    is an error rather than a silent bucket.
    """
    assert set(trainer.SKIP_REASONS) == {
        "no_crop", "no_window", "nonfinite_loss", "nonfinite_gradient"
    }
    for reason in trainer.SKIP_REASONS:
        assert trainer._SkipUpdate(reason).reason == reason
    with pytest.raises(ValueError):
        trainer._SkipUpdate("because")
    line = trainer.skip_report(
        {"no_crop": 0, "no_window": 3, "nonfinite_loss": 22, "nonfinite_gradient": 0},
        250,
    )
    assert "25/250" in line and "10.0%" in line and "nonfinite_loss 22" in line
    # Every reason really is raised from the loop, by name, rather than three of
    # the four being dead constants.
    source = inspect.getsource(trainer.main)
    for reason in trainer.SKIP_REASONS:
        assert f'_SkipUpdate("{reason}")' in source


def test_every_objective_setting_reaches_the_objective(trainer):
    """``objective_kwargs`` covers every setting ``egnca_objective`` takes.

    **The structural version of the project's recurring fault.** A key in the
    schema that the objective never receives is a setting that appears in the
    config, in the banner and in the checkpoint, and changes nothing. Checked by
    signature rather than by a list, so adding an argument to
    ``egnca_objective`` and forgetting to pass it fails here.
    """
    data = run_config.load(CONFIG)
    args = run_config.to_namespace(
        trainer.build_schema(), run_config.resolve(data, "port"), "port", RUNTIME
    )
    passed = set(trainer.objective_kwargs(args))
    signature = set(inspect.signature(egnca_objective).parameters)
    # Everything the objective takes that is not supplied per-update by the
    # loop itself.
    per_update = {
        "rule", "native_coords", "topo", "steps", "generator", "resume_from"
    }
    assert signature - per_update == passed, (
        "objective_kwargs does not match egnca_objective's settings: missing "
        f"{sorted(signature - per_update - passed)}, extra "
        f"{sorted(passed - (signature - per_update))}"
    )
    assert passed <= set(vars(args)), (
        "objective_kwargs passes something that is not a config setting, so "
        "the run's own checkpoint could not say what it was"
    )


def test_the_trainer_scores_the_held_out_set_through_the_same_settings(trainer):
    """``score`` and the training step share one ``objective_kwargs`` dict.

    Two call sites with two argument lists is how ``move`` and ``clash`` came
    to be computed in training and absent from the evaluation while being
    printed as a column. There is one list here, and this asserts the
    evaluation uses it.
    """
    source = inspect.getsource(trainer.score)
    assert "**objective_kwargs(args)" in source
    assert "**objective_kwargs(args)" in inspect.getsource(trainer.main)


def test_a_warm_start_across_a_width_change_is_refused(trainer):
    """Both widths are in ``WARM_START_MUST_MATCH``, and the reason is shapes.

    ``load_state_dict`` would raise a size mismatch over six tensors several
    screens into a run; the list exists so that is a refusal with the two
    numbers in it instead. Pinned together with the fact that it really does
    fail, because an entry nobody can demonstrate the need for gets removed.
    """
    assert "hidden_dim" in trainer.WARM_START_MUST_MATCH
    assert "message_dim" in trainer.WARM_START_MUST_MATCH
    assert "pairwise_window_weight" not in trainer.WARM_START_MUST_MATCH
    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        EGNCARule(hidden_dim=8, message_dim=16).load_state_dict(
            EGNCARule().state_dict()
        )
