"""The emitted move must be expressed in the residue's own frame.

**This is the bug that explains the whole g1-g9 record, found on 7 Nov after
seven arms and four dead hypotheses.** The perception features are invariant by
construction, so the network's output is unchanged by a global rotation of the
structure -- measured at 1.0e-08 for the rotation half and 1.2e-08 for the
translation half, which is numerical noise. And `rigid_relax` applied that
output in the **lab frame**:

    moved = (coords - ca) @ rot.T + ca + translation

Those two facts are incompatible. A lab-frame direction is exactly what an
invariant encoder cannot represent, so the rule could not express "move toward
my target" however it was trained, and no objective could teach it. It explains
every observation in the record:

  * teacher-forced, the emitted translation is anti-correlated with the ideal
    one (cosine -0.259 over 12 crops, spread straddling zero);
  * two arms with no placement term in common park at the same 12.23 A, because
    the move is a near-constant vector applied 120 times;
  * the emitted step length is nearly identical on every crop
    (0.0302-0.0315 A), which is what a constant looks like;
  * four reconstruction arms froze at their own baselines;
  * `g6_noid`, with no identity gradient, asked for 0.3 % of its budget -- the
    best constant vector being zero;
  * and FAPE "supplied a magnitude and not a bearing", which is now mechanical:
    a distance is invariant and learnable, a direction in the lab frame is not
    representable.

The fix makes the output local: interpret the translation in the residue's own
N-CA-C frame, and compose the rotation on the right. `--equivariant-move`
selects it, defaulting off so every earlier arm stays reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from aanca.growth import GrowthConfig, growth_links, rigid_relax
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, step_once
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

NATIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "1UBQ.pdb"


def rotation(degrees: float, dtype) -> torch.Tensor:
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


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords, growth_links(topo)


def _emitted(rule, topo, structure, alive, config):
    moves: list = []
    state = initial_state(structure, rule, alive=alive)
    with torch.no_grad():
        new, _mu, _kappa = step_once(
            rule, state, topo, config, alive,
            generator=torch.Generator().manual_seed(0), moves=moves,
        )
    axis_angle, translation, _before, _moving = moves[0]
    return axis_angle.clone(), translation.clone(), new.coords.clone()


def _perturbed_rule():
    rule = AANCARule(growth=True, birth="none", identity=False)
    torch.manual_seed(0)
    # Zero-initialised output, so without a perturbation every emitted move is
    # exactly zero and the question cannot be asked at all.
    for parameter in rule.frame_head[-1].parameters():
        parameter.data.add_(torch.randn_like(parameter) * 0.05)
    rule.eval()
    return rule


def test_the_network_output_cannot_see_the_global_orientation(ubiquitin):
    """The premise: perception is invariant, so the output is too.

    Pinned because the bug is only a bug in combination with a lab-frame
    application, and someone changing the features to be equivariant would make
    this test fail for a good reason -- at which point the lab-frame
    application becomes correct and this file needs rereading rather than
    patching.
    """
    topo, coords, _links = ubiquitin
    rule = _perturbed_rule()
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    config = RolloutConfig(
        steps=1, truncate=12, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )
    turn = rotation(40.0, coords.dtype)

    aa_plain, t_plain, _ = _emitted(rule, topo, coords, alive, config)
    aa_turned, t_turned, _ = _emitted(rule, topo, coords @ turn.T, alive, config)

    assert float(aa_plain.norm(dim=-1).mean()) > 1e-3, "nothing was emitted"
    assert float(t_plain.norm(dim=-1).mean()) > 1e-3, "nothing was emitted"
    assert float((aa_turned - aa_plain).norm(dim=-1).mean()) < 1e-6
    assert float((t_turned - t_plain).norm(dim=-1).mean()) < 1e-6


def test_a_lab_frame_translation_is_not_equivariant(ubiquitin):
    """`rigid_relax` as it stood: rotating the input does not rotate the output.

    The number is the point. The equivariance error is 0.36 A per step against
    a move of 1.32 A -- a third of the move, every step, compounding over the
    120 a rollout takes.
    """
    topo, coords, _links = ubiquitin
    n = topo.n_residues
    moving = torch.ones(n, dtype=torch.bool)
    turn = rotation(40.0, coords.dtype)

    torch.manual_seed(1)
    axis_angle = torch.randn(n, 3) * 0.05
    translation = torch.randn(n, 3) * 0.1

    plain = rigid_relax(coords, axis_angle, translation, topo, moving)
    turned = rigid_relax(coords @ turn.T, axis_angle, translation, topo, moving)

    gap = (turned - plain @ turn.T).norm(dim=-1)
    move = (plain - coords).norm(dim=-1)
    assert float(gap.mean()) > 0.1 * float(move.mean()), (
        "this test exists to record that the move is NOT equivariant; if it "
        "now is, rigid_relax changed and the docstring above needs rereading"
    )


def test_the_equivariant_form_rotates_with_the_structure(ubiquitin):
    """The fix: interpret the output in the residue's own frame.

    Translation becomes ``R_i @ t_i`` and the rotation composes on the right,
    so a globally rotated input produces a globally rotated output and the same
    local neighbourhood always produces the same local move.
    """
    topo, coords, _links = ubiquitin
    n = topo.n_residues
    moving = torch.ones(n, dtype=torch.bool)
    turn = rotation(40.0, coords.dtype)

    torch.manual_seed(1)
    axis_angle = torch.randn(n, 3) * 0.05
    translation = torch.randn(n, 3) * 0.1

    plain = rigid_relax(
        coords, axis_angle, translation, topo, moving, local=True
    )
    turned = rigid_relax(
        coords @ turn.T, axis_angle, translation, topo, moving, local=True
    )

    gap = (turned - plain @ turn.T).norm(dim=-1)
    move = (plain - coords).norm(dim=-1)
    assert float(move.mean()) > 1e-3, "nothing moved, so nothing is tested"
    assert float(gap.mean()) < 1e-6, (
        f"equivariance error {float(gap.mean()):.2e} A on a "
        f"{float(move.mean()):.3f} A move"
    )


def test_the_local_form_still_caps_the_move(ubiquitin):
    """Rotating the translation must not change its length.

    A rotation preserves norm, so the 0.3 A cap `_bounded` applies to the
    network's output still bounds the move after the frame change. Checked
    because a fix that quietly widened the step budget would confound every
    comparison against an earlier arm.
    """
    topo, coords, _links = ubiquitin
    n = topo.n_residues
    moving = torch.ones(n, dtype=torch.bool)
    torch.manual_seed(2)
    axis_angle = torch.zeros(n, 3)
    translation = torch.randn(n, 3)
    translation = translation / translation.norm(dim=-1, keepdim=True) * 0.3

    lab = rigid_relax(coords, axis_angle, translation, topo, moving)
    local = rigid_relax(
        coords, axis_angle, translation, topo, moving, local=True
    )
    for moved in (lab, local):
        ca = moved[torch.arange(n), topo.frame_idx[torch.arange(n), 1]]
        ca0 = coords[torch.arange(n), topo.frame_idx[torch.arange(n), 1]]
        step = (ca - ca0).norm(dim=-1)
        assert float(step.max()) == pytest.approx(0.3, abs=1e-6)
