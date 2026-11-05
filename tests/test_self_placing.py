"""Self-placing nodes: every slot alive from step 0, nobody's child.

Under ``birth="torsion"``, ``"cartesian"`` and ``"free"`` a residue's pose is
``rot_parent @ rot_reference @ rot_spin`` -- inherited through a chain of
parents, so an orientation error at one birth rotates everything grown through
it. g5 varied all three and added a direct geodesic charge on the bond rotation
and got the same answer four times: a rule collapsed onto its helical
initialisation whose every departure from that one conformation made the
rotation error worse.

``birth="none"`` removes the inheritance rather than one of its terms. Every
slot is alive from step 0, placed by a fixed reference walk off one native
anchor per covalent run, and each residue then moves itself with the frame head
reading its own neighbourhood. These tests pin what that has to mean: that no
birth machinery is on the path, that the baseline is a real conformation and not
the answer, that the information budget matches a grown run's, and that the loss
reaches the one head still doing work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.backbone import backbone_atoms
from aanca.fape import backbone_rmsd
from aanca.growth import (
    BASELINE_MODES,
    GrowthConfig,
    baseline_coords,
    count_runs,
    give_birth,
    growth_links,
    growth_objective,
    peptide_bond_restraint,
    run_anchors,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def ubiquitin():
    path = DATA / "1UBQ.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    topo, coords = feature_topology(load_protein(path))
    return topo, coords, growth_links(topo)


def _anchors(topo, links, seed=0):
    return run_anchors(topo.n_residues, links, torch.Generator().manual_seed(seed))


def test_no_birth_head_exists_and_the_rule_is_free_placements_size():
    """Absent, not frozen, and provably the same architecture as ``free``.

    ``free`` and ``none`` both lack the birth head, so an identical parameter
    count is the check that ``none`` removed the *inheritance* and not some
    additional capacity -- otherwise a difference between the two arms could be
    a difference in model size.
    """
    rule = AANCARule(growth=True, birth="none")
    assert not hasattr(rule, "birth_head")
    assert rule.parameter_count() == AANCARule(
        growth=True, birth="free"
    ).parameter_count()
    assert rule.parameter_count() < AANCARule(
        growth=True, birth="cartesian"
    ).parameter_count()

    node = torch.randn(8, rule.node_dim)
    hidden = torch.randn(8, rule.hidden_dim)
    _activation, birth, _axis, _translation = rule.growth_outputs(node, hidden)
    assert birth is None


def test_give_birth_under_none_says_what_went_wrong():
    """Reaching it at all is a bug in the rollout, so the message says so.

    ``'unknown birth mode'`` would be wrong twice over: the mode is known, and
    the caller's problem is not the mode but that a frontier was built in a
    rollout that cannot have one.
    """
    rule_free = AANCARule(growth=True, birth="none")
    assert rule_free.birth == "none"
    topo, coords = feature_topology(load_protein(DATA / "1UBQ.pdb"))
    links = growth_links(topo)
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[10] = True
    decided = torch.zeros(topo.n_residues)
    decided[10] = 1.0
    with pytest.raises(ValueError, match="no births to give"):
        give_birth(
            coords, topo, alive, links, torch.zeros(topo.n_residues), decided,
            None, birth_mode="none",
        )


@pytest.mark.parametrize("mode", BASELINE_MODES)
def test_the_baseline_has_ideal_peptide_geometry(ubiquitin, mode):
    """Every link at 3.804 A, whatever the baseline is.

    The reason the walk reuses the Cartesian birth's construction rather than
    scattering frames: a baseline whose bonds were already broken would make
    ``peptide_bond_restraint`` the whole problem from step 0, and a run that
    then failed would say nothing about placement. The crystal's own value is
    ~0.002, so the baselines have to be close to it and not merely finite.
    """
    topo, coords, links = ubiquitin
    anchors = _anchors(topo, links)
    start = baseline_coords(coords, topo, links, anchors, mode)

    ca = backbone_atoms(start, topo)[:, 1]
    index = torch.nonzero(links, as_tuple=False).flatten()
    spans = (ca[index + 1] - ca[index]).norm(dim=-1)
    assert torch.allclose(
        spans, torch.full_like(spans, 3.804), atol=5e-3
    ), f"{mode}: CA-CA from {float(spans.min()):.3f} to {float(spans.max()):.3f}"

    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    bond = float(peptide_bond_restraint(start, topo, alive, links))
    assert bond < 0.05, f"{mode}: bond term {bond:.4f}"


@pytest.mark.parametrize("mode", BASELINE_MODES)
def test_the_baseline_is_not_the_answer(ubiquitin, mode):
    """A self-placing rollout must not start at the structure it is building.

    The growth rollout gets this for free: ``initial_state`` parks every
    dormant slot, so the native structure is present only for the seeds. Here
    every slot is alive, so nothing parks anything and the guarantee has to
    come from the baseline itself.
    """
    topo, coords, links = ubiquitin
    anchors = _anchors(topo, links)
    start = baseline_coords(coords, topo, links, anchors, mode)
    alive = torch.ones(topo.n_residues, dtype=torch.bool)
    rmsd = float(backbone_rmsd(start, coords, topo, alive))
    assert rmsd > 5.0, f"{mode}: baseline is {rmsd:.2f} A from native"
    assert not start.requires_grad


def test_the_anchor_keeps_its_crystal_pose_and_nothing_else_does(ubiquitin):
    """One native frame per run: the same budget a grown run's seed gets.

    If a second residue kept its crystal pose the arm would be handed a second
    anchor and the comparison with growth would be unfair in its favour. The
    walk reaches every other residue in the run, so this also checks the walk
    did not stop early.
    """
    topo, coords, links = ubiquitin
    anchors = _anchors(topo, links)
    assert int(anchors.sum()) == count_runs(links)

    start = baseline_coords(coords, topo, links, anchors, "helix")
    moved = (start - coords).norm(dim=-1).amax(dim=-1)
    assert float(moved[anchors].max()) == pytest.approx(0.0, abs=1e-9)
    assert float(moved[~anchors].min()) > 0.1


def test_a_chain_break_is_walked_as_its_own_run(ubiquitin):
    """A run is a separate rigid body, so each one needs its own anchor.

    Growth cannot cross a chain break and neither can this walk. With a break
    inserted by hand, the two runs have to be built independently -- which
    shows up as the residues on either side of the break no longer sitting
    3.804 A apart, because nothing placed one from the other.
    """
    topo, coords, _links = ubiquitin
    links = growth_links(topo).clone()
    cut = topo.n_residues // 2
    links[cut] = False
    assert count_runs(links) == 2

    anchors = torch.zeros(topo.n_residues, dtype=torch.bool)
    anchors[0] = True
    anchors[cut + 1] = True
    start = baseline_coords(coords, topo, links, anchors, "helix")

    ca = backbone_atoms(start, topo)[:, 1]
    index = torch.nonzero(links, as_tuple=False).flatten()
    spans = (ca[index + 1] - ca[index]).norm(dim=-1)
    assert torch.allclose(spans, torch.full_like(spans, 3.804), atol=5e-3)
    # And the break itself is unconstrained, which is the point: residue
    # ``cut`` is the far end of the first run's walk and ``cut + 1`` is the
    # second run's anchor, so nothing placed either from the other and the gap
    # between them is wherever the two walks happened to end up. A gap near
    # 3.804 here would mean the walk had crossed the break.
    gap = float((ca[cut + 1] - ca[cut]).norm())
    assert abs(gap - 3.804) > 1.0, f"the walk crossed the chain break: {gap:.2f} A"


def test_the_random_baseline_depends_on_the_seed_and_the_others_do_not(ubiquitin):
    """Reproducible from the generator, like every other random choice here.

    The fixed baselines must be *identical* across seeds, or two arms that
    differ only in their training seed would also differ in where they started.
    """
    topo, coords, links = ubiquitin
    anchors = _anchors(topo, links)
    for mode in ("helix", "extended"):
        a = baseline_coords(
            coords, topo, links, anchors, mode, torch.Generator().manual_seed(0)
        )
        b = baseline_coords(
            coords, topo, links, anchors, mode, torch.Generator().manual_seed(7)
        )
        assert torch.allclose(a, b, atol=1e-12), mode

    a = baseline_coords(
        coords, topo, links, anchors, "random", torch.Generator().manual_seed(0)
    )
    b = baseline_coords(
        coords, topo, links, anchors, "random", torch.Generator().manual_seed(7)
    )
    again = baseline_coords(
        coords, topo, links, anchors, "random", torch.Generator().manual_seed(0)
    )
    assert not torch.allclose(a, b, atol=1e-3)
    assert torch.allclose(a, again, atol=1e-12)


def test_an_unknown_baseline_is_refused(ubiquitin):
    topo, coords, links = ubiquitin
    with pytest.raises(ValueError, match="unknown baseline mode"):
        baseline_coords(coords, topo, links, _anchors(topo, links), "spiral")


def test_a_self_placing_rollout_grows_nothing_and_trains_the_frame_head(ubiquitin):
    """The whole mechanism, end to end, and what the loss may reach.

    Three claims at once because they are one situation: every slot stays alive
    (nothing was born and nothing died), the two terms that exist to make
    growth grow are exactly zero, and the frame head -- the only head left with
    a job -- takes gradient.

    The frame head is perturbed first because every output layer in this
    project is zero-initialised, so an untrained rule emits exactly zero and
    the gradient into the node features is zero with it. That is an
    initialisation artefact rather than a dead path, and the same perturbation
    is used in ``test_free_placement`` for the same reason.
    """
    topo, coords, links = ubiquitin
    rule = AANCARule(growth=True, birth="none")
    torch.manual_seed(0)
    for p in rule.frame_head[-1].parameters():
        p.data.add_(torch.randn_like(p) * 0.01)

    config = RolloutConfig(
        steps=8, truncate=99, k=24,
        growth=GrowthConfig(birth="none", initial="helix"),
    )
    terms, result = growth_objective(
        rule, coords, topo, config, torch.Generator().manual_seed(0)
    )

    assert bool(result.state.alive.all())
    assert terms.alive_fraction == pytest.approx(1.0)
    assert terms.segments == count_runs(links)
    assert float(terms.completion) == 0.0
    assert float(terms.reach) == 0.0
    assert terms.frontier == 0
    assert torch.isfinite(terms.total)

    terms.total.backward()
    frame = sum(
        float(p.grad.abs().sum())
        for p in rule.frame_head.parameters()
        if p.grad is not None
    )
    assert frame > 0.0, "the frame head is not being trained"

    # The activation head is kept so a checkpoint from a grown run can be warm
    # started here, and it is asserted dead rather than quietly carried: there
    # is nothing for it to decide, so a non-zero gradient would mean some birth
    # path is still live.
    activation = sum(
        float(p.grad.abs().sum())
        for p in rule.activation_head.parameters()
        if p.grad is not None
    )
    assert activation == 0.0, "the activation head takes gradient it has no job for"


def test_the_rollout_actually_moves_the_baseline(ubiquitin):
    """A rule with something in its frame head must not end where it started.

    The counterpart to ``test_the_baseline_is_not_the_answer``: that one says
    the start is not the target, this one says the start is not a fixed point
    of the step. Between them a run that reports no change is reporting a real
    result rather than a disconnected head.
    """
    topo, coords, links = ubiquitin
    rule = AANCARule(growth=True, birth="none")
    torch.manual_seed(0)
    for p in rule.frame_head[-1].parameters():
        p.data.add_(torch.randn_like(p) * 0.5)

    config = RolloutConfig(
        steps=8, truncate=99, k=24,
        growth=GrowthConfig(birth="none", initial="helix"),
    )
    _terms, result = growth_objective(
        rule, coords, topo, config, torch.Generator().manual_seed(0)
    )
    anchors = _anchors(topo, links)
    start = baseline_coords(
        coords, topo, links, anchors, "helix", torch.Generator().manual_seed(0)
    )
    moved = (result.state.coords.detach() - start).norm(dim=-1).max()
    assert float(moved) > 0.1
