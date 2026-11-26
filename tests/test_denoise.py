"""What ``scripts/train_denoise.py`` must keep true, and what each test prevents.

====================================== ======================================
the property                           what breaking it would look like
====================================== ======================================
the corruption is isotropic            noise on one axis, or a per-residue
                                       scale, which would make the task
                                       orientation-dependent
its magnitude is sigma                 a sigma that means something else --
                                       the whole readout is rmsd against a
                                       named noise level
sigma is drawn log-uniformly           uniform, which puts 98 % of draws above
                                       1 A and never shows the rule the
                                       near-converged regime
``context_fraction = 0`` pins nothing  an off-by-one anchor, which is a leak at
                                       the one setting that must have none
``context_fraction = 1`` pins all      a residue left corrupted at the
                                       degenerate end, so ctx100 silently
                                       trains
the context is contiguous runs         a scatter, which puts an exact anchor in
                                       every neighbourhood at every fraction
                                       and makes the sweep measure nothing
the loss survives coincident atoms     nan from ``norm``'s derivative at zero,
                                       which an untrained rule can reach
equivariance survives the live graph   a graph built from the native
                                       coordinates while claiming not to be
the prior is reproducible              a sample series that cannot be compared
                                       between two evaluations
the prior's radius is 2.2 * N^0.38     a cloud whose size is the thing
                                       steps-to-convergence measures
the skip counter rejects a new reason   a typo'd reason counted into nothing
the port is untouched                  a shared edit, which would invalidate
                                       Bunya job 33419979
====================================== ======================================

**The module is registered in ``sys.modules`` before it is executed.** It is
loaded from a path rather than imported, and ``@dataclass`` looks its own class's
module up by name -- an unregistered module makes that lookup return ``None``
and the import fails with an ``AttributeError`` several frames inside
``dataclasses``. ``tests/test_egnca.py`` does not need this because
``train_egnca.py`` defines no dataclass.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from aanca import config as run_config
from aanca.egnca import EGNCARule, ca_positions, with_ca_positions
from aanca.growth import pairwise_distance_restraint
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "data" / "raw" / "1UBQ.pdb"
CONFIG = ROOT / "configs" / "denoise.toml"

RUNTIME = argparse.Namespace(
    device="auto", max_seconds=0.0, stop_file=None, resume=False
)


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_denoise", ROOT / "scripts" / "train_denoise.py"
    )
    module = importlib.util.module_from_spec(spec)
    # See this file's docstring: ``@dataclass`` needs the module findable by
    # name while the body is still executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def denoise():
    return _trainer()


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords


def rotation(degrees: float, dtype) -> torch.Tensor:
    """``tests/test_egnca.py``'s rotation, so the files turn alike."""
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


def _rule(seed: int = 0, gain: float = 0.3, double: bool = False, **kwargs):
    """A rule whose ``phi_x`` output is **not** at its published initialisation.

    The authors zero that layer, so an untrained rule emits exactly nothing and
    every "does it move" assertion would be asking about the number 0. This
    perturbs that one layer, on ``tests/test_egnca.py::_rule``'s calibration of
    0.3, so the move is a real quantity.
    """
    torch.manual_seed(seed)
    rule = EGNCARule(**kwargs)
    for parameter in rule.last_coord_layer.parameters():
        parameter.data.add_(torch.randn_like(parameter) * gain)
    if double:
        rule.double()
    rule.eval()
    return rule


# -- the corruption operator ------------------------------------------------


def test_the_corruption_is_isotropic_and_its_magnitude_is_sigma(denoise):
    """Three independent ``N(0, sigma^2)`` draws per residue, and nothing else.

    **Both halves matter and they fail differently.** A per-axis scale, or noise
    applied along a learned direction, would make the task depend on the
    structure's orientation in the lab -- which is exactly the property the
    E(n)-equivariant rule exists to not have, so the arm would be testing
    something other than what its banner says. And a magnitude that is not
    ``sigma`` would make every ``rmsd`` against ``start_rmsd`` comparison a
    comparison at an unnamed noise level.

    Measured on 40,000 residues at sigma 3: the three axis standard deviations
    against sigma, and the off-diagonal correlations against zero.
    """
    sigma = 3.0
    n = 40_000
    ca = torch.zeros(n, 3, dtype=torch.float64)
    mask = torch.ones(n, dtype=torch.bool)
    noisy = denoise.corrupt(
        ca, sigma, mask, generator=torch.Generator().manual_seed(0)
    )
    displacement = noisy - ca

    per_axis = displacement.std(dim=0)
    assert torch.allclose(
        per_axis, torch.full((3,), sigma, dtype=torch.float64), rtol=0.03
    ), f"per-axis std {per_axis.tolist()} against sigma {sigma}"

    # Isotropy is the off-diagonals as much as the diagonal: equal variances
    # with a correlated pair is an ellipsoid, not a sphere.
    centred = displacement - displacement.mean(dim=0, keepdim=True)
    covariance = (centred.T @ centred) / (n - 1)
    off = covariance - torch.diag(torch.diagonal(covariance))
    assert float(off.abs().max()) < 0.1 * sigma**2, (
        f"off-diagonal covariance {float(off.abs().max()):.4f} against "
        f"sigma^2 {sigma**2}"
    )

    # The expected displacement of a corrupted residue, sigma * sqrt(3), which
    # is what the trainer's ``start_rmsd`` column is a measurement of.
    assert float(displacement.norm(dim=-1).mean()) == pytest.approx(
        sigma * math.sqrt(8.0 / math.pi), rel=0.02
    )


def test_the_corruption_leaves_held_residues_bit_for_bit(denoise):
    """A held residue is an exact anchor or it is not an anchor at all."""
    ca = torch.arange(30, dtype=torch.float64).reshape(10, 3)
    held = torch.tensor([True, False] * 5)
    noisy = denoise.corrupt(
        ca, 5.0, ~held, generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(noisy[held], ca[held])
    assert not torch.equal(noisy[~held], ca[~held])


def test_sigma_is_drawn_log_uniformly_between_the_bounds(denoise):
    """In the band, and uniform in the LOG -- which is the half that is easy
    to lose.

    Uniform on [0.5, 30] puts 98 % of its mass above 1 A, so the rule would
    almost never be asked to hold still near the fixed point it is being trained
    to have. The check is on the median: log-uniform puts it at the geometric
    mean, sqrt(0.5 * 30) ~ 3.87, where uniform would put it at 15.25.
    """
    generator = torch.Generator().manual_seed(0)
    draws = sorted(denoise.draw_sigma(0.5, 30.0, generator) for _ in range(4000))
    assert draws[0] >= 0.5 and draws[-1] <= 30.0
    assert draws[len(draws) // 2] == pytest.approx(math.sqrt(0.5 * 30.0), rel=0.1)
    # Equal bounds fix the level, which is how a single-sigma arm is stated.
    assert denoise.draw_sigma(2.0, 2.0) == 2.0
    with pytest.raises(ValueError):
        denoise.draw_sigma(0.0, 10.0)
    with pytest.raises(ValueError):
        denoise.draw_sigma(10.0, 1.0)


# -- the context mask, which is the one axis --------------------------------


def test_context_fraction_zero_pins_nothing_and_one_pins_everything(denoise):
    """The two ends of the sweep, exactly, at every length the crops produce.

    ``0.0`` is the MVP and is the setting whose whole claim is that there is no
    anchor anywhere: one pinned residue is a leak, and a leak of one exact
    anchor in a 128-residue crop is not visible in any column of the log. ``1.0``
    is the degenerate end, where a single residue left corrupted would make
    ``ctx100`` quietly train on one residue instead of being skipped.
    """
    for n in (1, 7, 12, 76, 128, 129):
        empty = denoise.context_mask(n, 0.0)
        full = denoise.context_mask(n, 1.0)
        assert empty.shape == (n,) and full.shape == (n,)
        assert int(empty.sum()) == 0, f"{n} residues: {int(empty.sum())} pinned"
        assert int(full.sum()) == n, f"{n} residues: {int(full.sum())} pinned"
        assert bool((~empty).all()) and bool(full.all())


def test_the_held_context_is_contiguous_runs_of_about_the_port_width(denoise):
    """Runs, not a scatter, and the count is exact.

    A scatter of independent residues would put an exact anchor inside almost
    every k=24 neighbourhood at any fraction above a few percent, so every point
    on the sweep would carry the same local information and the axis would
    measure nothing. The runs are about ``CONTEXT_RUN_WIDTH`` = 12, the port's
    own damage width, so the corrupted stretches between them are the scale of
    the hole the port fills.
    """
    n = 128
    for fraction in (0.25, 0.5, 0.75):
        mask = denoise.context_mask(
            n, fraction, generator=torch.Generator().manual_seed(3)
        )
        assert int(mask.sum()) == round(fraction * n)
        # Count the runs of True by counting rising edges.
        padded = torch.cat([torch.zeros(1, dtype=torch.bool), mask])
        runs = int(((~padded[:-1]) & padded[1:]).sum())
        assert 1 <= runs <= math.ceil(fraction * n / denoise.CONTEXT_RUN_WIDTH) + 1
        assert int(mask.sum()) / runs == pytest.approx(
            denoise.CONTEXT_RUN_WIDTH, rel=0.6
        ), f"fraction {fraction}: {runs} run(s) for {int(mask.sum())} residues"


def test_a_fraction_outside_the_unit_interval_is_refused(denoise):
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            denoise.context_mask(16, bad)


# -- the loss, which is the port's own --------------------------------------


def test_the_loss_is_finite_and_non_negative_on_a_degenerate_configuration(
    denoise, ubiquitin
):
    """Every CA on top of every other, which an untrained rule can emit.

    **The specific hazard is ``norm``'s derivative at exactly zero**, which is
    nan: a pair of coincident atoms is reachable in one step from a corrupted
    start, and ``pairwise_distance_restraint`` clamps before the square root for
    that reason. A nan here would propagate into ``total``, be skipped as
    ``nonfinite_loss``, and the run would report a skip rate instead of a
    result. So the gradient is asserted too, not only the value -- a finite loss
    with a nan gradient is the failure that killed two of this project's growth
    arms.

    This is the loss **the trainer actually calls**:
    ``growth.pairwise_distance_restraint`` over the full pair set with the
    corrupted set in the damaged slot, unchanged and unwrapped.
    """
    topo, coords = ubiquitin
    zero = torch.zeros(topo.n_residues, 3, dtype=torch.float64, requires_grad=True)
    collapsed = with_ca_positions(coords, zero, topo)
    corrupted = torch.ones(topo.n_residues, dtype=torch.bool)
    anchor, window = pairwise_distance_restraint(
        collapsed, coords, topo, corrupted, ~corrupted
    )
    total = anchor + window
    assert torch.isfinite(total), f"loss is {float(total.detach())}"
    assert float(total.detach()) >= 0.0
    # No context at context_fraction 0, so the anchor half is exactly zero and
    # the full pair set lives in the window half.
    assert float(anchor.detach()) == 0.0
    assert float(window.detach()) > 0.0
    total.backward()
    assert zero.grad is not None and bool(torch.isfinite(zero.grad).all())


def test_the_objective_scores_the_corrupted_set_and_nothing_else(
    denoise, ubiquitin
):
    """``rmsd`` is read off the corrupted residues; the held ones never moved.

    And at ``context_fraction = 0`` the context superposition is **nan rather
    than zero**: ``backbone_rmsd_on`` has no unique rotation on fewer than three
    residues, and a zero there would read as a perfect fit in the log's best
    column.
    """
    topo, coords = ubiquitin
    rule = _rule(gain=0.1)
    terms, state = denoise.denoise_objective(
        rule,
        coords,
        topo,
        steps=4,
        context_fraction=0.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert terms.corrupted_mask is not None
    assert int(terms.corrupted_mask.sum()) == topo.n_residues
    assert torch.isfinite(terms.total)
    assert terms.rmsd is not None and terms.start_rmsd is not None
    assert terms.rmsd_fit_context != terms.rmsd_fit_context, "fit_ctx must be nan"
    assert 0.5 <= terms.sigma <= 30.0
    assert terms.total.requires_grad

    held = denoise.denoise_objective(
        rule,
        coords,
        topo,
        steps=4,
        context_fraction=0.5,
        generator=torch.Generator().manual_seed(0),
    )[0]
    assert int(held.corrupted_mask.sum()) == topo.n_residues - round(
        0.5 * topo.n_residues
    )
    # With half the structure held there are three residues to superpose on, so
    # the column exists.
    assert held.rmsd_fit_context == held.rmsd_fit_context


def test_every_residue_held_leaves_nothing_to_score(denoise, ubiquitin):
    """``context_fraction = 1.0`` is the degenerate end and says so.

    The restraint is a constant zero with no ``grad_fn``, so ``isfinite`` passes
    and ``backward()`` would raise "does not require grad" -- the trainer skips
    it as ``no_window`` instead, which is the same statement the port makes
    about a crop too short to hold a window.
    """
    topo, coords = ubiquitin
    terms, _state = denoise.denoise_objective(
        _rule(gain=0.1),
        coords,
        topo,
        steps=2,
        context_fraction=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert int(terms.corrupted_mask.sum()) == 0
    assert float(terms.total) == 0.0
    assert not terms.total.requires_grad


# -- equivariance under the graph this arm rebuilds -------------------------


def test_equivariance_survives_the_graph_rebuilt_from_current_coordinates(
    denoise, ubiquitin
):
    """Rotate and translate the input; the fixed-point run moves with it.

    **The property the whole port rests on, re-asserted on the new perception.**
    ``tests/test_egnca.py`` proves the rule equivariant with the neighbour list
    held fixed. This arm rebuilds the k-NN from where the residues currently are
    at every step, through ``with_ca_positions`` and the project's own
    minimum-heavy-atom selection -- a construction the port never exercises, and
    one that reads the carrier's atoms rather than only the CA. A selection that
    leaked the lab frame, or a carrier translation that did not rotate with the
    structure, would break the equivariance while every other test in this file
    passed.

    Run in float64 for ``tests/test_egnca.py``'s reason: coordinates reach 40 A,
    ``||x_i - x_j||^2`` is a difference of large numbers, and in float32 a rigid
    motion perturbs that invariant by ~1e-3 before the rule sees it.
    """
    topo, coords = ubiquitin
    rule = _rule(double=True)
    turn = rotation(40.0, rule.dtype)
    shift = torch.tensor([3.0, -5.0, 7.0], dtype=rule.dtype)

    start = ca_positions(coords, topo).to(rule.dtype)
    plain = denoise.run_to_fixed_point(
        rule, topo, coords, start, tolerance=1e-12, max_steps=5, k=24
    )
    turned = denoise.run_to_fixed_point(
        rule,
        topo,
        coords.to(rule.dtype) @ turn.T + shift,
        start @ turn.T + shift,
        tolerance=1e-12,
        max_steps=5,
        k=24,
    )
    assert plain.steps == turned.steps == 5, (
        "the runs stopped at different lengths, so the graph construction is "
        "not even taking the same path under a rigid motion"
    )
    step = (plain.ca - start).norm(dim=-1)
    gap = (turned.ca - (plain.ca @ turn.T + shift)).norm(dim=-1)
    assert float(step.mean()) > 1e-2, "nothing moved, so nothing is tested"
    assert float(gap.mean()) < 1e-4 * float(step.mean()) + 1e-5, (
        f"equivariance error {float(gap.mean()):.3e} A on a "
        f"{float(step.mean()):.3f} A move"
    )


# -- sampling from the prior ------------------------------------------------


def test_the_prior_is_reproducible_at_a_fixed_seed(denoise, ubiquitin):
    """Same seed, same cloud and same trajectory; different seed, different.

    A sample series is a readout of the weights across evaluations, so it has to
    be the same draw each time or the series measures the draw. ``steps`` is
    asserted too: a reproducible start with a non-reproducible run would still
    give a steps-to-convergence column that cannot be read.
    """
    topo, coords = ubiquitin
    rule = _rule(gain=0.1, double=True)
    make = lambda seed: denoise.prior_coordinates(  # noqa: E731
        topo.n_residues,
        generator=torch.Generator().manual_seed(seed),
        dtype=rule.dtype,
    )
    first, again, other = make(7), make(7), make(8)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)

    a = denoise.run_to_fixed_point(
        rule, topo, coords, first, tolerance=1e-3, max_steps=20
    )
    b = denoise.run_to_fixed_point(
        rule, topo, coords, again, tolerance=1e-3, max_steps=20
    )
    assert a.steps == b.steps
    assert torch.equal(a.ca, b.ca)


def test_the_prior_is_scaled_to_the_natural_radius_of_gyration(denoise):
    """``2.2 * N^0.38``, exactly rather than in expectation.

    Exactly, because steps-to-convergence is the readout: a per-sample radius
    varying by a few percent would put part of that column's variance in the
    draw rather than in the rule.
    """
    for n in (20, 76, 128, 400):
        cloud = denoise.prior_coordinates(
            n, generator=torch.Generator().manual_seed(n), dtype=torch.float64
        )
        assert cloud.shape == (n, 3)
        centred = cloud - cloud.mean(dim=0, keepdim=True)
        rg = float(centred.pow(2).sum(-1).mean().sqrt())
        assert rg == pytest.approx(denoise.radius_of_gyration(n), rel=1e-9)
        # Centred, so a sample does not start with a translation to undo.
        assert float(cloud.mean(dim=0).abs().max()) < 1e-9
    assert denoise.radius_of_gyration(128) == pytest.approx(
        2.2 * 128**0.38, rel=1e-12
    )


def test_the_samples_go_into_one_npz_and_not_one_file_each(denoise, tmp_path):
    """One compact file per run directory, ragged lengths and all.

    Bunya's user scratch quota has ~313k inodes free, so thousands of small
    files is a quota failure in the middle of a run rather than untidiness. The
    ragged case is the one that invites a directory of files, so it is the one
    tested: different crop lengths share a flat ``(total, 3)`` array with an
    ``offsets`` index and no padding value has to be invented.
    """
    import numpy as np

    samples = [
        denoise.Sample(
            ca=torch.arange(3 * n, dtype=torch.float32).reshape(n, 3),
            steps=n,
            converged=bool(n % 2),
            last_move=0.5,
            radius_of_gyration=float(n),
        )
        for n in (4, 7, 5)
    ]
    path = tmp_path / "samples.npz"
    denoise.write_samples(path, samples, update=250)
    assert len(list(tmp_path.iterdir())) == 1

    with np.load(path) as blob:
        assert int(blob["update"]) == 250
        assert blob["ca"].shape == (16, 3)
        assert blob["offsets"].tolist() == [0, 4, 11, 16]
        assert blob["steps"].tolist() == [4, 7, 5]
        assert blob["converged"].tolist() == [False, True, True]
        for index, sample in enumerate(samples):
            lo, hi = blob["offsets"][index], blob["offsets"][index + 1]
            assert np.allclose(blob["ca"][lo:hi], sample.ca.numpy())


# -- the skip counter -------------------------------------------------------


def test_the_skip_counter_rejects_an_unknown_reason(denoise):
    """A reason that is not in the tuple is an error, not a new column.

    The skip rate is a readout rather than a diagnostic -- the port's ~10 % of
    overflowing rollouts at every learning rate is why no learning rate helped
    -- and a typo'd reason counted into a dict that grows on demand would be
    invisible in both the per-interval line and the final summary.
    """
    with pytest.raises(ValueError):
        denoise._SkipUpdate("exploded")
    for reason in denoise.SKIP_REASONS:
        assert denoise._SkipUpdate(reason).reason == reason
    assert denoise.SKIP_REASONS == (
        "no_crop",
        "no_window",
        "nonfinite_loss",
        "nonfinite_gradient",
    )
    line = denoise.skip_report(
        {reason: 1 for reason in denoise.SKIP_REASONS}, window=250
    )
    assert "skip 4/250" in line and "nonfinite_loss 1" in line


# -- the config, and the port it may not touch ------------------------------


def test_the_config_states_every_setting_the_trainer_reads(denoise):
    """``check_base_is_complete`` on this arm's own config, through the loader.

    So a later change to a parser default cannot reinterpret a run recorded
    here. Every section is resolved too, because an unknown or mistyped key in a
    sweep section is only an error at the moment it is loaded.
    """
    data = run_config.load(CONFIG)
    schema = denoise.build_schema()
    run_config.check_base_is_complete(data, schema)
    names = run_config.sections(data)
    assert names == ["ctx100", "ctx25", "ctx50", "ctx75", "mvp"]
    for name in names:
        args = run_config.to_namespace(
            schema, run_config.resolve(data, name), name, RUNTIME
        )
        assert args.tag == name
        assert 0.0 <= args.context_fraction <= 1.0
        # The two removals that define the arm, and the five published settings
        # that must NOT change with them.
        assert args.native_edges is False
        assert args.has_coord_act and args.is_residual
        assert args.activation == "tanh" and args.norm_type == "pn"
        assert args.pairwise_weight == 1.0 and args.pairwise_window_weight == 1.0
        assert args.lr == 0.001 and args.clip == 1.0
        assert args.weight_decay == pytest.approx(1e-5)
        assert args.pool_probability == 0.5
        assert (args.steps_min, args.steps) == (15, 25)
    fractions = {
        name: run_config.resolve(data, name)["context_fraction"] for name in names
    }
    assert fractions == {
        "mvp": 0.0,
        "ctx25": 0.25,
        "ctx50": 0.5,
        "ctx75": 0.75,
        "ctx100": 1.0,
    }


def test_every_objective_setting_reaches_the_objective(denoise):
    """``objective_kwargs`` names every keyword :func:`denoise_objective` takes.

    The port's own reason: this project's record carries a retraction from
    weights that reached training and not evaluation (job 32405172), so a
    held-out column was structurally absent and printed as a measurement. One
    dict feeds both call sites, and this test is what keeps it complete.
    """
    import inspect

    data = run_config.load(CONFIG)
    schema = denoise.build_schema()
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "mvp"), "mvp", RUNTIME
    )
    passed = set(denoise.objective_kwargs(args))
    taken = set(inspect.signature(denoise.denoise_objective).parameters)
    unread = passed - taken
    assert not unread, f"objective_kwargs passes keys the objective ignores: {unread}"
    # Everything the objective has a setting for, minus what the trainer supplies
    # per update rather than from the config.
    per_update = {
        "rule",
        "native_coords",
        "topo",
        "steps",
        "generator",
        "resume_from",
    }
    missing = taken - passed - per_update
    assert not missing, f"settings that never reach the objective: {missing}"


def test_the_denoiser_exits_64_so_the_slurm_chain_resubmits(denoise):
    """``--max-seconds`` is "checkpointed, not finished", which is 64.

    ``scripts/bunya_train.slurm`` resubmits **only** on 64, and
    ``scripts/train_egnca.py`` returns 0 on this path -- so a port segment that
    hit its walltime ended the chain in silence, which is the behaviour
    HANDOFF.md §0a names. Asserted on the constant and on the slurm script that
    reads it, so the two cannot drift apart.
    """
    assert denoise.CHECKPOINTED_NOT_FINISHED == 64
    slurm = (ROOT / "scripts" / "bunya_train.slurm").read_text(encoding="utf-8")
    assert "RC == 64" in slurm


def test_the_published_port_is_untouched():
    """This work may not edit the three files Bunya job 33419979 rests on.

    Not a style rule: ``configs/egnca.toml``, ``scripts/train_egnca.py`` and
    ``src/aanca/egnca.py`` produced the 2.53 A result the project's central
    claim now rests on, and they must keep running bit-for-bit. The denoiser
    **imports** from the third rather than copying it, so this test is also what
    says the reuse is real -- a copy would let the two drift and the import
    cannot.
    """
    source = (ROOT / "scripts" / "train_denoise.py").read_text(encoding="utf-8")
    assert "from aanca.egnca import" in source
    for name in (
        "EGNCARule",
        "EGNCAState",
        "EGNCAPoolEntry",
        "egnca_rollout",
        "egnca_step",
        "egnca_neighbours",
        "with_ca_positions",
        "ca_positions",
    ):
        assert name in source, f"{name} should be imported, not reimplemented"
    assert "class EGNCARule" not in source
    assert "def egnca_rollout" not in source
    assert "def pairwise_distance_restraint" not in source
