"""What the CONDITIONED arm must keep true, and what each test prevents.

====================================== ======================================
the property                           what breaking it would look like
====================================== ======================================
the four protected files are unchanged a shared edit, which would invalidate
                                       Bunya job 33419979 and break both
                                       probes against ``runs/mvp/best.pt``
the label enters the invariant branch  equivariance lost, so the arm would be
                                       testing something other than Gala's
                                       rule
equivariance survives the live k-NN    a graph or a carrier that leaks the lab
                                       frame, which no other column would show
the embedding reaches the output       a silently disconnected label -- the
                                       exact failure readout 0 exists to
                                       catch, and the one that would make
                                       every other number in the table a
                                       measurement of nothing
dropout 1.0 is truly unconditional     a ``nolabel`` control that still reads
                                       the class, so the one-key comparison
                                       would be a no-key comparison
``structure_class`` is called          a second copy of the thresholds, which
                                       would drift from the validated one and
                                       from the natural reference quoted
                                       against it
the parameter cost stays ~148          a label that grows the rule, which
                                       trades away the "one small local rule"
                                       claim without arguing the trade
``history.json`` is valid JSON         a bare ``NaN`` token, which is not JSON
                                       and broke a reader of the MVP's history
``--max-seconds`` exits 64             a Slurm chain that ends in silence
nothing that changes a result is a flag a run whose settings are a shell line
====================================== ======================================

**The modules are registered in ``sys.modules`` before they are executed**, for
``tests/test_denoise.py``'s reason: they are loaded from a path rather than
imported, and ``@dataclass`` looks its own class's module up by name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from aanca import config as run_config
from aanca.conditioned import (
    CONDITIONED_COUNT,
    FOLD_CLASSES,
    LABEL_DIM,
    UNCONDITIONAL,
    ConditionedEGNCARule,
    class_index,
    class_name,
    conditioning_cost,
    effective_label,
)
from aanca.egnca import PUBLISHED_COUNT, EGNCARule, ca_positions
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "data" / "raw" / "1UBQ.pdb"
CONFIG = ROOT / "configs" / "conditioned.toml"

RUNTIME = argparse.Namespace(
    device="auto", max_seconds=0.0, stop_file=None, resume=False
)

#: SHA-256 of each protected file's text with newlines normalised to ``\n``.
#:
#: **Normalised rather than raw bytes** because ``core.autocrlf`` is true in
#: this checkout: a fresh clone on Windows can hand back CRLF for the same
#: committed content, and a test that failed on a re-clone would be a false
#: alarm that teaches people to ignore it. Any real edit -- a character, a line,
#: a key -- still changes the digest.
PROTECTED = {
    "scripts/train_denoise.py": (
        "3e8c5f3eda9da5ecbe0c7856b78a7f131e310c896f34f67530059e933742d007"
    ),
    "scripts/train_egnca.py": (
        "6e327b80df008ba25cb5c4c641980f9668fa293ccc290250caeb44005ee253dc"
    ),
    "src/aanca/egnca.py": (
        "3835c261be173f9669cf3825d2febdde66c109174a9de59e41ff9dc6a902d099"
    ),
    "configs/egnca.toml": (
        "377cccb7b9bd57b1cf4437e7e1d04ff6de55383c2cea7d6183031a8fa873e67a"
    ),
}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    # See this file's docstring: ``@dataclass`` needs the module findable by
    # name while the body is still executing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _script("train_conditioned")


@pytest.fixture(scope="module")
def metrics():
    return _script("generation_metrics")


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords


def rotation(degrees: float, dtype) -> torch.Tensor:
    """``tests/test_denoise.py``'s rotation, so the files turn alike."""
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
    """A conditioned rule whose ``phi_x`` output is NOT at its published init.

    The authors zero that layer, so an untrained rule emits exactly nothing and
    every "does it move" assertion would be asking about the number 0. The
    **label embedding is perturbed too**, and for the same reason squared: it is
    initialised from ``N(0, 1)`` so it is not degenerate, but ``phi_x`` at zero
    would make the label unable to reach a coordinate no matter what it said.
    """
    torch.manual_seed(seed)
    rule = ConditionedEGNCARule(**kwargs)
    for parameter in rule.last_coord_layer.parameters():
        parameter.data.add_(torch.randn_like(parameter) * gain)
    if double:
        rule.double()
    rule.eval()
    return rule


# -- the four files this work may not touch ---------------------------------


def test_the_four_protected_files_are_unchanged():
    """``src/aanca/egnca.py`` and the two trainers and config beside it.

    Not a style rule, and two separate stakes.
    ``configs/egnca.toml``/``scripts/train_egnca.py``/``src/aanca/egnca.py``
    produced the 2.53 A result the project's central claim rests on, and
    ``scripts/train_denoise.py`` is the refuted MVP that is itself a quoted
    result. **Beyond that**, ``runs/mvp/best.pt`` must stay readable by
    ``scripts/prior_fixed_point.py`` and ``scripts/denoise_floor.py``, which
    refuse a checkpoint missing any key their trainer's ``build_schema()``
    defines -- so a key added to the denoiser's schema would break both probes
    against the very checkpoint they exist to measure. That is why the
    conditioned arm is a new trainer and a new schema rather than a key here.
    """
    for relative, digest in PROTECTED.items():
        text = (ROOT / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        found = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert found == digest, (
            f"{relative} has changed (sha256 {found}, expected {digest}). "
            "These four files are quoted results and must stay bit-for-bit; "
            "the conditioned arm is a separate trainer for exactly this reason."
        )


def test_the_conditioned_trainer_imports_rather_than_copies(trainer):
    """The reuse is real: by-path imports, not a forked body.

    ``tests/test_denoise.py``'s pattern. A copy would let the two arms drift and
    an import cannot, which is what makes "the only difference is the label" a
    checkable statement rather than a claim in a docstring.
    """
    source = (ROOT / "scripts" / "train_conditioned.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "prior_coordinates",
        "run_to_fixed_point",
        "denoise_objective",
        "write_samples",
        "Sample",
        "DenoiseTerms",
        "skip_report",
        "on_disk",
    ):
        assert f"denoise.{name}" in source, (
            f"{name} should come from train_denoise by path, not be rewritten"
        )
    # The bodies that must not be here.
    assert "def denoise_objective" not in source
    assert "def run_to_fixed_point" not in source
    assert "def prior_coordinates" not in source
    assert "def draw_sigma" not in source
    assert "def context_mask" not in source
    assert "def corrupt(" not in source
    assert "class EGNCARule" not in source
    assert "def pairwise_distance_restraint" not in source
    # And the rule, which lives in its own module beside egnca.py.
    assert "ConditionedEGNCARule" in source
    assert "class ConditionedEGNCARule" not in source


def test_structure_class_is_called_and_not_reimplemented(trainer, metrics):
    """The thresholds live in ``generation_metrics`` and nowhere else.

    P-SEA is validated there -- 1.000 per-residue agreement with biotite's
    ``annotate_sse`` on all four crystal structures in ``data/raw/`` -- and the
    natural reference's 35/14/10/1 class split is quoted against those exact
    thresholds. A second copy would drift from both, and the drift would show up
    as a confusion table about a different label than the one the arm was
    trained on.
    """
    source = (ROOT / "scripts" / "train_conditioned.py").read_text(
        encoding="utf-8"
    )
    rule_source = (ROOT / "src" / "aanca" / "conditioned.py").read_text(
        encoding="utf-8"
    )
    assert "metrics.secondary_metrics" in source
    assert "def structure_class" not in source
    assert "def structure_class" not in rule_source
    # No restated threshold anywhere in the arm: the four numbers P-SEA's class
    # rule turns on are 0.20, 0.10, 0.20, 0.10 and none of them is written here.
    assert "def psea" not in source and "def psea" not in rule_source

    # And the four names this arm indexes are exactly the four that function
    # returns, checked by calling it rather than by restating its table.
    produced = {
        metrics.structure_class(helix, strand)
        for helix in (0.0, 0.05, 0.25, 0.6)
        for strand in (0.0, 0.05, 0.25, 0.6)
    }
    assert produced == set(FOLD_CLASSES), (
        f"structure_class returns {sorted(produced)}, FOLD_CLASSES names "
        f"{sorted(FOLD_CLASSES)}"
    )
    for name in FOLD_CLASSES:
        assert class_name(class_index(name)) == name
    assert class_name(UNCONDITIONAL) == "unconditional"
    with pytest.raises(ValueError):
        class_index("mainly-gamma")


# -- the label reaches the output, which is readout 0's own failure mode -----


def test_two_different_class_labels_give_different_coordinates(ubiquitin):
    """**The guard against a silently disconnected embedding.**

    Same input, same seed, same weights, two different class labels: if the
    coordinates come out equal the label is not reaching ``phi_x`` at all, and
    every other number this arm prints would be a measurement of the
    unconditional model wearing a label's name. That is the exact failure
    readout 0 exists to catch, and it is cheaper to catch here.

    Asserted on the COORDINATES rather than on the messages, because an
    embedding that reaches ``phi_m`` and then dies in ``phi_x``'s zeroed output
    layer would pass a test on the hidden state and still move nothing.
    """
    topo, coords = ubiquitin
    rule = _rule(gain=0.3, double=True)
    start = ca_positions(coords, topo).to(rule.dtype)

    runs = {}
    for name in FOLD_CLASSES:
        with rule.conditioned_on(class_index(name)):
            runs[name] = torch.stack([
                _one_step(rule, topo, coords, start) for _ in range(1)
            ])[0]

    base = runs["mainly-alpha"]
    gaps = {
        name: float((runs[name] - base).norm(dim=-1).mean())
        for name in FOLD_CLASSES
        if name != "mainly-alpha"
    }
    move = float((base - start).norm(dim=-1).mean())
    assert move > 1e-3, "nothing moved, so nothing is tested"
    for name, gap in gaps.items():
        assert gap > 1e-6, (
            f"conditioning on {name!r} instead of 'mainly-alpha' changed the "
            f"coordinates by {gap:.3e} A on a {move:.3f} A move -- the label "
            "is not reaching the output"
        )
    # And the unconditional row is a fifth, distinct condition.
    with rule.conditioned_on(UNCONDITIONAL):
        unconditional = _one_step(rule, topo, coords, start)
    assert float((unconditional - base).norm(dim=-1).mean()) > 1e-6


def _one_step(rule, topo, coords, start):
    """One deterministic step of the rule, for a label-versus-label comparison."""
    from aanca.egnca import egnca_neighbours, egnca_step, with_ca_positions
    from aanca.egnca import EGNCAState

    selection = egnca_neighbours(
        with_ca_positions(coords, start, topo), topo, 24, 10.0, False
    )
    state = EGNCAState(
        ca=start,
        hidden=rule.initial_hidden(
            topo.n_residues, device=start.device, dtype=rule.dtype
        ),
    )
    with torch.no_grad():
        state = egnca_step(
            rule, state, selection.neighbours, selection.edge_mask
        )
    return state.ca


def test_label_dropout_of_one_makes_every_class_identical(ubiquitin):
    """The ``nolabel`` control is a control, in training AND in evaluation.

    :func:`effective_label` sends every class to the unconditional row at
    ``label_dropout >= 1.0`` whether a dropout draw is offered or not. Without
    the "or not", evaluation would still condition on the true class and the
    control would be a *conditional* model measured against a conditional model
    -- so the one-key comparison would be a no-key comparison and nothing would
    say so.
    """
    topo, coords = ubiquitin
    rule = _rule(gain=0.3, double=True)
    start = ca_positions(coords, topo).to(rule.dtype)

    outputs = []
    for name in FOLD_CLASSES:
        row = effective_label(class_index(name), 1.0, draw=None)
        assert row == UNCONDITIONAL
        with rule.conditioned_on(row):
            outputs.append(_one_step(rule, topo, coords, start))
    for other in outputs[1:]:
        assert torch.equal(outputs[0], other), (
            "label dropout 1.0 left two classes giving different coordinates, "
            "so the class still reaches the rule"
        )
    # A training draw cannot rescue the class either.
    assert effective_label(0, 1.0, draw=0.999) == UNCONDITIONAL
    # And at dropout 0 nothing is ever dropped, whatever the draw.
    assert effective_label(2, 0.0, draw=0.0) == 2
    assert effective_label(2, 0.0, draw=None) == 2
    # The intended behaviour in between: a draw below the probability drops.
    assert effective_label(1, 0.5, draw=0.4) == UNCONDITIONAL
    assert effective_label(1, 0.5, draw=0.6) == 1
    # Evaluation never drops, which is what makes the held-out column one model.
    assert effective_label(1, 0.5, draw=None) == 1
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            effective_label(0, bad, draw=None)


def test_the_label_is_restored_after_a_conditioned_block():
    """A rollout left conditioned on the previous sample's class is a wrong
    measurement that no column would show.
    """
    rule = ConditionedEGNCARule()
    assert rule.label == UNCONDITIONAL, (
        "a rule that defaults to row 0 would silently condition on "
        "'mainly-alpha' for any caller that forgot to set a label"
    )
    with rule.conditioned_on(1):
        assert rule.label == 1
        with rule.conditioned_on(3):
            assert rule.label == 3
        assert rule.label == 1
    assert rule.label == UNCONDITIONAL
    # Restored even when the block raises, which is the case that matters:
    # a skipped update must not leak its condition into the next one.
    with pytest.raises(RuntimeError):
        with rule.conditioned_on(2):
            raise RuntimeError("boom")
    assert rule.label == UNCONDITIONAL
    with pytest.raises(ValueError):
        rule.label = 99


# -- equivariance, which the label must not break ---------------------------


def test_equivariance_survives_the_live_graph_with_a_label(trainer, ubiquitin):
    """Rotate and translate the input; the conditioned run moves with it.

    **The property the whole port rests on, re-asserted with the label live.**
    The embedding is an E(n)-invariant scalar vector entering ``phi_m``'s
    invariant branch -- the same place ``||x_i - x_j||^2`` enters -- so
    equivariance is preserved by construction. "By construction" is an argument
    and not a measurement: a label accidentally mixed into the coordinate branch
    would break this while every other test in this file passed.

    Run with the LIVE k-NN, rebuilt from where the residues currently are at
    every step, because that is what the arm runs. Run in float64 for
    ``tests/test_egnca.py``'s reason: coordinates reach 40 A,
    ``||x_i - x_j||^2`` is a difference of large numbers, and in float32 a rigid
    motion perturbs that invariant by ~1e-3 before the rule sees it.
    """
    topo, coords = ubiquitin
    rule = _rule(double=True)
    turn = rotation(40.0, rule.dtype)
    shift = torch.tensor([3.0, -5.0, 7.0], dtype=rule.dtype)
    start = ca_positions(coords, topo).to(rule.dtype)

    for name in FOLD_CLASSES:
        with rule.conditioned_on(class_index(name)):
            plain = trainer.run_to_fixed_point(
                rule, topo, coords, start, tolerance=1e-12, max_steps=5, k=24
            )
            turned = trainer.run_to_fixed_point(
                rule,
                topo,
                coords.to(rule.dtype) @ turn.T + shift,
                start @ turn.T + shift,
                tolerance=1e-12,
                max_steps=5,
                k=24,
            )
        assert plain.steps == turned.steps == 5, (
            f"{name}: the runs stopped at different lengths, so the graph "
            "construction is not even taking the same path under a rigid "
            "motion"
        )
        step = (plain.ca - start).norm(dim=-1)
        gap = (turned.ca - (plain.ca @ turn.T + shift)).norm(dim=-1)
        assert float(step.mean()) > 1e-2, f"{name}: nothing moved"
        assert float(gap.mean()) < 1e-4 * float(step.mean()) + 1e-5, (
            f"{name}: equivariance error {float(gap.mean()):.3e} A on a "
            f"{float(step.mean()):.3f} A move"
        )


# -- the parameter cost, which is load-bearing for the framing --------------


def test_the_label_costs_about_a_hundred_parameters_and_not_thousands():
    """5,328 + 148, derived block by block rather than quoted.

    **The smallness is load-bearing for the whole framing.** The
    pre-registration's argument is that a ~5.4k-parameter local rule is being
    handed one global scalar -- if this arm needed to grow the rule to work, the
    "one small local rule" claim is what would be traded away, and that trade
    gets argued before it is made rather than discovered in a parameter count
    afterwards.
    """
    plain = EGNCARule()
    conditioned = ConditionedEGNCARule()
    assert plain.parameter_count() == PUBLISHED_COUNT == 5_328
    cost = conditioned.parameter_count() - plain.parameter_count()
    assert cost == conditioning_cost() == LABEL_DIM * 32 + (
        len(FOLD_CLASSES) + 1
    ) * LABEL_DIM
    assert cost == 148, f"the label costs {cost} parameters, not 148"
    assert conditioned.parameter_count() == CONDITIONED_COUNT == 5_476
    assert conditioned.parameter_count() < 6_000, (
        f"{conditioned.parameter_count()} parameters is not 'on the order of "
        "5,328 plus a hundred'"
    )
    # The embedding has one row per class plus the unconditional one, and the
    # widened input is ``phi_m``'s first layer and nothing else.
    assert conditioned.label_embedding.weight.shape == (
        len(FOLD_CLASSES) + 1,
        LABEL_DIM,
    )
    assert conditioned.phi_m[0].in_features == plain.phi_m[0].in_features + LABEL_DIM
    assert conditioned.phi_h[0].in_features == plain.phi_h[0].in_features
    assert conditioned.phi_x[0].in_features == plain.phi_x[0].in_features


def test_the_label_enters_phi_m_and_not_the_hidden_state(ubiquitin):
    """Where it enters, asserted on behaviour and not only on the source.

    Two alternatives the pre-registration rules out, and each would pass a test
    that only looked at the parameter count:

      * **the initial hidden state** -- which would make the arm a question
        about whether the rule PRESERVES a signal over 15-25 steps, confounded
        with whether it uses one. ``initial_hidden`` is asserted to be the
        published all-ones, label or no label.
      * **added to the hidden state** -- which ``norm_type = pn`` would
        normalise away: PairNorm centres the rows over the whole structure, so a
        constant offset added to every row is removed exactly. Asserted by
        showing that the label is still live when PairNorm is on.
    """
    topo, coords = ubiquitin
    rule = _rule(gain=0.3, double=True)
    assert rule.norm_type == "pn", "the PairNorm argument needs PairNorm on"
    for name in FOLD_CLASSES:
        with rule.conditioned_on(class_index(name)):
            hidden = rule.initial_hidden(topo.n_residues, dtype=rule.dtype)
        assert torch.equal(hidden, torch.ones_like(hidden)), (
            "the label reached the seed, which is the alternative the "
            "pre-registration rules out"
        )
    # With PairNorm on, the label still changes the coordinates -- so it is not
    # entering anywhere PairNorm can remove it.
    start = ca_positions(coords, topo).to(rule.dtype)
    with rule.conditioned_on(0):
        first = _one_step(rule, topo, coords, start)
    with rule.conditioned_on(1):
        second = _one_step(rule, topo, coords, start)
    assert float((first - second).norm(dim=-1).mean()) > 1e-6

    # The rows differ, which is what makes "two labels" mean two things.
    rows = rule.label_embedding.weight
    assert rows.shape[0] == len(FOLD_CLASSES) + 1
    for i in range(rows.shape[0]):
        for j in range(i + 1, rows.shape[0]):
            assert not torch.equal(rows[i], rows[j])

    # And the label is genuinely in ``phi_m``'s input, as ``e_ij``: an
    # explicitly passed edge_attr still wins, so the inherited contract holds.
    assert rule.edge_dim == rule.label_dim == LABEL_DIM


def test_the_label_is_global_and_carries_no_address(ubiquitin):
    """**The property the whole discrimination rests on.**

    A fold class is the same number for every residue, so it adds exactly zero
    positional information -- that is why it discriminates H-ambiguity from
    H-address instead of being another lever, and why it does not violate the
    project's standing position that no cell may read a global *coordinate*. A
    per-residue or per-edge-varying channel would be address information and the
    arm would answer neither hypothesis.

    Asserted on the broadcast tensor itself: every residue and every edge sees
    the identical row.
    """
    topo, _coords = ubiquitin
    rule = _rule(double=True)
    with rule.conditioned_on(2):
        row = rule.label_row()
    assert row.shape == (LABEL_DIM,)
    broadcast = row.expand(topo.n_residues, 24, LABEL_DIM)
    assert broadcast.shape == (topo.n_residues, 24, LABEL_DIM)
    # Zero variance across residues and across edges, exactly.
    assert float(broadcast.std(dim=0).abs().max()) == 0.0
    assert float(broadcast.std(dim=1).abs().max()) == 0.0


# -- the config, and the one axis it adds -----------------------------------


def test_the_config_states_every_setting_the_trainer_reads(trainer):
    """``check_base_is_complete`` on this arm's config, through the loader.

    So a later change to a parser default cannot reinterpret a run recorded
    here. Every section is resolved too, because an unknown or mistyped key in a
    section is only an error at the moment it is loaded.
    """
    data = run_config.load(CONFIG)
    schema = trainer.build_schema()
    run_config.check_base_is_complete(data, schema)
    names = run_config.sections(data)
    assert names == ["cond", "nolabel"]
    for name in names:
        args = run_config.to_namespace(
            schema, run_config.resolve(data, name), name, RUNTIME
        )
        assert args.tag == name
        # Everything kept from configs/denoise.toml's mvp, identical.
        assert args.context_fraction == 0.0
        assert args.native_edges is False
        assert args.has_coord_act and args.is_residual
        assert args.activation == "tanh" and args.norm_type == "pn"
        assert args.pairwise_weight == 1.0 and args.pairwise_window_weight == 1.0
        assert args.lr == 0.001 and args.clip == 1.0
        assert args.weight_decay == pytest.approx(1e-5)
        assert args.pool_probability == 0.5
        assert (args.steps_min, args.steps) == (15, 25)
        assert (args.sigma_min, args.sigma_max) == (0.5, 30.0)
        assert args.crop_size == 128 and args.k == 24 and args.cutoff == 10.0
        assert args.hidden_dim == 16 and args.message_dim == 32
        assert args.updates == 20_000 and args.eval_every == 250
        assert args.structures == 400
        # Equal numbers per class, which readouts 0 and 5 need.
        assert args.sample_count % len(FOLD_CLASSES) == 0
        assert args.label_dim == LABEL_DIM
    dropouts = {
        name: run_config.resolve(data, name)["label_dropout"] for name in names
    }
    assert dropouts == {"cond": 0.1, "nolabel": 1.0}, (
        "the control is label dropout 1.0 -- always unconditional -- and the "
        "arm is a small probability, and nothing else differs between them"
    )
    # ONE key between the arm and its control, which is what makes the
    # comparison single-variable rather than two log headers diffed by eye.
    cond = run_config.resolve(data, "cond")
    nolabel = run_config.resolve(data, "nolabel")
    assert {
        key for key in set(cond) | set(nolabel) if cond.get(key) != nolabel.get(key)
    } == {"label_dropout"}


def test_the_schema_is_the_denoisers_plus_exactly_two_keys(trainer):
    """Key for key, so this arm cannot quietly change a setting the MVP shares.

    The comparison with ``configs/denoise.toml``'s ``mvp`` is the whole point of
    the arm. If this schema drifted -- a different default ``k``, a different
    ``pool_probability`` -- the difference in a result would no longer be the
    label, and nothing in either config would say so.
    """
    denoise = _script("train_denoise")
    mine = {
        action.dest: action.default
        for action in trainer.build_schema()._actions  # noqa: SLF001
        if action.dest != "help"
    }
    theirs = {
        action.dest: action.default
        for action in denoise.build_schema()._actions  # noqa: SLF001
        if action.dest != "help"
    }
    added = set(mine) - set(theirs)
    assert added == set(trainer.NEW_KEYS) == {"label_dim", "label_dropout"}
    assert not set(theirs) - set(mine), (
        f"the conditioned arm dropped settings the MVP has: "
        f"{sorted(set(theirs) - set(mine))}"
    )
    differing = {
        key: (theirs[key], mine[key])
        for key in theirs
        if theirs[key] != mine[key]
    }
    assert not differing, (
        f"shared settings whose defaults differ from the denoiser's: "
        f"{differing}"
    )


def test_every_objective_setting_reaches_the_objective(trainer):
    """``objective_kwargs`` names every keyword the objective takes.

    The port's own reason: this project's record carries a retraction from
    weights that reached training and not evaluation (job 32405172), so a
    held-out column was structurally absent and printed as a measurement. One
    dict feeds both call sites and this test is what keeps it complete.
    """
    import inspect

    data = run_config.load(CONFIG)
    schema = trainer.build_schema()
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "cond"), "cond", RUNTIME
    )
    passed = set(trainer.objective_kwargs(args))
    taken = set(inspect.signature(trainer.conditioned_objective).parameters)
    # The objective forwards the rest to ``denoise_objective`` through **kwargs.
    forwarded = set(
        inspect.signature(trainer.denoise_objective).parameters
    )
    unread = passed - taken - forwarded
    assert not unread, f"objective_kwargs passes keys nothing reads: {unread}"
    per_update = {
        "rule",
        "native_coords",
        "topo",
        "steps",
        "label",
        "dropout_draw",
        "generator",
        "resume_from",
        "kwargs",
    }
    missing = (taken | forwarded) - passed - per_update
    assert not missing, f"settings that never reach the objective: {missing}"
    assert "label_dropout" in passed, (
        "label dropout changes what is trained and must reach both call sites"
    )


def test_nothing_that_changes_a_result_is_a_command_line_flag(trainer):
    """``src/aanca/config.py``'s rule, checked on this trainer's own parser.

    Fifteen growth arms were launched as twenty-token shell lines and the only
    record of any of them is an ``args :`` line in a Slurm log. The runtime
    parser may only address the config or name a resource.
    """
    allowed = {
        "help",
        "config",
        "section",
        "list_sections",
        "diff",
        "device",
        "max_seconds",
        "stop_file",
        "resume",
    }
    found = {
        action.dest
        for action in trainer.build_parser()._actions  # noqa: SLF001
    }
    assert found <= allowed, f"result-changing flags on the command line: {found - allowed}"
    # And the two new keys are config keys, not flags.
    for key in trainer.NEW_KEYS:
        assert key not in found


# -- what the trainer writes -------------------------------------------------


def test_the_history_is_valid_json_with_null_and_never_a_bare_nan(
    trainer, tmp_path
):
    """``scripts/train_denoise.py``'s one bug, not reproduced.

    It dumps ``rmsd_fit_context`` -- which is ``nan`` for every sample at
    ``context_fraction = 0.0``, so every row -- straight through
    ``json.dumps``, which emits a bare ``NaN``. That token is not valid JSON by
    the spec, a strict parser refuses the whole file, and that is how a reader of
    that history broke. ``null`` is the honest encoding of a quantity that does
    not exist, and it round-trips.
    """
    history = [
        {
            "update": 250,
            "total": 87.4,
            "rmsd_fit_context": float("nan"),
            "sigma": float("inf"),
            "move": -float("inf"),
            "skips": {"no_crop": 0, "nonfinite_loss": float("nan")},
            "sample_helix_fraction": [0.3, float("nan")],
            "confusion": {"mainly-alpha -> irregular": 2},
        }
    ]
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(trainer.json_safe(history), indent=1), encoding="utf-8"
    )
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw, (
        "a bare NaN or Infinity token is not valid JSON and breaks a strict "
        f"parser:\n{raw}"
    )
    assert "null" in raw
    back = json.loads(raw)
    assert back[0]["rmsd_fit_context"] is None
    assert back[0]["sigma"] is None and back[0]["move"] is None
    assert back[0]["skips"]["nonfinite_loss"] is None
    assert back[0]["sample_helix_fraction"] == [0.3, None]
    assert back[0]["total"] == 87.4
    assert back[0]["confusion"] == {"mainly-alpha -> irregular": 2}
    # And the trainer uses it, rather than having it available and not calling
    # it -- which is exactly the shape of the bug being avoided.
    source = (ROOT / "scripts" / "train_conditioned.py").read_text(
        encoding="utf-8"
    )
    assert "json.dumps(json_safe(history)" in source


def test_the_samples_go_into_one_npz_with_the_class_they_were_conditioned_on(
    trainer, tmp_path
):
    """ONE file, the denoiser's flat layout, plus the conditioning columns.

    Bunya's user scratch quota has ~313k inodes free, so thousands of small
    files is a quota failure mid-run rather than untidiness -- and two files per
    evaluation would be the same problem starting. The ragged case is the one
    that invites a directory of files, so it is the one tested.

    **The conditioned class has to be in the file**, or readout 5's confusion
    table is not computable from the samples and would have to be recovered by
    rerunning the sampler against weights that have since moved.
    """
    import numpy as np

    samples = [
        trainer.ConditionedSample(
            sample=trainer.Sample(
                ca=torch.arange(3 * n, dtype=torch.float32).reshape(n, 3),
                steps=n,
                converged=bool(n % 2),
                last_move=0.5,
                radius_of_gyration=float(n),
            ),
            label=label,
            produced=produced,
            helix_fraction=0.1 * label,
            strand_fraction=0.2,
        )
        for n, label, produced in ((4, 0, "mainly-alpha"), (7, 1, "irregular"), (5, 3, ""))
    ]
    path = tmp_path / "samples.npz"
    trainer.write_samples(path, samples, update=250)
    assert len(list(tmp_path.iterdir())) == 1, "one file per run directory"

    with np.load(path, allow_pickle=False) as blob:
        assert int(blob["update"]) == 250
        # The denoiser's own layout, written by its own function.
        assert blob["ca"].shape == (16, 3)
        assert blob["offsets"].tolist() == [0, 4, 11, 16]
        assert blob["steps"].tolist() == [4, 7, 5]
        assert blob["converged"].tolist() == [False, True, True]
        # The conditioning columns readout 5 needs.
        assert blob["label"].tolist() == [0, 1, 3]
        assert [str(v) for v in blob["conditioned_class"]] == [
            "mainly-alpha",
            "mainly-beta",
            "irregular",
        ]
        assert [str(v) for v in blob["produced_class"]] == [
            "mainly-alpha",
            "irregular",
            "",
        ]
        assert blob["helix_fraction"].tolist() == pytest.approx([0.0, 0.1, 0.3])
        assert [str(v) for v in blob["fold_classes"]] == list(FOLD_CLASSES)
        for index, entry in enumerate(samples):
            lo, hi = blob["offsets"][index], blob["offsets"][index + 1]
            assert np.allclose(blob["ca"][lo:hi], entry.sample.ca.numpy())


def test_readout_zero_is_a_paired_contrast_and_reports_its_own_n(trainer):
    """``helix a/b``, and ``n`` beside it because two samples is not an error bar.

    The instrument check: if the two distributions are indistinguishable the
    conditioning is inert and nothing below may be read. A mean over zero
    samples must print as ``nan`` with the count saying why, not as 0.0.
    """
    make = lambda label, helix: trainer.ConditionedSample(  # noqa: E731
        sample=trainer.Sample(
            ca=torch.zeros(4, 3),
            steps=1,
            converged=False,
            last_move=1.0,
            radius_of_gyration=1.0,
        ),
        label=label,
        produced="",
        helix_fraction=helix,
    )
    alpha, beta = class_index("mainly-alpha"), class_index("mainly-beta")
    gap = trainer.helix_contrast([
        make(alpha, 0.6),
        make(alpha, 0.4),
        make(beta, 0.1),
        make(beta, 0.1),
        make(class_index("alpha/beta"), 0.9),
        make(UNCONDITIONAL, 0.0),
    ])
    assert gap["helix_alpha"] == pytest.approx(0.5)
    assert gap["helix_beta"] == pytest.approx(0.1)
    assert gap["helix_gap"] == pytest.approx(0.4)
    assert (gap["n_alpha"], gap["n_beta"]) == (2.0, 2.0)

    # An all-unconditional batch -- every sample of the ``nolabel`` control --
    # reads nan on both sides with n 0, which is the honest answer.
    empty = trainer.helix_contrast([make(UNCONDITIONAL, 0.5)])
    assert empty["helix_alpha"] != empty["helix_alpha"]
    assert (empty["n_alpha"], empty["n_beta"]) == (0.0, 0.0)
    # A non-finite helix fraction is counted out rather than poisoning the mean.
    one = trainer.helix_contrast([make(alpha, float("nan")), make(alpha, 0.4)])
    assert one["helix_alpha"] == pytest.approx(0.4) and one["n_alpha"] == 1.0


def test_the_class_counts_report_names_every_class_even_at_zero(trainer):
    """A class with too few examples to train is a reported fact.

    ``irregular`` is rare -- the 60-entry natural reference reads 1 of 60 -- and
    a missing column reads as a class that was never asked about rather than one
    with nothing to learn from. That distinction is what readout 5's confusion
    table has to be read against.
    """
    from collections import Counter

    line = trainer.class_counts_report(
        Counter({"alpha/beta": 200, "mainly-beta": 90, "mainly-alpha": 80}),
        unreadable=14,
        total=384,
    )
    for name in FOLD_CLASSES:
        assert name in line
    assert "irregular 0" in line
    assert "TOO FEW TO TRAIN" in line and "irregular" in line
    assert "14 could not be cropped" in line
    # And a balanced split says nothing about imbalance.
    plenty = trainer.class_counts_report(
        Counter({name: 50 for name in FOLD_CLASSES}), unreadable=0, total=200
    )
    assert "TOO FEW TO TRAIN" not in plenty


def test_the_confusion_table_is_conditioned_against_produced(trainer):
    """Readout 5, and the only readout that could be a positive result alone.

    A rule that reliably produces more helix when asked for helix has learned
    something about secondary structure even if its bonds are wrong. A diverged
    sample has no class and is left out rather than counted as a miss.
    """
    make = lambda label, produced: trainer.ConditionedSample(  # noqa: E731
        sample=trainer.Sample(
            ca=torch.zeros(4, 3),
            steps=1,
            converged=False,
            last_move=1.0,
            radius_of_gyration=1.0,
        ),
        label=label,
        produced=produced,
    )
    table = trainer.confusion([
        make(0, "mainly-alpha"),
        make(0, "mainly-alpha"),
        make(1, "irregular"),
        make(2, ""),
    ])
    assert table == {
        "mainly-alpha -> mainly-alpha": 2,
        "mainly-beta -> irregular": 1,
    }


def test_the_trainer_exits_64_so_the_slurm_chain_resubmits(trainer, tmp_path):
    """``--max-seconds`` is "checkpointed, not finished", which is 64.

    ``scripts/bunya_train.slurm`` resubmits **only** on 64, and
    ``scripts/train_egnca.py`` and ``scripts/train_growth.py`` return 0 on this
    path -- so a segment that hit its walltime ended the chain in silence. Run
    end to end rather than asserted on the constant, because the constant being
    right and the return path taking a different branch is exactly how that bug
    survived.
    """
    assert trainer.CHECKPOINTED_NOT_FINISHED == 64
    slurm = (ROOT / "scripts" / "bunya_train.slurm").read_text(encoding="utf-8")
    assert "RC == 64" in slurm

    # The real ``cond`` section over 400 structures would spend a minute of the
    # suite labelling crops it is not going to train on, so the run is the arm's
    # own settings with the data narrowed -- written out through
    # ``config.snapshot``, which is the same loadable shape the trainer writes
    # into every run directory. The tag follows ``runs/pytest_*``'s precedent.
    section = f"pytest_cond_{os.getpid()}"
    schema = trainer.build_schema()
    args = run_config.to_namespace(
        schema, run_config.resolve(run_config.load(CONFIG), "cond"), section, RUNTIME
    )
    args.structures, args.eval_structures = 6, 2
    args.sample_count, args.sample_max_steps = 4, 4
    args.updates, args.eval_every = 2, 1
    tiny = tmp_path / "tiny.toml"
    tiny.write_text(
        run_config.snapshot(
            args, section, CONFIG, trainer="scripts/train_conditioned.py"
        ),
        encoding="utf-8",
    )
    finished = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_conditioned.py"),
            str(tiny),
            "--section",
            section,
            "--device",
            "cpu",
            "--max-seconds",
            "0.001",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=900,
    )
    assert finished.returncode == 64, (
        f"exited {finished.returncode}, so the Slurm chain would stop.\n"
        f"{finished.stdout[-3000:]}\n{finished.stderr[-3000:]}"
    )
    out = finished.stdout
    # The startup block the pre-registration requires, and the printing
    # conventions: class counts once, and the arm named as diagnostic.
    assert "DIAGNOSTIC" in out
    assert "class counts over" in out
    assert "5476 parameters" in out
    assert "CATH" in out, "the label must be named as ours and not as CATH's"
    assert "exiting 64" in out
