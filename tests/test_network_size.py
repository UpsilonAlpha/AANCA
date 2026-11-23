"""The network's size is a setting, and the setting reaches the model.

Two things are asserted here and they are not the same thing. The first is that
``hidden_dim`` and ``message_dim`` *do* something: a section that halves them
gets a smaller rule, and the counts are pinned so a change to the architecture
that moves them has to come through this file. The second is the one the
project keeps paying for -- that the keys are **wired**, not merely parsed.
A setting that appears to be used and is not has cost retractions here (job
32405172: ``move`` and ``clash`` were computed ``if weight:`` while ``score()``
never passed the weight, so a held-out column was structurally absent and
printed as a measurement), so these tests construct the rule through
``train_growth.build_rule`` -- the same call ``main`` makes -- rather than
calling ``AANCARule`` directly, and they go through the config loader rather
than through a hand-built namespace.

Cut the wiring and they fail. Measured, by hardcoding ``hidden_dim=96`` in
``build_rule`` instead of ``args.hidden_dim``:
``test_the_size_settings_change_the_parameter_count`` fails with
``30180 != 229588`` at the small configuration, and
``test_each_size_key_moves_the_count_on_its_own`` fails on the ``hidden_dim``
half while still passing on ``message_dim`` -- which is why the two keys are
also tested one at a time.

**Why there is a floor, and why it is 17,552 rather than the published 5,000.**
The published rule that succeeds at the analogous task has ~5 kparam
(``docs/claims/the-port-must-reproduce-the-published-result-on-point-clouds-
before-it-is-run-on-proteins.md``). It is not reachable from these two keys and
that is not a defect in them: every head in ``AANCARule`` is
``Linear(state_dim, 64)`` with the 64 a literal, ``state_dim`` is
``node_dim + hidden_dim``, and a growth rule's ``node_dim`` is 56 -- the 32
featuriser channels plus the 24-channel closure block. Four heads at
``hidden_dim = 1`` therefore cost ~14.6 kparam on their own, before a single
message is passed. Reaching 5 kparam needs the head width exposed or the
featuriser narrowed, and neither is a width argument the constructor has.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from aanca import config as run_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "regeneration.toml"

RUNTIME = argparse.Namespace(
    device="auto", max_seconds=0.0, stop_file=None, resume=False
)

#: ``build_rule`` at the default pair, and at the smallest pair the two keys
#: can express. Both measured on this tree; the first is the count every arm
#: from g13 to g28 was built at, because the trainer had no key for either
#: width and took ``AANCARule``'s own defaults.
DEFAULT_COUNT = 229_588
FLOOR_COUNT = 17_552


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_growth", ROOT / "scripts" / "train_growth.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainer():
    return _trainer()


@pytest.fixture(scope="module")
def schema(trainer):
    return trainer.build_schema()


@pytest.fixture(scope="module")
def data():
    return run_config.load(CONFIG)


def _rule(trainer, schema, data, **overrides):
    """A rule built the way a run builds it: config section -> namespace -> rule.

    Through ``g25_equivariant`` rather than a bare namespace, so the path under
    test is the one a launch takes -- the loader's type checking included.
    """
    inputs = dict(run_config.resolve(data, "g25_equivariant"))
    inputs.update(overrides)
    args = run_config.to_namespace(schema, inputs, "probe", RUNTIME)
    return trainer.build_rule(args)


def test_the_default_is_the_size_every_recorded_arm_was_built_at(
    trainer, schema, data
):
    """229,588 -- the number the port exists to bring down.

    Pinned because it is quoted in two claim documents as *ours*, against the
    published ~5,000. If the architecture changes, this is where the quoted
    number has to be updated rather than being left stale in prose.
    """
    rule = _rule(trainer, schema, data)
    assert (rule.hidden_dim, rule.message_dim) == (96, 128)
    assert rule.parameter_count() == DEFAULT_COUNT


def test_the_size_settings_change_the_parameter_count(trainer, schema, data):
    """The assertion that the keys are not decoration.

    Strictly monotone over the whole range, and both ends pinned exactly: a
    bound would pass against a hardcoded width as long as it happened to lie
    inside it.
    """
    pairs = [(96, 128), (64, 96), (32, 64), (16, 32), (8, 16), (1, 2)]
    counts = [
        _rule(trainer, schema, data, hidden_dim=h, message_dim=m).parameter_count()
        for h, m in pairs
    ]
    assert counts[0] == DEFAULT_COUNT
    assert counts[-1] == FLOOR_COUNT
    assert counts == sorted(counts, reverse=True), (
        f"the size settings do not order the rule by size: {list(zip(pairs, counts))}"
    )
    assert len(set(counts)) == len(counts), (
        "two different sizes built the same rule, so at least one of the two "
        f"keys is not reaching the model: {list(zip(pairs, counts))}"
    )


@pytest.mark.parametrize(
    "key,other",
    [("hidden_dim", "message_dim"), ("message_dim", "hidden_dim")],
)
def test_each_size_key_moves_the_count_on_its_own(
    trainer, schema, data, key, other
):
    """One at a time, because a pair moved together hides a dead key.

    ``test_the_size_settings_change_the_parameter_count`` moves both at once
    and would still pass with one of them hardcoded -- the same shape of fault
    as a weight that reaches training and not evaluation.
    """
    big = _rule(trainer, schema, data, **{key: 96 if key == "hidden_dim" else 128})
    small = _rule(trainer, schema, data, **{key: 8})
    assert small.parameter_count() < big.parameter_count(), (
        f"{key} does not change the rule's size with {other} held fixed, so "
        "the section cannot set it"
    )


def test_the_keys_are_recorded_in_the_checkpoint_like_any_other_setting(
    trainer, schema, data
):
    """``recorded_args`` is what the checkpoint stores, and it is ``vars(args)``.

    So the assertion is that these are real schema settings rather than
    something the trainer reads from elsewhere: a key absent from
    ``recorded_args`` is a run whose own checkpoint cannot say what size of
    rule is inside it.
    """
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "g25_equivariant"), "probe", RUNTIME
    )
    stored = trainer.recorded_args(args)
    assert stored["hidden_dim"] == 96
    assert stored["message_dim"] == 128
    assert stored["pairwise_window_weight"] == 0.0


def test_a_warm_start_across_a_size_change_is_refused(trainer):
    """Both widths are in ``WARM_START_MUST_MATCH``, and the reason is shapes.

    Unlike every other entry in that list, these two do not *reinterpret* the
    stored weights -- they make them unloadable. ``load_state_dict`` would
    raise a size mismatch over twenty tensors, several screens into a run, and
    the list exists precisely so that is a refusal with the two numbers in it
    instead.
    """
    assert "hidden_dim" in trainer.WARM_START_MUST_MATCH
    assert "message_dim" in trainer.WARM_START_MUST_MATCH
    # And the weight is deliberately not, for the same reason `pairwise_weight`
    # is not: continuing training under a changed objective is something this
    # project does on purpose, and no stored tensor changes shape.
    assert "pairwise_window_weight" not in trainer.WARM_START_MUST_MATCH
    assert "pairwise_weight" not in trainer.WARM_START_MUST_MATCH


def test_splicing_weights_across_a_size_change_really_does_fail(
    trainer, schema, data
):
    """The claim behind the list entry, asserted rather than argued.

    If a warm start across ``hidden_dim`` were harmless the entry would be
    over-strict and someone would eventually remove it. It is not harmless.
    """
    big = _rule(trainer, schema, data)
    small = _rule(trainer, schema, data, hidden_dim=8, message_dim=16)
    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        small.load_state_dict(big.state_dict())


def test_the_published_five_thousand_is_out_of_reach_from_these_keys(
    trainer, schema, data
):
    """**A negative result about the port, recorded where it was measured.**

    The port table says "small network -- 5K succeeded; ours is 229,588" as
    though the gap were a setting away. It is not. At the smallest widths these
    two keys can express the rule still holds 17,552 parameters, and the
    remainder is the four heads' hardcoded width of 64 over a 56-channel node
    featuriser -- neither of which is a width argument ``AANCARule`` exposes.
    Quantified here so the port is launched knowing its floor rather than
    discovering it in a log.
    """
    floor = _rule(trainer, schema, data, hidden_dim=1, message_dim=2)
    assert floor.parameter_count() == FLOOR_COUNT
    assert floor.parameter_count() > 5_000
    heads = sum(
        sum(p.numel() for p in getattr(floor, name).parameters())
        for name in ("delta_head", "kappa_head", "activation_head", "frame_head")
    )
    assert heads > 14_000, (
        "the floor is supposed to be the heads; if it is not, the paragraph "
        f"in this file's docstring is wrong (heads hold {heads})"
    )
