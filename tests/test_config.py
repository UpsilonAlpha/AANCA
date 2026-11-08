"""The config loader, and that the ported arms are the arms that actually ran.

The second half is the one that matters. Moving from command-line flags to
named sections is only safe if each section resolves to *exactly* the settings
its run used, and the authority for that is not memory or HANDOFF: it is the
``args :`` line each job printed into its own Slurm log. The three lines below
are transcribed verbatim from

    /scratch/user/uquadhik/AANCA/logs/aanca-gpu-32489939.out
    /scratch/user/uquadhik/AANCA/logs/aanca-gpu-32489940.out
    /scratch/user/uquadhik/AANCA/logs/aanca-gpu-32494313.out

and parsed with the same schema the trainer uses, so a config that would have
changed an arm fails here rather than quietly producing a fourth curve on the
same axis.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from aanca import config as run_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "regeneration.toml"


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_growth", ROOT / "scripts" / "train_growth.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: What each job's own log says it was run with. Do not edit these to make a
#: test pass -- they are the record of what happened.
RECORDED = {
    "g13_natedge": (
        "--tag g13_natedge --split train --updates 2000 --eval-every 250 "
        "--seeds 16 --steps 120 --resume --birth none --initial helix "
        "--damage 12 --fape-weight 0.1 --clash-weight 1.0 --move-weight 1.0 "
        "--identity-weight 0 --equivariant-move --lab-rotation --native-edges"
    ),
    "g14_pairwise": (
        "--tag g14_pairwise --split train --updates 2000 --eval-every 250 "
        "--seeds 16 --steps 120 --resume --birth none --initial helix "
        "--damage 12 --fape-weight 0.1 --clash-weight 1.0 --move-weight 1.0 "
        "--identity-weight 0 --equivariant-move --lab-rotation "
        "--pairwise-weight 0.01"
    ),
    "g15_both": (
        "--tag g15_both --split train --updates 2000 --eval-every 250 "
        "--seeds 16 --steps 120 --resume --birth none --initial helix "
        "--damage 12 --fape-weight 0.1 --clash-weight 1.0 --move-weight 1.0 "
        "--identity-weight 0 --equivariant-move --lab-rotation "
        "--native-edges --pairwise-weight 0.01"
    ),
}

RUNTIME = argparse.Namespace(
    device="auto", max_seconds=0.0, stop_file=None, resume=True
)


@pytest.fixture(scope="module")
def schema():
    return _trainer().build_schema()


@pytest.fixture(scope="module")
def data():
    return run_config.load(CONFIG)


@pytest.mark.parametrize("section", sorted(RECORDED))
def test_section_reproduces_the_run_that_happened(section, schema, data):
    old = schema.parse_args(RECORDED[section].split())
    inputs = run_config.resolve(data, section)
    new = run_config.to_namespace(schema, inputs, section, RUNTIME)

    # `resume` is runtime on the new path and was a flag on the old one; the
    # recorded lines all carry it, and RUNTIME sets it, so they agree. Compare
    # everything.
    differ = {
        key: (getattr(old, key), getattr(new, key))
        for key in vars(old)
        if getattr(old, key) != getattr(new, key)
    }
    assert not differ, (
        f"section {section} would not reproduce job's own recorded args: "
        f"{differ}"
    )


def test_every_recorded_arm_has_a_section(data):
    assert set(RECORDED) <= set(run_config.sections(data))


def test_base_states_every_setting(schema, data):
    run_config.check_base_is_complete(data, schema)


def test_base_missing_a_key_is_an_error(schema):
    data = {"base": {"inputs": {"split": "train"}}, "arm": {"inputs": {}}}
    with pytest.raises(run_config.ConfigError, match="state every setting"):
        run_config.check_base_is_complete(data, schema)


def test_unknown_key_is_refused_with_a_suggestion(schema, data):
    inputs = dict(run_config.resolve(data, "g15_both"))
    inputs["pairwise_wieght"] = 0.01
    with pytest.raises(run_config.ConfigError) as exc:
        run_config.to_namespace(schema, inputs, "g15_both", RUNTIME)
    assert "pairwise_weight" in str(exc.value)


def test_a_flag_must_be_a_bool(schema, data):
    inputs = dict(run_config.resolve(data, "g13_natedge"))
    inputs["native_edges"] = "yes"
    with pytest.raises(run_config.ConfigError, match="true or false"):
        run_config.to_namespace(schema, inputs, "g13_natedge", RUNTIME)


def test_an_int_key_refuses_a_float(schema, data):
    inputs = dict(run_config.resolve(data, "g13_natedge"))
    inputs["updates"] = 2000.5
    with pytest.raises(run_config.ConfigError, match="must be an integer"):
        run_config.to_namespace(schema, inputs, "g13_natedge", RUNTIME)


def test_tag_cannot_be_set_in_a_section(schema, data):
    inputs = dict(run_config.resolve(data, "g13_natedge"))
    inputs["tag"] = "something_else"
    with pytest.raises(run_config.ConfigError, match="derived"):
        run_config.to_namespace(schema, inputs, "g13_natedge", RUNTIME)


def test_a_resource_cannot_be_set_in_a_section(schema, data):
    inputs = dict(run_config.resolve(data, "g13_natedge"))
    inputs["device"] = "cpu"
    with pytest.raises(run_config.ConfigError, match="names a resource"):
        run_config.to_namespace(schema, inputs, "g13_natedge", RUNTIME)


def test_tag_is_the_section_name(schema, data):
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "g14_pairwise"), "g14_pairwise", RUNTIME
    )
    assert args.tag == "g14_pairwise"


def test_unknown_section_names_the_known_ones(data):
    with pytest.raises(run_config.ConfigError) as exc:
        run_config.resolve(data, "g15_bath")
    assert "g15_both" in str(exc.value)


def test_base_is_not_runnable(data):
    with pytest.raises(run_config.ConfigError, match="not an arm"):
        run_config.resolve(data, "base")


def test_inherits_chain_overlays_in_order(schema):
    data = {
        "base": {"inputs": {"split": "train", "updates": 10}},
        "a": {"inputs": {"updates": 20, "steps": 5}},
        "b": {"inherits": "a", "inputs": {"steps": 7}},
    }
    merged = run_config.resolve(data, "b")
    assert merged == {"split": "train", "updates": 20, "steps": 7}


def test_inherits_cycle_is_an_error():
    data = {
        "a": {"inherits": "b", "inputs": {}},
        "b": {"inherits": "a", "inputs": {}},
    }
    with pytest.raises(run_config.ConfigError, match="cycle"):
        run_config.resolve(data, "a")


def test_snapshot_round_trips(schema, data, tmp_path):
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "g15_both"), "g15_both", RUNTIME
    )
    path = tmp_path / "config.toml"
    path.write_text(run_config.snapshot(args, "g15_both", CONFIG), encoding="utf-8")

    again = run_config.to_namespace(
        schema, run_config.resolve(run_config.load(path), "g15_both"),
        "g15_both", RUNTIME,
    )
    assert vars(again) == vars(args)


def test_a_snapshot_can_be_run_as_a_config(schema, data, tmp_path):
    """A snapshot must go back in through the door the trainer uses.

    This test used to call ``check_base_is_complete({"base": load(path)[tag]})``
    -- rearranging the data by hand into the shape the loader wanted, which the
    loader will not do for itself. It passed while
    ``train_growth.py <snapshot> --section g15_both`` raised ConfigError,
    because the snapshot had no ``[base]`` at all. It tested the content and
    asserted away the interface. Now it walks the trainer's own path, in the
    trainer's own order.
    """
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "g13_natedge"), "g13_natedge", RUNTIME
    )
    path = tmp_path / "config.toml"
    path.write_text(run_config.snapshot(args, "g13_natedge", CONFIG), encoding="utf-8")

    reloaded = run_config.load(path)
    run_config.check_base_is_complete(reloaded, schema)       # as main() does
    again = run_config.to_namespace(
        schema, run_config.resolve(reloaded, "g13_natedge"), "g13_natedge", RUNTIME
    )
    assert vars(again) == vars(args)


#: Every setting's default **as the pre-port trainer declared it**, read out of
#: ``git show 3b97972^:scripts/train_growth.py`` with ``ast`` rather than from
#: memory. The three weights come from ``src/aanca/growth.py``'s module
#: constants, which the port did not touch.
#:
#: Why this exists: ``RECORDED`` above carries about twelve settings and the
#: schema has forty-five. For the other thirty-three,
#: ``test_section_reproduces_the_run_that_happened`` was comparing
#: ``action.default`` against a base value that had itself been transcribed
#: from those same defaults -- both sides of the assertion reading the same
#: live object. A default that changed between g13 and today would pass that
#: test with both sides wrong together. These numbers are the independent
#: half, and they are in git rather than on a cluster.
HISTORICAL_DEFAULTS = {
    "activation": "sample", "activation_init": -2.0, "birth": "cartesian",
    "bond_weight": 0.122, "clash_tolerance": 0.5, "clash_weight": 0.0,
    "clip": 1.0, "completion_weight": 5.23, "coupling_radius": 5.0,
    "crop_size": 128, "cward_only": False, "damage": 0,
    "equivariant_move": False, "eval_activation": "sample", "eval_every": 500,
    "eval_structures": 12, "fape_clamp": 10.0, "fape_clamp_prob": 1.0,
    "fape_weight": 1.0, "identity_keep": 2.0, "identity_sampling": "sample",
    "identity_start": "masked", "identity_track": "auto",
    "identity_weight": 0.0, "init_from": "", "initial": "helix", "k": 32,
    "lab_rotation": False, "lr": 0.0003, "max_births": 8, "max_rotation": 0.1,
    "max_translation": 0.3, "morphogen": False, "move_weight": 0.0,
    "native_edges": False, "orientation_weight": 0.0, "pairwise_weight": 0.0,
    "placement_rotation": 3.14159265, "placement_steps": 0,
    "placement_translation": 4.0, "reach_weight": 0.086, "seed": 0,
    "seeds": 4, "split": "train", "steps": 40, "structures": 400,
    "truncate": 12, "update_schedule": "bernoulli", "updates": 4000,
}


def test_the_schema_still_declares_the_defaults_the_arms_ran_under(schema):
    """A default that drifted silently reinterprets every recorded arm.

    ``configs/regeneration.toml``'s base was transcribed from the schema's
    defaults, so if one of them moves, the base moves with it and the arms'
    meaning changes with no diff anywhere. This is the only assertion in the
    file whose right-hand side does not come from the live code.
    """
    live = vars(schema.parse_args([]))
    drifted = {
        key: (was, live[key])
        for key, was in sorted(HISTORICAL_DEFAULTS.items())
        if key in live and live[key] != was
    }
    assert not drifted, (
        "a schema default no longer matches what g13/g14/g15 ran under: "
        f"{drifted}. If the change is deliberate, pin the old value in the "
        "affected sections of configs/regeneration.toml FIRST, then update "
        "this dict in the same commit."
    )


@pytest.mark.parametrize("section", sorted(RECORDED))
def test_unstated_settings_resolve_to_the_historical_defaults(section, schema, data):
    """The thirty-three settings no recorded command line mentions.

    ``bunya_train.slurm`` echoes ``args : $*``, i.e. the explicit flags only,
    so for the settings an arm did not state there is no record anywhere of
    what was in force -- only the code at that commit. This asserts the config
    against that, so the base cannot quietly disagree with the run.
    """
    stated = {
        token[2:].replace("-", "_")
        for token in RECORDED[section].split()
        if token.startswith("--")
    }
    resolved = run_config.resolve(data, section)
    wrong = {
        key: (resolved[key], was)
        for key, was in sorted(HISTORICAL_DEFAULTS.items())
        if key not in stated and key in resolved and resolved[key] != was
    }
    assert not wrong, f"{section} would not reproduce the defaults in force: {wrong}"


def test_g16_is_a_single_variable_arm_against_g15(schema, data):
    """The property the config system exists to make checkable."""
    g15 = run_config.to_namespace(
        schema, run_config.resolve(data, "g15_both"), "g15_both", RUNTIME
    )
    g16 = run_config.to_namespace(
        schema, run_config.resolve(data, "g16_localrot"), "g16_localrot", RUNTIME
    )
    differ = {
        key
        for key in vars(g15)
        if key != "tag" and getattr(g15, key) != getattr(g16, key)
    }
    assert differ == {"lab_rotation"}


def test_each_arm_differs_from_its_parent_in_at_most_two_settings(schema, data):
    """A section states only its difference from what it inherits.

    Against its *parent*, not against the base: g16_localrot inherits g15_both
    and so differs from the base in three settings while differing from the arm
    it is a variant of in one, which is the property that matters. A section
    that changes more than a couple of things at once is an arm whose result
    cannot be attributed, and that is what this bound is for.
    """
    for section in run_config.sections(data):
        parent = data[section].get("inherits")
        reference = (
            run_config.resolve(data, parent)
            if parent
            else dict((data.get("base") or {}).get("inputs") or {})
        )
        inputs = run_config.resolve(data, section)
        differ = {k for k, v in inputs.items() if reference.get(k) != v}
        assert len(differ) <= 2, f"{section} changes {sorted(differ)}"


def test_resume_drift_is_empty_when_the_config_still_describes_the_run(schema, data):
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "g15_both"), "g15_both", RUNTIME
    )
    stored = {
        k: v for k, v in vars(args).items() if k not in run_config.RUNTIME_ONLY
    }
    assert run_config.resume_drift(stored, args) == {}


def test_resume_drift_catches_a_section_edited_mid_chain(schema, data):
    """The failure this check exists for: g15's weights, g16's settings.

    Edit a section while a 24 h chain is in flight and the next segment used
    to continue a different experiment on the previous segment's weights,
    silently, on the same curve. runs/ is gitignored, so there was no record
    either.
    """
    g15 = run_config.to_namespace(
        schema, run_config.resolve(data, "g15_both"), "g15_both", RUNTIME
    )
    g16 = run_config.to_namespace(
        schema, run_config.resolve(data, "g16_localrot"), "g16_localrot", RUNTIME
    )
    stored = {k: v for k, v in vars(g15).items() if k not in run_config.RUNTIME_ONLY}
    drift = run_config.resume_drift(stored, g16)
    assert set(drift) == {"lab_rotation", "tag"}
    assert drift["lab_rotation"] == (True, False)


def test_resume_drift_allows_extending_a_run_and_ignores_resources(schema, data):
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "g15_both"), "g15_both", RUNTIME
    )
    stored = {k: v for k, v in vars(args).items() if k not in run_config.RUNTIME_ONLY}
    stored["updates"] = 2000
    args.updates = 10000
    args.max_seconds = 86000.0
    args.device = "cpu"
    assert run_config.resume_drift(stored, args) == {}
