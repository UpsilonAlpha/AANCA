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


def test_snapshot_is_complete_on_its_own(schema, data, tmp_path):
    """A snapshot must not need the base it came from."""
    args = run_config.to_namespace(
        schema, run_config.resolve(data, "g13_natedge"), "g13_natedge", RUNTIME
    )
    path = tmp_path / "config.toml"
    path.write_text(run_config.snapshot(args, "g13_natedge", CONFIG), encoding="utf-8")
    run_config.check_base_is_complete(
        {"base": run_config.load(path)["g13_natedge"]}, schema
    )


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
