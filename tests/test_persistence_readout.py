"""The persistence probe must honour the run, and must not call a drift a hold.

`scripts/persistence_readout.py` asks whether the trained E(n)-GNCA port has an
*attractor* rather than a trajectory that looks good at the step count it was
trained on. Three things can make its answer a fiction, and each has a test
here:

  * **It could measure its own defaults.** Four results in this project were
    produced that way (`docs/claims/a-probe-pointed-at-best-pt-silently-used-
    its-own-defaults.md`), so a checkpoint whose recorded settings the probe
    cannot honour has to be refused rather than run. Both halves are pinned: a
    recorded key the schema does not define, and a schema key the checkpoint
    does not carry.
  * **It could report a fixed point that is not one.** The convergence test is
    a threshold on a measured displacement, so the measurement is tested
    against a rule whose output genuinely *is* its input -- the published
    initialisation, where `phi_x`'s output layer is zeroed, emits exactly no
    displacement, so step 1 is already a fixed point and must be reported as
    one.
  * **The displacement could be non-zero for an identity step.** Then every
    rollout would look as though it were still moving and nothing would ever
    converge, which would make the headline claim unfalsifiable in the safe
    direction.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from aanca import checkpoint as ckpt
from aanca.egnca import EGNCARule, EGNCAState, ca_positions, egnca_neighbours
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "data" / "raw" / "1UBQ.pdb"


def _probe():
    spec = importlib.util.spec_from_file_location(
        "persistence_readout", ROOT / "scripts" / "persistence_readout.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["persistence_readout"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _probe()


@pytest.fixture(scope="module")
def schema(probe):
    return probe.trainer_module().build_schema()


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords


def _recorded(schema, **overrides) -> dict:
    """A complete, honourable `args` dict: every schema key at its own default.

    Built from the schema rather than written out, so a setting added to the
    trainer appears here automatically and the refusal tests stay about the
    refusal rather than about this fixture going stale.
    """
    from aanca import config as run_config

    out = {
        action.dest: action.default
        for action in schema._actions
        if action.dest not in ("help",) and action.dest not in run_config.RUNTIME_ONLY
    }
    out["tag"] = "port"
    out.update(overrides)
    return out


# -- it must not measure its own defaults -----------------------------------


def test_a_checkpoint_with_no_arguments_is_refused(probe, schema, tmp_path):
    """The original fault: a `best.pt` holding nothing but the weights.

    Every rollout setting would then be the probe's own, and a default is
    indistinguishable from a measurement once it is printed under a column
    heading.
    """
    rule = EGNCARule()
    path = tmp_path / "best.pt"
    ckpt.save(path, {"state_dict": rule.state_dict()})
    state = ckpt.load(path)

    with pytest.raises(probe.ProbeRefusal):
        probe.settings_from_checkpoint(state, schema)


def test_a_recorded_setting_the_schema_cannot_honour_is_refused(probe, schema):
    """A key the probe has no name for: refuse rather than ignore it.

    A setting the checkpoint recorded and the code no longer defines means the
    weights were trained under a rule this probe cannot reconstruct. Ignoring
    it would measure a *different* rule and announce nothing.
    """
    recorded = _recorded(schema, actuator_cap=0.3)

    with pytest.raises(probe.ProbeRefusal) as refusal:
        probe.settings_from_checkpoint({"args": recorded}, schema)
    assert "actuator_cap" in str(refusal.value)


def test_a_schema_setting_the_checkpoint_omits_is_refused(probe, schema):
    """The other half, and it is where a silent default would actually enter."""
    recorded = _recorded(schema)
    recorded.pop("steps")

    with pytest.raises(probe.ProbeRefusal) as refusal:
        probe.settings_from_checkpoint({"args": recorded}, schema)
    assert "steps" in str(refusal.value)


def test_a_recorded_value_outside_the_schemas_choices_is_refused(probe, schema):
    """`activation` and `norm_type` are closed sets, and a value outside one
    means the checkpoint and the code disagree about what the setting is."""
    with pytest.raises(probe.ProbeRefusal):
        probe.settings_from_checkpoint(
            {"args": _recorded(schema, activation="relu")}, schema
        )


def test_the_run_s_own_settings_are_accepted_and_reach_the_rule(probe, schema):
    """The accepting path, and that it hands the trainer's two functions the
    run's values rather than the schema's.

    This is the positive half of the refusal: a probe that refused everything
    would also never measure its own defaults.
    """
    recorded = _recorded(schema, steps=25, k=24, damage=12, activation="tanh")
    args = probe.settings_from_checkpoint({"args": recorded}, schema)
    assert args.steps == 25 and args.k == 24 and args.damage == 12

    trainer = probe.trainer_module()
    rule = trainer.build_rule(args)
    assert rule.activation == "tanh" and rule.norm_type == "pn"
    assert rule.hidden_dim == recorded["hidden_dim"]
    objective = trainer.objective_kwargs(args)
    assert objective["k"] == 24 and objective["damage"] == 12


def test_a_checkpoint_over_a_different_pool_is_refused(probe, schema, tmp_path):
    """Recorded `eval_pdb_ids` that do not match the catalogue on disk.

    Same crop index, different protein -- a laptop pool and a cluster pool give
    `available[:416]` a different 416 proteins, so a number from one is not
    comparable with a number from the other. The probe must not quietly score
    the wrong structures.
    """

    class Spec:
        def __init__(self, pdb_id):
            self.pdb_id = pdb_id
            self.path = Path("nowhere")
            self.group = None

    trainer = argparse.Namespace(on_disk=lambda specs: [Spec("9XYZ"), Spec("8ABC")])
    args = probe.settings_from_checkpoint(
        {"args": _recorded(schema, structures=1, eval_structures=1)}, schema
    )
    state = {"eval_indices": [0], "eval_pdb_ids": ["1UBQ"]}

    class _Dataset:
        def __init__(self, *rest, **kwargs):
            pass

    probe_dataset = probe.RepackingDataset
    probe.RepackingDataset = _Dataset
    try:
        with pytest.raises(probe.ProbeRefusal) as refusal:
            probe.held_out(state, args, trainer, torch.device("cpu"))
    finally:
        probe.RepackingDataset = probe_dataset
    assert "not the run's pool" in str(refusal.value)


# -- it must not call a drift a hold ----------------------------------------


def test_the_displacement_is_zero_for_an_identity_step(probe):
    """An unmoved state has moved nothing, in both the max and the mean.

    If this were non-zero -- off by a dtype cast, or read against the wrong
    tensor -- no rollout would ever fall below the 0.1 A threshold and the
    probe would report "never converged" for every crop, which is the
    unfalsifiable-in-the-safe-direction failure.
    """
    ca = torch.randn(40, 3, dtype=torch.float32) * 10.0
    assert probe.step_displacement(ca, ca.clone()) == (0.0, 0.0)

    mask = torch.zeros(40, dtype=torch.bool)
    mask[3:15] = True
    assert probe.step_displacement(ca, ca.clone(), mask) == (0.0, 0.0)

    # And it is the per-residue Euclidean length, not a component or a mean of
    # components: one residue moved 3 A along one axis reads 3.0.
    moved = ca.clone()
    moved[7, 0] += 3.0
    largest, average = probe.step_displacement(ca, moved, mask)
    assert largest == pytest.approx(3.0, abs=1e-5)
    assert average == pytest.approx(3.0 / 12, abs=1e-5)


def test_a_rule_whose_output_is_its_input_is_detected_as_a_fixed_point(
    probe, ubiquitin
):
    """The published initialisation emits nothing, so step 1 is a fixed point.

    `EGNCARule.__init__` zeroes `phi_x`'s output layer, as the authors' `EGC`
    does, so the coordinate update is exactly `x' = x`: the rule is handed its
    own output at every step. The probe must report `converged_at == 1` and a
    flat zero displacement -- and this is the assertion that keeps the
    convergence test a measurement rather than a step counter, because a probe
    that reported `None` here could never report a genuine fixed point either.
    """
    topo, coords = ubiquitin
    rule = EGNCARule()
    rule.eval()
    ca = ca_positions(coords, topo).to(rule.dtype)
    selection = egnca_neighbours(coords, topo, k=24)
    damaged = torch.zeros(topo.n_residues, dtype=torch.bool)
    damaged[10:22] = True

    with torch.no_grad():
        trace, end = probe.trace_rollout(
            rule,
            EGNCAState(
                ca=ca, hidden=rule.initial_hidden(topo.n_residues, dtype=rule.dtype)
            ),
            lambda state: selection,
            coords,
            topo,
            damaged,
            marks=(1, 5, 20),
            tol=probe.FIXED_POINT_TOL,
        )

    assert trace.converged_at == 1, "a rule that emits nothing is already at rest"
    assert trace.nonfinite_at is None
    assert end is not None
    for step in (1, 5, 20):
        assert trace.marks[step]["finite"]
        assert trace.marks[step]["max_step"] == 0.0
        assert trace.marks[step]["mean_step"] == 0.0
    # The coordinates did not move, so neither did the metric: no drift.
    assert trace.drift() == pytest.approx(0.0, abs=1e-9)
    assert torch.equal(end.ca, ca)


def test_a_moving_rule_is_not_reported_as_a_fixed_point(probe, ubiquitin):
    """The other side of the same threshold, so it is not vacuously satisfied.

    Perturbing `phi_x`'s output layer is what `tests/test_egnca.py::_rule`
    does and for the same reason: at the published initialisation every "does
    it move" question is about the number zero. With a real displacement the
    probe must *not* declare an immediate fixed point.
    """
    topo, coords = ubiquitin
    torch.manual_seed(0)
    rule = EGNCARule()
    for parameter in rule.last_coord_layer.parameters():
        parameter.data.add_(torch.randn_like(parameter) * 0.3)
    rule.eval()
    ca = ca_positions(coords, topo).to(rule.dtype)
    selection = egnca_neighbours(coords, topo, k=24)
    damaged = torch.zeros(topo.n_residues, dtype=torch.bool)
    damaged[10:22] = True

    with torch.no_grad():
        trace, _end = probe.trace_rollout(
            rule,
            EGNCAState(
                ca=ca, hidden=rule.initial_hidden(topo.n_residues, dtype=rule.dtype)
            ),
            lambda state: selection,
            coords,
            topo,
            damaged,
            marks=(1, 5),
            tol=probe.FIXED_POINT_TOL,
        )

    assert trace.marks[1]["max_step"] > probe.FIXED_POINT_TOL
    assert trace.converged_at != 1


# -- the table's own rules --------------------------------------------------


def test_the_aggregate_counts_a_lost_crop_rather_than_averaging_round_it(probe):
    """`n` and `diverged` at *every* step count, because a crop can be finite
    at 25 and gone at 1,000.

    A mean over survivors is not a result, and a single `diverged` for the
    whole table would hide exactly the drift this probe exists to find.
    """
    crops = [
        {
            "from_start": {
                "marks": {
                    "25": {
                        "finite": True,
                        "rmsd_dmg": 2.0,
                        "rmsd_dmg_fit_intact": 2.1,
                        "max_step": 0.05,
                        "mean_step": 0.01,
                    },
                    "1000": {
                        "finite": True,
                        "rmsd_dmg": 4.0,
                        "rmsd_dmg_fit_intact": 4.1,
                        "max_step": 0.05,
                        "mean_step": 0.01,
                    },
                },
                "converged_at": None,
                "drift": None,
            }
        },
        {
            "from_start": {
                "marks": {
                    "25": {
                        "finite": True,
                        "rmsd_dmg": 3.0,
                        "rmsd_dmg_fit_intact": 3.1,
                        "max_step": 0.05,
                        "mean_step": 0.01,
                    },
                    "1000": {
                        "finite": False,
                        "rmsd_dmg": None,
                        "rmsd_dmg_fit_intact": None,
                        "max_step": None,
                        "mean_step": None,
                    },
                },
                "converged_at": None,
                "drift": None,
            }
        },
    ]
    rows = {row["steps"]: row for row in probe.aggregate(crops, "from_start", (25, 1000))}

    assert rows[25]["n"] == 2 and rows[25]["diverged"] == 0
    assert rows[1000]["n"] == 1 and rows[1000]["diverged"] == 1
    assert rows[25]["rmsd_dmg"]["mean"] == pytest.approx(2.5)
    assert rows[25]["rmsd_dmg"]["min"] == pytest.approx(2.0)
    # The 1,000-step row is the survivor's own number, and `n` 1 says so.
    assert rows[1000]["rmsd_dmg"]["mean"] == pytest.approx(4.0)
