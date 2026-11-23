"""``native_edges`` reaches the rollout, and changes the edge set it perceives over.

**Why this file exists rather than a config test.** ``tests/test_config.py``
proves the setting can be *expressed*: the schema parses it and a section can
state it. That is not the same claim as the setting doing anything, and the
difference has already cost this project retractions -- ``--damage`` was
silently dropped under any ``--birth`` but ``none``, and job 32241769's
diagnostic announced ``birth free`` for a ``--birth none`` run. A setting that
appears to be honoured and is not produces a plausible number under the wrong
tag, which is the worst failure mode available here.

So the assertion is on what ``torch_graph.build_graph`` is actually handed
inside ``step_once``, captured at every step of a real ``growth_objective``
rollout. Three things have to hold, and the first two are the ones that fail if
the wiring is cut anywhere between the trainer's flag and the graph:

  1. with the setting on, every step is handed a ``FixedNeighbours``, and the
     neighbour list the graph is built with is byte-identical at every step --
     a selection that is being rebuilt from live coordinates cannot be, because
     the damaged window is moving;
  2. that list is the one the NATIVE coordinates select, not the live ones;
  3. with the setting off, nothing is handed and the list is *not* the native
     one -- without which (1) and (2) could both pass on a structure whose
     native and live selections happen to agree, and the test would be
     measuring nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca import rollout as rollout_module  # noqa: E402
from aanca.growth import (  # noqa: E402
    GrowthConfig,
    growth_objective,
    native_edge_selection,
)
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import RolloutConfig  # noqa: E402
from aanca.structure import load_protein  # noqa: E402
from aanca.torch_graph import FixedNeighbours, feature_topology  # noqa: E402

PDB = ROOT / "data" / "raw" / "1UBQ.pdb"

STEPS = 6
K = 16


@pytest.fixture(scope="module")
def protein():
    if not PDB.exists():
        pytest.skip(f"no {PDB}")
    return feature_topology(load_protein(str(PDB)))


@pytest.fixture(scope="module")
def rule():
    """A rule that actually moves atoms.

    An untrained ``AANCARule`` is the identity map -- the delta head's last
    layer is zero-initialised -- so its live selection would never change and
    "the selection is held" would be trivially true. The same reason
    ``tests/test_growth_pool.py`` perturbs its weights.
    """
    r = AANCARule(growth=True, birth="none")
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in r.parameters():
            parameter.add_(
                torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                ) * 0.05
            )
    r.eval()
    return r


def config() -> RolloutConfig:
    return RolloutConfig(
        steps=STEPS, truncate=STEPS, k=K, growth=GrowthConfig(birth="none")
    )


def run(rule, topo, native, native_edges: bool, monkeypatch):
    """``growth_objective`` once, returning what every step handed ``build_graph``.

    ``(handed, selected)`` -- the ``fixed_neighbours`` argument per step, and
    the neighbour list the graph was actually built with per step. Patched in
    ``aanca.rollout``'s namespace, which is where ``step_once`` looks the name
    up, so the real ``build_graph`` still does the work.
    """
    handed: list[FixedNeighbours | None] = []
    selected: list[torch.Tensor] = []
    real = rollout_module.build_graph

    def spy(*args, **kwargs):
        handed.append(kwargs.get("fixed_neighbours"))
        graph = real(*args, **kwargs)
        selected.append(graph.neighbours.clone())
        return graph

    monkeypatch.setattr(rollout_module, "build_graph", spy)
    growth_objective(
        rule, native, topo, config(),
        generator=torch.Generator().manual_seed(0),
        damage=12,
        fape_weight=1.0,
        native_edges=native_edges,
    )
    assert selected, "the rollout built no graph at all"
    return handed, selected


def test_native_edges_holds_the_selection_at_the_native_edge_set(
    rule, protein, monkeypatch
):
    """On: every step gets the native selection, and it does not move."""
    topo, native = protein
    native_list = native_edge_selection(native, topo, config()).neighbours

    handed, selected = run(rule, topo, native, True, monkeypatch)

    assert all(isinstance(each, FixedNeighbours) for each in handed), (
        "a step built its graph with no fixed selection, so the setting did "
        "not reach step_once: "
        f"{sum(each is None for each in handed)} of {len(handed)} steps"
    )
    for step, got in enumerate(selected):
        assert torch.equal(got, native_list), (
            f"step {step} perceived over a different edge set from the one the "
            "native coordinates select, so the selection is being recomputed "
            "somewhere despite the setting"
        )


def test_without_it_the_selection_is_rebuilt_from_the_live_coordinates(
    rule, protein, monkeypatch
):
    """Off: nothing is handed, and the edge set is NOT the native one.

    This is what makes the test above a measurement. If the two selections
    agreed on this structure -- they do not; a 12-residue window starting ~13 A
    out of place is the regeneration task -- the assertion that the rollout
    perceives the native set would hold whether or not the setting did
    anything.
    """
    topo, native = protein
    native_list = native_edge_selection(native, topo, config()).neighbours

    handed, selected = run(rule, topo, native, False, monkeypatch)

    assert all(each is None for each in handed), (
        "a rollout with the setting OFF was handed a fixed selection"
    )
    assert not torch.equal(selected[0], native_list), (
        "the live selection and the native one are identical on this "
        "structure, so this file cannot distinguish them: pick a structure or "
        "a damage window where the baseline actually displaces residues"
    )


def test_the_setting_is_recorded_so_a_probe_can_tell(monkeypatch):
    """It is in ``recorded_args``, like any other setting.

    A checkpoint whose ``args`` do not say which graph the rule was trained
    over cannot be read back: the readout would run it under whichever
    perception the current tree defaults to and print the table under the run's
    own tag. That is the 6 Nov failure that cost a retraction, and it is why
    ``recorded_args`` takes the whole namespace minus the resources rather than
    an allowlist -- this test pins the consequence rather than the mechanism.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_growth", ROOT / "scripts" / "train_growth.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_schema().parse_args(["--native-edges"])
    recorded = module.recorded_args(args)
    assert recorded.get("native_edges") is True
    assert "native_edges" not in module.runtime_args(args), (
        "native_edges is part of what the run IS, not a resource it got"
    )


def test_every_trainer_call_forwards_it():
    """Both ``growth_objective`` call sites in the trainer pass it.

    The gap this closes is specific and this project has fallen into it: the
    flag parses, the objective honours it, and the trainer never passes it --
    so the run records ``native_edges = true``, trains under ordinary k-NN
    perception, and every number is filed under the wrong task. Static, because
    the alternative is a GPU-shaped training step in a unit test; it reads the
    source the trainer actually runs rather than a copy.

    The *held-out* call site matters as much as the training one: a mismatch
    there makes the reported curve a different experiment from the run, which
    is how g8_ceiling's first 1,250 updates became unreadable.
    """
    import ast

    tree = ast.parse((ROOT / "scripts" / "train_growth.py").read_text("utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "growth_objective"
    ]
    assert len(calls) == 2, (
        f"expected the training and held-out call sites, found {len(calls)}: "
        "a new one must forward native_edges too"
    )
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "native_edges" in passed, (
            f"the growth_objective call on line {call.lineno} does not forward "
            "native_edges, so that rollout perceives over the live graph "
            "whatever the config says"
        )


def test_two_selections_for_one_rollout_is_an_error(rule, protein, monkeypatch):
    """``native_edges`` plus an explicit ``fixed_neighbours`` must not pick one.

    ``FixedNeighbours.checked`` refuses loudly rather than recomputing for the
    same reason: a selection that is silently overridden means the rollout is
    not the one the caller asked for.
    """
    topo, native = protein
    fixed = native_edge_selection(native, topo, config())
    with pytest.raises(ValueError, match="two"):
        growth_objective(
            rule, native, topo, config(),
            generator=torch.Generator().manual_seed(0),
            damage=12,
            fape_weight=1.0,
            native_edges=True,
            fixed_neighbours=fixed,
        )
