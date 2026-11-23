"""``--native-edges`` has to actually change the edge set, or it measures nothing.

The flag's whole content is one substitution: the rollout's neighbour
*selection* comes from the native coordinates instead of the live ones. If it
were wired up wrongly -- parsed and never passed, passed and silently dropped,
or passed a selection that happens to equal the live one -- the readout would
print a table under a banner announcing a ceiling, and the table would be the
ordinary regeneration number with a different heading over it.

**That exact failure has already cost this project a retraction**: job 32241769
announced ``birth free`` for a ``--birth none`` run, and the four arms whose
``--damage`` was silently dropped under a non-``none`` birth mode. A probe that
appears to honour a setting and does not is worse than one that crashes,
because a default is indistinguishable from a measurement once it is printed in
a column. So this file does not test that the flag exists or that the code path
runs; it records the neighbour tensor the rule is handed at every step of every
crop, through the readout's own ``main()``, and asserts the two runs differ in
the one way they are supposed to.

``FixedNeighbours`` already validates shape, dtype, device and index range and
refuses to fall back -- see its docstring. These tests cover the half that
validation cannot: that the thing being injected is a *different* edge set, and
that it reaches the rule.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import aanca.rollout as rollout_module
from aanca import checkpoint as ckpt
from aanca.growth import GrowthConfig
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology, select_neighbours

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "ceiling_readout", ROOT / "scripts" / "ceiling_readout.py"
)
ceiling_readout = importlib.util.module_from_spec(SPEC)
sys.modules["ceiling_readout"] = ceiling_readout
SPEC.loader.exec_module(ceiling_readout)

NATIVE = ROOT / "data" / "raw" / "1UBQ.pdb"

#: Captured at import, before any patching. Reading
#: ``rollout_module.build_graph`` inside the recorder's installer instead would
#: pick up a recorder left in place by an earlier call in the same test -- the
#: second run then feeds both lists and the counts stop matching, which is how
#: this was found.
REAL_BUILD_GRAPH = rollout_module.build_graph

#: Captured at import for the same reason, and it is the same trap: a test that
#: runs the readout twice shares one ``monkeypatch``, so reading
#: ``ceiling_readout._bearing`` inside the recorder's installer picks up the
#: recorder the first call left in place. The second run then appends to both
#: lists and the pairing comparison is between a two-entry list and a one-entry
#: one, which is how this was found.
REAL_BEARING = ceiling_readout._bearing

#: Small on purpose: these tests are about which residues are talking to each
#: other, and that is settled in the first few steps. The stored arguments are
#: what the readout configures the rollout from, so they double as the record
#: of what the fixed selection has to match -- same ``k``, same crop.
#: Every key here is one ``scripts/train_growth.py``'s schema still has, which
#: ``test_d`` relies on: the readout refuses a checkpoint recording a setting
#: the schema has lost, so a stale key in this fixture would stop the other
#: tests running at all. ``blind`` is deliberately absent -- a real run derives
#: it from ``identity_track`` and never records it, so the readout's
#: ``stored.get("blind", True)`` default is what a real checkpoint gets too.
STORED = {
    "damage": 12,
    "steps": 4,
    "truncate": 12,
    "k": 16,
    "initial": "helix",
    "seed": 0,
    "split": "train",
    "identity_weight": 0.0,
    "crop_size": 128,
}


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords


class OneCrop:
    """The crop source ``held_out`` would build, without needing the pool.

    The readout's pool comes from ``data/manifest.json`` and ``data/pdb``, which
    are not in the repository; 1UBQ is. Standing in here keeps the test offline
    while leaving everything downstream -- the rule, ``growth_objective``, the
    rollout, the selection -- the real code.
    """

    def __init__(self, topo, coords):
        self._topo, self._coords = topo, coords

    def load_crop(self, index, seed_residue=None, rng=None):
        return self._topo, self._coords, None, None


def _checkpoint(tmp_path, extra: dict | None = None):
    """A ``best.pt`` for a self-placing damage run, with a head that moves.

    The frame head is zero-initialised, so an untouched rule emits the zero
    vector, nothing moves and the live edge set never leaves its start value.
    That would still pass the assertions below -- the start state is not the
    native one -- but it would pass them for the wrong reason, with a live
    selection that is constant because the structure is frozen rather than
    because the selection was fixed. Perturbing the head makes the live edge set
    genuinely drift, which is the case the flag exists to remove.
    """
    rule = AANCARule(growth=True, birth="none")
    generator = torch.Generator().manual_seed(11)
    for parameter in rule.frame_head[-1].parameters():
        parameter.data.add_(
            torch.randn(parameter.shape, generator=generator) * 0.5
        )
    path = tmp_path / "best.pt"
    stored = dict(STORED)
    stored.update(extra or {})
    ckpt.save(path, {"state_dict": rule.state_dict(), "args": stored})
    return path


def _readout(
    monkeypatch, tmp_path, topo, coords, native_edges: bool,
    extra: dict | None = None, expect: int = 0,
    bearings: list | None = None, cripple_pairing: bool = False,
):
    """Run the readout's ``main()`` and return every selection the rule saw.

    Recorded by wrapping the ``build_graph`` that ``rollout.step_once`` calls,
    so what is captured is the selection the *rule* was handed, not the one the
    script believes it supplied. A flag that reached the script and not the
    rollout is exactly the bug being tested for, and reading the script's own
    variable back would not see it.
    """
    seen: list[torch.Tensor] = []

    def recording(*args, **kwargs):
        graph = REAL_BUILD_GRAPH(*args, **kwargs)
        seen.append(graph.neighbours.clone())
        return graph

    if bearings is not None:
        def recording_bearing(
            result, native, topo_, damaged, start_coords, config, local=False
        ):
            out = REAL_BEARING(
                result, native, topo_, damaged, start_coords, config,
                local=local,
            )
            need = out[4]
            if cripple_pairing:
                # **The defect, put back.** `need` read off the rollout's own
                # `moves[0]` -- which `rollout.rollout` leaves holding the
                # start of the LAST truncation window, not the start of the
                # rollout. Nothing else about the run changes, so a `need`
                # that moves between the two perceptions under this and not
                # otherwise is the whole of the bug and the whole of the fix.
                from aanca.fape import ca_rmsd_on
                from aanca.growth import backbone_atoms

                first = backbone_atoms(result.moves[0][2], topo_)[:, 1]
                ca_native = backbone_atoms(native, topo_)[:, 1]
                every = torch.ones_like(damaged)
                need = float(ca_rmsd_on(
                    first[every], ca_native[every],
                    first[damaged], ca_native[damaged],
                ))
            bearings.append({
                "start": start_coords.detach().clone(),
                "damaged": damaged.detach().clone(),
                "need": need,
                "align": out[0],
            })
            return out

        monkeypatch.setattr(ceiling_readout, "_bearing", recording_bearing)
    monkeypatch.setattr(rollout_module, "build_graph", recording)
    monkeypatch.setattr(
        ceiling_readout, "held_out",
        lambda stored, state, extra=0: (OneCrop(topo, coords), [0], 1),
    )
    argv = [
        "ceiling_readout.py",
        "--checkpoint", str(_checkpoint(tmp_path, extra)),
    ]
    if native_edges:
        argv.append("--native-edges")
    monkeypatch.setattr(sys, "argv", argv)

    assert ceiling_readout.main() == expect, (
        f"the readout returned something other than {expect}"
    )
    if expect == 0:
        assert seen, "nothing was recorded, so this test asserts nothing"
    return seen


def test_a_the_native_edge_set_differs_from_the_live_one(ubiquitin):
    """The substitution is a real one: native k-NN != start-state k-NN.

    If the two coincided, every other assertion here would hold vacuously and
    the diagnostic would be a no-op with a banner. They do not coincide because
    the damaged window starts about 13 A from where it belongs, which is the
    whole premise of
    ``docs/claims/the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is.md``.
    """
    topo, coords = ubiquitin
    config = RolloutConfig(
        steps=STORED["steps"], truncate=STORED["truncate"], k=STORED["k"],
        # The readout's own default for a key no run records.
        blind=True,
        growth=GrowthConfig(birth="none", initial=STORED["initial"]),
    )
    # **The readout must not carry its own copy of this.** It did until 23 Nov,
    # when ``aanca.growth`` gained the same function for the trainer; two
    # functions whose whole content is "nothing differs but the coordinates" is
    # where a divergence hides in silence, so the readout's name is now an
    # alias and this pins it. The argument is ``carries_identity=False``: the
    # parameter used to be the identity STATE, tested with ``is not None``, so
    # ``False`` turned blinding on -- which is what three tests in this file,
    # this one included, were silently doing. It is a bool read by value now.
    from aanca.growth import native_edge_selection

    assert ceiling_readout.native_selection is native_edge_selection
    fixed = ceiling_readout.native_selection(
        coords, topo, config, carries_identity=False
    )

    live, _mask = select_neighbours(
        coords.to(config.feature_dtype), topo, config.k, config.cutoff
    )
    # Same selector, same k, same coordinates: the helper must not be doing
    # anything of its own beyond choosing which coordinates to select from.
    assert torch.equal(fixed.neighbours, live)

    # And the selection it is standing in for -- the one the rollout computes
    # from a baseline-filled window -- is a different set of edges.
    from aanca.growth import baseline_coords, damage_window, growth_links

    links = growth_links(topo)
    damaged = damage_window(
        topo.n_residues, links, STORED["damage"],
        torch.Generator().manual_seed(STORED["seed"]), coords.device,
    )
    start = baseline_coords(
        coords, topo, links, ~damaged,
        STORED["initial"], torch.Generator().manual_seed(STORED["seed"]),
    )
    start_edges, _m = select_neighbours(
        start.to(config.feature_dtype), topo, config.k, config.cutoff
    )
    assert not torch.equal(start_edges, fixed.neighbours), (
        "the start state already selects the native edge set, so this crop "
        "cannot test the flag"
    )

    # The validator accepts it for the shape the rollout will ask for.
    fixed.checked(topo, config.k, False, coords.device)


def test_b_the_flag_changes_the_edge_set_the_rollout_uses(
    monkeypatch, tmp_path, ubiquitin
):
    """**The point of this file.** With the flag, every step's selection is the
    native one; without it, it is not.

    Three separate assertions, because each catches a different way of being a
    no-op:

      * the flag run's selections are all *identical to each other* -- a flag
        that was dropped would leave them drifting with the coordinates;
      * they are identical to the edge set ``native_selection`` computes -- a
        flag that reached ``build_graph`` with the wrong tensor would pass the
        first check and fail this one;
      * the no-flag run's selections are *not* that set -- which is what makes
        the two runs a comparison rather than two readings of one thing.
    """
    topo, coords = ubiquitin
    config = RolloutConfig(
        steps=STORED["steps"], truncate=STORED["truncate"], k=STORED["k"],
        # The readout's own default for a key no run records.
        blind=True,
        growth=GrowthConfig(birth="none", initial=STORED["initial"]),
    )
    native = ceiling_readout.native_selection(
        coords, topo, config, carries_identity=False
    ).neighbours

    with_flag = _readout(monkeypatch, tmp_path, topo, coords, native_edges=True)
    without = _readout(monkeypatch, tmp_path, topo, coords, native_edges=False)

    assert len(with_flag) == len(without) == STORED["steps"], (
        "both runs must take the same number of steps, or the comparison is "
        "between two different rollouts"
    )
    assert all(torch.equal(s, with_flag[0]) for s in with_flag), (
        "under --native-edges the selection is held, so it cannot change "
        "between steps"
    )
    assert torch.equal(with_flag[0], native), (
        "the held selection is not the one the native coordinates give"
    )
    assert any(not torch.equal(s, native) for s in without), (
        "without the flag the rollout already selected the native edge set at "
        "every step, so passing the flag would change nothing and the "
        "comparison would be vacuous"
    )
    # Stated as the one-line claim as well: the two runs saw different graphs.
    assert any(
        not torch.equal(a, b) for a, b in zip(with_flag, without)
    ), "the two runs used the same edge set, so the flag is a no-op"


def test_c_a_dropped_flag_is_what_this_would_catch(
    monkeypatch, tmp_path, ubiquitin
):
    """Make the flag a no-op on purpose; test b's assertions must then fail.

    The verification of the test rather than of the code. ``native_selection``
    is replaced by one returning ``None``, which is precisely the regression the
    file guards against -- the flag parsed, the banner printed, and nothing
    handed to the rollout -- and the recorded selections must then stop being
    the native ones.
    """
    topo, coords = ubiquitin
    config = RolloutConfig(
        steps=STORED["steps"], truncate=STORED["truncate"], k=STORED["k"],
        # The readout's own default for a key no run records.
        blind=True,
        growth=GrowthConfig(birth="none", initial=STORED["initial"]),
    )
    native = ceiling_readout.native_selection(
        coords, topo, config, carries_identity=False
    ).neighbours

    monkeypatch.setattr(
        ceiling_readout, "native_selection",
        lambda *a, **k: None,
    )
    crippled = _readout(monkeypatch, tmp_path, topo, coords, native_edges=True)

    held = all(torch.equal(s, crippled[0]) for s in crippled) and torch.equal(
        crippled[0], native
    )
    assert not held, (
        "with the flag made a no-op the rollout still used the native edge "
        "set, which means test b would pass on a broken flag and proves "
        "nothing"
    )


def test_d_a_recorded_setting_the_schema_has_lost_is_refused(
    monkeypatch, tmp_path, ubiquitin, capsys
):
    """A key nothing in this tree reads must stop the readout, by name.

    Every rollout setting is pulled out of ``stored`` with ``.get(...)``, so a
    key with no reader is dropped without a word -- which is how
    ``native_edges`` behaved before this check. ``g13_natedge``, ``g15_both``
    and ``g17_long`` all record it, so the readout would have run their weights
    under ordinary k-NN-every-step perception and printed the table under their
    checkpoint path. That is the 6 Nov failure: a probe substituting its own
    behaviour for the run's with nothing in the output to say so.
    """
    topo, coords = ubiquitin
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=False,
        extra={"completely_made_up": 7}, expect=1,
    )
    printed = capsys.readouterr().out
    assert "completely_made_up" in printed, "the refusal must name the key"
    assert "REFUSING" in printed
    assert "crop" not in printed.split("REFUSING")[-1], (
        "a refusal must not also print the table it is refusing to stand behind"
    )


def test_e_a_recorded_perception_is_refused_unless_it_is_reproduced(
    monkeypatch, tmp_path, ubiquitin, capsys
):
    """``native_edges`` recorded ON: refuse, and say which flag reproduces it.

    **The grounds changed on 23 Nov and the refusal did not.** ``native_edges``
    was restored to ``scripts/train_growth.py``'s schema as the task statement
    rather than a leak, so it is an ordinary recorded setting again and
    ``unresolved_settings`` no longer fires on it -- which is exactly the
    condition under which ``stored.get`` drops it in silence, as it did before
    9 Nov. What mattered was never that the key had no reader; it was that those
    weights were trained with the target graph in the rollout, so reading them
    without ``--native-edges`` is a different experiment from the one that
    produced them, printed under the run's tag. Reading them *with* it is a
    reproduction. Both are sayable; silently doing the first is not, and
    ``ceiling_readout.perception_mismatch`` is where that survives.
    """
    topo, coords = ubiquitin
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=False,
        extra={"native_edges": True}, expect=1,
    )
    printed = capsys.readouterr().out
    assert "REFUSING" in printed
    assert "native_edges" in printed and "--native-edges" in printed, (
        "the refusal must name the recorded setting and the flag that "
        "reproduces it"
    )
    assert "crop" not in printed.split("REFUSING")[-1], (
        "a refusal must not also print the table it is refusing to stand behind"
    )

    # With the flag, the run is reproducible and the readout proceeds -- and it
    # says in the log that this is the run's OWN perception rather than a
    # ceiling imposed on an arm that trained without it.
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=True,
        extra={"native_edges": True}, expect=0,
    )
    printed = capsys.readouterr().out
    assert "NATIVE EDGES" in printed
    assert "RECORDED native_edges" in printed, (
        "the log must distinguish reproducing the run's own perception from "
        "imposing a ceiling on an arm that never had it"
    )


def test_f_native_edges_recorded_off_is_not_an_obstacle_either_way(
    monkeypatch, tmp_path, ubiquitin, capsys
):
    """``native_edges = False`` read without the flag is simply the run.

    An arm that recorded the setting and left it off ran the k-NN-every-step
    perception every live-edge arm runs, so there is nothing to reconcile and
    refusing would be a false alarm -- and the log must still state which
    perception was used, because that statement is the only thing in the output
    that attributes the table.

    And the reverse direction must stay open: the same arm read *with*
    ``--native-edges`` is the one-variable ceiling comparison this whole script
    exists for, so it cannot be refused as a mismatch. Only trained-on-native
    read as live is an error, which is what ``perception_mismatch`` encodes.
    """
    topo, coords = ubiquitin
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=False,
        extra={"native_edges": False}, expect=0,
    )
    printed = capsys.readouterr().out
    assert "REFUSING" not in printed
    assert "rebuilt from the live coordinates every step" in printed

    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=True,
        extra={"native_edges": False}, expect=0,
    )
    printed = capsys.readouterr().out
    assert "REFUSING" not in printed, (
        "a live-edge arm read with --native-edges is the ceiling comparison, "
        "not a mismatch, and refusing it would remove the diagnostic"
    )
    assert "CEILING" in printed and "did NOT record native_edges" in printed


def test_g_a_checkpoint_whose_settings_all_resolve_is_untouched(
    monkeypatch, tmp_path, ubiquitin, capsys
):
    """The check must be invisible to every ordinary checkpoint.

    ``g25_equivariant`` and ``g28_ceil_local`` record nothing retired, and a
    readout that started refusing or warning on them would have broken the
    comparison this flag exists for.
    """
    topo, coords = ubiquitin
    _readout(monkeypatch, tmp_path, topo, coords, native_edges=False)
    printed = capsys.readouterr().out
    assert "REFUSING" not in printed
    assert "retired" not in printed.lower()
    assert "recorded settings" not in printed


def test_h_the_authority_is_the_trainers_schema_not_a_list_here(ubiquitin):
    """``unresolved_settings`` reads ``build_schema``, and resolves a real run.

    Two ways this goes quietly wrong. Reading ``build_parser`` -- the three
    options that address the config -- makes *every* recorded key unresolved,
    which is how it was first written and would refuse every checkpoint. A
    hand-kept list of valid keys in the readout goes stale the next time the
    trainer gains a setting, and goes stale silently.
    """
    assert ceiling_readout.unresolved_settings(dict(STORED)) == []
    # ``native_edges`` is a schema key again as of 23 Nov, so it resolves and
    # only the genuinely unknown key is named. That the KEY resolves is not the
    # end of the matter -- test e covers the perception it implies -- but the
    # schema check must report the schema and nothing else, or it is the
    # hand-kept list this function exists to avoid.
    assert ceiling_readout.unresolved_settings(
        {"native_edges": True, "nonsense": 1}
    ) == ["nonsense"]
    assert ceiling_readout.RETIRED_SETTINGS == {}, (
        "nothing is retired today; a non-empty map needs a test of its own"
    )
    # The resource keys are excluded by ``recorded_args`` on the way in, and an
    # older checkpoint that carries them anyway is not drift.
    assert ceiling_readout.unresolved_settings(
        {"device": "cuda", "max_seconds": 3600.0}
    ) == []


#: ``steps`` past ``truncate``, which is the regime every real arm runs in --
#: ``g25_equivariant`` and ``g28_ceil_local`` are 120 steps at truncate 12 --
#: and the regime the pairing defect lived in. ``rollout.rollout`` clears its
#: ``moves`` list at every truncation boundary, so with two segments
#: ``moves[0]`` is the pose four perception-driven steps downstream of the
#: start. The rest of this file runs at ``steps = 4``, a single segment, where
#: ``moves[0]`` happens to BE the start and the defect is invisible: that is
#: why it was not caught here.
TRUNCATED = {"steps": 8, "truncate": 4}


def test_i_the_two_runs_share_one_start_state_and_differ_only_in_perception(
    monkeypatch, tmp_path, ubiquitin
):
    """The comparison is exactly paired: same start, different edge set.

    **This is the property ``--native-edges`` is worthless without.** ``need``
    is the window's distance from native at the start state, which is settled
    before a step is taken and cannot depend on perception -- so if it moves
    between the two runs of one checkpoint, something other than the edge set
    moved too and no difference in ``align`` can be attributed. The four jobs of
    23 Nov show exactly that: ``need`` 15.90 -> 16.14 A on ``g25_equivariant``,
    12.34 -> 12.28 A on ``g28_ceil_local``, which is the same order as
    ``g25``'s ``align`` 0.141 -> 0.208 and is what stopped that being readable
    as a rise.

    Three assertions, and the third is what keeps the first two from being
    satisfied by a readout that does nothing:

      * the damaged window and the start coordinates are **bit-identical** --
        ``torch.equal``, not a tolerance. They are drawn by ``damage_start``
        from a generator seeded off the checkpoint's own seed, above and before
        the perception choice, so there is no float path by which they could
        differ and nothing to justify a tolerance on;
      * ``need`` is identical as a float, for the same reason: identical inputs
        through identical ops;
      * ``align`` **differs**, because it is the one thing that is supposed to.
        Equal ``align`` would mean the flag changed nothing measurable and the
        pairing had been bought by making the comparison vacuous.
    """
    topo, coords = ubiquitin
    flagged: list = []
    live: list = []
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=True,
        extra=TRUNCATED, bearings=flagged,
    )
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=False,
        extra=TRUNCATED, bearings=live,
    )
    assert len(flagged) == len(live) == 1, "one crop each, or this compares runs"
    a, b = flagged[0], live[0]

    assert torch.equal(a["damaged"], b["damaged"]), (
        "the two runs damaged different residues, so they are not two readings "
        "of one start state"
    )
    assert torch.equal(a["start"], b["start"]), (
        "the two runs started from different coordinates, so any difference in "
        "align is confounded with a difference in the task"
    )
    assert a["need"] == b["need"], (
        f"need moved between the two perceptions: {b['need']} -> {a['need']}. "
        "It is a property of the start state and cannot depend on the edge set"
    )
    assert a["need"] == a["need"], "need must be a number on this crop"
    assert a["align"] != b["align"], (
        "align came out identical under the two edge sets, so either the flag "
        "reached nothing or the rule is not moving -- and the pairing above is "
        "then vacuous rather than earned"
    )


def test_j_undoing_the_pairing_makes_test_i_fail(
    monkeypatch, tmp_path, ubiquitin
):
    """Proof by crippling: restore the defect, and ``need`` moves again.

    ``cripple_pairing`` computes ``need`` the way the readout did before --
    off ``result.moves[0]``, the start of the live truncation segment -- and
    changes nothing else. If ``need`` still came out equal under that, test i
    would be passing because the rollout is too static to tell the difference,
    not because the start state is shared, and it would keep passing if the fix
    were reverted.

    The assertion is deliberately the *negation* of test i's: these two runs
    still share a start state bit for bit, and the ``need`` computed from it is
    still equal. It is only the ``need`` read off the rollout that moves. So
    this localises the defect as well as proving the test: not the crop, not
    the damage, not the seed -- the state ``need`` was read from.
    """
    topo, coords = ubiquitin
    flagged: list = []
    live: list = []
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=True,
        extra=TRUNCATED, bearings=flagged, cripple_pairing=True,
    )
    _readout(
        monkeypatch, tmp_path, topo, coords, native_edges=False,
        extra=TRUNCATED, bearings=live, cripple_pairing=True,
    )
    a, b = flagged[0], live[0]
    # The paired half is untouched, which is what makes this a localisation.
    assert torch.equal(a["start"], b["start"])
    assert torch.equal(a["damaged"], b["damaged"])
    assert a["need"] != b["need"], (
        "with `need` read back off the rollout's own moves it came out equal "
        "anyway, so test i's assertion would pass on the unfixed readout and "
        "proves nothing. Either this crop's rollout moves too little to show "
        "the defect, or `moves` is no longer being cleared at the truncation "
        "boundary"
    )


def test_k_the_reconstructed_start_is_the_one_the_rollout_begins_at(
    monkeypatch, tmp_path, ubiquitin
):
    """``damage_start`` reproduces ``growth_objective``'s draws, not a lookalike.

    The fix trades one liability for another: ``need`` no longer depends on the
    perception, but it is now computed from a state this script builds itself,
    and a drift in ``growth_objective``'s draw ORDER would make it a plausible
    number from a pose no rollout ever started at. That is the 6 Nov failure in
    a third costume, so it is pinned rather than trusted.

    At ``steps = 4`` and ``truncate = 12`` no boundary is crossed, nothing
    clears ``moves``, and ``moves[0]`` therefore IS the rollout's start state --
    which makes it the witness. The readout checks the same thing on every such
    crop at runtime and prints if it fails; this asserts it.
    """
    topo, coords = ubiquitin
    seen_moves: list = []

    def capture(result, native, topo_, damaged, start_coords, config, local=False):
        from aanca.growth import backbone_atoms

        seen_moves.append((
            len(result.moves),
            backbone_atoms(result.moves[0][2], topo_)[:, 1].clone(),
            backbone_atoms(start_coords, topo_)[:, 1].clone(),
        ))
        return REAL_BEARING(
            result, native, topo_, damaged, start_coords, config, local=local
        )

    # Installed without ``bearings``, which would patch ``_bearing`` a second
    # time and leave this one unreached.
    monkeypatch.setattr(ceiling_readout, "_bearing", capture)
    _readout(monkeypatch, tmp_path, topo, coords, native_edges=False)
    assert seen_moves, "nothing was captured"
    n_moves, first, reconstructed = seen_moves[0]
    assert n_moves == STORED["steps"], (
        "this crop crossed a truncation boundary, so `moves[0]` is not the "
        "start state and cannot witness the reconstruction"
    )
    assert torch.equal(first, reconstructed), (
        "the start state `damage_start` reconstructs is not the one the rollout "
        "began at, so `need` is computed from a pose no run ever held"
    )
