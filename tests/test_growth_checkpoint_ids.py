"""A growth checkpoint has to name the proteins it held out, not just index them.

``eval_indices`` are offsets into ``available[:structures + eval_structures]``,
so the same integer names a different protein on a differently sized pool --
954 structures on the laptop against 23,760 on Bunya. ``eval_pdb_ids`` is what
lets a probe refuse that mismatch instead of silently measuring the wrong
proteins, and for two days it was not written at all: ``scripts/train_growth.py``
had two variables called ``pool`` in one function scope -- the structure list
and, assigned ~230 lines later, the Growing-NCA sample pool, a dict keyed by
``(structure index, crop seed)``. The second shadowed the first, so the
checkpoint writer's ``pool[i].pdb_id`` indexed the dict with an integer and
every run died with ``KeyError`` at its FIRST best checkpoint, after the
evaluation it was checkpointing (job 33304426: nine minutes and a full held-out
evaluation lost).

What the two tests here protect, and what they cannot:

``test_a_growth_run_writes_the_held_out_ids_it_actually_evaluated`` runs the
real driver to its first best checkpoint and resolves the ids independently.
It is the only version of the claim that covers the writer, the split and the
save together. It cannot run without ``data/manifest.json`` and coordinates on
disk, it says nothing about a GPU run or about any arm's settings but the tiny
ones below, and it would not notice ids that are correct here and wrong at
another pool size -- that is what the ids in the checkpoint exist to catch
downstream.

``test_the_structure_list_and_the_sample_pool_do_not_share_a_name`` reads the
source instead, so it runs with no data and in milliseconds, and it fails for
the specific shape of the bug -- one name for both containers -- rather than
for its symptom. It cannot catch a wrong-container error written with two
distinct names, nor any other reason the ids could come out wrong.
"""

from __future__ import annotations

import ast
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aanca import checkpoint as ckpt
from aanca.data import read_manifest

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts" / "train_growth.py"
CONFIG = ROOT / "configs" / "regeneration.toml"
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"

#: The section name is also the run tag, so it carries the pid: two suites
#: running at once must not write each other's ``runs/<tag>/best.pt``. See the
#: same note in ``tests/test_train_resume.py``, which cost half an hour of
#: chasing a regression that was not one.
SECTION = f"pytest_eval_ids_{os.getpid()}"

#: Deliberately 6 + 3 and not 3 + 1. ``indices`` is shuffled before the split,
#: and with nine structures seed 0 holds out offsets 7, 5 and 1 -- which a
#: dict keyed by anything else cannot serve and a list can. A test on 0..n-1
#: would pass against several wrong containers.
TINY = """
[{section}.inputs]
structures = 6
eval_structures = 3
updates = 1
eval_every = 1
steps = 2
truncate = 1
crop_size = 48
damage = 4
seeds = 1
pool_probability = 0.0
"""


@pytest.fixture
def tiny_config(tmp_path):
    """The real config plus one tiny arm, written outside the repo.

    ``[base.inputs]`` must state every setting (``check_base_is_complete``), so
    the section is appended to a copy of the committed config rather than
    written from scratch -- and to a copy, because a scratch arm that exists
    only to be fast is not an arm and does not belong in ``configs/``.
    """
    path = tmp_path / "tiny.toml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8") + TINY.format(section=SECTION),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def clean_run_dir():
    shutil.rmtree(ROOT / "runs" / SECTION, ignore_errors=True)
    yield ROOT / "runs" / SECTION
    shutil.rmtree(ROOT / "runs" / SECTION, ignore_errors=True)


@pytest.mark.needs_data
@pytest.mark.slow
def test_a_growth_run_writes_the_held_out_ids_it_actually_evaluated(
    tiny_config, clean_run_dir
):
    if not MANIFEST.exists() or not any(PDB.glob("*.bcif")):
        pytest.skip("needs data/manifest.json and downloaded coordinates")

    done = subprocess.run(
        [sys.executable, str(TRAINER), str(tiny_config),
         "--section", SECTION, "--device", "cpu"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "new best, checkpointed" in done.stdout, (
        "the run never reached a best checkpoint, so it did not exercise the "
        "writer this test is about: " + done.stdout[-3000:]
    )

    # The ids the run should have written, resolved here from the manifest and
    # the trainer's own split rule -- not read back out of the file it wrote.
    catalogue = _catalogue(6 + 3)
    order = list(range(len(catalogue)))
    random.Random(0).shuffle(order)
    expected_indices = order[:3]
    expected_ids = [catalogue[i].pdb_id for i in expected_indices]
    assert expected_indices != sorted(expected_indices), (
        "the split came out in order, so this run would not have caught the bug"
    )

    for name in ("best.pt", "last.pt"):
        state = ckpt.load(clean_run_dir / name)
        assert "eval_pdb_ids" in state, f"{name} does not name its held-out set"
        ids = state["eval_pdb_ids"]
        assert isinstance(ids, list), f"{name}: {type(ids)}"
        assert all(isinstance(one, str) for one in ids), ids
        assert len(ids) == len(state["eval_indices"]), (ids, state["eval_indices"])
        assert state["eval_indices"] == expected_indices
        assert ids == expected_ids, f"{name} names the wrong proteins"


def test_the_structure_list_and_the_sample_pool_do_not_share_a_name():
    """The bug itself, read off the source, with no data and no training run.

    Two containers, one name, in one function scope: the list of structures and
    the Growing-NCA sample pool. Nothing about the shadowing is visible at the
    use site -- ``pool[i].pdb_id`` reads correctly either way -- so the only
    cheap guard is that the names differ, and that the sample pool is never the
    thing a ``.pdb_id`` is read from.
    """
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    main = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    pool_names = {
        node.target.id
        for node in ast.walk(main)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and "GrowthPoolEntry" in ast.unparse(node.annotation)
    }
    catalogue_names = {
        target.id
        for node in ast.walk(main) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        if "available[" in ast.unparse(node.value)
    }
    assert pool_names, "no sample pool found in main; this test has gone stale"
    assert catalogue_names, "no structure list found in main; this test has gone stale"
    assert not pool_names & catalogue_names, (
        f"{pool_names & catalogue_names} names both the structure list and the "
        "sample pool in main's scope; the later assignment shadows the earlier "
        "one and the checkpoint writer indexes whichever it got"
    )

    for node in ast.walk(main):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "pdb_id"
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in pool_names
        ):
            raise AssertionError(
                f"{ast.unparse(node)} reads a pdb_id out of the sample pool, "
                "which is keyed by (structure index, crop seed) tuples"
            )


def _catalogue(wanted: int) -> list:
    """``available[:wanted]``, by the trainer's own helpers."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_growth", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    available = module.on_disk(read_manifest(module.MANIFEST)["train"])
    return available[:wanted] if len(available) >= wanted else available
