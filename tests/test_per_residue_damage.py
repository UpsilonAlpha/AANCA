"""Per-residue damage: the fix for what ``scripts/clamp_test.py`` measured.

Until 2026-11-02 every training sample was damaged *globally* -- one
concentration per structure, applied to every chi -- so the distribution held
only two corners, all-displaced and all-native, and never a side chain that is
wrong inside a neighbourhood that is right. The clamp test showed the rule is
*worse* in that state (57.1 % well-right with three quarters of the protein
pinned at crystal conformations) than in the globally damaged one (77.3 %).

The tests that matter here are the two that are easy to get wrong rather than
the one that is obvious:

* that ``(1.0,)`` consumes **exactly** the random stream the old code did, so
  v5's curve is still reproducible and a warm start lands on a distribution the
  weights recognise; and
* that a partial draw leaves the undamaged residues at *exactly* native, not
  nearly, since "nearly" is a different experiment.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.data import (
    RepackingDataset,
    StructureSpec,
    damaged_residues,
    uniform_deltas,
    von_mises_deltas,
)
from aanca.torch_geometry import apply_chi_deltas, measure_chis
from aanca.rotamers import nearest_well

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture
def dataset():
    paths = [DATA / f"{i}.pdb" for i in ("1UBQ", "193L")]
    if not all(p.exists() for p in paths):
        pytest.skip("run scripts/fetch_test_structures.py")
    specs = [StructureSpec(pdb_id=p.stem, path=p, group=p.stem) for p in paths]
    return RepackingDataset(specs, concentrations=(50.0, 0.0))


# --- the subset draw itself -------------------------------------------------


def test_the_subset_is_per_residue_not_per_chi():
    """A residue is damaged on all its chis or none. The unit is a side chain
    in the wrong place, not a chi2 disagreeing with its own chi1."""
    mask = torch.ones(50, 4, dtype=torch.bool)
    keep = damaged_residues(mask, 0.5, torch.Generator().manual_seed(0))
    assert keep.shape == (50,)
    assert keep.dtype == torch.bool


def test_the_subset_scales_with_the_fraction():
    g = torch.Generator().manual_seed(7)
    mask = torch.ones(4000, 4, dtype=torch.bool)
    shares = [
        float(damaged_residues(mask, f, g).to(torch.float64).mean())
        for f in (0.0, 0.1, 0.5, 1.0)
    ]
    assert shares[0] == 0.0
    assert shares[3] == 1.0
    assert 0.07 < shares[1] < 0.13
    assert 0.47 < shares[2] < 0.53
    assert shares == sorted(shares)


def test_the_subset_is_reproducible_from_the_seed():
    mask = torch.ones(200, 4, dtype=torch.bool)
    a = damaged_residues(mask, 0.3, torch.Generator().manual_seed(11))
    b = damaged_residues(mask, 0.3, torch.Generator().manual_seed(11))
    c = damaged_residues(mask, 0.3, torch.Generator().manual_seed(12))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_an_out_of_range_fraction_is_refused():
    mask = torch.ones(8, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        damaged_residues(mask, 1.5)
    with pytest.raises(ValueError):
        damaged_residues(mask, -0.1)


# --- the stream, which is the part a mistake would hide ----------------------


def test_fraction_one_consumes_the_old_random_stream_exactly(dataset):
    """The whole reason ``(1.0,)`` skips the subset draw. If this fails, v5's
    curve is no longer reproducible and ``--init-from`` is warm-starting onto a
    distribution that is not the one it reports."""
    recipe = torch.Generator().manual_seed(3)
    topo, native, _indices, _seed = dataset.load_crop(0, seed_residue=0)
    pick = torch.randint(len(dataset.concentrations), (1,), generator=recipe).item()
    level = dataset.concentrations[pick]
    mask = topo.torsion.chi_mask
    deltas = (
        uniform_deltas(mask, recipe)
        if level <= 0
        else von_mises_deltas(mask, level, recipe)
    )
    expected = apply_chi_deltas(native, deltas, topo.torsion)

    sample = dataset.sample(0, generator=torch.Generator().manual_seed(3), crop_seed=0)
    assert sample.damage_fraction == 1.0
    assert torch.equal(sample.damaged, expected)


def test_a_single_fraction_below_one_also_draws_nothing_to_choose(dataset):
    """One option is not a choice, so no ``randint`` is spent on it. Compared
    against passing the same fraction explicitly: a spurious draw would advance
    the generator before the subset is chosen and the two would diverge."""
    dataset.damage_fractions = (0.25,)
    drawn = dataset.sample(0, generator=torch.Generator().manual_seed(5), crop_seed=0)
    given = dataset.sample(
        0, fraction=0.25, generator=torch.Generator().manual_seed(5), crop_seed=0
    )
    assert drawn.damage_fraction == 0.25
    assert torch.equal(drawn.damaged, given.damaged)


# --- what the damaged structure actually looks like --------------------------


def _chi1_moved(sample, topo):
    native = measure_chis(sample.native, topo.torsion)[:, 0]
    damaged = measure_chis(sample.damaged, topo.torsion)[:, 0]
    return (damaged - native).abs()


def test_the_undamaged_residues_sit_at_exactly_native(dataset):
    """Not nearly native. The clamp test's native arm holds residues at the
    crystal coordinates, and this is the training case for that arm."""
    sample = dataset.sample(
        0, fraction=0.3, generator=torch.Generator().manual_seed(2), crop_seed=0
    )
    untouched = ~(sample.deltas != 0).any(dim=-1)
    assert bool(untouched.any()), "nothing was left alone"
    assert bool((~untouched).any()), "nothing was damaged"
    assert torch.equal(sample.damaged[untouched], sample.native[untouched])


def test_a_partial_draw_damages_some_residues_and_not_others(dataset):
    topo, _native, _indices, _seed = dataset.load_crop(0, seed_residue=0)
    sample = dataset.sample(
        0, concentration=0.0, fraction=0.5,
        generator=torch.Generator().manual_seed(4), crop_seed=0,
    )
    eligible = topo.torsion.chi_mask[:, 0]
    moved = _chi1_moved(sample, topo) > 1e-3
    assert bool((moved & eligible).any()), "nothing was damaged"
    assert bool((~moved & eligible).any()), "everything was damaged"


def test_fraction_zero_is_a_native_start(dataset):
    """It subsumes ``--native-probability``, which is worth knowing because the
    two knobs now overlap and only one of them is pinned in old checkpoints."""
    sample = dataset.sample(
        0, fraction=0.0, generator=torch.Generator().manual_seed(1), crop_seed=0
    )
    assert torch.equal(sample.damaged, sample.native)
    assert float(sample.deltas.abs().max()) == 0.0


def test_the_damaged_residues_really_leave_their_wells(dataset):
    """A partial draw that moved chi1 by a degree would train nothing. At
    concentration 0 the draw is uniform, so most damaged residues should land
    in a different rotamer well than they started in."""
    topo, _native, _indices, _seed = dataset.load_crop(0, seed_residue=0)
    sample = dataset.sample(
        0, concentration=0.0, fraction=1.0,
        generator=torch.Generator().manual_seed(9), crop_seed=0,
    )
    eligible = topo.torsion.chi_mask[:, 0]
    native_well, _ = nearest_well(
        measure_chis(sample.native, topo.torsion)[:, 0].rad2deg()
    )
    damaged_well, _ = nearest_well(
        measure_chis(sample.damaged, topo.torsion)[:, 0].rad2deg()
    )
    changed = (native_well != damaged_well)[eligible]
    assert float(changed.to(torch.float64).mean()) > 0.5


# --- the mixture, and its guard rails ----------------------------------------


def test_the_fractions_are_a_mixture_drawn_per_sample(dataset):
    dataset.damage_fractions = (0.1, 1.0)
    g = torch.Generator().manual_seed(21)
    seen = {
        dataset.sample(0, generator=g, crop_seed=0).damage_fraction
        for _ in range(30)
    }
    assert seen == {0.1, 1.0}


def test_an_empty_or_impossible_fraction_list_is_refused():
    spec = [StructureSpec("X", DATA / "x.pdb", "X")]
    with pytest.raises(ValueError):
        RepackingDataset(spec, damage_fractions=())
    with pytest.raises(ValueError):
        RepackingDataset(spec, damage_fractions=(0.5, 1.5))


def test_damage_fractions_is_pinned_against_a_resume():
    """A resume that changed the damage distribution would continue one curve
    under a different problem. A *warm start* is expected to change it, which
    is why it is absent from train.check_warm_start's list."""
    from aanca import checkpoint as ckpt

    assert "damage_fractions" in ckpt.PINNED
    stored = {"damage_fractions": [1.0]}
    with pytest.raises(ValueError, match="damage_fractions"):
        ckpt.check_fingerprint(
            stored, {"damage_fractions": [0.1, 1.0]}, {"damage_fractions": [1.0]}
        )


def test_an_old_checkpoint_without_the_key_still_resumes_at_the_default():
    """The ``defaults`` escape hatch: pinning a new setting must not strand a
    run already chaining through its segments."""
    from aanca import checkpoint as ckpt

    ckpt.check_fingerprint({}, {"damage_fractions": [1.0]}, {"damage_fractions": [1.0]})
    with pytest.raises(ValueError):
        ckpt.check_fingerprint(
            {}, {"damage_fractions": [0.25]}, {"damage_fractions": [1.0]}
        )
