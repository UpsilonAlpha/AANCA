"""Dataset assembly: cluster grouping, leak-free splitting, and damage sampling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.data import (
    ClusterIndex,
    RepackingDataset,
    StructureSpec,
    read_manifest,
    split_by_cluster,
    uniform_deltas,
    von_mises_deltas,
    write_manifest,
)
from aanca.torch_geometry import measure_chis

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture
def clusters():
    # Two clusters: {1ABC, 2DEF} and {3GHI, 4JKL, 5MNO}; 9XYZ unlisted.
    return ClusterIndex(
        [
            "1ABC_1 2DEF_1 2DEF_2\n",
            "3GHI_1 4JKL_1\n",
            "4JKL_2 5MNO_1\n",
        ]
    )


def _specs(clusters, ids):
    return [
        StructureSpec(pdb_id=i, path=DATA / f"{i}.pdb", group=clusters.group_of(i))
        for i in ids
    ]


# -- grouping --------------------------------------------------------------


def test_entries_sharing_a_cluster_are_grouped(clusters):
    assert clusters.group_of("1ABC") == clusters.group_of("2DEF")


def test_grouping_is_deliberately_not_transitive(clusters):
    """4JKL shares one cluster with 3GHI and another with 5MNO, but 3GHI and
    5MNO are *not* grouped together.

    Transitive grouping was tried and measured: on 7,601 X-ray entries at 1.2 A
    it produced a single group of 6,682 entries -- 88% of the pool -- because a
    promiscuous entity chains unrelated entries and the merging cascades
    globally. Grouping by the primary entity cut that to a largest group of 751
    and tripled the usable pool. The cost is borne by the test-leakage filter,
    which is sequence-based and does not rely on this at all.
    """
    assert clusters.group_of("3GHI") != clusters.group_of("5MNO")
    # 4JKL's primary entity is the one shared with 3GHI, so it groups there.
    assert clusters.group_of("4JKL") == clusters.group_of("3GHI")


def test_clusters_of_is_the_conservative_view(clusters):
    """Exclusion uses every cluster an entry touches, not just the primary one:
    a test target's homolog may sit at any entity number."""
    assert len(clusters.clusters_of("4JKL")) == 2
    assert clusters.clusters_of("9XYZ") == set()


def test_holdout_exclusion_follows_every_cluster(clusters):
    """Even a non-primary shared cluster must remove the entry."""
    specs = _specs(clusters, ["3GHI", "4JKL", "5MNO", "9XYZ"])
    splits = split_by_cluster(specs, {"5MNO"}, clusters, validation_fraction=0.0)
    excluded = {s.pdb_id for s in splits["excluded"]}
    # 4JKL shares 5MNO's cluster through its second entity, so it goes too.
    assert excluded == {"5MNO", "4JKL"}
    assert {s.pdb_id for s in splits["train"]} == {"3GHI", "9XYZ"}


def test_unlisted_entries_are_their_own_group(clusters):
    """The conservative reading: an unclustered entry pulls no homolog across."""
    assert clusters.group_of("9XYZ") == "9XYZ"
    assert clusters.group_of("9XYZ") != clusters.group_of("1ABC")


def test_grouping_is_case_insensitive(clusters):
    assert clusters.group_of("1abc") == clusters.group_of("1ABC")


def test_group_ids_are_stable_not_insertion_ordered():
    """Group ids must not depend on file order, or a split is not reproducible."""
    forward = ClusterIndex(["AAAA_1 BBBB_1\n", "BBBB_2 CCCC_1\n"])
    backward = ClusterIndex(["BBBB_2 CCCC_1\n", "AAAA_1 BBBB_1\n"])
    assert forward.group_of("CCCC") == backward.group_of("CCCC")


def test_group_size_histogram_exposes_over_merging(clusters):
    """Counts members *present in the given list*, so the cost of grouping is
    visible for the pool actually being split rather than for all of RCSB.

    This is the diagnostic that caught the union-find collapse, so it earns its
    place: a largest-group number is the cheapest signal that grouping has gone
    wrong.
    """
    histogram = clusters.group_size_histogram(
        ["1ABC", "2DEF", "3GHI", "4JKL", "5MNO", "9XYZ"]
    )
    # {1ABC,2DEF} share a primary cluster = 2; {3GHI,4JKL} = 2; 5MNO and 9XYZ
    # are each alone, 5MNO because its only cluster is not 4JKL's primary.
    assert histogram == {1: 2, 2: 2}
    assert max(histogram) == 2, "no group should swallow the pool"


# -- splitting -------------------------------------------------------------


def test_holdout_removes_the_whole_group(clusters):
    """Dropping only the test target leaves its homologs in training."""
    specs = _specs(clusters, ["1ABC", "2DEF", "3GHI", "9XYZ"])
    splits = split_by_cluster(specs, {"1ABC"}, clusters, validation_fraction=0.0)

    excluded = {s.pdb_id for s in splits["excluded"]}
    assert excluded == {"1ABC", "2DEF"}, "2DEF is a homolog and must go too"
    assert {s.pdb_id for s in splits["train"]} == {"3GHI", "9XYZ"}


def test_an_empty_holdout_is_refused(clusters):
    """Silently training on the test set is how an indefensible number happens."""
    specs = _specs(clusters, ["1ABC", "3GHI"])
    with pytest.raises(ValueError, match="holdout_ids is empty"):
        split_by_cluster(specs, set(), clusters)


def test_empty_holdout_can_be_opted_into(clusters):
    specs = _specs(clusters, ["1ABC", "3GHI"])
    splits = split_by_cluster(
        specs, set(), clusters, validation_fraction=0.0, require_holdout=False
    )
    assert len(splits["train"]) == 2


def test_no_group_straddles_train_and_validation(clusters):
    many = ClusterIndex([f"X{i:03d}A_1 X{i:03d}B_1\n" for i in range(50)])
    ids = [f"X{i:03d}{s}" for i in range(50) for s in "AB"]
    specs = _specs(many, ids)

    splits = split_by_cluster(
        specs, {"X000A"}, many, validation_fraction=0.2, seed=0
    )
    train_groups = {s.group for s in splits["train"]}
    validation_groups = {s.group for s in splits["validation"]}
    assert not (train_groups & validation_groups)
    assert len(splits["validation"]) > 0


def test_split_is_reproducible(clusters):
    many = ClusterIndex([f"Y{i:03d}A_1\n" for i in range(40)])
    specs = _specs(many, [f"Y{i:03d}A" for i in range(40)])
    a = split_by_cluster(specs, {"Y000A"}, many, validation_fraction=0.25, seed=7)
    b = split_by_cluster(specs, {"Y000A"}, many, validation_fraction=0.25, seed=7)
    assert [s.pdb_id for s in a["validation"]] == [s.pdb_id for s in b["validation"]]


def test_rejects_a_nonsense_fraction(clusters):
    specs = _specs(clusters, ["1ABC"])
    with pytest.raises(ValueError):
        split_by_cluster(specs, {"9XYZ"}, clusters, validation_fraction=1.0)


def test_manifest_round_trips(tmp_path, clusters):
    specs = _specs(clusters, ["1ABC", "3GHI"])
    splits = split_by_cluster(specs, {"9XYZ"}, clusters, validation_fraction=0.0)
    path = tmp_path / "manifest.json"
    write_manifest(path, splits)
    restored = read_manifest(path)
    assert {s.pdb_id for s in restored["train"]} == {s.pdb_id for s in splits["train"]}
    assert restored["train"][0].group == splits["train"][0].group


# -- damage sampling -------------------------------------------------------


def test_damage_only_touches_real_chis():
    mask = torch.tensor([[True, False, True, False]])
    for deltas in (
        von_mises_deltas(mask, 20.0),
        uniform_deltas(mask),
    ):
        assert torch.all(deltas[~mask] == 0.0)
        assert torch.any(deltas[mask] != 0.0)


def test_concentration_grades_the_damage():
    mask = torch.ones(4000, 4, dtype=torch.bool)
    spreads = [
        float(von_mises_deltas(mask, kappa).abs().mean())
        for kappa in (200.0, 50.0, 10.0, 2.0)
    ]
    assert spreads == sorted(spreads), spreads


def test_zero_concentration_is_uniform():
    mask = torch.ones(4000, 4, dtype=torch.bool)
    spread = float(von_mises_deltas(mask, 0.0).abs().mean())
    # A uniform angle on (-pi, pi] has mean |x| = pi/2.
    assert spread == pytest.approx(torch.pi / 2, rel=0.05)


# -- the dataset -----------------------------------------------------------


@pytest.fixture
def dataset():
    paths = [DATA / f"{i}.pdb" for i in ("1UBQ", "193L")]
    if not all(p.exists() for p in paths):
        pytest.skip("run scripts/fetch_test_structures.py")
    specs = [
        StructureSpec(pdb_id=p.stem, path=p, group=p.stem) for p in paths
    ]
    return RepackingDataset(specs, concentrations=(50.0, 0.0))


def test_dataset_yields_a_damaged_native_pair(dataset):
    sample = dataset.sample(0, concentration=20.0)
    assert sample.pdb_id == "1UBQ"
    assert sample.native.shape == sample.damaged.shape
    assert float((sample.damaged - sample.native).abs().max()) > 0.1


def test_damage_is_exactly_reversible(dataset):
    """The recorded deltas make the damage auditable, not merely applied."""
    from aanca.torch_geometry import apply_chi_deltas

    sample = dataset.sample(0, concentration=10.0)
    restored = apply_chi_deltas(sample.damaged, -sample.deltas, sample.topology.torsion)
    assert float((restored - sample.native).abs().max()) < 1e-9


def test_native_chis_are_recoverable(dataset):
    sample = dataset.sample(1, concentration=5.0)
    native = measure_chis(sample.native, sample.topology.torsion)
    damaged = measure_chis(sample.damaged, sample.topology.torsion)
    mask = sample.topology.torsion.chi_mask
    assert not torch.allclose(native[mask], damaged[mask])


def test_topologies_are_cached(dataset):
    first = dataset.load(0)
    second = dataset.load(0)
    assert first[0] is second[0], "building a topology parses the whole structure"


def test_random_concentration_comes_from_the_mixture(dataset):
    """The curriculum is a mixture, not a schedule -- the rule is never told how
    badly it was damaged, which is the noise-level conditioning DiffPack has and
    this deliberately does not."""
    g = torch.Generator().manual_seed(0)
    seen = {dataset.sample(0, generator=g).concentration for _ in range(30)}
    assert seen <= set(dataset.concentrations)
    assert len(seen) > 1


def test_epoch_visits_every_structure(dataset):
    g = torch.Generator().manual_seed(0)
    seen = [s.pdb_id for s in dataset.iter_epoch(generator=g)]
    assert sorted(seen) == ["193L", "1UBQ"]


def test_rejects_empty_inputs():
    with pytest.raises(ValueError):
        RepackingDataset([])
    with pytest.raises(ValueError):
        RepackingDataset(
            [StructureSpec("X", DATA / "x.pdb", "X")], concentrations=()
        )
