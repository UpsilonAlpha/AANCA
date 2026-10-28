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


def test_sharing_a_cluster_with_an_excluded_entry_is_not_itself_leakage():
    """The distinction that produced a false leak report, encoded.

    An excluded entry can touch two clusters: one banned, because it holds a
    chain homologous to a test target, and one innocent. An entry sharing only
    the innocent cluster is 40%-similar to *a chain of the excluded entry*,
    which says nothing about its similarity to any test target -- so it must
    survive.

    Checking "shares any cluster with something excluded" instead of "touches a
    banned cluster" reported 60 false positives on the real manifest.
    """
    index = ClusterIndex(
        [
            # TGT and BRIDGE share cluster C1 -- BRIDGE is a genuine homolog.
            "C1 TGT_1 BRIDGE_1\n",
            # BRIDGE's second entity sits in C2 with INNOCENT. C2 is not banned.
            "C2 BRIDGE_2 INNOCENT_1\n",
        ]
    )
    specs = [
        StructureSpec(pdb_id=i, path=DATA / f"{i}.pdb", group=index.group_of(i))
        for i in ("BRIDGE", "INNOCENT")
    ]
    splits = split_by_cluster(specs, {"TGT"}, index, validation_fraction=0.0)

    assert {s.pdb_id for s in splits["excluded"]} == {"BRIDGE"}
    assert {s.pdb_id for s in splits["train"]} == {"INNOCENT"}
    # INNOCENT shares C2 with the excluded BRIDGE, but touches no banned cluster.
    assert index.clusters_of("INNOCENT") & index.clusters_of("BRIDGE")
    assert not index.clusters_of("INNOCENT") & index.clusters_of("TGT")


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


def test_split_survives_a_change_to_the_pool(clusters):
    """Adding or removing entries must not reassign unrelated groups.

    This is a regression test for a measured failure, not a hypothetical. The
    split used to be a ``randperm`` over the sorted group list, so every
    group's side depended on how many groups existed. Correcting the leakage
    filter moved **one** group into ``excluded``, the permutation shifted, and
    945 of 988 validation entries changed -- meaning the whole validation set
    had to be re-downloaded to evaluate the same model against the same split.
    """
    many = ClusterIndex([f"Z{i:03d}A_1\n" for i in range(200)])
    ids = [f"Z{i:03d}A" for i in range(200)]
    full = split_by_cluster(
        _specs(many, ids), {"Z000A"}, many, validation_fraction=0.2, seed=3
    )
    # Drop one entry from the middle of the pool, as a filter correction would.
    fewer = split_by_cluster(
        _specs(many, [i for i in ids if i != "Z100A"]),
        {"Z000A"},
        many,
        validation_fraction=0.2,
        seed=3,
    )
    before = {s.pdb_id for s in full["validation"]}
    after = {s.pdb_id for s in fewer["validation"]}
    assert before - after <= {"Z100A"}
    assert not after - before
    assert len(after) > 10, "too few validation entries for this to mean anything"


def test_the_split_fraction_is_roughly_the_one_asked_for(clusters):
    """Hashing makes the realised fraction binomial rather than exact. At this
    pool size the tolerance below is about four standard deviations."""
    many = ClusterIndex([f"W{i:04d}A_1\n" for i in range(4000)])
    ids = [f"W{i:04d}A" for i in range(4000)]
    splits = split_by_cluster(
        _specs(many, ids), {"W0000A"}, many, validation_fraction=0.1, seed=0
    )
    share = len(splits["validation"]) / len(ids)
    assert 0.08 < share < 0.12, share


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


def test_oversized_structures_fail_loudly(dataset):
    """A size guard behind the manifest filter.

    The smoke test on real data found entries from 12 to 3,570 residues. At the
    measured ~124 MB of retained activations per 300 residues per step, a 3,570
    residue assembly needs ~13 GB for one 30-step rollout -- which would surface
    as an opaque OOM mid-training rather than as a data problem.
    """
    dataset.max_residues = 10
    dataset._cache.clear()
    with pytest.raises(ValueError, match="over the max_residues limit"):
        dataset.load(0)


def test_tiny_structures_fail_loudly(dataset):
    """The manifest filter cannot catch these on its own.

    RCSB's deposited_polymer_monomer_count includes nucleotides, so a
    DNA-protein complex passes a 40-residue filter and then yields almost no
    amino acids once the loader keeps only the standard twenty. The smoke test
    on real data found entries down to 2 residues that way.
    """
    dataset.min_residues = 10_000
    dataset._cache.clear()
    with pytest.raises(ValueError, match="under the min_residues limit"):
        dataset.load(0)


def test_size_guards_can_be_disabled(dataset):
    dataset.max_residues = None
    dataset.min_residues = None
    dataset._cache.clear()
    topo, _coords = dataset.load(0)
    assert topo.n_residues > 10


def test_rejects_empty_inputs():
    with pytest.raises(ValueError):
        RepackingDataset([])
    with pytest.raises(ValueError):
        RepackingDataset(
            [StructureSpec("X", DATA / "x.pdb", "X")], concentrations=()
        )


def test_the_parse_cache_is_bounded(dataset):
    """An unbounded cache over 23,761 entries, some of them 4,000-residue
    assemblies, would grow until training died of it."""
    dataset.cache_limit = 2
    for index in range(min(len(dataset), 4)):
        dataset.protein(index)
    assert len(dataset._proteins) <= 2


# -- cropping, at the dataset level ----------------------------------------


@pytest.fixture
def big_dataset():
    """4HHB: 574 residues across four chains, so a 512 crop really crops."""
    path = DATA / "4HHB.pdb"
    if not path.exists():
        pytest.skip("run scripts/fetch_test_structures.py")
    return RepackingDataset(
        [StructureSpec("4HHB", path, "4HHB")],
        concentrations=(50.0,),
        crop_size=512,
    )


def test_a_large_entry_is_cropped_and_partly_supervised(big_dataset):
    topo, _coords, indices = big_dataset.load_crop(0)
    assert topo.n_residues == 512
    assert indices is not None and len(indices) == 512
    assert 0.0 < float(topo.supervised_mask.double().mean()) < 1.0


def test_a_crop_is_not_cached_so_the_region_can_vary(big_dataset):
    """Caching the topology would freeze one region per entry and turn the
    crop back into truncation."""
    import random as _random

    first = big_dataset.load_crop(0, rng=_random.Random(1))[2]
    second = big_dataset.load_crop(0, rng=_random.Random(2))[2]
    assert first != second
    assert not big_dataset._cache


def test_a_small_entry_is_taken_whole_and_fully_supervised(dataset):
    topo, _coords, indices = dataset.load_crop(0)
    assert indices is None
    assert topo.supervised is None
    assert topo.supervised_mask.all()


def test_cropping_can_be_disabled(big_dataset):
    big_dataset.crop_size = None
    topo, _coords, indices = big_dataset.load_crop(0)
    assert indices is None
    assert topo.n_residues > 512
