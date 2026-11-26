"""The CA-only metric suite behind ``scripts/generation_metrics.py``.

Every assertion here is an anchor against something external to the code: ideal
helix and strand geometry built from textbook parameters, the algebraic
identities TM-score has to satisfy, biotite's independent implementation of the
same P-SEA paper, and a crystal structure. A metric suite that is only checked
against itself reports its own bugs as results, and this one exists precisely
so that a generated number has something to be compared against.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "generation_metrics", ROOT / "scripts" / "generation_metrics.py"
)
gm = importlib.util.module_from_spec(SPEC)
sys.modules["generation_metrics"] = gm
SPEC.loader.exec_module(gm)

RAW = ROOT / "data" / "raw"
UBIQUITIN = RAW / "1UBQ.pdb"


# ---------------------------------------------------------------------------
# ideal geometry, built from standard parameters rather than asserted
# ---------------------------------------------------------------------------


def ideal_helix(n: int = 40, radius: float = 2.28, rise: float = 1.50) -> np.ndarray:
    """CA trace of an ideal right-handed alpha helix.

    Textbook parameters: 3.6 residues per turn, so 100 degrees of twist per
    residue; 1.50 A of rise per residue; a CA radius of 2.28 A about the helix
    axis. Those three numbers fix the virtual CA-CA distance at 3.80 A, which
    is what :data:`generation_metrics.CA_CA_IDEAL` claims -- so the helix is an
    independent construction, not a fit to the metric.
    """
    turn = np.deg2rad(100.0)
    i = np.arange(n)
    return np.stack(
        [radius * np.cos(turn * i), radius * np.sin(turn * i), rise * i], axis=-1
    )


def ideal_strand(n: int = 40, bond: float = 3.80, angle: float = 124.0) -> np.ndarray:
    """CA trace of an ideal extended beta strand: a planar pleated zigzag.

    The pleat angle is P-SEA's own strand centre for the CA pseudo-bond angle,
    124 degrees, with a pseudo-torsion of 180 degrees -- a fully extended
    strand. The resulting CA(i)-CA(i+2/3/4) distances are not put in by hand;
    they come out of the construction and are what the test checks.
    """
    half = np.deg2rad(angle) / 2.0
    advance = bond * np.sin(half)
    lateral = bond * np.cos(half)
    i = np.arange(n)
    return np.stack(
        [advance * i, lateral * (i % 2), np.zeros(n)], axis=-1
    )


def rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(degrees)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


# ---------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------


def test_ideal_helix_bond_length_is_3_8():
    """The construction's own CA-CA distance, which fixes the suite's constant."""
    bonds = gm.ca_bond_lengths(ideal_helix())
    assert bonds.mean() == pytest.approx(3.80, abs=0.01)
    assert bonds.std() == pytest.approx(0.0, abs=1e-9)
    assert gm.geometry_metrics(ideal_helix())["ca_ca_outlier_fraction"] == 0.0


def test_ideal_strand_bond_length_is_3_8():
    bonds = gm.ca_bond_lengths(ideal_strand())
    assert bonds.mean() == pytest.approx(3.80, abs=1e-9)
    assert bonds.std() == pytest.approx(0.0, abs=1e-9)


def test_clash_rate_excludes_the_short_separations():
    """An ideal helix has i/i+2 at ~5.4 A and i/i+3 at ~5.0 A: no clash either way.

    The point of the |i-j| >= 3 rule is that i/i+1 and i/i+2 are short by
    construction. This checks the rule is applied, by confirming a helix -- the
    most compact local geometry a backbone has -- scores zero.
    """
    rate, clashes, pairs = gm.clash_rate(ideal_helix())
    assert clashes == 0
    assert rate == 0.0
    n = 40
    assert pairs == (n * (n - 1) // 2) - (n - 1) - (n - 2)


def test_clash_rate_counts_a_planted_clash():
    ca = ideal_strand(n=20).copy()
    ca[15] = ca[0] + np.array([0.0, 0.0, 1.0])
    rate, clashes, pairs = gm.clash_rate(ca)
    assert clashes >= 1
    assert rate == clashes / pairs


def test_radius_of_gyration_of_a_known_shape():
    """A uniform spherical shell of radius R has Rg = R, exactly."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20000, 3))
    shell = 10.0 * v / np.linalg.norm(v, axis=-1, keepdims=True)
    assert gm.radius_of_gyration(shell) == pytest.approx(10.0, abs=0.05)


def test_rg_ratio_is_against_the_stated_scaling():
    ca = ideal_helix(n=100)
    metrics = gm.geometry_metrics(ca)
    assert metrics["radius_of_gyration_expected"] == pytest.approx(
        2.2 * 100**0.38
    )
    assert metrics["radius_of_gyration_ratio"] == pytest.approx(
        metrics["radius_of_gyration"] / metrics["radius_of_gyration_expected"]
    )
    # A 100-residue helix is a rod, not a globule, so it must be far above the
    # globular scaling. If it were not, the ratio would be measuring nothing.
    assert metrics["radius_of_gyration_ratio"] > 2.0


# ---------------------------------------------------------------------------
# 2. P-SEA
# ---------------------------------------------------------------------------


def test_ideal_helix_is_mainly_alpha():
    ca = ideal_helix(n=40)
    metrics = gm.secondary_metrics(ca)
    assert metrics["class"] == "mainly-alpha"
    assert metrics["strand_fraction"] == 0.0
    # The last residues have no CA(i+4) and so no strict criterion; the relaxed
    # extension recovers one of them. 38/40 = 0.95 is the ceiling for a
    # 40-residue helix, and a longer helix approaches 1.0.
    assert metrics["helix_fraction"] == pytest.approx(0.95, abs=1e-9)
    assert gm.secondary_metrics(ideal_helix(n=200))["helix_fraction"] > 0.98


def test_ideal_strand_is_strand():
    ca = ideal_strand(n=40)
    metrics = gm.secondary_metrics(ca)
    assert metrics["class"] == "mainly-beta"
    assert metrics["helix_fraction"] == 0.0
    assert metrics["strand_fraction"] > 0.9


def test_ideal_geometry_lands_inside_the_published_windows():
    """The distances P-SEA keys on, measured on both ideal constructions.

    This is the test that would catch a mistyped threshold: it reads the
    windows out of the module constants and checks the ideal geometry falls
    inside them, rather than trusting the assignment that uses them.
    """
    helix = ideal_helix(n=40)
    for k, key in ((2, "d2"), (3, "d3"), (4, "d4")):
        d = np.linalg.norm(helix[k:] - helix[:-k], axis=-1).mean()
        centre, tol = gm.PSEA_HELIX[key]
        assert abs(d - centre) <= tol, (key, d)
    strand = ideal_strand(n=40)
    for k, key in ((2, "d2"), (3, "d3"), (4, "d4")):
        d = np.linalg.norm(strand[k:] - strand[:-k], axis=-1).mean()
        centre, tol = gm.PSEA_STRAND[key]
        assert abs(d - centre) <= tol, (key, d)


def test_a_random_walk_is_coil_not_structure():
    """A chain with correct bond lengths and no structure must read as coil.

    Without this the suite could report high helix content for noise, which is
    the failure mode that would most flatter a generated sample.
    """
    rng = np.random.default_rng(7)
    steps = rng.normal(size=(120, 3))
    steps /= np.linalg.norm(steps, axis=-1, keepdims=True)
    ca = np.cumsum(3.8 * steps, axis=0)
    metrics = gm.secondary_metrics(ca)
    assert metrics["helix_fraction"] < 0.05
    assert metrics["coil_fraction"] > 0.85


def test_structure_class_rule_is_the_documented_one():
    assert gm.structure_class(0.0, 0.0) == "irregular"
    assert gm.structure_class(0.05, 0.05) == "irregular"
    assert gm.structure_class(0.60, 0.00) == "mainly-alpha"
    assert gm.structure_class(0.00, 0.40) == "mainly-beta"
    assert gm.structure_class(0.30, 0.30) == "alpha/beta"


def test_psea_matches_biotite_on_a_crystal_structure():
    """Agreement with biotite's independent implementation of Labesse 1997.

    biotite's ``annotate_sse`` is a separate implementation of the same paper,
    written from the paper rather than from this file, and it works on the
    AtomArray rather than on an extracted CA trace. Exact agreement is
    therefore evidence that the thresholds in ``PSEA_HELIX`` / ``PSEA_STRAND``
    and the run-length rules are the paper's, which is the one thing in this
    suite that a published diffusion number is compared against.
    """
    if not UBIQUITIN.exists():
        pytest.skip(f"{UBIQUITIN} missing; run scripts/fetch_test_structures.py")
    import biotite.structure as struc

    from aanca.structure import load_protein

    protein = load_protein(UBIQUITIN)
    ca, chain = gm.ca_trace(protein)
    mine = gm.psea_string(ca)
    theirs = "".join(
        struc.PSeaElement.to_symbols(
            struc.annotate_sse(protein.atoms[protein.atoms.chain_id == chain])
        )
    )
    assert len(mine) == len(theirs)
    assert mine == theirs
    # And it is the right answer for ubiquitin: a five-strand beta sheet with
    # one alpha helix of about three and a half turns. P-SEA puts that helix at
    # residues 23-33 (eleven residues, zero-based 22-32); DSSP calls it 23-34,
    # so P-SEA is one residue short at the C-terminal end -- the known cost of
    # assigning from the CA trace alone, with no hydrogen bonds to see.
    assert mine[22:33] == "a" * 11
    assert mine[21] == "c" and mine[33] == "c"
    assert mine.count("b") > mine.count("a")


# ---------------------------------------------------------------------------
# 3. TM-score
# ---------------------------------------------------------------------------


def test_d0_is_the_published_formula():
    assert gm.d0_of(100) == pytest.approx(1.24 * (100 - 15) ** (1 / 3) - 1.8)
    assert gm.d0_of(22) == pytest.approx(1.24 * 7 ** (1 / 3) - 1.8)
    assert gm.d0_of(20) == 0.5  # the documented floor below L = 22


def test_tm_score_of_a_structure_against_itself_is_one():
    for ca in (ideal_helix(n=60), ideal_strand(n=60)):
        assert gm.tm_score(ca, ca) == pytest.approx(1.0, abs=1e-9)


def test_tm_score_of_a_real_structure_against_itself_is_one():
    if not UBIQUITIN.exists():
        pytest.skip(f"{UBIQUITIN} missing; run scripts/fetch_test_structures.py")
    from aanca.structure import load_protein

    ca, _chain = gm.ca_trace(load_protein(UBIQUITIN))
    assert gm.tm_score(ca, ca) == pytest.approx(1.0, abs=1e-9)


def test_tm_score_is_invariant_to_rotation_and_translation():
    ca = ideal_helix(n=60)
    moved = ca @ rotation(np.array([0.3, -0.7, 0.5]), 137.0).T + np.array(
        [12.0, -300.0, 4.5]
    )
    assert gm.tm_score(moved, ca) == pytest.approx(1.0, abs=1e-9)
    other = ideal_strand(n=60)
    plain = gm.tm_score(other, ca)
    shifted = gm.tm_score(
        other @ rotation(np.array([1.0, 1.0, 0.0]), 65.0).T + 50.0, ca
    )
    assert shifted == pytest.approx(plain, abs=1e-9)


def test_tm_score_is_invariant_to_rotating_the_target_too():
    ca = ideal_strand(n=60)
    other = ideal_helix(n=60)
    rot = rotation(np.array([0.1, 0.2, 0.9]), 22.0)
    assert gm.tm_score(other, ca) == pytest.approx(
        gm.tm_score(other, ca @ rot.T - 7.0), abs=1e-9
    )


def test_tm_score_rejects_a_mirror_image():
    """A reflection is not a rotation, and must not score as a match.

    The Kabsch step removes reflections explicitly. Without that, a
    mirror-image backbone -- a left-handed helix -- would score 1.0 against the
    right-handed one, and the suite would call an unphysical structure a hit.
    """
    ca = ideal_helix(n=60)
    mirrored = ca * np.array([1.0, 1.0, -1.0])
    assert gm.tm_score(mirrored, ca) < 0.6


def test_tm_score_is_bounded_and_falls_for_unrelated_shapes():
    helix = ideal_helix(n=60)
    strand = ideal_strand(n=60)
    score = gm.tm_score(helix, strand)
    assert 0.0 < score < 0.5
    assert gm.tm_score(helix, helix) > score


def test_tm_score_handles_unequal_lengths_by_threading():
    """A fragment of a structure must score well against the whole.

    Normalised by the target length, a 30-residue piece of a 60-residue helix
    can reach at most about 0.5, and the threading has to find the offset or it
    would score far lower.
    """
    whole = ideal_helix(n=60)
    piece = whole[20:50]
    score = gm.tm_score(piece, whole)
    assert 0.35 < score <= 0.55
    # Normalised by the fragment instead, the same correspondence is near 1.
    assert gm.tm_score(piece, whole, l_norm=len(piece)) > 0.95


def test_kabsch_agrees_with_the_projects_own_superposition():
    """The numpy Kabsch here must match ``aanca.fape.ca_rmsd_on`` in torch.

    Two superposition routines in one project is one too many unless they
    agree; the suite keeps its own only to avoid a torch dependency.
    """
    torch = pytest.importorskip("torch")
    from aanca.fape import ca_rmsd_on

    rng = np.random.default_rng(3)
    a = ideal_helix(n=40) + rng.normal(scale=0.4, size=(40, 3))
    b = ideal_helix(n=40)
    rot, trans = gm.kabsch(a, b)
    mine = float(np.sqrt(np.mean(np.sum((a @ rot + trans - b) ** 2, axis=-1))))
    theirs = float(
        ca_rmsd_on(
            torch.tensor(a), torch.tensor(b), torch.tensor(a), torch.tensor(b)
        )
    )
    assert mine == pytest.approx(theirs, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. diversity and novelty
# ---------------------------------------------------------------------------


def test_two_identical_samples_give_one_cluster():
    ca = ideal_helix(n=60)
    out = gm.diversity_metrics([ca, ca.copy()])
    assert out["mean_pairwise_tm"] == pytest.approx(1.0, abs=1e-9)
    assert out["max_pairwise_tm"] == pytest.approx(1.0, abs=1e-9)
    assert out["clusters"]["tm_0.5"] == 1
    assert out["clusters"]["tm_0.6"] == 1
    assert out["cluster_fraction"]["tm_0.5"] == 0.5


def test_two_identical_samples_are_one_cluster_after_rigid_motion():
    ca = ideal_helix(n=60)
    moved = ca @ rotation(np.array([0.0, 1.0, 1.0]), 99.0).T + 20.0
    out = gm.diversity_metrics([ca, moved])
    assert out["mean_pairwise_tm"] == pytest.approx(1.0, abs=1e-9)
    assert out["clusters"]["tm_0.6"] == 1


def test_unrelated_samples_stay_separate_clusters():
    traces = [ideal_helix(n=60), ideal_strand(n=60)]
    out = gm.diversity_metrics(traces)
    assert out["max_pairwise_tm"] < 0.5
    assert out["clusters"]["tm_0.5"] == 2
    assert out["clusters"]["tm_0.6"] == 2


def test_single_linkage_chains_near_duplicates_together():
    sim = np.array(
        [
            [1.0, 0.7, 0.1],
            [0.7, 1.0, 0.7],
            [0.1, 0.7, 1.0],
        ]
    )
    assert len(set(gm.single_linkage_clusters(sim, 0.5))) == 1
    assert len(set(gm.single_linkage_clusters(sim, 0.8))) == 3


def test_novelty_finds_the_structure_a_sample_was_copied_from():
    reference = [ideal_helix(n=60), ideal_strand(n=60), ideal_helix(n=45)]
    sample = reference[1] @ rotation(np.array([1.0, 0.0, 0.0]), 41.0).T + 3.0
    out = gm.novelty_metrics([sample], reference, reference_names=["h60", "s60", "h45"])
    assert out["nearest_reference"] == ["s60"]
    assert out["max_tm"][0] == pytest.approx(1.0, abs=1e-9)
    assert out["fraction_above_0.5"] == 1.0


def test_novelty_is_low_for_a_shape_absent_from_the_reference_set():
    reference = [ideal_strand(n=60), ideal_strand(n=45)]
    out = gm.novelty_metrics([ideal_helix(n=60)], reference)
    assert out["mean_max_tm"] < 0.5
    assert out["fraction_above_0.5"] == 0.0


# ---------------------------------------------------------------------------
# the suite and its two input modes
# ---------------------------------------------------------------------------


def test_generated_and_natural_modes_share_one_code_path(tmp_path):
    """The same CA trace scored as a .npy sample and as a structure must agree.

    This is the claim the whole suite rests on: a generated number and an
    in-house reference number are comparable because they are produced by the
    same functions. If the two modes ever diverged, every comparison in the
    write-up would be between two different measurements.
    """
    if not UBIQUITIN.exists():
        pytest.skip(f"{UBIQUITIN} missing; run scripts/fetch_test_structures.py")
    from aanca.structure import load_protein

    ca, _chain = gm.ca_trace(load_protein(UBIQUITIN))
    path = tmp_path / "samples.npy"
    np.save(path, ca[None])
    through_npy = gm.load_generated(path)[0]
    assert np.allclose(through_npy, ca)

    from_structure = gm.score_traces([ca], names=["struct"], with_diversity=False)
    from_array = gm.score_traces(
        [through_npy], names=["struct"], with_diversity=False
    )
    assert from_structure.per_structure == from_array.per_structure


def test_crystal_structure_clash_rate_is_near_zero():
    """1UBQ reads 0.00037 on the CA clash rate: 1 pair in 2700.

    The bound asserted is the measured value plus a little, not a guess. The
    single pair inside 4.0 A is a real crystal contact between CA atoms three
    residues apart in a tight turn, which is why the bound is not exactly zero.
    """
    if not UBIQUITIN.exists():
        pytest.skip(f"{UBIQUITIN} missing; run scripts/fetch_test_structures.py")
    from aanca.structure import load_protein

    ca, _chain = gm.ca_trace(load_protein(UBIQUITIN))
    rate, _clashes, _pairs = gm.clash_rate(ca)
    assert rate < 0.001


def test_ca_trace_keeps_a_gap_free_run():
    """The trace must not bridge a chain break, or every number downstream lies."""
    if not UBIQUITIN.exists():
        pytest.skip(f"{UBIQUITIN} missing; run scripts/fetch_test_structures.py")
    from aanca.structure import load_protein

    protein = load_protein(UBIQUITIN)
    ca, chain = gm.ca_trace(protein)
    assert chain == "A"
    bonds = gm.ca_bond_lengths(ca)
    assert bonds.max() < 4.2
    assert len(ca) == 76


def test_aggregate_pools_fractions_by_residue_count():
    traces = [ideal_helix(n=20), ideal_strand(n=100)]
    suite = gm.score_traces(traces, names=["h", "s"], with_diversity=False)
    agg = suite.aggregate()
    per_structure = np.mean(
        [row["strand_fraction"] for row in suite.per_structure]
    )
    assert agg["pooled_strand_fraction"] != pytest.approx(per_structure)
    total = 120
    assert agg["pooled_strand_fraction"] == pytest.approx(
        sum(
            row["strand_fraction"] * row["n_residues"] for row in suite.per_structure
        )
        / total
    )
    assert agg["class_counts"] == {"mainly-alpha": 1, "mainly-beta": 1}


def test_table_renders_every_row():
    suite = gm.score_traces(
        [ideal_helix(n=40), ideal_strand(n=40)], names=["helix", "strand"]
    )
    table = gm.format_table(suite)
    assert "helix" in table and "strand" in table
    assert "mainly-alpha" in table and "mainly-beta" in table
    assert "clusters at 0.5" in table


def test_json_round_trips(tmp_path):
    import json

    suite = gm.score_traces([ideal_helix(n=40), ideal_strand(n=40)])
    path = tmp_path / "out.json"
    path.write_text(json.dumps(suite.as_json(), indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["thresholds"]["psea_helix"]["d3"] == [5.3, 0.5]
    assert len(loaded["per_structure"]) == 2
    assert loaded["diversity"]["clusters"]["tm_0.5"] == 2


def test_main_runs_the_generated_mode(tmp_path, capsys):
    path = tmp_path / "samples.npy"
    np.save(path, np.stack([ideal_helix(n=40), ideal_strand(n=40)]))
    out = tmp_path / "out.json"
    assert gm.main(["--generated", str(path), "--json", str(out)]) == 0
    captured = capsys.readouterr().out
    assert "mainly-alpha" in captured
    assert out.exists()


def test_rejects_a_badly_shaped_array(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((4, 10, 4)))
    with pytest.raises(ValueError, match="n_residues, 3"):
        gm.load_generated(path)
