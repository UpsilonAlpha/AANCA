"""The benchmark's target set and its reference table.

Neither is interesting code and both are easy to get quietly wrong: a target
set off by a few entries, or a reference row with a column in the wrong place,
produces a comparison table that looks right and says something false.
"""

from __future__ import annotations

from aanca.casp import (
    CASP_RANGES,
    OTHER_PROTOCOLS,
    REFERENCE,
    target_number,
    targets_for,
)


def archive(root, layout: dict[str, list[str]]):
    """Write empty ``.pdb`` files in the archive's directory layout."""
    for method, stems in layout.items():
        directory = root / method / "casp_all"
        directory.mkdir(parents=True)
        for stem in stems:
            (directory / f"{stem}.pdb").write_text("")
    return root


def test_older_casp_targets_are_not_counted_as_casp13(tmp_path):
    """The archive holds CASP10 and CASP11 entries too, and T0658 sorts into
    the same prefix as T0949. Filtering on 'T0' would silently widen the set
    and give an average that matches no published number."""
    root = archive(tmp_path, {
        "native": ["T0658-D1", "T0949-D1", "T1022-D1", "T1024-D1"],
        "faspr": ["T0658-D1", "T0949-D1", "T1022-D1", "T1024-D1"],
    })
    assert targets_for(root, 13, "faspr") == ["T0949-D1", "T1022-D1"]
    assert targets_for(root, 14, "faspr") == ["T1024-D1"]


def test_the_set_is_the_intersection_over_methods(tmp_path):
    """Our row has to be an average over the same structures theirs is. A
    target one baseline failed on is missing from its directory, and keeping
    it for us only would compare two different averages."""
    root = archive(tmp_path, {
        "native": ["T0949-D1", "T0950-D1", "T0951-D1"],
        "faspr": ["T0949-D1", "T0950-D1", "T0951-D1"],
        "scwrl": ["T0949-D1", "T0951-D1"],
    })
    assert targets_for(root, 13, ["faspr", "scwrl"]) == ["T0949-D1", "T0951-D1"]


def test_a_missing_method_directory_yields_no_targets(tmp_path):
    """Better an empty table than one silently averaged over everything."""
    root = archive(tmp_path, {"native": ["T0949-D1"]})
    assert targets_for(root, 13, "diffpack") == []


def test_target_number_ignores_names_it_does_not_understand():
    assert target_number("T0949-D1") == 949
    assert target_number("1ABC") is None
    assert target_number("native") is None


def test_every_reference_row_has_the_right_shape():
    for casp, methods in REFERENCE.items():
        assert casp in CASP_RANGES
        for method, (mae, accuracy, rmsd) in methods.items():
            assert len(mae) == 4, f"{method}: not four chi"
            assert len(accuracy) == 3, f"{method}: not all/core/surface"
            assert len(rmsd) == 3, f"{method}: not all/core/surface"
            assert all(0 < v < 180 for v in mae), f"{method}: MAE out of range"
            assert all(0 < v < 100 for v in accuracy), f"{method}: not a percent"
            assert all(0 < v < 10 for v in rmsd), f"{method}: RMSD out of range"


def test_diffpack_is_the_best_published_row_on_both_sets():
    """A transcription slip would most likely show up as the state of the art
    no longer being the state of the art."""
    for casp in (13, 14):
        best = min(REFERENCE[casp].items(), key=lambda kv: kv[1][0][0])
        assert best[0] == "diffpack", f"CASP{casp}: {best[0]} beats diffpack on chi1"


def test_flowpacker_is_kept_out_of_the_comparable_table():
    """Its accuracy is per chi, not per residue, and its targets are a
    different set. Merging it in would understate everyone else by ten
    points without anything in the table saying so."""
    assert "flowpacker" not in REFERENCE[13]
    assert "flowpacker" not in REFERENCE[14]
    other = OTHER_PROTOCOLS["flowpacker"]
    assert len(other[13][1]) == 4, "per-chi accuracy has four columns, not three"
    assert "per-chi accuracy" in other["source"]
