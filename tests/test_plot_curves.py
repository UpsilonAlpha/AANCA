"""The log parser behind ``scripts/plot_curves.py``.

The drawing is eyeballed; the parsing is not, because every way it can go wrong
produces a plausible-looking curve. An unnumbered first block silently becoming
update 0 of the *wrong* arm, or two chained segments drawing as two lines over
the same x range, would both look like a result.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "plot_curves", ROOT / "scripts" / "plot_curves.py"
)
plot_curves = importlib.util.module_from_spec(SPEC)
sys.modules["plot_curves"] = plot_curves
SPEC.loader.exec_module(plot_curves)

HEADER = (
    "                   chi1   chi2   chi3   chi4  acc all    core    surf"
    "  rmsd all    core    surf"
)


def block(label: str, pooled: float, kappa100: float, drift: float) -> str:
    return "\n".join([
        label,
        HEADER,
        f"kappa 0           56.39  41.84  73.84  57.22     17.1    24.8    10.3"
        f"     1.489   1.399   1.541",
        f"kappa 100          {kappa100:.2f}   5.11   4.61   5.06     98.7    99.0"
        f"    98.2     0.165   0.155   0.170",
        f"pooled            {pooled:.2f}  19.00  29.49  24.28     66.1    70.0"
        f"    62.1     0.647   0.605   0.672",
        "mean |delta| per step (deg): 0.1  3.2  4.2",
        f"drift from a native start: {drift:.3f} A",
        "",
        "     125  loss  -1.500  grad   1.713   2.70 s/update",
    ])


def write(path: Path, tag: str, extra: str, *blocks: str) -> Path:
    path.write_text("\n".join([
        "job        : 1 on bun161",
        f"args       : --tag {tag} --split train --updates 60000{extra}",
        "",
        *blocks,
    ]))
    return path


def test_the_first_unnumbered_block_is_update_zero(tmp_path):
    """It is labelled by what the weights are, not by a number, in both the
    untrained and the warm-started case -- and it is the row every later point
    is read against, so losing it loses the comparison."""
    log = write(tmp_path / "a.out", "v4_gpu", "",
                block("untrained:", 23.17, 5.01, 1.697),
                block("held out after 500 updates:", 22.44, 4.76, 1.439))
    _, _, evaluations = plot_curves.parse_log(log)
    assert [e.update for e in evaluations] == [0, 500]
    assert evaluations[0].metrics["pooled.chi1"] == 23.17
    assert evaluations[1].metrics["drift"] == 1.439

    warm = write(tmp_path / "w.out", "v4_gpu", "",
                 block("at update 0, warm-started from runs/v3_gpu/last.pt:",
                       23.17, 5.01, 1.697))
    _, _, warm_evaluations = plot_curves.parse_log(warm)
    assert [e.update for e in warm_evaluations] == [0]


def test_the_label_names_the_setting_the_arms_differ_by():
    """A tag says which run it was; the comparison needs what was varied."""
    assert plot_curves.NOOP.search("--noop-weight 4").group(1) == "4"


def test_an_arm_with_a_noop_weight_is_labelled_with_it(tmp_path):
    log = write(tmp_path / "n.out", "v4_noop4", " --noop-weight 4",
                block("untrained:", 23.17, 5.01, 1.697))
    tag, label, _ = plot_curves.parse_log(log)
    assert tag == "v4_noop4"
    assert "noop weight 4" in label


def test_segments_of_one_run_merge_into_one_curve(tmp_path):
    """Each chained segment writes its own log under its own job id. Two logs
    with the same tag are one arm, in update order."""
    first = write(tmp_path / "s1.out", "v4_gpu", "",
                  block("untrained:", 23.17, 5.01, 1.697),
                  block("held out after 500 updates:", 22.44, 4.76, 1.439))
    second = write(tmp_path / "s2.out", "v4_gpu", "",
                   block("held out after 1000 updates:", 22.17, 4.21, 1.193))
    arms = plot_curves.merge([first, second])
    assert len(arms) == 1
    label, points = arms[0]
    assert label == "v4_gpu"
    assert [p.update for p in points] == [0, 500, 1000]


def test_a_resumed_segment_replaces_ground_it_recovers(tmp_path):
    """A segment killed after an evaluation resumes from the last checkpoint,
    so an update can appear in two logs. The later log is the one that ran."""
    first = write(tmp_path / "s1.out", "v4_gpu", "",
                  block("held out after 500 updates:", 22.44, 4.76, 1.439))
    second = write(tmp_path / "s2.out", "v4_gpu", "",
                   block("held out after 500 updates:", 21.00, 4.00, 1.000))
    _, points = plot_curves.merge([first, second])[0]
    assert len(points) == 1
    assert points[0].metrics["pooled.chi1"] == 21.00


def test_two_tags_are_two_arms(tmp_path):
    first = write(tmp_path / "a.out", "v4_gpu", "",
                  block("untrained:", 23.17, 5.01, 1.697))
    second = write(tmp_path / "b.out", "v4_noop4", " --noop-weight 4",
                   block("untrained:", 23.17, 5.01, 1.697))
    assert len(plot_curves.merge([first, second])) == 2


def test_training_lines_after_a_block_are_not_read_as_data(tmp_path):
    """The drift line closes a block. Without that, the ``125 loss ...`` lines
    and the next banner would keep landing in the last evaluation."""
    log = write(tmp_path / "a.out", "v4_gpu", "",
                block("untrained:", 23.17, 5.01, 1.697))
    _, _, evaluations = plot_curves.parse_log(log)
    assert set(evaluations[0].metrics) == {
        f"{row}.{column}"
        for row in ("kappa 0", "kappa 100", "pooled")
        for column in plot_curves.COLUMNS
    } | {"drift"}
    assert evaluations[0].profile == [0.1, 3.2, 4.2]


def test_an_svg_comes_out_with_both_series(tmp_path):
    """End to end, including the panel that is only drawn with --profile."""
    first = write(tmp_path / "a.out", "v4_gpu", "",
                  block("untrained:", 23.17, 5.01, 1.697),
                  block("held out after 500 updates:", 22.44, 4.76, 1.439))
    second = write(tmp_path / "b.out", "v4_noop4", " --noop-weight 4",
                   block("untrained:", 23.17, 5.01, 1.697),
                   block("held out after 500 updates:", 22.61, 4.62, 1.331))
    out = tmp_path / "curves.svg"
    sys.argv = ["plot_curves.py", "--profile", "--out", str(out),
                str(first), str(second)]
    assert plot_curves.main() == 0
    svg = out.read_text()
    assert svg.startswith("<svg")
    for colour, _ in plot_curves.SERIES[:2]:
        assert colour in svg
    assert "mean |delta| per rollout step at update 500" in svg
    # Four panels: the three in PANELS plus the profile.
    assert svg.count('class="h"') == len(plot_curves.PANELS) + 1


def test_a_log_with_no_evaluations_is_an_error(tmp_path, capsys):
    empty = tmp_path / "e.out"
    empty.write_text("job        : 1 on bun161\n")
    sys.argv = ["plot_curves.py", "--out", str(tmp_path / "x.svg"), str(empty)]
    assert plot_curves.main() == 1
    assert "no evaluations" in capsys.readouterr().out
