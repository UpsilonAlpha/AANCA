"""Draw the training curves straight out of the SLURM logs, as one SVG.

    python scripts/plot_curves.py logs/aanca-gpu-30976061.out logs/aanca-gpu-30976062.out
    python scripts/plot_curves.py --out /tmp/v4.svg logs/aanca-gpu-3097606*.out

The logs are the record. ``runs/<tag>/history.json`` holds the same numbers, but
only for a run that is still on disk under its own tag, and a chained run's
segments each write a separate log -- so parsing the logs is what lets a curve be
redrawn for any arm, from any machine, including after a run is moved aside.

One positional argument per log. Logs are grouped into arms by the ``--tag`` on
the job banner's ``args`` line, so the segments of a chained run merge into one
curve automatically and two arms submitted from the same script come out as two
series. Where segments overlap -- a segment that was killed after an evaluation
and resumed from an earlier checkpoint -- the later log wins.

Deliberately stdlib-only. This has to run on a login node and on a laptop
without touching the environment that the training itself depends on, so it
writes the SVG by hand rather than importing a plotting library. See
``PANELS`` to add a metric.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Fields on the per-kappa rows of an evaluation block, after the label.
COLUMNS = ("chi1", "chi2", "chi3", "chi4", "acc", "acc_core", "acc_surf",
           "rmsd", "rmsd_core", "rmsd_surf")

TAG = re.compile(r"--tag\s+(\S+)")
NOOP = re.compile(r"--noop-weight\s+(\S+)")
NUMBERED = re.compile(r"^held out after (\d+) updates:")
#: The first evaluation of a run is printed before update 1 and labelled by what
#: the weights are ("untrained:", or a warm start's own line), not by a number.
#: A resumed segment prints no such block at all, so an unnumbered block is
#: always update 0.
UNNUMBERED = re.compile(r"^(untrained:|at update 0, warm-started)")
ROW = re.compile(r"^(kappa \d+|pooled)\s+(.*)")
PROFILE = re.compile(r"^mean \|delta\| per step \(deg\):\s*(.*)")
DRIFT = re.compile(r"^drift from a native start:\s*([\d.]+)")


@dataclass
class Evaluation:
    """One held-out evaluation block, keyed by the update it was taken at."""

    update: int
    metrics: dict[str, float] = field(default_factory=dict)
    profile: list[float] = field(default_factory=list)


@dataclass
class Panel:
    """One stacked plot: a label, what to pull out of each evaluation, a unit."""

    title: str
    key: str
    unit: str


#: The four views worth having side by side: the accuracy the run is optimising,
#: the two numbers that say whether the native structure is a fixed point, and
#: the per-step profile that says *where* in the rollout the motion is.
PANELS = (
    Panel("pooled chi1 MAE", "pooled.chi1", "deg"),
    Panel("kappa 100 chi1 MAE -- near-native start", "kappa 100.chi1", "deg"),
    Panel("drift from a native start", "drift", "A"),
)

#: Categorical, in fixed order, with a dash pattern so the series are still
#: distinguishable in greyscale and to a colourblind reader.
SERIES = (("#2a78d6", "none"), ("#eb6834", "7 4"), ("#1baf7a", "2 3"),
          ("#6250d6", "10 3 2 3"))

WIDTH = 900
PLOT_H = 190
PAD = {"left": 62, "right": 14, "top": 22, "bottom": 36}
GAP = 30


def parse_log(path: Path) -> tuple[str, str, list[Evaluation]]:
    """Return ``(tag, label, evaluations)`` for one log.

    The label carries the one setting the arms differ by, because a tag alone
    ("v4_gpu" against "v4_noop4") says which run it was and not what was varied.
    """
    tag, noop, evaluations = path.stem, None, []
    current: Evaluation | None = None

    for line in path.read_text(errors="replace").splitlines():
        line = line.rstrip()
        if line.startswith("args"):
            found = TAG.search(line)
            tag = found.group(1) if found else tag
            weight = NOOP.search(line)
            noop = weight.group(1) if weight else "0"
            continue

        numbered = NUMBERED.match(line)
        if numbered or UNNUMBERED.match(line):
            current = Evaluation(int(numbered.group(1)) if numbered else 0)
            evaluations.append(current)
            continue
        if current is None:
            continue

        row = ROW.match(line)
        if row:
            values = row.group(2).split()
            for name, value in zip(COLUMNS, values):
                current.metrics[f"{row.group(1)}.{name}"] = float(value)
            continue
        profile = PROFILE.match(line)
        if profile:
            current.profile = [float(v) for v in profile.group(1).split()]
            continue
        drift = DRIFT.match(line)
        if drift:
            current.metrics["drift"] = float(drift.group(1))
            # The drift line closes a block; anything after it is training
            # output until the next header.
            current = None

    label = tag if noop in (None, "0") else f"{tag} (noop weight {noop})"
    return tag, label, evaluations


def merge(logs: list[Path]) -> list[tuple[str, list[Evaluation]]]:
    """Group logs into arms by tag and splice each arm's segments together.

    Keyed by update number so a resumed segment that re-covers ground already
    in an earlier log replaces it rather than drawing a second line back
    through the same x range.
    """
    arms: dict[str, tuple[str, dict[int, Evaluation]]] = {}
    for path in logs:
        tag, label, evaluations = parse_log(path)
        _, merged = arms.setdefault(tag, (label, {}))
        for evaluation in evaluations:
            merged[evaluation.update] = evaluation
    return [
        (label, [points[u] for u in sorted(points)])
        for label, points in arms.values()
        if points
    ]


def ticks(low: float, high: float, target: int = 5) -> list[float]:
    """Round values spanning ``[low, high]``, about ``target`` of them."""
    if high <= low:
        return [low]
    raw = (high - low) / target
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(
        (m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw),
        10 * magnitude,
    )
    value = step * math.ceil(low / step)
    out = []
    while value <= high + step * 1e-9:
        out.append(round(value, 10))
        value += step
    return out or [low, high]


def fmt(value: float) -> str:
    return f"{value:g}"


class Canvas:
    """Accumulates SVG fragments. Panels are stacked, each with its own axes."""

    def __init__(self, panels: int) -> None:
        self.height = panels * (PLOT_H + PAD["top"] + PAD["bottom"] + GAP) + 46
        self.parts: list[str] = []

    def text(self, x: float, y: float, body: str, cls: str = "t",
             anchor: str = "start") -> None:
        safe = body.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
            f'text-anchor="{anchor}">{safe}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str) -> None:
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="{cls}"/>'
        )

    def path(self, points: list[tuple[float, float]], colour: str,
             dash: str) -> None:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
            for i, (x, y) in enumerate(points)
        )
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-dasharray="{dash}"/>'
        )

    def svg(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{self.height}" viewBox="0 0 {WIDTH} {self.height}" '
            f'font-family="system-ui, sans-serif">'
            "<style>"
            ".t{font-size:11px;fill:#52514e}"
            ".h{font-size:13px;fill:#0b0b0b}"
            ".grid{stroke:#e1e0d9;stroke-width:1}"
            ".axis{stroke:#c3c2b7;stroke-width:1}"
            "@media (prefers-color-scheme: dark){"
            ".t{fill:#c3c2b7}.h{fill:#f0efec}"
            ".grid{stroke:#2c2c2a}.axis{stroke:#383835}}"
            "</style>"
            + "".join(self.parts)
            + "</svg>"
        )


def draw_panel(canvas: Canvas, top: float, title: str, unit: str,
               x_label: str, series: list[tuple[str, list[tuple[float, float]]]]
               ) -> None:
    """One plot. ``series`` is ``(label, [(x, y), ...])`` already in data units."""
    left, right = PAD["left"], WIDTH - PAD["right"]
    y0, y1 = top + PAD["top"], top + PAD["top"] + PLOT_H

    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points]
    if not xs:
        return
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    margin = (y_hi - y_lo) * 0.12 or max(abs(y_hi) * 0.1, 0.1)
    y_lo, y_hi = y_lo - margin, y_hi + margin

    def sx(value: float) -> float:
        span = x_hi - x_lo or 1
        return left + (value - x_lo) / span * (right - left)

    def sy(value: float) -> float:
        span = y_hi - y_lo or 1
        return y1 - (value - y_lo) / span * (y1 - y0)

    canvas.text(left, top + 14, f"{title} ({unit})", "h")
    for value in ticks(y_lo, y_hi):
        y = sy(value)
        canvas.line(left, y, right, y, "grid")
        canvas.text(left - 8, y + 4, fmt(value), "t", "end")
    canvas.line(left, y1, right, y1, "axis")
    for value in ticks(x_lo, x_hi, 6):
        if x_lo <= value <= x_hi:
            canvas.text(sx(value), y1 + 16, fmt(value), "t", "middle")
    canvas.text((left + right) / 2, y1 + 31, x_label, "t", "middle")

    for index, (_, points) in enumerate(series):
        colour, dash = SERIES[index % len(SERIES)]
        canvas.path([(sx(x), sy(y)) for x, y in points], colour, dash)


def legend(canvas: Canvas, y: float, labels: list[str]) -> None:
    x = PAD["left"]
    for index, label in enumerate(labels):
        colour, dash = SERIES[index % len(SERIES)]
        canvas.parts.append(
            f'<line x1="{x}" y1="{y - 4:.1f}" x2="{x + 20}" y2="{y - 4:.1f}" '
            f'stroke="{colour}" stroke-width="2" stroke-dasharray="{dash}"/>'
        )
        canvas.text(x + 26, y, label, "t")
        x += 34 + 7.0 * len(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("curves.svg"))
    parser.add_argument(
        "--profile", action="store_true",
        help="add the mean |delta| per rollout step at each arm's last evaluation",
    )
    args = parser.parse_args()

    arms = merge(args.logs)
    if not arms:
        print("no evaluations found -- are these training logs?")
        return 1

    panels = list(PANELS)
    canvas = Canvas(len(panels) + (1 if args.profile else 0))
    legend(canvas, 16, [label for label, _ in arms])

    top = 26.0
    for panel in panels:
        series = [
            (label, [(e.update, e.metrics[panel.key]) for e in points
                     if panel.key in e.metrics])
            for label, points in arms
        ]
        draw_panel(canvas, top, panel.title, panel.unit, "update", series)
        top += PLOT_H + PAD["top"] + PAD["bottom"] + GAP

    if args.profile:
        series = []
        for label, points in arms:
            with_profile = [e for e in points if e.profile]
            if not with_profile:
                continue
            last = with_profile[-1]
            series.append((
                label,
                [(i + 1, v) for i, v in enumerate(last.profile)],
            ))
        updates = {
            e.update for _, points in arms for e in points[-1:] if e.profile
        }
        at = f" at update {max(updates)}" if updates else ""
        draw_panel(canvas, top, f"mean |delta| per rollout step{at}", "deg",
                   "rollout step", series)

    args.out.write_text(canvas.svg())
    for label, points in arms:
        print(f"{label}: {len(points)} evaluations, to update {points[-1].update}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
