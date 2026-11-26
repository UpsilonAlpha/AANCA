"""The fold-class arm's readouts, computed from the run's artefacts alone.

Pre-registered at
``docs/claims/a-global-fold-class-label-tells-us-whether-option-2-failed-on-ambiguity-or-on-address.md``.
The trainer prints its readouts at every evaluation; this reads the FINAL state
off ``runs/<tag>/history.json`` and ``runs/<tag>/samples.npz`` so a verdict does
not depend on scraping a Slurm log, and so **readout 0a** -- which the trainer
does not print -- can be computed at all.

Readout 0a is the paired per-residue displacement between samples that differ
in nothing but the label. :func:`train_conditioned.sample_batch` emits the
classes in a fixed inner loop over one shared crop and one shared noise seed,
so ``samples[p * n_classes + c]`` is position ``p`` conditioned on class ``c``
and the whole block is paired by construction. **A zero here means the
conditioning never reached the coordinates**, which is a bug and not a result;
nothing else in the table may be read until it is non-zero.

``sigma`` and ``start_rmsd`` are checked single-valued because this arm is
compared crop-for-crop against the refuted MVP at the same two numbers. If
either moved, the comparison is void and the script says so rather than
printing a table that reads as if it were not.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

#: ``json`` refuses a bare ``NaN`` written by an older trainer.
#: ``train_conditioned.py`` emits ``null``, so a hit here is a REGRESSION and
#: the caller is told about it rather than it being silently repaired.
BARE_NAN = re.compile(r"\bNaN\b")


def read_history(path: Path) -> tuple[list[dict], int]:
    """``(rows, nan_tokens)`` from a history file, bare ``NaN`` tolerated."""
    text = path.read_text()
    hits = len(BARE_NAN.findall(text))
    if hits:
        text = BARE_NAN.sub("null", text)
    rows = json.loads(text)
    if isinstance(rows, dict):
        rows = rows.get("history", [])
    return list(rows), hits


def single_valued(rows: list[dict], key: str) -> list[float]:
    """The distinct finite values of ``key`` over ``rows``, rounded to 4 dp."""
    seen: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or not isinstance(value, (int, float)):
            continue
        if value != value:
            continue
        rounded = round(float(value), 4)
        if rounded not in seen:
            seen.append(rounded)
    return seen


def block_means(rows: list[dict], key: str, width: int) -> list[tuple[int, float, int]]:
    """``(block_end, mean, n)`` per ``width``-update block of ``key``."""
    buckets: dict[int, list[float]] = {}
    for row in rows:
        update = int(row.get("update", 0))
        value = row.get(key)
        if value is None or not isinstance(value, (int, float)) or value != value:
            continue
        block = ((update - 1) // width + 1) * width
        buckets.setdefault(block, []).append(float(value))
    return [
        (block, sum(values) / len(values), len(values))
        for block, values in sorted(buckets.items())
    ]


def split_by_offsets(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """The ragged sample layout, which is flat ``(total, 3)`` plus ``offsets``."""
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(flat):
        raise ValueError(
            f"`offsets` does not span the {len(flat)} rows of the coordinates"
        )
    return [flat[int(a) : int(b)] for a, b in zip(offsets, offsets[1:])]


def paired_displacement(
    traces: list[np.ndarray], labels: np.ndarray, n_classes: int
) -> dict[str, float]:
    """READOUT 0a. Mean per-residue displacement between paired samples.

    Paired means same crop and same noise seed, two different labels, so a
    non-zero here is the label and nothing else. No superposition is applied
    and none is wanted: both samples were started from the *same* prior
    coordinates, so the raw difference is the displacement the label caused.
    """
    per_pair: list[float] = []
    alpha_beta: list[float] = []
    positions = len(traces) // n_classes
    for position in range(positions):
        block = traces[position * n_classes : (position + 1) * n_classes]
        rows = labels[position * n_classes : (position + 1) * n_classes]
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                left, right = block[i], block[j]
                if left.shape != right.shape or not left.size:
                    continue
                if not (np.isfinite(left).all() and np.isfinite(right).all()):
                    continue
                moved = float(
                    np.linalg.norm(left - right, axis=1).mean()
                )
                per_pair.append(moved)
                if {int(rows[i]), int(rows[j])} == {0, 1}:
                    alpha_beta.append(moved)
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    return {
        "mean": mean(per_pair),
        "min": min(per_pair) if per_pair else float("nan"),
        "max": max(per_pair) if per_pair else float("nan"),
        "n_pairs": float(len(per_pair)),
        "alpha_beta": mean(alpha_beta),
        "n_alpha_beta": float(len(alpha_beta)),
        "positions": float(positions),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("runs/cond"))
    p.add_argument("--block", type=int, default=2000)
    p.add_argument("--expected-updates", type=int, default=20000)
    args = p.parse_args(argv)

    history = args.run / "history.json"
    rows, nan_tokens = read_history(history)
    print(f"history {history}  rows {len(rows)}")
    if nan_tokens:
        print(f"  REGRESSION: {nan_tokens} bare NaN tokens -- this trainer emits null")
    last = max((int(r.get("update", 0)) for r in rows), default=0)
    verdict = "FINISHED" if last >= args.expected_updates else "TRUNCATED"
    print(f"  last update {last} of {args.expected_updates}  {verdict}")

    for key, want in (("sigma", 8.899), ("start_rmsd", 15.205)):
        values = single_valued(rows, key)
        ok = len(values) == 1 and abs(values[0] - want) < 5e-4
        state = "single-valued, matches the MVP" if ok else "MOVED -- comparison VOID"
        print(f"  {key}: {values}  {state}")

    for key in ("rmsd", "total", "window", "move"):
        blocks = block_means(rows, key, args.block)
        if not blocks:
            continue
        shown = "  ".join(f"{m:.3f}" for _b, m, _n in blocks)
        print(f"  {key} by {args.block}: {shown}")

    samples = args.run / "samples.npz"
    if not samples.exists():
        print(f"no samples at {samples}")
        return 0
    with np.load(samples, allow_pickle=False) as blob:
        flat = np.asarray(blob["ca"], dtype=np.float64)
        offsets = np.asarray(blob["offsets"], dtype=np.int64)
        labels = np.asarray(blob["label"], dtype=np.int64)
        classes = [str(x) for x in blob["fold_classes"]]
        conditioned = [str(x) for x in blob["conditioned_class"]]
        produced = [str(x) for x in blob["produced_class"]]
        helix = np.asarray(blob["helix_fraction"], dtype=np.float64)
        strand = np.asarray(blob["strand_fraction"], dtype=np.float64)
        converged = np.asarray(blob["converged"])
        steps = np.asarray(blob["steps"], dtype=np.float64)
        last_move = np.asarray(blob["last_move"], dtype=np.float64)
    traces = split_by_offsets(flat, offsets)
    print(f"\nsamples {samples}  n {len(traces)}  classes {classes}")

    zero_a = paired_displacement(traces, labels, len(classes))
    print(
        "READOUT 0a paired displacement: mean {mean:.4f} A  min {min:.4f}  "
        "max {max:.4f}  over {n_pairs:.0f} pairs, {positions:.0f} positions".format(
            **zero_a
        )
    )
    print(
        f"  alpha-vs-beta only: {zero_a['alpha_beta']:.4f} A "
        f"over {zero_a['n_alpha_beta']:.0f} pairs"
    )
    if not zero_a["mean"] > 0:
        print("  INERT -- this is a BUG and no readout below may be read")
        return 1

    def mean_where(name: str, values: np.ndarray) -> tuple[float, int]:
        rows = [
            float(v)
            for c, v in zip(conditioned, values)
            if c == name and v == v
        ]
        return (sum(rows) / len(rows) if rows else float("nan"), len(rows))

    helix_a, n_a = mean_where("mainly-alpha", helix)
    helix_b, n_b = mean_where("mainly-beta", helix)
    print(
        f"READOUT 0b helix gap: alpha {helix_a:.4f} (n {n_a})  "
        f"beta {helix_b:.4f} (n {n_b})  gap {helix_a - helix_b:+.4f}"
    )
    pooled = float(np.nansum(helix) + np.nansum(strand))
    if pooled <= 0:
        print(
            "  NO DYNAMIC RANGE -- the output carries no secondary structure, so "
            "0b is not a measurement of the label (per the amendment)"
        )

    table: Counter[str] = Counter()
    for c, q in zip(conditioned, produced):
        if q:
            table[f"{c} -> {q}"] += 1
    print("READOUT 5 confusion, conditioned -> produced:")
    for key, count in sorted(table.items()):
        print(f"  {key}  {count}")

    settled = int(np.count_nonzero(converged))
    print(
        f"sampler: converged {settled}/{len(traces)}  "
        f"steps median {float(np.median(steps)):.1f}  "
        f"last_move median {float(np.median(last_move)):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
