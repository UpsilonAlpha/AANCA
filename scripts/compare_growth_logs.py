"""Put several growth runs' logs side by side.

`scripts/train_growth.py` writes checkpoints and **no history file** -- unlike
`scripts/train.py`, which writes `runs/<tag>/history.json`. So a growth run's
curve exists only in its SLURM `.out`, and comparing four arms means reading
four logs by eye and keeping the update numbers aligned in your head. That is
how g3 came to be called flat at 2,500 of 8,000 updates.

This parses the logs instead. It reads each run's tag out of its own banner, so
the job id never has to be matched to an arm by hand, and prints every arm's
numbers at each update where all of them have one.

    python scripts/compare_growth_logs.py logs/aanca-gpu-*.out
    python scripts/compare_growth_logs.py --column twist logs/*.out
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

#: The per-interval line, e.g.
#: ``   250  total  280.900  fape  0.808  rmsd  11.18 A  alive 95.7%  twist
#: 73.9d  rec  7.1%   2.01 s/u``
LINE = re.compile(
    r"^\s*(?P<update>\d+)\s+total\s+(?P<total>[-\d.]+)\s+"
    r"fape\s+(?P<fape>[-\d.]+)\s+rmsd\s+(?P<rmsd>[-\d.]+)\s*A\s+"
    r"alive\s+(?P<alive>[\d.]+)%"
    r"(?:\s+twist\s+(?P<twist>[\d.]+)d)?"
    r"(?:\s+rec\s+(?P<rec>[\d.]+)%)?"
)
TAG = re.compile(r"^args\s*:.*?--tag\s+(\S+)")
BIRTH = re.compile(r"(\w+) birth, crop")


def parse(path: Path) -> tuple[str, str, dict[int, dict[str, float]]]:
    tag, note, rows = path.stem, "", {}
    for line in path.read_text(errors="replace").splitlines():
        found = TAG.match(line.strip())
        if found:
            tag = found.group(1)
            # Flags that change what the arm *is*, pulled from the recorded
            # command rather than from the tag, because a tag is a name
            # somebody chose and the command is what ran.
            flags = []
            if "--morphogen" in line:
                flags.append("field")
            if "--orientation-weight" in line:
                weight = re.search(r"--orientation-weight\s+(\S+)", line)
                if weight and float(weight.group(1)) > 0:
                    flags.append(f"twist w={weight.group(1)}")
            if "--placement-steps" in line:
                steps = re.search(r"--placement-steps\s+(\S+)", line)
                if steps:
                    flags.append(f"window {steps.group(1)}")
            note = ", ".join(flags)
            continue
        birth = BIRTH.search(line)
        if birth:
            note = f"{birth.group(1)}{', ' + note if note else ''}"
            continue
        found = LINE.match(line)
        if not found:
            continue
        row = {
            key: float(value)
            for key, value in found.groupdict().items()
            if value is not None and key != "update"
        }
        rows[int(found.group("update"))] = row
    return tag, note, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument(
        "--column", default="rmsd",
        choices=("rmsd", "total", "fape", "alive", "twist", "rec"),
        help="which quantity to tabulate across arms",
    )
    parser.add_argument(
        "--every", type=int, default=0,
        help="only print updates that are a multiple of this, to shorten the table",
    )
    args = parser.parse_args()

    runs = []
    for name in args.logs:
        path = Path(name)
        if not path.exists():
            continue
        tag, note, rows = parse(path)
        if rows:
            runs.append((tag, note, rows))
    if not runs:
        print("no growth logs with readable interval lines among those paths")
        return 1

    # Sorted by tag so repeated invocations print the columns in one order,
    # which matters when the output is pasted beside an earlier run of this.
    runs.sort(key=lambda r: r[0])
    print(f"column: {args.column}")
    for tag, note, rows in runs:
        print(f"  {tag:<12} {note or '-':<28} {len(rows)} intervals, "
              f"to update {max(rows)}")

    updates = sorted(set().union(*(set(r[2]) for r in runs)))
    if args.every:
        updates = [u for u in updates if u % args.every == 0]
    header = "".join(f"{tag:>12}" for tag, _n, _r in runs)
    print(f"\n{'update':>8}{header}")
    for update in updates:
        cells = ""
        for _tag, _note, rows in runs:
            value = rows.get(update, {}).get(args.column)
            cells += f"{value:>12.2f}" if value is not None else f"{'-':>12}"
        print(f"{update:>8}{cells}")

    # The last value is the one a reader will quote, so give it its own line
    # rather than leaving it to be found at the bottom of a long table. And
    # name the run length: g3 was called flat at 2,500 of 8,000 and then
    # improved, so a comparison without the denominator is not a comparison.
    print(f"\nlatest {args.column}:")
    for tag, _note, rows in runs:
        last = max(rows)
        value = rows[last].get(args.column)
        print(f"  {tag:<12} {value if value is None else f'{value:.2f}'} "
              f"at update {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
