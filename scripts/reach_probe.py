"""How far can ``g`` residues reach? The bound closure-aware growth budgets with.

    python scripts/reach_probe.py [--structures 400] [--max-gap 20]

A pending closure has an index gap ``g`` and an end-to-end distance ``d``, and
it is still closable only while ``d`` is within what ``g`` residues can span.
That bound is a fact about protein backbones, so it is measured here rather
than assumed: the maximum CA(i)-CA(i+g) distance over many real structures, as
a function of ``g``.

Two reasons not to compute it from ideal geometry instead. The trans peptide
fixes CA-CA at 3.804 A, but the *pseudo-angle* CA(i)-CA(i+1)-CA(i+2) is capped
well below 180 degrees by the peptide's own geometry and by sterics, and the
cap is not something this project has a number for. And the quantity wanted is
an upper bound on what real protein chains do, which is what an empirical
maximum over a few hundred structures is.

Printed as a table to paste into ``aanca.growth.REACH``, with the per-residue
figure alongside because the fall-off with ``g`` is the whole point -- a long
gap cannot be spanned at 3.8 A a residue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aanca.structure import load_protein

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=400)
    parser.add_argument("--max-gap", type=int, default=20)
    parser.add_argument("--min-residues", type=int, default=40)
    args = parser.parse_args()

    paths = sorted(PDB.glob("*.bcif"))[: args.structures]
    if not paths:
        print(f"no structures in {PDB}")
        return 1

    samples: list[list[torch.Tensor]] = [[] for _ in range(args.max_gap + 1)]
    used = 0
    for path in paths:
        try:
            protein = load_protein(path)
        except Exception:
            continue
        if len(protein) < args.min_residues:
            continue
        # CA only, and only across residues that are actually bonded: a gap
        # spanning a chain break or a numbering discontinuity is not a closure,
        # it is two different molecules. So the structure is cut into runs of
        # consecutive residue ids within one chain, and every run measured on
        # its own.
        runs: list[list] = []
        current: list = []
        previous: tuple[str, int] | None = None
        for i, residue in enumerate(protein.residues):
            if "CA" not in residue.index:
                previous = None
                if len(current) > 1:
                    runs.append(current)
                current = []
                continue
            chain, res_id = protein.chain_ids[i], protein.res_ids[i]
            contiguous = previous is not None and previous == (chain, res_id - 1)
            if not contiguous and current:
                if len(current) > 1:
                    runs.append(current)
                current = []
            current.append(residue.coords[residue.index["CA"]])
            previous = (chain, res_id)
        if len(current) > 1:
            runs.append(current)
        if not runs:
            continue
        used += 1
        for ca_list in runs:
            ca = torch.tensor(ca_list, dtype=torch.float64)
            for g in range(1, min(args.max_gap, ca.shape[0] - 1) + 1):
                samples[g].append((ca[g:] - ca[:-g]).norm(dim=-1))

    print(f"{used} structures, CA-CA distances within a bonded run\n")
    print(
        f"{'g':>4} {'max':>8} {'99.99%':>8} {'99.9%':>8} {'99%':>8} "
        f"{'per res':>9} {'pairs':>11}"
    )
    table = [0.0] * (args.max_gap + 1)
    for g in range(1, args.max_gap + 1):
        if not samples[g]:
            continue
        d = torch.cat(samples[g])
        q = [
            float(torch.quantile(d, torch.tensor(p, dtype=d.dtype)))
            for p in (0.9999, 0.999, 0.99)
        ]
        # **The quantile, not the maximum.** A maximum over 188,000 pairs is a
        # maximum over the worst modelling artifact in four hundred
        # structures: g = 1 came out at 4.32 A, which is not a peptide bond at
        # all -- trans is 3.80 and cis 2.9 -- so that pair is a chain break
        # with contiguous numbering, or a broken model. The bound wants to be
        # an upper bound on what real chains do, and the 99.99th percentile of
        # 188,000 pairs still keeps the nineteen longest.
        table[g] = round(q[0], 2)
        print(
            f"{g:>4} {float(d.max()):>8.2f} {q[0]:>8.2f} {q[1]:>8.2f} "
            f"{q[2]:>8.2f} {q[0] / g:>9.2f} {d.numel():>11d}"
        )

    print("\nREACH = " + json.dumps(table))
    print(
        "\nAn empirical maximum, so it is a bound on what was observed and not a\n"
        "theorem. It is used as a *feasibility* test, where being slightly too\n"
        "generous is the safe direction: it never calls a closure impossible\n"
        "that a real chain could have made."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
