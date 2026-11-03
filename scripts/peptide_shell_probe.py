"""Can the chain be threaded off a cloud from the C-to-N peptide bond?

    python scripts/peptide_shell_probe.py [--structures 300]

The CA-CA shell does not separate cleanly -- ``scripts/ca_shell_probe.py``
measured a 0.86 % overlap and 4 % of residues carrying a third CA at peptide
range -- so thresholding CA-CA leaves a chain that cannot be read off
unambiguously.

The peptide bond should be far better, for two reasons. It is **short**: C-N is
1.329 A, while a non-bonded carbon and nitrogen are held apart by van der Waals
contact at roughly 1.70 + 1.55 = 3.25 A, so the two populations have no reason
to meet. And it is **directed**: the bond runs from residue *i*'s C to residue
*i+1*'s N, so finding it tells you not only that two residues are joined but
which way round, which is what fixes the N-to-C direction of a cloud that has
no sequence indices in it.

Reports the bonded and non-bonded C-N populations, the gap between them, and
the in- and out-degree of the directed C-to-N graph at a threshold -- which must
be at most one each way, or the result is not a set of paths and cannot be a
set of chains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from aanca.structure import load_protein

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "pdb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=300)
    parser.add_argument("--min-residues", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()

    paths = sorted(PDB.glob("*.bcif"))[: args.structures]
    if not paths:
        print(f"no structures in {PDB}")
        return 1

    bonded: list[torch.Tensor] = []
    other_close: list[torch.Tensor] = []
    other_min = float("inf")
    worst: tuple[float, str] | None = None
    out_degree: dict[int, int] = {}
    in_degree: dict[int, int] = {}
    wrong_way = 0
    total_bonds = 0
    used = 0

    for path in paths:
        try:
            protein = load_protein(path)
        except Exception:
            continue
        if len(protein) < args.min_residues:
            continue
        keep = [
            i
            for i, r in enumerate(protein.residues)
            if {"N", "C"} <= set(r.index)
        ]
        if len(keep) < 3:
            continue
        used += 1
        nitrogen = torch.tensor(
            np.array(
                [
                    protein.residues[i].coords[protein.residues[i].index["N"]]
                    for i in keep
                ]
            ),
            dtype=torch.float64,
        )
        carbon = torch.tensor(
            np.array(
                [
                    protein.residues[i].coords[protein.residues[i].index["C"]]
                    for i in keep
                ]
            ),
            dtype=torch.float64,
        )
        chain = [protein.chain_ids[i] for i in keep]
        res_id = torch.tensor([protein.res_ids[i] for i in keep])

        # distance[i, j] = |C(i) - N(j)|, so it is directed by construction.
        distance = torch.cdist(carbon, nitrogen)
        same_chain = torch.tensor(
            [[a == b for b in chain] for a in chain], dtype=torch.bool
        )
        forward = (res_id[None, :] - res_id[:, None]) == 1
        covalent = same_chain & forward
        self_pair = torch.eye(len(keep), dtype=torch.bool)

        bonded.append(distance[covalent])
        rest = distance[~covalent & ~self_pair]
        other_close.append(rest[rest < 6.0])
        if rest.numel():
            smallest = float(rest.min())
            if smallest < other_min:
                other_min, worst = smallest, (smallest, path.stem)

        found = distance < args.threshold
        found = found & ~self_pair
        for value in found.sum(dim=1).tolist():
            out_degree[value] = out_degree.get(value, 0) + 1
        for value in found.sum(dim=0).tolist():
            in_degree[value] = in_degree.get(value, 0) + 1
        # A found edge that is not a real peptide bond, or runs backwards.
        wrong_way += int((found & ~covalent).sum())
        total_bonds += int(covalent.sum())

    if not bonded:
        print("nothing measured")
        return 1

    bond_d = torch.cat(bonded)
    rest_d = torch.cat(other_close)
    print(f"{used} structures\n")
    def q(values, p):
        return float(torch.quantile(values, torch.tensor(p, dtype=values.dtype)))

    print(
        f"  bonded C(i)-N(i+1)   n={bond_d.numel():>8d}  "
        f"min {float(bond_d.min()):.3f}  median {float(bond_d.median()):.3f}  "
        f"max {float(bond_d.max()):.3f}"
    )
    # The min and max are broken models, as at g = 1 in scripts/reach_probe.py.
    # The percentiles are the real spread of a peptide bond.
    print(
        f"    percentiles        0.01% {q(bond_d, 0.0001):.3f}  "
        f"0.1% {q(bond_d, 0.001):.3f}  1% {q(bond_d, 0.01):.3f}  "
        f"99% {q(bond_d, 0.99):.3f}  99.9% {q(bond_d, 0.999):.3f}  "
        f"99.99% {q(bond_d, 0.9999):.3f}"
    )
    print(
        f"  every other C-N      n={rest_d.numel():>8d}  "
        f"min {other_min:.3f}  "
        f"1st pct "
        f"{float(torch.quantile(rest_d, torch.tensor(0.01, dtype=rest_d.dtype))):.3f}"
    )
    if worst:
        print(f"  closest non-bonded C-N: {worst[0]:.3f} A in {worst[1]}")
    print(f"\n  gap between the populations: {other_min - float(bond_d.max()):+.3f} A")
    print(
        f"  non-bonded C-N closer than the longest real bond: "
        f"{int((rest_d < float(bond_d.max())).sum())}"
    )

    print(f"\n  directed C-to-N graph at {args.threshold:g} A:")
    for name, counts in (("out-degree", out_degree), ("in-degree", in_degree)):
        total = sum(counts.values())
        detail = "  ".join(
            f"{d}: {counts[d] / total:.4%}" for d in sorted(counts) if d <= 3
        )
        print(f"    {name:<11} {detail}")
    print(
        f"\n  spurious edges found: {wrong_way} against {total_bonds} real bonds "
        f"({wrong_way / max(total_bonds, 1):.2e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
