"""Is the 3.8 A CA-CA shell exclusively covalent?

    python scripts/ca_shell_probe.py [--structures 400]

The precondition for growing a protein as a *point cloud* rather than as a
chain of indexed slots. If a residue is created by local density matching --
"my neighbourhood is sparse over there, there should be a residue there" -- it
arrives with a position and no sequence index, so the chain has to be
recoverable from the geometry afterwards.

It is recoverable if and only if the covalent neighbours are the *only* CAs at
peptide-bond range. In a real protein a bonded pair sits at 3.80 A, while
non-bonded CAs are kept apart by van der Waals contact between the residues
they belong to -- adjacent beta strands are ~4.8-5.5 A, helix i to i+3 is
~5-6 A. If the two populations are cleanly separated then thresholding the
CA-CA graph at the gap reads the chain straight off the cloud, and sequence
assignment stops being a problem the generator has to solve. If they overlap,
every overlapping pair is an ambiguity that no local rule can resolve.

Reports the bonded maximum, the non-bonded minimum, the size of the gap, and
how many CAs sit within the threshold of each residue -- which should be one at
a chain terminus and two everywhere else, or the chain is not a path.
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
    parser.add_argument("--structures", type=int, default=400)
    parser.add_argument("--min-residues", type=int, default=40)
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.2,
        help="candidate cut between the bonded and non-bonded populations",
    )
    args = parser.parse_args()

    paths = sorted(PDB.glob("*.bcif"))[: args.structures]
    if not paths:
        print(f"no structures in {PDB}")
        return 1

    bonded: list[torch.Tensor] = []
    nonbonded_close: list[torch.Tensor] = []
    nonbonded_min = float("inf")
    degree_counts: dict[int, int] = {}
    used = 0
    worst: tuple[float, str, int] | None = None

    for path in paths:
        try:
            protein = load_protein(path)
        except Exception:
            continue
        if len(protein) < args.min_residues:
            continue

        keep = [
            i for i, r in enumerate(protein.residues) if "CA" in r.index
        ]
        if len(keep) < 3:
            continue
        used += 1
        ca = torch.tensor(
            np.array(
                [
                    protein.residues[i].coords[protein.residues[i].index["CA"]]
                    for i in keep
                ]
            ),
            dtype=torch.float64,
        )
        chain = [protein.chain_ids[i] for i in keep]
        res_id = torch.tensor([protein.res_ids[i] for i in keep])

        distance = torch.cdist(ca, ca)
        same_chain = torch.tensor(
            [[a == b for b in chain] for a in chain], dtype=torch.bool
        )
        adjacent = (res_id[:, None] - res_id[None, :]).abs() == 1
        covalent = same_chain & adjacent
        self_pair = torch.eye(len(keep), dtype=torch.bool)

        bonded.append(distance[covalent])
        other = distance[~covalent & ~self_pair]
        nonbonded_close.append(other[other < 8.0])
        if other.numel():
            smallest = float(other.min())
            if smallest < nonbonded_min:
                nonbonded_min = smallest
                worst = (smallest, path.stem, len(keep))

        # How many CAs sit within the threshold of each residue?
        within = ((distance < args.threshold) & ~self_pair).sum(dim=1)
        for value in within.tolist():
            degree_counts[value] = degree_counts.get(value, 0) + 1

    if not bonded:
        print("nothing measured")
        return 1

    covalent_d = torch.cat(bonded)
    other_d = torch.cat(nonbonded_close)
    print(f"{used} structures\n")
    print(f"  bonded CA-CA      n={covalent_d.numel():>8d}  "
          f"min {float(covalent_d.min()):.2f}  median {float(covalent_d.median()):.2f}  "
          f"max {float(covalent_d.max()):.2f}")
    print(f"  non-bonded CA-CA  n={other_d.numel():>8d}  "
          f"min {nonbonded_min:.2f}  "
          f"1st pct {float(torch.quantile(other_d, torch.tensor(0.01, dtype=other_d.dtype))):.2f}")
    if worst:
        print(f"  closest non-bonded pair: {worst[0]:.2f} A in {worst[1]} "
              f"({worst[2]} residues)")

    overlap = int((other_d < float(covalent_d.max())).sum())
    print(
        f"\n  non-bonded pairs closer than the longest bonded one: {overlap} "
        f"({overlap / other_d.numel():.2e} of them)"
    )
    print(f"  gap between the populations: "
          f"{nonbonded_min - float(covalent_d.max()):+.2f} A")

    print(f"\n  CAs within {args.threshold:g} A of a residue (should be 1 or 2):")
    total = sum(degree_counts.values())
    for degree in sorted(degree_counts):
        share = degree_counts[degree] / total
        print(f"    {degree:>3d} neighbours: {degree_counts[degree]:>8d}  {share:>7.3%}")
    clean = sum(degree_counts.get(d, 0) for d in (1, 2)) / total
    print(
        f"\n  {clean:.3%} of residues have exactly one or two CAs within "
        f"{args.threshold:g} A, which is what makes the chain a readable path."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
