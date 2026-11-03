"""Is a chain terminus locally distinguishable, and is valence a local property?

    python scripts/valence_probe.py [--structures 300]

Two preconditions for growth driven purely by "there should be another residue
here", with no sequence indices anywhere.

**Valence.** If every residue carries at most one outgoing and one incoming
peptide bond, the bond graph is automatically a disjoint union of paths, which
is what a set of chains is. ``scripts/peptide_shell_probe.py`` already measured
that native structures satisfy this exactly. The question here is whether it is
a *local* property -- whether a residue can tell its own valence from what it
can see -- because if so then "the chain is a path" is enforceable by a local
rule and is not a global constraint at all.

**Termination.** A residue whose valence is unsatisfied has a dangling peptide
bond, which is locally visible, so "there should be another residue here" is
just "my valence is unsatisfied" and growth stops when none are left. Except
that a real chain has exactly two dangling bonds -- its termini -- so the rule
has to learn that a dangling bond is *acceptable* in a terminus-like
environment. That is only learnable if termini differ locally from interior
residues, which is what this measures: neighbour count and burial for termini
against everything else.
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
    parser.add_argument("--cutoff", type=float, default=10.0)
    args = parser.parse_args()

    paths = sorted(PDB.glob("*.bcif"))[: args.structures]
    if not paths:
        print(f"no structures in {PDB}")
        return 1

    terminal_neighbours: list[torch.Tensor] = []
    interior_neighbours: list[torch.Tensor] = []
    covalent_share: list[torch.Tensor] = []
    used = 0

    for path in paths:
        try:
            protein = load_protein(path)
        except Exception:
            continue
        if len(protein) < args.min_residues:
            continue
        keep = [i for i, r in enumerate(protein.residues) if "CA" in r.index]
        if len(keep) < 10:
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
        self_pair = torch.eye(len(keep), dtype=torch.bool)
        near = (distance < args.cutoff) & ~self_pair
        count = near.sum(dim=1).to(torch.float64)

        same_chain = torch.tensor(
            [[a == b for b in chain] for a in chain], dtype=torch.bool
        )
        adjacent = (res_id[:, None] - res_id[None, :]).abs() == 1
        covalent = same_chain & adjacent
        # How much of a residue's neighbourhood is covalent? This is the number
        # that says a created residue is almost certainly *not* a chain
        # neighbour of its creator.
        covalent_share.append(
            covalent.sum(dim=1).to(torch.float64) / count.clamp_min(1.0)
        )

        # A terminus is a residue missing a covalent partner on one side.
        has_prev = (same_chain & ((res_id[None, :] - res_id[:, None]) == -1)).any(dim=1)
        has_next = (same_chain & ((res_id[None, :] - res_id[:, None]) == 1)).any(dim=1)
        terminal = ~has_prev | ~has_next
        terminal_neighbours.append(count[terminal])
        interior_neighbours.append(count[~terminal])

    if not used:
        print("nothing measured")
        return 1

    term = torch.cat(terminal_neighbours)
    inter = torch.cat(interior_neighbours)
    share = torch.cat(covalent_share)

    def q(values, p):
        return float(torch.quantile(values, torch.tensor(p, dtype=values.dtype)))

    print(f"{used} structures, neighbours within {args.cutoff:g} A of CA\n")
    print(
        f"  interior residues  n={inter.numel():>8d}  "
        f"mean {float(inter.mean()):>6.2f}  median {q(inter, 0.5):>6.1f}  "
        f"10th {q(inter, 0.10):>5.1f}  90th {q(inter, 0.90):>5.1f}"
    )
    print(
        f"  chain termini      n={term.numel():>8d}  "
        f"mean {float(term.mean()):>6.2f}  median {q(term, 0.5):>6.1f}  "
        f"10th {q(term, 0.10):>5.1f}  90th {q(term, 0.90):>5.1f}"
    )
    gap = float(inter.mean() - term.mean())
    pooled = float(
        ((inter.var() * inter.numel() + term.var() * term.numel())
         / (inter.numel() + term.numel())).sqrt()
    )
    print(
        f"\n  termini are {gap:.2f} fewer neighbours on average, "
        f"{gap / pooled:.2f} pooled standard deviations"
    )
    print(
        f"  a terminus is below the interior 10th percentile "
        f"{float((term < q(inter, 0.10)).to(torch.float64).mean()):.1%} of the time"
    )
    print(
        f"\n  covalent share of a neighbourhood: mean "
        f"{float(share.mean()):.3f}, median {q(share, 0.5):.3f} -- so "
        f"{1 - float(share.mean()):.1%} of what a residue sees is NOT a chain "
        f"neighbour of its own"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
