"""Does the C-to-N graph survive the error the placement head actually makes?

    python scripts/threading_error_probe.py [--structures 200]

`scripts/peptide_shell_probe.py` established that threading works **on native
coordinates**: bonded C-N lengths lie in [1.262, 1.392] A for 99.98 % of pairs,
the nearest non-bonded C-N anywhere in 670k pairs is 2.509 A, and at a 2.0 A cut
there are zero false edges with in- and out-degree never above 1. So the
directed graph is exactly a set of chains and it fixes the N-to-C direction.

**That margin was measured on perfect geometry, and growth will not have
perfect geometry.** `scripts/hole_by_structure.py` measures the placement head
at **0.92 A** on covalently-bonded positions (three hops; 0.95 A at one), and
`scripts/train_hole.py`'s hops ablation showed depth does not reduce it. The
whole Cartesian-birth-then-thread design rests on a margin that has never been
tested against that error, which is the gap this closes.

**The quantity that matters is relative error, not absolute, and that is the
point of this probe.** A peptide bond is a constraint between *two* residues, so
what threading sees is the error in |C(i) - N(i+1)| -- a difference of two
placement errors, not one. If residues are placed independently at sigma, that
difference carries sigma*sqrt(2): at sigma = 0.92 A that is **1.3 A**, which is
larger than the entire 1.1 A gap between the bonded range's top and the nearest
non-bonded pair. Threading would collapse outright. If neighbouring residues err
*together*, the relative geometry is preserved and the absolute error is almost
irrelevant to threading.

So the sweep is over a **correlation length** as well as a magnitude. Each
residue is displaced by a smooth Gaussian field whose correlation length is L:
at L = 0 displacements are independent, and as L grows they approach a rigid
translation of the whole structure, which threading cannot see at all. This
converts a yes/no worry into the number actually worth knowing: **how much
neighbour-correlation does threading require at the error we have?** And that is
checkable against the head, because the common-mode result -- three hops halving
per-witness error while leaving the fused error alone -- is a statement that the
residual *is* correlated over a neighbourhood.

**This is the optimistic case, deliberately, and the result should be read as an
upper bound.** Each residue is moved as a rigid body, so its own N-CA-C geometry
stays exact and only the inter-residue contribution is perturbed. Real placement
error includes orientation error, which would move C and N within the residue as
well. A design that fails here fails for certain; one that survives here has
only cleared the easier half.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from aanca.structure import load_protein

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "pdb"

#: Magnitudes in Angstrom. 0.92 is the measured three-hop placement error and
#: 0.95 the one-hop; the smaller values say how much better it would have to be.
SIGMAS = (0.0, 0.2, 0.4, 0.6, 0.92, 1.3)
#: Correlation lengths in Angstrom. 0 is independent placement. 3.8 is one CA-CA
#: step, so it is the shortest length that couples covalent neighbours at all.
LENGTHS = (0.0, 3.8, 8.0, 16.0)


def correlated_field(
    centres: torch.Tensor, sigma: float, length: float, generator: torch.Generator
) -> torch.Tensor:
    """``(n, 3)`` displacements with per-residue RMS ``sigma`` and correlation ``length``.

    Built by smoothing an independent draw with a Gaussian kernel over the
    residues' own positions, then rescaling to the requested RMS. Smoothing is
    what creates the correlation, and the rescale is what stops it also
    shrinking the magnitude -- averaging ``k`` independent draws reduces the
    RMS by ``sqrt(k)``, so without the rescale a longer correlation length
    would silently also mean a smaller error and the sweep would confound the
    two things it is meant to separate.
    """
    raw = torch.randn(centres.shape[0], 3, generator=generator, dtype=torch.float64)
    if sigma == 0.0:
        return torch.zeros_like(raw)
    field = raw
    if length > 0.0:
        weight = torch.exp(
            -torch.cdist(centres, centres) ** 2 / (2.0 * length * length)
        )
        field = (weight @ raw) / weight.sum(dim=1, keepdim=True)
    # ``sigma`` is the RMS *displacement magnitude*, so both branches are
    # normalised the same way rather than one scaling per component and the
    # other per vector -- an inconsistency there would have made the L=0 column
    # sqrt(3) times noisier than the rest of its row and looked like a result.
    rms = float(field.pow(2).sum(dim=-1).mean().sqrt())
    if rms < 1e-12:
        return torch.zeros_like(raw)
    return field * (sigma / rms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=200)
    parser.add_argument("--min-residues", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(
        p
        for p in PDB.iterdir()
        if p.suffix in (".bcif", ".cif", ".pdb")
    )[: args.structures]
    if not paths:
        print(f"no coordinates in {PDB}")
        return 1

    # Load once; the sweep re-perturbs the same structures so every cell of the
    # table is measured on identical geometry and the only difference is noise.
    loaded = []
    for path in paths:
        try:
            protein = load_protein(path)
        except Exception:
            continue
        if len(protein) < args.min_residues:
            continue
        keep = [
            i for i, r in enumerate(protein.residues) if {"N", "C", "CA"} <= set(r.index)
        ]
        if len(keep) < 3:
            continue
        def atom(name):
            return torch.tensor(
                np.array(
                    [
                        protein.residues[i].coords[protein.residues[i].index[name]]
                        for i in keep
                    ]
                ),
                dtype=torch.float64,
            )
        chain = [protein.chain_ids[i] for i in keep]
        res_id = torch.tensor([protein.res_ids[i] for i in keep])
        same_chain = torch.tensor(
            [[a == b for b in chain] for a in chain], dtype=torch.bool
        )
        forward = (res_id[None, :] - res_id[:, None]) == 1
        loaded.append(
            {
                "n": atom("N"),
                "c": atom("C"),
                "ca": atom("CA"),
                "covalent": same_chain & forward,
                "self": torch.eye(len(keep), dtype=torch.bool),
            }
        )
    if not loaded:
        print("nothing loaded")
        return 1

    total_res = sum(e["ca"].shape[0] for e in loaded)
    print(
        f"{len(loaded)} structures, {total_res} residues, threshold "
        f"{args.threshold:g} A\n"
    )
    print(
        "  Each residue is translated rigidly by a correlated Gaussian field, so\n"
        "  its internal N-CA-C geometry stays exact: this is the optimistic case\n"
        "  and an upper bound on how well threading can do.\n"
    )
    header = (
        f"  {'sigma':>7}" + "".join(f"{f'L={l:g}':>22}" for l in LENGTHS)
    )
    print(header)
    print(f"  {'':>7}" + "".join(f"{'false/miss/deg':>22}" for _ in LENGTHS))

    for sigma in SIGMAS:
        cells = []
        for length in LENGTHS:
            generator = torch.Generator().manual_seed(args.seed)
            false_edges = missed = degree_bad = bonds = 0
            for entry in loaded:
                shift = correlated_field(entry["ca"], sigma, length, generator)
                carbon = entry["c"] + shift
                nitrogen = entry["n"] + shift
                distance = torch.cdist(carbon, nitrogen)
                found = (distance < args.threshold) & ~entry["self"]
                covalent = entry["covalent"]
                false_edges += int((found & ~covalent).sum())
                missed += int((covalent & ~found).sum())
                degree_bad += int((found.sum(dim=1) > 1).sum()) + int(
                    (found.sum(dim=0) > 1).sum()
                )
                bonds += int(covalent.sum())
            cells.append(
                f"{false_edges / max(bonds, 1):.1%}/"
                f"{missed / max(bonds, 1):.1%}/"
                f"{degree_bad / max(total_res, 1):.1%}"
            )
        print(f"  {sigma:>7.2f}" + "".join(f"{c:>22}" for c in cells))

    print(
        "\n  false = edges found that are not real peptide bonds, as a fraction of\n"
        "  the real ones; miss = real bonds the cut no longer finds; deg = residues\n"
        "  whose in- or out-degree exceeds 1, so the graph is no longer a set of\n"
        "  paths. All three must be near zero for threading to work unaided.\n"
        "\n  The row at sigma 0.92 is the measured placement error. Read along it:\n"
        "  the leftmost cell is independent placement and the rightmost is error\n"
        "  that moves neighbours together. If threading only survives at large L,\n"
        "  the design needs the placement residual to be correlated over that\n"
        "  distance, which is a property to verify in the head rather than hope\n"
        "  for."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
