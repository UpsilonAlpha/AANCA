"""How much peptide-bond error does a given placement error cause?

    python scripts/bond_sensitivity_probe.py [--structures 80]

Closes a gap in an argument built over several sections. The hole head's
accuracy is measured as a **CA** error -- 0.95 A on bonded positions -- and the
threading tolerance is a **C-N** distance, about 0.6 A before a peptide bond
stops being unambiguous (`scripts/peptide_shell_probe.py`). Those were being
compared directly, and they are not the same quantity: the bond runs from the
parent's C to the child's **N**, and where the child's N sits depends on its
*orientation* as well as its CA. A residue whose CA is 0.9 A out but whose
frame is rotated can have a much worse bond than 0.9 A; one that is translated
along the bond axis can have a better one.

So this measures the transfer function. Take a real residue, perturb its frame
by a controlled amount -- a translation of known magnitude in a random
direction, or a rotation of known angle about a random axis, or both -- and
measure what happens to the C(i-1)-N(i) distance that holds it to its parent.
No model and no training: it is a property of the geometry, so it bounds what
any amount of either could achieve.

The question it answers: is the ~0.6 A threading tolerance a tolerance on
*placement*, or does placement have to be several times tighter than that?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from aanca.backbone import backbone_atoms, rotation_from_vector
from aanca.growth import growth_links
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "pdb"

#: The ideal peptide bond, and the threshold beyond which the directed C-to-N
#: graph stops reading the chain unambiguously (``peptide_shell_probe``: real
#: bonds are 99.98 % within [1.262, 1.392] and the nearest non-bonded C-N
#: anywhere is 2.509, so 2.0 is the usable cut).
IDEAL_C_N = 1.329
THRESHOLD = 2.0


def perturb(
    coords: torch.Tensor,
    topo,
    index: torch.Tensor,
    shift: float,
    angle: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Move residues ``index``: rotate by ``angle`` about their own CA, then translate.

    Rotation about CA and not the centroid, so the two perturbations are
    independent: the rotation leaves the CA where it was and the translation is
    the only thing that moves it. That matters for reading the table -- the
    ``shift`` column is exactly the CA placement error, which is what the hole
    head's number is.
    """
    out = coords.clone()
    n = index.numel()
    axis = torch.randn(n, 3, generator=generator, dtype=coords.dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    rotation = rotation_from_vector(axis * angle)

    direction = torch.randn(n, 3, generator=generator, dtype=coords.dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    ca = backbone_atoms(coords, topo)[index, 1]
    block = out[index] - ca.unsqueeze(1)
    block = block @ rotation.transpose(-1, -2)
    block = block + ca.unsqueeze(1) + (direction * shift).unsqueeze(1)
    return out.index_copy(0, index, block)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=80)
    parser.add_argument("--min-residues", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(PDB.glob("*.bcif"))[: args.structures]
    cases = []
    for path in paths:
        try:
            protein = load_protein(path)
        except Exception:
            continue
        if len(protein) < args.min_residues:
            continue
        try:
            topo, coords = feature_topology(protein)
        except Exception:
            continue
        links = growth_links(topo)
        joined = torch.nonzero(links, as_tuple=False).flatten()
        if joined.numel() < 10:
            continue
        cases.append((topo, coords, joined))
        if len(cases) >= 20:
            break
    if not cases:
        print(f"no usable structures in {PDB}")
        return 1

    print(
        f"{len(cases)} structures. A residue's frame is perturbed and the "
        f"C(i-1)-N(i) bond\nholding it to its parent is remeasured. Ideal is "
        f"{IDEAL_C_N} A; the threading cut is {THRESHOLD} A.\n"
    )
    print(
        f"  {'CA shift':>9}{'rotation':>10}{'mean C-N':>11}{'median':>9}"
        f"{'90th':>8}{'past cut':>10}"
    )

    settings = [
        (0.0, 0.0), (0.3, 0.0), (0.6, 0.0), (0.9, 0.0), (1.5, 0.0),
        (0.0, 0.05), (0.0, 0.10), (0.0, 0.20),
        (0.9, 0.10), (0.9, 0.20),
    ]
    for shift, angle in settings:
        bonds = []
        for repeat in range(3):
            generator = torch.Generator().manual_seed(args.seed + repeat)
            for topo, coords, joined in cases:
                child = joined + 1
                moved = perturb(coords, topo, child, shift, angle, generator)
                bb_parent = backbone_atoms(coords, topo)
                bb_child = backbone_atoms(moved, topo)
                # parent's C is untouched; the child's N has moved with it
                bonds.append(
                    (bb_child[child, 0] - bb_parent[joined, 2]).norm(dim=-1)
                )
        d = torch.cat(bonds)
        q = lambda p: float(torch.quantile(d, torch.tensor(p, dtype=d.dtype)))
        past = float((d > THRESHOLD).to(torch.float64).mean())
        print(
            f"  {shift:>8.1f} {math.degrees(angle):>8.0f}d"
            f"{float(d.mean()):>10.2f} A{q(0.5):>8.2f}{q(0.90):>8.2f}"
            f"{past:>9.1%}"
        )

    print(
        "\n  'past cut' is the fraction whose bond exceeds the 2.0 A threshold,\n"
        "  i.e. would no longer be read as a peptide bond at all. Read the\n"
        "  rotation rows carefully: a rotation leaves the CA exactly where it\n"
        "  was, so any bond error there is placement error the CA metric\n"
        "  cannot see."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
