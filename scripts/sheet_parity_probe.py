"""Can an edge tell a parallel beta sheet from an antiparallel one?

    python scripts/sheet_parity_probe.py --structures 120

**The sharpest test of whether local messages carry enough geometry.** Two
paired strands are, in a distance-only representation, *identical*: the CA-CA
distances between paired residues are the same either way. What differs is the
direction the two chains run. So if the edge featurisation cannot separate them,
no amount of training fixes it -- the information is absent -- and if it can,
then the deficit is somewhere else and "give the cells more local geometry" is
the wrong next move.

``edge_features`` carries a ``relative_rotation`` block, ``R_i^T R_j`` flattened
to nine numbers, which *should* settle it: in a parallel pair the two backbone
frames are related by a near-identity rotation about the strand axis, in an
antiparallel pair by a near-180-degree one. This measures whether that
separation is actually there in real structures, in the quantity the model
receives, rather than in principle.

Operational definition, geometric and DSSP-free so it needs no extra tool:
a pair (i, j) is strand-paired when they are more than four residues apart in
sequence, their CAs are within 5.5 A, and both sit in an extended local
conformation. The parity is the sign of the dot product of the two chain
direction vectors ``CA(i+1) - CA(i-1)`` and ``CA(j+1) - CA(j-1)``: positive is
parallel, negative antiparallel.

Reported as the distribution of the relative rotation's **angle**, which is the
single scalar the nine numbers determine (``theta = arccos((tr(R) - 1) / 2)``).
If the two parities occupy different parts of that axis, an edge can tell them
apart from this block alone.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from aanca.structure import load_protein
from aanca.torch_graph import backbone_frames, feature_topology, frame_atom_coords

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "pdb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=120)
    parser.add_argument("--min-residues", type=int, default=40)
    parser.add_argument("--pair-cutoff", type=float, default=5.5)
    parser.add_argument("--min-separation", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(
        p for p in PDB.iterdir() if p.suffix in (".bcif", ".cif", ".pdb")
    )[: args.structures]

    parallel: list[float] = []
    anti: list[float] = []
    used = 0

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
        rot, _origin = backbone_frames(coords, topo)
        ca = frame_atom_coords(coords, topo)[:, 1, :]
        n = topo.n_residues
        if n < 8:
            continue
        used += 1

        # Chain direction at each interior residue, from its own neighbours.
        tangent = ca[2:] - ca[:-2]
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        res_ids = topo.res_ids
        distance = torch.cdist(ca, ca)
        for i in range(1, n - 1):
            if not bool(topo.has_frame[i]):
                continue
            close = torch.nonzero(distance[i] < args.pair_cutoff).flatten().tolist()
            for j in close:
                if j <= i or j < 1 or j >= n - 1:
                    continue
                if not bool(topo.has_frame[j]):
                    continue
                if abs(int(res_ids[j]) - int(res_ids[i])) <= args.min_separation:
                    continue
                # Extended: the local tangent is nearly straight, which is what
                # a strand is and what excludes helical contacts.
                straight_i = float((ca[i + 1] - ca[i - 1]).norm())
                straight_j = float((ca[j + 1] - ca[j - 1]).norm())
                if straight_i < 6.0 or straight_j < 6.0:
                    continue
                dot = float(tangent[i - 1] @ tangent[j - 1])
                if abs(dot) < 0.5:
                    continue  # neither clearly parallel nor clearly anti
                relative = rot[i].transpose(-1, -2) @ rot[j]
                trace = float(relative.diagonal().sum())
                theta = math.degrees(
                    math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
                )
                (parallel if dot > 0 else anti).append(theta)

    if not parallel or not anti:
        print(f"not enough pairs: {len(parallel)} parallel, {len(anti)} anti")
        return 1

    def stats(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
        return m, sd

    pm, ps = stats(parallel)
    am, asd = stats(anti)
    print(
        f"{used} structures, {len(parallel)} parallel and {len(anti)} "
        f"antiparallel strand pairs\n"
    )
    print(f"  relative rotation angle, theta = arccos((tr(R_i^T R_j) - 1) / 2):")
    print(f"    {'parity':<16}{'mean':>9}{'sd':>8}{'n':>7}")
    print(f"    {'parallel':<16}{pm:>8.1f} deg{ps:>7.1f}{len(parallel):>7}")
    print(f"    {'antiparallel':<16}{am:>8.1f} deg{asd:>7.1f}{len(anti):>7}")
    pooled = ((ps ** 2 + asd ** 2) / 2) ** 0.5
    print(f"\n  separation {abs(pm - am):.1f} deg, pooled sd {pooled:.1f}, "
          f"Cohen's d {abs(pm - am) / max(pooled, 1e-9):.2f}")
    # Where a single threshold would put them, which is the operational question:
    # can one number off this block classify the parity at all?
    cut = (pm + am) / 2
    right = sum(1 for x in parallel if (x < cut) == (pm < am)) + sum(
        1 for x in anti if (x < cut) == (am < pm)
    )
    print(
        f"  best single-threshold accuracy at {cut:.0f} deg: "
        f"{right / (len(parallel) + len(anti)):.1%}"
    )
    print(
        "\n  A large separation means the relative-rotation block already\n"
        "  distinguishes sheet parity and the local representation is not what\n"
        "  is missing -- so the deficit is global position, not local geometry.\n"
        "  Near-complete overlap would mean the opposite: the edge cannot tell\n"
        "  two paired strands apart and no training would fix it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
