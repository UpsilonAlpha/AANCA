"""Report the perception graph's shape, and how much it churns per step.

    python scripts/graph_report.py

Two numbers matter for the per-step rebuild decision.

**k_binding_fraction** -- the fraction of residues where the k limit, not the
distance cutoff, excluded a neighbour. The cosine envelope makes an edge leaving
through the *cutoff* smooth, but an edge dropped because k was full still pops.
Choose the default k so this is zero, or the rebuild reintroduces the gradient
discontinuity the envelope exists to remove.

**Edge churn** -- the fraction of the neighbour list that changes when every
side chain is randomised. If this were near zero the per-step rebuild would be
an expensive no-op and the graph could be built once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aanca.damage import randomise_rotamers, torsion_noise
from aanca.graph import build_graph
from aanca.structure import load_protein

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
ENTRIES = ("193L", "3NIR", "1UBQ")
CUTOFF = 10.0


def churn(before, after) -> float:
    """Fraction of real edges whose neighbour set changed."""
    total = changed = 0
    for i in range(before.n_residues):
        a = set(before.neighbours[i][before.edge_mask[i]].tolist())
        b = set(after.neighbours[i][after.edge_mask[i]].tolist())
        total += max(len(a), 1)
        changed += len(a ^ b)
    return changed / max(total, 1)


def main() -> int:
    paths = [(e, DATA / f"{e}.pdb") for e in ENTRIES]
    paths = [(e, p) for e, p in paths if p.exists()]
    if not paths:
        print("no structures; run scripts/fetch_test_structures.py")
        return 1

    print(f"Graph shape at cutoff {CUTOFF:.0f} A -- choosing k")
    print(f"{'entry':<8}{'k':>5}{'mean deg':>10}{'max deg':>9}{'k binds':>10}")
    for entry, path in paths:
        protein = load_protein(path)
        for k in (16, 24, 32, 48, 64):
            g = build_graph(protein, k=k, cutoff=CUTOFF)
            degrees = g.edge_mask.sum(axis=1)
            print(
                f"{entry:<8}{k:>5}{g.stats.mean_degree:>10.1f}"
                f"{degrees.max():>9}{g.stats.k_binding_fraction:>9.0%}"
            )
        print()

    print("Edge churn per step (how much the graph actually moves)")
    print(f"{'entry':<8}{'mild noise':>13}{'randomised':>13}")
    for entry, path in paths:
        protein = load_protein(path)
        base = build_graph(protein, k=48, cutoff=CUTOFF)
        mild = torsion_noise(
            protein, concentration=20.0, rng=np.random.default_rng(0)
        ).apply(protein)
        wild = randomise_rotamers(protein, rng=np.random.default_rng(0)).apply(protein)
        c_mild = churn(base, build_graph(mild, k=48, cutoff=CUTOFF))
        c_wild = churn(base, build_graph(wild, k=48, cutoff=CUTOFF))
        print(f"{entry:<8}{c_mild:>12.1%}{c_wild:>13.1%}")

    print("\nFeature layout")
    protein = load_protein(paths[0][1])
    g = build_graph(protein, k=48, cutoff=CUTOFF)
    for name, sl in sorted(g.feature_slices.items(), key=lambda kv: kv[1].start):
        print(f"  {name:<20}{sl.stop - sl.start:>5} dims  [{sl.start}:{sl.stop}]")
    print(f"  {'TOTAL edge':<20}{g.edge_features.shape[-1]:>5} dims")
    print(f"  {'node':<20}{g.node_features.shape[-1]:>5} dims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
