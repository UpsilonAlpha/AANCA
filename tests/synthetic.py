"""Deterministic synthetic residues, so the geometry tests need no downloads.

Atoms are placed by forward kinematics over the residue's own bond graph with
plausible-but-arbitrary internal coordinates. The torsion code never assumes
ideal geometry, so arbitrary-but-valid coordinates are a fair test; the real
structures in ``data/raw/`` are what check atom naming and disorder handling.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from aanca import residue_constants as rc
from aanca.torsions import Residue

_BOND = 1.52
_ANGLE = np.deg2rad(111.0)


def place_atom(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, bond: float, angle: float, torsion: float
) -> np.ndarray:
    """Position atom ``d`` from ``a-b-c`` plus internal coordinates (NeRF)."""
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.stack([bc, np.cross(n, bc), n], axis=-1)
    d2 = np.array(
        [
            -bond * np.cos(angle),
            bond * np.sin(angle) * np.cos(torsion),
            bond * np.sin(angle) * np.sin(torsion),
        ]
    )
    return c + m @ d2


def _residue_adjacency(resname: str) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    bonds = (("N", "CA"), ("CA", "C"), ("C", "O")) + rc.SIDECHAIN_BONDS[resname]
    for u, v in bonds:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def synthetic_residue(resname: str, seed: int = 0) -> Residue:
    """Build a residue with distinct, chemically plausible heavy-atom positions."""
    rng = np.random.default_rng(seed)
    adj = _residue_adjacency(resname)

    coords: dict[str, np.ndarray] = {
        "N": np.array([0.0, 0.0, 0.0]),
        "CA": np.array([1.458, 0.0, 0.0]),
        "C": np.array([2.009, 1.420, 0.0]),
    }
    parents: dict[str, tuple[str, str, str]] = {}

    # Breadth-first over the bond graph; each new atom hangs off three already
    # placed ones so that its torsion is well defined.
    queue: deque[str] = deque(["N", "CA", "C"])
    placed = {"N", "CA", "C"}
    chains = {"N": ("C", "CA"), "CA": ("C", "N"), "C": ("N", "CA")}

    while queue:
        cur = queue.popleft()
        gp, p = chains[cur]
        for nxt in sorted(adj.get(cur, ())):
            if nxt in placed:
                continue
            coords[nxt] = place_atom(
                coords[gp],
                coords[p],
                coords[cur],
                bond=_BOND,
                angle=_ANGLE,
                torsion=float(rng.uniform(-np.pi, np.pi)),
            )
            parents[nxt] = (gp, p, cur)
            chains[nxt] = (p, cur)
            placed.add(nxt)
            queue.append(nxt)

    names = tuple(sorted(coords))
    return Residue(
        resname=resname,
        atom_names=names,
        coords=np.stack([coords[n] for n in names]),
    )
