"""Per-residue chemical constants: chi definitions, bond graphs, symmetries.

Everything a torsion operation needs to know about an amino acid lives here. The
only hand-written tables are ``CHI_ATOMS``, ``SIDECHAIN_BONDS`` and
``CHI_PI_PERIODIC``; the atom sets that a chi rotation moves are *derived* from
the bond graph rather than listed, so they cannot silently disagree with it.
"""

from __future__ import annotations

from collections import deque

#: The four atoms defining each chi, in order, for every residue type. Standard
#: IUPAC/AlphaFold convention. The axis of chi_i is the bond between atoms 1 and
#: 2 of its tuple.
CHI_ATOMS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "ALA": (),
    "ARG": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "NE"),
        ("CG", "CD", "NE", "CZ"),
    ),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ),
    "GLU": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ),
    "GLY": (),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ),
    "MET": (
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}

AA3 = tuple(sorted(CHI_ATOMS))

MAX_CHI = 4

#: Chis whose terminal group is two-fold symmetric: the 180-degree-flipped
#: torsion describes the *same* physical structure. Both the loss and any
#: reported MAE must quotient by this or the numbers are wrong and not obviously
#: so. ASN/GLN/HIS are deliberately absent -- their terminal substituents differ
#: (O vs N), so their chis are not periodic.
CHI_PI_PERIODIC: dict[str, tuple[bool, ...]] = {
    "ASP": (False, True),
    "GLU": (False, False, True),
    "PHE": (False, True),
    "TYR": (False, True),
}

#: Backbone heavy-atom bonds present in every residue.
_BACKBONE_BONDS: tuple[tuple[str, str], ...] = (
    ("N", "CA"),
    ("CA", "C"),
    ("C", "O"),
)

#: Side-chain heavy-atom bonds, including the CA-CB link. PRO's CD-N bond closes
#: the pyrrolidine ring and is what makes its chis non-rotatable; it is listed
#: here rather than omitted so that the ring is detected rather than assumed.
SIDECHAIN_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "ALA": (("CA", "CB"),),
    "ARG": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "NE"),
        ("NE", "CZ"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
    ),
    "ASN": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "ASP": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "CYS": (("CA", "CB"), ("CB", "SG")),
    "GLN": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "NE2"),
    ),
    "GLU": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "OE2"),
    ),
    "GLY": (),
    "HIS": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "ND1"),
        ("CG", "CD2"),
        ("ND1", "CE1"),
        ("CD2", "NE2"),
        ("CE1", "NE2"),
    ),
    "ILE": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
    "LEU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "MET": (("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "PHE": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    ),
    "PRO": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")),
    "SER": (("CA", "CB"), ("CB", "OG")),
    "THR": (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
    "TRP": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "NE1"),
        ("NE1", "CE2"),
        ("CD2", "CE2"),
        ("CD2", "CE3"),
        ("CE3", "CZ3"),
        ("CZ3", "CH2"),
        ("CH2", "CZ2"),
        ("CZ2", "CE2"),
    ),
    "TYR": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    ),
    "VAL": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
}


class CyclicTorsion(ValueError):
    """Raised when a chi's rotation axis lies inside a ring.

    Rotating about such a bond cannot be expressed as a rigid rotation of the
    downstream atoms, because the "downstream" set loops back to the axis. PRO
    is the only standard residue affected.
    """


def _adjacency(resname: str) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {}
    for u, v in _BACKBONE_BONDS + SIDECHAIN_BONDS[resname]:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _downstream(resname: str, chi_index: int) -> frozenset[str]:
    """Atoms that move when chi_(chi_index+1) of ``resname`` is rotated.

    Walk the residue's bond graph outward from atom ``c`` of the chi tuple,
    forbidden to cross back through ``b``. If the walk reaches ``a`` or ``b``
    anyway, the axis is in a ring.
    """
    a, b, c, _d = CHI_ATOMS[resname][chi_index]
    adj = _adjacency(resname)

    seen = {b, c}
    moving: set[str] = set()
    queue = deque([c])
    while queue:
        for nxt in adj[queue.popleft()]:
            if nxt in seen:
                continue
            if nxt in (a, b):
                raise CyclicTorsion(
                    f"{resname} chi{chi_index + 1}: axis {b}-{c} lies in a ring "
                    f"(reached {nxt})"
                )
            seen.add(nxt)
            moving.add(nxt)
            queue.append(nxt)
    return frozenset(moving)


#: ``CHI_DOWNSTREAM[resname][i]`` is the set of atom names rotated by chi_(i+1).
#: Derived from the bond graph at import time, so it cannot drift out of sync
#: with ``SIDECHAIN_BONDS``.
CHI_DOWNSTREAM: dict[str, tuple[frozenset[str], ...]] = {}

#: Residues whose side chain this module treats as rigid, with the reason. PRO
#: lands here by ring detection, not by being named.
RIGID_RESIDUES: dict[str, str] = {}

for _res in AA3:
    try:
        CHI_DOWNSTREAM[_res] = tuple(
            _downstream(_res, i) for i in range(len(CHI_ATOMS[_res]))
        )
    except CyclicTorsion as exc:
        CHI_DOWNSTREAM[_res] = ()
        RIGID_RESIDUES[_res] = str(exc)


def num_chi(resname: str) -> int:
    """Number of rotatable chis. Zero for ALA, GLY and the rigid residues."""
    if resname in RIGID_RESIDUES:
        return 0
    return len(CHI_ATOMS[resname])


def chi_periodic_mask(resname: str) -> tuple[bool, ...]:
    """Per-chi two-fold symmetry flags, padded to the residue's chi count."""
    flags = CHI_PI_PERIODIC.get(resname, ())
    n = len(CHI_ATOMS.get(resname, ()))
    return tuple(flags[i] if i < len(flags) else False for i in range(n))


def sidechain_atoms(resname: str) -> frozenset[str]:
    """Heavy side-chain atom names, i.e. everything beyond CA except C and O."""
    names = {n for bond in SIDECHAIN_BONDS[resname] for n in bond}
    return frozenset(names - {"N", "CA", "C", "O"})
