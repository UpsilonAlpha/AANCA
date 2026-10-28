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

#: Three-letter to one-letter residue codes, for sequence-identity queries.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

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


BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


def downstream_atoms(resname: str, b: str, c: str) -> frozenset[str]:
    """Side-chain atoms that move when the ``b-c`` bond is driven.

    Walk the residue's bond graph outward from ``c``, forbidden to cross back
    through ``b``. This is the moving set for *any* internal coordinate hinged on
    that bond -- a chi rotation about it, a stretch along it, or a bend of the
    angle that ends at it.

    Reaching a backbone atom means the walk came round a ring, since no
    legitimate side-chain moving set contains backbone. PRO is the only standard
    residue where this happens.
    """
    adj = _adjacency(resname)
    if c not in adj or b not in adj.get(c, ()):
        raise KeyError(f"{resname}: {b}-{c} is not a bond")

    # Walk outward from c with b deleted from the graph.
    seen = {b, c}
    moving: set[str] = set()
    queue = deque([c])
    while queue:
        for nxt in adj[queue.popleft()]:
            if nxt in seen:
                continue
            if nxt in BACKBONE_ATOMS:
                raise CyclicTorsion(
                    f"{resname}: the {b}-{c} bond lies in a ring "
                    f"(the walk reached backbone atom {nxt})"
                )
            seen.add(nxt)
            moving.add(nxt)
            queue.append(nxt)

    # If any *other* neighbour of b is reachable without passing through b, then
    # b sits on a cycle and driving this bond would break the ring closure.
    # Checking only for backbone atoms is not enough: PRO's ring contains N and
    # is caught that way, but an aromatic ring contains no backbone at all, so
    # PHE's CG-CD1 would otherwise look stretchable.
    looped = (adj[b] - {c}) & moving
    if looped:
        raise CyclicTorsion(
            f"{resname}: the {b}-{c} bond lies in a ring "
            f"(reached {sorted(looped)[0]}, also bonded to {b})"
        )
    return frozenset(moving)


def _downstream(resname: str, chi_index: int) -> frozenset[str]:
    """Atoms that move when chi_(chi_index+1) of ``resname`` is rotated."""
    _a, b, c, _d = CHI_ATOMS[resname][chi_index]
    return downstream_atoms(resname, b, c)


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


def _oriented_sidechain_bonds(resname: str) -> list[tuple[str, str]]:
    """Side-chain bonds as ``(near, far)``, ordered by bond count from CA."""
    depth = sidechain_depth(resname)
    depth["CA"] = 0
    out: list[tuple[str, str]] = []
    for u, v in SIDECHAIN_BONDS[resname]:
        if u in BACKBONE_ATOMS and v in BACKBONE_ATOMS:
            continue
        du, dv = depth.get(u), depth.get(v)
        if du is None or dv is None or du == dv:
            # A ring-closing bond joins two atoms at equal depth, or an atom the
            # side-chain walk never reaches (PRO's CD-N). Neither is drivable.
            continue
        out.append((u, v) if du < dv else (v, u))
    return out


def movable_bonds(resname: str) -> tuple[tuple[str, str, frozenset[str]], ...]:
    """Stretchable side-chain bonds as ``(near, far, moving_atoms)``.

    A bond inside a ring is excluded: its length cannot change without the rest
    of the ring absorbing it, which a rigid translation of one side cannot do.
    """
    out = []
    for near, far in _oriented_sidechain_bonds(resname):
        try:
            out.append((near, far, downstream_atoms(resname, near, far) | {far}))
        except CyclicTorsion:
            continue
    return tuple(out)


def movable_angles(resname: str) -> tuple[tuple[str, str, str, frozenset[str]], ...]:
    """Bendable side-chain angles as ``(a, b, c, moving_atoms)``.

    The angle ``a-b-c`` is changed by rotating ``c`` and everything beyond it
    about the axis through ``b`` perpendicular to the ``a-b-c`` plane. ``a`` may
    be a backbone atom (CA for the N-CA-CB style angles at the attachment
    point), but ``b-c`` must not be in a ring.
    """
    adj = _adjacency(resname)
    depth = sidechain_depth(resname)
    depth["CA"] = 0

    out = []
    for near, far in _oriented_sidechain_bonds(resname):
        try:
            moving = downstream_atoms(resname, near, far) | {far}
        except CyclicTorsion:
            continue
        # The apex is `near`; `a` is any atom bonded to it that is closer to CA.
        for other in sorted(adj[near]):
            if other == far or depth.get(other) is None:
                continue
            if depth[other] < depth[near] or (
                depth[other] == depth[near] and other < far
            ):
                out.append((other, near, far, moving))
                break
    return tuple(out)


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


def sidechain_depth(resname: str) -> dict[str, int]:
    """Bond count from CA to each side-chain atom, CB being depth 1.

    Topological, so it does not change when the side chain moves. PRO's CD-N
    bond is not followed, since N is backbone.
    """
    adj: dict[str, set[str]] = {}
    for u, v in SIDECHAIN_BONDS[resname]:
        if u in ("N", "C", "O") or v in ("N", "C", "O"):
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    depth = {"CA": 0}
    frontier = ["CA"]
    while frontier:
        nxt: list[str] = []
        for cur in frontier:
            for other in adj.get(cur, ()):
                if other not in depth:
                    depth[other] = depth[cur] + 1
                    nxt.append(other)
        frontier = nxt
    depth.pop("CA")
    return depth


def _tip_atoms(resname: str) -> frozenset[str]:
    depth = sidechain_depth(resname)
    if not depth:
        return frozenset()
    deepest = max(depth.values())
    return frozenset(n for n, d in depth.items() if d == deepest)


#: Atom relabellings that produce a chemically identical structure.
#:
#: These exist because the terminal atoms of a symmetric group are
#: indistinguishable: a carboxylate's two oxygens, an aromatic ring's two
#: halves, a guanidinium's two nitrogens. Comparing such a residue to a
#: reference **by atom name** scores a large RMSD for two structures that are
#: physically the same, which is why side-chain RMSD must be minimised over
#: these permutations. The same chemistry is what makes the corresponding chis
#: two-fold periodic (``CHI_PI_PERIODIC``).
#:
#: The aromatic swap moves both pairs at once, because it is a single 180-degree
#: ring flip -- swapping CD1/CD2 without CE1/CE2 is not a symmetry of anything.
#:
#: LEU's CD1/CD2 and VAL's CG1/CG2 are deliberately absent. Those methyls are
#: chemically equivalent but IUPAC assigns them by geometry, and swapping them
#: changes the defining chi by ~120 degrees rather than 180, so it is not a
#: periodicity. AlphaFold's table excludes them and the packing baselines follow
#: it; matching that is what keeps the tier-1 comparison auditable.
SYMMETRIC_RELABELLINGS: dict[str, tuple[dict[str, str], ...]] = {
    "ASP": ({"OD1": "OD2", "OD2": "OD1"},),
    "GLU": ({"OE1": "OE2", "OE2": "OE1"},),
    "ARG": ({"NH1": "NH2", "NH2": "NH1"},),
    "PHE": (
        {"CD1": "CD2", "CD2": "CD1", "CE1": "CE2", "CE2": "CE1"},
    ),
    "TYR": (
        {"CD1": "CD2", "CD2": "CD1", "CE1": "CE2", "CE2": "CE1"},
    ),
}


def symmetric_relabellings(resname: str) -> tuple[dict[str, str], ...]:
    """Relabellings giving a chemically identical structure, identity first."""
    return ({},) + SYMMETRIC_RELABELLINGS.get(resname, ())


#: The most distal side-chain atoms, by bond count from CA. Where several tie
#: they are averaged, which is what you want for the chemically equivalent pairs
#: (ARG's NH1/NH2, LEU's CD1/CD2, VAL's CG1/CG2).
#:
#: Chosen topologically rather than by distance from CA on purpose: the furthest
#: atom *by distance* can change identity as the side chain rotates, which would
#: put a discontinuity into the edge features at the moment the argmax flips.
SIDECHAIN_TIP_ATOMS: dict[str, frozenset[str]] = {
    res: _tip_atoms(res) for res in AA3
}

#: Internal coordinates beyond the chis that the rule *may* drive, if enabled.
#: Frozen in v1 -- see ``docs/PLAN_V1.md`` -- but the topology is defined here so
#: the degrees of freedom exist rather than having to be retro-fitted. A
#: classical forcefield has bond and angle terms, so a learned one needs the
#: scope for them. What stops them becoming cheap clash relief is the harmonic
#: restraint in ``losses.py``, not their absence.
#:
#: Computed after ``sidechain_depth`` because the bond orientation depends on it.
MOVABLE_BONDS: dict[str, tuple[tuple[str, str, frozenset[str]], ...]] = {
    res: movable_bonds(res) for res in AA3
}
MOVABLE_ANGLES: dict[str, tuple[tuple[str, str, str, frozenset[str]], ...]] = {
    res: movable_angles(res) for res in AA3
}

#: Padding widths for the batched tensors in ``torch_geometry``.
MAX_BONDS = max(len(v) for v in MOVABLE_BONDS.values())
MAX_ANGLES = max(len(v) for v in MOVABLE_ANGLES.values())
