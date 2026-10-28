"""Loading structures into per-residue views, and writing them back out.

Parsing is biotite's job. This module's job is to produce the one representation
the rest of the project uses -- a list of :class:`~aanca.torsions.Residue` plus
the index mapping back into the source ``AtomArray`` -- and to keep that mapping
intact so a modified side chain can be written to a file and compared against
the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, filter_amino_acids
from biotite.structure.io import load_structure, save_structure

from . import residue_constants as rc
from .torsions import Residue, apply_chi_deltas, chi_mask, measure_chis


@dataclass
class Protein:
    """A parsed structure, residue-indexed.

    ``atoms`` is the filtered source array; ``atom_indices[i]`` gives the rows of
    ``atoms`` belonging to residue ``i``, in the same order as
    ``residues[i].atom_names``.

    ``residues`` holds float64 coordinates and is the authoritative state;
    ``atoms`` is a float32 mirror kept in sync for file output. Rollouts
    therefore accumulate no float32 rounding, but any comparison made on
    ``atoms.coord`` is only accurate to ~1e-6 A -- measure on ``residues``.
    """

    atoms: AtomArray
    residues: list[Residue]
    atom_indices: list[np.ndarray]
    chain_ids: list[str]
    res_ids: list[int]

    def __len__(self) -> int:
        return len(self.residues)

    # -- torsions -----------------------------------------------------------

    def chis(self) -> tuple[np.ndarray, np.ndarray]:
        """``(n_res, MAX_CHI)`` torsions in radians and the matching bool mask."""
        chis = np.full((len(self), rc.MAX_CHI), np.nan)
        mask = np.zeros((len(self), rc.MAX_CHI), dtype=bool)
        for i, res in enumerate(self.residues):
            chis[i], mask[i] = measure_chis(res)
        return chis, mask

    def chi_mask(self) -> np.ndarray:
        return np.stack([chi_mask(r) for r in self.residues])

    def apply_chi_deltas(self, deltas: np.ndarray) -> "Protein":
        """Return a new Protein with ``(n_res, MAX_CHI)`` deltas applied."""
        deltas = np.asarray(deltas, dtype=np.float64)
        if deltas.shape != (len(self), rc.MAX_CHI):
            raise ValueError(
                f"expected deltas of shape {(len(self), rc.MAX_CHI)}, "
                f"got {deltas.shape}"
            )
        new_residues = [
            apply_chi_deltas(res, deltas[i]) for i, res in enumerate(self.residues)
        ]
        atoms = self.atoms.copy()
        for idx, res in zip(self.atom_indices, new_residues):
            atoms.coord[idx] = res.coords
        return Protein(
            atoms=atoms,
            residues=new_residues,
            atom_indices=self.atom_indices,
            chain_ids=self.chain_ids,
            res_ids=self.res_ids,
        )

    def sequences(self) -> dict[str, str]:
        """One-letter sequences per chain, in residue order.

        Used for the sequence-identity leakage filter. Built from the residues
        actually present, so a gap in the model shortens the sequence rather
        than inserting a placeholder -- which is what we want for an identity
        search, since the alternative invents residues the structure lacks.
        """
        out: dict[str, list[str]] = {}
        for index, residue in enumerate(self.residues):
            code = rc.THREE_TO_ONE.get(residue.resname)
            if code is not None:
                out.setdefault(self.chain_ids[index], []).append(code)
        return {chain: "".join(codes) for chain, codes in out.items()}

    # -- rigid motion -------------------------------------------------------

    def transformed(
        self, rotation: np.ndarray | None = None, translation: np.ndarray | None = None
    ) -> "Protein":
        """Rigidly move the whole structure: ``x -> rotation @ x + translation``.

        This is the rotation augmentation the training recipe calls for. It is
        also the test instrument for invariance: every chi and every graph
        feature must be unchanged by it, since the alternative to an equivariant
        architecture is invariant features plus this augmentation. Translation
        augmentation is meaningless in torsion space but is included because it
        is the cheaper half of the same assertion.
        """
        r = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        t = np.zeros(3) if translation is None else np.asarray(
            translation, dtype=np.float64
        )
        if r.shape != (3, 3):
            raise ValueError(f"rotation must be (3, 3), got {r.shape}")

        new_residues = [res.with_coords(res.coords @ r.T + t) for res in self.residues]
        atoms = self.atoms.copy()
        for idx, res in zip(self.atom_indices, new_residues):
            atoms.coord[idx] = res.coords
        return Protein(
            atoms=atoms,
            residues=new_residues,
            atom_indices=self.atom_indices,
            chain_ids=self.chain_ids,
            res_ids=self.res_ids,
        )

    # -- io -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        save_structure(str(path), self.atoms)


def _repaired_pdb_lines(path: Path) -> list[str]:
    """Fill in blank occupancy/B-factor columns so a strict parser accepts the file.

    Third-party packers write PDB files to varying standards. SCWRL4's output in
    AttnPacker's benchmark release leaves the B-factor column empty, which makes
    biotite raise while converting it to a float -- the file is otherwise fine
    and its coordinates are what we want. Only columns 55-66 of ATOM/HETATM
    records are touched, and only when blank, so no coordinate can be altered.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            padded = line.ljust(80)
            if not padded[54:60].strip():
                padded = padded[:54] + "  1.00" + padded[60:]
            if not padded[60:66].strip():
                padded = padded[:60] + "  0.00" + padded[66:]
            line = padded.rstrip()
        out.append(line)
    return out


def load_protein(
    path: str | Path, model: int = 1, repair_columns: bool = False
) -> Protein:
    """Load a PDB/mmCIF file as a :class:`Protein`.

    Keeps standard amino acids and heavy atoms only, and keeps a single altloc
    per atom. Non-standard residue names are dropped rather than guessed at;
    silently mis-typing a residue would corrupt every chi downstream of it.

    ``repair_columns`` retries a PDB that a strict parse rejects, filling blank
    occupancy and B-factor fields. Off by default: it should be an explicit
    decision for third-party files, not a silent fallback that hides a genuinely
    broken input.
    """
    path = Path(path)
    try:
        atoms = load_structure(str(path), model=model, altloc="first")
    except ValueError:
        if not repair_columns or path.suffix.lower() not in (".pdb", ".ent"):
            raise
        import io

        from biotite.structure.io import pdb

        # Read through the normal path rather than assigning ``.lines``, which
        # skips the index building that ``get_structure`` then relies on.
        handle = pdb.PDBFile.read(
            io.StringIO("\n".join(_repaired_pdb_lines(path)) + "\n")
        )
        atoms = handle.get_structure(model=model, altloc="first")
    atoms = atoms[filter_amino_acids(atoms)]
    atoms = atoms[atoms.element != "H"]
    atoms = atoms[np.isin(atoms.res_name, rc.AA3)]

    residues: list[Residue] = []
    atom_indices: list[np.ndarray] = []
    chain_ids: list[str] = []
    res_ids: list[int] = []

    keys = list(zip(atoms.chain_id, atoms.res_id, atoms.ins_code, atoms.res_name))
    start = 0
    for stop in range(1, len(keys) + 1):
        if stop < len(keys) and keys[stop] == keys[start]:
            continue
        idx = np.arange(start, stop)
        chain, res_id, _ins, res_name = keys[start]
        names = tuple(str(n) for n in atoms.atom_name[idx])
        if len(set(names)) != len(names):
            # Duplicate atom names within one residue means the altloc filter
            # did not collapse it. Skip rather than pick arbitrarily.
            start = stop
            continue
        residues.append(
            Residue(
                resname=str(res_name),
                atom_names=names,
                coords=np.asarray(atoms.coord[idx], dtype=np.float64),
            )
        )
        atom_indices.append(idx)
        chain_ids.append(str(chain))
        res_ids.append(int(res_id))
        start = stop

    return Protein(
        atoms=atoms,
        residues=residues,
        atom_indices=atom_indices,
        chain_ids=chain_ids,
        res_ids=res_ids,
    )


def sidechain_rmsd(
    a: Protein, b: Protein, respect_symmetry: bool = True
) -> np.ndarray:
    """Per-residue heavy-atom side-chain RMSD in Angstrom, NaN where undefined.

    No superposition: the backbone is fixed, so the structures are already in a
    common frame and superposing would hide exactly the error being measured.

    ``respect_symmetry`` minimises over ``rc.symmetric_relabellings`` -- the atom
    permutations that give a chemically identical structure. Without it, a PHE
    ring flipped by 180 degrees scores a large RMSD against a reference that is
    physically the same structure, which inflates the number in exactly the
    tier-1 comparison against published packers that has to be auditable.
    Measured on real entries, a symmetric flip moves atoms by ~2.4 A, so the
    error this removes is not small. Pass False only to see the unsymmetrised
    number.
    """
    if len(a) != len(b):
        raise ValueError(f"residue count differs: {len(a)} vs {len(b)}")
    out = np.full(len(a), np.nan)
    for i, (ra, rb) in enumerate(zip(a.residues, b.residues)):
        if ra.resname != rb.resname or ra.atom_names != rb.atom_names:
            raise ValueError(f"residue {i} differs between structures")
        wanted = rc.sidechain_atoms(ra.resname) - {"CB"}
        sel = [j for j, n in enumerate(ra.atom_names) if n in wanted]
        if not sel:
            continue

        index = ra.index
        options = (
            rc.symmetric_relabellings(ra.resname) if respect_symmetry else ({},)
        )
        best = np.inf
        for relabel in options:
            # Compare a's atom j against b's atom that a's label maps to.
            other = [index.get(relabel.get(ra.atom_names[j], ra.atom_names[j]), j) for j in sel]
            d = ra.coords[sel] - rb.coords[other]
            best = min(best, float(np.sqrt(np.mean(np.sum(d * d, axis=-1)))))
        out[i] = best
    return out
