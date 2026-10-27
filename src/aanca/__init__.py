"""AANCA -- an amino-acid neural cellular automaton for side-chain repacking.

Milestone 1 is the geometry and data layer: extract chi1-chi4 for all 20 residue
types, modify them exactly, and prove the round trip. Nothing here trains a
model yet.
"""

from . import damage, geometry, residue_constants, structure, torsions
from .damage import patch_damage, randomise_rotamers, torsion_noise
from .structure import Protein, load_protein, sidechain_rmsd
from .torsions import Residue, apply_chi_deltas, chi_error, measure_chis, set_chis

__all__ = [
    "Protein",
    "Residue",
    "apply_chi_deltas",
    "chi_error",
    "damage",
    "geometry",
    "patch_damage",
    "randomise_rotamers",
    "torsion_noise",
    "load_protein",
    "measure_chis",
    "residue_constants",
    "set_chis",
    "sidechain_rmsd",
    "structure",
    "torsions",
]
