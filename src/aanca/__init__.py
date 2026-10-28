"""AANCA -- an amino-acid neural cellular automaton for side-chain repacking.

The geometry and data layer: extract chi1-chi4 for all 20 residue types, modify
them exactly, build the per-step perception graph. See ``docs/PLAN_V1.md`` for
the model and training plan.

**Importing this package does not import Torch.** ``model``, ``losses`` and
``torch_geometry`` are deliberately left out of these re-exports so the data
layer stays usable without a Torch install -- import them by module path when
you want them.
"""

from . import damage, geometry, graph, residue_constants, structure, torsions
from .damage import patch_damage, randomise_rotamers, torsion_noise
from .graph import ResidueGraph, build_graph
from .structure import Protein, load_protein, sidechain_rmsd
from .torsions import Residue, apply_chi_deltas, chi_error, measure_chis, set_chis

__all__ = [
    "Protein",
    "Residue",
    "ResidueGraph",
    "apply_chi_deltas",
    "build_graph",
    "chi_error",
    "damage",
    "geometry",
    "graph",
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
