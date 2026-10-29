"""The published side-chain packing benchmark: target sets and reference rows.

Shared by ``scripts/validate_benchmark.py``, which scores the *baselines' own*
predictions to check that our metrics reproduce their table, and
``scripts/benchmark.py``, which scores a trained rule on the same targets. Both
have to agree about which targets count and what the published numbers are, so
that lives here rather than in either script.

**What the numbers below are, exactly.** DiffPack (NeurIPS 2023) Tables 1 and 2,
transcribed from the paper. Angle MAE in degrees per chi; Angle Accuracy as the
percentage of *residues* whose every chi is within 20 degrees, split all / core
/ surface; Atom RMSD in Angstrom over side-chain atoms, same split. Core is a
residue with at least 20 CB atoms within 10 A, surface at most 15. Those
definitions, and the 20-degree threshold, are DiffPack's section 5.1, and
:class:`aanca.metrics.Convention` implements them.

**What they are not.** FlowPacker (MLSB 2024; Bioinformatics 41(3) btaf010,
2025) reports a table that looks the same and is not: its accuracy columns are
*per chi* rather than per residue, which reads about ten points high, and its
test set is 20 / 34 / 45 targets for CASP13 / 14 / 15 taken from the Prediction
Center rather than the target set in AttnPacker's release. Its rows therefore
cannot be pasted into this table, and are kept in :data:`OTHER_PROTOCOLS` with
the protocol attached. The same paper's re-run of DiffPack ("DiffPack-fix",
with idealised bond geometry as input) does not reproduce DiffPack's own
published numbers either, which is worth knowing before anyone treats a
one-decimal difference as a result.
"""

from __future__ import annotations

import re
from pathlib import Path

#: CASP13 regular targets are numbered T0949-T1022, CASP14 T1024-T1101. The
#: archive also holds older CASP targets (T0658, T0734, ...), so filtering on
#: the T0/T1 prefix alone would pull CASP10/11 entries into "CASP13" and give
#: numbers that match nothing published.
CASP_RANGES = {13: (949, 1022), 14: (1024, 1101)}

#: ``method -> (chi MAE x4, accuracy all/core/surface %, RMSD all/core/surface)``.
#: Keys match the directory names in AttnPacker's archive where they
#: correspond; ``diffpack`` has no directory, since the archive predates it.
REFERENCE = {
    13: {
        "scwrl": ([27.64, 28.97, 49.75, 61.54], (56.2, 71.3, 43.4), (0.934, 0.495, 1.027)),
        "faspr": ([27.04, 28.41, 50.30, 60.89], (56.4, 70.3, 43.6), (0.910, 0.502, 1.002)),
        "rosetta_packer": ([25.88, 28.25, 48.13, 59.82], (58.6, 75.3, 35.7), (0.872, 0.422, 1.001)),
        "dl_packer": ([22.18, 27.00, 51.22, 70.04], (58.8, 73.9, 45.4), (0.772, 0.402, 0.876)),
        "AttnPacker_PP": ([18.92, 23.17, 44.89, 58.98], (62.1, 73.7, 47.6), (0.669, 0.366, 0.775)),
        "diffpack": ([15.35, 19.19, 37.30, 50.19], (69.5, 82.7, 57.3), (0.579, 0.298, 0.696)),
    },
    14: {
        "scwrl": ([33.50, 33.05, 51.61, 55.28], (45.4, 62.5, 33.2), (1.062, 0.567, 1.216)),
        "faspr": ([33.04, 32.49, 50.15, 54.82], (46.3, 62.4, 34.0), (1.048, 0.594, 1.205)),
        "rosetta_packer": ([31.79, 28.25, 50.54, 56.16], (47.5, 67.2, 33.5), (1.006, 0.501, 1.183)),
        "dl_packer": ([29.01, 33.00, 53.98, 72.88], (48.0, 66.9, 33.9), (0.929, 0.476, 1.107)),
        "AttnPacker_PP": ([25.34, 28.19, 48.77, 51.92], (50.9, 66.2, 36.3), (0.823, 0.438, 1.001)),
        # DiffPack's own prose says 0.793 A for all residues where Table 2 says
        # 0.770. The table is used here; the discrepancy is theirs and is worth
        # remembering before reading anything into a third decimal.
        "diffpack": ([21.91, 25.54, 44.27, 55.03], (57.5, 77.8, 43.5), (0.770, 0.356, 0.956)),
    },
}

#: Rows measured under a *different* protocol, kept apart on purpose. Each is
#: ``(chi MAE x4, per-chi accuracy %, RMSD all/core/surface)`` -- note the
#: middle tuple is not the same quantity as in :data:`REFERENCE`.
OTHER_PROTOCOLS = {
    "flowpacker": {
        "source": "MLSB 2024 Table 1; 20 CASP13 / 34 CASP14 targets from the "
        "Prediction Center, per-chi accuracy, not the set or the definition "
        "used in REFERENCE",
        13: ([16.44, 24.38, 42.82, 52.16], (85.69, 75.55, 50.31, 51.00), (0.674, 0.388, 0.865)),
        14: ([22.17, 29.52, 45.33, 50.48], (77.37, 66.66, 44.00, 46.64), (0.830, 0.503, 1.053)),
    },
}


def target_number(stem: str) -> int | None:
    match = re.match(r"^T(\d{4})", stem)
    return int(match.group(1)) if match else None


def targets_for(
    results: Path, casp: int, methods: str | list[str], backbone: str = "casp_all"
) -> list[str]:
    """Target stems in ``casp``'s range that every one of ``methods`` covers.

    Passing several methods is how a run of ours is scored on the *same* set
    the published rows cover: a target one baseline failed on is missing from
    its directory, and including it for us only would compare two different
    averages.
    """
    if isinstance(methods, str):
        methods = [methods]
    low, high = CASP_RANGES[casp]
    native_dir = results / "native" / backbone
    if not native_dir.is_dir():
        return []

    available: set[str] | None = None
    for method in methods:
        method_dir = results / method / backbone
        if not method_dir.is_dir():
            return []
        stems = {p.stem for p in method_dir.glob("*.pdb")}
        available = stems if available is None else (available & stems)

    out = []
    for path in sorted(native_dir.glob("*.pdb")):
        number = target_number(path.stem)
        if number is None or not low <= number <= high:
            continue
        if available is None or path.stem in available:
            out.append(path.stem)
    return out
