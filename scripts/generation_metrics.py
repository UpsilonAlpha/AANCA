"""Metric suite for generated protein backbones, from CA coordinates alone.

    python scripts/generation_metrics.py --generated samples.npy --json out.json
    python scripts/generation_metrics.py --natural --structures 60 --json out.json
    python scripts/generation_metrics.py --generated samples.npy \
        --reference data/pdb --reference-structures 100

The comparison target is the diffusion-model literature, which reports
designability as the ProteinMPNN + ESMFold self-consistency rate. We cannot
assume either network is installed and we have no internet, so no number here
depends on a learned model: everything is geometry over a CA trace.

The point of the suite is **not** that the numbers are good in isolation. A
clash rate of 0.3% means nothing until the same code says what a crystal
structure scores. So ``--natural`` runs this project's own PDB entries through
exactly the same functions, residue for residue, and emits exactly the same
table. Every generated number has an in-house reference computed identically,
or it is not reportable.

Four blocks, all from CA coordinates:

1. ``geometry``   -- virtual CA-CA bond length, clash rate, radius of gyration.
2. ``secondary``  -- P-SEA helix/strand/coil assignment and a per-structure class.
3. ``diversity``  -- pairwise TM-score over the sample set, and cluster counts.
4. ``novelty``    -- max TM-score of each sample to a local reference set.

numpy only. No torch, no GPU: the project's torch code is for the model, and a
metric that cannot run on a login node is a metric nobody runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aanca.structure import load_protein  # noqa: E402

DATA_PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"

#: Ideal virtual CA-CA distance for consecutive residues in a trans peptide,
#: and the tolerance the outlier fraction is counted against. 3.80 A is the
#: standard value; 0.10 A is roughly the spread seen in crystal structures, so
#: the outlier fraction on natural proteins is a few percent, not zero.
CA_CA_IDEAL = 3.80
CA_CA_TOLERANCE = 0.10

#: Two residues at least this far apart in sequence are counted as clashing if
#: their CA atoms are closer than ``CLASH_DISTANCE``. |i-j| >= 3 excludes the
#: i,i+1 bond and the i,i+2 pair, which are short by construction in a helix
#: (~5.4 A) and cannot clash in any meaningful sense.
CLASH_MIN_SEPARATION = 3
CLASH_DISTANCE = 4.0

#: Empirical globular-protein scaling Rg ~ 2.2 * N^0.38 A. The ratio to it is
#: the number that matters: a generated backbone that is too extended or
#: collapsed shows up here even when every bond length is perfect.
RG_PREFACTOR = 2.2
RG_EXPONENT = 0.38


# ---------------------------------------------------------------------------
# superposition
# ---------------------------------------------------------------------------


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking ``mobile`` onto ``target``.

    Returns ``(rot, trans)`` with ``mobile @ rot + trans`` optimally
    superposed. Reflections are removed by flipping the sign of the smallest
    singular value, so the result is always a proper rotation -- a reflection
    would let a mirror-image backbone score as a match.

    The project's own superposition (``aanca.fape.ca_rmsd_on``) is torch; this
    is the same arithmetic in numpy, kept here so the suite has no torch
    dependency. The two are checked against each other in the tests.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"need matching (n, 3) arrays, got {mobile.shape} and {target.shape}")
    if len(mobile) < 3:
        raise ValueError(f"need at least 3 points to superpose, got {len(mobile)}")
    mc = mobile.mean(axis=0)
    tc = target.mean(axis=0)
    u, _s, vh = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(u @ vh))
    correction = np.diag([1.0, 1.0, d])
    rot = u @ correction @ vh
    return rot, tc - mc @ rot


# ---------------------------------------------------------------------------
# 1. geometry
# ---------------------------------------------------------------------------


def ca_bond_lengths(ca: np.ndarray) -> np.ndarray:
    """``(n-1,)`` virtual CA-CA distances for consecutive residues."""
    ca = np.asarray(ca, dtype=np.float64)
    return np.linalg.norm(ca[1:] - ca[:-1], axis=-1)


def clash_rate(
    ca: np.ndarray,
    min_separation: int = CLASH_MIN_SEPARATION,
    cutoff: float = CLASH_DISTANCE,
) -> tuple[float, int, int]:
    """Fraction of ``|i-j| >= min_separation`` CA pairs closer than ``cutoff``.

    Returns ``(rate, clashes, pairs)``. ``rate`` is NaN when there are no such
    pairs, which happens only for traces shorter than ``min_separation + 1``.
    """
    ca = np.asarray(ca, dtype=np.float64)
    n = len(ca)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    considered = sep >= min_separation
    pairs = int(considered.sum() // 2)
    if pairs == 0:
        return float("nan"), 0, 0
    clashes = int((considered & (d < cutoff)).sum() // 2)
    return clashes / pairs, clashes, pairs


def radius_of_gyration(ca: np.ndarray) -> float:
    """Root-mean-square distance of the CA atoms from their centroid."""
    ca = np.asarray(ca, dtype=np.float64)
    centred = ca - ca.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centred * centred, axis=-1))))


def expected_rg(n_residues: int) -> float:
    """``2.2 * N^0.38`` A, the empirical globular scaling."""
    return RG_PREFACTOR * float(n_residues) ** RG_EXPONENT


def geometry_metrics(ca: np.ndarray) -> dict:
    """Bond length, clash and compactness numbers for one CA trace."""
    ca = np.asarray(ca, dtype=np.float64)
    bonds = ca_bond_lengths(ca)
    lo, hi = CA_CA_IDEAL - CA_CA_TOLERANCE, CA_CA_IDEAL + CA_CA_TOLERANCE
    rate, clashes, pairs = clash_rate(ca)
    rg = radius_of_gyration(ca)
    expected = expected_rg(len(ca))
    return {
        "n_residues": int(len(ca)),
        "ca_ca_mean": float(bonds.mean()) if len(bonds) else float("nan"),
        "ca_ca_sd": float(bonds.std(ddof=0)) if len(bonds) else float("nan"),
        "ca_ca_outlier_fraction": (
            float(np.mean((bonds < lo) | (bonds > hi))) if len(bonds) else float("nan")
        ),
        "ca_ca_window": [lo, hi],
        "clash_rate": rate,
        "clashes": clashes,
        "clash_pairs_considered": pairs,
        "radius_of_gyration": rg,
        "radius_of_gyration_expected": expected,
        "radius_of_gyration_ratio": rg / expected if expected > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# 2. secondary structure: P-SEA
# ---------------------------------------------------------------------------

#: P-SEA thresholds, exactly as published in Labesse, Colloc'h, Pothier and
#: Mornon, "P-SEA: a new efficient assignment of secondary structure from Ca
#: trace of proteins", Bioinformatics 13(3):291-295, 1997, Table 1.
#:
#: Distances in Angstrom between CA(i) and CA(i+k); ``r`` is the CA pseudo-bond
#: angle CA(i-1)-CA(i)-CA(i+1) in degrees; ``a`` is the CA pseudo-torsion
#: CA(i-1)-CA(i)-CA(i+1)-CA(i+2) in degrees. Every value below is
#: ``centre +/- tolerance`` from the paper.
PSEA_HELIX = {
    "d2": (5.5, 0.5),
    "d3": (5.3, 0.5),
    "d4": (6.4, 0.6),
    "r": (89.0, 12.0),
    "a": (50.0, 20.0),
}
PSEA_STRAND = {
    "d2": (6.7, 0.6),
    "d3": (9.9, 0.9),
    "d4": (12.4, 1.1),
    "r": (124.0, 14.0),
    "a": (-170.0, 45.0),
}

#: Run-length requirements from the paper: a helix seed is five consecutive
#: residues meeting the strict criterion, a strand seed four; a three-residue
#: strand seed is accepted only if the region has at least five CA neighbours
#: in the 4.2-5.2 A shell, which is the paper's test for a strand paired into a
#: sheet. Seeds are then extended by at most one residue at each end if that
#: residue meets the relaxed criterion.
PSEA_HELIX_RUN = 5
PSEA_STRAND_RUN = 4
PSEA_SHORT_STRAND_RUN = 3
PSEA_SHEET_CONTACTS = 5
PSEA_SHEET_SHELL = (4.2, 5.2)

COIL, HELIX, STRAND = 0, 1, 2
PSEA_SYMBOLS = {COIL: "c", HELIX: "a", STRAND: "b"}


def _pseudo_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle a-b-c in degrees, elementwise over leading axes."""
    u = a - b
    v = c - b
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)))


def _pseudo_torsion(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Torsion a-b-c-d in degrees on ``(-180, 180]``, elementwise."""
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.degrees(np.arctan2(y, x))


def _within(values: np.ndarray, spec: tuple[float, float]) -> np.ndarray:
    centre, tolerance = spec
    with np.errstate(invalid="ignore"):
        return (values >= centre - tolerance) & (values <= centre + tolerance)


def _within_torsion(values: np.ndarray, spec: tuple[float, float]) -> np.ndarray:
    """``_within`` with the torsion wrapped, so -170 +/- 45 spans +-180."""
    centre, tolerance = spec
    with np.errstate(invalid="ignore"):
        delta = np.abs((values - centre + 180.0) % 360.0 - 180.0)
        return delta <= tolerance


def _mask_consecutive(mask: np.ndarray, number: int) -> np.ndarray:
    """True for every element inside a run of ``number`` consecutive Trues."""
    if number > len(mask):
        return np.zeros(len(mask), dtype=bool)
    counts = np.zeros(len(mask) - number + 1, dtype=int)
    for i in range(number):
        counts += mask[i : i + len(counts)].astype(int)
    seed = counts == number
    out = np.zeros(len(mask), dtype=bool)
    for i in range(number):
        out[i : i + len(seed)] |= seed
    return out


def _extend_region(base: np.ndarray, relaxed: np.ndarray) -> np.ndarray:
    """Grow each True run of ``base`` by one element per side where ``relaxed``."""
    out = base.copy()
    n = len(base)
    change = np.diff(np.concatenate(([False], base)))
    for i in np.where(change & base)[0]:  # run starts
        if i - 1 >= 0 and relaxed[i - 1]:
            out[i - 1] = True
    for i in np.where(change & ~base)[0]:  # first element after a run
        if i < n and relaxed[i]:
            out[i] = True
    return out


def _sheet_paired(ca: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Keep candidate runs with enough CA neighbours in the sheet-pairing shell."""
    if not candidate.any():
        return np.zeros(len(candidate), dtype=bool)
    lo, hi = PSEA_SHEET_SHELL
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    contacts = np.where(candidate, ((d > lo) & (d <= hi)).sum(axis=1), 0)
    out = np.zeros(len(candidate), dtype=bool)
    start = None
    for i in range(len(candidate) + 1):
        inside = i < len(candidate) and candidate[i]
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            if contacts[start:i].sum() >= PSEA_SHEET_CONTACTS:
                out[start:i] = True
            start = None
    return out


def psea(ca: np.ndarray) -> np.ndarray:
    """Per-residue secondary structure from a CA trace, by P-SEA.

    Returns an ``(n,)`` int array of ``COIL`` / ``HELIX`` / ``STRAND``.

    Implemented from the published criteria in Labesse, Colloc'h, Pothier and
    Mornon, Bioinformatics 13(3):291-295 (1997). The exact thresholds used are
    the module constants ``PSEA_HELIX`` and ``PSEA_STRAND``, each written as
    ``centre +/- tolerance``:

    =========================  ===================  ====================
    quantity                   helix                strand
    =========================  ===================  ====================
    d2 = |CA(i) - CA(i+2)|     5.5 +/- 0.5 A        6.7 +/- 0.6 A
    d3 = |CA(i) - CA(i+3)|     5.3 +/- 0.5 A        9.9 +/- 0.9 A
    d4 = |CA(i) - CA(i+4)|     6.4 +/- 0.6 A        12.4 +/- 1.1 A
    r  = CA(i-1,i,i+1) angle   89 +/- 12 deg        124 +/- 14 deg
    a  = CA(i-1..i+2) torsion  50 +/- 20 deg        -170 +/- 45 deg
    =========================  ===================  ====================

    The decision rule, also from the paper:

    * strict helix  = (d3 and d4 in range) or (r and a in range)
    * strict strand = (d2 and d3 and d4 in range) or (r and a in range)
    * relaxed helix = d3 in range or r in range; relaxed strand = d3 in range
    * a helix needs ``PSEA_HELIX_RUN`` = 5 consecutive strict residues, a
      strand ``PSEA_STRAND_RUN`` = 4, or 3 if the run has at least
      ``PSEA_SHEET_CONTACTS`` = 5 CA neighbours in the 4.2-5.2 A shell
    * each accepted run is extended by at most one residue per side where the
      relaxed criterion holds
    * everything else is coil, and strands win over helices where both fire

    Two deliberate departures, both of which the paper's own text licenses.
    The helix ``d2`` window is not used: the paper's decision rule for helices
    is written on d3 and d4 (d2 is tabulated for reference only), and adding it
    rejects the first and last turn of real helices. The strand torsion window
    is applied with the angle wrapped, since -170 +/- 45 deg straddles +-180.
    Both choices are the ones biotite's ``annotate_sse`` makes, which is how
    this implementation is cross-checked in the tests -- it is an independent
    implementation of the same paper, not a wrapper, so agreement between them
    is evidence about the thresholds rather than a tautology.

    This is the one axis where a published diffusion number can be compared,
    so the thresholds above are the thing to argue with, not the code.
    """
    ca = np.asarray(ca, dtype=np.float64)
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError(f"need an (n, 3) CA trace, got {ca.shape}")
    n = len(ca)
    if n <= 5:
        return np.full(n, COIL, dtype=int)

    nan = float("nan")
    d2 = np.full(n, nan)
    d3 = np.full(n, nan)
    d4 = np.full(n, nan)
    r = np.full(n, nan)
    a = np.full(n, nan)

    # Everything is indexed on the *middle* residue i, matching the paper:
    # d2i is |CA(i-1) - CA(i+1)|, ri is the angle at i, ai the torsion from
    # i-1. Index 0 and the tail are left NaN, and NaN fails every comparison.
    d2[1 : n - 1] = np.linalg.norm(ca[2:n] - ca[0 : n - 2], axis=-1)
    d3[1 : n - 2] = np.linalg.norm(ca[3:n] - ca[0 : n - 3], axis=-1)
    d4[1 : n - 3] = np.linalg.norm(ca[4:n] - ca[0 : n - 4], axis=-1)
    r[1 : n - 1] = _pseudo_angle(ca[0 : n - 2], ca[1 : n - 1], ca[2:n])
    a[1 : n - 2] = _pseudo_torsion(
        ca[0 : n - 3], ca[1 : n - 2], ca[2 : n - 1], ca[3:n]
    )

    h_d3 = _within(d3, PSEA_HELIX["d3"])
    h_d4 = _within(d4, PSEA_HELIX["d4"])
    h_r = _within(r, PSEA_HELIX["r"])
    h_a = _within_torsion(a, PSEA_HELIX["a"])
    strict_helix = (h_d3 & h_d4) | (h_r & h_a)
    relaxed_helix = h_d3 | h_r

    s_d2 = _within(d2, PSEA_STRAND["d2"])
    s_d3 = _within(d3, PSEA_STRAND["d3"])
    s_d4 = _within(d4, PSEA_STRAND["d4"])
    s_r = _within(r, PSEA_STRAND["r"])
    s_a = _within_torsion(a, PSEA_STRAND["a"])
    strict_strand = (s_d2 & s_d3 & s_d4) | (s_r & s_a)
    relaxed_strand = s_d3

    helix = _extend_region(_mask_consecutive(strict_helix, PSEA_HELIX_RUN), relaxed_helix)
    strand = _mask_consecutive(strict_strand, PSEA_STRAND_RUN)
    short = _sheet_paired(ca, _mask_consecutive(strict_strand, PSEA_SHORT_STRAND_RUN))
    strand = _extend_region(strand | short, relaxed_strand)

    out = np.full(n, COIL, dtype=int)
    out[helix] = HELIX
    out[strand] = STRAND
    return out


def psea_string(ca: np.ndarray) -> str:
    """P-SEA assignment as the paper's one-letter string (``a``/``b``/``c``)."""
    return "".join(PSEA_SYMBOLS[int(v)] for v in psea(ca))


def structure_class(helix_fraction: float, strand_fraction: float) -> str:
    """Fold class from the two fractions.

    This project's convention, not a published one, chosen to line up roughly
    with CATH's four classes (Orengo et al. 1997) while being computable from
    P-SEA fractions alone:

    * ``irregular``    -- helix + strand < 0.20
    * ``mainly-alpha`` -- helix >= 0.20 and strand < 0.10
    * ``mainly-beta``  -- strand >= 0.20 and helix < 0.10
    * ``alpha/beta``   -- anything else, i.e. both present in quantity
    """
    if helix_fraction + strand_fraction < 0.20:
        return "irregular"
    if helix_fraction >= 0.20 and strand_fraction < 0.10:
        return "mainly-alpha"
    if strand_fraction >= 0.20 and helix_fraction < 0.10:
        return "mainly-beta"
    return "alpha/beta"


def secondary_metrics(ca: np.ndarray) -> dict:
    """Per-residue P-SEA fractions and the fold class for one CA trace."""
    sse = psea(ca)
    n = max(len(sse), 1)
    helix = float(np.mean(sse == HELIX)) if len(sse) else float("nan")
    strand = float(np.mean(sse == STRAND)) if len(sse) else float("nan")
    coil = float(np.mean(sse == COIL)) if len(sse) else float("nan")
    return {
        "helix_fraction": helix,
        "strand_fraction": strand,
        "coil_fraction": coil,
        "n_assigned": int(n),
        "class": structure_class(helix, strand),
    }


# ---------------------------------------------------------------------------
# 3. TM-score
# ---------------------------------------------------------------------------


def d0_of(length: int) -> float:
    """TM-score's length normalisation ``d0 = 1.24 * (L-15)^(1/3) - 1.8``.

    Defined for ``L > 21``; below that TM-score's own code falls back to a
    floor of 0.5 A, which is kept here so short traces do not produce a
    negative or imaginary d0.
    """
    if length > 21:
        return 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8
    return 0.5


def _tm_on_correspondence(
    mobile: np.ndarray,
    target: np.ndarray,
    d0: float,
    l_norm: int,
    max_starts: int,
    max_iterations: int = 20,
) -> float:
    """Best TM-score over seed-fragment superpositions of a fixed correspondence.

    ``mobile[k]`` is aligned to ``target[k]``. This is TM-score's own search:
    superpose a seed fragment, score every pair, keep the pairs inside a
    distance cutoff, re-superpose on those, iterate to a fixed point; repeat
    over seed lengths L, L/2, L/4, ... >= 4 at several start offsets.
    """
    n = len(mobile)
    if n < 3:
        return 0.0
    d0_search = min(max(d0, 4.5), 8.0)
    best = 0.0
    lengths: list[int] = []
    size = n
    while size >= 4:
        lengths.append(size)
        size //= 2
    if 4 not in lengths and n >= 4:
        lengths.append(4)
    if not lengths:
        lengths = [n]

    for l_init in lengths:
        n_starts = min(max_starts, n - l_init + 1)
        starts = (
            np.unique(np.linspace(0, n - l_init, n_starts).round().astype(int))
            if n_starts > 0
            else np.array([0])
        )
        for start in starts:
            selection = np.arange(start, start + l_init)
            if len(selection) < 3:
                continue
            for _ in range(max_iterations):
                rot, trans = kabsch(mobile[selection], target[selection])
                d = np.linalg.norm(mobile @ rot + trans - target, axis=-1)
                score = float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / l_norm)
                if score > best:
                    best = score
                cutoff = d0_search - 1.0
                new = np.where(d < cutoff)[0]
                while len(new) < 3 and cutoff < 1e3:
                    cutoff += 0.5
                    new = np.where(d < cutoff)[0]
                if len(new) < 3 or (
                    len(new) == len(selection) and np.array_equal(new, selection)
                ):
                    break
                selection = new
    return best


def tm_score(
    mobile: np.ndarray,
    target: np.ndarray,
    l_norm: int | None = None,
    max_starts: int = 8,
    max_offsets: int = 64,
) -> float:
    """TM-score of ``mobile`` against ``target``, both CA traces.

    ``d0 = 1.24 * (L - 15)^(1/3) - 1.8`` with ``L = l_norm``, which defaults to
    ``len(target)``, as TM-score normalises by the reference length.

    **This is a TM-score with a simplified search, not TM-align's.** The
    residue correspondence here is restricted to gap-free threadings: for equal
    lengths that is the identity alignment (which is what you want when
    comparing generated samples that share a residue indexing), and for unequal
    lengths it is the best sequential offset. TM-align instead runs dynamic
    programming over a distance-derived score matrix and permits gaps, so it
    can find correspondences this cannot. Within each threading the iterative
    seed-fragment superposition is TM-score's own, with ``max_starts`` start
    offsets per seed length rather than all of them.

    Consequently **every value from this function is a lower bound** on the
    TM-align value. That is the safe direction for the two uses here: it can
    only make the sample set look more diverse and more novel than it is, never
    less, so a diversity or novelty claim made on these numbers is
    conservative. It is the wrong direction for claiming two structures match.

    Returns a float in ``(0, 1]``.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    for name, arr in (("mobile", mobile), ("target", target)):
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must be (n, 3), got {arr.shape}")
    la, lb = len(mobile), len(target)
    if la < 3 or lb < 3:
        raise ValueError(f"need at least 3 residues in each trace, got {la} and {lb}")
    l_norm = int(lb if l_norm is None else l_norm)
    d0 = d0_of(l_norm)

    if la == lb:
        return _tm_on_correspondence(
            mobile, target, d0, l_norm, max_starts=max_starts
        )

    # Gap-free threading. Offset o pairs mobile[i] with target[i + o]; the
    # overlap is whatever both have. Screen every offset cheaply (one
    # full-overlap seed), then re-run the full seed search on the winner.
    offsets = np.arange(-(la - 3), lb - 3 + 1)
    if len(offsets) > max_offsets:
        offsets = np.unique(np.linspace(offsets[0], offsets[-1], max_offsets).round().astype(int))

    def pair(offset: int) -> tuple[np.ndarray, np.ndarray]:
        lo = max(0, -offset)
        hi = min(la, lb - offset)
        return mobile[lo:hi], target[lo + offset : hi + offset]

    best = 0.0
    best_offset = 0
    for offset in offsets:
        a, b = pair(int(offset))
        if len(a) < 3:
            continue
        score = _tm_on_correspondence(a, b, d0, l_norm, max_starts=1)
        if score > best:
            best = score
            best_offset = int(offset)
    a, b = pair(best_offset)
    if len(a) >= 3:
        best = max(best, _tm_on_correspondence(a, b, d0, l_norm, max_starts=max_starts))
    return best


def tm_matrix(traces: list[np.ndarray], max_starts: int = 8) -> np.ndarray:
    """Symmetric ``(k, k)`` TM-score matrix with 1.0 on the diagonal.

    For equal-length traces TM-score is symmetric, since the optimal
    superposition and hence the distance set is the same either way and d0
    depends only on the common length. For unequal lengths it is not, so each
    off-diagonal entry is the larger of the two directions -- the generous
    choice, which makes the cluster count a lower bound on the true diversity.
    """
    k = len(traces)
    out = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            a = tm_score(traces[i], traces[j], max_starts=max_starts)
            if len(traces[i]) == len(traces[j]):
                b = a
            else:
                b = tm_score(traces[j], traces[i], max_starts=max_starts)
            out[i, j] = out[j, i] = max(a, b)
    return out


def single_linkage_clusters(similarity: np.ndarray, threshold: float) -> list[int]:
    """Cluster labels under single linkage: joined if similarity > threshold.

    Union-find rather than scipy, which is not a dependency of this project.
    """
    k = len(similarity)
    parent = list(range(k))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(k):
        for j in range(i + 1, k):
            if similarity[i, j] > threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = {}
    labels = []
    for i in range(k):
        root = find(i)
        labels.append(roots.setdefault(root, len(roots)))
    return labels


def diversity_metrics(
    traces: list[np.ndarray],
    thresholds: tuple[float, ...] = (0.5, 0.6),
    max_starts: int = 8,
) -> dict:
    """Pairwise TM-score statistics and cluster counts over a sample set.

    Both 0.5 and 0.6 are reported because the diffusion literature uses both:
    0.5 is the conventional same-fold threshold, 0.6 the stricter one some
    papers quote for diversity. Clusters are single-linkage, so a chain of
    near-duplicates collapses to one cluster -- the pessimistic reading of
    diversity, which is the honest one for a generative model.
    """
    k = len(traces)
    if k < 2:
        return {
            "n_samples": k,
            "mean_pairwise_tm": float("nan"),
            "max_pairwise_tm": float("nan"),
            "clusters": {f"tm_{t}": k for t in thresholds},
            "cluster_fraction": {f"tm_{t}": float("nan") for t in thresholds},
        }
    matrix = tm_matrix(traces, max_starts=max_starts)
    iu = np.triu_indices(k, 1)
    values = matrix[iu]
    out = {
        "n_samples": k,
        "mean_pairwise_tm": float(values.mean()),
        "max_pairwise_tm": float(values.max()),
        "min_pairwise_tm": float(values.min()),
        "clusters": {},
        "cluster_fraction": {},
    }
    for t in thresholds:
        n_clusters = len(set(single_linkage_clusters(matrix, t)))
        out["clusters"][f"tm_{t}"] = n_clusters
        out["cluster_fraction"][f"tm_{t}"] = n_clusters / k
    out["tm_matrix"] = matrix.tolist()
    return out


def novelty_metrics(
    traces: list[np.ndarray],
    reference: list[np.ndarray],
    reference_names: list[str] | None = None,
    max_starts: int = 8,
) -> dict:
    """Max TM-score of each sample to a local reference set.

    No database download: the reference set is whatever the caller passes,
    normally the structures the model was trained on. A sample whose max TM to
    the training set is high is a memorisation, not a generation. Because the
    TM-score here is a lower bound (see :func:`tm_score`), a *high* value is
    trustworthy evidence of memorisation and a *low* value is weaker evidence
    of novelty than the same number from TM-align would be.
    """
    if not reference:
        return {"n_reference": 0, "max_tm": [], "mean_max_tm": float("nan")}
    best = []
    nearest = []
    for trace in traces:
        scores = [
            tm_score(trace, ref, max_starts=max_starts) for ref in reference
        ]
        index = int(np.argmax(scores))
        best.append(float(scores[index]))
        nearest.append(
            reference_names[index] if reference_names else index
        )
    return {
        "n_reference": len(reference),
        "max_tm": best,
        "nearest_reference": nearest,
        "mean_max_tm": float(np.mean(best)),
        "median_max_tm": float(np.median(best)),
        "fraction_above_0.5": float(np.mean(np.asarray(best) > 0.5)),
        "fraction_above_0.6": float(np.mean(np.asarray(best) > 0.6)),
    }


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------


@dataclass
class Suite:
    """Everything the suite computes, per structure and in aggregate."""

    names: list[str] = field(default_factory=list)
    per_structure: list[dict] = field(default_factory=list)
    diversity: dict = field(default_factory=dict)
    novelty: dict = field(default_factory=dict)
    source: str = ""

    def as_json(self) -> dict:
        return {
            "source": self.source,
            "names": self.names,
            "per_structure": self.per_structure,
            "aggregate": self.aggregate(),
            "diversity": self.diversity,
            "novelty": self.novelty,
            "thresholds": {
                "ca_ca_ideal": CA_CA_IDEAL,
                "ca_ca_tolerance": CA_CA_TOLERANCE,
                "clash_distance": CLASH_DISTANCE,
                "clash_min_separation": CLASH_MIN_SEPARATION,
                "rg_prefactor": RG_PREFACTOR,
                "rg_exponent": RG_EXPONENT,
                "psea_helix": PSEA_HELIX,
                "psea_strand": PSEA_STRAND,
            },
        }

    def aggregate(self) -> dict:
        keys = (
            "n_residues",
            "ca_ca_mean",
            "ca_ca_sd",
            "ca_ca_outlier_fraction",
            "clash_rate",
            "radius_of_gyration",
            "radius_of_gyration_ratio",
            "helix_fraction",
            "strand_fraction",
            "coil_fraction",
        )
        out: dict = {}
        for key in keys:
            values = np.asarray(
                [row[key] for row in self.per_structure if key in row], dtype=float
            )
            values = values[np.isfinite(values)]
            out[key] = {
                "mean": float(values.mean()) if len(values) else float("nan"),
                "sd": float(values.std(ddof=0)) if len(values) else float("nan"),
                "min": float(values.min()) if len(values) else float("nan"),
                "max": float(values.max()) if len(values) else float("nan"),
            }
        classes: dict[str, int] = {}
        for row in self.per_structure:
            classes[row["class"]] = classes.get(row["class"], 0) + 1
        out["class_counts"] = classes
        # Pooled residue-level fractions: the per-residue number a paper quotes
        # is over all residues, not the mean of per-structure means, and the two
        # differ whenever the structures have different lengths.
        total = sum(row["n_residues"] for row in self.per_structure) or 1
        for name in ("helix", "strand", "coil"):
            out[f"pooled_{name}_fraction"] = (
                sum(
                    row[f"{name}_fraction"] * row["n_residues"]
                    for row in self.per_structure
                )
                / total
            )
        return out


def score_traces(
    traces: list[np.ndarray],
    names: list[str] | None = None,
    reference: list[np.ndarray] | None = None,
    reference_names: list[str] | None = None,
    thresholds: tuple[float, ...] = (0.5, 0.6),
    max_starts: int = 8,
    source: str = "",
    with_diversity: bool = True,
) -> Suite:
    """Score a list of CA traces. The one entry point both modes use.

    This is the whole reason the suite exists in this shape: ``--generated``
    and ``--natural`` differ only in how they get to a list of ``(n, 3)``
    arrays. Every number after that point is computed by the same code on both.
    """
    names = list(names) if names is not None else [f"sample_{i}" for i in range(len(traces))]
    if len(names) != len(traces):
        raise ValueError(f"{len(names)} names for {len(traces)} traces")
    suite = Suite(names=names, source=source)
    for name, trace in zip(names, traces):
        row = {"name": name}
        row.update(geometry_metrics(trace))
        row.update(secondary_metrics(trace))
        suite.per_structure.append(row)
    if with_diversity:
        suite.diversity = diversity_metrics(
            traces, thresholds=thresholds, max_starts=max_starts
        )
    if reference:
        suite.novelty = novelty_metrics(
            traces, reference, reference_names=reference_names, max_starts=max_starts
        )
    return suite


# ---------------------------------------------------------------------------
# input: generated arrays and natural structures
# ---------------------------------------------------------------------------


def load_generated(path: str | Path) -> list[np.ndarray]:
    """Read ``(n_samples, n_residues, 3)`` CA coordinates from .npy or .npz."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as handle:
            key = "ca" if "ca" in handle else list(handle.keys())[0]
            array = handle[key]
    else:
        array = np.load(path)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"expected (n_samples, n_residues, 3) CA coordinates, got {array.shape}"
        )
    return [array[i] for i in range(len(array))]


def ca_trace(
    protein,
    chain: str | None = None,
    require_contiguous: bool = True,
) -> tuple[np.ndarray, str]:
    """The CA trace of one chain of a :class:`~aanca.structure.Protein`.

    Returns ``(coords, label)``. The chain defaults to the one with most
    residues.

    ``require_contiguous`` keeps the longest run of residues with consecutive
    ``res_id`` and a CA atom, which is what makes the comparison honest: a
    generated backbone is a gap-free chain, so a natural chain with an
    unmodelled loop must not be scored across the gap. Scoring across it would
    put a 15 A "bond" in the CA-CA distribution and tell P-SEA that two
    unrelated turns are consecutive -- the reference would then look worse than
    the generated samples for a reason that has nothing to do with the model.
    """
    by_chain: dict[str, list[int]] = {}
    for i, c in enumerate(protein.chain_ids):
        by_chain.setdefault(c, []).append(i)
    if chain is None:
        chain = max(by_chain, key=lambda c: len(by_chain[c]))
    if chain not in by_chain:
        raise KeyError(f"no chain {chain!r}; have {sorted(by_chain)}")

    indices = [i for i in by_chain[chain] if "CA" in protein.residues[i].index]
    if not indices:
        raise ValueError(f"chain {chain!r} has no CA atoms")

    if require_contiguous:
        runs: list[list[int]] = [[indices[0]]]
        for prev, cur in zip(indices, indices[1:]):
            if protein.res_ids[cur] == protein.res_ids[prev] + 1:
                runs[-1].append(cur)
            else:
                runs.append([cur])
        indices = max(runs, key=len)

    coords = np.stack(
        [
            protein.residues[i].coords[protein.residues[i].index["CA"]]
            for i in indices
        ]
    )
    return coords, chain


def _structure_path(pdb_id: str, directory: Path) -> Path | None:
    for suffix in (".bcif", ".cif", ".pdb", ".ent"):
        candidate = directory / f"{pdb_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def natural_traces(
    directory: Path = DATA_PDB,
    limit: int = 60,
    min_residues: int = 50,
    max_residues: int = 400,
    manifest: Path | None = None,
    split: str = "train",
    skip: int = 0,
) -> tuple[list[np.ndarray], list[str]]:
    """CA traces from this project's own PDB entries, through its own loader.

    The length window exists so the reference set is comparable to generated
    samples, which are a fixed modest length; it is not a quality filter.
    Entries are taken in manifest order when a manifest is given, so the set is
    reproducible, and otherwise in sorted filename order.
    """
    candidates: list[tuple[str, Path]] = []
    if manifest is not None and manifest.exists():
        from aanca.data import read_manifest

        for spec in read_manifest(manifest).get(split, []):
            path = _structure_path(spec.pdb_id, directory)
            if path is not None:
                candidates.append((spec.pdb_id, path))
    else:
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() in (".bcif", ".cif", ".pdb", ".ent"):
                candidates.append((path.stem, path))

    traces: list[np.ndarray] = []
    names: list[str] = []
    seen = 0
    for pdb_id, path in candidates:
        if len(traces) >= limit:
            break
        try:
            protein = load_protein(path)
            coords, chain = ca_trace(protein)
        except Exception:
            continue
        if not (min_residues <= len(coords) <= max_residues):
            continue
        seen += 1
        if seen <= skip:
            continue
        traces.append(coords)
        names.append(f"{pdb_id}_{chain}")
    return traces, names


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

_HEADER = (
    "name            N   CA-CA   sd   out%  clash%     Rg  Rg/exp"
    "   helix  strand   coil  class"
)


def format_table(suite: Suite) -> str:
    lines = [_HEADER]
    for row in suite.per_structure:
        lines.append(
            "{name:<12} {n:>4} {bond:>7.3f} {sd:>5.3f} {out:>5.1f} {clash:>7.3f}"
            " {rg:>6.1f} {ratio:>7.3f} {helix:>7.3f} {strand:>7.3f} {coil:>6.3f}"
            "  {cls}".format(
                name=row["name"][:12],
                n=row["n_residues"],
                bond=row["ca_ca_mean"],
                sd=row["ca_ca_sd"],
                out=100 * row["ca_ca_outlier_fraction"],
                clash=100 * row["clash_rate"],
                rg=row["radius_of_gyration"],
                ratio=row["radius_of_gyration_ratio"],
                helix=row["helix_fraction"],
                strand=row["strand_fraction"],
                coil=row["coil_fraction"],
                cls=row["class"],
            )
        )
    agg = suite.aggregate()
    lines.append("")
    lines.append(f"aggregate over {len(suite.per_structure)} structures ({suite.source})")
    for key in (
        "n_residues",
        "ca_ca_mean",
        "ca_ca_sd",
        "ca_ca_outlier_fraction",
        "clash_rate",
        "radius_of_gyration",
        "radius_of_gyration_ratio",
        "helix_fraction",
        "strand_fraction",
        "coil_fraction",
    ):
        s = agg[key]
        lines.append(
            f"  {key:<26} mean {s['mean']:>8.4f}  sd {s['sd']:>8.4f}"
            f"  min {s['min']:>8.4f}  max {s['max']:>8.4f}"
        )
    lines.append(
        "  pooled per-residue        "
        f"helix {agg['pooled_helix_fraction']:.4f}"
        f"  strand {agg['pooled_strand_fraction']:.4f}"
        f"  coil {agg['pooled_coil_fraction']:.4f}"
    )
    lines.append(f"  class counts               {agg['class_counts']}")
    if suite.diversity:
        d = suite.diversity
        lines.append("")
        lines.append("diversity (TM-score, simplified search -- a lower bound)")
        lines.append(
            f"  mean pairwise TM {d['mean_pairwise_tm']:.4f}"
            f"   max {d['max_pairwise_tm']:.4f}"
            f"   min {d.get('min_pairwise_tm', float('nan')):.4f}"
        )
        for key, value in d["clusters"].items():
            frac = d["cluster_fraction"][key]
            lines.append(f"  clusters at {key[3:]:<5} {value:>4}  fraction {frac:.4f}")
    if suite.novelty:
        nv = suite.novelty
        lines.append("")
        lines.append(f"novelty against {nv['n_reference']} reference structures")
        lines.append(
            f"  mean max TM {nv['mean_max_tm']:.4f}"
            f"   median {nv['median_max_tm']:.4f}"
            f"   >0.5 {nv['fraction_above_0.5']:.4f}"
            f"   >0.6 {nv['fraction_above_0.6']:.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--generated", help="an .npy/.npz of (n_samples, n_residues, 3) CA coordinates"
    )
    source.add_argument(
        "--natural",
        action="store_true",
        help="score this project's own PDB entries through the same code path",
    )
    parser.add_argument("--structures", type=int, default=60)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--min-residues", type=int, default=50)
    parser.add_argument("--max-residues", type=int, default=400)
    parser.add_argument("--pdb-dir", default=str(DATA_PDB))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--reference",
        default=None,
        help="directory of PDB/mmCIF reference structures for the novelty block",
    )
    parser.add_argument("--reference-structures", type=int, default=50)
    parser.add_argument("--reference-skip", type=int, default=0)
    parser.add_argument("--max-starts", type=int, default=8)
    parser.add_argument("--no-diversity", action="store_true")
    parser.add_argument("--psea-report", default=None, help="print the P-SEA string for this name")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    if args.generated:
        traces = load_generated(args.generated)
        names = [f"sample_{i}" for i in range(len(traces))]
        source_label = f"generated:{args.generated}"
    else:
        traces, names = natural_traces(
            directory=Path(args.pdb_dir),
            limit=args.structures,
            min_residues=args.min_residues,
            max_residues=args.max_residues,
            manifest=Path(args.manifest),
            split=args.split,
            skip=args.skip,
        )
        source_label = f"natural:{args.pdb_dir}:{args.split}"
    if not traces:
        print("nothing to score")
        return 1

    reference: list[np.ndarray] | None = None
    reference_names: list[str] | None = None
    if args.reference:
        reference, reference_names = natural_traces(
            directory=Path(args.reference),
            limit=args.reference_structures,
            min_residues=args.min_residues,
            max_residues=args.max_residues,
            manifest=Path(args.manifest),
            split=args.split,
            skip=args.reference_skip,
        )

    suite = score_traces(
        traces,
        names=names,
        reference=reference,
        reference_names=reference_names,
        max_starts=args.max_starts,
        source=source_label,
        with_diversity=not args.no_diversity,
    )
    print(format_table(suite))

    if args.psea_report:
        for name, trace in zip(names, traces):
            if name.startswith(args.psea_report):
                print()
                print(f"P-SEA string for {name} ({len(trace)} residues)")
                string = psea_string(trace)
                for start in range(0, len(string), 60):
                    print(f"  {start + 1:>5}  {string[start:start + 60]}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(suite.as_json(), indent=2), encoding="utf-8"
        )
        print()
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
