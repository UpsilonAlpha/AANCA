"""The packing benchmark, copied from the published protocol rather than invented.

AttnPacker (PNAS 2023), DiffPack (NeurIPS 2023) and FlowPacker (Bioinformatics
2025) all report the same three numbers on the same test sets, stratified the
same way. Matching them exactly is the only way the tier-1 comparison is
auditable, so this module implements *their* definitions:

* **chi MAE** in degrees, **per chi index**, over all residues, symmetric chis
  folded.
* **chi accuracy**, the fraction of chis within 20 degrees, **pooled over chis**
  and split All / Core / Surface.
* **side-chain atom RMSD**, per residue, split All / Core / Surface,
  symmetry-aware.

Note the asymmetry in how they stratify, which is easy to get wrong: MAE is
per-chi and *not* split by burial; accuracy and RMSD are split by burial and
*not* per-chi. :func:`benchmark_table` reproduces that layout.

**Surface residues are the hard ones, not core.** DiffPack reports 82.7% core
against 57.3% surface accuracy on CASP13, and their headline gain over
AttnPacker was +20.4% on *surface* specifically. Core side chains are
sterically boxed in and nearly solved; surface ones are floppy and are where
methods actually differ. An earlier version of this docstring had that
backwards.

The burial convention, from AttnPacker via DiffPack 5.1: a residue is **core**
with at least 20 CB atoms within 10 A and **surface** with at most 15. That is
deliberately *not* a partition -- 16 to 19 neighbours is neither band, and is
reported separately here rather than silently dropped.

One judgement call: GLY has no CB, so the virtual CB from
``torch_graph.frame_atom_coords`` is used for it when counting neighbours. The
published descriptions do not say what they do with glycine; the count is
dominated by the other residues either way, but it is a real ambiguity and
``include_glycine_cb`` exists so the alternative can be checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import residue_constants as rc
from .torch_geometry import TorsionTopology, chi_residual, measure_chis
from .torch_graph import FeatureTopology, frame_atom_coords

#: AttnPacker's burial thresholds, in CB neighbours within 10 Angstrom.
CORE_MIN_NEIGHBOURS = 20
SURFACE_MAX_NEIGHBOURS = 15
BURIAL_RADIUS = 10.0

#: A chi counts as correct within this many degrees.
ACCURACY_TOLERANCE_DEGREES = 20.0

#: Chis the published evaluation treats as two-fold symmetric, as
#: ``(residue, chi_index)`` with chi1 at index 0.
#:
#: **Reverse-engineered, not quoted.** None of the papers state their symmetry
#: set, so it was recovered by recomputing their tables from the predictions they
#: released and sweeping candidate sets (see ``scripts/validate_benchmark.py``).
#: This set is AlphaFold's plus ASN chi2 and HIS chi2 -- amide and imidazole
#: terminal groups being experimentally indistinguishable in density -- but
#: **excluding GLN chi3**, which is the odd part: GLN's terminal amide is as
#: ambiguous as ASN's. Folding GLN chi3 moves chi3 MAE from a 0.4-2.2 degree
#: agreement to a 4-6 degree disagreement across all four methods, so the data
#: is unambiguous even though the chemistry is not. Treat it as an observed
#: convention of theirs rather than a defensible rule.
PUBLICATION_PERIODIC: frozenset[tuple[str, int]] = frozenset(
    {
        ("ASP", 1),
        ("GLU", 2),
        ("PHE", 1),
        ("TYR", 1),
        ("ASN", 1),
        ("HIS", 1),
    }
)


@dataclass(frozen=True)
class Convention:
    """How to compute the metrics.

    Two instances matter. :data:`PUBLISHED` reproduces the packing literature so
    the tier-1 comparison is auditable. :data:`CHEMICAL` is what the geometry
    and the *loss* use, which is a different question: the loss should fold what
    is physically identical, while the metric should fold what the people we are
    comparing against fold. Conflating the two would either make our numbers
    incomparable or put a reverse-engineered convention into the training signal.
    """

    periodic: frozenset[tuple[str, int]]
    #: True scores a *residue* correct only when every one of its chis is within
    #: tolerance. That is the published definition of "Angle Accuracy" --
    #: pooling per chi instead reads about 10 points high.
    accuracy_per_residue: bool
    #: Published RMSD includes CB, which is almost always correct and therefore
    #: lowers the number; excluding it reads ~0.15 A high.
    rmsd_include_cb: bool
    #: Published RMSD does *not* minimise over symmetry-equivalent labellings,
    #: even though the angle error folds symmetry. Inconsistent of them, but it
    #: is what matches: 0.907 against a published 0.910 for FASPR.
    rmsd_symmetry: bool


PUBLISHED = Convention(
    periodic=PUBLICATION_PERIODIC,
    accuracy_per_residue=True,
    rmsd_include_cb=True,
    rmsd_symmetry=False,
)

CHEMICAL = Convention(
    periodic=frozenset(
        (resname, index)
        for resname, flags in rc.CHI_PI_PERIODIC.items()
        for index, flag in enumerate(flags)
        if flag
    ),
    accuracy_per_residue=False,
    rmsd_include_cb=False,
    rmsd_symmetry=True,
)


def periodic_mask(topo: TorsionTopology, convention: Convention) -> Tensor:
    """``(n_res, MAX_CHI)`` bool mask of the chis this convention folds."""
    mask = torch.zeros_like(topo.chi_mask)
    for row, resname in enumerate(topo.resnames):
        for name, index in convention.periodic:
            if resname == name:
                mask[row, index] = True
    return mask


def cb_neighbour_counts(
    coords: Tensor, topo: FeatureTopology, include_glycine_cb: bool = True
) -> Tensor:
    """``(n_res,)`` count of other residues' CB atoms within ``BURIAL_RADIUS``.

    Uses the virtual CB where a residue has none, which is only GLY.
    """
    cb = frame_atom_coords(coords, topo)[:, 4, :]
    keep = torch.ones(cb.shape[0], dtype=torch.bool, device=cb.device)
    if not include_glycine_cb:
        keep = torch.tensor(
            [name != "GLY" for name in topo.torsion.resnames], device=cb.device
        )

    distances = torch.cdist(cb, cb[keep])
    within = distances <= BURIAL_RADIUS
    counts = within.sum(dim=-1)
    # Remove self-counting for the residues present in the reference set.
    counts = counts - keep.to(counts.dtype)
    return counts.clamp_min(0)


def classify_burial(counts: Tensor) -> dict[str, Tensor]:
    """Core / surface / intermediate masks. **Not** a partition -- see module docs."""
    core = counts >= CORE_MIN_NEIGHBOURS
    surface = counts <= SURFACE_MAX_NEIGHBOURS
    return {
        "all": torch.ones_like(core),
        "core": core,
        "surface": surface,
        "intermediate": ~core & ~surface,
    }


def _masked_mean(values: Tensor, mask: Tensor) -> float:
    count = int(mask.sum())
    if count == 0:
        return float("nan")
    return float((values * mask).sum() / count)


def chi_mae_degrees(
    pred: Tensor, native: Tensor, topo: TorsionTopology, residues: Tensor | None = None
) -> list[float]:
    """Per-chi-index MAE in degrees, symmetric chis folded. NaN where empty.

    Per index rather than pooled: chi1 and chi4 accuracy differ a lot, and every
    paper in this line reports them separately.
    """
    errors = chi_residual(pred, native, topo.periodic).abs()
    mask = topo.chi_mask
    if residues is not None:
        mask = mask & residues.unsqueeze(-1)
    return [
        float(torch.rad2deg(torch.tensor(_masked_mean(errors[:, i], mask[:, i]))))
        for i in range(rc.MAX_CHI)
    ]


def chi_accuracy(
    pred: Tensor,
    native: Tensor,
    topo: TorsionTopology,
    residues: Tensor | None = None,
    tolerance_degrees: float = ACCURACY_TOLERANCE_DEGREES,
) -> list[float]:
    """Per-chi-index fraction within ``tolerance_degrees``, symmetric chis folded."""
    tolerance = torch.deg2rad(torch.tensor(tolerance_degrees, dtype=pred.dtype))
    correct = (chi_residual(pred, native, topo.periodic).abs() <= tolerance).to(
        pred.dtype
    )
    mask = topo.chi_mask
    if residues is not None:
        mask = mask & residues.unsqueeze(-1)
    return [_masked_mean(correct[:, i], mask[:, i]) for i in range(rc.MAX_CHI)]


def chi_accuracy_pooled(
    pred: Tensor,
    native: Tensor,
    topo: TorsionTopology,
    residues: Tensor | None = None,
    tolerance_degrees: float = ACCURACY_TOLERANCE_DEGREES,
) -> float:
    """One accuracy over **all** chis of the selected residues.

    This is the published layout: their Angle Accuracy column is a single number
    per burial band, pooled across chi1-chi4, not a per-chi breakdown. Averaging
    the per-chi accuracies instead would weight chi4 -- which only ARG and LYS
    have -- equally with chi1, and give a different number.
    """
    tolerance = torch.deg2rad(torch.tensor(tolerance_degrees, dtype=pred.dtype))
    correct = (chi_residual(pred, native, topo.periodic).abs() <= tolerance).to(
        pred.dtype
    )
    mask = topo.chi_mask
    if residues is not None:
        mask = mask & residues.unsqueeze(-1)
    return _masked_mean(correct, mask)


@dataclass
class BurialMetrics:
    label: str
    n_residues: int
    chi_mae: list[float]
    chi_accuracy: list[float]
    pooled_accuracy: float
    rmsd: float

    def summary(self) -> str:
        mae = " ".join(
            "-" if m != m else f"{m:5.1f}" for m in self.chi_mae
        )
        acc = " ".join(
            "-" if a != a else f"{a:5.1%}" for a in self.chi_accuracy
        )
        rmsd = "-" if self.rmsd != self.rmsd else f"{self.rmsd:.3f}"
        return (
            f"{self.label:<13}{self.n_residues:>6}  MAE {mae}  acc {acc}  "
            f"RMSD {rmsd}"
        )


def evaluate_packing(
    predicted: Tensor,
    native: Tensor,
    topo: FeatureTopology,
    include_glycine_cb: bool = True,
) -> dict[str, BurialMetrics]:
    """The full tier-1 table for one structure, stratified by burial.

    ``predicted`` and ``native`` are padded coordinate tensors on the same
    topology. RMSD is computed per residue and averaged, symmetry-aware, with no
    superposition -- the backbone is fixed, so superposing would hide the error
    being measured.
    """
    torsion = topo.torsion
    pred_chis = measure_chis(predicted, torsion)
    native_chis = measure_chis(native, torsion)

    counts = cb_neighbour_counts(native, topo, include_glycine_cb)
    bands = classify_burial(counts)
    rmsd = sidechain_rmsd_torch(predicted, native, topo)

    out: dict[str, BurialMetrics] = {}
    for label, residues in bands.items():
        out[label] = BurialMetrics(
            label=label,
            n_residues=int(residues.sum()),
            chi_mae=chi_mae_degrees(pred_chis, native_chis, torsion, residues),
            chi_accuracy=chi_accuracy(pred_chis, native_chis, torsion, residues),
            pooled_accuracy=chi_accuracy_pooled(
                pred_chis, native_chis, torsion, residues
            ),
            rmsd=_masked_mean(torch.nan_to_num(rmsd), residues & ~torch.isnan(rmsd)),
        )
    return out


@dataclass
class BenchmarkRow:
    """One method's row in the published layout: MAE per chi, then All/Core/Surface."""

    n_structures: int
    n_residues: int
    chi_mae: list[float]
    accuracy: dict[str, float]
    rmsd: dict[str, float]

    def header(self) -> str:
        return (
            f"{'':<16}{'chi1':>7}{'chi2':>7}{'chi3':>7}{'chi4':>7}"
            f"{'acc all':>9}{'core':>8}{'surf':>8}"
            f"{'rmsd all':>10}{'core':>8}{'surf':>8}"
        )

    def row(self, label: str) -> str:
        def fmt(value, width, spec):
            return (
                f"{'-':>{width}}" if value != value else f"{value:>{width}{spec}}"
            )

        mae = "".join(fmt(m, 7, ".2f") for m in self.chi_mae)
        acc = "".join(
            fmt(self.accuracy.get(b, float("nan")) * 100, w, ".1f")
            for b, w in (("all", 9), ("core", 8), ("surface", 8))
        )
        rmsd = "".join(
            fmt(self.rmsd.get(b, float("nan")), w, ".3f")
            for b, w in (("all", 10), ("core", 8), ("surface", 8))
        )
        return f"{label:<16}{mae}{acc}{rmsd}"


def benchmark_table(
    pairs: list[
        tuple[Tensor, Tensor, FeatureTopology]
        | tuple[Tensor, Tensor, FeatureTopology, Tensor | None]
    ],
    include_glycine_cb: bool = True,
    convention: Convention | None = None,
) -> BenchmarkRow:
    """Aggregate the published table over many structures.

    ``convention`` defaults to :data:`PUBLISHED`, which reproduces the packing
    literature's definitions (see :class:`Convention`).

    Each pair is ``(predicted, native, topology)`` or
    ``(predicted, native, topology, residue_mask)``. The optional mask selects
    residues to score, which is needed when a prediction does not cover every
    native residue -- scoring an unmatched residue against whatever happens to
    sit at its index is how a plausible but meaningless number gets produced.

    **A cropped topology restricts scoring to its supervised residues**, and an
    explicit mask is intersected with them rather than overriding them. The
    boundary shell of a crop was never predicted from complete information, so
    reporting its accuracy would quietly mix a different, harder task into the
    number. Burial bands stay exact under this: a residue counted as a CB
    neighbour within ``BURIAL_RADIUS`` has its minimum heavy-atom distance
    within the same radius, so for a supervised residue every neighbour the
    burial count needs is still present in the crop.

    Pooled across structures by residue, not by averaging per-structure numbers:
    the papers report per-residue statistics, and averaging per-structure means
    would weight a 50-residue target the same as a 500-residue one.
    """
    convention = convention or PUBLISHED
    mae_num = [0.0] * rc.MAX_CHI
    mae_den = [0] * rc.MAX_CHI
    acc_num: dict[str, float] = {}
    acc_den: dict[str, int] = {}
    rmsd_num: dict[str, float] = {}
    rmsd_den: dict[str, int] = {}
    n_residues = 0

    tolerance = torch.deg2rad(torch.tensor(ACCURACY_TOLERANCE_DEGREES))

    for pair in pairs:
        predicted, native, topo = pair[0], pair[1], pair[2]
        scored = pair[3] if len(pair) > 3 else None
        torsion = topo.torsion
        complete = topo.supervised_mask
        scored = complete if scored is None else (scored & complete)

        pred_chis = measure_chis(predicted, torsion)
        native_chis = measure_chis(native, torsion)
        errors = chi_residual(
            pred_chis, native_chis, periodic_mask(torsion, convention)
        ).abs()
        # Burial is a property of the native structure, so it is computed on the
        # full structure and only then restricted to the scored residues -- a
        # residue's neighbour count must not depend on which residues matched.
        bands = classify_burial(cb_neighbour_counts(native, topo, include_glycine_cb))
        bands = {label: mask & scored for label, mask in bands.items()}
        rmsd = sidechain_rmsd_torch(
            predicted,
            native,
            topo,
            include_cb=convention.rmsd_include_cb,
            respect_symmetry=convention.rmsd_symmetry,
        )
        n_residues += int(scored.sum())

        for i in range(rc.MAX_CHI):
            column = torsion.chi_mask[:, i] & scored
            mae_num[i] += float((errors[:, i] * column).sum())
            mae_den[i] += int(column.sum())

        within = errors <= tolerance.to(errors.dtype)
        for label, residues in bands.items():
            chi_mask = torsion.chi_mask & residues.unsqueeze(-1)
            if convention.accuracy_per_residue:
                # A residue counts once, and only if every chi it has is right.
                has_chi = torsion.chi_mask.any(dim=1) & residues
                good = (within | ~torsion.chi_mask).all(dim=1) & has_chi
                acc_num[label] = acc_num.get(label, 0.0) + float(good.sum())
                acc_den[label] = acc_den.get(label, 0) + int(has_chi.sum())
            else:
                acc_num[label] = acc_num.get(label, 0.0) + float(
                    (within.to(errors.dtype) * chi_mask).sum()
                )
                acc_den[label] = acc_den.get(label, 0) + int(chi_mask.sum())

            defined = residues & ~torch.isnan(rmsd)
            rmsd_num[label] = rmsd_num.get(label, 0.0) + float(
                (torch.nan_to_num(rmsd) * defined).sum()
            )
            rmsd_den[label] = rmsd_den.get(label, 0) + int(defined.sum())

    nan = float("nan")
    return BenchmarkRow(
        n_structures=len(pairs),
        n_residues=n_residues,
        chi_mae=[
            float(torch.rad2deg(torch.tensor(mae_num[i] / mae_den[i])))
            if mae_den[i]
            else nan
            for i in range(rc.MAX_CHI)
        ],
        accuracy={
            b: (acc_num[b] / acc_den[b] if acc_den.get(b) else nan) for b in acc_num
        },
        rmsd={
            b: (rmsd_num[b] / rmsd_den[b] if rmsd_den.get(b) else nan) for b in rmsd_num
        },
    )


def sidechain_rmsd_torch(
    predicted: Tensor,
    native: Tensor,
    topo: FeatureTopology,
    include_cb: bool = False,
    respect_symmetry: bool = True,
) -> Tensor:
    """``(n_res,)`` side-chain heavy-atom RMSD, NaN where undefined.

    The Torch counterpart of ``structure.sidechain_rmsd``. The two switches are
    the ones the published convention needs (see :class:`Convention`):
    ``include_cb=True`` and ``respect_symmetry=False`` reproduce their numbers to
    ~0.003 A, while the defaults here are the chemically sensible pair.

    Symmetry matters a lot when it is on: without it a 180-degree-flipped PHE
    scores ~1.8 A against a structure it is chemically identical to, inflating
    the whole-protein mean by 0.26-0.36 A. That the published numbers match the
    *unsymmetrised* RMSD while their angle error *is* folded is inconsistent of
    them, but it is what the data says.
    """
    n_res = predicted.shape[0]
    out = torch.full((n_res,), float("nan"), dtype=predicted.dtype)

    for r in range(n_res):
        resname = topo.torsion.resnames[r]
        names = topo.torsion.atom_names[r]
        index = {name: i for i, name in enumerate(names)}
        wanted = rc.sidechain_atoms(resname)
        if not include_cb:
            wanted = wanted - {"CB"}
        selected = [i for i, name in enumerate(names) if name in wanted]
        if not selected:
            continue

        options = (
            rc.symmetric_relabellings(resname) if respect_symmetry else ({},)
        )
        best = float("inf")
        for relabel in options:
            other = [index.get(relabel.get(names[i], names[i]), i) for i in selected]
            diff = predicted[r, selected] - native[r, other]
            best = min(best, float(diff.pow(2).sum(-1).mean().sqrt()))
        out[r] = best
    return out
