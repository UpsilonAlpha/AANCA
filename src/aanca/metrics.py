"""The packing benchmark, copied from the published protocol rather than invented.

AttnPacker (PNAS 2023), DiffPack (NeurIPS 2023) and FlowPacker (Bioinformatics
2025) all report the same three numbers on the same test sets, stratified the
same way. Matching them exactly is the only way the tier-1 comparison is
auditable, so this module implements *their* definitions:

* **chi MAE** in degrees, per chi index, with the two-fold symmetric chis folded.
* **chi accuracy**, the fraction of chis within 20 degrees. This is their
  headline number alongside MAE and is easy to omit by accident.
* **side-chain atom RMSD**, symmetry-aware (see ``structure.sidechain_rmsd``).

And stratified by burial, which was the thing most worth copying: **core**
residues are the hard ones, and a pooled number hides whether a method is good
at packing or merely good at solvent-exposed side chains that barely matter.

AttnPacker's convention, as cited by the later papers: a residue is **core** with
at least 20 CB atoms within 10 A and **surface** with at most 15. Note that is
deliberately *not* a partition -- residues with 16 to 19 neighbours belong to
neither band and are reported separately here rather than silently dropped.

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


@dataclass
class BurialMetrics:
    label: str
    n_residues: int
    chi_mae: list[float]
    chi_accuracy: list[float]
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
            rmsd=_masked_mean(torch.nan_to_num(rmsd), residues & ~torch.isnan(rmsd)),
        )
    return out


def sidechain_rmsd_torch(
    predicted: Tensor, native: Tensor, topo: FeatureTopology
) -> Tensor:
    """``(n_res,)`` symmetry-aware side-chain heavy-atom RMSD, NaN where undefined.

    The Torch counterpart of ``structure.sidechain_rmsd``. Minimising over
    ``rc.symmetric_relabellings`` matters: without it a 180-degree-flipped PHE
    scores ~1.8 A against a structure it is chemically identical to, which
    inflates the whole-protein mean by 0.26-0.36 A -- about the entire gap
    between competing methods.
    """
    n_res = predicted.shape[0]
    out = torch.full((n_res,), float("nan"), dtype=predicted.dtype)

    for r in range(n_res):
        resname = topo.torsion.resnames[r]
        names = topo.torsion.atom_names[r]
        index = {name: i for i, name in enumerate(names)}
        wanted = rc.sidechain_atoms(resname) - {"CB"}
        selected = [i for i, name in enumerate(names) if name in wanted]
        if not selected:
            continue

        best = float("inf")
        for relabel in rc.symmetric_relabellings(resname):
            other = [index.get(relabel.get(names[i], names[i]), i) for i in selected]
            diff = predicted[r, selected] - native[r, other]
            best = min(best, float(diff.pow(2).sum(-1).mean().sqrt()))
        out[r] = best
    return out
