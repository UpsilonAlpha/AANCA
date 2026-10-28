"""Reproduce the published packing numbers from the published predictions.

    python scripts/validate_benchmark.py [--casp 13|14] [--methods scwrl,faspr,...]

AttnPacker's Zenodo release (record 7713779, ``scp_results.zip``) ships the
native test structures *and* every baseline's predicted structures. That makes a
far stronger check available than any self-consistency test: compute our metrics
on **their** predictions and see whether we recover **their** published table.

If the numbers match, `metrics.py` implements the same thing they do and the
tier-1 comparison is auditable. If they do not, the discrepancy is ours and
needs finding before a single training run.

Reference values below are from DiffPack (NeurIPS 2023) Tables 1 and 2, which
is also where the burial definition and the 20-degree accuracy threshold come
from.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from aanca.metrics import benchmark_table
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "external" / "attnpacker" / "extracted" / "standard_results"

#: DiffPack Table 1 (CASP13) and Table 2 (CASP14). Keys match the directory
#: names in the archive where they correspond.
REFERENCE = {
    13: {
        "scwrl": ([27.64, 28.97, 49.75, 61.54], (56.2, 71.3, 43.4), (0.934, 0.495, 1.027)),
        "faspr": ([27.04, 28.41, 50.30, 60.89], (56.4, 70.3, 43.6), (0.910, 0.502, 1.002)),
        "rosetta_packer": ([25.88, 28.25, 48.13, 59.82], (58.6, 75.3, 35.7), (0.872, 0.422, 1.001)),
        "dl_packer": ([22.18, 27.00, 51.22, 70.04], (58.8, 73.9, 45.4), (0.772, 0.402, 0.876)),
        "AttnPacker_PP": ([18.92, 23.17, 44.89, 58.98], (62.1, 73.7, 47.6), (0.669, 0.366, 0.775)),
    },
    14: {
        "scwrl": ([33.50, 33.05, 51.61, 55.28], (45.4, 62.5, 33.2), (1.062, 0.567, 1.216)),
        "faspr": ([33.04, 32.49, 50.15, 54.82], (46.3, 62.4, 34.0), (1.048, 0.594, 1.205)),
        "rosetta_packer": ([31.79, 28.25, 50.54, 56.16], (47.5, 67.2, 33.5), (1.006, 0.501, 1.183)),
    },
}

#: CASP13 regular targets are numbered T0949-T1022, CASP14 T1024-T1101. The
#: archive also contains older CASP targets (T0658, T0734, ...), so filtering by
#: the T0/T1 prefix alone is wrong -- it would pull CASP10/11 entries into
#: "CASP13" and give numbers that match nothing.
CASP_RANGES = {13: (949, 1022), 14: (1024, 1101)}


def target_number(stem: str) -> int | None:
    match = re.match(r"^T(\d{4})", stem)
    return int(match.group(1)) if match else None


def targets_for(casp: int, method: str, backbone: str = "casp_all") -> list[str]:
    low, high = CASP_RANGES[casp]
    native_dir = RESULTS / "native" / backbone
    method_dir = RESULTS / method / backbone
    if not native_dir.is_dir() or not method_dir.is_dir():
        return []
    available = {p.stem for p in method_dir.glob("*.pdb")}
    out = []
    for path in sorted(native_dir.glob("*.pdb")):
        number = target_number(path.stem)
        if number is None or not low <= number <= high:
            continue
        if path.stem in available:
            out.append(path.stem)
    return out


def align_prediction(native, predicted):
    """Put ``predicted``'s coordinates into ``native``'s atom layout.

    Matching by *position* in the two files is wrong and silently so: the two
    were written by different programs, and any difference in chain order,
    residue numbering or atom ordering misaligns the comparison while still
    producing numbers. Residues are matched on ``(chain, res_id, ins_code)``
    with the residue name required to agree, and atoms by name within the
    residue.

    Returns ``(coords, scored)`` where ``coords`` is in the native topology's
    layout and ``scored`` marks the residues that matched.
    """
    import torch

    from aanca.torch_geometry import MAX_ATOMS

    def keys(protein, with_chain: bool):
        out = {}
        for index, res in enumerate(protein.residues):
            key = (
                (protein.chain_ids[index], protein.res_ids[index], res.resname)
                if with_chain
                else (protein.res_ids[index], res.resname)
            )
            out[key] = index
        return out

    # Prefer the chain-qualified match. Some packers rewrite or blank the chain
    # id, so fall back to residue number alone -- but only for single-chain
    # structures, where that is unambiguous.
    with_chain = True
    lookup = keys(predicted, True)
    native_keys = keys(native, True)
    if not (set(lookup) & set(native_keys)):
        single = len(set(native.chain_ids)) == 1 and len(set(predicted.chain_ids)) == 1
        if single:
            with_chain = False
            lookup = keys(predicted, False)

    coords = torch.zeros(len(native), MAX_ATOMS, 3, dtype=torch.float64)
    scored = torch.zeros(len(native), dtype=torch.bool)

    for index, res in enumerate(native.residues):
        key = (
            (native.chain_ids[index], native.res_ids[index], res.resname)
            if with_chain
            else (native.res_ids[index], res.resname)
        )
        other = lookup.get(key)
        if other is None:
            continue
        source = predicted.residues[other]
        source_index = source.index
        # Default to the native coordinate so an atom the prediction omits
        # contributes zero error rather than a spurious one; its chi is only
        # scored if every defining atom was actually present.
        row = torch.from_numpy(res.coords).clone()
        complete = True
        for atom_index, name in enumerate(res.atom_names):
            if name in source_index:
                row[atom_index] = torch.from_numpy(
                    source.coords[source_index[name]]
                )
            elif name not in ("N", "CA", "C", "O"):
                complete = False
        coords[index, : len(res.atom_names)] = row
        scored[index] = complete

    return coords, scored


def evaluate(casp: int, method: str, backbone: str = "casp_all"):
    stems = targets_for(casp, method, backbone)
    pairs = []
    skipped = []
    for stem in stems:
        try:
            # repair_columns because these are outputs from six different
            # programs: SCWRL4's files leave the B-factor column blank, which a
            # strict parse rejects even though the coordinates are fine.
            native = load_protein(
                RESULTS / "native" / backbone / f"{stem}.pdb", repair_columns=True
            )
            predicted = load_protein(
                RESULTS / method / backbone / f"{stem}.pdb", repair_columns=True
            )
        except Exception as exc:  # malformed or unreadable file
            skipped.append((stem, f"{type(exc).__name__}: {exc}"))
            continue

        topo, native_coords = feature_topology(native)
        pred_coords, scored = align_prediction(native, predicted)
        if not bool(scored.any()):
            skipped.append(
                (stem, f"no residues matched ({len(native)} vs {len(predicted)})")
            )
            continue
        pairs.append((pred_coords, native_coords, topo, scored))
    return (benchmark_table(pairs) if pairs else None), len(stems), skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casp", type=int, default=13, choices=[13, 14])
    parser.add_argument("--methods", default="")
    parser.add_argument("--backbone", default="casp_all")
    parser.add_argument("--verbose-skips", action="store_true")
    args = parser.parse_args()

    if not RESULTS.is_dir():
        print(f"{RESULTS} missing -- run scripts/fetch_benchmark.py first")
        return 1

    methods = (
        args.methods.split(",") if args.methods else list(REFERENCE[args.casp])
    )

    print(f"CASP{args.casp}, backbone set '{args.backbone}'")
    print(f"targets numbered T{CASP_RANGES[args.casp][0]}-T{CASP_RANGES[args.casp][1]}\n")

    printed_header = False
    for method in methods:
        row, n_candidates, skipped = evaluate(args.casp, method, args.backbone)
        if row is None:
            print(f"{method}: no usable structures ({n_candidates} candidates)")
            for stem, why in skipped[: 10 if args.verbose_skips else 2]:
                print(f"     {stem}: {why}")
            continue
        if not printed_header:
            print(row.header())
            printed_header = True
        print(row.row(f"{method} (ours)"))

        if method in REFERENCE[args.casp]:
            mae, acc, rmsd = REFERENCE[args.casp][method]
            from aanca.metrics import BenchmarkRow

            published = BenchmarkRow(
                n_structures=0,
                n_residues=0,
                chi_mae=mae,
                accuracy={"all": acc[0] / 100, "core": acc[1] / 100, "surface": acc[2] / 100},
                rmsd={"all": rmsd[0], "core": rmsd[1], "surface": rmsd[2]},
            )
            print(published.row("  published"))
            deltas = [
                abs(a - b) for a, b in zip(row.chi_mae, mae) if a == a
            ]
            worst = max(deltas) if deltas else float("nan")
            print(
                f"  -> {row.n_structures}/{n_candidates} structures, "
                f"{row.n_residues} residues, worst chi MAE gap {worst:.2f} deg"
            )
        if skipped:
            print(f"  skipped {len(skipped)}")
            if args.verbose_skips:
                for stem, why in skipped[:10]:
                    print(f"     {stem}: {why}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
