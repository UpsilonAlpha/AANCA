"""Reproduce the published packing numbers from the published predictions.

    python scripts/validate_benchmark.py [--casp 13|14] [--methods scwrl,faspr,...]

AttnPacker's Zenodo release (record 7713779, ``scp_results.zip``) ships the
native test structures *and* every baseline's predicted structures. That makes a
far stronger check available than any self-consistency test: compute our metrics
on **their** predictions and see whether we recover **their** published table.

If the numbers match, `metrics.py` implements the same thing they do and the
tier-1 comparison is auditable. If they do not, the discrepancy is ours and
needs finding before a single training run.

Reference values live in ``aanca.casp``: DiffPack (NeurIPS 2023) Tables 1 and 2,
which is also where the burial definition and the 20-degree accuracy threshold
come from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aanca.casp import CASP_RANGES, REFERENCE, targets_for
from aanca.metrics import benchmark_table
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "external" / "attnpacker" / "extracted" / "standard_results"


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
    stems = targets_for(RESULTS, casp, method, backbone)
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
