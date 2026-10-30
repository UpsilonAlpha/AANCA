"""Score a trained rule on CASP13/CASP14, in the published table's layout.

    python scripts/benchmark.py runs/v5_native/last.pt --casp 13
    python scripts/benchmark.py runs/v5_native/last.pt --casp 14 --steps 200

This is tier 1 of `HANDOFF.md` §4: the table that says whether the method is
competitive with existing packers. ``scripts/validate_benchmark.py`` is its
companion and should be run first -- it scores the *baselines' own* predictions
with our metrics and recovers their published table, which is what makes a row
printed here comparable rather than merely adjacent.

**The task.** Every chi of every residue is randomised, which is the packing
problem as the literature poses it: the backbone is given, the side chains are
not. ``--kappa 0`` is the uniform draw; a larger value starts nearer the native
answer and is a different, easier question.

**The target set** is every CASP target in range that *all* the published
methods also covered, so our average is over the same structures theirs is. A
target one baseline failed to produce is dropped for everyone.

**Rollout length is a free parameter here and is not in training.** The rule is
a fixed point, so ``--steps`` can exceed anything it was trained at; running
the benchmark at several lengths is the tier-2 stability claim and the tier-1
number at once. 100 steps is the default because the training rollout is
64--96.

On a card this is minutes; on CPU it is hours, so run it as a short GPU job
rather than on a login node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.casp import OTHER_PROTOCOLS, REFERENCE, targets_for
from aanca.data import von_mises_deltas
from aanca.metrics import BenchmarkRow, benchmark_table
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.structure import load_protein
from aanca.torch_geometry import apply_chi_deltas
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "external" / "attnpacker" / "extracted" / "standard_results"


def published_row(mae, accuracy, rmsd) -> BenchmarkRow:
    return BenchmarkRow(
        n_structures=0,
        n_residues=0,
        chi_mae=list(mae),
        accuracy={
            "all": accuracy[0] / 100,
            "core": accuracy[1] / 100,
            "surface": accuracy[2] / 100,
        },
        rmsd={"all": rmsd[0], "core": rmsd[1], "surface": rmsd[2]},
    )


def pack(rule, native_coords, topo, config, seed: int):
    """Randomise every chi, then roll the rule out and return where it lands."""
    deltas = von_mises_deltas(
        topo.torsion.chi_mask, 0.0, generator=torch.Generator().manual_seed(seed)
    )
    start = apply_chi_deltas(native_coords, deltas, topo.torsion)
    result = rollout(
        rule,
        initial_state(start, rule),
        topo,
        config,
        generator=torch.Generator().manual_seed(seed),
        record=False,
    )
    return result.state.coords


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--casp", type=int, default=13, choices=[13, 14])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backbone", default="casp_all")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=0, help="first N targets only, for a smoke test"
    )
    args = parser.parse_args()

    if not RESULTS.is_dir():
        print(f"{RESULTS} missing -- run scripts/fetch_benchmark.py first")
        return 1

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    state = ckpt.load(args.checkpoint, map_location=device)
    stored = state.get("args") or {}
    rule = AANCARule().to(device)
    rule.load_state_dict(ckpt.rule_weights(state))
    rule.eval()
    config = RolloutConfig(
        steps=args.steps,
        k=stored.get("k", 64),
        cutoff=10.0,
        # Provably the same graph to within a zero-weight edge at the cutoff
        # (``scripts/check_neighbours.py``), and about five times quicker.
        device_neighbours=device != "cpu",
    )

    # Only the directories that exist in the archive constrain the target set;
    # ``diffpack`` has no directory, so its published row covers whatever
    # DiffPack's authors ran on and is compared to on that understanding.
    covered = [m for m in REFERENCE[args.casp] if (RESULTS / m / args.backbone).is_dir()]
    stems = targets_for(RESULTS, args.casp, covered, args.backbone)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        print(f"no CASP{args.casp} targets covered by all of {covered}")
        return 1

    print(
        f"{args.checkpoint} at update {state['update']}, CASP{args.casp}, "
        f"{len(stems)} targets, {args.steps} steps, every chi randomised, {device}"
    )

    pairs, skipped = [], []
    with torch.no_grad():
        for index, stem in enumerate(stems):
            try:
                native = load_protein(
                    RESULTS / "native" / args.backbone / f"{stem}.pdb",
                    repair_columns=True,
                )
                topo, native_coords = feature_topology(native, device=device)
            except Exception as exc:  # malformed or unreadable file
                skipped.append((stem, f"{type(exc).__name__}: {exc}"))
                continue
            predicted = pack(rule, native_coords, topo, config, args.seed + index)
            pairs.append((predicted.cpu(), native_coords.cpu(), topo))

    if not pairs:
        print("no usable targets")
        for stem, why in skipped[:5]:
            print(f"  {stem}: {why}")
        return 1

    row = benchmark_table(pairs)
    print()
    print(row.header())
    print(row.row("AANCA (ours)"))
    for method, (mae, accuracy, rmsd) in REFERENCE[args.casp].items():
        print(published_row(mae, accuracy, rmsd).row(method))
    if skipped:
        print(f"\n{len(skipped)} targets skipped; first: {skipped[0]}")

    print(
        "\naudited columns: chi MAE, accuracy all/core/surface, RMSD all. The\n"
        "  RMSD core and surface columns do not reproduce from the baselines' own\n"
        "  structures (see aanca.metrics) -- ours there is not their measurement."
    )

    other = OTHER_PROTOCOLS["flowpacker"]
    print(
        f"\nnot in the table above -- flowpacker chi MAE {other[args.casp][0]}, "
        f"RMSD {other[args.casp][2]}.\n  {other['source']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
