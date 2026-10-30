"""What the chi1 deficit on the CASP table is actually made of.

    python scripts/chi1_histogram.py runs/v5_native/best.pt --casp 13

`scripts/benchmark.py` says the rule reaches 31.6 degrees chi1 MAE where
DiffPack reaches 15.35. That number is an average, and three very different
failures average to it:

**Wrong well.** chi1 is roughly trimodal -- gauche-minus near -60, trans near
180, gauche-plus near +60 -- and the rule picks a real rotamer, just not the
native one. Predictions then sit *on* the wells and the errors cluster near
120 degrees.

**Between wells.** The rule cannot resolve which well and lands at the
circular mean of the plausible answers, which for g- and t is an angle no
side chain occupies. Predictions sit *between* the wells and the errors
cluster near 50-60 degrees. This is what a von Mises NLL against the native
rewards under ambiguity, and it is the hypothesis this script was written to
test.

**Just imprecise.** The well is right and the angle is sloppy. Predictions sit
on the wells, errors are small and unimodal.

The arithmetic already argues against the first: if the 42.6 % of residues
that are correct sit near 8 degrees and every failure were a clean swap to a
neighbouring well, the MAE would be about 80 degrees rather than 31.6. But
arithmetic on summary statistics cannot see a mixture, so this measures the
distributions directly.

**What is printed.** Three histograms -- the signed chi1 error, the predicted
chi1 angle, and the native chi1 angle as the control that says the wells are
where they are claimed to be -- then the decomposition that decides it: how
far predictions sit from the nearest canonical well against how far natives
do, how often the predicted well is the native one, and the error conditional
on the well being right or wrong.

**Proline is excluded from the well analysis** and counted separately. Its
chi1 is constrained by the pyrrolidine ring to about +/-25 degrees, so the
three-well model does not describe it and including it would blur the
control.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.casp import pack_from_random, targets_for
from aanca.casp import REFERENCE
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_geometry import measure_chis, wrap_to_pi
from aanca.torch_graph import feature_topology

RESULTS = Path("data/benchmark")

#: The three canonical chi1 rotamer wells, in degrees. Lovell et al. (2000),
#: the penultimate rotamer library; every packing paper in this line uses the
#: same three names.
WELLS = {"g-": -60.0, "t": 180.0, "g+": 60.0}


def nearest_well(degrees: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(index, signed distance)`` to the nearest of :data:`WELLS`.

    Distance is wrapped, so a prediction at 170 is 10 degrees from trans and
    not 350 from it.
    """
    centres = torch.tensor(list(WELLS.values()), dtype=degrees.dtype)
    offsets = wrap_to_pi(
        torch.deg2rad(degrees.unsqueeze(-1) - centres)
    ).rad2deg()
    index = offsets.abs().argmin(dim=-1)
    return index, offsets.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def histogram(values: torch.Tensor, low: float, high: float, bins: int) -> list[int]:
    """Counts per equal-width bin; values outside the range are clamped in."""
    if values.numel() == 0:
        return [0] * bins
    width = (high - low) / bins
    index = ((values - low) / width).floor().to(torch.long).clamp(0, bins - 1)
    counts = torch.zeros(bins, dtype=torch.long)
    return counts.scatter_add_(0, index, torch.ones_like(index)).tolist()


def draw(counts: list[int], low: float, high: float, width: int = 48) -> str:
    """An ASCII bar chart, because this is read in a job log over a git push."""
    peak = max(counts) or 1
    step = (high - low) / len(counts)
    total = sum(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        left = low + i * step
        bar = "#" * round(width * count / peak)
        lines.append(
            f"  {left:7.0f} to {left + step:7.0f} | {bar:<{width}} "
            f"{count:6d}  {100 * count / total:5.1f}%"
        )
    return "\n".join(lines)


def collect(rule, config, stems, backbone: str, device: str, seed: int):
    """Pack every target and return the chi1 angles, in degrees, as three
    flat tensors: predicted, native, and whether the residue is a proline."""
    predicted, native_out, proline = [], [], []
    with torch.no_grad():
        for index, stem in enumerate(stems):
            try:
                protein = load_protein(
                    RESULTS / "native" / backbone / f"{stem}.pdb",
                    repair_columns=True,
                )
                topo, native_coords = feature_topology(protein, device=device)
            except Exception:  # malformed or unreadable file; benchmark.py reports these
                continue
            packed = pack_from_random(rule, native_coords, topo, config, seed + index)

            torsion = topo.torsion
            has_chi1 = torsion.chi_mask[:, 0].cpu()
            pred_chi1 = measure_chis(packed, torsion)[:, 0].cpu()
            nat_chi1 = measure_chis(native_coords, torsion)[:, 0].cpu()
            is_pro = torch.tensor(
                [name == "PRO" for name in torsion.resnames], dtype=torch.bool
            )

            predicted.append(torch.rad2deg(pred_chi1[has_chi1]))
            native_out.append(torch.rad2deg(nat_chi1[has_chi1]))
            proline.append(is_pro[has_chi1])
    if not predicted:
        return None
    return (
        torch.cat(predicted),
        torch.cat(native_out),
        torch.cat(proline),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--casp", type=int, default=13, choices=[13, 14])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--backbone", default="casp_all")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=36, help="10-degree bins by default")
    parser.add_argument("--limit", type=int, default=0, help="first N targets only")
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
        device_neighbours=device != "cpu",
    )

    # The same target set the published row is averaged over, so the
    # decomposition explains that number and not a neighbouring one.
    covered = [m for m in REFERENCE[args.casp] if (RESULTS / m / args.backbone).is_dir()]
    stems = targets_for(RESULTS, args.casp, covered, args.backbone)
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        print(f"no CASP{args.casp} targets covered by all of {covered}")
        return 1

    print(
        f"{args.checkpoint} at update {state.get('update', '?')}, "
        f"CASP{args.casp}, {len(stems)} targets, {args.steps} steps, "
        f"every chi randomised, {device}"
    )

    gathered = collect(rule, config, stems, args.backbone, device, args.seed)
    if gathered is None:
        print("no usable targets")
        return 1
    pred, native, is_pro = gathered

    keep = ~is_pro
    pred_ok, native_ok = pred[keep], native[keep]
    error = wrap_to_pi(torch.deg2rad(pred_ok - native_ok)).rad2deg()

    print(
        f"\n{keep.sum().item()} residues with a chi1 "
        f"({is_pro.sum().item()} prolines excluded from here on)\n"
    )

    print("signed chi1 error, predicted minus native (deg)")
    print(draw(histogram(error, -180.0, 180.0, args.bins), -180.0, 180.0))

    print("\npredicted chi1 (deg) -- wells at -60, +60, 180")
    print(draw(histogram(pred_ok, -180.0, 180.0, args.bins), -180.0, 180.0))

    print("\nnative chi1 (deg) -- the control: this is what on-the-wells looks like")
    print(draw(histogram(native_ok, -180.0, 180.0, args.bins), -180.0, 180.0))

    pred_well, pred_offset = nearest_well(pred_ok)
    native_well, native_offset = nearest_well(native_ok)
    matched = pred_well == native_well

    print("\n--- what the error is made of ---\n")
    print(
        f"distance to the nearest well : predicted {pred_offset.abs().mean():6.2f} deg "
        f"mean, {pred_offset.abs().median():6.2f} median"
    )
    print(
        f"                               native    {native_offset.abs().mean():6.2f} deg "
        f"mean, {native_offset.abs().median():6.2f} median"
    )
    print(
        f"\npredicted well is the native one : {100 * matched.float().mean():.1f}% "
        f"(chance is about 33%)"
    )
    for name, index in zip(WELLS, range(len(WELLS))):
        share_pred = 100 * (pred_well == index).float().mean()
        share_native = 100 * (native_well == index).float().mean()
        print(
            f"  {name:3s} predicted {share_pred:5.1f}% of residues, "
            f"native {share_native:5.1f}%"
        )

    if matched.any():
        print(
            f"\nwhen the well is right ({matched.sum().item()} residues): "
            f"|error| {error[matched].abs().mean():6.2f} deg mean, "
            f"{100 * (error[matched].abs() <= 20).float().mean():.1f}% within 20"
        )
    if (~matched).any():
        print(
            f"when it is wrong      ({(~matched).sum().item()} residues): "
            f"|error| {error[~matched].abs().mean():6.2f} deg mean"
        )

    print(
        "\nReading it: predictions sitting as close to the wells as the natives do,"
        "\nwith a well-match rate near chance, is well *selection* failing -- the"
        "\nrule picks real rotamers and picks wrong. Predictions sitting markedly"
        "\nfurther from the wells than the natives is *averaging* -- the rule lands"
        "\nbetween wells, at an angle no side chain occupies, which is what a von"
        "\nMises NLL against the native rewards when it cannot resolve the mode."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
