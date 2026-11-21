"""Does the trained rule actually displace the surviving context?

**Written because the number it checks was an inference, not a measurement.**
On 21 Nov `g22_random`'s context displacement was estimated at 1.20 A barely
trained and 2.83 A settled, by decomposing the log's `rmsd` against its
`rmsd_dmg`. That decomposition assumes the two share one superposition, and the
conclusion -- that training teaches the rule to corrupt the reference frame its
own placement is measured against -- was load-bearing enough to register an arm
on (`[g24_frozen]`). So it has to be measured directly before that arm is read,
and arguably before it is run.

    python scripts/context_drift.py runs/g22_random/best.pt --damage 12

**Reports N/CA/C separately from O, and that separation is the whole point.**
A first attempt at this measured all atoms together and found 2.1 A of apparent
context motion in an untrained rollout -- which was one oxygen. `correct_oxygens`
recomputes each O from its backbone every step, so O moves by the difference
between the crystal's O and the idealised reconstruction whether or not the rule
touches the residue. It is identical with the context frozen and unfrozen and it
is not drift. N, CA and C are the atoms that say whether a residue moved.

**An untrained rule is not a null here.** Its move heads emit ~0, so N/CA/C
drift reads exactly 0.0000 and a frozen/unfrozen comparison is vacuous. Run this
on a trained checkpoint or it says nothing.

Also prints the rmsd-decomposition estimate on the same rollouts, so the
inference and the direct measurement can be read against each other: if they
disagree, the decomposition is what to stop quoting.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca.data import RepackingDataset, read_manifest        # noqa: E402
from aanca.fape import backbone_rmsd, backbone_rmsd_on        # noqa: E402
from aanca.growth import (                                    # noqa: E402
    GrowthConfig,
    baseline_coords,
    damage_window,
    growth_links,
)
from aanca.model import AANCARule                             # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from regrowth_dispersion import rule_from_checkpoint          # noqa: E402
from aanca.rollout import RolloutConfig, initial_state, rollout  # noqa: E402

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"

#: ``N CA C O CB`` is the growth atom set; O is index 3 and is reconstructed.
BACKBONE_NCAC = (0, 1, 2)
OXYGEN = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="a trained best.pt. Omit to run the untrained "
                             "control, which should read exactly 0.0000 and "
                             "proves only that the instrument is wired")
    parser.add_argument("--split", default="train")
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--pool", type=int, default=400)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--damage", type=int, default=12)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial", default="random")
    parser.add_argument(
        "--freeze", action="store_true",
        help="run with the context frozen, which must drive N/CA/C drift to "
             "exactly 0. This is g24_frozen's second pre-registered bucket: if "
             "it does not read 0, `frozen` was not applied and the arm is void",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[args.split]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = args.pool + args.eval_structures
    pool = available[:wanted] if len(available) >= wanted else available
    if not pool:
        print("no structures on disk for that split")
        return 1
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size, device=device,
        cache_limit=max(len(pool), 1),
    )
    order = list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    indices = order[: args.eval_structures][: args.structures]

    # **Rebuild the rule AND the rollout config from the run's own recorded
    # args, never from this script's defaults.** `best.pt` carries `args`
    # precisely because a probe that substitutes its own silently measures a
    # different rollout -- it happened twice on 6 Nov and the save site in
    # `train_growth.py` says so. The first version of this script hand-built a
    # RolloutConfig and guessed `initial`, `lab_rotation`, `truncate` and `k`;
    # it also failed outright on the weights, because `best.pt` stores them
    # under `state_dict` and `last.pt` under `rule`.
    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
        rule, run_args, config = rule_from_checkpoint(blob, device)
        label = f"{args.checkpoint} at update {blob.get('update', '?')}"
        initial = run_args.initial
        # The run's OWN held-out crops, so this is measured on the proteins
        # g22_random was judged on rather than on a fresh shuffle.
        recorded = list(blob.get("eval_indices") or [])
        if recorded:
            indices = recorded[: args.structures]
        steps = config.steps
    else:
        import train_growth as tg
        fallback = tg.build_schema().parse_args([])
        fallback.initial = args.initial
        fallback.steps = args.steps
        rule = AANCARule(growth=True, birth="none").to(device)
        config = tg.build_config(fallback, fallback.eval_activation)
        label = "UNTRAINED CONTROL (expect exactly 0.0000; wiring only)"
        initial = args.initial
        steps = args.steps
    rule.eval()

    print(f"checkpoint : {label}")
    print(f"{len(pool)} structures on disk, {len(indices)} held out at crop "
          f"{args.crop_size}, window {args.damage}, start {initial!r}, "
          f"steps {steps}, context {'FROZEN' if args.freeze else 'FREE'}")
    print(f"held-out ids {[pool[i].pdb_id for i in indices]}")
    print()
    print("N/CA/C drift is whether the context MOVED. O is reconstructed by")
    print("correct_oxygens every step and moves whether or not the rule touches")
    print("the residue, so it is reported apart and is not drift.")
    print()
    print(f"{'id':>6}  {'ncac_mean':>9}  {'ncac_max':>8}  {'O_mean':>7}  "
          f"{'rmsd':>6}  {'rmsd_dmg':>8}  {'implied':>7}")

    ncac_means: list[float] = []
    ncac_maxes: list[float] = []
    o_means: list[float] = []
    implied: list[float] = []
    for index in indices:
        try:
            topo, native, _crop, _used = dataset.load_crop(
                index, rng=random.Random(args.seed + index)
            )
        except Exception:
            continue
        if topo.n_residues < args.damage:
            continue
        links = growth_links(topo)
        alive = torch.ones(topo.n_residues, dtype=torch.bool, device=device)
        generator = torch.Generator().manual_seed(args.seed + index)
        damaged = damage_window(
            topo.n_residues, links, args.damage, generator, device
        )
        if not bool(damaged.any()):
            continue
        start = baseline_coords(
            native, topo, links, ~damaged, initial, generator
        )
        begin = initial_state(start, rule, alive=alive)
        with torch.no_grad():
            result = rollout(
                rule, begin, topo, config,
                generator=torch.Generator().manual_seed(args.seed + index),
                record=False,
                frozen=(~damaged) if args.freeze else None,
            )
            coords = result.state.coords
            delta = (coords[~damaged] - native[~damaged]).norm(dim=-1)
            mask = topo.atom_mask[~damaged]
            ncac = torch.cat([
                delta[:, j][mask[:, j]] for j in BACKBONE_NCAC
                if bool(mask[:, j].any())
            ])
            oxy = (
                delta[:, OXYGEN][mask[:, OXYGEN]]
                if bool(mask[:, OXYGEN].any())
                else torch.zeros(1)
            )
            rmsd = float(backbone_rmsd(coords, native, topo, alive))
            dmg = float(backbone_rmsd_on(coords, native, topo, alive, damaged))
        n_all, n_dmg = topo.n_residues, int(damaged.sum())
        resid = (rmsd**2 * n_all - dmg**2 * n_dmg) / max(n_all - n_dmg, 1)
        est = resid**0.5 if resid > 0 else float("nan")
        ncac_means.append(float(ncac.mean()))
        ncac_maxes.append(float(ncac.max()))
        o_means.append(float(oxy.mean()))
        implied.append(est)
        print(f"{pool[index].pdb_id:>6}  {float(ncac.mean()):>9.4f}  "
              f"{float(ncac.max()):>8.3f}  {float(oxy.mean()):>7.4f}  "
              f"{rmsd:>6.2f}  {dmg:>8.2f}  {est:>7.2f}")

    if not ncac_means:
        print("no crop produced a window; nothing measured")
        return 1
    print()
    print(f"over {len(ncac_means)} crops:")
    print(f"  CONTEXT N/CA/C DRIFT   mean {statistics.fmean(ncac_means):6.4f} A"
          f"   worst atom {max(ncac_maxes):6.3f} A   <- the measurement")
    print(f"  O (reconstructed)      mean {statistics.fmean(o_means):6.4f} A"
          f"   not drift")
    print(f"  rmsd-decomposition     mean {statistics.fmean(implied):6.2f} A"
          f"   <- the INFERENCE, for comparison")
    print()
    print("If the decomposition is far above the direct N/CA/C figure then the")
    print("decomposition is measuring something else -- most likely the")
    print("reconstructed O and the window's own contribution leaking through a")
    print("shared superposition -- and it should stop being quoted. The direct")
    print("figure is the one that answers whether the context moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
