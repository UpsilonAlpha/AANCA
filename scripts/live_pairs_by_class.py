"""Which of FAPE's live pairs can actually move the window?

``scripts/baseline_probe.py`` reports one live fraction over every scored pair,
and on a damaged window it reads ~90 %, which was taken as evidence that the
10 A clamp does not bite in regeneration. But a crop is ~116 intact residues
sitting at their crystal poses and a 12-residue window ~13 A out of place, so
the overwhelming majority of pairs are intact-against-intact and *already
correct*. A statistic over all pairs cannot answer a question about the
informative ones.

This splits the same quantity by pair class. FAPE scores every (frame, point)
combination, so each pair has two ends that are independently damaged or
intact:

    intact  -> intact    both ends at crystal pose: error ~0, live, useless
    intact  -> window    the window's atoms read in an anchored frame
    window  -> intact    the context read in a displaced frame
    window  -> window    the window's internal shape

The two cross classes are the ones that carry placement information, and they
are what ``docs/claims/the-fape-clamp-is-what-gates-the-geometry.md`` is about.
If their live fraction is near zero while the total is near 90 %, then
``fape_weight = 0.1`` is demoting a term that was already blind, and the clamp
argument was never answered -- only out-voted by pairs with nothing to fix.

    python scripts/live_pairs_by_class.py --damage 12

**Laptop pools and Bunya pools hold different proteins at the same crop
index** (the cause of the voided 10.77 A baseline), so the held-out ids are
printed. The quantity here is geometric and should travel; the absolute RMSDs
should not be compared with a Bunya arm's.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca.backbone import backbone_atoms  # noqa: E402
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.fape import CLAMP, _frames, _to_local, backbone_rmsd_on  # noqa: E402
from aanca.growth import (  # noqa: E402
    baseline_coords,
    damage_window,
    growth_links,
)

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"

#: The four pair classes, as (frame end, point end) over "is damaged".
CLASSES = (
    ("intact -> intact", False, False),
    ("intact -> window", False, True),
    ("window -> intact", True, False),
    ("window -> window", True, True),
)


def pair_errors(coords, native, topo, mask):
    """``(F, P)`` unclamped frame-local error, and the residue behind each end.

    Mirrors ``baseline_probe.live_fraction`` exactly -- same frames, same
    points, same order -- but returns the error per pair instead of one mean,
    plus the residue index of each axis so the pairs can be classified.

    ``backbone_atoms(...)[index, :3]`` is residue-major with three atoms each,
    so point ``p`` belongs to ``index[p // 3]``.
    """
    index = torch.nonzero(mask, as_tuple=False).flatten()
    if index.numel() < 2:
        return None, None, None
    rot, origin = _frames(coords, topo, index)
    rot_native, origin_native = _frames(native, topo, index)
    atoms = backbone_atoms(coords, topo)[index, :3].reshape(1, -1, 3)
    atoms_native = backbone_atoms(native, topo)[index, :3].reshape(1, -1, 3)
    local = _to_local(rot, origin, atoms)
    local_native = _to_local(rot_native, origin_native, atoms_native)
    error = (local - local_native).norm(dim=-1)[0]
    frame_residue = index
    point_residue = index.repeat_interleave(3)
    return error, frame_residue, point_residue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--pool", type=int, default=400)
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--damage", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="helix")
    parser.add_argument("--device", default="cpu")
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
    print(f"{len(pool)} structures on disk, {len(indices)} held out at "
          f"crop {args.crop_size}, baseline {args.mode}, damage {args.damage}")
    print(f"pool: {len(available)} available; held-out ids "
          f"{[pool[i].pdb_id for i in indices]}")

    # counts[class] = [n_pairs, n_live, sum_error]
    totals = {name: [0, 0, 0.0] for name, _, _ in CLASSES}
    overall = [0, 0]
    rmsds: list[float] = []
    rng = random.Random(args.seed)
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 12:
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
                native, topo, links, ~damaged, args.mode, generator
            )
            rmsds.append(
                float(backbone_rmsd_on(start, native, topo, alive, damaged))
            )

            error, frame_residue, point_residue = pair_errors(
                start, native, topo, alive
            )
            if error is None:
                continue
            frame_dmg = damaged[frame_residue].unsqueeze(1)
            point_dmg = damaged[point_residue].unsqueeze(0)
            live = error < CLAMP
            overall[0] += error.numel()
            overall[1] += int(live.sum())
            for name, want_frame, want_point in CLASSES:
                sel = (frame_dmg == want_frame) & (point_dmg == want_point)
                n = int(sel.sum())
                if n == 0:
                    continue
                totals[name][0] += n
                totals[name][1] += int((live & sel).sum())
                totals[name][2] += float(error[sel].sum())

    if not rmsds:
        print("no crops produced a window")
        return 1

    print()
    print(f"window rmsd_dmg  {sum(rmsds) / len(rmsds):8.2f} A   "
          f"over {len(rmsds)} crops")
    print(f"live, all pairs  {100.0 * overall[1] / overall[0]:8.1f} %   "
          f"({overall[1]:,} of {overall[0]:,})")
    print()
    print(f"{'pair class':<18} {'pairs':>12} {'share':>8} {'live':>8} "
          f"{'mean err':>10}")
    for name, _, _ in CLASSES:
        n, n_live, total_error = totals[name]
        if n == 0:
            continue
        print(f"{name:<18} {n:>12,} {100.0 * n / overall[0]:>7.1f}% "
              f"{100.0 * n_live / n:>7.1f}% {total_error / n:>9.2f} A")

    cross = [0, 0]
    for name, want_frame, want_point in CLASSES:
        if want_frame != want_point:
            cross[0] += totals[name][0]
            cross[1] += totals[name][1]
    if cross[0]:
        print()
        print(f"window-to-intact pairs are {100.0 * cross[0] / overall[0]:.1f} %"
              f" of the total and {100.0 * cross[1] / cross[0]:.1f} % live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
