"""What does a trivial rule achieve on the hole task? The number training must beat.

    python scripts/hole_baseline.py [--structures 60]

Built after watching `scripts/train_hole.py` move from 6.93 to 6.60 A over two
thousand updates and realising there was no baseline to say whether that was
slow learning or a hard task. Three closed-form predictors, no parameters:

* **zero** -- "the hole is at my own CA". This is what the zero-initialised head
  emits, and it is also the best possible *constant* answer, because the hole
  lies in every direction from one witness or another, so the mean offset is
  nearly zero. That is why the task has no easy bias win and why training from
  zero looks flat at the start.
* **anti-centroid** -- the hole is opposite the centroid of my neighbours, at
  the median neighbour distance. One line, and it uses the only signal that
  survives a sum over neighbours: a neighbourhood with a gap has its mass on
  the other side.
* **least-occupied direction** -- of a set of candidate directions, take the one
  whose nearest neighbour is furthest away, at the median neighbour distance.
  A crude void detector, and closer to what the head is being asked to learn.

If the learned head cannot beat anti-centroid it has learned nothing useful. If
anti-centroid is already near the ~0.6 A the peptide bond needs, the task is
easier than it looks and the architecture is the problem.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from aanca.backbone import backbone_atoms, frame_from_triad
from aanca.data import RepackingDataset, read_manifest
from aanca.hole import SEE_RADIUS, make_task

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
MODES = ("hole", "frontier")


def on_disk(specs):
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def fibonacci_sphere(n: int, dtype=torch.float64) -> torch.Tensor:
    """``(n, 3)`` roughly uniform directions, for the void scan."""
    index = torch.arange(n, dtype=dtype)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * index / max(n - 1, 1)
    radius = (1.0 - z * z).clamp_min(0.0).sqrt()
    theta = phi * index
    return torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), z], -1)


def baselines(coords, topo, task, directions) -> dict[str, float]:
    """Error in Angstrom of each closed-form predictor, averaged over witnesses."""
    backbone = backbone_atoms(coords, topo)
    rotation = frame_from_triad(backbone[:, 0], backbone[:, 1], backbone[:, 2])
    ca = backbone[:, 1]
    index = torch.nonzero(task.witnesses, as_tuple=False).flatten()
    alive = task.alive.clone()
    truth = task.target  # already in each witness's frame

    out: dict[str, list[float]] = {"zero": [], "anti_centroid": [], "least_occupied": []}
    for i in index.tolist():
        target = truth[i]
        out["zero"].append(float(target.norm()))

        # Neighbours of i that exist, in i's own frame.
        offset = ca - ca[i]
        distance = offset.norm(dim=-1)
        near = alive & (distance < SEE_RADIUS) & (distance > 1e-6)
        if not bool(near.any()):
            out["anti_centroid"].append(float(target.norm()))
            out["least_occupied"].append(float(target.norm()))
            continue
        local = torch.einsum("ab,nb->na", rotation[i].transpose(-1, -2), offset[near])
        radius = float(local.norm(dim=-1).median())

        centroid = local.mean(dim=0)
        norm = float(centroid.norm())
        guess = (
            -centroid / norm * radius
            if norm > 1e-6
            else torch.zeros(3, dtype=local.dtype)
        )
        out["anti_centroid"].append(float((guess - target).norm()))

        # Of the candidate directions, the one whose closest neighbour -- by
        # angle -- is furthest away in angle. That is the emptiest cone.
        unit = local / local.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cosine = directions.to(unit.dtype) @ unit.transpose(0, 1)
        emptiest = directions[int(cosine.max(dim=1).values.argmin())].to(local.dtype)
        out["least_occupied"].append(float((emptiest * radius - target).norm()))

    return {name: sum(v) / max(len(v), 1) for name, v in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=60)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--directions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    splits = read_manifest(MANIFEST)
    pool = on_disk(splits["validation"])[: args.structures]
    if not pool:
        print(f"no coordinates in {PDB}")
        return 1
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size,
        device="cpu", cache_limit=max(len(pool), 1),
    )
    directions = fibonacci_sphere(args.directions)

    print(f"{len(pool)} structures, mean error in Angstrom over witnesses\n")
    print(f"  {'mask':<10}{'zero':>9}{'anti-centroid':>15}{'least-occupied':>16}{'n':>5}")
    for mode in MODES:
        rng = random.Random(args.seed)
        totals: dict[str, list[float]] = {}
        for i in range(len(pool)):
            try:
                topo, coords, _c, _u = dataset.load_crop(i, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 12:
                continue
            try:
                task = make_task(coords, topo, mode, rng)
            except ValueError:
                continue
            if not bool(task.witnesses.any()):
                continue
            for name, value in baselines(coords, topo, task, directions).items():
                totals.setdefault(name, []).append(value)
        if not totals:
            continue
        n = len(next(iter(totals.values())))
        print(
            f"  {mode:<10}{sum(totals['zero']) / n:>7.2f} A"
            f"{sum(totals['anti_centroid']) / n:>13.2f} A"
            f"{sum(totals['least_occupied']) / n:>14.2f} A{n:>5d}"
        )

    print(
        "\n  zero is what a zero-initialised head emits, and the best constant\n"
        "  answer there is -- the hole lies in every direction from some\n"
        "  witness, so the mean offset is near zero and there is no bias win.\n"
        "  A learned head that does not beat anti-centroid has learned nothing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
