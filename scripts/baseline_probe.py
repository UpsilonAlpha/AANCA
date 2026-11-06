"""Where does a self-placing rollout start, and can the objective see it?

``birth="none"`` puts every slot alive at step 0 on a baseline conformation --
a fixed reference walk off one native anchor per covalent run -- and asks each
residue to move itself into place. Which baseline is not a detail:

* **It sets the floor the arm has to beat.** The baseline's own RMSD is what a
  rule that learns nothing will report, so a run quoted without it cannot be
  read. ``helix`` is the null model on purpose: it is where untrained growth
  ends up, and the g5 arms measured a rule collapsed onto exactly it.
* **It decides how much of FAPE is alive.** FAPE clamps at 10 A, and
  ``clamp_max`` has zero gradient above the clamp -- so a residue whose
  frame-local error is 40 A contributes a constant and no signal. A baseline
  can therefore be *too far away to train from*, which is a property of the
  objective and not of the architecture, and it would be a waste of a GPU to
  discover it after eight thousand updates. The ``live`` column below is the
  fraction of scored pairs under the clamp.

So this runs no model at all. It builds each baseline on the same held-out
crops a run would see and reports what the rule would be handed.

    python scripts/baseline_probe.py
    python scripts/baseline_probe.py --crop-size 128 --structures 12
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
from aanca.fape import CLAMP, SCALE, fape as fape_loss  # noqa: E402
from aanca.growth import (  # noqa: E402
    BASELINE_MODES,
    baseline_coords,
    count_runs,
    damage_window,
    growth_links,
    peptide_bond_restraint,
    relative_frame_restraint,
    run_anchors,
)
from aanca.fape import backbone_rmsd, backbone_rmsd_on  # noqa: E402

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"


def live_fraction(coords, native, topo, mask) -> float:
    """What share of FAPE's scored pairs sit under the clamp, and so carry gradient.

    Recomputed here rather than returned by ``fape`` because every other caller
    wants the loss and this is the only one that wants to know whether the loss
    can move anything.
    """
    from aanca.fape import _frames, _to_local

    index = torch.nonzero(mask, as_tuple=False).flatten()
    if index.numel() < 2:
        return float("nan")
    rot, origin = _frames(coords, topo, index)
    rot_native, origin_native = _frames(native, topo, index)
    atoms = backbone_atoms(coords, topo)[index, :3].reshape(1, -1, 3)
    atoms_native = backbone_atoms(native, topo)[index, :3].reshape(1, -1, 3)
    local = _to_local(rot, origin, atoms)
    local_native = _to_local(rot_native, origin_native, atoms_native)
    error = (local - local_native).norm(dim=-1)
    return float((error < CLAMP).to(torch.float32).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--pool", type=int, default=400)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--damage", type=int, default=0,
        help=(
            "measure the REGENERATION start state instead: a contiguous window "
            "of this many residues destroyed and every other residue left at "
            "its crystal pose. Reports rmsd_dmg, the window alone superposed "
            "on the intact context, which is the only number comparable with "
            "a --damage training run's. **Run this before reading a "
            "regeneration arm's curve**: the whole-structure baselines are a "
            "different task, and 12.23 against 10.76 is not a comparison"
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[args.split]:
        # The same redirection the trainer does: the manifest records a .cif
        # path while what is downloaded is .bcif, so filtering without
        # redirecting drops every entry.
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
    # The held-out indices a run with this seed would hold out, so the numbers
    # below are comparable with a training log's held-out column.
    order = list(range(len(pool)))
    random.Random(args.seed).shuffle(order)
    indices = order[: args.eval_structures][: args.structures]
    print(f"{len(pool)} structures on disk, {len(indices)} held out at "
          f"crop {args.crop_size}")

    rows: dict[str, list[tuple[float, float, float, float, float]]] = {
        mode: [] for mode in BASELINE_MODES
    }
    runs_seen: list[int] = []
    windows: dict[str, list[float]] = {mode: [] for mode in BASELINE_MODES}
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
            runs_seen.append(count_runs(links))
            for mode in BASELINE_MODES:
                generator = torch.Generator().manual_seed(args.seed + index)
                if args.damage > 0:
                    # Everything outside the window keeps its crystal pose, so
                    # ``anchors`` is the intact context rather than one residue
                    # per covalent run.
                    damaged = damage_window(
                        topo.n_residues, links, args.damage, generator, device
                    )
                    if not bool(damaged.any()):
                        continue
                    anchors = ~damaged
                else:
                    damaged = None
                    anchors = run_anchors(
                        topo.n_residues, links, generator, device
                    )
                start = baseline_coords(
                    native, topo, links, anchors, mode, generator
                )
                if damaged is not None:
                    windows[mode].append(
                        float(
                            backbone_rmsd_on(
                                start, native, topo, alive, damaged
                            )
                        )
                    )
                rows[mode].append((
                    float(backbone_rmsd(start, native, topo, alive)),
                    float(fape_loss(start, native, topo, alive)),
                    live_fraction(start, native, topo, alive),
                    float(peptide_bond_restraint(start, topo, alive, links)),
                    relative_frame_restraint(start, native, topo, alive, links)[1],
                ))

    if not runs_seen:
        print("no crops loaded")
        return 1
    print(f"\n{sum(runs_seen) / len(runs_seen):.1f} covalent runs per crop, so "
          f"that many native anchors")
    print(f"\n{'baseline':<10} {'rmsd':>8} {'fape':>7} {'live':>7} {'bond':>8} "
          f"{'twist':>7}")
    for mode in BASELINE_MODES:
        values = rows[mode]
        if not values:
            continue
        mean = [sum(v[i] for v in values) / len(values) for i in range(5)]
        print(f"{mode:<10} {mean[0]:>8.2f} {mean[1]:>7.3f} {mean[2]:>6.1%} "
              f"{mean[3]:>8.4f} {mean[4]:>7.1f}")

    if args.damage > 0:
        print(
            "\n"
            f"REGENERATION: a {args.damage}-residue window destroyed, the rest "
            "native.\n"
            "rmsd_dmg is the window alone, superposed on the intact context, "
            "which is\n"
            "the only figure comparable with a --damage training run's.\n"
        )
        print(f"{'baseline':<10} {'rmsd_dmg':>9} {'crops':>7}")
        for mode in BASELINE_MODES:
            values = windows[mode]
            if values:
                print(
                    f"{mode:<10} {sum(values) / len(values):>9.2f} "
                    f"{len(values):>7d}"
                )
        print(
            "\n"
            "This is what 'learned nothing' looks like on this task, the way "
            "17.45 A\n"
            "was on whole-structure reconstruction. An arm above its own line "
            "here has\n"
            "made the window worse than leaving it where the walk put it."
        )
    print(f"\nFAPE's clamp is {CLAMP:g} A over a scale of {SCALE:g}, so a loss "
          f"near 1.0 is\nsaturated and `live` is the share of scored pairs that "
          f"can still move.\nA baseline with a small live fraction is not a hard "
          f"problem, it is an\ninvisible one -- the near-diagonal pairs are what "
          f"is left of the gradient.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
