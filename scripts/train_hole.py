"""Train the hole head: how accurately can a residue say where one is missing?

    python scripts/train_hole.py --split validation --structures 200 --updates 4000

The decisive measurement for growth as a point cloud, and the cheapest one in
this project: **no rollout.** One message-passing hop, one head, one loss. A
full training run is minutes on a card rather than days, because nothing is
unrolled and there is no sample pool.

What the numbers mean. ``scripts/peptide_shell_probe.py`` measured that a
newborn's N has to land within about 0.6 A for its peptide bond to be
unambiguous, and `scripts/drift_probe.py` measured the untrained growth rule
drifting 1.25 A in a single birth. So:

* **consensus error below ~0.6 A** -- density-driven growth can thread its own
  chain, and the cloud representation is worth the refactor.
* **around 1 A** -- it works for the surrounded case and not at a frontier,
  which is the regime growth is actually in.
* **above 2 A** -- the local neighbourhood does not determine where a residue
  goes, and no amount of machinery downstream will fix that.

Two mask modes, and the gap between them is the real result. ``hole`` deletes
one residue from an otherwise complete structure, which is over-determined and
an upper bound. ``frontier`` deletes a contiguous sequence stretch so the hole
sits on the boundary of a partial structure, which is what growth faces. A rule
that does well on the first and badly on the second has learned interpolation,
not growth.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.hole import (
    centroid_error,
    consensus_error,
    hole_loss,
    hole_predictions,
    make_task,
    placement_error,
)
from aanca.model import AANCARule

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"
RUNS = ROOT / "runs"

MODES = ("hole", "frontier")


def on_disk(specs) -> list:
    """Specs whose coordinates are present, with the path pointed at the file.

    The same helper as ``scripts/train.py``, and it has to redirect the path
    rather than just filter: the manifest records a ``.cif`` path while what is
    downloaded is ``.bcif``, so filtering alone yields specs that pass the
    existence check and then fail to open.
    """
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(rule, dataset, indices, args, seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean per-witness and consensus error, per mask mode, on held-out entries."""
    rule.eval()
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for mode in MODES:
            rng = random.Random(seed)
            per_witness, consensus, witnesses, detected = [], [], [], []
            centroid = []
            for index in indices:
                try:
                    topo, coords, _crop, _used = dataset.load_crop(index, rng=rng)
                except Exception:
                    continue
                if topo.n_residues < 12:
                    continue
                try:
                    task = make_task(
                        coords, topo, mode, rng, args.frontier_fraction
                    )
                except ValueError:
                    continue
                if not bool(task.witnesses.any()):
                    continue
                offset, logit = hole_predictions(
                    rule, coords, topo, task, k=args.k, hops=args.hops
                )
                per_witness.append(float(placement_error(offset, task)))
                consensus.append(
                    float(consensus_error(offset, logit, task, coords, topo))
                )
                centroid.append(float(centroid_error(task, coords, topo)))
                witnesses.append(float(task.witnesses.sum()))
                # Did the most confident residue actually see the hole?
                detected.append(float(task.witnesses[int(logit.argmax())]))
            n = max(len(consensus), 1)
            out[mode] = {
                "per_witness": sum(per_witness) / n,
                "consensus": sum(consensus) / n,
                "centroid": sum(centroid) / n,
                "witnesses": sum(witnesses) / n,
                "top1": sum(detected) / n,
                "n": len(consensus),
            }
    rule.train()
    return out


def report(scores: dict[str, dict[str, float]]) -> str:
    lines = [
        f"  {'mask':<10}{'fused':>9}{'centroid':>11}{'gain':>8}"
        f"{'per-witness':>13}{'witnesses':>11}{'top-1':>8}{'n':>5}"
    ]
    for mode in MODES:
        s = scores.get(mode)
        if not s:
            continue
        gain = s["centroid"] - s["consensus"]
        lines.append(
            f"  {mode:<10}{s['consensus']:>7.2f} A{s['centroid']:>9.2f} A"
            f"{gain:>+8.2f}{s['per_witness']:>11.2f} A"
            f"{s['witnesses']:>11.1f}{s['top1']:>7.1%}{int(s['n']):>5d}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--structures", type=int, default=200)
    parser.add_argument("--eval-structures", type=int, default=40)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        help="message-passing hops before the head reads. 1 is what a single "
        "step of the real rule can see, and the honest default. Because the "
        "hole task has no time, hops here ARE depth -- so how many it needs is "
        "an estimate of how many steps a growth rollout would need before a "
        "newborn could be placed well. A matched sweep at 1 and 3 is the "
        "experiment; see aanca.hole.hole_predictions.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument(
        "--mask",
        choices=("hole", "frontier", "both"),
        default="both",
        help="what to train on. 'both' alternates, which is the honest default: "
        "a rule trained only on surrounded holes learns interpolation.",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="give each residue its own envelope-weighted neighbour count. "
        "**The cheap test of whether burial helps placement.** The aggregation "
        "is a weighted sum so the count is in its magnitude, and the following "
        "LayerNorm removes it exactly -- measured at 8.000 for degrees 10 "
        "through 40. Burial IS that count (core is 20+ CB neighbours within "
        "10 A), and the block correlates with the CB count at r = 0.897. This "
        "task is the right place to test it because it has no rollout, so a "
        "run is minutes rather than hours, and what it scores -- where a "
        "missing residue goes -- is exactly the geometry problem growth fails "
        "at.",
    )
    parser.add_argument("--frontier-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="hole")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    splits = read_manifest(MANIFEST)
    if args.split not in splits:
        print(f"no split {args.split!r}; available: {sorted(splits)}")
        return 1
    available = on_disk(splits[args.split])
    if not available:
        print(f"no coordinates in {PDB} -- run scripts/fetch_structures.py")
        return 1

    wanted = args.structures + args.eval_structures
    pool = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        pool,
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=max(len(pool), 1),
    )
    indices = list(range(len(pool)))
    random.Random(args.seed).shuffle(indices)
    eval_indices = indices[: args.eval_structures]
    train_indices = indices[args.eval_structures :]
    if not train_indices:
        print("no training entries left after holding some out")
        return 1
    print(
        f"device {device}, split {args.split}: {len(train_indices)} training, "
        f"{len(eval_indices)} held out (disjoint)"
    )

    rule = AANCARule(hole=True, density=args.density).to(device)
    optimiser = torch.optim.Adam(rule.parameters(), lr=args.lr)
    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "STOP"

    start_update = 0
    best = float("inf")
    if args.resume and last_path.exists():
        state = ckpt.load(last_path, map_location=device)
        rule.load_state_dict(state["rule"])
        optimiser.load_state_dict(state["optimiser"])
        start_update = int(state["update"])
        best = float(state.get("best", float("inf")))
        ckpt.set_rng_state(state["rng"])
        ckpt.restore_python_rng(rng, state["python_rng"])
        print(f"resumed from {last_path} at update {start_update}")

    # ``args.hops``, not the word "one". The banner read "one hop" while the
    # run used three, because an earlier patch to this line did not match and
    # failed silently -- and a log that misreports its own configuration is
    # worse than one that reports nothing.
    print(
        f"{rule.parameter_count()} parameters, {args.hops} hop(s), no rollout.\n"
        "The baseline to beat is the witnesses' own centroid -- 2.51 A\n"
        "surrounded, 4.76 A at a frontier -- and an untrained head reproduces\n"
        "it exactly. The target is ~0.6 A, the peptide bond's tolerance.\n"
    )
    started = time.perf_counter()
    recent: list[float] = []
    recent_error: list[float] = []

    for update in range(start_update + 1, args.updates + 1):
        mode = args.mask
        if mode == "both":
            mode = MODES[update % len(MODES)]
        index = rng.choice(train_indices)
        try:
            topo, coords, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 12:
            continue
        try:
            task = make_task(coords, topo, mode, rng, args.frontier_fraction)
        except ValueError:
            continue
        if not bool(task.witnesses.any()):
            continue

        offset, logit = hole_predictions(
            rule, coords, topo, task, k=args.k, hops=args.hops
        )
        total, fused, error = hole_loss(offset, logit, task, coords, topo)
        if not torch.isfinite(total):
            continue
        optimiser.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)
        optimiser.step()
        recent.append(float(total.detach()))
        recent_error.append(float(fused))

        if update % max(args.eval_every // 4, 1) == 0:
            done = update - start_update
            print(
                f"  {update:>6}  loss "
                f"{sum(recent) / max(len(recent), 1):>7.3f}  "
                f"fused {sum(recent_error) / max(len(recent_error), 1):>6.2f} A  "
                f"{(time.perf_counter() - started) / max(done, 1):>5.3f} s/update",
                flush=True,
            )
            recent.clear()
            recent_error.clear()

        if update % args.eval_every == 0 or update == args.updates:
            scores = evaluate(rule, dataset, eval_indices, args)
            print(f"\nheld out after {update} updates:")
            print(report(scores) + "\n", flush=True)
            combined = sum(scores[m]["consensus"] for m in MODES if m in scores)
            if combined < best:
                best = combined
                ckpt.save(out_dir / "best.pt", {"state_dict": rule.state_dict()})
                print("  new best, checkpointed\n", flush=True)
            ckpt.save(
                last_path,
                {
                    "rule": rule.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "update": update,
                    "best": best,
                    "rng": ckpt.rng_state(),
                    "python_rng": rng.getstate(),
                    "args": vars(args),
                },
            )

        if stop_file.exists():
            print(f"stopped at update {update} by {stop_file}")
            return ckpt.UNFINISHED
        if args.max_seconds and time.perf_counter() - started > args.max_seconds:
            print(f"stopped at update {update} after {args.max_seconds:g} s")
            return ckpt.UNFINISHED

    minutes = (time.perf_counter() - started) / 60
    print(f"done in {minutes:.1f} min; {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
