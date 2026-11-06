"""How far can a self-placing rollout move a residue at all, and how far does it?

Four arms have now frozen at their own starting conformation: `g6_self` and
`g6_noid` at the helix baseline's 17.45 A, `g6_ext` at the extended baseline's
36.73 A, and `g6_open` at 17.45 A with AF2's partial clamp. Three different
hypotheses have been tested and all three failed -- identity starvation, the
initialisation, and the clamp. This probe tests the one that is arithmetic
rather than a guess.

**`AANCARule` bounds every step's translation to 0.3 A and every step's
rotation to 0.1 rad** (`max_translation`, `max_rotation`, enforced by
`_bounded`, which is a tanh on the norm so the bound is a true ceiling). The
rollout is 120 steps. So:

    120 steps x 0.3 A  =  36 A      the most a residue can *ever* move,
                                    and only if all 120 steps push the same way
     12 steps x 0.3 A  =  3.6 A     the most that can move inside the
                                    truncation window, which is all the
                                    gradient ever sees

The helix baseline is 17.45 A from native, so the rule must produce ~58 steps
of coherent motion while being taught by a window that covers 3.6 A of it. The
extended baseline is 36.73 A away, which is **past the 36 A ceiling** -- so
`g6_ext` could not have reached the native in 120 steps with a perfect rule,
and its freeze needs no explanation beyond the arithmetic.

What the arithmetic cannot say is whether a trained rule is using its budget.
Two very different failures look identical in a held-out RMSD:

  * **path length near the ceiling, net displacement near zero** -- the rule is
    moving hard and incoherently, and the fix is about credit assignment over
    steps (a longer truncation window, or a loss on intermediate steps).
  * **path length near zero** -- the rule has learned to ask for nothing, and
    the fix is about why the frame head has no reason to act.

So this reports both, per residue, against the ceiling. It runs on the CPU and
needs no card.

    python scripts/reach_ceiling_probe.py --checkpoint runs/g6_self/last.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.growth import (
    GrowthConfig,
    baseline_coords,
    count_runs,
    growth_links,
    run_anchors,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_identity, initial_state, step_once

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"


def ca_positions(coords: torch.Tensor, topo) -> torch.Tensor:
    """``(R, 3)``: every residue's CA, which is what displacement means here."""
    index = torch.arange(topo.n_residues)
    return coords[index, topo.frame_idx[index, 1]]


def held_out_crop(stored: dict, state: dict):
    """The crop the run was evaluated on, at its own crop size and split."""
    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[str(stored.get("split") or "train")]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = int(stored.get("structures", 400)) + int(
        stored.get("eval_structures", 12)
    )
    pool = available[:wanted] if len(available) >= wanted else available
    if not pool:
        raise SystemExit("no structures on disk for that split")
    dataset = RepackingDataset(
        pool,
        concentrations=(0.0,),
        crop_size=int(stored.get("crop_size") or 128),
        device=torch.device("cpu"),
        cache_limit=max(len(pool), 1),
    )
    indices = list(state.get("eval_indices") or [])
    if not indices:
        order = list(range(len(pool)))
        random.Random(int(stored.get("seed", 0))).shuffle(order)
        indices = order[: int(stored.get("eval_structures", 12))]
    # **Every held-out crop, not the first one.** This probe reported a single
    # crop until 7 Nov, and the number it produced -- 7.67 A travelled against
    # 8.64 A needed, which is the whole evidence for "FAPE supplies a magnitude
    # and not a bearing" -- was therefore k=1. The pMHC binder session made the
    # general point while correcting its own rho: **the reportable quantity is
    # the unit count, not the interval**, and a statistic from too few
    # independent units does not mean what its spread implies. One crop is the
    # degenerate case of that, so the probe now reports the spread across crops
    # and the count beside it.
    crops = []
    rng = random.Random(int(stored.get("seed", 0)))
    for index in indices:
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 12:
            continue
        crops.append((index, topo, native))
    if not crops:
        raise SystemExit("no held-out crop loaded")
    return crops, f"{len(crops)} held-out crops of {len(pool)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--steps", type=int, default=0, help="default: the run's own"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    state = ckpt.load(path, map_location="cpu")
    stored = state.get("args") or {}
    if not stored:
        sibling = path.with_name("last.pt")
        if sibling.exists():
            beside = ckpt.load(sibling, map_location="cpu")
            stored = beside.get("args") or {}
            if not state.get("eval_indices"):
                state["eval_indices"] = beside.get("eval_indices") or []
    if not stored:
        print(
            f"{path} stores no run arguments and {path.with_name('last.pt')} is "
            "missing, so every rollout setting would be this probe's default "
            "rather than the run's. Point --checkpoint at the run's last.pt."
        )
        return 1

    weights = ckpt.rule_weights(state)
    birth = str(stored.get("birth") or "none")
    if birth != "none":
        print(
            f"this run is --birth {birth}, not a self-placing one. The ceiling "
            "argument is about a rollout that must move every residue from a "
            "baseline, so it does not apply to a grown run."
        )
        return 1

    steps = args.steps or int(stored.get("steps") or 120)
    initial = str(stored.get("initial") or "helix")
    crops, source = held_out_crop(stored, state)

    rule = AANCARule(
        growth=True,
        birth="none",
        identity="identity_head.0.weight" in weights,
        morphogen="emit_head.0.weight" in weights,
        identity_keep=float(stored.get("identity_keep", 2.0)),
    )
    rule.load_state_dict(weights)
    rule.eval()

    ceiling = steps * rule.max_translation
    truncated = min(int(stored.get("truncate") or 12), steps)
    window = truncated * rule.max_translation

    rows = []
    for index, topo, native in crops:
        links = growth_links(topo)
        generator = torch.Generator().manual_seed(args.seed + index)
        anchors = run_anchors(topo.n_residues, links, generator, native.device)
        start = baseline_coords(native, topo, links, anchors, initial, generator)
        alive = torch.ones(topo.n_residues, dtype=torch.bool)
        config = RolloutConfig(
            steps=steps,
            truncate=truncated,
            k=int(stored.get("k") or 24),
            blind=bool(stored.get("blind", True)),
            growth=GrowthConfig(birth="none", initial=initial),
        )
        identity = None
        if rule.identity:
            identity = initial_identity(
                topo, str(stored.get("identity_start", "masked")), alive=alive
            )
        rollout_state = initial_state(start, rule, identity=identity, alive=alive)

        begin = ca_positions(start, topo)
        previous = begin
        path_length = torch.zeros(topo.n_residues)
        with torch.no_grad():
            for _ in range(steps):
                rollout_state, _mu, _kappa = step_once(
                    rule, rollout_state, topo, config, alive,
                    generator=generator,
                )
                here = ca_positions(rollout_state.coords, topo)
                path_length += (here - previous).norm(dim=-1)
                previous = here
        net = (previous - begin).norm(dim=-1)
        needed = (begin - ca_positions(native, topo)).norm(dim=-1)
        rows.append((
            index,
            float(path_length.mean()),
            float(net.mean()),
            float(needed.mean()),
            float(net.mean() / path_length.mean().clamp_min(1e-9)),
        ))

    print(f"rule: {path} ({initial} baseline, {steps} steps)")
    print(f"{source}\n")
    print(f"per-step translation cap   {rule.max_translation:g} A")
    print(f"ceiling over {steps:4d} steps     {ceiling:.1f} A   (all steps coherent)")
    print(f"inside the {truncated:2d}-step window  {window:.1f} A   (all the gradient sees)\n")

    print(f"{'crop':>6} {'path':>8} {'net':>8} {'needed':>8} {'net/path':>9}")
    for index, path_len, net_len, need, ratio in rows:
        print(f"{index:>6} {path_len:>8.2f} {net_len:>8.2f} {need:>8.2f} "
              f"{ratio:>9.3f}")
    k = len(rows)
    mean_path = sum(r[1] for r in rows) / k
    mean_net = sum(r[2] for r in rows) / k
    mean_need = sum(r[3] for r in rows) / k
    mean_ratio = sum(r[4] for r in rows) / k
    print(f"\n{'mean':>6} {mean_path:>8.2f} {mean_net:>8.2f} {mean_need:>8.2f} "
          f"{mean_ratio:>9.3f}")
    print(f"{'spread':>6} {min(r[1] for r in rows):>8.2f}"
          f"-{max(r[1] for r in rows):<7.2f}"
          f"{min(r[2] for r in rows):>8.2f}-{max(r[2] for r in rows):<7.2f}")
    print(f"\n  k = {k} independent crops. **Report the unit count, not just "
          f"the mean.**\n"
          "  This probe read a single crop until 7 Nov, so the figure that\n"
          "  carries 'FAPE supplies a magnitude and not a bearing' was k=1.\n"
          "  A spread across crops that straddles the mean says the split\n"
          "  between arms is real; one that does not says the arms differ by\n"
          "  less than the crops do.\n")
    print(
        "  Read net/path first. Near 1 means the rule moves in a straight\n"
        "  line and is simply out of budget; near 0 means it spends the\n"
        "  budget going nowhere, a credit-assignment problem rather than a\n"
        "  reach problem. A path length near zero means the rule asks for\n"
        "  nothing at all, and neither the ceiling nor the window stops it."
    )
    if mean_need > ceiling:
        print(
            f"\n  ** The baseline is {mean_need:.1f} A from native on average "
            f"and the\n  ceiling is {ceiling:.1f} A, so this rollout cannot "
            f"reach the native\n  structure in {steps} steps however good the "
            "rule is. The arm's\n  freeze needs no further explanation than "
            "that."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
