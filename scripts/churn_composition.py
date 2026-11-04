"""When the identity head flips a label, does it flip to a similar residue?

    python scripts/churn_composition.py runs/v8c_if/last.pt [--steps 64]

v8c_if's label churn plateaus at **0.12** -- twelve per cent of positions change
their committed label every step, indefinitely, after 20,000 updates
(`scripts/score_identity.py`). Whether that is a problem depends entirely on
*what* it flips to. A flip from LEU to ILE is the head being uncertain between
two choices the structure barely distinguishes, and the energy surface there is
nearly flat; a flip from LEU to ASP is not. The first is a sampler doing its
job, the second is a rule that has not learned what the position wants.

**The null is the whole measurement.** "Seventy per cent of flips stay in
class" sounds like a result and means nothing on its own: with six classes of
unequal size, a *random* flip stays in class a good fraction of the time
anyway, and the more so if the head's output distribution is concentrated. So
every statistic here is reported against a null drawn from the head's **own**
marginal over the labels it actually commits, which is the distribution a
flip would come from if the choice ignored the position entirely. The claim is
the gap, not the level.

Reported three ways, because "similar" means more than one thing and the user's
question named several: chemical class, side-chain size, and the specific pairs.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca import residue_constants as rc
from aanca.data import RepackingDataset, read_manifest
from aanca.identity import index_resnames, type_index
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    initial_identity,
    initial_state,
    rollout,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"


def on_disk(specs):
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    state = ckpt.load(Path(args.checkpoint), map_location="cpu")
    stored = state.get("args") or {}
    if "eval_indices" not in state:
        print("no eval_indices in this checkpoint; pass the run's last.pt")
        return 1
    indices = list(state["eval_indices"])[: args.structures]

    split = stored.get("split", "validation")
    pool = on_disk(read_manifest(MANIFEST)[split])
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=int(stored.get("crop_size", 384)),
        device="cpu", cache_limit=max(len(pool), 1),
    )

    rule = AANCARule(
        identity=True, identity_keep=float(stored.get("identity_keep", 8.0))
    )
    weights = Path(args.weights) if args.weights else Path(args.checkpoint)
    rule.load_state_dict(ckpt.rule_weights(ckpt.load(weights, map_location="cpu")))
    rule.eval()

    config = RolloutConfig(
        steps=args.steps, truncate=args.steps, k=int(stored.get("k", 64)),
        blind=True,
        identity_start=str(stored.get("identity_start", "masked")),
        identity_sampling=str(stored.get("identity_sampling", "sample")),
        update_schedule=str(stored.get("update_schedule", "bernoulli")),
        coupling_radius=float(stored.get("coupling_radius", 5.0)),
    )

    flips: Counter = Counter()
    committed: Counter = Counter()
    rng = random.Random(args.seed)

    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _c, _u = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            supervised = topo.supervised_mask
            start = initial_identity(topo, native, config.identity_start)
            state_ = initial_state(native, rule, identity=start)
            previous = state_.identity.argmax(-1)
            from aanca.rollout import step_once, update_mask
            from aanca.update_schedule import coupling_graph, random_independent_set

            generator = torch.Generator().manual_seed(args.seed)
            for _step in range(args.steps):
                if config.update_schedule == "independent":
                    mask = random_independent_set(
                        coupling_graph(state_.coords, topo, config.coupling_radius),
                        generator,
                    )
                else:
                    mask = update_mask(
                        topo.n_residues, config.update_probability,
                        state_.coords.device, generator,
                    )
                state_, _mu, _lk = step_once(
                    rule, state_, topo, config, mask, generator=generator
                )
                current = state_.identity.argmax(-1)
                changed = (current != previous) & supervised
                for position in torch.nonzero(changed, as_tuple=False).flatten().tolist():
                    old = index_resnames(previous[position : position + 1])[0]
                    new = index_resnames(current[position : position + 1])[0]
                    flips[(old, new)] += 1
                # The head's own marginal over what it commits, which is the
                # null a flip would be drawn from if it ignored the position.
                for position in torch.nonzero(supervised, as_tuple=False).flatten().tolist():
                    committed[index_resnames(current[position : position + 1])[0]] += 1
                previous = current

    total = sum(flips.values())
    if not total:
        print("no flips observed")
        return 1

    # The null: pairs drawn independently from the committed marginal.
    weight = {name: n / sum(committed.values()) for name, n in committed.items()}
    null_same_class = sum(
        weight[a] * weight[b]
        for a in weight for b in weight
        if a != b and rc.chemical_class(a) == rc.chemical_class(b)
    ) / max(sum(weight[a] * weight[b] for a in weight for b in weight if a != b), 1e-12)
    null_size = sum(
        weight[a] * weight[b] * abs(rc.sidechain_size(a) - rc.sidechain_size(b))
        for a in weight for b in weight if a != b
    ) / max(sum(weight[a] * weight[b] for a in weight for b in weight if a != b), 1e-12)

    same_class = sum(n for (a, b), n in flips.items() if rc.chemical_class(a) == rc.chemical_class(b))
    size_delta = sum(
        n * abs(rc.sidechain_size(a) - rc.sidechain_size(b))
        for (a, b), n in flips.items()
    ) / total

    print(f"{weights}, {len(indices)} held-out structures, {args.steps} steps")
    print(f"{total} label flips observed\n")
    print(f"  {'statistic':<26}{'observed':>11}{'null':>10}{'ratio':>8}")
    print(
        f"  {'flip stays in class':<26}{same_class / total:>10.1%}"
        f"{null_same_class:>10.1%}{(same_class / total) / max(null_same_class, 1e-9):>8.2f}"
    )
    print(
        f"  {'mean |size change|':<26}{size_delta:>10.2f}{null_size:>10.2f}"
        f"{size_delta / max(null_size, 1e-9):>8.2f}"
    )
    print(
        "\n  The null draws both ends of a flip independently from the head's own\n"
        "  marginal over committed labels, so it is what a flip would look like if\n"
        "  the choice ignored the position. A ratio above 1 on class, or below 1 on\n"
        "  size, is the head preferring similar residues; a ratio of 1 is a flip\n"
        "  that carries no information about the site."
    )

    # **Is a residue a hub, or is it just common?** The top-flip list is
    # dominated by a handful of residues, which looks like the head using a
    # low-commitment default and bouncing off it. But a residue it commits
    # often heads the list for that reason alone, so participation has to be
    # read against the same marginal as everything else here. A ratio near 1
    # means "appears in flips as often as it is used"; well above 1 means the
    # label is unstable *per use*, which is what a hub actually is.
    endpoints: Counter = Counter()
    for (first, second), n in flips.items():
        endpoints[first] += n
        endpoints[second] += n
    total_ends = max(sum(endpoints.values()), 1)
    print("\n  flip participation per use (ratio > 1 = unstable per use):")
    print(f"    {'residue':<9}{'endpoints':>10}{'share':>8}{'committed':>11}{'ratio':>8}")
    rows = []
    for name, n in endpoints.items():
        share = n / total_ends
        used = weight.get(name, 0.0)
        rows.append((share / used if used else float("inf"), name, n, share, used))
    for ratio, name, n, share, used in sorted(rows, reverse=True)[:10]:
        print(f"    {name:<9}{n:>10}{share:>8.1%}{used:>11.1%}{ratio:>8.2f}")

    print("\n  most frequent flips:")
    for (a, b), n in flips.most_common(12):
        same = "same" if rc.chemical_class(a) == rc.chemical_class(b) else "CROSS"
        print(
            f"    {a} -> {b:<4} {n:>6}  {same:>5}  "
            f"{rc.chemical_class(a)}/{rc.chemical_class(b)}  "
            f"size {rc.sidechain_size(a)}->{rc.sidechain_size(b)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
