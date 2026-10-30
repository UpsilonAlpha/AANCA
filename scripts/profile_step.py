"""Where do the milliseconds in a rollout step go?

    python scripts/profile_step.py [--crop-size 384] [--steps 80] [--device auto]

Measured on the v5 settings, an update costs 2.8 s. Two points -- 0.91 s at 20
rollout steps and 2.80 at 64-96, averaging 80 -- give 0.28 s fixed plus about
31.5 ms per step. That is a very large number for 384 residues through a
200k-parameter model over ~24,000 edges, which is microseconds of arithmetic
on this card, so the step is overhead rather than compute and this says which
overhead.

The suspect is ``torch_graph.select_neighbours``: it runs in NumPy on the CPU,
so every step pays a device transfer, a synchronisation, a spatial search and
some Python set arithmetic, and none of it gets faster on a GPU. The last
timing below is a *lower bound* on a GPU replacement -- distances over the
atom cloud, reduced to residue pairs, then ``topk``. It is not the same
function (no sequence-neighbour forcing, no cutoff bookkeeping), so read it as
"the device can do the geometry in this long", not as a drop-in.

Numbers from here justify a change; they do not authorise one mid-run. The
neighbour path and the batch size both move the random stream, so changing
either would break the comparison v5 exists to make.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from aanca.data import RepackingDataset, read_manifest
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, step_once, update_mask
from aanca.torch_graph import (
    build_graph,
    edge_features,
    node_features,
    select_neighbours,
    select_neighbours_device,
    sequence_links,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"


def on_disk(specs, limit: int):
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
        if len(out) >= limit:
            break
    return out


def timed(label: str, fn, device: str, repeats: int = 20) -> float:
    """Milliseconds per call, synchronising so the GPU timings are real."""
    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    with torch.no_grad():
        fn()
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        sync()
    per_call = (time.perf_counter() - start) / repeats * 1000
    print(f"  {label:<44}{per_call:8.2f} ms")
    return per_call


def gpu_neighbour_lower_bound(coords, topo, k: int):
    """Residue-pair minimum heavy-atom distance, then ``topk``, all on device.

    A lower bound, not a replacement: no sequence-neighbour forcing and no
    cutoff bookkeeping. It answers how long the geometry itself takes when it
    never leaves the card.
    """
    atoms = coords[topo.atom_mask]
    owner = torch.as_tensor(topo.owner, device=coords.device)
    distances = torch.cdist(atoms, atoms)
    n = topo.n_residues
    by_residue = torch.full((n, atoms.shape[0]), float("inf"), device=coords.device)
    by_residue = by_residue.scatter_reduce(
        0, owner.unsqueeze(-1).expand(-1, atoms.shape[0]), distances, "amin"
    )
    pairs = torch.full((n, n), float("inf"), device=coords.device)
    pairs = pairs.scatter_reduce(
        1, owner.unsqueeze(0).expand(n, -1), by_residue, "amin"
    )
    return pairs.topk(min(k, n), dim=-1, largest=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--min-residues",
        type=int,
        default=0,
        help="skip entries smaller than this, so the graph matches training's",
    )
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    splits = read_manifest(MANIFEST)
    dataset = RepackingDataset(
        on_disk(splits[args.split], 24),
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=8,
    )
    # Two filters, both learned the hard way. Entries that are mostly nucleic
    # acid pass the manifest filter and yield almost no protein. And a crop is
    # ``min(structure, crop_size)``, so a small entry silently profiles a
    # smaller graph than training uses -- which matters here, because the cost
    # being measured grows with the residue count.
    topo = native = None
    for index in range(len(dataset.specs)):
        try:
            candidate_topo, candidate_native = dataset.load(index, rng=random.Random(0))
        except ValueError:
            continue
        if candidate_topo.n_residues >= args.min_residues:
            topo, native = candidate_topo, candidate_native
            break
    if topo is None:
        print(
            f"no structure of at least {args.min_residues} residues in the first "
            f"{len(dataset.specs)} of {args.split}"
        )
        return 1
    rule = AANCARule().to(device)
    rule.eval()
    config = RolloutConfig(steps=1, k=args.k, cutoff=10.0)
    state = initial_state(native, rule)
    coords32 = state.coords.to(torch.float32)
    mask = update_mask(topo.n_residues, 1.0, device)

    print(
        f"{topo.n_residues} residues, {int(topo.atom_mask.sum())} side-chain atoms, "
        f"k={args.k}, {device}"
    )
    print(f"{sum(p.numel() for p in rule.parameters())} parameters\n")

    neighbours, edge_mask = select_neighbours(coords32, topo, args.k, 10.0)
    graph = build_graph(coords32, topo, args.k, 10.0)
    node = graph.node_features.to(torch.float32)

    print("one rollout step, by part:")
    t_select = timed(
        "select_neighbours (CPU numpy + transfers)",
        lambda: select_neighbours(coords32, topo, args.k, 10.0),
        device, args.repeats,
    )
    timed(
        "node_features",
        lambda: node_features(coords32, topo), device, args.repeats,
    )
    timed(
        "edge_features",
        lambda: edge_features(coords32, topo, neighbours, edge_mask, 10.0, None, True),
        device, args.repeats,
    )
    timed(
        "rule forward",
        lambda: rule(
            node, state.hidden, graph.edge_features.to(torch.float32),
            graph.neighbours, graph.edge_mask, graph.envelope.to(torch.float32),
        ),
        device, args.repeats,
    )
    t_step = timed(
        "step_once (everything above, plus geometry)",
        lambda: step_once(rule, state, topo, config, mask),
        device, args.repeats,
    )
    t_device = timed(
        "select_neighbours_device (the replacement)",
        lambda: select_neighbours_device(coords32, topo, args.k, 10.0),
        device, args.repeats,
    )
    links = sequence_links(topo, device)
    timed(
        "  the same, sequence links hoisted out",
        lambda: select_neighbours_device(coords32, topo, args.k, 10.0, links=links),
        device, args.repeats,
    )
    t_bound = timed(
        "-- lower bound, geometry only",
        lambda: gpu_neighbour_lower_bound(coords32, topo, args.k),
        device, args.repeats,
    )

    exact = torch.equal(
        select_neighbours(coords32, topo, args.k, 10.0)[0],
        select_neighbours_device(coords32, topo, args.k, 10.0)[0],
    )
    print(f"\nthe two agree on this structure: {exact}")

    print(
        f"select_neighbours is {t_select / t_step:.0%} of a step. At {args.steps} "
        f"steps that is {t_select * args.steps / 1000:.2f} s of an update's "
        f"{t_step * args.steps / 1000:.2f} s."
    )
    print(
        f"The replacement takes {t_device:.2f} ms, "
        f"{t_select / max(t_device, 1e-9):.0f}x faster, which would make a step "
        f"{t_step - t_select + t_device:.2f} ms and an update about "
        f"{(t_step - t_select + t_device) * args.steps / 1000:.2f} s."
    )
    print(f"Geometry alone, with none of the bookkeeping, is {t_bound:.2f} ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
