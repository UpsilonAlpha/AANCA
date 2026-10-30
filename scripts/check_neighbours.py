"""Do the two neighbour selections agree across the training distribution?

    python scripts/check_neighbours.py [--structures 60] [--perturbations 4]

``tests/test_torch_graph.py`` pins the agreement on two deposited structures.
That is enough to say the implementations mean the same thing; it is not
enough to switch a running experiment onto the new one, because the risk is
not a logic error -- it is a residue pair whose minimum heavy-atom distance
sits within a float ulp of the cutoff, where ``cKDTree`` on the host and
``cdist`` on the card can disagree about whether it is inside. One flip is one
edge in one step, and the run would simply be a slightly different run with
nothing to say so.

So this asks the question at the scale the switch implies: real crops from the
training split, on the device, in the off-distribution geometry a rollout
actually visits, counting every graph and every edge. It also counts pairs
within a hair of the cutoff, which is the exposure -- if that number is large
and the disagreement is still zero, the agreement is not luck.

Run it on a card. On CPU the comparison is the same but the arithmetic is not
the arithmetic the run would use.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from aanca.data import RepackingDataset, read_manifest
from aanca.torch_geometry import apply_chi_deltas
from aanca.torch_graph import select_neighbours, select_neighbours_device

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"

#: A pair this close to the cutoff is one an implementation could put on the
#: other side of it. Angstrom.
HAIR = 1e-4


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


def randomised(coords, topo, generator):
    """A rollout-like state: every chi moved, so the graph is off-distribution."""
    deltas = torch.zeros_like(topo.torsion.chi_mask, dtype=coords.dtype)
    count = int(topo.torsion.chi_mask.sum())
    draws = torch.rand(count, generator=generator, dtype=torch.float64)
    deltas[topo.torsion.chi_mask] = (
        (draws * 2 - 1) * torch.pi
    ).to(coords.dtype).to(coords.device)
    return apply_chi_deltas(coords, deltas, topo.torsion)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=60)
    parser.add_argument("--perturbations", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    dataset = RepackingDataset(
        on_disk(read_manifest(MANIFEST)[args.split], args.structures * 2),
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=4,
    )

    graphs = edges = disagreements = near = 0
    residues = 0
    failures: list[str] = []
    generator = torch.Generator().manual_seed(args.seed)

    for index in range(len(dataset.specs)):
        if graphs >= args.structures * (1 + args.perturbations):
            break
        try:
            topo, native = dataset.load(index, rng=random.Random(index))
        except ValueError:
            continue

        states = [native] + [
            randomised(native, topo, generator) for _ in range(args.perturbations)
        ]
        for step, coords in enumerate(states):
            coords32 = coords.to(torch.float32)
            expected = select_neighbours(coords32, topo, args.k, args.cutoff)
            actual = select_neighbours_device(coords32, topo, args.k, args.cutoff)
            graphs += 1
            residues += topo.n_residues
            edges += int(expected[1].sum())
            if not (
                torch.equal(actual[0], expected[0])
                and torch.equal(actual[1], expected[1])
            ):
                disagreements += 1
                differing = int((actual[0] != expected[0]).any(dim=-1).sum())
                failures.append(
                    f"{dataset.specs[index].pdb_id} state {step}: "
                    f"{differing} of {topo.n_residues} rows differ"
                )

            # Exposure: how many pairs sit close enough to the cutoff that the
            # two could have put them on opposite sides of it.
            atoms = coords32.detach()[topo.atom_mask]
            owner = torch.as_tensor(topo.owner, device=device, dtype=torch.long)
            n_atoms = atoms.shape[0]
            distances = torch.cdist(
                atoms, atoms, compute_mode="donot_use_mm_for_euclid_dist"
            )
            by_residue = torch.full(
                (topo.n_residues, n_atoms), float("inf"), device=device
            ).scatter_reduce(
                0, owner.unsqueeze(-1).expand(-1, n_atoms), distances, "amin"
            )
            pair = torch.full(
                (topo.n_residues, topo.n_residues), float("inf"), device=device
            ).scatter_reduce(
                1, owner.unsqueeze(0).expand(topo.n_residues, -1), by_residue, "amin"
            )
            near += int(((pair - args.cutoff).abs() < HAIR).sum() // 2)

        print(
            f"  {dataset.specs[index].pdb_id}: {graphs} graphs, "
            f"{disagreements} disagreements",
            flush=True,
        )

    print(
        f"\n{graphs} graphs over {residues} residues and {edges} edges, {device}\n"
        f"pairs within {HAIR} A of the cutoff: {near}\n"
        f"disagreements: {disagreements}"
    )
    for line in failures[:10]:
        print(f"  {line}")
    return 1 if disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
