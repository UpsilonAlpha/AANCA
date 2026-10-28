"""Measure what a BPTT rollout actually costs, rather than estimating it.

    python scripts/measure_rollout_memory.py [--steps 8]

`docs/PLAN_V1.md` §4 budgets rollout memory from tensor arithmetic. Arithmetic
is not a measurement, and the plan's step count depends on this number, so
measure it before committing a Bunya run.

**What is measured, and why not RSS.** Process RSS was tried first and is
useless here: the CPU allocator neither returns pages promptly nor allocates
only what is asked for, so RSS deltas came out negative and contradicted their
own per-step figures. What actually runs a rollout out of memory is the
*activations retained for backward*, so that is measured directly --
``torch.autograd.graph.saved_tensors_hooks`` fires once per tensor saved for the
backward pass, and the accountant below sums their distinct storages. That is
allocator-independent and is the number that scales with step count.

Three configurations, two of which test a claim:

* ``baseline`` -- everything in the graph, features in float64 as the featuriser
  produces them.
* ``detached backbone`` -- the backbone block detached. **The backbone never
  moves** (only side chains rotate, and the backbone is excluded from every
  downstream atom set), so its gradient is identically zero and it does not
  belong in the graph. That is 400 of 575 edge dimensions.
* ``float32 + detached`` -- also featurises from float32 coordinates. Torsion
  arithmetic stays float64 for exactness across hundreds of rotations, but
  distances and RBF expansions do not need it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from aanca.graph import build_graph as np_build_graph
from aanca.model import AANCARule
from aanca.structure import load_protein
from aanca.torch_geometry import apply_chi_deltas
from aanca.torch_graph import build_graph, feature_topology

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
ENTRIES = ("1UBQ", "193L", "4HHB")
K = 64
CUTOFF = 10.0
MB = 1024.0 * 1024.0


class SavedTensorAccountant:
    """Sums the distinct storages torch retains for the backward pass.

    Keyed on ``data_ptr`` so two saved references to one storage count once,
    which is what the allocator sees. Saved tensors are not freed while the
    graph is alive, so a repeated pointer cannot be a different tensor.
    """

    def __init__(self) -> None:
        self.seen: dict[int, int] = {}

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        self.seen[tensor.data_ptr()] = tensor.numel() * tensor.element_size()
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def megabytes(self) -> float:
        return sum(self.seen.values()) / MB


def rollout(
    coords: torch.Tensor,
    topo,
    rule: AANCARule,
    steps: int,
    detach_backbone: bool,
    feature_dtype: torch.dtype,
) -> tuple[list[float], float]:
    """Run ``steps`` keeping the graph. Returns (retained MB after each step, seconds)."""
    hidden = rule.initial_hidden(topo.n_residues, dtype=torch.float32)
    accountant = SavedTensorAccountant()
    trace: list[float] = []
    started = time.perf_counter()

    with torch.autograd.graph.saved_tensors_hooks(
        accountant.pack, accountant.unpack
    ):
        for _ in range(steps):
            graph = build_graph(
                coords.to(feature_dtype),
                topo,
                k=K,
                cutoff=CUTOFF,
                fixed_backbone=detach_backbone,
            )
            edge = graph.edge_features

            mu, _log_kappa, hidden = rule(
                graph.node_features.to(torch.float32),
                hidden,
                edge.to(torch.float32),
                graph.neighbours,
                graph.edge_mask,
                graph.envelope.to(torch.float32),
            )
            coords = apply_chi_deltas(coords, mu.to(coords.dtype), topo.torsion)
            trace.append(accountant.megabytes)

    return trace, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    configs = [
        ("baseline", False, torch.float64),
        ("detached backbone", True, torch.float64),
        ("float32 + detached", True, torch.float32),
    ]

    print(f"Activations retained for backward, k={K}, cutoff={CUTOFF:.0f} A")
    print(f"Measured over {args.steps} steps via saved_tensors_hooks.\n")

    for entry in ENTRIES:
        path = DATA / f"{entry}.pdb"
        if not path.exists():
            print(f"{entry}: missing -- run scripts/fetch_test_structures.py {entry}")
            continue

        protein = load_protein(path)
        reference = np_build_graph(protein, k=K, cutoff=CUTOFF)
        topo, coords = feature_topology(protein)

        print(
            f"{entry}: {len(protein)} residues, "
            f"{len(set(protein.chain_ids))} chain(s), "
            f"mean degree {reference.stats.mean_degree:.1f}"
        )
        print(
            f"  {'config':<21}{'per step':>11}{'ms/step':>9}"
            f"{'30 steps':>11}{'50 steps':>11}"
        )

        for name, detach, dtype in configs:
            torch.manual_seed(0)
            rule = AANCARule()
            trace, seconds = rollout(coords, topo, rule, args.steps, detach, dtype)
            # Marginal cost of a step, from the second half: the first step also
            # allocates one-off buffers that do not repeat.
            half = len(trace) // 2
            per_step = (trace[-1] - trace[half]) / max(len(trace) - 1 - half, 1)
            print(
                f"  {name:<21}{per_step:>8.1f} MB{seconds / args.steps * 1000:>8.0f}"
                f"{per_step * 30:>8.0f} MB{per_step * 50:>8.0f} MB"
            )

        print()

    print("Cost is linear in residue count at fixed k, so these extrapolate by")
    print("residue count. Compare against the GPU's HBM when sizing a batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
