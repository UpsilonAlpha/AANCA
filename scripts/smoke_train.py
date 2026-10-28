"""End-to-end smoke test: real structures -> dataset -> rollout -> objective -> step.

    python scripts/smoke_train.py [--structures 40] [--steps 8]

Every component is tested in isolation, but until this runs, the path from a
downloaded coordinate file to a parameter update has never executed on real data.
That is where integration problems live, and they tend to be unglamorous: a dtype
mismatch, a padding assumption, a residue with more heavy atoms than
``MAX_ATOMS``, a structure whose chains confuse the loader.

So this deliberately does **not** try to train anything. It takes a handful of
structures from the validation split, runs a few optimiser steps, and reports
what broke and what the numbers look like. Loss going down over a dozen steps on
forty structures means nothing about whether the idea works; loss being finite
and every parameter receiving a gradient means the plumbing is connected.

What it checks that a unit test cannot:

* real entries parse, including the awkward ones, and failures are counted rather
  than fatal;
* no residue exceeds the padding width;
* the objective is finite on real geometry at several damage levels;
* every parameter receives a gradient;
* the native structure really is a near-fixed-point for the no-op term;
* wall time and memory per step at realistic sizes.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from aanca.data import RepackingDataset, read_manifest
from aanca.metrics import benchmark_table
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, SamplePool, initial_state, rollout, training_objective
from aanca.torch_geometry import MAX_ATOMS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"


def usable(specs, limit: int):
    """Specs whose coordinate file is actually on disk."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=40)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if not MANIFEST.exists():
        print("no manifest -- run scripts/build_training_manifest.py")
        return 1

    specs = usable(read_manifest(MANIFEST)["validation"], args.structures)
    if not specs:
        print(f"no coordinates in {PDB} -- run scripts/fetch_structures.py")
        return 1
    print(f"{len(specs)} structures with coordinates on disk")

    # Stage 1: can we parse them at all, and does the padding width hold?
    dataset = RepackingDataset(specs, concentrations=(100.0, 20.0, 5.0, 0.0))
    loaded, failures, oversized = [], [], []
    sizes = []
    for index in range(len(dataset)):
        try:
            topo, _coords = dataset.load(index)
        except Exception as exc:
            failures.append((specs[index].pdb_id, f"{type(exc).__name__}: {exc}"))
            continue
        widest = max((len(n) for n in topo.torsion.atom_names), default=0)
        if widest > MAX_ATOMS:
            oversized.append((specs[index].pdb_id, widest))
        loaded.append(index)
        sizes.append(topo.n_residues)

    print(f"\nparsed {len(loaded)}/{len(specs)}; {len(failures)} failed")
    for pdb_id, why in failures[:5]:
        print(f"   {pdb_id}: {why}")
    if oversized:
        print(f"  PADDING EXCEEDED on {len(oversized)}: {oversized[:5]}")
    if sizes:
        sizes.sort()
        print(
            f"  residues: min {sizes[0]}, median {sizes[len(sizes) // 2]}, "
            f"max {sizes[-1]}"
        )
    if not loaded:
        return 1

    # Stage 2: the objective on real geometry, at every damage level.
    rule = AANCARule()
    config = RolloutConfig(steps=args.steps, truncate=4, k=64, cutoff=10.0)
    optimiser = torch.optim.Adam(rule.parameters(), lr=args.lr)
    pool = SamplePool(capacity=64, seed=0)
    rng = random.Random(0)
    generator = torch.Generator().manual_seed(0)

    print(f"\n{args.updates} updates, {args.steps} steps each")
    print(f"  {'#':<4}{'pdb':<7}{'res':>5}{'kappa':>7}{'loss':>10}{'recon':>9}"
          f"{'noop':>9}{'s/step':>8}")

    history = []
    for update in range(args.updates):
        index = rng.choice(loaded)
        sample = dataset.sample(index, generator=generator)
        start = time.perf_counter()
        try:
            terms, result = training_objective(
                rule,
                sample.damaged,
                sample.native,
                sample.topology,
                config,
                generator=generator,
                rng=rng,
            )
            optimiser.zero_grad()
            terms.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(rule.parameters(), 10.0)
            optimiser.step()
        except Exception as exc:
            print(f"  {update:<4}{sample.pdb_id:<7} FAILED {type(exc).__name__}: {exc}")
            return 1
        elapsed = (time.perf_counter() - start) / args.steps

        missing = [n for n, p in rule.named_parameters() if p.grad is None]
        if missing:
            print(f"  no gradient for {len(missing)} parameters: {missing[:3]}")
            return 1
        if not torch.isfinite(terms.total):
            print(f"  non-finite loss at update {update}")
            return 1

        pool.insert(index, result.state)
        history.append(float(terms.total))
        print(
            f"  {update:<4}{sample.pdb_id:<7}{sample.topology.n_residues:>5}"
            f"{sample.concentration:>7.0f}{float(terms.total):>10.3f}"
            f"{float(terms.reconstruction):>9.3f}{float(terms.noop):>9.3f}"
            f"{elapsed:>8.2f}"
        )

    print(f"\ngradient norm on the last update: {float(grad_norm):.3f}")
    print(f"pool holds {len(pool)} states")
    print(f"loss first {history[0]:.3f} -> last {history[-1]:.3f} "
          f"(12 updates on {len(loaded)} structures proves nothing about learning)")

    # Stage 3: is the native structure a fixed point for an untrained rule, and
    # does the benchmark table run on real output?
    fresh = AANCARule()
    topo, native = dataset.load(loaded[0])
    result = rollout(fresh, initial_state(native, fresh), topo, config)
    drift = float((result.state.coords.detach() - native).abs().max())
    print(f"\nuntrained rule on a native structure: max drift {drift:.2e} A "
          f"over {config.steps} steps")

    table = benchmark_table([(result.state.coords.detach(), native, topo)])
    print(f"benchmark table on that rollout: {table.row('identity')}")
    print("\nPlumbing connected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
