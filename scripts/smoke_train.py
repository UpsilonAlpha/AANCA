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
* large entries crop, and the supervised fraction of a crop is what the
  standalone measurement said it would be;
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


def _failure_class(why: str) -> str:
    """Collapse a failure message to its class, so counts per class are visible.

    Deliberately crude: the point is that every failure lands in some named
    bucket with a count beside it, not that the buckets are elegant. An
    unrecognised message keeps its exception type, which is enough to notice
    that something new is happening.
    """
    for marker, label in (
        ("under the min_residues limit", "too few amino acids (min_residues)"),
        ("over the max_residues limit", "too large (max_residues)"),
        ("more than MAX_ATOMS", "residue wider than the atom padding"),
        ("is not a bond", "bad side-chain topology"),
    ):
        if marker in why:
            return label
    return why.split(":", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=int, default=40)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=512)
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
    dataset = RepackingDataset(
        specs, concentrations=(100.0, 20.0, 5.0, 0.0), crop_size=args.crop_size
    )
    loaded, failures, oversized = [], [], []
    sizes, entry_sizes, supervised = [], [], []
    cropped = 0
    for index in range(len(dataset)):
        try:
            topo, _coords, indices, _seed = dataset.load_crop(index)
        except Exception as exc:
            failures.append((specs[index].pdb_id, f"{type(exc).__name__}: {exc}"))
            continue
        widest = max((len(n) for n in topo.torsion.atom_names), default=0)
        if widest > MAX_ATOMS:
            oversized.append((specs[index].pdb_id, widest))
        loaded.append(index)
        sizes.append(topo.n_residues)
        entry_sizes.append(len(dataset.protein(index)))
        supervised.append(float(topo.supervised_mask.to(torch.float64).mean()))
        cropped += indices is not None

    print(f"\nparsed {len(loaded)}/{len(specs)}; {len(failures)} failed")
    # Grouped by class, not the first five. Printing a sample is how a whole
    # failure class hides: an earlier run reported its 154 failures as "all the
    # size guards" because the five shown happened to be size guards, while
    # eight were really a padding-width bug that silently dropped every
    # structure whose chain ended in TRP.
    classes: dict[str, list[str]] = {}
    for pdb_id, why in failures:
        classes.setdefault(_failure_class(why), []).append(pdb_id)
    for label, ids in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        print(f"   {len(ids):>4}  {label}  e.g. {', '.join(ids[:3])}")
    if oversized:
        print(f"  PADDING EXCEEDED on {len(oversized)}: {oversized[:5]}")
    if sizes:
        ordered = sorted(sizes)
        whole = sorted(entry_sizes)
        print(
            f"  entries: min {whole[0]}, median {whole[len(whole) // 2]}, "
            f"max {whole[-1]} residues"
        )
        print(
            f"  samples: min {ordered[0]}, median {ordered[len(ordered) // 2]}, "
            f"max {ordered[-1]} residues after cropping"
        )
        # The cost of cropping, stated rather than assumed: the boundary shell
        # is in the graph as context for the interior but carries no gradient.
        print(
            f"  cropped {cropped}/{len(loaded)} entries; mean supervised "
            f"fraction {sum(supervised) / len(supervised):.1%}"
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
    print(f"  {'#':<4}{'pdb':<7}{'res':>5}{'sup':>6}{'kappa':>7}{'loss':>10}"
          f"{'recon':>9}{'noop':>9}{'s/step':>8}")

    history = []
    for update in range(args.updates):
        index = rng.choice(loaded)
        # Pass the rng so a large entry gives a different crop each time, which
        # is what makes cropping augmentation rather than truncation.
        sample = dataset.sample(index, generator=generator, rng=rng)
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

        pool.insert(index, result.state, crop_seed=sample.crop_seed)
        total = float(terms.total.detach())
        history.append((sample.concentration, total))
        print(
            f"  {update:<4}{sample.pdb_id:<7}{sample.topology.n_residues:>5}"
            f"{sample.supervised_fraction:>6.0%}"
            f"{sample.concentration:>7.0f}{total:>10.3f}"
            f"{float(terms.reconstruction):>9.3f}{float(terms.noop):>9.3f}"
            f"{elapsed:>8.2f}"
        )

    print(f"\ngradient norm on the last update: {float(grad_norm):.3f}")
    print(f"pool holds {len(pool)} states")

    # Report by damage level, not first-versus-last. Loss is not comparable
    # across levels -- uniform damage costs about three times mild noise -- so a
    # first-to-last comparison says only which levels those two samples drew.
    # An earlier version of this summary did exactly that and read as a 3x
    # regression when nothing had regressed.
    by_level: dict[float, list[float]] = {}
    for concentration, value in history:
        by_level.setdefault(concentration, []).append(value)
    print("\nmean loss by damage level (kappa 0 is uniform, higher is milder):")
    for level in sorted(by_level, reverse=True):
        values = by_level[level]
        print(
            f"  kappa {level:>5.0f}  n={len(values):<3} mean {sum(values) / len(values):.3f}"
        )
    print(
        f"\n{args.updates} updates on {len(loaded)} structures says nothing about "
        "whether this learns. It says the plumbing is connected."
    )

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
