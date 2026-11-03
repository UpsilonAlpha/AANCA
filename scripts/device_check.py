"""Does every code path actually run on this device? Not "does torch see the card".

    python scripts/device_check.py [--device auto] [--identity] [--growth]

**Written because job 31516313 did not need to happen.** The first
inverse-folding run took a GPU allocation and died at update 0 with
``RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'``.
``scripts/bunya_gpu_check.slurm`` had already passed on that card, because it
answers a different question -- whether the ROCm wheel enumerates the GPU --
and 806 local tests had passed, because they all run on CPU, where a CPU
generator against a CPU tensor satisfies the constraint for free.

So this script exercises the paths that *consume a generator*, on whatever
device it is given, and names each one. On a CPU it is nearly a no-op and
passes trivially; on a card it is the cheapest possible rehearsal of a
training run. Thirty seconds against a queue wait and a dead chain.

Every check is a real call into the library rather than a reimplementation --
a check that paraphrases the code it is checking will pass while the code
fails.
"""

from __future__ import annotations

import argparse
import random
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pdb", default="1UBQ")
    parser.add_argument("--identity", action="store_true", default=True)
    parser.add_argument("--growth", action="store_true", default=True)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device {device}")
    if device.type == "cuda":
        print(f"  arch list: {torch.cuda.get_arch_list()}")
        print(f"  name: {torch.cuda.get_device_name(0)}")

    path = RAW / f"{args.pdb}.pdb"
    if not path.exists():
        print(f"{path} missing; run scripts/fetch_test_structures.py")
        return 1

    from aanca.growth import GrowthConfig, growth_objective, seed_alive
    from aanca.hole import hole_loss, hole_predictions, make_task
    from aanca.identity import gumbel_softmax, straight_through
    from aanca.model import AANCARule
    from aanca.rollout import (
        RolloutConfig,
        initial_identity,
        initial_state,
        rollout,
        training_objective,
    )
    from aanca.rng import draw_multinomial, draw_uniform
    from aanca.structure import load_protein
    from aanca.torch_graph import feature_topology

    # **Built on the device, not moved to it.** ``FeatureTopology`` has no
    # ``.to()`` -- only the ``TorsionTopology`` inside it does -- so a
    # ``hasattr`` guard silently left the whole topology on the CPU while the
    # coordinates went to the card, and five of these ten checks failed with
    # "found at least two devices" on the first GPU run. That was this script's
    # bug, not the library's: ``scripts/train.py`` never moves a topology
    # either, it passes ``device`` to ``RepackingDataset`` and the topology is
    # constructed there. Do the same.
    topo, coords_cpu = feature_topology(load_protein(path), device=device)
    coords = coords_cpu.to(device)
    # **A CPU generator on purpose**, which is the whole point: this project
    # draws every random choice on the CPU so a run's stream is a property of
    # its seed and not of the hardware. A device generator would hide exactly
    # the fault this script exists to catch.
    generator = torch.Generator().manual_seed(0)
    rng = random.Random(0)

    checks: list[tuple[str, callable]] = []

    def check(name):
        def wrap(fn):
            checks.append((name, fn))
            return fn
        return wrap

    @check("rng.draw_multinomial")
    def _m():
        p = torch.softmax(torch.randn(7, 54, device=device), dim=-1)
        out = draw_multinomial(p, generator)
        assert out.shape == (7,) and out.device.type == device.type

    @check("rng.draw_uniform")
    def _u():
        out = draw_uniform((5,), generator, torch.float32, device)
        assert out.device.type == device.type

    @check("identity.straight_through sample")
    def _st():
        logits = torch.randn(9, 54, device=device)
        out = straight_through(logits, generator, sample=True)
        assert out.shape == (9, 54) and out.device.type == device.type

    @check("identity.gumbel_softmax")
    def _gs():
        logits = torch.randn(9, 54, device=device)
        out = gumbel_softmax(logits, 1.0, generator)
        assert out.device.type == device.type

    @check("rollout.initial_identity random")
    def _ii():
        out = initial_identity(topo, coords, "random", generator)
        assert out.device.type == device.type

    @check("rollout one packing step")
    def _pack():
        rule = AANCARule().to(device)
        out = rollout(
            rule, initial_state(coords, rule), topo,
            RolloutConfig(steps=2, truncate=2, k=24),
            generator=generator, record=False,
        )
        assert torch.isfinite(out.state.coords).all()

    if args.identity:
        @check("training_objective with identity (what 31516313 hit)")
        def _idobj():
            rule = AANCARule(identity=True, identity_keep=2.0).to(device)
            config = RolloutConfig(
                steps=4, truncate=4, k=24, blind=True,
                identity_sampling="sample",
            )
            terms, _r = training_objective(
                rule, coords, coords, topo, config,
                generator=generator, rng=rng, noop_weight=0.0,
                identity_weight=1.0,
            )
            terms.total.backward()
            assert torch.isfinite(terms.total)

    if args.growth:
        @check("growth_objective, Cartesian birth, identity on")
        def _grow():
            rule = AANCARule(
                growth=True, birth="cartesian", identity=True, identity_keep=2.0
            ).to(device)
            config = RolloutConfig(
                steps=6, truncate=6, k=24, blind=True,
                growth=GrowthConfig(seeds=3, birth="cartesian"),
            )
            terms, _r = growth_objective(
                rule, coords, topo, config, generator, identity_weight=1.0
            )
            terms.total.backward()
            assert torch.isfinite(terms.total)

        @check("growth seed_alive")
        def _seed():
            out = seed_alive(topo.n_residues, 3, generator, device)
            assert out.device.type == device.type and int(out.sum()) == 3

    @check("evaluate: the eval path a run reaches 500 updates in")
    def _eval():
        # **Added after the third device fault in a row.** Each of the first
        # two hid the next: a CPU generator raised before a CPU topology could,
        # and a CPU topology raised before ``rotamers.nearest_well``'s CPU
        # constant could. A run that clears all three still dies at its *first
        # eval* if the eval path has one, and that costs 500 updates rather
        # than nothing -- so the check covers it too, through the real
        # functions.
        from aanca.data import RepackingDataset, read_manifest
        from aanca.evaluate import build_eval_set, evaluate

        manifest = ROOT / "data" / "manifest.json"
        pdb = ROOT / "data" / "pdb"
        if not manifest.exists():
            print("      (skipped: no manifest)")
            return
        specs = []
        for spec in read_manifest(manifest)["validation"]:
            for suffix in (".bcif", ".cif", ".pdb"):
                candidate = pdb / f"{spec.pdb_id}{suffix}"
                if candidate.exists():
                    specs.append(type(spec)(spec.pdb_id, candidate, spec.group))
                    break
            if len(specs) >= 24:
                break
        if not specs:
            print("      (skipped: no coordinates)")
            return
        dataset = RepackingDataset(
            specs, concentrations=(100.0, 0.0), crop_size=96,
            device=device, cache_limit=32,
        )
        # **Pre-filtered, because the dataset's own filters reject entries the
        # manifest accepts.** The first attempt took the first two specs
        # alphabetically and got 10GZ, which has 12 amino acids against a
        # min_residues of 40 -- RCSB's polymer-monomer count includes
        # nucleotides, so a nucleic-acid complex passes the manifest and has
        # almost no protein. That is the dataset behaving correctly and this
        # check asking the wrong thing of it.
        usable = []
        for index in range(len(specs)):
            try:
                dataset.load_crop(index)
            except Exception:
                continue
            usable.append(index)
            if len(usable) >= 2:
                break
        if not usable:
            print("      (skipped: no entry passed the dataset's filters)")
            return
        cases = build_eval_set(dataset, usable, (100.0,), seed=0)
        rule = AANCARule().to(device)
        result = evaluate(
            rule, cases, RolloutConfig(steps=2, truncate=2, k=24)
        )
        assert result is not None

    @check("hole task and loss")
    def _hole():
        rule = AANCARule(hole=True).to(device)
        task = make_task(coords, topo, "hole", rng)
        offset, logit = hole_predictions(rule, coords, topo, task, k=24)
        total, _f, _p = hole_loss(offset, logit, task, coords, topo)
        total.backward()
        assert torch.isfinite(total)

    failures = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print(f"\n{len(checks) - failures}/{len(checks)} paths run on {device}")
    if failures:
        print(
            "A failure here is worth more than a passing test suite: the suite "
            "runs on CPU, where a CPU generator and a CPU tensor agree for free."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
