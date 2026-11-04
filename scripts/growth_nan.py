"""Name the module that first emits a non-finite value in a growth step.

    python scripts/growth_nan.py runs/g1/last.pt --seeds 16

g3 died at **step 0** with `6912 of 6912` identity logits non-finite and the
rollout's own guard reporting that the incoming **coordinates were finite**. The
checkpoint is clean too: `runs/g1/last.pt` holds **0** non-finite parameter
tensors at update 8000. So finite weights and finite coordinates produce an
entirely non-finite output, which means the non-finiteness is created inside the
forward pass -- and every logit being bad points at a shared reduction rather
than a local edge case, since one infinity entering a sum or a normalisation
poisons everything downstream of it.

This is `scripts/find_nan.py`'s method applied to a growth step: hook every leaf
module, run **one** step, and report the first module whose output is not finite,
with the magnitudes on the way in. `find_nan.py` itself cannot be used -- it is
built around the packing trainer's pool and no-op term and has no growth path.

**The seed count is the one configuration difference that matters.** g2 ran 1,375
updates from this checkpoint at 4 seeds; g3 failed immediately at 16. Four seeds
is the control, so run both: a module that is finite at 4 and infinite at 16 is
the answer, and one that fails at both means the difference lies elsewhere and
g2's survival needs its own explanation.

A likely family, recorded so the result can confirm or kill it rather than being
rationalised afterwards: `park_coords` places dormant residues at
`PARK_SPACING * index`, which for a 128-slot crop is up to 63,500 Å. That is
finite, and a squared distance from it is 4e9 which is also finite in float32 --
but a sum of squares over many such pairs, or anything cubic, is not. More seeds
means more alive residues and a different masking pattern, so if the parked
coordinates are reaching a reduction they should not, the seed count would change
whether it overflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.growth import GrowthConfig, growth_links, seed_alive
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    initial_identity,
    initial_state,
    step_once,
    update_mask,
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


def summarise(value) -> str:
    if torch.is_tensor(value):
        if not value.is_floating_point():
            return f"{tuple(value.shape)} {value.dtype}"
        finite = torch.isfinite(value)
        bad = int((~finite).sum())
        peak = float(value[finite].abs().max()) if bool(finite.any()) else float("nan")
        return f"{tuple(value.shape)} max|x|={peak:.3e} nonfinite={bad}"
    if isinstance(value, (tuple, list)):
        return " | ".join(summarise(v) for v in value[:3])
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    state = ckpt.load(Path(args.checkpoint), map_location="cpu")
    stored = state.get("args") or {}
    weights = ckpt.rule_weights(state)
    split = args.split or str(stored.get("split", "train"))
    pool = on_disk(read_manifest(MANIFEST)[split])
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size,
        device="cpu", cache_limit=max(len(pool), 1),
    )
    index = (list(state.get("eval_indices") or [args.index]))[args.index]

    import random

    topo, native, _c, _u = dataset.load_crop(index, rng=random.Random(0))
    links = growth_links(topo)
    print(
        f"{args.checkpoint}, update {state.get('update', '?')}, split {split}, "
        f"crop index {index}, {topo.n_residues} residues\n"
        f"checkpoint non-finite tensors: "
        f"{sum(1 for v in weights.values() if not torch.isfinite(v).all())}\n"
    )

    for seeds in args.seeds:
        rule = AANCARule(
            growth=True,
            birth=str(stored.get("birth", "cartesian")),
            identity=True,
            identity_keep=float(stored.get("identity_keep", 2.0)),
        )
        rule.load_state_dict(weights)
        rule.eval()

        first: list[str] = []
        handles = []

        def make_hook(name):
            def hook(_module, inputs, output):
                ok = True
                if torch.is_tensor(output):
                    ok = bool(torch.isfinite(output).all())
                elif isinstance(output, (tuple, list)):
                    ok = all(
                        bool(torch.isfinite(o).all())
                        for o in output
                        if torch.is_tensor(o) and o.is_floating_point()
                    )
                if not ok and not first:
                    first.append(
                        f"{name}\n      in : {summarise(inputs)}\n"
                        f"      out: {summarise(output)}"
                    )
            return hook

        for name, module in rule.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(make_hook(name)))

        config = RolloutConfig(
            steps=1, truncate=1, k=args.k, blind=True,
            identity_start=str(stored.get("identity_start", "masked")),
            growth=GrowthConfig(
                seeds=seeds, birth=str(stored.get("birth", "cartesian"))
            ),
        )
        generator = torch.Generator().manual_seed(0)
        alive = seed_alive(
            topo.n_residues, seeds, generator, native.device, links=links
        )
        start_identity = initial_identity(
            topo, native, config.identity_start, generator, alive=alive
        )
        st = initial_state(native, rule, identity=start_identity, alive=alive)
        coords_ok = bool(torch.isfinite(st.coords).all())
        peak = float(st.coords[torch.isfinite(st.coords)].abs().max())

        mask = update_mask(
            topo.n_residues, config.update_probability, native.device, generator
        )
        failure = None
        try:
            with torch.no_grad():
                st2, _mu, _lk = step_once(
                    rule, st, topo, config, mask, generator=generator
                )
            logits_ok = (
                bool(torch.isfinite(st2.identity_logits).all())
                if st2.identity_logits is not None
                else True
            )
        except Exception as exc:
            logits_ok = False
            failure = f"{type(exc).__name__}: {str(exc)[:200]}"
        for h in handles:
            h.remove()

        print(f"  seeds {seeds}:")
        print(f"    alive {int(alive.sum())}/{topo.n_residues}, "
              f"start coords finite={coords_ok} max|x|={peak:.3e}")
        print(f"    identity logits finite after one step: {logits_ok}")
        if failure:
            print(f"    raised: {failure}")
        if first:
            print(f"    FIRST non-finite module output:\n      {first[0]}")
        else:
            print("    no module emitted a non-finite output")
    print(
        "\n  A module finite at 4 seeds and non-finite at 16 is the answer. One\n"
        "  failing at both means the seed count is not the difference and g2's\n"
        "  1,375 surviving updates from this same checkpoint need another\n"
        "  explanation. Neither failing means a single step is not enough to\n"
        "  reproduce it and the trigger is in the training loop rather than the\n"
        "  forward pass."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
