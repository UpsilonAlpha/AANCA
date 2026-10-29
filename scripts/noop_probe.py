"""What does the no-op term actually supervise?

    python scripts/noop_probe.py runs/v4_gpu/last.pt [--structures 3] [--steps 30]

``training_objective`` evaluates the no-op term with a single ``step_once`` on
``initial_state(native_coords, rule)`` -- step 0, hidden state all zeros. Two
things could be true of that, and they call for opposite responses:

* the rule outputs a near-zero delta at the native structure *because it was
  trained to*, in which case the term works and the fixed point is held; or
* the rule outputs a near-zero delta at step 0 **whatever it is shown**,
  because the hidden state carries no information yet, in which case the term
  is supervising a step where the rule is structurally silent and says nothing
  about the fixed point at all.

The two are told apart by one comparison: |mu| at step 0 from a native start
against |mu| at step 0 from a *damaged* start. If a rule that is 56 degrees
away from native also proposes nothing at step 0, the term is vacuous and no
value of ``--noop-weight`` will fix it.

The rest of the output is the trajectory the term never sees: per-step
displacement from native over a native start, mean as well as max, because
``EvalResult.native_drift`` is a max over atoms over structures and one
flipped arginine sets it.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest, von_mises_deltas
from aanca.losses import masked_mean, noop_nll
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, step_once, update_mask
from aanca.torch_geometry import apply_chi_deltas, wrap_to_pi

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


def trace(rule, coords, native, topo, config, seed):
    """Roll out from ``coords``, returning per-step proposal and displacement.

    Written out rather than calling ``rollout`` because the displacement from
    native at *each* step is the quantity in question, and the rollout helper
    records only the proposal.
    """
    generator = torch.Generator().manual_seed(seed)
    state = initial_state(coords, rule)
    # The same mask the loss uses: chi angles of residues with complete
    # perception. Scoring the boundary shell here would measure the crop.
    chi_mask = topo.torsion.chi_mask & topo.supervised_mask.unsqueeze(-1)
    atoms = topo.torsion.atom_mask & topo.supervised_mask.unsqueeze(-1)

    proposals, mean_move, max_move, noops = [], [], [], []
    for _ in range(config.steps):
        mask = update_mask(
            topo.n_residues, config.update_probability, state.coords.device, generator
        )
        state, mu, log_kappa = step_once(rule, state, topo, config, mask)
        proposals.append(float(torch.rad2deg(masked_mean(wrap_to_pi(mu).abs(), chi_mask))))
        noops.append(float(noop_nll(mu, log_kappa, chi_mask, topo.torsion.periodic)))
        moved = (state.coords - native).abs()
        mean_move.append(float(moved[atoms].mean()))
        max_move.append(float(moved[atoms].max()))
    return proposals, mean_move, max_move, noops


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--structures", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--split", default=None, help="defaults to the run's own")
    args = parser.parse_args()

    state = ckpt.load(args.checkpoint, map_location="cpu")
    stored = state.get("args") or {}
    split = args.split or stored.get("split", "validation")
    print(
        f"{args.checkpoint} at update {state['update']}, split {split}, "
        f"noop weight {stored.get('noop_weight')}"
    )

    dataset = RepackingDataset(
        on_disk(read_manifest(MANIFEST)[split]),
        concentrations=(100.0, 20.0, 5.0, 0.0),
        crop_size=stored.get("crop_size", 384),
        device="cpu",
        cache_limit=max(args.structures, 1),
    )
    rule = AANCARule()
    rule.load_state_dict(state["rule"])
    rule.eval()
    config = RolloutConfig(steps=args.steps, k=stored.get("k", 64), cutoff=10.0)

    indices = list(state["eval_indices"])[: args.structures]
    native_first, damaged_first = [], []
    rows = []

    with torch.no_grad():
        for index in indices:
            topo, native = dataset.load(index, rng=random.Random(index))
            deltas = von_mises_deltas(
                topo.torsion.chi_mask, 0.0,
                generator=torch.Generator().manual_seed(index),
            )
            damaged = apply_chi_deltas(native, deltas, topo.torsion)

            proposals, mean_move, max_move, noops = trace(
                rule, native, native, topo, config, index
            )
            from_damage, _, _, _ = trace(
                rule, damaged, native, topo, config, index
            )
            native_first.append(proposals[0])
            damaged_first.append(from_damage[0])
            rows.append((dataset.specs[index].pdb_id, proposals, mean_move,
                         max_move, noops, from_damage))

    print(
        "\nstep 0, the step the no-op term is evaluated at:\n"
        f"  |mu| from native  {sum(native_first) / len(native_first):.3f} deg\n"
        f"  |mu| from kappa 0 {sum(damaged_first) / len(damaged_first):.3f} deg"
    )
    if sum(damaged_first) / len(damaged_first) < 0.5:
        print(
            "  -- the rule proposes nothing at step 0 whatever it is shown, so\n"
            "     the no-op term is supervising a step where it is silent."
        )

    for pdb_id, proposals, mean_move, max_move, noops, from_damage in rows:
        print(f"\n{pdb_id}, from a native start:")
        print("  step        " + "".join(f"{s:7d}" for s in (0, 1, 2, 5, 10, 20, 29)
                                         if s < len(proposals)))
        for label, series, scale in (
            ("|mu| (deg)", proposals, 1.0),
            ("mean move (A)", mean_move, 1.0),
            ("max move (A)", max_move, 1.0),
            ("noop nll", noops, 1.0),
        ):
            cells = "".join(
                f"{series[s] * scale:7.3f}" for s in (0, 1, 2, 5, 10, 20, 29)
                if s < len(series)
            )
            print(f"  {label:<12}" + cells)
        print(f"  |mu| at step 0 from kappa 0 damage: {from_damage[0]:.3f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
