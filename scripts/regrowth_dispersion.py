"""Regrow one window many times: how many ways can it sit?

    python scripts/regrowth_dispersion.py runs/g20_orient
    python scripts/regrowth_dispersion.py runs/g20_orient --members 24 --amplitudes 0,5,15,45

**The measurement the project exists to make, and the first one whose product is
a distribution rather than a number.** If low entropic cost is what makes a
binder both good and predictable, the quantity to measure is local
conformational freedom -- and a rule that can destroy and regrow a window
repeatedly can be asked for it directly. See
``docs/claims/destruction-and-regrowth-yields-a-local-statistical-potential.md``.

**Why the amplitude sweep IS the control.** In a ``birth = none``,
``identity_weight = 0`` rollout there are only two stochastic sources: which
window is damaged, and the Bernoulli update mask. Hold the window fixed, as an
ensemble must, and the mask is all that is left -- so "N rollouts differing only
in the generator" would measure the update schedule, not the protein. Mask order
is not conformational multiplicity.

So the members here differ in their **initial condition**: each damaged residue
is rigidly perturbed about its own CA by a random rotation of up to
``amplitude`` degrees. Amplitude 0 is the unperturbed helix baseline, where the
members differ by update mask alone -- **that row is the noise floor** and every
other row has to beat it to mean anything.

And the sweep answers the question a single N cannot:

  * settled spread SATURATES as the start spread grows -> the rule has basins,
    and the spread is a property of the window's environment.
  * settled spread TRACKS the start spread -> the rule is carrying its input
    through, and this measures the perturbation rather than the protein.

**Rolled to convergence, not to a fixed horizon**, on the rule's **own**
successive coordinates: the chunk-to-chunk CA movement of the damaged window
falling below ``--tol``. Never on RMSD to native -- a stopping rule that reads
the target puts the answer back inside the rollout, which is what ``native_edges``
was removed for. Runs that hit ``--max-steps`` are reported as unconverged
rather than silently truncated, because some graph-CA rules oscillate instead of
settling. ``steps`` is reported beside the spread: a rule that never moves
converges immediately, so settling time means nothing on its own.

CPU only and no GPU allocation needed. The checkpoint supplies its own
``eval_indices``, so this scores the crops the run was judged on -- a laptop
pool and a Bunya pool hold different proteins at the same index, so run it where
the run ran.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aanca.backbone import backbone_atoms  # noqa: E402
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.fape import backbone_rmsd_on  # noqa: E402
from aanca.growth import (  # noqa: E402
    baseline_coords,
    damage_window,
    growth_links,
    peptide_bond_restraint,
)
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import initial_state, rollout  # noqa: E402

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"


def rule_from_checkpoint(ckpt, device):
    """Rebuild the rule the run trained, from the run's own recorded args.

    Imports ``train_growth``'s own helpers rather than reconstructing the
    namespace by hand: a hand-built config that drifts from the trainer's is
    exactly the failure ``tests/test_config.py`` exists to catch, and it would
    be silent here.
    """
    import train_growth as tg

    args = tg.build_schema().parse_args([])
    for key, value in (ckpt.get("args") or {}).items():
        setattr(args, key, value)
    rule = AANCARule(
        growth=True,
        birth=args.birth,
        morphogen=args.morphogen,
        activation_init=args.activation_init,
        max_rotation=args.max_rotation,
        max_translation=args.max_translation,
        identity=tg.identity_track(args),
        identity_keep=args.identity_keep,
    ).to(device)
    # **The two checkpoint files name the weights differently.** `last.pt`
    # stores them under `rule` and `best.pt` under `state_dict` -- two writers,
    # two keys, which cost a run here before it was noticed. Accept either and
    # say so rather than assuming.
    weights = ckpt.get("rule")
    if weights is None:
        weights = ckpt.get("state_dict")
    if weights is None:
        raise KeyError(
            "checkpoint holds neither 'rule' nor 'state_dict'; keys are "
            f"{sorted(ckpt)}"
        )
    rule.load_state_dict(weights)
    rule.eval()
    config = tg.build_config(args, args.eval_activation)
    return rule, args, config


def perturb(coords, topo, damaged, amplitude_deg, generator):
    """Rigidly rotate each damaged residue about the WINDOW'S centroid.

    **About the centroid, not about each residue's own CA.** The first version
    of this rotated every residue about its own CA, which leaves the CA exactly
    where it was -- so the CA-RMSF readout this script reports came back
    identically 0.000 at every amplitude, and the perturbation did not touch the
    quantity being measured at all. Caught by the smoke test before any run.
    Rotating about a shared centre moves each residue bodily, by roughly
    ``r * theta`` for ``r`` its distance from the centre, so a 15 deg amplitude
    on a 12-residue window is a couple of Angstrom.

    **Only real atoms.** ``coords`` carries padded atom slots, and rotating
    those about a distant centre produced a 22 A displacement in the same smoke
    test. Masked by ``topo.atom_mask``.

    Torsion-space jitter would keep peptide geometry ideal and is the more
    physical perturbation, but ``baseline_coords`` builds from fixed torsion
    references and exposes no scale, and reimplementing the walk here would put
    a second copy of it in the tree. The cost of rigid perturbation is a
    strained start, so the starting ``bond`` is reported -- the distortion is
    visible rather than assumed harmless, and amplitude 0 touches nothing.
    """
    if amplitude_deg <= 0:
        return coords
    out = coords.clone()
    index = torch.nonzero(damaged, as_tuple=False).flatten()
    if index.numel() == 0:
        return out
    real = topo.atom_mask.to(torch.bool)
    ca = backbone_atoms(coords, topo)[:, 1]
    centre = ca[damaged].mean(dim=0)
    axis = torch.randn(index.numel(), 3, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angle = (
        torch.rand(index.numel(), generator=generator)
        * (amplitude_deg * math.pi / 180.0)
    )
    eye = torch.eye(3, device=coords.device, dtype=coords.dtype)
    for row, residue in enumerate(index.tolist()):
        k = axis[row].to(coords.device, coords.dtype)
        theta = angle[row].to(coords.device, coords.dtype)
        kx = torch.tensor(
            [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
            device=coords.device, dtype=coords.dtype,
        )
        rot = eye + torch.sin(theta) * kx + (1.0 - torch.cos(theta)) * (kx @ kx)
        keep = real[residue]
        moved = (coords[residue][keep] - centre) @ rot.T + centre
        slots = out[residue].clone()
        slots[keep] = moved
        out[residue] = slots
    return out


def roll_to_convergence(rule, start, topo, config, damaged, generator,
                        chunk, max_steps, tol):
    """Chunked rollout until the damaged window stops moving. Self-referential.

    Returns ``(coords, steps, converged)``. The criterion is the CA movement of
    the damaged residues between the ends of successive chunks -- the rule's own
    coordinates, never a distance to native.
    """
    import dataclasses

    cfg = dataclasses.replace(config, steps=chunk, truncate=chunk)

    alive = torch.ones(topo.n_residues, dtype=torch.bool, device=start.device)
    state = initial_state(start, rule, alive=alive)
    previous = backbone_atoms(state.coords, topo)[:, 1][damaged].clone()
    steps = 0
    with torch.no_grad():
        while steps < max_steps:
            result = rollout(
                rule, state, topo, cfg, generator=generator, record=False
            )
            state = result.state
            steps += chunk
            now = backbone_atoms(state.coords, topo)[:, 1][damaged]
            moved = float((now - previous).norm(dim=-1).pow(2).mean().sqrt())
            previous = now.clone()
            if moved < tol:
                return state.coords, steps, True
    return state.coords, steps, False


def spread(poses, topo, damaged):
    """Per-residue CA RMSF over an ensemble, in Angstrom.

    ``poses`` is a list of full coordinate tensors. The mean pose is taken
    without superposition on purpose: the intact context is identical in every
    member and pins the frame already, so superposing would discard exactly the
    placement differences this is trying to see.
    """
    cas = torch.stack([backbone_atoms(p, topo)[:, 1][damaged] for p in poses])
    mean = cas.mean(dim=0, keepdim=True)
    per_residue = (cas - mean).pow(2).sum(-1).mean(dim=0).sqrt()
    return per_residue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="a run directory, e.g. runs/g20_orient")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument(
        "--amplitudes", default="0,5,15,45",
        help="degrees of per-residue rigid perturbation; 0 is the noise floor",
    )
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--tol", type=float, default=0.02,
        help="Angstrom of window CA movement per chunk below which it has settled",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run = Path(args.run)
    ckpt = torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)
    rule, run_args, config = rule_from_checkpoint(ckpt, device)
    indices = list(ckpt.get("eval_indices") or [])
    if not indices:
        print("checkpoint carries no eval_indices")
        return 1
    indices = indices[: args.structures]
    amplitudes = [float(a) for a in args.amplitudes.split(",")]

    print(f"run        : {run}  [{args.checkpoint}] at update {ckpt.get('update')}")
    print(f"settings   : damage {run_args.damage}, crop {run_args.crop_size}, "
          f"update_schedule {config.update_schedule}, "
          f"update_probability {config.update_probability}")
    print(f"eval crops : {indices}")
    print(f"ensemble   : {args.members} members per amplitude, amplitudes "
          f"{amplitudes} deg, chunk {args.chunk}, cap {args.max_steps}, "
          f"tol {args.tol} A")

    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[run_args.split]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = run_args.structures + run_args.eval_structures
    pool = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=run_args.crop_size,
        device=device, cache_limit=max(len(pool), 1),
    )
    print(f"pool       : {len(available)} available, ids "
          f"{[pool[i].pdb_id for i in indices if i < len(pool)]}")

    rows = {a: {"start": [], "settled": [], "steps": [], "conv": 0, "n": 0,
                "dmg": [], "bond0": []} for a in amplitudes}
    per_residue_by_amp = {a: [] for a in amplitudes}
    rng = random.Random(args.seed)

    for index in indices:
        if index >= len(pool):
            continue
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        links = growth_links(topo)
        gen = torch.Generator().manual_seed(args.seed + index)
        damaged = damage_window(
            topo.n_residues, links, run_args.damage, gen, device
        )
        if not bool(damaged.any()):
            continue
        base = baseline_coords(native, topo, links, ~damaged, run_args.initial, gen)
        alive = torch.ones(topo.n_residues, dtype=torch.bool, device=device)

        for amp in amplitudes:
            starts, poses, steps_list = [], [], []
            for member in range(args.members):
                g = torch.Generator().manual_seed(
                    10_000 * (member + 1) + 7 * index + int(amp)
                )
                start = perturb(base, topo, damaged, amp, g)
                starts.append(start)
                coords, steps, converged = roll_to_convergence(
                    rule, start, topo, config, damaged, g,
                    args.chunk, args.max_steps, args.tol,
                )
                poses.append(coords)
                steps_list.append(steps)
                rows[amp]["conv"] += int(converged)
                rows[amp]["n"] += 1
            start_rmsf = spread(starts, topo, damaged)
            settled_rmsf = spread(poses, topo, damaged)
            rows[amp]["start"].append(float(start_rmsf.mean()))
            rows[amp]["settled"].append(float(settled_rmsf.mean()))
            rows[amp]["steps"].append(sum(steps_list) / len(steps_list))
            rows[amp]["dmg"].append(
                sum(float(backbone_rmsd_on(p, native, topo, alive, damaged))
                    for p in poses) / len(poses)
            )
            rows[amp]["bond0"].append(
                sum(float(peptide_bond_restraint(s, topo, alive, links))
                    for s in starts) / len(starts)
            )
            per_residue_by_amp[amp].append(settled_rmsf.tolist())

    if not rows[amplitudes[0]]["start"]:
        print("no crop produced a window")
        return 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print()
    print(f"{'amp deg':>8} {'start RMSF':>11} {'settled RMSF':>13} "
          f"{'ratio':>7} {'steps':>7} {'converged':>10} {'rmsd_dmg':>9} "
          f"{'start bond':>11}")
    for amp in amplitudes:
        r = rows[amp]
        s0, s1 = mean(r["start"]), mean(r["settled"])
        ratio = (s1 / s0) if s0 > 1e-9 else float("nan")
        print(f"{amp:>8.0f} {s0:>11.3f} {s1:>13.3f} {ratio:>7.2f} "
              f"{mean(r['steps']):>7.0f} {r['conv']}/{r['n']:<8} "
              f"{mean(r['dmg']):>9.2f} {mean(r['bond0']):>11.2f}")

    floor = mean(rows[amplitudes[0]]["settled"])
    print()
    print(f"  Amplitude {amplitudes[0]:.0f} is the NOISE FLOOR: members there "
          f"differ by update mask alone.")
    print(f"  Settled RMSF at the floor: {floor:.3f} A. Every other row has to "
          f"beat it to mean anything.")
    print("  Read the ratio column: saturating as the start spread grows means "
          "basins;")
    print("  tracking it means the rule is carrying its input through.")

    print()
    print("per-residue settled RMSF, window position 0..n, mean over crops:")
    for amp in amplitudes:
        cols = per_residue_by_amp[amp]
        if not cols:
            continue
        width = min(len(c) for c in cols)
        avg = [mean([c[i] for c in cols]) for i in range(width)]
        print(f"  {amp:>5.0f} deg  " + " ".join(f"{v:5.2f}" for v in avg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
