"""Write an AANCA rollout out as a PyMOL trajectory: what grew, and what moved.

    python scripts/growth_traj.py --run g13_natedge
    python scripts/growth_traj.py --run g13_natedge --crop 3 --out viz/g13
    python scripts/growth_traj.py --run v8b_if --untrained --perturb 0.05

Produces three files in ``--out``:

  ``traj.pdb``    one MODEL per rollout step, dormant residues simply absent
  ``native.pdb``  the crystal structure, as a reference object
  ``view.pml``    ``pymol view.pml`` and the animation is set up

**Load it with ``discrete=1``, which is the whole trick.** A growth rollout
changes its atom count as residues wake, and no fixed-topology trajectory
format -- DCD, XTC, the usual choices -- can represent that; they assume one
atom set for every frame. PyMOL's ``discrete=1`` gives each state its own
atoms, so a residue that is dormant at step 0 and alive at step 40 is *absent*
from the first forty models and appears on the fortieth. That is birth, drawn
without any tricks with occupancy or parking coordinates off-screen.
``view.pml`` passes the flag; if you load the file by hand, pass it yourself or
PyMOL will try to reconcile the atom sets and show you nonsense.

**Two channels, and which is which depends on the run.**

``B-factor`` is always *churn*: how far this residue's CA moved on this step,
in Angstrom. It is the quantity the name of this project keeps implying and
nothing has ever plotted -- a residue that has settled reads 0, a residue being
dragged reads up to the per-step cap (0.3 A), and a rule at its fixed point is
blue everywhere. ``spectrum b`` is set up in ``view.pml``.

``occupancy`` depends on the rollout:

  * a **growth** run writes *freshness* -- steps since this residue woke,
    over 20 -- which draws the growth front: a residue born this step reads 0
    and one that has been settling reads 1;
  * a **regeneration** run writes 1.0 for the intact context and 0.25 for the
    damaged window, because everything is alive from step 0 and the useful
    distinction is which residues are the task. ``select window, q < 0.5``.

**The untrained warning, and read it before concluding the exporter is
broken.** The frame head is zero-initialised on purpose, so an untrained rule
emits *exactly* zero rotation and translation and every model in the
trajectory is identical to the first. That is the rule, not this script. Pass
``--perturb`` to get a rule that moves, and expect it to move randomly.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aanca import checkpoint as ckpt  # noqa: E402
from aanca.backbone import backbone_atoms  # noqa: E402
from aanca.growth import (  # noqa: E402
    GrowthConfig,
    damage_window,
    growth_links,
    baseline_coords,
    run_anchors,
)
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import RolloutConfig, initial_state, rollout  # noqa: E402

# The held-out selection, imported rather than copied: it carries the
# ``args``-missing fallback that exists because job 32241768 silently used a
# probe's defaults when ``best.pt`` turned out to have no ``args``, and a
# second copy of that logic here would be free to drift from it.
from ceiling_readout import held_out  # noqa: E402

PDB_DIR = ROOT / "data" / "pdb"
ATOM_NAMES = ("N", "CA", "C", "O")
THREE = {
    0: "ALA", 1: "ARG", 2: "ASN", 3: "ASP", 4: "CYS", 5: "GLN", 6: "GLU",
    7: "GLY", 8: "HIS", 9: "ILE", 10: "LEU", 11: "LYS", 12: "MET", 13: "PHE",
    14: "PRO", 15: "SER", 16: "THR", 17: "TRP", 18: "TYR", 19: "VAL",
}


def atom_records(
    atoms, topo, alive, churn, occupancy, names, serial: int = 1
) -> tuple[list[str], int]:
    """ATOM lines for the alive residues of one frame.

    Dormant residues are **left out** rather than written somewhere harmless.
    Under ``discrete=1`` that is what makes them appear at birth; writing them
    parked would put a stray atom 200 A away in every early frame and rescale
    every view in the session.
    """
    lines = []
    for i in range(atoms.shape[0]):
        if not bool(alive[i]):
            continue
        chain = topo.chain_ids[i] if i < len(topo.chain_ids) else "A"
        res_id = (
            topo.res_id_list[i] if i < len(topo.res_id_list) else i + 1
        )
        resname = names[i]
        b = min(float(churn[i]), 99.99)
        q = float(occupancy[i])
        for a, name in enumerate(ATOM_NAMES):
            x, y, z = (float(v) for v in atoms[i, a])
            # **Fixed columns, and they are counted rather than eyeballed.**
            # The first version of this put resName at 17 and left altLoc out
            # altogether, which shifted every field after it by one and would
            # have loaded as a structure with no residue names. The layout:
            #   1-6 ATOM, 7-11 serial, 13-16 name, 17 altLoc, 18-20 resName,
            #   22 chain, 23-26 resSeq, 27 iCode, 31-38/39-46/47-54 xyz,
            #   55-60 occupancy, 61-66 tempFactor, 77-78 element.
            # A name of three characters or fewer starts at 14, not 13, because
            # 13-14 is the element symbol's field.
            field = name if len(name) >= 4 else " " + name
            lines.append(
                f"ATOM  {serial:>5d} {field:<4s} {resname:>3s}"
                f" {chain[:1]:1s}{res_id:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{q:>6.2f}{b:>6.2f}"
                f"          {name[0]:>2s}"
            )
            serial += 1
    return lines, serial


def residue_names(topo, identity) -> list[str]:
    """Three-letter names from the committed labels, or the crystal's.

    A growth rollout that carries an identity state has *chosen* these, and
    showing the choice is half the point of watching it. Without one the
    crystal's own types are used, so the movie is never silently all-alanine.
    """
    source = identity.argmax(dim=-1) if identity is not None else None
    if source is None and topo.aa_onehot is not None:
        source = topo.aa_onehot.argmax(dim=-1)
    if source is None:
        return ["ALA"] * topo.n_residues
    return [THREE.get(int(v), "ALA") for v in source]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="tag under runs/, reads best.pt")
    group.add_argument("--checkpoint", help="path to a .pt directly")
    parser.add_argument("--crop", type=int, default=0,
                        help="which held-out crop, by position")
    parser.add_argument("--out", default=None)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--perturb", type=float, default=0.0,
                        help="noise on the frame head's last layer, so an "
                             "untrained rule moves at all")
    parser.add_argument("--every", type=int, default=1,
                        help="write every Nth step; 120 models is fine, but "
                             "this thins a long rollout")
    args = parser.parse_args()

    path = (
        Path(args.checkpoint) if args.checkpoint
        else ROOT / "runs" / args.run / "best.pt"
    )
    if not path.exists():
        print(f"no checkpoint at {path}")
        return 1
    state = torch.load(path, map_location="cpu", weights_only=False)
    stored = state.get("args") or {}
    weights = ckpt.rule_weights(state)
    dataset, indices, _pool = held_out(stored, state)
    if args.crop >= len(indices):
        print(f"--crop {args.crop} but only {len(indices)} held-out crops")
        return 1

    damage = int(stored.get("damage") or 0)
    birth = str(stored.get("birth") or "cartesian")
    steps = int(stored.get("steps") or 120)
    seed = int(stored.get("seed", 0))
    config = RolloutConfig(
        steps=steps,
        truncate=int(stored.get("truncate") or 12),
        k=int(stored.get("k") or 32),
        blind=bool(stored.get("blind", True)),
        growth=GrowthConfig(
            birth=birth, initial=str(stored.get("initial") or "helix")
        ),
    )

    rng = random.Random(seed)
    topo = native = None
    for position, index in enumerate(indices):
        try:
            topo, native, _c, _u = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if position == args.crop:
            break
    if topo is None:
        print("no crop loaded")
        return 1

    rule = AANCARule(
        growth=True, birth=birth,
        identity="identity_head.0.weight" in weights,
        morphogen="emit_head.0.weight" in weights,
        identity_keep=float(stored.get("identity_keep", 2.0)),
    )
    if not args.untrained:
        rule.load_state_dict(weights)
    if args.perturb:
        generator = torch.Generator().manual_seed(seed)
        for parameter in rule.frame_head[-1].parameters():
            parameter.data.add_(
                torch.randn(parameter.shape, generator=generator)
                * args.perturb
            )
    rule.eval()

    links = growth_links(topo)
    generator = torch.Generator().manual_seed(seed + args.crop)
    self_placing = birth == "none"
    damaged = torch.zeros(topo.n_residues, dtype=torch.bool)
    if self_placing:
        alive = torch.ones(topo.n_residues, dtype=torch.bool)
        if damage > 0:
            damaged = damage_window(
                topo.n_residues, links, damage, generator,
                torch.device("cpu"),
            )
            anchors = ~damaged
        else:
            anchors = run_anchors(
                topo.n_residues, links, generator, torch.device("cpu")
            )
        start = baseline_coords(
            native, topo, links, anchors, config.growth.initial, generator
        )
    else:
        # A growing rollout starts from the seeds alone and the alive mask is
        # what the rule extends; ``initial_state`` parks the rest.
        alive = run_anchors(
            topo.n_residues, links, generator, torch.device("cpu")
        )
        start = native

    out = Path(args.out) if args.out else ROOT / "viz" / (
        args.run or path.parent.name
    )
    out.mkdir(parents=True, exist_ok=True)

    names = residue_names(topo, None)
    frames: list[str] = []
    born_at = torch.full((topo.n_residues,), -1, dtype=torch.long)
    churn_trace: list[tuple[int, int, float, float]] = []

    def trace(step, rolled, previous, _mask):
        live = (
            rolled.alive if rolled.alive is not None
            else torch.ones(topo.n_residues, dtype=torch.bool)
        )
        # **The bookkeeping runs on every step, the writing does not.** With
        # ``--every 2`` a birth on an unwritten step would otherwise never be
        # recorded, and the residue would read as born on the next frame --
        # which is the one place this script could quietly lie about when
        # something happened.
        newly = live & (born_at < 0)
        born_at[newly] = step
        if step % args.every:
            return
        atoms = backbone_atoms(rolled.coords, topo)
        rows = torch.arange(topo.n_residues)
        ca_now = rolled.coords[rows, topo.frame_idx[:, 1]]
        ca_was = previous[rows, topo.frame_idx[:, 1]]
        churn = (ca_now - ca_was).norm(dim=-1)
        # **A newborn's displacement is birth, not churn, and it is enormous.**
        # ``park_coords`` translates every dormant residue out of reach, so on
        # the step a residue wakes its CA moves from the parked position to a
        # real one -- measured here at up to 2e4 A. Left in, it is 60000x the
        # per-step cap, so the B-factor scale saturates and every residue in
        # that frame reads maximum red: one artefact would repaint the whole
        # movie on exactly the frames growth is happening. Birth is already
        # drawn by the atom appearing, so the honest value for that step is 0.
        churn = torch.where(born_at == step, torch.zeros_like(churn), churn)
        if bool(damaged.any()):
            occupancy = torch.where(
                damaged, torch.full_like(churn, 0.25),
                torch.ones_like(churn),
            )
        else:
            # **Freshness, not the activation alpha.** ``RolloutState`` carries
            # no alpha -- ``alive`` is already a boolean by the time the state
            # exists -- so there is nothing to read, and inventing a channel
            # that looks like confidence would be worse than having none.
            # Steps since birth is a quantity this script can actually compute,
            # and it draws the growth front: a residue woken on this step reads
            # 0 and one that has been settling for 20 steps reads 1.
            occupancy = torch.where(
                live,
                ((step - born_at).clamp(min=0).to(churn.dtype) / 20.0
                 ).clamp(0.0, 1.0),
                torch.zeros_like(churn),
            )
        step_names = (
            residue_names(topo, rolled.identity)
            if rolled.identity is not None else names
        )
        lines, _ = atom_records(
            atoms, topo, live, churn, occupancy, step_names
        )
        frames.append(
            f"MODEL     {len(frames) + 1:>4d}\n"
            + "\n".join(lines)
            + "\nENDMDL"
        )
        moved = churn[live]
        churn_trace.append((
            step, int(live.sum()),
            float(moved.mean()) if moved.numel() else 0.0,
            float(moved.max()) if moved.numel() else 0.0,
        ))

    with torch.no_grad():
        rollout(
            rule,
            initial_state(start, rule, alive=alive),
            topo, config, generator=generator, record=False, trace=trace,
        )

    (out / "traj.pdb").write_text("\n".join(frames) + "\nEND\n")
    native_atoms = backbone_atoms(native, topo)
    ones = torch.ones(topo.n_residues)
    lines, _ = atom_records(
        native_atoms, topo,
        torch.ones(topo.n_residues, dtype=torch.bool),
        torch.zeros(topo.n_residues), ones, names,
    )
    (out / "native.pdb").write_text("\n".join(lines) + "\nEND\n")
    (out / "view.pml").write_text(pml(out, damage, len(frames)))

    print(f"{len(frames)} models, {topo.n_residues} residues, "
          f"{'regeneration, %d damaged' % int(damaged.sum()) if damage else birth + ' growth'}")
    print(f"\n{'step':>5} {'alive':>6} {'churn_mean':>11} {'churn_max':>10}")
    for row in churn_trace[:: max(1, len(churn_trace) // 12)]:
        print(f"{row[0]:>5d} {row[1]:>6d} {row[2]:>11.4f} {row[3]:>10.4f}")
    print(f"\nwrote {out}\n  pymol {out / 'view.pml'}")
    return 0


def pml(out: Path, damage: int, models: int) -> str:
    window = (
        "select window, traj and q < 0.5\n"
        "show spheres, window and name CA\n"
        "set sphere_scale, 0.25, window\n"
        "color grey70, window\n"
        if damage else
        "# growth run: occupancy is the activation alpha, not a window flag\n"
        "# select committed, traj and q > 0.9\n"
    )
    return f"""# Written by scripts/growth_traj.py. pymol {out / 'view.pml'}
#
# discrete=1 is required: a growth rollout's atom count changes as residues
# wake, so each state needs its own atom set. Without it PyMOL reconciles the
# frames against each other and the animation is meaningless.
load {out / 'traj.pdb'}, traj, discrete=1
load {out / 'native.pdb'}, native

hide everything
show cartoon, native
color grey80, native
set cartoon_transparency, 0.6, native

show ribbon, traj
set ribbon_width, 4

# B-factor is the per-step CA displacement in Angstrom. The per-step cap is
# 0.3 A, so 0.3 is the top of the scale by construction -- a residue at the
# maximum is one the rule is moving as hard as it is allowed to.
spectrum b, blue_white_red, traj, minimum=0, maximum=0.3
{window}
# The two objects share a frame, so do not let PyMOL fit them to each other.
set all_states, off
set movie_fps, 15
mset 1 x{models}
orient native
mplay

# Useful afterwards:
#   set ray_trace_mode, 1 ; mpng frame_      # dump PNGs for a gif
#   iterate_state 60, traj and name CA, print(resi, b)
"""


if __name__ == "__main__":
    raise SystemExit(main())
