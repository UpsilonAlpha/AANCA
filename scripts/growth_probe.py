"""Can a protein be grown, and how far does torsion-space growth get?

    python scripts/growth_probe.py [--pdb 1UBQ] [--steps 60] [--checkpoint PATH]

Three measurements, in the order they decide things.

**The lever arm.** Regrow a crystal backbone from its own measured torsions,
once with the structure's own bond lengths and angles and once with ideal ones.
The first is a round trip and must be exact; the second is the approximation
every de novo growth model lives with, because a structure being grown has no
measured geometry to read. The gap between them, as a function of how far along
the chain you are, is the number that decides whether relaxation can be done in
torsion space at all. It cannot: see the table.

**Growth itself.** Seed, run the rule, and report what came out -- how much of
the protein exists, whether the chain is still a chain, FAPE against the native
backbone and CA RMSD after superposition. With no checkpoint this is an
untrained rule, which is a real baseline rather than a smoke test: an untrained
rule grows alpha helices from its initialisation, so these numbers are what a
trained rule has to beat.

**The synthesis-order control.** One seed with ``--cward-only`` is growth in
synthesis order, which is the thing this project argues folding is not. Several
seeds growing both ways is the alternative. They are reported side by side,
because an argument that spatial growth is better should be made with a number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.backbone import (
    BackboneGeometry,
    backbone_atoms,
    extend_cward,
    ideal_geometry,
    measure_backbone_torsions,
    measured_geometry,
)
from aanca.fape import backbone_rmsd, fape
from aanca.growth import (
    GrowthConfig,
    completion,
    growth_links,
    peptide_bond_restraint,
    seed_alive,
)
from aanca.identity import library_index, type_index, type_marginal
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_identity, initial_state, rollout
from aanca.torch_geometry import measure_chis
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def _pair(geometry: BackboneGeometry, parent: int, child: int) -> BackboneGeometry:
    """The peptide bond's values belong to the parent, the residue's to the child."""
    if geometry.n_ca.dim() == 0:
        return geometry
    return BackboneGeometry(
        n_ca=geometry.n_ca[child],
        ca_c=geometry.ca_c[child],
        c_n=geometry.c_n[parent],
        n_ca_c=geometry.n_ca_c[child],
        ca_c_n=geometry.ca_c_n[parent],
        c_n_ca=geometry.c_n_ca[parent],
    )


def lever_arm(topo, coords, links) -> dict[str, list[float]]:
    """CA error against the crystal as a function of residues grown."""
    phi, psi, omega = measure_backbone_torsions(coords, topo, links)
    bb = backbone_atoms(coords, topo)
    out: dict[str, list[float]] = {}
    for label, geometry in (
        ("measured", measured_geometry(coords, topo, links)),
        ("ideal", ideal_geometry(coords.device, coords.dtype)),
    ):
        n, ca, c = bb[0, 0], bb[0, 1], bb[0, 2]
        errors = []
        for i in range(coords.shape[0] - 1):
            if not bool(links[i]):
                break
            n, ca, c = extend_cward(
                n, ca, c, psi[i], omega[i], phi[i + 1], _pair(geometry, i, i + 1)
            )
            errors.append(float((ca - bb[i + 1, 1]).norm()))
        out[label] = errors
    return out


def grow(rule, topo, native, links, config, seed, identity=False) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    alive = seed_alive(
        topo.n_residues, config.growth.seeds, generator, native.device
    )
    start = None
    if identity:
        start = initial_identity(topo, native, "masked", generator, alive=alive)
    with torch.no_grad():
        result = rollout(
            rule,
            initial_state(native, rule, identity=start, alive=alive),
            topo,
            config,
            generator=generator,
            record=False,
        )
        final = result.state
        scored = final.alive & topo.supervised_mask
        out = {
            "alive": completion(final.alive),
            "bond": float(peptide_bond_restraint(final.coords, topo, final.alive, links)),
            "fape": float(fape(final.coords, native, topo, scored)),
            "rmsd": float(backbone_rmsd(final.coords, native, topo, scored)),
        }
        if identity:
            chi1 = torch.rad2deg(measure_chis(native, topo.torsion)[:, 0])
            target = library_index(topo.torsion.resnames, chi1)
            predicted = type_marginal(
                torch.softmax(final.identity_logits, dim=-1)
            ).argmax(-1)
            right = (predicted == type_index(target)) & scored
            out["recovery"] = float(right.sum() / scored.sum().clamp_min(1))
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default="1UBQ")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--birth",
        choices=("torsion", "cartesian", "both"),
        default="both",
        help="how a newborn is placed. The two are initialised to the same "
        "helical reference birth, so running both is a comparison of what is "
        "learnable rather than of where they start.",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="grow the sequence too: the rule is blind beyond CB, the state "
        "holds no side chains at all, and a newborn arrives unlabelled. Adds "
        "a sequence-recovery column, which is the number ProteinMPNN (~58 %), "
        "PottsMPNN (~44 %) and RFD4-Proteina report.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="a trained rule. Without one the untrained baseline is reported, "
        "which is what a trained rule has to beat.",
    )
    args = parser.parse_args()

    path = RAW / f"{args.pdb}.pdb"
    if not path.exists():
        print(f"{path} missing; run scripts/fetch_test_structures.py")
        return 1
    topo, native = feature_topology(load_protein(path))
    links = growth_links(topo)
    print(f"{args.pdb}: {topo.n_residues} residues\n")

    curves = lever_arm(topo, native, links)
    marks = [m for m in (1, 2, 5, 10, 20, 40, 60, 75) if m <= len(curves["ideal"])]
    print("Regrowing the backbone from its own torsions -- CA error, Angstrom")
    print("  residues grown  " + "".join(f"{m:>9d}" for m in marks))
    for label in ("measured", "ideal"):
        cells = "".join(f"{curves[label][m - 1]:>9.3f}" for m in marks)
        print(f"  {label:<14}" + cells)
    print(
        "\n  measured is the round trip and must be ~0: it is the control on the\n"
        "  placement primitive. ideal is what growth has to use, and the error\n"
        "  it accumulates is why relaxation is a rigid frame move and not\n"
        "  another torsion.\n"
    )

    loaded = None
    if args.checkpoint:
        loaded = ckpt.rule_weights(ckpt.load(Path(args.checkpoint)))
        print(f"rule from {args.checkpoint}\n")
    else:
        print("untrained rule (the baseline)\n")

    if args.identity:
        # Without this the recovery column is unreadable. "Guess the most
        # frequent type everywhere" is the only baseline a sequence model has
        # to beat before its number means anything, and on a single small
        # protein it is high: 1UBQ is 11.8 % leucine.
        from collections import Counter

        counts = Counter(topo.torsion.resnames)
        common, hits = counts.most_common(1)[0]
        print(
            f"background composition: {hits / topo.n_residues:.1%} "
            f"({common}, the most frequent type) -- recovery below this is "
            "worse than guessing\n"
        )

    print(f"Growth over {args.steps} steps, mean of {args.repeats} seeds")
    print(
        "  birth       seeds   mode         alive      bond      fape   CA rmsd"
        + ("  recovery" if args.identity else "")
    )
    modes = ("torsion", "cartesian") if args.birth == "both" else (args.birth,)
    for mode in modes:
        rule = AANCARule(
            growth=True, birth=mode, identity=args.identity, identity_keep=2.0
        )
        if args.checkpoint:
            rule.load_state_dict(ckpt.rule_weights(loaded), strict=False)
        rule.eval()
        for n_seeds in args.seeds:
          for cward_only in (True, False):
            if cward_only and n_seeds != 1:
                # Synthesis order is a single-seed statement; several seeds
                # growing one way is neither control nor model.
                continue
            config = RolloutConfig(
                steps=args.steps,
                truncate=args.steps,
                k=args.k,
                blind=args.identity,
                growth=GrowthConfig(
                    seeds=n_seeds, cward_only=cward_only, birth=mode
                ),
            )
            runs = [
                grow(rule, topo, native, links, config, seed, args.identity)
                for seed in range(args.repeats)
            ]
            mean = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
            label = "C-ward only" if cward_only else "both ways"
            print(
                f"  {mode:<12s}{n_seeds:>3d}   {label:<12s}"
                f"{mean['alive']:>7.1%}{mean['bond']:>10.4f}"
                f"{mean['fape']:>10.4f}{mean['rmsd']:>10.2f}"
                + (f"{mean['recovery']:>10.1%}" if args.identity else "")
            )

    print(
        "\n  bond is the peptide restraint, unweighted, and a crystal structure\n"
        "  scores about 0.002: a value far above that is a torn chain, which is\n"
        "  the failure a rigid frame move can cause and a torsion cannot.\n"
        "  fape saturates at its 10 A clamp, so read rmsd alongside it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
