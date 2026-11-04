"""Do two adjacent newborns err in the same direction?

    python scripts/adjacent_error_probe.py runs/hole_h3/best.pt --hops 3

**The one condition the threading design rests on, and the one nobody had
measured.** `scripts/threading_error_probe.py` showed that the directed C-to-N
graph survives placement error *only if that error is correlated between
neighbours*: independent error at the measured 0.92 Å loses **34 % of all
peptide bonds** at the 2.0 Å cut, and no threshold recovers it, while
correlation over a single CA-CA step brings it to 0.8 %. So the design is safe
if and only if adjacent residues err together.

The common-mode result is **not** that evidence, and conflating them would be
easy. Three hops halving per-witness error while leaving the fused error alone
says the residual is correlated *across the witnesses of one hole* -- several
predictors of a single position agreeing with each other. Whether *two
different* holes err in the same direction is a different claim about different
quantities, and this measures it.

**The statistic is the error in the difference, because that is what a peptide
bond sees.** A bond constrains |C(i) - N(i+1)|, so what matters is
``e_i - e_(i+1)`` rather than either error alone. If the two are independent
with RMS sigma, the difference has RMS ``sigma * sqrt(2)``; if they are
perfectly correlated it has RMS 0. The reported ratio is therefore

    d / (sigma * sqrt(2))      1 = independent, 0 = moving together

and ``d`` itself is quoted in Ångström so it can go straight into the threading
probe's table in place of its independent-error assumption.

**Both holes are deleted at once**, which is the growth case: a gap of two, each
predicted from the same surviving structure. Predicting them one at a time from
otherwise-complete structures would let each see the other's true position and
measure something strictly easier than growth ever faces.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.hole import SEE_RADIUS, HoleTask, _frames, fused_position, hole_predictions
from aanca.model import AANCARule
from aanca.torch_graph import frame_atom_coords

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


def task_for(coords, topo, alive: torch.Tensor, hole: int) -> HoleTask:
    """A HoleTask for ``hole`` against a caller-supplied alive mask.

    The same construction as ``hole.make_task``, with the alive set given
    rather than derived, so both holes of a pair can be asked against one
    partial structure.
    """
    rotation, centre = _frames(coords, topo)
    offset = centre - centre[hole]
    distance = offset.norm(dim=-1)
    witnesses = alive & (distance < SEE_RADIUS) & topo.has_frame
    local = torch.einsum(
        "iab,ib->ia", rotation.transpose(-1, -2), centre[hole] - centre
    )
    target = torch.where(witnesses.unsqueeze(-1), local, torch.zeros_like(local))
    return HoleTask(
        alive=alive, hole=hole, witnesses=witnesses, target=target, distance=distance
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--structures", type=int, default=40)
    parser.add_argument("--pairs-per-structure", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rule = AANCARule(hole=True)
    rule.load_state_dict(
        ckpt.rule_weights(ckpt.load(Path(args.checkpoint), map_location="cpu"))
    )
    rule.eval()

    pool = on_disk(read_manifest(MANIFEST)["validation"])[: args.structures]
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size,
        device="cpu", cache_limit=max(len(pool), 1),
    )
    rng = random.Random(args.seed)

    singles: list[float] = []      # |e| per hole
    diffs: list[float] = []        # |e_i - e_(i+1)| per pair
    cosines: list[float] = []      # cos angle between the two error vectors
    with torch.no_grad():
        for index in range(len(pool)):
            try:
                topo, coords, _c, _u = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 12:
                continue
            ca = frame_atom_coords(coords, topo)[:, 1, :]
            # Candidate pairs: covalently consecutive, both with frames.
            resid = torch.tensor([int(r) for r in topo.torsion.res_ids]) \
                if hasattr(topo.torsion, "res_ids") else None
            candidates = []
            for i in range(topo.n_residues - 1):
                if not (bool(topo.has_frame[i]) and bool(topo.has_frame[i + 1])):
                    continue
                # Covalent continuity, judged by CA-CA as elsewhere in this
                # project: 3.804 A is the trans value and 4.2 is generous.
                if float((ca[i + 1] - ca[i]).norm()) > 4.2:
                    continue
                candidates.append(i)
            if not candidates:
                continue
            rng.shuffle(candidates)
            for i in candidates[: args.pairs_per_structure]:
                alive = torch.ones(topo.n_residues, dtype=torch.bool)
                alive[i] = False
                alive[i + 1] = False
                errors = []
                ok = True
                for hole in (i, i + 1):
                    task = task_for(coords, topo, alive, hole)
                    if not bool(task.witnesses.any()):
                        ok = False
                        break
                    offset, logit = hole_predictions(
                        rule, coords, topo, task, k=args.k, hops=args.hops
                    )
                    predicted = fused_position(offset, logit, task, coords, topo)
                    errors.append(predicted - ca[hole])
                if not ok or len(errors) != 2:
                    continue
                e0, e1 = errors
                singles.append(float(e0.norm()))
                singles.append(float(e1.norm()))
                diffs.append(float((e0 - e1).norm()))
                n0, n1 = float(e0.norm()), float(e1.norm())
                if n0 > 1e-9 and n1 > 1e-9:
                    cosines.append(float((e0 @ e1) / (n0 * n1)))

    if not diffs:
        print("no adjacent pairs measured")
        return 1

    def rms(xs):
        return (sum(x * x for x in xs) / len(xs)) ** 0.5

    sigma = rms(singles)
    d = rms(diffs)
    independent = sigma * (2 ** 0.5)
    ratio = d / max(independent, 1e-12)
    mean_cos = sum(cosines) / len(cosines) if cosines else float("nan")

    print(f"{args.checkpoint}, {args.hops} hop(s), {len(diffs)} adjacent pairs "
          f"from {len(pool)} structures\n")
    print(f"  {'per-hole error (RMS |e|)':<34}{sigma:>8.3f} A")
    print(f"  {'difference error (RMS |e0-e1|)':<34}{d:>8.3f} A")
    print(f"  {'independent prediction (sigma*sqrt2)':<34}{independent:>8.3f} A")
    print(f"  {'ratio  (1 = independent, 0 = together)':<34}{ratio:>8.3f}")
    print(f"  {'mean cos(e0, e1)':<34}{mean_cos:>8.3f}")
    print(
        "\n  The difference error is what a peptide bond sees, so it is the number\n"
        "  that goes into scripts/threading_error_probe.py in place of that\n"
        "  probe's independent-error assumption. For reference that probe loses\n"
        "  34 % of bonds at a 2.0 A cut when the relative error is 1.3 A, and\n"
        "  0.8 % when it is small -- so a difference error near or above 1 A\n"
        "  means Cartesian birth cannot thread unaided, and well below means it\n"
        "  can.\n\n"
        "  A ratio near 1 with a mean cosine near 0 is independent placement: the\n"
        "  common-mode result does not transfer, and the correlation threading\n"
        "  needs is absent. A ratio well below 1 with a positive cosine is\n"
        "  neighbours moving together, which is the condition the design\n"
        "  requires."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
