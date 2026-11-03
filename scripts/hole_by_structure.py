"""Where does the hole head do well, and where does it fail?

    python scripts/hole_by_structure.py runs/hole_v2/best.pt [--structures 60]

The aggregate number hides the result. `scripts/hole_baseline.py` and the
circle argument below say the trained head reaches about 1.65 A while chain
geometry alone gives 2.26 A, so it has bought only 27 % of the irreducible
radius -- which sounds like a model that has learned nothing but the chain.
That reading is only right if the error is uniform, and there is a strong
reason to think it is not.

**The circle, and why it reframes the whole measurement.** A hole whose two
covalent neighbours exist is not free in three dimensions. Its CA must sit
3.804 A from each of them, so it lies on a *circle* whose radius follows from
the CA(i-1)-CA(i+1) distance: ``r = sqrt(3.804^2 - (d/2)^2)``. Measured over
68,277 residues in 143 structures, that radius is **2.26 A mean, 2.47 A
median**. So locating a hole to 0.6 A is not a matter of shrinking a spherical
error -- it is a matter of knowing **where on the circle**, to about a sixth of
it.

And where on the circle is exactly what phi and psi determine. So the error
should be small wherever the backbone's conformation is pinned by regular
secondary structure, and large in loops where it is not. If so, "1.65 A" is a
mixture of two very different populations and the average is the least
informative summary of it.

**The proxy, which needs no DSSP.** The CA(i-1)-CA(i+1) distance separates the
regular conformations cleanly and is already computed for the circle: an alpha
helix sits near 5.4-5.6 A, an extended strand near 6.5-7.0, and the values
between and below are turns, bridges and irregular loops. So the error is
binned by ``d`` and the bins are named.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from aanca import checkpoint as ckpt
from aanca.backbone import backbone_atoms
from aanca.data import RepackingDataset, read_manifest
from aanca.growth import growth_links
from aanca.hole import consensus_error, hole_predictions, make_task
from aanca.model import AANCARule

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"

#: Twice the trans peptide's CA-CA distance: the largest CA(i-1)-CA(i+1)
#: separation a bonded triple can have, reached only when the three are
#: collinear. **Anything at or beyond it is not a bonded triple**, and the
#: first version of this script did not check, so its "very extended" bin
#: reported 3.70 A and a ratio of 0.33 -- apparently three times worse than
#: chain geometry alone. Those were chain breaks: a hole whose neighbours are
#: not bonded to it has no circle to lie on and nothing to be measured against.
#: Covalent continuity is now required, which is what ``growth_links`` is for.
MAX_TRIPLE = 2 * 3.804

#: ``(label, low, high)`` for the CA(i-1)-CA(i+1) distance, Angstrom.
BINS = (
    ("tight turn", 0.0, 5.2),
    ("alpha helix", 5.2, 5.9),
    ("irregular", 5.9, 6.4),
    ("extended", 6.4, 7.1),
    ("near collinear", 7.1, MAX_TRIPLE),
)


def on_disk(specs):
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--structures", type=int, default=60)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pool = on_disk(read_manifest(MANIFEST)["validation"])[: args.structures]
    if not pool:
        print(f"no coordinates in {PDB}")
        return 1
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size,
        device="cpu", cache_limit=max(len(pool), 1),
    )

    rule = AANCARule(hole=True)
    state = ckpt.load(Path(args.checkpoint))
    rule.load_state_dict(ckpt.rule_weights(state))
    rule.eval()
    print(f"rule from {args.checkpoint}, {args.hops} hop(s)\n")

    buckets: dict[str, list[float]] = {label: [] for label, _l, _h in BINS}
    radii: dict[str, list[float]] = {label: [] for label, _l, _h in BINS}
    no_pair = []

    with torch.no_grad():
        for repeat in range(args.repeats):
            rng = random.Random(args.seed + repeat)
            for index in range(len(pool)):
                try:
                    topo, coords, _c, _u = dataset.load_crop(index, rng=rng)
                except Exception:
                    continue
                if topo.n_residues < 12:
                    continue
                try:
                    task = make_task(coords, topo, "hole", rng)
                except ValueError:
                    continue
                if not bool(task.witnesses.any()):
                    continue
                links = growth_links(topo)
                offset, logit = hole_predictions(
                    rule, coords, topo, task, k=args.k, hops=args.hops
                )
                error = float(consensus_error(offset, logit, task, coords, topo))

                hole = task.hole
                ca = backbone_atoms(coords, topo)[:, 1]
                # The hole must be **covalently bonded** to both neighbours,
                # not merely adjacent in index. Without this the bins collect
                # chain breaks, where there is no circle and no bound.
                if hole == 0 or hole + 1 >= topo.n_residues:
                    no_pair.append(error)
                    continue
                if not (bool(links[hole - 1]) and bool(links[hole])):
                    no_pair.append(error)
                    continue
                d = float((ca[hole + 1] - ca[hole - 1]).norm())
                if d >= MAX_TRIPLE:
                    # Bonded by numbering but not by geometry: a broken model.
                    no_pair.append(error)
                    continue
                placed = False
                for label, low, high in BINS:
                    if low <= d < high:
                        buckets[label].append(error)
                        radii[label].append(
                            math.sqrt(3.804**2 - (d / 2) ** 2)
                        )
                        placed = True
                        break
                if not placed:
                    no_pair.append(error)

    print(
        f"  {'conformation':<16}{'d range':>12}{'fused error':>13}"
        f"{'circle r':>11}{'ratio':>8}{'n':>7}"
    )
    for label, low, high in BINS:
        values = buckets[label]
        if len(values) < 5:
            continue
        error = sum(values) / len(values)
        radius = sum(radii[label]) / len(radii[label]) if radii[label] else float("nan")
        ratio = radius / error if error > 0 else float("nan")
        print(
            f"  {label:<16}{f'{low:.1f}-{high:.1f}':>12}{error:>11.2f} A"
            f"{radius:>10.2f} A{ratio:>8.2f}{len(values):>7d}"
        )
    if no_pair:
        print(
            f"\n  {len(no_pair)} holes excluded: at a chain end, or not "
            f"covalently bonded to both neighbours (mean error "
            f"{sum(no_pair) / len(no_pair):.2f} A -- no circle, so no bound)"
        )
    everything = [v for values in buckets.values() for v in values]
    if everything:
        print(
            f"\n  {'all':<16}{'':>12}"
            f"{sum(everything) / len(everything):>11.2f} A"
            f"{'':>10}  {'':>8}{len(everything):>7d}"
        )
    print(
        "\n  circle r is the positional freedom chain geometry alone leaves:\n"
        "  sqrt(3.804^2 - (d/2)^2), the radius of the circle the hole's CA must\n"
        "  lie on given its two covalent neighbours. ratio is how much tighter\n"
        "  than that the head is -- the only honest way to read the error,\n"
        "  since the circle itself is twice as wide in a helix as in a strand."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
