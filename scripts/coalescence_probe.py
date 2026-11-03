"""How bad is coalescence, in Angstrom?

Grow 1UBQ from several seeds and look at every covalently bonded pair that is
alive at both ends. A pair created by a birth should sit at the peptide bond's
1.33 A. A pair where two independently grown segments became adjacent has
never been placed by anything, so it sits wherever the two growth processes
left it.
"""

from __future__ import annotations

import torch

from aanca.backbone import backbone_atoms
from aanca.growth import GrowthConfig, growth_links, seed_alive
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

topo, native = feature_topology(load_protein("data/raw/1UBQ.pdb"))
links = growth_links(topo)
rule = AANCARule(growth=True, birth="cartesian")
rule.eval()

print(f"{'seeds':>6} {'alive':>7} {'segments':>9} {'bonded':>7} {'joined':>7} "
      f"{'junctions':>10} {'worst gap':>10} {'median gap':>11}")

for n_seeds in (1, 2, 4, 8, 16):
    rows = []
    for seed in range(3):
        generator = torch.Generator().manual_seed(seed)
        alive = seed_alive(topo.n_residues, n_seeds, generator)
        config = RolloutConfig(
            steps=60, truncate=60, k=24,
            growth=GrowthConfig(seeds=n_seeds, birth="cartesian"),
        )
        with torch.no_grad():
            out = rollout(
                rule, initial_state(native, rule, alive=alive), topo, config,
                generator=generator, record=False,
            )
        final = out.state
        live = final.alive
        bb = backbone_atoms(final.coords, topo)

        # Segments: maximal runs of alive residues joined by a real link.
        segments = 0
        for i in range(topo.n_residues):
            if not bool(live[i]):
                continue
            joined_back = i > 0 and bool(live[i - 1]) and bool(links[i - 1])
            if not joined_back:
                segments += 1

        # Every bonded pair alive at both ends, and its realised C-N distance.
        gaps = []
        for i in range(topo.n_residues - 1):
            if bool(live[i]) and bool(links[i]) and bool(live[i + 1]):
                gaps.append(float((bb[i + 1, 0] - bb[i, 2]).norm()))
        gaps_t = torch.tensor(gaps) if gaps else torch.zeros(0)
        # A junction is a bonded alive pair that is nowhere near a peptide bond.
        junctions = gaps_t[gaps_t > 2.0]
        rows.append(
            (
                float(live.sum()) / topo.n_residues,
                segments,
                len(gaps),
                int((gaps_t <= 2.0).sum()),
                junctions,
            )
        )

    alive_mean = sum(r[0] for r in rows) / len(rows)
    seg_mean = sum(r[1] for r in rows) / len(rows)
    bonded = sum(r[2] for r in rows) / len(rows)
    joined = sum(r[3] for r in rows) / len(rows)
    allj = torch.cat([r[4] for r in rows]) if any(r[4].numel() for r in rows) else torch.zeros(0)
    worst = f"{float(allj.max()):.1f}" if allj.numel() else "-"
    median = f"{float(allj.median()):.1f}" if allj.numel() else "-"
    print(f"{n_seeds:>6} {alive_mean:>6.1%} {seg_mean:>9.1f} {bonded:>7.1f} "
          f"{joined:>7.1f} {allj.numel() / len(rows):>10.1f} {worst:>10} {median:>11}")

print(
    "\nbonded = covalently bonded pairs alive at both ends.\n"
    "joined = those within 2 A, i.e. an actual peptide bond made by a birth.\n"
    "junctions = those beyond 2 A: two segments that became sequence-adjacent\n"
    "            without ever being placed relative to each other.\n"
    "gap = the realised C-N distance, Angstrom, against an ideal 1.33."
)
