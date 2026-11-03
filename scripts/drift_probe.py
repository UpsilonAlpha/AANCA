"""Is a junction gap explained by the two sides' accumulated drift?

A parent-child pair is at exactly 3.804 A because that is what a birth sets,
so the cross-segment junctions are precisely the adjacent alive pairs that are
*not* at 3.804. For each of those, compare the gap against the two sides' own
errors against native. If the gap tracks the drift, then closure difficulty is
a function of per-birth placement error -- and the untrained numbers say
nothing about a trained rule.
"""

from __future__ import annotations

import torch

from aanca.backbone import backbone_atoms
from aanca.growth import GrowthConfig, growth_links, seed_alive
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, rollout
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

TRANS = 3.804

topo, native = feature_topology(load_protein("data/raw/1UBQ.pdb"))
links = growth_links(topo)
native_ca = backbone_atoms(native, topo)[:, 1]
rule = AANCARule(growth=True, birth="cartesian")
rule.eval()

rows = []
for seed in range(8):
    generator = torch.Generator().manual_seed(seed)
    alive0 = seed_alive(topo.n_residues, 6, generator)
    config = RolloutConfig(
        steps=60, truncate=60, k=24,
        growth=GrowthConfig(seeds=6, birth="cartesian"),
    )
    with torch.no_grad():
        out = rollout(
            rule, initial_state(native, rule, alive=alive0), topo, config,
            generator=generator, record=False,
        )
    final = out.state
    ca = backbone_atoms(final.coords, topo)[:, 1]
    drift = (ca - native_ca).norm(dim=-1)

    for i in range(topo.n_residues - 1):
        if not (bool(final.alive[i]) and bool(links[i]) and bool(final.alive[i + 1])):
            continue
        gap = float((ca[i + 1] - ca[i]).norm())
        if abs(gap - TRANS) < 0.01:
            continue  # a birth placed this pair
        rows.append((gap, float(drift[i]), float(drift[i + 1])))

if not rows:
    print("no cross-segment junctions formed")
else:
    data = torch.tensor(rows)
    gap, a, b = data[:, 0], data[:, 1], data[:, 2]
    combined = a + b
    print(f"{gap.numel()} cross-segment junctions over 8 runs\n")
    print(f"  junction CA-CA gap      mean {float(gap.mean()):6.2f} A  "
          f"median {float(gap.median()):6.2f}  max {float(gap.max()):6.2f}")
    print(f"  sum of the two drifts   mean {float(combined.mean()):6.2f} A  "
          f"median {float(combined.median()):6.2f}  max {float(combined.max()):6.2f}")
    centred_gap = gap - gap.mean()
    centred_sum = combined - combined.mean()
    r = float(
        (centred_gap * centred_sum).sum()
        / (centred_gap.norm() * centred_sum.norm()).clamp_min(1e-9)
    )
    print(f"\n  correlation between the gap and the summed drift: r = {r:.3f}")
    print(
        f"  fraction of junctions already inside REACH[1] = 3.96 A: "
        f"{float((gap < 3.96).to(torch.float64).mean()):.1%}"
    )
