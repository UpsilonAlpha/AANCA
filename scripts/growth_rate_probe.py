"""Does growth finish, and in what fraction of the rollout?

Reason 3 against a joint growth-and-packing objective was that a growing
structure has incomplete neighbourhoods where growth is happening, so the chi
target is unlearnable there. That objection only bites for as long as growth
is still going. ``loss_window = 0.5`` already supervises the back half of the
rollout, so if growth finishes in the first half the objection dissolves.

So the question is a number: when is the structure complete? The activation
bias is the knob -- at the default -2 an untrained rule wakes a frontier
residue with probability 0.12 per step.
"""

from __future__ import annotations

import torch

from aanca.growth import GrowthConfig, completion, growth_links, seed_alive
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, initial_state, step_once, update_mask
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

topo, native = feature_topology(load_protein("data/raw/1UBQ.pdb"))
links = growth_links(topo)
STEPS = 120

print(f"1UBQ, {topo.n_residues} residues, {STEPS} steps, mean of 3 seeds")
print(f"{'bias':>6} {'p(wake)':>8} {'seeds':>6} {'max_births':>11} "
      f"{'complete at':>12} {'fraction':>9} {'final alive':>12}")

for bias in (-2.0, 0.0, 2.0, 4.0):
    for n_seeds, births in ((1, 8), (4, 8), (4, 2)):
        firsts, finals = [], []
        for seed in range(3):
            rule = AANCARule(growth=True, birth="cartesian", activation_init=bias)
            rule.eval()
            generator = torch.Generator().manual_seed(seed)
            alive = seed_alive(topo.n_residues, n_seeds, generator)
            config = RolloutConfig(
                steps=STEPS, truncate=STEPS, k=24,
                growth=GrowthConfig(
                    seeds=n_seeds, birth="cartesian", max_births=births
                ),
            )
            state = initial_state(native, rule, alive=alive)
            mask_p = config.update_probability
            done = None
            with torch.no_grad():
                for step in range(STEPS):
                    mask = update_mask(topo.n_residues, mask_p, native.device, generator)
                    state, _mu, _lk = step_once(
                        rule, state, topo, config, mask, generator=generator
                    )
                    if done is None and bool(state.alive.all()):
                        done = step + 1
            firsts.append(done)
            finals.append(completion(state.alive))

        got = [f for f in firsts if f is not None]
        when = f"{sum(got) / len(got):.0f}" if len(got) == len(firsts) else (
            f">{STEPS} ({len(got)}/3)" if got else f">{STEPS}"
        )
        fraction = f"{sum(got) / len(got) / STEPS:.0%}" if len(got) == len(firsts) else "-"
        final = sum(finals) / len(finals)
        print(f"{bias:>6.1f} {torch.sigmoid(torch.tensor(bias)):>8.2f} {n_seeds:>6} "
              f"{births:>11} {when:>12} {fraction:>9} {final:>11.1%}")

print(
    "\ncomplete at = first step where every slot is alive, mean over seeds.\n"
    "fraction = that step as a share of the rollout. Below 50% means growth\n"
    "           finishes before the loss window opens, which is what reason 3\n"
    "           against a joint objective actually needs."
)
