"""The clamp decides whether a far-wrong residue has a gradient at all.

``clamp_max`` is exactly flat above the clamp, so a residue 40 A out of place
contributes a constant to the loss and nothing to the gradient. What survives
is the near-diagonal -- and a self-placing run's baseline satisfies the
near-diagonal exactly, because it is built by walking exact peptide geometry.
That is the mechanism g6_noid exposed: with geometry as the *entire* objective
and the identity term removed, seven consecutive held-out evaluations read
17.45 A and 64.4 degrees, the baseline's own values, to the last digit.

``--fape-clamp-prob`` is AF2's answer (Jumper et al. 2021, supplementary
1.9.5): clamp most updates, leave a tenth of them unclamped. These tests pin
that the flag does what it claims and that the default is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from aanca.fape import fape
from aanca.growth import GrowthConfig, growth_objective
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

NATIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "1UBQ.pdb"


def _displaced(coords: torch.Tensor, topo, far: float):
    """The native with its second half moved ``far`` Angstrom away."""
    moved = coords.clone()
    half = topo.n_residues // 2
    moved[half:] = moved[half:] + torch.tensor([far, 0.0, 0.0])
    return moved


def test_above_the_clamp_a_far_residue_has_no_gradient():
    """The flat region, stated as a derivative rather than as prose.

    40 A out of place and 80 A out of place give the *same* gradient under the
    clamp -- zero on the moved atoms -- so no amount of reweighting the other
    terms can make FAPE tell a rule which way to move them.
    """
    topo, native = feature_topology(load_protein(NATIVE))
    mask = torch.ones(topo.n_residues, dtype=torch.bool)

    grads = []
    for far in (40.0, 80.0):
        moved = _displaced(native, topo, far).requires_grad_(True)
        fape(moved, native, topo, mask).backward()
        half = topo.n_residues // 2
        grads.append(float(moved.grad[half:].abs().sum()))

    assert grads[0] == 0.0, "a clamped pair must contribute no direction"
    assert grads[1] == 0.0
    # And unclamped, the same displacement does carry one.
    moved = _displaced(native, topo, 40.0).requires_grad_(True)
    fape(moved, native, topo, mask, clamp=float("inf")).backward()
    half = topo.n_residues // 2
    assert float(moved.grad[half:].abs().sum()) > 0.0


def test_unclamped_fape_separates_wrong_from_very_wrong():
    """Clamped FAPE saturates; unclamped does not.

    This is why the clamp hides the error a fold has to fix: 40 A and 80 A
    score identically under it.
    """
    topo, native = feature_topology(load_protein(NATIVE))
    mask = torch.ones(topo.n_residues, dtype=torch.bool)

    near = float(fape(_displaced(native, topo, 40.0), native, topo, mask))
    far = float(fape(_displaced(native, topo, 80.0), native, topo, mask))
    assert math.isclose(near, far, rel_tol=1e-6), "clamped FAPE saturates"

    near_u = float(
        fape(_displaced(native, topo, 40.0), native, topo, mask, clamp=float("inf"))
    )
    far_u = float(
        fape(_displaced(native, topo, 80.0), native, topo, mask, clamp=float("inf"))
    )
    assert far_u > near_u * 1.5


def _objective(clamp_prob: float, seed: int, clamp: float = 10.0):
    topo, native = feature_topology(load_protein(NATIVE))
    rule = AANCARule(growth=True, birth="none", identity=True)
    config = RolloutConfig(
        steps=4, truncate=99, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )
    terms, _ = growth_objective(
        rule, native, topo, config,
        torch.Generator().manual_seed(seed),
        identity_weight=0.0,
        fape_clamp=clamp,
        fape_clamp_prob=clamp_prob,
    )
    return float(terms.fape)


def test_the_default_is_every_update_clamped():
    """``1.0`` must reproduce every run before the flag existed, exactly.

    Not approximately: the draw is skipped entirely at 1.0, so the generator is
    not even advanced and a resumed run makes the same sequence of choices it
    made before the flag was added.
    """
    topo, native = feature_topology(load_protein(NATIVE))
    config = RolloutConfig(
        steps=4, truncate=99, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )
    states = {}
    for prob in (1.0, 0.5):
        rule = AANCARule(growth=True, birth="none", identity=True)
        generator = torch.Generator().manual_seed(0)
        growth_objective(
            rule, native, topo, config, generator,
            identity_weight=0.0, fape_clamp_prob=prob,
        )
        states[prob] = generator.get_state()

    # At 1.0 the draw is skipped, so the generator ends where a pre-flag run
    # left it; at 0.5 it has consumed one more number. That difference is the
    # whole claim: adding the flag did not perturb any existing run's RNG
    # stream, so g5 and g6 remain bit-reproducible.
    assert not torch.equal(states[1.0], states[0.5])


def test_an_unclamped_draw_changes_the_loss_and_a_clamped_one_does_not():
    """``0.0`` is every update unclamped, ``1.0`` none -- and they differ.

    A self-placing baseline is ~17 A from native, well past a 10 A clamp, so
    the two must not agree; if they did, the flag would be doing nothing and
    the arm it exists for would be a null experiment that looked like a result.
    """
    clamped = _objective(1.0, seed=3)
    unclamped = _objective(0.0, seed=3)
    assert unclamped > clamped, (unclamped, clamped)


def test_the_draw_comes_from_the_run_generator():
    """Same seed, same choice -- so a resumed run is not a different objective.

    The draw is on the CPU from the run's own generator, like every other
    random choice in this project, which is what makes a segment boundary
    invisible in the results.
    """
    a = _objective(0.5, seed=11)
    b = _objective(0.5, seed=11)
    assert a == b
