"""The von Mises likelihood, checked against the textbook formula and its own shape."""

from __future__ import annotations

import math

import pytest
import torch

from aanca import residue_constants as rc
from aanca.losses import (
    chi_mae_degrees,
    chi_nll,
    log_i0,
    log_i0e,
    noop_nll,
    von_mises_nll,
)

ALL = torch.ones(1, rc.MAX_CHI, dtype=torch.bool)


def _reference_nll(x: float, mu: float, kappa: float) -> float:
    """-log p(x) for a von Mises, written straight from the definition.

    Independent of the implementation's algebra, which is the point: this is
    what caught the normaliser pairing bug.
    """
    i0 = float(torch.special.i0(torch.tensor(kappa, dtype=torch.float64)))
    return -kappa * math.cos(x - mu) + math.log(2.0 * math.pi * i0)


@pytest.mark.parametrize("kappa", [0.5, 1.0, 5.0, 20.0, 100.0])
@pytest.mark.parametrize("residual", [0.0, 0.3, 1.5, 3.0])
def test_matches_the_textbook_formula(kappa, residual):
    mu = torch.full((1, rc.MAX_CHI), residual, dtype=torch.float64)
    log_kappa = torch.full((1, rc.MAX_CHI), math.log(kappa), dtype=torch.float64)
    target = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)

    got = float(von_mises_nll(mu, log_kappa, target, ALL)[0, 0])
    assert got == pytest.approx(_reference_nll(residual, 0.0, kappa), abs=1e-9)


def test_confidence_helps_when_right_and_hurts_when_wrong():
    """The sign check the normaliser bug got backwards."""
    target = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)
    kappas = [math.log(k) for k in (1.0, 5.0, 25.0, 125.0)]

    perfect = [
        float(
            von_mises_nll(
                torch.zeros(1, rc.MAX_CHI, dtype=torch.float64),
                torch.full((1, rc.MAX_CHI), lk, dtype=torch.float64),
                target,
                ALL,
            )[0, 0]
        )
        for lk in kappas
    ]
    assert perfect == sorted(perfect, reverse=True), perfect

    wrong = [
        float(
            von_mises_nll(
                torch.full((1, rc.MAX_CHI), 2.5, dtype=torch.float64),
                torch.full((1, rc.MAX_CHI), lk, dtype=torch.float64),
                target,
                ALL,
            )[0, 0]
        )
        for lk in kappas
    ]
    assert wrong == sorted(wrong), wrong


def test_minimised_at_the_true_delta():
    target = torch.full((1, rc.MAX_CHI), 0.7, dtype=torch.float64)
    log_kappa = torch.full((1, rc.MAX_CHI), 1.5, dtype=torch.float64)
    candidates = torch.linspace(-math.pi, math.pi, 721, dtype=torch.float64)
    values = [
        float(
            von_mises_nll(
                torch.full((1, rc.MAX_CHI), float(c), dtype=torch.float64),
                log_kappa,
                target,
                ALL,
            )[0, 0]
        )
        for c in candidates
    ]
    best = float(candidates[min(range(len(values)), key=values.__getitem__)])
    assert best == pytest.approx(0.7, abs=0.01)


def test_gradient_pushes_mu_toward_the_target():
    mu = torch.full((1, rc.MAX_CHI), 0.0, dtype=torch.float64, requires_grad=True)
    target = torch.full((1, rc.MAX_CHI), 1.0, dtype=torch.float64)
    log_kappa = torch.full((1, rc.MAX_CHI), 1.0, dtype=torch.float64)
    chi_nll(mu, log_kappa, target, ALL).backward()
    # Target is above mu, so descending the loss must raise mu.
    assert float(mu.grad[0, 0]) < 0.0


# -- the symmetry mixture --------------------------------------------------


def test_symmetric_chis_score_a_180_degree_flip_identically():
    """ASP chi2, GLU chi3, PHE chi2, TYR chi2: the flip is the same structure."""
    periodic = torch.tensor([[False, True, False, False]])
    log_kappa = torch.full((1, rc.MAX_CHI), 1.2, dtype=torch.float64)
    target = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)

    at_zero = von_mises_nll(
        torch.zeros(1, rc.MAX_CHI, dtype=torch.float64), log_kappa, target, ALL, periodic
    )
    at_pi = von_mises_nll(
        torch.full((1, rc.MAX_CHI), math.pi, dtype=torch.float64),
        log_kappa,
        target,
        ALL,
        periodic,
    )
    assert float(at_zero[0, 1]) == pytest.approx(float(at_pi[0, 1]), abs=1e-9)
    # The non-periodic chi in the same row must be unaffected by the folding.
    assert float(at_zero[0, 0]) != pytest.approx(float(at_pi[0, 0]), abs=1e-3)


def test_the_mixture_is_a_mixture_not_a_minimum():
    """Folding the residual would give -log max(...); we want -log mean(...).

    Both limits are checked, because they are what distinguishes the two:

    * one branch dominating -- the mixture sits exactly log 2 *above* the single
      component, which is the normalisation that folding throws away;
    * both branches equal -- averaging two equal densities returns the same
      density, so the NLL is unchanged.
    """
    periodic = torch.tensor([[True]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    log_kappa = torch.full((1, 1), 3.0, dtype=torch.float64)
    target = torch.zeros(1, 1, dtype=torch.float64)

    # Dominating branch: mu sits on one of the two equivalent targets.
    on_target = torch.zeros(1, 1, dtype=torch.float64)
    mixture = float(von_mises_nll(on_target, log_kappa, target, mask, periodic)[0, 0])
    single = float(von_mises_nll(on_target, log_kappa, target, mask, None)[0, 0])
    assert mixture == pytest.approx(single + math.log(2.0), abs=1e-6)

    # Equidistant: both components contribute equally.
    halfway = torch.full((1, 1), math.pi / 2, dtype=torch.float64)
    mixture_half = float(von_mises_nll(halfway, log_kappa, target, mask, periodic)[0, 0])
    single_half = float(von_mises_nll(halfway, log_kappa, target, mask, None)[0, 0])
    assert mixture_half == pytest.approx(single_half, abs=1e-9)


# -- masking and the no-op term --------------------------------------------


def test_masked_chis_contribute_nothing():
    mask = torch.tensor([[True, False, False, False]])
    mu = torch.full((1, rc.MAX_CHI), 2.0, dtype=torch.float64)
    log_kappa = torch.full((1, rc.MAX_CHI), 1.0, dtype=torch.float64)
    target = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)
    per_chi = von_mises_nll(mu, log_kappa, target, mask, None)
    assert (per_chi[~mask] == 0.0).all()
    assert float(chi_nll(mu, log_kappa, target, mask)) == pytest.approx(
        float(per_chi[0, 0])
    )


def test_noop_term_wants_zero_mean_and_high_confidence():
    """Native in, zero out -- and be certain about it."""
    mask = torch.ones(1, rc.MAX_CHI, dtype=torch.bool)

    mu = torch.full((1, rc.MAX_CHI), 0.4, dtype=torch.float64, requires_grad=True)
    log_kappa = torch.full((1, rc.MAX_CHI), 1.0, dtype=torch.float64, requires_grad=True)
    noop_nll(mu, log_kappa, mask).backward()
    # Mean is above zero, so descending must lower it.
    assert float(mu.grad[0, 0]) > 0.0

    # At a perfect zero update, the loss still rewards raising confidence.
    mu2 = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)
    lk = torch.full((1, rc.MAX_CHI), 1.0, dtype=torch.float64, requires_grad=True)
    noop_nll(mu2, lk, mask).backward()
    assert float(lk.grad[0, 0]) < 0.0


def test_empty_mask_gives_zero_not_nan():
    mask = torch.zeros(1, rc.MAX_CHI, dtype=torch.bool)
    mu = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)
    log_kappa = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)
    out = chi_nll(mu, log_kappa, mu, mask)
    assert torch.isfinite(out) and float(out) == 0.0


# -- numerics --------------------------------------------------------------


@pytest.mark.parametrize("kappa", [1e-6, 1.0, 100.0, 403.0, 1e4])
def test_bessel_helpers_stay_finite(kappa):
    k = torch.tensor([kappa], dtype=torch.float64)
    assert torch.isfinite(log_i0e(k)).all()
    assert torch.isfinite(log_i0(k)).all()


def test_log_i0_matches_the_unscaled_bessel_where_it_is_representable():
    k = torch.tensor([0.5, 2.0, 20.0], dtype=torch.float64)
    assert torch.allclose(log_i0(k), torch.log(torch.special.i0(k)), atol=1e-10)


def test_nll_is_finite_at_the_kappa_cap():
    from aanca.model import MAX_LOG_KAPPA

    log_kappa = torch.full((1, rc.MAX_CHI), MAX_LOG_KAPPA, dtype=torch.float64)
    mu = torch.full((1, rc.MAX_CHI), 0.1, dtype=torch.float64, requires_grad=True)
    target = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)
    loss = chi_nll(mu, log_kappa, target, ALL)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(mu.grad).all()


# -- reporting -------------------------------------------------------------


def test_mae_is_zero_for_a_perfect_prediction():
    pred = torch.randn(7, rc.MAX_CHI, dtype=torch.float64)
    mask = torch.ones(7, rc.MAX_CHI, dtype=torch.bool)
    mae = chi_mae_degrees(pred, pred, mask)
    assert mae.shape == (rc.MAX_CHI,)
    assert torch.allclose(mae, torch.zeros_like(mae), atol=1e-9)


def test_mae_folds_symmetric_chis():
    pred = torch.full((1, rc.MAX_CHI), math.pi, dtype=torch.float64)
    target = torch.zeros(1, rc.MAX_CHI, dtype=torch.float64)
    mask = torch.ones(1, rc.MAX_CHI, dtype=torch.bool)
    periodic = torch.tensor([[False, True, False, False]])
    mae = chi_mae_degrees(pred, target, mask, periodic)
    assert float(mae[0]) == pytest.approx(180.0, abs=1e-6)
    assert float(mae[1]) == pytest.approx(0.0, abs=1e-6)
