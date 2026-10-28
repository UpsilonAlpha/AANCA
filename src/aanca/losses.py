"""Von Mises likelihood over chi deltas, and the no-op term.

The head emits a distribution rather than a point, so the loss is a likelihood
rather than a regression: *how surprised was the model by the delta that was
actually needed?* That rewards being confident where it deserves to be and
uncertain where it does not, and it is what makes the concentration parameter
mean something -- see `docs/PLAN_V1.md` §2.

The no-op term is not a separate function. It is this same likelihood evaluated
on an undamaged input with a target delta of zero, which pushes the mean to zero
and the concentration to its cap. One loss, two uses.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from . import residue_constants as rc

#: Computed in Python, not through a tensor: ``torch.tensor(2 * torch.pi)``
#: defaults to float32 and the constant would carry only ~1e-8 of precision,
#: which is above the tolerance the loss is checked against.
LOG_2PI = math.log(2.0 * math.pi)


def log_i0e(kappa: Tensor) -> Tensor:
    """``log(I_0(kappa) * exp(-kappa))`` -- the exponentially scaled Bessel log.

    ``torch.special.i0e`` is scaled by construction, which is what keeps this
    finite: ``I_0`` itself overflows around kappa ~ 700, and the scaled form also
    keeps the gradient sane near the cap.
    """
    return torch.log(torch.special.i0e(kappa).clamp_min(1e-30))


def log_i0(kappa: Tensor) -> Tensor:
    """``log I_0(kappa)``, stable for large kappa."""
    return log_i0e(kappa) + kappa


def _nll_terms(residual: Tensor, kappa: Tensor) -> Tensor:
    """The data-dependent part of the von Mises NLL: ``kappa * (1 - cos(residual))``.

    The textbook form is ``-kappa*cos(d) + log(2*pi*I_0(kappa))``. Rewriting the
    first term as ``kappa*(1 - cos(d))`` -- so it is zero at a perfect
    prediction, which keeps the logged numbers interpretable -- moves a ``+kappa``
    into the normaliser, and ``log I_0(kappa) - kappa`` is exactly
    ``log_i0e(kappa)``. The two must therefore be used as a pair: pairing this
    form with the *unscaled* ``log I_0`` double-counts kappa and makes the loss
    increase with confidence at a perfect prediction, which is backwards.
    """
    return kappa * (1.0 - torch.cos(residual))


def von_mises_nll(
    mu: Tensor,
    log_kappa: Tensor,
    target: Tensor,
    chi_mask: Tensor,
    periodic: Tensor | None = None,
) -> Tensor:
    """Per-chi negative log likelihood, ``(n_res, MAX_CHI)``, zero where masked.

    ``mu`` and ``target`` are angles in radians; ``log_kappa`` is the head's
    bounded confidence output.

    For the two-fold symmetric terminal chis (ASP chi2, GLU chi3, PHE chi2, TYR
    chi2) a 180-degree flip is physically the same structure. Those are scored as
    a proper **two-component mixture**, ``0.5 p(x) + 0.5 p(x + pi)``, rather than
    by folding the residual and pretending the density is unchanged. Folding
    would compute ``-log max(...)`` instead of ``-log mean(...)``, which is not a
    normalised density and quietly biases kappa upward on exactly those chis.
    """
    kappa = torch.exp(log_kappa)
    # Scaled, to pair with the ``1 - cos`` form in ``_nll_terms``. See its docstring.
    normaliser = LOG_2PI + log_i0e(kappa)

    residual = mu - target
    plain = _nll_terms(residual, kappa) + normaliser

    if periodic is not None and bool(periodic.any()):
        flipped = _nll_terms(residual + torch.pi, kappa) + normaliser
        # -log( 0.5 e^-a + 0.5 e^-b ) = log 2 - logsumexp(-a, -b)
        mixture = torch.log(torch.tensor(2.0, dtype=plain.dtype, device=plain.device))
        mixture = mixture - torch.logsumexp(
            torch.stack([-plain, -flipped], dim=0), dim=0
        )
        nll = torch.where(periodic, mixture, plain)
    else:
        nll = plain

    return torch.where(chi_mask, nll, torch.zeros_like(nll))


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over the masked entries; zero when nothing is masked in."""
    count = mask.sum()
    if count == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / count


def chi_nll(
    mu: Tensor,
    log_kappa: Tensor,
    target: Tensor,
    chi_mask: Tensor,
    periodic: Tensor | None = None,
) -> Tensor:
    """Scalar mean NLL over every real chi."""
    per_chi = von_mises_nll(mu, log_kappa, target, chi_mask, periodic)
    return masked_mean(per_chi, chi_mask)


def noop_nll(
    mu: Tensor,
    log_kappa: Tensor,
    chi_mask: Tensor,
    periodic: Tensor | None = None,
) -> Tensor:
    """The no-op term: native in, zero update out.

    Without this nothing stops the rule drifting off a correct answer, and the
    fixed point is not a fixed point. Note what it does to the concentration as
    well as the mean: at a target of zero the likelihood is maximised by pushing
    kappa to its cap, so "native in, zero out" becomes "and be certain about
    it", which is a stronger and better-posed condition than a plain
    zero-magnitude penalty.
    """
    return chi_nll(mu, log_kappa, torch.zeros_like(mu), chi_mask, periodic)


def harmonic_restraint(
    current: Tensor, reference: Tensor, mask: Tensor, stiffness: float
) -> Tensor:
    """``stiffness * mean((current - reference)^2)`` over the masked entries.

    This is what makes bond and angle control safe to expose. A classical
    forcefield's stretching and bending terms are stiff springs about ideal
    values, and so is this -- the reference is the *crystal* geometry, not an
    ideal-geometry table, so the restraint preserves whatever the structure
    actually had rather than pulling it toward a library mean.

    Without it, bond geometry is a physically wrong shortcut: deforming a bond
    relieves a steric clash more cheaply than finding the right rotamer, which
    lowers the clash term, raises chi MAE, and does not show up in any
    aggregate metric.
    """
    return stiffness * masked_mean((current - reference) ** 2, mask)


def chi_mae_degrees(
    pred: Tensor, target: Tensor, chi_mask: Tensor, periodic: Tensor | None = None
) -> Tensor:
    """Per-chi-index MAE in degrees, ``(MAX_CHI,)``, symmetric chis folded.

    Reported per chi index rather than pooled: chi1 and chi4 accuracy differ a
    lot, and a single pooled number hides which one moved.
    """
    from .torch_geometry import chi_residual

    errors = chi_residual(pred, target, periodic).abs()
    out = torch.zeros(rc.MAX_CHI, dtype=errors.dtype, device=errors.device)
    for i in range(rc.MAX_CHI):
        out[i] = masked_mean(errors[:, i], chi_mask[:, i])
    return torch.rad2deg(out)
