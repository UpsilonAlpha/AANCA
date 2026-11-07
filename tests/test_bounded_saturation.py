"""``_bounded`` destroys the gradient at BOTH ends, and only one was known.

``_bounded``'s docstring records the failure at the **origin** -- a
zero-initialised head gives ``limit * tanh(0) / 0 = 0 / eps = 0``, which
multiplies the output by zero and leaves the head with no gradient for every
update forever -- and calls the branch guarding it load-bearing, with a measured
``0.000e+00`` behind it. The same docstring says tanh was chosen over a hard
clamp so that "the gradient does not vanish for a head that is asking for too
much".

**That second half was an assumption and it is false at scale.** The derivative
of the bounded norm with respect to the raw one is ``1 - tanh^2(raw/limit)``, so
a head asking for several times the cap sits in the flat tail. On
``g10_equiv_ceil`` (job 32423296) ``step`` reads 0.3000 on all twelve held-out
crops -- 100 % of the 0.3 A cap -- which inverts to a request of at least 4.7x
the cap and a magnitude gradient at most 3.3e-4 of its near-origin value.

These tests pin the arithmetic that claim rests on, so it stays true if
``_bounded`` is ever changed. See
``docs/claims/the-move-heads-magnitude-gradient-is-attenuated-three-thousandfold.md``.
"""

import math

import pytest
import torch

from aanca.model import _bounded

CAP = 0.3


def _bounded_norm(raw_norm: float, limit: float = CAP) -> float:
    vector = torch.tensor([[raw_norm, 0.0, 0.0]])
    return float(_bounded(vector, limit).norm(dim=-1)[0])


def test_in_exact_arithmetic_the_bound_is_asymptotic_but_float32_reaches_it():
    """**The reason 0.3000 is readable as a request -- and its limit.**

    In exact arithmetic tanh never reaches 1, so a bounded norm equal to the cap
    would be impossible and any approach to it says how much more the head asked
    for. That is what makes ``step`` invertible.

    In **float32 it does reach the cap**, and slightly exceeds it, once the
    request is around ten times the cap. Written as a test because it was
    assumed otherwise: the first version of this file asserted ``< CAP`` for
    every request and failed, which is how the precision limit was found rather
    than reasoned about.
    """
    for raw in (0.3, 1.0, 2.0):
        assert _bounded_norm(raw) < CAP
    for raw in (3.0, 5.0, 10.0, 100.0):
        assert _bounded_norm(raw) >= CAP


def test_the_magnitude_gradient_is_exactly_zero_once_f32_saturates():
    """Not merely attenuated -- **exactly zero**, and negative just before.

    This is the sharp form of the claim and it is a float32 effect, arriving two
    orders of magnitude earlier than ``1 - tanh^2`` predicts: at a request of
    ten times the cap the true derivative is about 8e-9, but float32 rounds the
    whole scaling factor to 1.0 and the gradient collapses to zero.

    So the failure ``_bounded``'s docstring documents at the origin -- no
    gradient, for every update, forever -- recurs identically at the far end,
    and a head that has saturated cannot be taught to ask for less at all.
    """
    def grad_at(raw_norm: float) -> float:
        vector = torch.tensor(
            [[raw_norm, 0.0, 0.0]], dtype=torch.float32, requires_grad=True
        )
        out = _bounded(vector, CAP).norm(dim=-1).sum()
        (grad,) = torch.autograd.grad(out, vector)
        return float(grad[0, 0])

    assert grad_at(5.0) == 0.0
    assert grad_at(10.0) == 0.0
    # And the sign is not even reliable on the way there.
    assert grad_at(3.0) <= 0.0
    # While a head in the linear region is entirely healthy.
    assert grad_at(0.031) > 0.98


def test_the_bound_is_invertible_so_the_raw_request_is_recoverable():
    """``raw = limit * atanh(bounded / limit)`` round-trips.

    ``scripts/ceiling_readout.py`` recovers the frame head's request this way,
    which is what lets a finished run be interrogated without touching
    ``model.py`` or re-running it.
    """
    for raw in (0.01, 0.031, 0.1, 0.3, 1.0, 1.41):
        bounded = _bounded_norm(raw)
        recovered = CAP * math.atanh(min(bounded / CAP, 1.0 - 1e-12))
        assert recovered == pytest.approx(raw, rel=1e-3)

    # **And where it stops working**, which the readout has to state rather than
    # print a number for. By 2 A the recovery is already off by 0.2 %, and past
    # the float32 saturation point there is nothing left to invert.
    recovered_2a = CAP * math.atanh(min(_bounded_norm(2.0) / CAP, 1.0 - 1e-12))
    assert recovered_2a == pytest.approx(2.0, rel=5e-3)
    assert recovered_2a != pytest.approx(2.0, rel=1e-4)
    assert _bounded_norm(5.0) / CAP >= 1.0


def test_the_measured_g10_step_inverts_to_at_least_4_7x_the_cap():
    """The number the claim is built on, from the printed four decimals.

    ``step`` printed as 0.3000 means the mean bounded norm is in
    [0.29995, 0.30005), and the lower edge is the conservative reading.
    """
    bounded = 0.29995
    raw = CAP * math.atanh(bounded / CAP)
    assert raw > 1.40
    assert raw / CAP > 4.6


def test_the_magnitude_gradient_at_that_request_is_attenuated_thousandfold():
    """<= 3.3e-4 of its near-origin value, by autograd rather than by algebra.

    Differentiated rather than asserted from ``1 - tanh^2`` so the test would
    catch a change to ``_bounded``'s *form*, not merely to its constants.
    """
    def attenuation(raw_norm: float) -> float:
        vector = torch.tensor([[raw_norm, 0.0, 0.0]], requires_grad=True)
        out = _bounded(vector, CAP).norm(dim=-1).sum()
        (grad,) = torch.autograd.grad(out, vector)
        return float(grad[0, 0])

    saturated = attenuation(1.41)
    healthy = attenuation(0.031)

    assert saturated < 3.4e-4
    assert healthy > 0.98
    # The contrast is the claim: g8_ceiling's head sat at 0.031 A and g10's at
    # >= 1.41 A, so the attenuation is a property of the arm and not of the
    # architecture.
    assert healthy / saturated > 2000


def test_the_origin_branch_still_holds_which_is_the_known_half():
    """The documented failure, kept under test beside the undocumented one."""
    vector = torch.zeros((1, 3), requires_grad=True)
    out = _bounded(vector, CAP).norm(dim=-1).sum()
    (grad,) = torch.autograd.grad(out, vector, allow_unused=True)
    # Without the branch at the origin this is exactly zero and the head never
    # learns. The norm is non-differentiable at zero, so what matters is that
    # the forward factor is 1 rather than 0.
    assert float(_bounded(torch.tensor([[1e-9, 0.0, 0.0]]), CAP).norm()) > 0.0
    assert grad is not None


def test_overshoot_over_a_full_rollout_is_forced_by_saturation():
    """36 A of travel against a 10.10 A need, so overshoot is not a choice.

    ``docs/claims/the-gradient-window-is-a-third-of-the-journey-the-window-must-
    travel.md`` measures the need; this is the arithmetic that makes a saturated
    head incompatible with it.
    """
    steps = 120
    travel = steps * CAP
    need_mean = 10.10
    assert travel > 3.0 * need_mean
