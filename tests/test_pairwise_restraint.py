"""The unclamped distance restraint: a gradient over the whole 13 A, not a third of it.

Every regeneration arm scores 12.23 A, which is exactly its own baseline
(``docs/claims/both-arms-park-at-12-23-angstrom-by-update-250.md``). The
suspected reason is the clamp: FAPE clamps at 10 A, ``clamp_max`` is exactly
flat above it, and the ideal-helix window starts ~13.3 A from where it belongs
-- so for most of the journey the loss hands the rule a constant and no
direction (``docs/claims/the-fape-clamp-is-what-gates-the-geometry.md``,
``docs/claims/the-gradient-window-is-a-third-of-the-journey-the-window-must-travel.md``).

`pairwise_distance_restraint` is the term the published equivariant graph NCA
trains on, restricted to the pairs that carry placement
(``docs/claims/an-equivariant-graph-nca-morphs-random-coordinates-into-a-target-point-cloud.md``).
The test that matters here is the interpolation one: it walks the window from
the baseline to the native pose and asserts this term falls at every step,
printing FAPE on the same walk beside it so the flat stretch is visible rather
than argued. The rest pin the things that would make the term silently useless
-- a nan where a zero belongs, and the two halves summed into one number that
cannot say whether the window moved or merely tidied itself up.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca.fape import backbone_rmsd_on, fape as fape_loss
from aanca.growth import (
    GrowthConfig,
    baseline_coords,
    damage_window,
    growth_links,
    growth_objective,
    pairwise_distance_restraint,
)
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

NATIVE = Path(__file__).resolve().parent.parent / "data" / "raw" / "1UBQ.pdb"


@pytest.fixture(scope="module")
def ubiquitin():
    topo, coords = feature_topology(load_protein(NATIVE))
    return topo, coords, growth_links(topo)


@pytest.fixture(scope="module")
def regeneration(ubiquitin):
    """The regeneration start state: a 12-residue hole, everything else crystal.

    Built exactly as ``growth_objective``'s ``damage > 0`` branch builds it, so
    the numbers here are the ones a run is handed rather than a reconstruction
    of them.
    """
    topo, native, links = ubiquitin
    generator = torch.Generator().manual_seed(0)
    damaged = damage_window(topo.n_residues, links, 12, generator)
    assert bool(damaged.any()), "1UBQ has a run long enough for a 12 window"
    anchors = ~damaged
    start = baseline_coords(native, topo, links, anchors, "helix", generator)
    return topo, native, start, damaged, anchors


def test_the_term_is_exactly_zero_on_the_native_structure(regeneration):
    """A crystal structure scores zero, like the bond and clash terms do.

    That is what makes the value physical and readable: this is not a loss
    whose floor has to be measured before it can be quoted.
    """
    topo, native, _start, damaged, anchors = regeneration
    anchor, window = pairwise_distance_restraint(
        native, native, topo, damaged, anchors
    )
    assert float(anchor) == pytest.approx(0.0, abs=1e-8)
    assert float(window) == pytest.approx(0.0, abs=1e-8)


def test_the_baseline_start_state_is_large(regeneration):
    """The ideal helix is far enough out that the term has something to say.

    The threshold is deliberately loose -- the point is the order of magnitude,
    not a pinned value, because the window is drawn from a seed and a different
    one lands somewhere else.
    """
    topo, native, start, damaged, anchors = regeneration
    anchor, window = pairwise_distance_restraint(
        start, native, topo, damaged, anchors
    )
    displacement = float(
        backbone_rmsd_on(start, native, topo, anchors, damaged)
    )
    print(
        f"\nbaseline: anchor {float(anchor):.2f} A^2, window "
        f"{float(window):.2f} A^2, rmsd_dmg {displacement:.2f} A"
    )
    assert float(anchor) > 10.0, "the term cannot see the window"
    assert torch.isfinite(anchor) and torch.isfinite(window)


def test_the_two_halves_are_reported_separately_and_differ(regeneration):
    """Window-to-anchor and window-to-window are different quantities.

    The anchor half is the only one that constrains *placement*: one end of
    every pair sits at its crystal pose. The window half constrains the
    window's own shape, which an ideal helix mostly has already -- so it is
    much the smaller of the two at the baseline, and a single summed number
    would be dominated by the anchor half while still being reducible by
    tidying the shape. Both are returned so a curve can show which moved.
    """
    topo, native, start, damaged, anchors = regeneration
    anchor, window = pairwise_distance_restraint(
        start, native, topo, damaged, anchors
    )
    assert float(anchor) != float(window)
    assert float(anchor) > float(window), (
        "placement error should dominate internal shape error on a helix "
        "baseline; if it does not, the window was not actually displaced"
    )
    # And the window half genuinely ignores placement: translate the whole
    # window rigidly and only the anchor half moves.
    shifted = start.clone()
    shifted[damaged] = shifted[damaged] + torch.tensor([7.0, -3.0, 2.0])
    anchor_shifted, window_shifted = pairwise_distance_restraint(
        shifted, native, topo, damaged, anchors
    )
    assert float(window_shifted) == pytest.approx(float(window), rel=1e-6)
    assert float(anchor_shifted) != pytest.approx(float(anchor), rel=1e-3)


def test_it_falls_monotonically_where_fape_goes_flat(regeneration):
    """**The test this term exists for.**

    Walk the window linearly from the baseline to its native pose and score
    both losses at every step. This term must fall at *every* step, because a
    loss that is flat anywhere over the 13 A the window has to travel supplies
    no gradient there -- which is precisely the complaint against the clamp.
    FAPE is printed on the same walk and is not asserted monotone: it is
    expected to sit near its saturated value over the first part of the journey
    and only start moving once the window comes inside the clamp, and the
    printed pair is the evidence for that rather than a claim about it.

    The interpolated window is not a physical backbone at the midpoints, and
    that is fine: the question is whether the loss surface has a slope along
    the straight line between the two poses, not whether every point on it is a
    protein.

    **The window-to-window half is deliberately not asserted monotone, and
    measured it is not.** On 1UBQ it *rises* from 8.7 to 44.7 over the first
    four tenths of the journey and only then falls: a straight line between two
    poses passes through a contracted intermediate, so the window's internal
    CA-CA distances get worse before they get better. That is the measurement
    that justifies keeping the halves apart rather than summing them. Summed,
    the term is still monotone here but spends only 44 % of its total fall on
    the first half of the journey against the anchor half's 82 %, so the
    internal half is actively cancelling the placement signal in exactly the
    far regime the term was built for.
    """
    topo, native, start, damaged, anchors = regeneration
    alive = torch.ones(topo.n_residues, dtype=torch.bool)

    rows = []
    for step in range(11):
        t = step / 10.0
        coords = start.clone()
        coords[damaged] = (1.0 - t) * start[damaged] + t * native[damaged]
        anchor, window = pairwise_distance_restraint(
            coords, native, topo, damaged, anchors
        )
        rows.append((
            t,
            float(backbone_rmsd_on(coords, native, topo, anchors, damaged)),
            float(anchor),
            float(window),
            float(fape_loss(coords, native, topo, alive)),
            float(fape_loss(coords, native, topo, alive, clamp=float("inf"))),
        ))

    print(
        f"\n{'t':>5}{'rmsd_dmg':>10}{'anchor':>10}{'window':>9}"
        f"{'fape':>8}{'fape_inf':>10}"
    )
    for t, rmsd, anchor, window, fape, unclamped in rows:
        print(
            f"{t:>5.1f}{rmsd:>10.2f}{anchor:>10.2f}{window:>9.2f}"
            f"{fape:>8.4f}{unclamped:>10.4f}"
        )

    anchors_curve = [row[2] for row in rows]
    for before, after in zip(anchors_curve, anchors_curve[1:]):
        assert after < before, (
            "the restraint is flat or rising somewhere on the journey: "
            f"{anchors_curve}"
        )
    assert anchors_curve[-1] == pytest.approx(0.0, abs=1e-8)

    # The clamped FAPE's *fractional* travel over the first half of the
    # journey, against this term's. Not asserted as a threshold, because the
    # number depends on the window drawn -- but printed, because it is the
    # quantity the whole design rests on.
    half = len(rows) // 2
    def travelled(curve):
        span = curve[0] - curve[-1]
        return (curve[0] - curve[half]) / span if span else float("nan")
    lumped = [row[2] + row[3] for row in rows]
    print(
        f"\nfraction of the total fall spent on the first half of the "
        f"journey: anchor {travelled(anchors_curve):.1%}, anchor+window "
        f"lumped {travelled(lumped):.1%}, clamped FAPE "
        f"{travelled([row[4] for row in rows]):.1%}, unclamped FAPE "
        f"{travelled([row[5] for row in rows]):.1%}"
    )


def test_a_degenerate_set_scores_zero_rather_than_nan(ubiquitin):
    """Exactly 0.0, never nan. A nan here would silently poison ``total``.

    None of these is exceptional. An undamaged run is the normal case for a
    crop whose longest covalent run is too short to hold a window, which
    `damage_window` returns an empty mask for on purpose, and the objective
    then adds this term to a loss it is meant not to change.
    """
    topo, coords, _links = ubiquitin
    empty = torch.zeros(topo.n_residues, dtype=torch.bool)
    full = torch.ones(topo.n_residues, dtype=torch.bool)
    one = empty.clone()
    one[5] = True

    for damaged, intact in (
        (empty, full),          # nothing damaged
        (full, empty),          # nothing intact
        (empty, empty),         # neither
        (one, ~one),            # a window of one: no window-to-window pair
    ):
        anchor, window = pairwise_distance_restraint(
            coords + 5.0, coords, topo, damaged, intact
        )
        assert torch.isfinite(anchor) and torch.isfinite(window)
        if not bool(damaged.any()) or not bool(intact.any()):
            assert float(anchor) == 0.0
        if int(damaged.sum()) < 2:
            assert float(window) == 0.0


def test_the_objective_reports_it_at_zero_weight_and_charges_it_only_when_weighted(
    ubiquitin,
):
    """**Measured on every run; added to the loss only when weighted.**

    Those are two separate decisions and job 32405172 made them one. `move` is
    computed `if move_weight:` and `score()` never passed the weight, so the
    held-out `move` column was *structurally absent* rather than unprinted --
    a zero from a branch nobody took, printed as though it were a measurement.
    So the assertion that matters here is the first one: with the weight at
    zero the term still reports a real value, which is what makes an
    unweighted arm comparable with a weighted one.
    """
    topo, coords, _links = ubiquitin
    config = RolloutConfig(
        steps=4, truncate=99, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )

    out = {}
    for name, weight in (("off", 0.0), ("on", 1.0)):
        rule = AANCARule(growth=True, birth="none", identity=True)
        terms, _ = growth_objective(
            rule, coords, topo, config, torch.Generator().manual_seed(0),
            identity_weight=0.0, damage=12, pairwise_weight=weight,
        )
        out[name] = terms

    for name in ("off", "on"):
        assert out[name].pairwise_anchor is not None
        assert float(out[name].pairwise_anchor) > 10.0, (
            f"the {name} arm reports no pairwise value, which is the "
            "32405172 failure"
        )
        assert float(out[name].pairwise_window) > 0.0
    # Reported unweighted, so the two arms agree on the measurement even
    # though their losses differ.
    assert float(out["off"].pairwise_anchor) == pytest.approx(
        float(out["on"].pairwise_anchor), rel=1e-5
    )
    # Off by default: the unweighted arm's loss is the loss every existing arm
    # had, and only the weighted one is charged.
    on_total = float(out["on"].total.detach())
    off_total = float(out["off"].total.detach())
    assert on_total > off_total
    assert on_total - off_total == pytest.approx(
        float(out["on"].pairwise_anchor), rel=1e-4
    )
    assert float(out["off"].fape) == pytest.approx(
        float(out["on"].fape), rel=1e-5
    )


def test_an_undamaged_run_is_charged_nothing(ubiquitin):
    """``damage=0`` leaves the loss identical with and without the weight."""
    topo, coords, _links = ubiquitin
    config = RolloutConfig(
        steps=4, truncate=99, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )
    totals = []
    for weight in (0.0, 1.0):
        rule = AANCARule(growth=True, birth="none", identity=True)
        terms, _ = growth_objective(
            rule, coords, topo, config, torch.Generator().manual_seed(0),
            identity_weight=0.0, damage=0, pairwise_weight=weight,
        )
        assert float(terms.pairwise_anchor) == 0.0
        assert float(terms.pairwise_window) == 0.0
        totals.append(float(terms.total.detach()))
    assert totals[0] == pytest.approx(totals[1], rel=1e-9)


def test_the_term_carries_a_gradient_to_the_window(regeneration):
    """A value is not a gradient, and this is the term's whole purpose.

    FAPE's value at the baseline is a constant in the window's coordinates, so
    its gradient there is zero. This asserts the replacement's is not, on the
    same state.
    """
    topo, native, start, damaged, anchors = regeneration
    moving = start.clone().requires_grad_(True)
    anchor, _window = pairwise_distance_restraint(
        moving, native, topo, damaged, anchors
    )
    anchor.backward()
    grad = moving.grad[damaged]
    assert torch.isfinite(grad).all()
    assert float(grad.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# The window-to-window half, weighted. A SECOND weight, not a widening of
# `pairwise_weight`: the port weights the full pair set (Ulban, 23 Nov,
# docs/claims/the-task-is-to-realise-coordinates-for-a-given-contact-graph.md)
# while every arm g14-g28 charged the anchor half alone, and both have to stay
# expressible. What the measurement above says about the internal half -- that
# a LINEAR walk to the native pose makes it rise from 8.7 to 44.7 A^2 before
# falling -- is a fact about a straight line and not about the loss, so it is
# the reason to watch `pair_window` rather than a reason to leave it unweighted.
# ---------------------------------------------------------------------------


def _regeneration_objective(coords, topo, **weights):
    """One objective evaluation on the regeneration task, seed fixed.

    A fresh rule per call, seeded identically, so two calls differ in nothing
    but the weights handed in -- which is what makes the totals subtractable.
    """
    config = RolloutConfig(
        steps=4, truncate=99, k=24, blind=True,
        growth=GrowthConfig(birth="none", initial="helix"),
    )
    torch.manual_seed(0)
    rule = AANCARule(growth=True, birth="none", identity=True)
    terms, _ = growth_objective(
        rule, coords, topo, config, torch.Generator().manual_seed(0),
        identity_weight=0.0, damage=12, **weights,
    )
    return terms


def test_the_window_half_is_unweighted_by_default_and_the_total_is_unchanged(
    ubiquitin,
):
    """**The compatibility assertion: zero is bit-identical, not merely close.**

    Every recorded arm's objective has to resolve to the sum it actually
    optimised, and the way that goes wrong is a new key that defaults to
    something, or that gets folded into an existing one. So this compares the
    total at ``pairwise_window_weight = 0`` against the total with the argument
    left out entirely -- i.e. against the call every existing arm makes -- and
    requires exact equality of the floats, with a nonzero ``pair_window`` on
    both sides so the comparison is not vacuous.

    Cut the wiring (drop ``+ pairwise_window_weight * pairwise_window`` from
    ``growth_objective``'s ``total``) and this test still passes, which is the
    point of it: it is the half that says the default is inert. The two tests
    below are the ones that fail.
    """
    topo, coords, _links = ubiquitin
    absent = _regeneration_objective(coords, topo, pairwise_weight=0.01)
    stated = _regeneration_objective(
        coords, topo, pairwise_weight=0.01, pairwise_window_weight=0.0
    )
    assert float(absent.pairwise_window) > 0.0, "nothing to be charged for"
    assert float(stated.total.detach()) == float(absent.total.detach())
    # And the two halves are still reported apart, which is what makes "the
    # window tidied itself up while its placement never moved" visible.
    assert float(stated.pairwise_anchor) != float(stated.pairwise_window)


def test_the_window_weight_changes_the_objective(ubiquitin):
    """The setting does what it claims: the total rises by weight x the half.

    Exactly, not approximately -- the term enters ``total`` linearly, so the
    difference between two weights is the measured half times the difference.
    That is the assertion a dead key cannot pass: with the wiring cut the two
    totals are equal and this fails with
    ``0.0 != 16.6...`` on the first comparison.
    """
    topo, coords, _links = ubiquitin
    off = _regeneration_objective(
        coords, topo, pairwise_weight=0.0, pairwise_window_weight=0.0
    )
    on = _regeneration_objective(
        coords, topo, pairwise_weight=0.0, pairwise_window_weight=1.0
    )
    half = float(off.pairwise_window)
    assert half > 0.0
    # Reported unweighted, so both arms agree on the measurement.
    assert float(on.pairwise_window) == pytest.approx(half, rel=1e-6)
    assert float(on.total.detach()) - float(off.total.detach()) == pytest.approx(
        half, rel=1e-4
    )


def test_the_two_halves_are_charged_independently(ubiquitin):
    """``pairwise_weight`` still means the anchor half ALONE.

    The failure this guards is the tempting simplification: make
    ``pairwise_weight`` charge both halves and add a key to switch the second
    off. That would silently change what g14's 0.01 meant. So the four corners
    are checked -- neither, anchor only, window only, both -- and the total
    must be additive across them, which is only true if each weight multiplies
    its own half.
    """
    topo, coords, _links = ubiquitin
    neither = _regeneration_objective(
        coords, topo, pairwise_weight=0.0, pairwise_window_weight=0.0
    )
    anchor_only = _regeneration_objective(
        coords, topo, pairwise_weight=1.0, pairwise_window_weight=0.0
    )
    window_only = _regeneration_objective(
        coords, topo, pairwise_weight=0.0, pairwise_window_weight=1.0
    )
    both = _regeneration_objective(
        coords, topo, pairwise_weight=1.0, pairwise_window_weight=1.0
    )

    base = float(neither.total.detach())
    anchor = float(neither.pairwise_anchor)
    window = float(neither.pairwise_window)
    assert anchor > 10.0 and window > 0.0

    assert float(anchor_only.total.detach()) - base == pytest.approx(
        anchor, rel=1e-4
    ), "pairwise_weight no longer charges the anchor half alone"
    assert float(window_only.total.detach()) - base == pytest.approx(
        window, rel=1e-4
    ), "pairwise_window_weight does not charge the window half"
    assert float(both.total.detach()) - base == pytest.approx(
        anchor + window, rel=1e-4
    ), "the two halves are not charged independently"


def test_both_of_the_trainer_s_objective_calls_pass_the_new_weight():
    """**The 32405172 failure, which was a call site and not a term.**

    ``move`` and ``clash`` were computed ``if weight:`` while ``score()`` never
    passed the weight, so the held-out column was not unprinted -- it was
    structurally absent, a zero from a branch nobody took, printed as a number
    that looked real, and g8_ceiling's first 1,250 updates cannot be read. The
    term being right is therefore not enough: ``scripts/train_growth.py`` calls
    ``growth_objective`` twice, once for the held-out score and once for the
    training step, and a weight that reaches one and not the other makes the
    evaluation a different experiment from the run.

    Read off the source with ``ast`` rather than by running a rollout, because
    the alternative is a GPU and 400 structures. Delete either keyword and this
    fails naming the line.
    """
    import ast

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "train_growth.py"
    ).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "growth_objective"
    ]
    assert len(calls) == 2, (
        f"expected the held-out call and the training call, found {len(calls)}"
    )
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "pairwise_weight" in passed and "pairwise_window_weight" in passed, (
            "the growth_objective call at line "
            f"{call.lineno} does not pass both pairwise weights, so one half "
            "of the term is charged in training and not in the evaluation "
            f"(it passes {sorted(passed)})"
        )
