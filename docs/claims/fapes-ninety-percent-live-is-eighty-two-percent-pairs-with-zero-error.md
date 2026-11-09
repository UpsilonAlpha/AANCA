---
type: observation
status: supported
evidence: experimental
sources: [live_pairs_by_class.py]
---

# FAPE's 90 % live fraction is 82 % pairs at exactly zero error, and the pairs that carry orientation are the deadest class

`scripts/live_pairs_by_class.py --damage 12`, no model at all, 12 held-out
crops at crop 128, helix baseline. Classed `experimental`: the quantity is the
frame-local error between a constructed start state and **deposited crystal
structures**.

    window rmsd_dmg     10.77 A   over 12 crops
    live, all pairs      90.1 %   (530,216 of 588,300)

    pair class                pairs    share     live   mean err
    intact -> intact        483,036    82.1%   100.0%      0.00 A
    intact -> window         50,040     8.5%    55.7%     10.05 A
    window -> intact         50,040     8.5%    31.0%     21.79 A
    window -> window          5,184     0.9%    73.0%      6.73 A

    window-to-intact pairs are 17.0 % of the total and 43.4 % live

A pair is a (frame, point) combination, which is what FAPE averages over, and
each end is independently damaged or intact. The run reproduces the all-pairs
figure of 90.1 % exactly, so this is the same quantity, split.

## What this settles

**The 90 % cannot support "the clamp does not bite in regeneration".** 82.1 %
of all scored pairs are intact-against-intact at a mean error of **0.00 A**.
They are live, and they are live because both ends sit at their crystal poses
and there is nothing to fix. They carry no gradient at any clamp. The statistic
that was read as "FAPE can see the window" is dominated four-to-one by pairs
that say only that the context was not moved.

## What this refutes, and it is my own prediction

Written before the run: the window-to-intact live fraction "at a 13.3 A
displacement should be near zero". **It is 43.4 %.** Not near zero, and not the
90 % either. FAPE does reach the window across a little under half of the pairs
that carry placement information, so the clamp removes a majority of that
signal rather than all of it.

## The result worth keeping: the asymmetry

The two cross classes are the *same pairs* with the roles swapped, and they are
nothing alike:

    intact frame, window point    55.7 % live, mean 10.05 A
    window frame, intact point    31.0 % live, mean 21.79 A

The error is frame-local, so this is a statement about what a *displaced and
mis-oriented* frame does. Read from an anchored frame, a window atom is wrong by
roughly the window's displacement. Read from a window's own frame, the intact
context is wrong by that displacement **plus a rotation lever arm over the whole
crop** -- a frame rotation error multiplies by the distance to the point, and
the intact context extends across all 116 residues. Hence twice the error and
half the live fraction.

**So the pairs that would tell the window its orientation is wrong are the most
clamped-out class in the objective.** That matters because FAPE is the only term
in the objective that constrains orientation at all -- `pairwise_anchor`,
`bond` and `clash` are distances and blind to twist -- and it sits at
`fape_weight = 0.1`. The one orientation signal is demoted by weight and then
69 % removed by the clamp.

This is consistent with the two standing observations about rotation, and was
measured independently of both: `twist` rose 10.4 -> 14.3 in `g17_long` *as
placement improved*
([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]]), and the
rotation half of the equivariance fix is still applied in the lab frame
([[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]).

## Scope

  * **Laptop pool, so the proteins are not Bunya's.** 954 available, held-out
    ids `2DWZ 1P6F 3H6J 4ZSX 5D2U 3C2X 4J29 2EKN 4H80 4GJJ 1XHK 1ZL2`, and the
    10.77 A it reproduces is the voided laptop baseline rather than Bunya's
    12.23 ([[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]).
    The *fractions* are geometric and should travel; the absolute RMSD must not
    be quoted against an arm.
  * **No model.** This is the start state every regeneration arm is handed, not
    a trained rollout. What the live fractions look like at 6.69 A rather than
    10.77 is unmeasured, and they should be much higher.
  * It says nothing about whether opening the clamp *helps*. `g6_open` tested
    that on whole-structure growth and it did not
    ([[g6-open-did-not-move-with-the-clamp-open-a-tenth-of-the-time]]).

## Supports
- [[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]

## Refutes
- [[the-regeneration-baseline-is-10-77-angstrom-and-90-percent-live]]
