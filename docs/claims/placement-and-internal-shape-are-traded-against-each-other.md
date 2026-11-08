---
type: claim
status: void
sources: [32489939, 32489940, 32494313]
---

# Retracted: the arms were never shown to be trading placement against internal shape

**Held 7-9 Nov, withdrawn 9 Nov.** The claim was that the three lever arms buy
placement by spending internal shape, on the evidence that all three end above
the untrained helix's `window` of 16.88 and that the three arms' `window`
values order the same way as their placement quality:

    arm             rmsd_dmg settled   window at 2000
    g13_natedge               11.42            24.81
    g14_pairwise              10.91            29.14
    g15_both                  10.24            31.15

Two things are wrong with reading that as a trade.

**It was compared against the wrong point.** 16.88 is the window's value at
*zero* displacement. The arms have moved roughly a sixth of the way from 12.23
to native, and a window in transit contracts
([[a-straight-line-between-two-poses-passes-through-a-contracted-intermediate]]):
on a straight-line walk the internal half rises to 4.95x its baseline before it
falls. Against that comparator the arms sit at 1.47-1.85x, i.e. **better** than
the shortest path at matched displacement. The ordering that looked like a
trade is the ordering of how far each arm travelled.

**The column cannot show a trade even in principle.** `window` is an unsigned
mean squared error over internal CA-CA distances, so a collapsed window and a
stretched one are the same number. A signed, separation-binned readout has not
been taken.

The honest statement is that **nothing is known about the sign or the cause of
the arms' internal-shape number**, and in particular the window-to-window
weight arm has no measured motivation. It may still be worth running; it is not
indicated by this.

## What it cost

One turn of analysis reported to the user as a finding, and a pre-registered
prediction in
[[the-two-levers-together-beat-either-alone-and-never-give-the-gain-back]]
whose causal half ("*because* nothing in either flag charges for internal
shape") was guaranteed by the transit geometry whether or not the objective had
anything to do with it. A prediction that the mechanism's absence would also
produce is not a test of the mechanism.

## Superseded by
- [[a-straight-line-between-two-poses-passes-through-a-contracted-intermediate]]

## Refuted by
- [[a-straight-line-between-two-poses-passes-through-a-contracted-intermediate]]
