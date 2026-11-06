---
type: claim
status: void
sources: [32215014]
---

# The identity term's gradient is starving the frame head, so the geometry cannot train

**Held for about six hours on 6 Nov, then refuted by the arm built to test
it.** Kept because the reasoning was sound and the conclusion was wrong, and
because the shape of the error recurs.

The case for it: at initialisation FAPE puts 6.9e-02 on the frame head while
identity puts 1.4e+01 on its own -- a factor of 200 -- and `g6_self` was
sitting at its baseline while identity trained normally. A gradient ratio of
200 between the two objectives looked like the explanation.

**The prediction was recorded before the test: if identity is starving
geometry, `g6_noid` leaves 17.45 A and `g6_self` does not. The reverse
happened** ([[g6-noid-never-left-the-helix-baseline]]). `g6_noid` is perfectly
static with geometry as its entire objective.

The flaw was named in the journal as a caveat at the time and then not weighted
properly: **Adam normalises per parameter**, so a uniformly smaller gradient on
a separate head still takes full-size steps and the ratio cancels. It bites
only on shared parameters. The measurement that would have settled it was the
trunk column, and three attempts at it were invalid
([[a-probe-pointed-at-best-pt-silently-used-its-own-defaults]]) -- so the
hypothesis survived six hours on an init-time number from the wrong place in
the model.

The lesson worth carrying is not "identity was innocent". It is that a ratio
between objectives says nothing until you know whether the parameters are
shared.

## Superseded by
- [[identity-supplies-the-trunk-that-geometry-reads]]

## Refuted by
- [[g6-noid-never-left-the-helix-baseline]]
