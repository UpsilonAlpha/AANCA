---
type: claim
status: supported
sources: [growth.py]
---

# A regeneration start state has to park the damaged region within the rollout's reach

A premise of regeneration. **It did not hold on 6 Nov and holds now**; the
numbers below are what it looked like broken, kept because they are the only
reason anyone would think to check.

`baseline_coords` already takes an arbitrary bool mask of which residues keep
their crystal pose, so handing it `~damaged` instead of one-anchor-per-run
preserves the context exactly. Measured on 1UBQ, window 40:52, with the other
64 residues native:

    intact residues moved      0.00e+00 A     context exactly preserved
    CA-CA inside the window    3.804 A        ideal geometry, as designed
    damaged residues moved     73.96 A mean   parked past the 36 A ceiling
    CA-CA at the N-ward join   64.78 A        not hinged off its neighbour

**The walk order is wrong for this use.** `baseline_coords` runs a walk out of
every anchor and later walks overwrite earlier ones, so a window surrounded by
anchors is last written by a walk from the *furthest* anchor in its run rather
than the adjacent one. With one anchor per run that is harmless -- there is
only ever one walk -- which is why it has never shown up.

So a regeneration start state needs the walk to begin at the intact residue
adjacent to the window. That is a change to anchor ordering, not to geometry.
Both joins should stay open afterwards so the rule has to close them: loop
closure is part of the task rather than something to hand it.

**Fixed on 7 Nov.** A C-ward walk now stops at the next anchor, and the N-ward
walks run afterwards onto residues no C-ward walk claimed -- so a non-anchor
residue is placed by walking C-ward from the nearest *preceding* anchor in its
run, or N-ward from the nearest following one when it has none. At one anchor
per run that is byte-identical to the old behaviour, which is what makes the
6 Nov arms still reproducible; `tests/test_regeneration.py` pins both the fix
and the equivalence. The window now joins its N-ward neighbour at 3.804 A and
starts inside the 36 A ceiling, over six different seeds.

The C-ward join is deliberately left open: closing it is the task, and a start
state with both joins closed would already be the answer. One consequence to
read carefully -- the bond term is high on a regeneration run by design,
because that open join is a broken peptide bond, so `bond` is not comparable
with a reconstruction run's either.

**On check 1's finding that the only observation beneath this is `ours`:** it
is, and it stands. The claim is about our own rollout's reach and our own
walk's output, so our own measurements are the subject. The per-step cap is not
a calculation at all -- it is a design choice read out of `model.py` -- and
`ours` is the nearest class the vocabulary has for that.

## Supported by
- [[the-per-step-translation-cap-is-0-3-angstrom]]
