---
type: claim
status: unsupported
---

# Starting from a known peptide motif, the rule can grow a protein scaffold around it

The second application, and the one that genuinely needs *growth* rather than
regeneration: hold a motif fixed as the surviving context and have the rule
build new structure outward from it.

**`unsupported` and untested**, and the furthest of the two from what is built.
Three reasons, worth separating from the interface case rather than folding in
with it:

The reach ceiling bites hardest here. A scaffold residue may legitimately
belong tens of Angstrom from the motif it grows off, which is the regime where
`steps x 0.3 A` is binding
([[the-per-step-translation-cap-is-0-3-angstrom]]) -- unlike loop or interface
regrowth, where a residue moves a few Angstrom.

There is no native answer to supervise against. Any of many scaffolds would do,
so this cannot be trained by FAPE to a crystal structure, which is how every
arm so far has been trained. It needs a different objective, not a different
start state.

And the structure genuinely grows rather than being repositioned, so the birth
machinery that `--birth none` replaced is relevant again here even though it is
not for regeneration.
