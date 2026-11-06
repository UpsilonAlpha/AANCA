# What this map argues

AANCA's thesis is that **a local per-residue rule can regenerate a damaged
region of a protein from the context that survived** -- the Growing-NCA
objective applied to protein geometry. The point is design: delete the part of
a protein you want to change and have the rule fill it back in from local
information. The two applications driving it are regrowing a designed binder's
interface against its real target, and growing a scaffold around a known
peptide motif.

This directory records what the project believes about that and why, as argmap
notes. `python docs/claims/check.py` validates them; the generic tool is
vendored unchanged in `../argmap/`.

**`HANDOFF.md` remains the journal and is never migrated into here.** Its
chronology is itself evidence -- when a prediction was timestamped, how long a
wrong finding stood -- and every node here cites the jobs or source files its
numbers came from rather than restating the history.

## Why the map exists

This project has been bitten twice by exactly the failure argmap is for: a
correction written into a *later* HANDOFF section while the retracted half
stayed greppable. On 6 Nov a wrong twist comparison was committed and corrected
within the hour (`the-orientation-term-buys-22-degrees-of-twist`), and an
identity-starvation hypothesis stood six hours on a number measured in the
wrong place (`the-identity-gradient-starves-the-geometry`). Both are kept here
as `void` nodes carrying their own `## Refuted by`, so the refutation travels
with the claim.

The map earned itself within the hour. It was first rooted in the wrong thesis
-- "one local rule can build a protein backbone", inferred from the last three
weeks of work instead of read from the README -- and getting that wrong meant
every node's relevance was being judged against a question the project is not
asking. The re-rooting then exposed
`the-four-arms-measured-the-architectures-ceiling`: a claim nobody had ever
stated plainly, which is how it survived a day of being assumed. It is now
`void`.

## Conventions

**Sources** are SLURM job ids (`32241722`, which also names the log file under
`logs/`) or a source file where the number is a design fact rather than a
measurement (`model.py`). The field is `sources`; see `argmap.json`.

**Expect check 1 to report that two claims rest only on `ours` evidence.** They
are claims *about our own model's gradient flow*, so our own measurements are
the subject rather than a weak proxy for an experimental one, and each body
says so. Apart from the baseline distances -- measured against deposited
structures and classed `experimental` -- every number in this project does come
from our own runs, and the validator saying so is the evidence hierarchy
working rather than a defect to engineer around.

**Check 4 is skipped**: no retractions index yet. Given the two corrections
above, adding one is worth doing.
