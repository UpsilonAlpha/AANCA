# What this map argues

AANCA's thesis is that **one local per-residue update rule can build a protein
backbone**. This directory records what the project believes about that and why,
as argmap notes: `python check.py` validates them.

**`HANDOFF.md` remains the journal and is never migrated into here.** Its
chronology is itself evidence -- when a prediction was timestamped, how long a
wrong finding stood -- and every node here cites the jobs or source files its
numbers came from rather than restating the history.

The map exists because this project has now twice been bitten by exactly the
failure argmap is for: a correction written into a *later* HANDOFF section while
the retracted half stayed greppable. On 6 Nov a wrong twist comparison was
committed and corrected within the hour
(`the-orientation-term-buys-22-degrees-of-twist`), and an identity-starvation
hypothesis stood for six hours on a number measured in the wrong place
(`the-identity-gradient-starves-the-geometry`). Both are kept here as `void`
with their refutations attached as edges, so the refutation travels with the
claim.

**Sources** are SLURM job ids (`32241722`, which also names the log file under
`logs/`) or a source file where the number is a design fact rather than a
measurement (`model.py`). `source_field` is `sources`; see `argmap.json`.

**Expect check 1 to report that nearly every claim rests only on `ours`
evidence.** That is true and worth seeing: apart from the baseline distances,
which are measured against deposited structures and classed `experimental`,
every number in this project comes from our own training runs. The validator
saying so on every claim is the hierarchy working, not a defect to engineer
around.
