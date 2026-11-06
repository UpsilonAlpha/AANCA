---
type: observation
status: supported
evidence: ours
sources: [32215014]
---

# g6_self spent 8,000 updates inside 0.09 A of its baseline while its bond term rose 80-fold

The reference arm: `--birth none --initial helix`, identity weighted 93.7.
8,000 updates, exit 0, 2 updates skipped for a non-finite gradient.

    updates    rmsd   twist    bond
       500   17.45    64.4   0.006
      3000   17.47    67.0   0.341
      5000   17.50    66.0   1.580
      7500   17.53    67.0   4.757
      8000   17.44    65.2   0.489

RMSD spans 17.44 to 17.53 about a baseline of 17.45, and twist 64.4 to 67.0
about 64.4. **That is noise around the baseline, not progress** -- a partial log
read at 6,500 updates was described in the journal as "motion against a control
that has none", and the full run does not support that reading.

The column that does move is `bond`, from 0.006 to 4.757. The rule is
progressively breaking the exact peptide geometry the baseline handed it and
buying no fold with the damage. `g6_ext` reproduces the same drift (0.004 to
5.415), so it belongs to the identity term rather than to the helix, and
`g6_noid` with identity removed holds bond at 0.002 throughout.

## Supports
- [[the-rule-keeps-whatever-conformation-it-is-given]]
- [[the-identity-term-perturbs-geometry-rather-than-helping-it]]
