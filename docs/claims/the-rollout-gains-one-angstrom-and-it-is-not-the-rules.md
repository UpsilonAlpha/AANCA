---
type: observation
status: supported
evidence: ours
sources: [32453593, 32453595]
---

# Two arms close the same 1.02 Angstrom, crop by crop to within 0.02, so the gain is not the learned rule's

The readout now measures the damaged window's distance from native **at the
start state** (`need`) with the same `ca_rmsd_on` as `rmsd_dmg`, so
`need - rmsd_dmg` is what the rollout bought. On the twelve held-out crops:

    crop     need   rmsd_dmg   gained(g8)   gained(g11)
     180     9.7        9.3       0.33         0.32
     167     8.9        8.8       0.14         0.14
     393    21.4       20.1       1.33         1.33
     264    22.8       20.8       2.00         2.02
     330    17.3       16.3       1.00         0.98
     193    15.3       12.7       2.53         2.54
     348    10.0        9.6       0.35         0.36
     297     0.5        0.5       0.03         0.01
       6    13.1       11.9       1.25         1.24
     319    25.9       23.9       1.97         1.98
     225     3.0        2.8       0.20         0.22
     129    11.3       10.2       1.10         1.10
    mean    13.3       12.3       1.02         1.02

Every crop gains, and the first reading of that was that the arms are not worse
than doing nothing after all. **That reading is wrong, and the agreement is why.**

`g8_ceiling` and `g11_split` are different arms with different weights,
different bearings (`align` -0.259 against -0.189) and step sizes 18 % apart
(0.0309 A against 0.0263 A), and they close the same distance **crop by crop to
within 0.02 A**. A quantity that is identical across two rules that behave
differently is not produced by either rule. It is the same reasoning that made
[[both-arms-park-at-12-23-angstrom-by-update-250]] suspicious -- two objectives
with no placement term in common landing on one number -- applied to the gain
rather than to the remainder.

**So the honest statement is stronger than "the rule makes the window worse":
the rule contributes nothing measurable to closing the gap.** The 1 A that
closes is geometric, and `gained` is a property of the setup.

What the 1 A actually is remains open. The first suspect is the window's
rigid-body component under re-superposition: `rmsd_dmg` fits on the surviving
context and reads on the window, and the rollout moves every damaged residue by
*something*, which can reduce a fitted residual without any residue going
anywhere useful. **An `--untrained --perturb 0.05` run of the same readout
settles it for free** -- an untrained rule with a perturbed frame head should
then gain the same 1.02 A, and if it does, the column is closed as a measurement
of the rule.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[the-equivariance-fix-did-not-release-a-bearing]]
