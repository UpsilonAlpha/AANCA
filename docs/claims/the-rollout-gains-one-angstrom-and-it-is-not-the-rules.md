---
type: observation
status: supported
evidence: ours
sources: [32453593, 32453594, 32453595, 32453659]
---

# An untrained rule closes the same 1.01 Angstrom and scores the same 12.26, so neither number is the rule's

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

## The untrained control settles it, and it settles more than the gain

Pre-registered above as the test that would close this column: an
`--untrained --perturb 0.05` rule should gain the same amount if the gain is not
the rule's. Job 32453659:

    rule                      step     align    gained   rmsd_dmg
    g8_ceiling  (trained)    0.0309   -0.259     1.02      12.25
    g11_split   (trained)    0.0263   -0.189     1.02      12.26
    g12_trunc   (trained)    0.0139   -0.015     1.01      12.26
    UNTRAINED, perturbed     0.0920   -0.344     1.01      12.26

**An untrained rule gains the same angstrom and arrives at the same RMSD.** Its
step is 3.5x g11's and 6.6x g12's, its `move` loss is four times theirs
(1.93 against 0.39-0.51), and it has never seen a gradient. Across a sevenfold
range of step size and every bearing from -0.02 to -0.34, `gained` is
1.01-1.02 A and `rmsd_dmg` is 12.25-12.26.

So the column is closed, and so is a bigger one: **`rmsd_dmg` on this task has
almost no dynamic range under any rule.** 12.25 A is not an attractor that
training finds -- it is what the metric reads for a rule picked at random, and
[[both-arms-park-at-12-23-angstrom-by-update-250]] now has an untrained control
parked in the same place. A damaged window's final RMSD is set by where the
damage put it, not by what the rule does with it, which is why nine arms have
read 12.18-12.85 whatever was changed about them.

That is the clearest statement of the project's position to date, and it is a
statement about the measurement rather than about the architecture. The capacity
question [[question-can-the-architecture-place-residues-under-any-supervision]]
cannot be answered with this quantity, because this quantity cannot tell a
trained rule from an untrained one.

What the 1 A itself is: the window's rigid-body component under
re-superposition is the remaining suspect -- `rmsd_dmg` fits on the surviving
context and reads on the window, and moving every damaged residue by *something*
can reduce a fitted residual without any residue going anywhere useful. Worth
knowing, but no longer load-bearing.

## Supports
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[the-equivariance-fix-did-not-release-a-bearing]]
