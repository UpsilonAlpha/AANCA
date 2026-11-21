---
type: observation
status: supported
evidence: ours
sources: [33249861, 33249842, 33249843, geometric_floor.py]
---

# From a neutral start the rule places a damaged window at 9.02 angstrom where closure and sterics alone reach 9.48, so ten thousand updates bought 0.46 angstrom against a 4.0 angstrom bar

`g22_random` (33249861) completed 20 Nov, exit 0, 10,000 updates, 397.1 min on
bun161. This is the first arm in the project read against **an absolute
threshold fixed before the run** and against **a measured floor**, and it is the
answer to the question Ulban put on 20 Nov: either the AANCAs can regrow a
backbone or they cannot.

**They cannot. Not yet, and not close.**

    on the same 12 held-out crops            fit_all   fit_intact
    g22_random start (untrained, update 250)   12.38        13.44
    GEOMETRIC FLOOR, random, mean               8.60         9.48
    g22_random SETTLED                          8.41         9.02
    geometric floor, best of 10 restarts        5.68         6.00
    bucket 0 PASS line                             --         4.00

`fit_intact` is the column the bar is set on -- the superposition fitted on the
surviving context alone, so the residues whose placement is the question are not
also part of the frame that measures them
(`configs/regeneration.toml`, bucket 0).

## Three readings, and the first is the verdict

**1. Bucket 0 FAILS, by a factor of two.** 9.02 A against a 4.0 A line that was
set from the CA-CA spacing of 3.8 A -- the scale at which every residue is
within one position of where it belongs. At 9.02 A the window is a different
conformation in a different place.

**2. The margin over a native-blind optimiser is 0.46 A.** The floor is
`scripts/geometric_floor.py`: the window's frames moved to satisfy **peptide-bond
closure and steric overlap only**, no distance to native, no torsion prior, no
fragment library, using the rule's own degrees of freedom so that the two search
the same set of structures and differ only in that one of them is told where
native is. Ten thousand updates over 400 proteins bought 0.46 A on top of that.

**The floor is not winning by cheating on chemistry, which is the first thing to
check.** Achieved `bond` on the floor run is **0.0204** against g22's **0.629**:
the optimiser closes the chain roughly thirty times better than the trained rule
while also landing closer to native. And a better optimiser would push the floor
*down*, so this comparison is conservative in the direction that matters.

**3. The rule is not selecting among valid completions.** The best of ten random
restarts reaches **6.00 A** -- so completions 3 A better than the rule's output
exist, satisfy the same constraints, and are findable by chance in ten draws.
The rule is not choosing well among the structures geometry permits; it is
landing near the middle of them. That is the sharper statement of the failure
than the bar is, because it does not depend on where the bar was put.

## The floor is a valid comparator, and that was checked rather than assumed

This is the whole load-bearing step, since a floor measured on a different task
proves nothing. Three checks, all passed:

  * **Same pool, same held-out set.** 23,760 structures available, held-out ids
    `['3B82','4CQE','1JNP','2FAZ','8QTF','6HJ6','2QB5','12XX','3N71','1T7V','3VYC','10XL']`,
    whose first six are the dispersion's own. Bunya's pool, not the laptop's --
    the trap that had the helix baseline quoted as 10.77 for months when Bunya
    reads 12.23 ([[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]).
  * **Same start state.** The floor's start reads 11.92 / 13.09 and g22's own
    first held-out evaluation reads 12.38 / 13.44. Those agree, so the optimiser
    and the rule were handed the same problem.
  * **The chain is closed.** Mean `bond` 0.0204 over 12 crops x 10 restarts, so
    these are valid completions and not a torn chain that happens to score well.
    One crop of twelve (`4CQE`) finished at 0.3237 and is the weakest row.

## What the twist guard did, and why bucket 0 survived it

`g22_random` settled at **twist 12.8 deg**, outside the 9-12 band, so its void
clause fired -- as it did for `g19_nomove` (14.26), `g21_pool` (13.25) and
`g23_localrot` (12.9). **Four consecutive arms have had a placement number
voided by that guard.**

Bucket 0 survives because the clause was narrowed on 20 Nov, before this run,
to void only buckets 1 and 2 -- the comparisons against g20 -- on the reasoning
that *a regime marker cannot invalidate a measurement that is not a comparison*.
An absolute threshold no other arm enters is immune to it. Had that narrowing
not been written, this arm would have joined the other three and the question
would still be open after a fifth 400-minute run.

**The cost of that same clause: bucket 2 is void**, and bucket 2 was the
interesting one -- the fraction of natively-helical window residues landing in
the helical Ramachandran basin, which is the direct test of learned secondary
structure and the thing Ulban asked for on 14 Nov. Voiding it on `twist` was
over-broad of me: twist is the geodesic error of a bonded pair's *relative
rotation* and the basin fraction is a *local torsion* readout, so they are not
the same quantity and the second did not need protecting by the first. The
readout can still be computed from the checkpoint as a diagnostic; it is not
pre-registered evidence and should not be quoted as though it were.

## What this does and does not retire

**It does not retire g20's 7.78 A.** That arm ran a helix start, where the
window begins with its secondary structure handed to it, and its number is
`fit_all`. Both differences flatter it and neither is measured here.

**It does bound what the 7.78 was.** g22 is g20 with one setting changed, and
from a neutral start the same rule lands at 9.02 A `fit_intact` -- 0.46 A from
what geometry alone forces. So the project's headline rested on a start state
doing a large share of the work, which is what [[the-start-state-hands-the-rule-its-secondary-structure]]
predicted and this is the measurement of it.

**And the discipline clause applies**: bucket 0 was registered with *"if it
fails, we do not add terms and re-run."* A local rule with this information,
this objective and this much training does not place a backbone. That is the
finding.

## The floors were re-measured on 21 Nov, and the margin is STILL not quotable

The crop-selection fault was fixed and both floors re-run (33282053 random,
33282054 helix, both exit 0). The fix is confirmed on the cluster, not just
locally: **the helix start now reads 12.23 A `fit_all`**, which is the recorded
Bunya baseline to the digit.

    corrected floors              start fit_all/int    floor fit_all/int   best_int
    helix                             12.23 / 13.26       6.20 /  6.73       6.73
    random                            14.76 / 16.28       9.21 / 10.05       5.91

**But a second comparability fault is now visible, and it runs the other way.**
The floor still picks its own crops by reshuffling a pool slice, so it is not
measured on the crops the arm was judged on. The random floor's start state is
**16.28 A `fit_intact`** against `g22_random`'s own untrained start of **13.44**:
the floor is being computed on **materially harder windows**, which inflates it
and therefore flatters the rule. Before the crop fix the two happened to agree
(13.09 against 13.44), which is why this was invisible.

**So no floor-versus-arm margin in this record should be quoted right now** --
not the original 0.46 A, not the 0.99 A that `g24_frozen` appears to have, and
not the 1.56 A that 8.49 against 10.05 would suggest. All three compare numbers
measured on different windows. `scripts/geometric_floor.py` is being given the
same checkpoint-driven crop selection `scripts/context_drift.py` already uses
(`eval_indices` off `best.pt`), and the floors re-run on each arm's own crops
are what settle it.

**And one number demands that re-run rather than waiting for it.** The helix
floor is **6.20 A `fit_all`**, while `g20_orient` -- the arm that produced this
project's headline, on a helix start -- settled at **7.78 A `fit_all`**. If that
survives a same-crops comparison, then **a network-free optimiser beats the
project's headline result**, and the 7.78 was at or below the floor: no signal at
all rather than a small one. Different crops, so it is a hint and not a finding,
but it is the sharpest open question in the record and it is a CPU job to answer.

## Scope

  * **One seed, one run.** No arm in this project has ever been replicated
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]), so
    0.46 A has no error bar either. The *sign* is what carries here, not the
    magnitude, and a 0.46 A margin is inside the noise of anything unmeasured.
  * **BOTH FLOOR RUNS ARE ON THE WRONG CROPS AND ARE BEING RE-MEASURED
    (21 Nov evening).** `geometric_floor.py` seeded a fresh rng per crop, and
    that rng picks the crop's SEED RESIDUE -- a crop being a spatial ball around
    it -- so both runs selected different 128-residue regions of the same
    proteins than `baseline_probe.py` and the trainer, which share one stream.
    Same held-out ids, different balls, different windows. The fix reproduces
    `baseline_probe`'s helix start digit for digit (10.77 / 11.83 locally).

    **It was a comparability fault, not an easier task**: over 46 crops the two
    conventions agree inside their standard error (helix `fit_all` 11.41 +- 0.78
    shared against 12.11 +- 0.77 fresh), and with a per-crop sd of ~5 A a
    12-crop draw moves the mean ~1.5 A on sampling noise alone -- which is how
    the helix run read 7.67 A against the recorded 12.23. The five crops under
    1.5 A are genuine: an ideal-helix walk reproduces a natively helical window
    to ~1 A, and such crops appear under both conventions.

    **Expected effect on the number above**: locally the fix moves the random
    floor `fit_intact` from 10.91 to 11.20, so **9.48 A should rise by ~0.3 A
    and the rule's 0.46 A margin should shrink to ~0.2-0.3 A** -- against the
    rule, so the verdict stands and strengthens, and bucket 0 still fails by a
    factor of two. **Quoted as an estimate until the cluster re-runs land.**
    The margin was already inside single-seed noise and is now more so.

    The helix run also dropped `--iters 1500` and ran at the default 600, which
    is a separate and smaller fault; it is being re-run with the iterations the
    random run used.
  * **The floor is one optimiser configuration**, Adam at lr 0.03 for 1500
    iterations. It is an upper bound on the true geometric floor, so the real
    margin over geometry is at most 0.46 A and probably less.
  * `g23_localrot`'s own number is void on the same guard: `rmsd_dmg` 8.33
    `fit_all`, twist 12.9. It predates the `fit_intact` column, so it has none.

## Supports
- [[the-start-state-hands-the-rule-its-secondary-structure]]

## Refutes
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Requires
- [[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]
