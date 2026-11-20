---
type: observation
status: supported
evidence: ours
sources: [33245103, 32904055, 32604037, regrowth_dispersion.py]
---

# Rolled to 600 steps the pool arm sits at 8.5 angstrom where g20 sat at 16.9: the divergence past the training horizon is a protocol fault, not an architectural one

`scripts/regrowth_dispersion.py` on `g21_pool`'s `best.pt` (job 33245103, CPU,
bun154, exit 0), read 20 Nov. This is that arm's **primary** pre-registered
bucket and the second of the three questions that decide the central claim --
*does the rule hold what it builds?*

The comparison is like-for-like and not merely similar: the same six held-out
crops (`3B82 4CQE 1JNP 2FAZ 8QTF 6HJ6`), 8 members per amplitude, the same three
amplitudes, the same 600-step cap and 0.02 A tolerance, and the same
`start bond` column to two decimals -- 7.49 / 7.53 / 7.88 in both runs, which is
the perturbation reproducing exactly.

    amp deg          g20_orient                      g21_pool
             settled RMSF  rmsd_dmg        settled RMSF  rmsd_dmg
          0         5.098     16.93               3.384      8.51
         15         5.139     16.35               4.085      8.72
         45         6.253     16.41               5.400      9.10

    untrained baseline                                      12.23
    g21 at step 120, the trained horizon (VOID, see below)    8.41

## The first bucket fires, and the margin is not small

Pre-registered: *"600-step rmsd_dmg <= 9.0 -- the rule holds its structure. The
statistical-potential programme is unblocked and the ensemble is a distribution
over poses rather than over drift."* At the noise floor it reads **8.51** and at
15 deg **8.72**; 45 deg at **9.10** sits a tenth of an Angstrom into the second
bucket, which is also a pass against the 12.23 A baseline.

The quantity that moved is not the placement accuracy, it is the **horizon
dependence**. g20 went from 7.78 A at step 120 to 16.93 at 600 -- it lost 9.15 A
by being allowed to keep running, ending **worse than leaving the window
alone**. g21 goes from 8.41 to 8.51: **0.10 A over 480 extra steps**. The rule
no longer destroys what it built.

**And this is still not a fixed point. 0 of 48 converged**, at every amplitude,
same as g20. The structure is still moving at 600 steps -- it is moving
*without going anywhere*. That is exactly the property this record named as the
physical one and distinguished from the degenerate case: *"an exact fixed point
is the property the degenerate identity map has... A bounded basin is physical
and distinguishable from never having moved."* A bounded basin is what this
measurement now shows, and `move 0.0000` is by construction
(`move_weight = 0.0`), so it is not the identity map that is producing it.

## The third bucket does not fire, but it is not a clean read either

Pre-registered: *"spread at 120 falls by more than half against g20 -- the pool
bought persistence by narrowing the distribution."* It did not: the settled
spread ratios are **0.66, 0.79, 0.86**, a third at the floor and less as the
start spread grows. Nothing halved.

Two honest qualifications, and the second is the larger.

**The column is not the one the bucket named.** The bucket asked for spread at
step 120; the script prints `start RMSF` and `settled RMSF` at the cap, so the
read is on settled spread at 600. g20's own record read that same column as the
ensemble spread, so the comparison is consistent, but it is not the quantity as
written.

**g20's denominator was drift.** Its record says so in terms: *"the spread at
600 steps measures how fast members come apart, not how many ways the loop can
sit."* So this is not a multiplicity ratio -- it is a diverging spread against a
bounded one, and the fact that it did not halve cannot be turned into *"the
pool kept the multiplicity"*. What can be said is the weaker and still useful
thing: **the narrowing failure mode the bucket was written to catch is not
visible**, and for the first time the spread is measured on a trajectory that
stays put, so it is the first number in this project that could be a
distribution over poses. Whether it is one is unmeasured.

Within g21 the spread behaves like basins rather than like noise: the amplitude
0 floor is 3.384 (members differing by update mask alone), 15 deg and 45 deg
both clear it, and the ratio column saturates 6.80 -> 3.17 as the start spread
grows, which is the signature the script's own legend assigns to basins.

## This reading is CONFOUNDED, as pre-registered -- and the gate on it is void

Said plainly, because three separate things weaken it and one of them is a
logical knot rather than a caveat.

**1. The checkpoint's twist is void.** `g21_pool` settled at twist 13.25 against
g20's 11.24, outside its own 9-12 band, so its placement number is void
([[the-pool-and-length-jitter-undo-the-orientation-terms-hold-on-twist]]). The
pre-registration anticipated exactly this and ruled that twist drift makes a
persistence comparison **confounded, not void**, because the checkpoint is in a
different conformational regime from g20's. That ruling stands and this is
reported under it.

**2. The gate that was supposed to license this reading is unavailable.** The
second bucket -- `settled` at step 120 within 1.0 A of g20's 7.78 -- was written
as a precondition: *"more than an Angstrom of it means the price is too high to
read the first bucket against g20 at all."* g21's 8.41 is 0.63 A, inside
tolerance, but **it is void on the twist guard**, so the gate cannot be read
either way. The first bucket is therefore being read without the check that was
built to license it. Stated rather than waved past: the 8.5 A figure is a real
measurement of this checkpoint, and the claim that it is comparable to g20's
16.9 rests on the crops, members, amplitudes and start bond matching, not on the
horizon-accuracy gate, which does not exist.

**3. Two variables, and the one not usually named is the likelier cause.** g21
changes `pool_probability = 0.5` and `steps_min = 40` together. Of the two,
**the length jitter is the more mechanically direct explanation for
horizon-independence**: drawing each rollout's length from [40, 120] removes the
privileged step the loss is applied at, which is precisely what
horizon-dependence is. The pool's contribution is to supply states the rule has
to be stable *on*. The arm cannot separate them and the record should stop
calling this "the pool" without the qualifier.

## What this does to the programme

**It unblocks the flexibility programme that this measurement blocked on 13
Nov.** The block was that `-log` of a distribution requires a distribution and
what existed was a diverging trajectory
([[the-rule-has-no-attractor-past-its-training-horizon]]). That is now fixed in
the arm that was built to fix it, by the two ingredients that record named as
the missing Growing-NCA ones. E(n)-GNCA reports its attractor held 1,000 steps
past training with a replay pool; this is the same mechanism arriving here.

It also changes the standing qualification on every number in the record. The
honest phrasing was *"the rule passes through 7.78 A at step 120"* because
nothing had been read anywhere but at the horizon and the trajectory did not
stay there. For this checkpoint the trajectory does stay there, so for the first
time a `settled` value is a settled value. It is the void one.

**So the second question is answered yes, at the cost of the arm that answered
it.** The arm holds what it builds and cannot report what it built. What is now
the cheapest thing in the queue is the arm that has both: `g20_orient`'s
orientation term with the jitter and the pool, twist inside the band, and
`eval_ids` pinning PDB ids so it is also the replicate that question 1 needs.

## Scope

  * **One checkpoint, `best.pt`**, best-on-held-out-total, so a slightly
    different rule from the one that wrote the final `history.json` row. Same
    choice as g20's reading, so the comparison is consistent.
  * **6 of 12 crops.** `g17_long` has still not been run through this script, so
    whether a double-ceiling arm holds its structure is still unmeasured and
    still costs only CPU.
  * The 9.10 at 45 deg is outside the first bucket by 0.10 A. Reported as the
    first bucket because the two lower amplitudes are inside it and the 45 deg
    row is the strained one, not because the boundary was moved.
  * Nothing here is replicated, at this or any seed
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]), so
    8.5 against 16.9 is a regime difference and not a significance test. It is a
    large regime difference.

## What this does NOT do to the no-attractor observation

It does not refute it, and a `refutes` edge was written here first and removed.
[[the-rule-has-no-attractor-past-its-training-horizon]] is a correct measurement
of `g20_orient`'s checkpoint, and its own scope says in terms: *"Not a statement
about training... Whether that is fixable by a pool, by a sampled supervision
step, or only by an architectural change is open."* This **answers that open
question** rather than contradicting the measurement, and the comparator numbers
here are that observation's own.

What it does do is bound how far that observation's **title** can be carried.
"The rule has no attractor past its training horizon" is true of the rule g20
trained and is not a property of the architecture, which is how it has been read
in the summary sections of this record. Both readings now have a checkpoint.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[the-growth-objective-supervises-a-fixed-final-step-while-the-track-that-converged-samples-one]]

## Requires
- [[the-rule-has-no-attractor-past-its-training-horizon]]
- [[the-pool-and-length-jitter-undo-the-orientation-terms-hold-on-twist]]
