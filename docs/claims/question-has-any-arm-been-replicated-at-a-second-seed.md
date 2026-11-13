---
type: question
status: open
---

# No growth arm has ever been run twice, so no difference between arms has an error bar

Every arm in the record is **one run at `seed = 0`, evaluated on 12 held-out
crops**. There has never been a seed replicate of any arm, and no node in this
map quotes a standard error.

That makes every comparison the project rests on a point estimate against a
hand-chosen threshold:

  * g14 settled 10.91 against g15 settled 10.24 -- a 0.67 A gap, no error bar
    on either.
  * g15's pre-registered bucket boundary was 10.70 and `settled` was 10.24, a
    0.46 A margin.
  * The two points `settled` averages on g15 are 0.60 A apart, four times the
    ~0.15 A adjacent-evaluation wobble the two-point rule was written to
    smooth -- noted in that node, and itself a hint that the run-to-run term is
    not small.

The per-crop spread exists in the Slurm logs and has never been reduced to a
number in this map.

## Pre-registration: `g18_seed1`, submitted as 32507434

`g15_both` at `seed = 1`, one setting, 2,000 updates. Running on `bun159` from
2026-11-09 09:00 AEST. Written before any of its evaluations were read.

**The quantity is `spread` = |g18 settled - g15 settled|**, where `settled` is
the mean of `rmsd_dmg` at updates 1750 and 2000 and g15's value is **10.24**.
The *sign* is deliberately not part of the prediction: a replicate at one extra
seed says how big the run-to-run term is, not which seed is better. Four
exclusive buckets, each named for the claim it would cost:

  * **`spread <= 0.15`** -- the run-to-run term is no larger than the
    adjacent-evaluation wobble the two-point rule was written to smooth. Every
    difference in the lever table is then bigger than the noise, and the record
    stands as read.
  * **`0.15 < spread <= 0.46`** -- noise eats into the margin by which g15
    landed inside its pre-registered bucket (10.24 against a 10.70 boundary)
    but not into the g14-to-g15 gap. **Bucket assignments become marginal;
    the ordering of the arms survives.**
  * **`0.46 < spread <= 0.67`** -- g15's "superadditive" bucket is **not
    resolvable by a single run**. The claim then needs replicates rather than
    another lever, and the four-bucket rule that produced it has to be rewritten
    with a stated error term.
  * **`spread > 0.67`** -- the whole lever table sits inside the run-to-run
    term. g13, g14 and g15 are then unordered, and all three "this lever moves
    the metric" readings revert to `open` pending replicates.

**What one replicate cannot do.** Two runs bound a spread; they do not estimate
a standard error, and nothing here licenses a confidence interval or a p-value.
The buckets are deliberately about which *existing* claims survive, not about
establishing a new number.

**Void condition.** If the run does not complete 2,000 updates -- Bunya goes
into maintenance today with no SLURM reservation registered -- `spread` is not
read and the arm is resubmitted with `--resume`.

## Required by
- [[the-two-levers-together-beat-either-alone-and-never-give-the-gain-back]]
- [[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]]
- [[four-arms-break-at-update-1000-and-only-the-amplitude-follows-the-frame]]

## `g18_seed1` ran, completed, and does not answer this question

32507434, `bun159`, 2,000 updates in 30.8 min, exit 0.

    update   dmg   anchor   window   twist   bond   crops
       250 10.23     43.5     36.2    10.3   3.30      10
      1000 10.32     45.1     37.0    10.1   3.27      10
      1250 10.06     42.6     32.8    10.0   2.86      10
      1750 10.32     44.1     31.9    10.4   3.35      10
      2000 10.09     42.1     33.4    10.7   3.38      10

    settled = (10.32 + 10.09) / 2 = 10.21   against g15's 10.24
    spread  = 0.03 A

On the pre-registered buckets that is the first one, `spread <= 0.15`, which
would have meant *the run-to-run term is no larger than the
adjacent-evaluation wobble; the record stands as read*. **It is not claimed,
because the arm does not measure what the buckets were written about.**

### `seed` chooses the held-out structures, so it cannot isolate a replicate

Read out of the two checkpoints rather than inferred:

    g15_both   seed 0   eval_indices [6, 129, 167, 180, 193, 225, 264, 297, 319, 330, 348, 393]
    g18_seed1  seed 1   eval_indices [31, 45, 131, 138, 154, 164, 174, 200, 289, 294, 327, 388]

**Completely disjoint.** `seed` feeds the split as well as the initialisation --
it sits in `WARM_START_MUST_MATCH` beside `split`, `structures` and
`eval_structures`, which is the trainer saying as much -- so changing it changed
*which proteins the metric is a mean over*. g18's rows also report **10 crops,
not 12**: the metric was computable on 10 of its 12, where g15 and g17 got 12
of 12.

So `spread = 0.03 A` is the agreement between two different measurements on two
disjoint structure sets of different size. That is not nothing -- jointly it
bounds *training seed plus held-out set* at 0.03 A, and it says the 12-crop mean
is remarkably insensitive to which crops you hold out. But **the run-to-run
term this node was opened about is still unmeasured**, and two numbers landing
0.03 A apart on different proteins is as easily a coincidence as a tight
instrument.

**This is an instrument fault, and the void condition did not cover it.** The
condition written at submission was "if the run does not complete 2,000
updates". It completed. What went wrong is that the knob does two things, which
no bucket could have caught -- the same class of fault as the voided 10.77 A
baseline, where a crop index meant a different protein
([[the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it]]).

### What a real replicate needs

A second seed that does **not** touch the split: either a separate
`init_seed` used only for weight initialisation and batch order while
`--seed` keeps choosing the structures, or an explicit `eval_indices` that
pins the held-out set so any seed can be varied against a fixed instrument. The
second is the smaller change and also fixes the unrecorded-pool problem, since
a pinned index list is a recorded one.

Until then every difference in the lever table remains a point estimate with no
error bar, and this question stays **open**.

## Supports
- [[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]]
