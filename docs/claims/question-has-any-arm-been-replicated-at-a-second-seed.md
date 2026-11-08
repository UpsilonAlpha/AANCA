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
