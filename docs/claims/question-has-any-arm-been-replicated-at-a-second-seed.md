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

**What settles it:** `g18_seed1` -- `g15_both` at `seed = 1`, one setting, 37
minutes. One replicate does not give a confidence interval, but it bounds the
run-to-run term against the 0.46-0.67 A differences the record is reading, and
it is the cheapest arm available.

## Required by
- [[the-two-levers-together-beat-either-alone-and-never-give-the-gain-back]]
- [[both-levers-move-the-window-and-the-placement-term-holds-more-of-it]]
- [[four-arms-break-at-update-1000-and-only-the-amplitude-follows-the-frame]]
