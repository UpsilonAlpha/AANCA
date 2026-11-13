---
type: observation
status: supported
evidence: ours
sources: [32505354, 32604034, 32604037, 32904055]
---

# The missing evaluation rows are a skipped update taking its evaluation with it, and four runs now show it

Open since 9 Nov, when `g17_long` came back with 38 history rows instead of 40
and the pre-registered `settled` pair did not exist. Diagnosed 13 Nov from
`g21_pool` failing the same way at a *different* update, which is the clue the
three identical failures could not give.

## The mechanism

`scripts/train_growth.py:1327`:

    if update % args.eval_every == 0 or update == args.updates:
        scores = score(rule, dataset, eval_indices, args, eval_config)

That is **inside the training loop, after every `continue`**. So an update that
skips its training step skips its *evaluation* as well, and the history row for
that update is never written. The `continue` paths are:

  * the crop failed to load (silent)
  * `topo.n_residues < 12` (silent)
  * `not torch.isfinite(terms.total)` (silent)
  * a non-finite gradient (**prints**, at counts 1, 10, 100 and every 500)

## Why three runs missed the same two updates and the fourth missed another

g17, g19 and g20 all ran at `seed = 0` on the same pool, so they draw the
**same crop sequence** and hit the same unusable crop at the same update --
hence 5750 and 9750 absent from all three, which looked like a property of those
update numbers and is not.

`g21_pool` is also `seed = 0`, but pool resumption consumes extra `rng` draws,
so its crop sequence diverges. It is missing **1250** instead, and nothing else
below 1750. Same mechanism, different victim, which is what rules out any
explanation tied to the update number itself.

Confirmed in the log and not only in `history.json`: the `.out` file prints
`held out after` for 250, 500, 750, 1000 and 1500 and **not for 1250**, so the
evaluation genuinely never ran. No skip message appears anywhere in the log, so
it was one of the three silent paths rather than the non-finite gradient one.

## It has cost three pre-registrations and nobody noticed for four days

Every arm since g17 pre-registered `settled` as the mean at updates 9750 and
10000, and in all three cases 9750 did not exist, so the rule applied was the
mean at 9500 and 10000. Recorded as a departure each time
([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]],
[[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]])
without the cause being found, because three runs agreeing looked like a fact
about the schedule rather than a reproducible bug.

## And my own pool code made it worse

`g21_pool` added **two new silent `continue` paths** to that loop -- a pool
entry whose crop fails to load, and one whose crop does not reproduce the stored
topology -- in a loop where a silent `continue` costs an evaluation. That is
strictly the wrong shape for those two cases: a failed *resume* should fall
through to a fresh baseline draw, which costs nothing, rather than discard the
update.

## The fix, in two parts, and why neither is applied yet

  1. **Ungate the evaluation.** Whether the held-out score is taken is a
     property of the update *number*, not of whether that update's training step
     succeeded. Either evaluate before the `continue`s or carry an "eval due"
     flag past them.
  2. **The pool's failure paths should fall back, not skip.** Drop the entry and
     draw a fresh crop in the same iteration.

**Not applied while 32904055 is running.** The chain resubmits with whatever is
in the tree, so a code change landing mid-run would splice two versions of the
trainer onto one curve if it ever reached segment 2 -- the failure
`resume_drift` was built for, except that check guards *settings* and not code.
At ~2.4 s/update the run needs ~6.7 h against a 24 h segment, so it should not
chain; the fix waits anyway, because "should not" is not "cannot".

## Supports
- [[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]
