---
type: observation
status: supported
evidence: ours
sources: [rollout.py, growth.py, train_growth.py]
---

# The mechanism documented as "what forces a fixed point" exists in this repository, is used by the track that converged, and is absent from the track that froze

Found 9 Nov while answering Ulban's question about whether BPTT over a fixed
rollout is the right approach. It is a two-line difference between two code
paths in the same repository.

## The packing path samples the supervised step

`rollout.sample_loss_step`, used by `rollout.training_objective`:

    def sample_loss_step(config, rng=None) -> int:
        """Draw the supervised step uniformly from the back ``loss_window``
        fraction.

        Supervising the *last* step teaches a trajectory of that length.
        Supervising an unpredictable late step is what forces a fixed point.
        """
        first = int(config.steps * (1.0 - config.loss_window))
        return rng.randrange(max(first, 0), config.steps)

`loss_window` defaults to 0.5, so the loss lands on a step drawn uniformly from
the back half of the rollout. The rule cannot know which step will be scored, so
it has to be right at all of them. **This is the track that converged**:
inverse folding reached 38.0 % held-out recovery and flattened.

## The growth path does not

`growth.growth_objective` scores `final = result.state` -- step 120, every
update, no sampling. `sample_loss_step` and `loss_window` appear nowhere in
`growth.py` or `scripts/train_growth.py`. **This is the track that froze**: four
arms of 8,000 updates each ended at their own baseline to two decimal places
([[the-rule-keeps-whatever-conformation-it-is-given]]).

## Narrowed within the hour: our own data contradicts the causal reading

**The freeze cannot be attributed to this.** `scripts/train_growth.py` calls
`growth_objective` and nothing else -- `training_objective` appears zero times
in it -- so *every* growth arm scored the fixed final step. That includes the
arms that **moved**: g13, g14, g15 and g17 all ran with the loss at step 120
every update, and g17 reached 6.69 A
([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]]).

So the fixed supervision step is present in the frozen arms and in the moving
ones alike, and it is therefore not what separates them. What separates them is
already correctly recorded: the task (whole-structure growth under clamped FAPE
versus interior-window regeneration) and the ceilings
([[the-rule-keeps-whatever-conformation-it-is-given]], narrowed 8 Nov).

Stated plainly because it was overstated when this node was written an hour
earlier: the asymmetry between the two code paths is real and its documented
intent is real, but **"the growth path froze because it supervises a fixed
step" is refuted by the arms that did not freeze.**

## What survives: persistence past the horizon, which nothing measures

What the sampled step is actually for is a state that *persists*, and that is a
different question from whether the window moves. Every growth number in the
record is read at step 120 and nowhere else, so the record cannot distinguish a
window that settled in a basin from one that is frozen from one that would drift
away by step 300. g17's 6.69 A is a reading at one step of one rollout.

That makes the cheap first test a **measurement**, not an arm: roll a trained
checkpoint past 120 steps and report the dispersion. E(n)-GNCA reports its
attractor maintained for 1000 steps past the training horizon; nothing here does
the equivalent. If g17's window holds, the fixed supervision step cost nothing
and this node is a curiosity. If it drifts, the sampled step becomes the fix for
a measured problem rather than a suspected one.

## Why the asymmetry is still worth recording

### Why a fixed step is an invitation to build a clock

A rule supervised at exactly step 120 is being asked for one trajectory of one
known length. A rule supervised at an unpredictable late step is being asked for
a *state that persists*. The second is the task; the first is satisfiable by a
schedule the rule times internally -- and timing is available to it even without
a step index, because the morphogen channels and the activation logits are a
state it writes itself and reads back.

So "no step index" does not by itself prevent the rule from learning a clock.
It prevents the clock being *handed over*. A fixed supervision step is an
invitation to build one.

Three further absences, all of them the same lever, and all already noted as
the neural-cubes levers missing from the growth trainer
([[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]):

  * **No sample pool.** `train_growth.py`'s own docstring: *"the cheapest thing
    that can: one structure, one rollout, one backward pass per update, no
    sample pool."* Growing NCA's pool is what makes persistence trainable -- the
    rule meets partially-grown states as inputs and must handle early and late
    states with one set of weights.
  * **Fixed rollout length.** `steps = args.steps`, constant at 120. Growing NCA
    draws its step count per sample, and E(n)-GNCA trains by BPTT over 15-25
    steps with a replay pool.
  * **The gradient reaches 12 of 120 steps.** `truncate = 12` detaches the graph
    every 12 steps and the loss is at the end, so only the final segment carries
    gradient -- through the iteration that E(n)-GNCA's authors identify as the
    source of global reach
    ([[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]).
    Whether the gradient *explodes or vanishes* over 120 steps is not the live
    question, because it never travels 120 steps.

## Scope

An observation about two code paths, and -- after the narrowing above -- **not
an explanation of anything measured.** No growth arm has been run with a sampled
supervision step, and the arms that froze shared the fixed step with the arms
that did not. The claim is the asymmetry and its documented intent only.

Order of work this implies: the dispersion measurement on `g17_long`'s
checkpoint first, because it is CPU-only, needs no queue, and decides whether
this matters at all. A sampled-step arm is worth a slot only if that measurement
says the window does not hold.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Requires
- [[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]
