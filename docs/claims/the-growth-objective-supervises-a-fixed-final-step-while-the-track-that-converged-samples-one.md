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

## Why this is more than a missing flag

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

This is an observation about two code paths, not a measurement of their effect.
**No growth arm has been run with a sampled supervision step**, so the claim here
is the asymmetry and its documented intent, not that sampling would have
unfrozen the four arms. The cheapest test available: `g14_pairwise`'s settings
with a sampled late supervision step, which is one setting and reuses an arm the
config can still express.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Requires
- [[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]
