---
type: observation
status: supported
evidence: ours
sources: [rollout.py, train_growth.py, regeneration.toml]
---

# The one Growing-NCA ingredient whose justification may not transfer is also the one setting no config records

Found 13 Nov, answering "could stochastic updates be interfering with no real
benefit?". The honest answer is **nobody can tell, because the knob is not
reachable from a config.**

## The finding

`RolloutConfig.update_probability` defaults to **0.5** and decides what
fraction of residues tick each step. `scripts/train_growth.py` builds its
`RolloutConfig` at line 190 and **never passes it**:

    update_schedule=args.update_schedule,     <- a config key
    coupling_radius=args.coupling_radius,     <- a config key
    (update_probability absent)               <- a dataclass default

So it has been 0.5 in **every arm this project has run**, and it appears
nowhere in `configs/regeneration.toml`, nowhere in `build_schema()`, nowhere in
`HISTORICAL_DEFAULTS`, and in no test. It is invisible to `--diff`, to
`snapshot()` and to `resume_drift` -- all three of which were built this week
precisely so that a result-affecting value could not live somewhere a diff
cannot see it
([[the-objective-should-restrain-the-output-not-teach-the-placement]]).

`update_schedule = "bernoulli"` **is** in the config, which is what made this
hard to spot: the schedule is recorded and its single parameter is not.

## Why it matters more than an ordinary unrecorded default

`update_mask` returns all-ones when `probability >= 1.0`, so **this one setting
is the difference between a stochastic rollout and a deterministic one** --
and in a `birth = none`, `identity_weight = 0` regeneration rollout it is the
*only* remaining source of nondeterminism once the damaged window is fixed.
Which means it silently decided the design of the dispersion measurement
([[question-does-regrowth-dispersion-measure-conformational-freedom]]).

## And its Growing-NCA justification may not transfer

Stochastic asynchronous update is inherited from Growing NCA, where it is a
**symmetry-breaking** device: grid cells are genuinely identical -- same
weights, same neighbourhood structure, no per-cell identity -- so a synchronous
step makes every cell in an identical local configuration do an identical
thing. Asynchrony is what stops that.

**Residues are not symmetric in that way.** Each has a different neighbourhood
geometry, a different closure-separation block, a different position along the
chain, and a different sequence context once identity is on. The symmetry
asynchrony exists to break may simply not exist here, while the cost is real:
a random mask inside a BPTT window adds gradient variance, and at 0.5 it ticks
about three times more residues than the `"independent"` schedule allows, which
is the quantity the oscillation worry is about (`rollout.py` records that
measurement on 1UBQ and 193L).

Note the contrast with the other schedule. `"independent"` -- a fresh maximal
independent set of the coupling graph, chromatic Gibbs, so no two residues that
could touch update from the same stale neighbourhood -- has a **protein-specific**
justification and was reasoned out for this problem. `"bernoulli"` at 0.5 is
the inherited one.

## The test, and it is one key

Expose `update_probability` as a config key with historical default 0.5, then
one arm at **1.0** off `g20_orient`: a fully synchronous, deterministic
rollout. Same crops, same budget, one setting. It answers three things at once
-- whether asynchrony buys anything on this task, whether the gradient variance
it adds was costing accuracy, and whether a deterministic rollout is available
as the baseline the dispersion control needs.

Exposing it also makes every past arm's value *recorded* rather than inferred
from a dataclass, which is the standard the config port set and this setting
escaped.

## Scope

An observation about what the code does and what the configs contain, not a
measurement of the setting's effect. **No arm has ever run at a value other
than 0.5**, so whether asynchrony helps, hurts or does nothing here is
unmeasured. The Growing-NCA argument above is a reason to test it, not a result.

## Supports
- [[question-does-regrowth-dispersion-measure-conformational-freedom]]

## Requires
- [[question-is-two-thousand-updates-enough-for-any-arm-to-have-converged]]
