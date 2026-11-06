---
type: claim
status: unsupported
---

# The rollout can move a residue far enough to reach its native position

A premise of the thesis that was never checked, and arithmetic says it is false
for at least one arm already run.

`AANCARule` bounds every step's translation to `max_translation = 0.3` A and
every step's rotation to `max_rotation = 0.1` rad, enforced by `_bounded`,
which is a tanh on the norm and therefore a true ceiling rather than a soft
preference (`model.py`). At 120 steps:

    120 x 0.3  =  36.0 A    the most a residue can ever move, and only if
                            all 120 steps push in the same direction
     12 x 0.3  =   3.6 A    the most that can move inside the truncation
                            window, which is all the gradient ever sees

The helix baseline is 17.45 A from native, so the rule must produce about 58
steps of coherent motion while being taught through a window covering 3.6 A of
it. **The extended baseline is 36.73 A away, past the 36.0 A ceiling** -- so
`g6_ext` could not have reached the native in 120 steps with a perfect rule.

Marked `unsupported` rather than `void` because the ceiling disqualifies one
arm outright and squeezes the others; whether it is what stopped the 17.45 A
arms is a separate question, and the probe that would settle it has not run.

## Requires
- [[question-does-a-trained-rule-use-its-movement-budget]]

## Refuted by
- [[the-per-step-translation-cap-is-0-3-angstrom]]
- [[the-baselines-are-17-45-and-36-73-angstrom-from-native]]
- [[g6-ext-never-left-the-extended-baseline]]

## Refutes
- [[the-fape-clamp-is-what-gates-the-geometry]]
