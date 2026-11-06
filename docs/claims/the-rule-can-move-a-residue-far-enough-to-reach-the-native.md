---
type: claim
status: open
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

**The probe has now run and the answer is: it depends on the arm, which is why
this is `open` rather than settled either way.**
[[the-reach-probes-split-the-three-frozen-arms]] measured `g6_self` moving
7.67 A against a mean need of 8.64 A, in an almost perfectly straight line and
using a fifth of its budget -- so for a helix arm the ceiling is not binding and
this premise effectively holds. For `g6_ext`, 5.74 A against a need of 19.48 A
mean and 69.50 A at worst, it does not.

So the ceiling explains `g6_ext` and explains nothing about the arms that
started 17.45 A out. That matters more than it sounds: it removes the last
mechanical excuse for those arms and leaves the direction of FAPE's gradient as
the only candidate.

## Requires
- [[question-does-a-trained-rule-use-its-movement-budget]]

## Supported by
- [[the-reach-probes-split-the-three-frozen-arms]]

## Refuted by
- [[the-per-step-translation-cap-is-0-3-angstrom]]
- [[the-baselines-are-17-45-and-36-73-angstrom-from-native]]
- [[g6-ext-never-left-the-extended-baseline]]

## Refutes
- [[the-fape-clamp-is-what-gates-the-geometry]]
