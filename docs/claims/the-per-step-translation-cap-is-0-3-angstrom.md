---
type: observation
status: supported
evidence: ours
sources: [model.py]
---

# Every step may move a residue at most 0.3 A and rotate it at most 0.1 rad

`AANCARule.__init__` defaults: `max_rotation = 0.1`, `max_translation = 0.3`.
Both are applied by `_bounded`, which scales the vector so its norm never
exceeds the limit -- a tanh on the norm, chosen so the bound is smooth and a
head asking for too much still has a gradient. It is therefore a **true
ceiling**, not a soft preference the rule can learn to exceed.

Classed `ours` because it is a design choice of ours, not a measurement of
anything.

The consequence is arithmetic: over `steps` applications the furthest a residue
can travel is `steps x 0.3` A, and only if every step pushes the same way. At
the 120 steps these runs use that is 36.0 A, and inside the 12-step truncation
window that the gradient can see, 3.6 A.

## Supports
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
