---
type: thesis
status: unsupported
---

# One local update rule, applied per residue, can build a whole protein backbone from a baseline

**The root, and deliberately the strongest assertion the project makes.** It
was briefly demoted on 6 Nov in favour of the regeneration claim, which was the
wrong way round: argmap's `supports` is evidence-for rather than entailment, so
the strongest assertion belongs at the root and the achievable results feed it.
Regeneration succeeding would be strong evidence that a local rule can place
residues from local information, without being the whole claim -- so it
`supports` this, not the reverse
([[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]).

`unsupported`: no run has produced a backbone measurably closer to its native
than the conformation it started from. Four arms of 8,000 updates each ended at
their own baseline, and because they tested *this* claim directly rather than
the regeneration one, they are evidence against it and are attached below as
refutations.

What works and is not evidence for this: identity prediction trains to 22 %
recovery with NLL below the ln-20 floor, the self-placing architecture does
what it was built to do, and the rollout, data pipeline and optimiser are
demonstrably functioning. The geometry is the part that has never moved.

Its premises are listed below, and both of the hard ones are failing.

## Supported by
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Refuted by
- [[the-rule-keeps-whatever-conformation-it-is-given]]

## Requires
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
- [[fape-can-teach-the-frame-head-which-way-to-move]]
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
