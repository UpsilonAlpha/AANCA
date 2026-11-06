---
type: question
status: open
---

# Should a secondary model or search calibrate the rollout parameters, rather than hand-tuning them?

Ulban's proposal, 7 Nov, recorded as a question because it is not yet a claim
the project makes: **if parameters keep needing hand-calibration, that is itself
the finding, and the calibration should be automated rather than repeated.**

The case for it is the count. Numbers that have had to be chosen by measurement
rather than derived, each costing a round trip:

    parameter            chosen           what settled it
    clash tolerance      0.5 A            a crystal's H-bond floor; 1.5 blind,
                                          0.25 worse than 0.5
    fape_weight          1.0 -> loose     four arms, and a wrong comment
    truncate             12, suspect      3.6 A against a 10.1 A journey
    max_translation      0.3 A            disqualified g6_ext at 120 steps
    identity_weight      93.7             inherited, never ablated
    damage               12 residues      arbitrary, never varied

Six, and at least four were wrong or unexamined when first set. That is the
pattern the proposal is aimed at.

**What makes it non-trivial here**: the objective is a fixed point reached over
a step count the rule was never told, so a parameter that looks better at 2,000
updates can be worse at 8,000, and the quantity to optimise -- `rmsd_dmg`
against a model-free baseline -- takes an hour of GPU to evaluate once. A naive
sweep is unaffordable on four QOS slots. What fits that shape is something
sample-efficient over a handful of parameters: a small Bayesian search, or a
bandit over arms that reports which parameter the objective is actually
sensitive to.

**What would have to be true first**, and is not yet: a single scalar worth
optimising. `rmsd_dmg` is the current candidate and
[[the-objective-should-restrain-the-output-not-teach-the-placement]] argues it
must *not* become the pass/fail criterion, because a structure that is valid,
closed, clash-free and 10 A from the crystal is the better-than-original
outcome the project wants and the failure mode look identical. Automating a
search against the wrong scalar would industrialise that confusion rather than
resolve it.

So the honest order is: answer
[[question-can-the-architecture-place-residues-under-any-supervision]], decide
what "good" means for a regenerated window, and only then automate the search
for it. Recorded now so the idea is not lost, and so the parameter count above
keeps being visible as it grows.

## Supported by
- [[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]
