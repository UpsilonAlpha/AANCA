---
type: claim
status: open
sources: []
checked: []
---

# The constant edge feature, not the fold class, is what gave the unconditional map a fixed point

**PRE-REGISTERED 27 Nov, BEFORE THE ARM WAS SUBMITTED, and the pre-registration
is binding.** The arm is `configs/conditioned.toml` `[nolabel]`, which differs
from the measured `[cond]` arm by **exactly one key**, `label_dropout = 1.0`.
Nothing else is written, no code changes, `tag = section` so it lands in
`runs/nolabel`.

## What is being attributed

Job 33834699 ([[a-global-fold-class-label-tells-us-whether-option-2-failed-on-ambiguity-or-on-address]])
produced a result nobody registered: rolled out from the prior with nothing
pinned, at
[[a-local-rule-with-an-attractor-is-a-diffusion-model-without-a-schedule]]'s own
registered 0.1 A/step and 500-step cap, the conditioned checkpoint settles **14
of 15** crops at median `last_move` 0.0969, against the refuted MVP's **2 of 15**
at median 0.5379.

That rollout read the **unconditional row**, which carries no class information,
so the class embedding was not in play while the map converged. Three candidate
causes were left unseparated, and they are named here before the measurement:

  * **A constant learned edge feature.** `EGNCARule`'s `edge_dim` slot now
    carries a non-zero 4-wide vector into `phi_m` beside `||x_i - x_j||^2` at
    every edge at every step. A uniform additive bias inside the message MLP can
    damp the coordinate update, which is exactly what a 5.5x smaller step is.
  * **148 extra parameters**, 2.8 % more capacity.
  * **Where `best.pt` happened to land** -- update 12250 for `cond`, and a
    different update for the MVP.

**`[nolabel]` separates the first two from the class and from nothing else.** It
carries the embedding, the widened `phi_m` input and all 148 parameters, and the
rule sees only the single unconditional row, in training and in evaluation. So:

    cond vs nolabel     ONE key. Isolates the CLASS.
    nolabel vs runs/mvp A different trainer, a different seed draw and 148
                        parameters. Does NOT isolate the edge feature.

The second line is the honest limit of this arm and it is written down first: a
`nolabel` fixed point attributes the convergence **away from the class**, which
is what the title claims, and it does **not** attribute it to the edge feature
rather than to capacity or to the checkpoint's landing point. Separating those
three needs a width sweep, and that arm is not registered.

## Registered predictions

**1. READOUT 0a MUST BE EXACTLY ZERO, and this is the one place in the project
where a zero is the correct answer.** `scripts/cond_readouts.py` computes the
paired per-residue displacement between samples that differ only in the label,
from the same crop and the same noise seed. With `label_dropout = 1.0` the label
never reaches the rule, so the four samples at a position are the **same forward
computation** and their coordinates must agree bit-for-bit: `0a == 0.0`, not
"small".

A non-zero 0a here is a **BUG in the dropout path** -- the class leaking into a
rule that was configured never to see it -- and it voids the arm. This is the
calibration the `[nolabel]` section was written for: the instrument check read
1.7988 A under `[cond]`, and it has never been read against a known null.

**2. The fixed-point count decides the title.**

    nolabel converged >= 12 of 15   ->  SUPPORTED. The class is not the cause.
    nolabel converged <= 4 of 15    ->  REFUTED, and the surprise is larger than
                                        the one it was built to explain: a class
                                        embedding the registered rollout does not
                                        read would be changing the unconditional
                                        row through training alone.
    5 to 11 of 15                   ->  WITHHELD, not split. 15 crops cannot
                                        resolve a partial effect, and no arm in
                                        this project has a second seed, so a
                                        middling count has no error bar to be
                                        read against.

The thresholds are set from the two measured ends -- 14/15 and 2/15 -- with a
gap left in the middle on purpose, because the alternative is reading a
3-crop difference as a mechanism.

## What this arm CANNOT do, stated before it runs

It cannot make option 2 true and it is not an attempt to. The thing `[cond]`
converges to is an all-coil, 0.9951-outlier `irregular` object with a clash rate
**worse** than the MVP's, and a `nolabel` fixed point would be the same kind of
object. **Convergence is a property of the map; being a protein is a property of
where it converges to.** This arm asks only which of two keys owns the first
property, and it is run because an unattributed 5.5x effect sitting in the
journal is how a later arm gets built on a guess.

## supports

- [[a-global-fold-class-label-tells-us-whether-option-2-failed-on-ambiguity-or-on-address]]

## requires

- [[a-probe-pointed-at-best-pt-silently-used-its-own-defaults]]
