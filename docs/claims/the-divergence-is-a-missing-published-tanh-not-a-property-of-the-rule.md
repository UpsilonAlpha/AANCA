---
type: observation
status: supported
evidence: literature
sources: [gengala/egnca, vgsatorras/egnn, arXiv:2102.09844]
---

# The port's divergence is a tanh we left out, and four other published settings the paper never printed

**The port was faithful to the paper and the paper is not the method.** The
divergence measured in
[[the-published-rule-ported-verbatim-moves-the-window-six-angstrom-in-250-updates]]
is not a property of E(n)-GNCA at protein scale. It is what the rule does when
`phi_x`'s output is **unbounded**, and in the authors' own code for *this exact
task* it is bounded by a tanh that is **on by default**.

## The mechanism, measured here, and why a bound dissolves it

A stabilisation agent instrumented 1,000 updates and found the coordinates --
not the weights, not the gradients -- going non-finite at step 5 to 9 of 25.
Weights stay finite and barely move (`phi_x.2.weight` norm 0.156 -> 0.269 over
1,000 updates); the non-finite-gradient counter fired **zero** times.
`||x_i - x_j||^2` enters `phi_m` unnormalised, so `phi_x` is close to linear in
the squared distance -- measured at `h = ones`, update 1,000:

    d (A)     3.8    6.0    8.0   10.0   15.0   20.0   33.0   50.0
    phi_x    0.02  -0.04  -0.13  -0.27  -0.73  -1.42  -4.07  -9.49

i.e. `phi_x ~ -0.0038 d^2`. One step maps an edge of length `d` to
`|1 + phi_x| d`, which **contracts while `-2 < phi_x < 0` and expands once
`phi_x < -2`, at about d = 23 A** -- and expanding raises `d^2`, which raises
`|phi_x|`, which raises `d`. Doubly exponential: 33 A to 1e7 A in four steps.

**With the published tanh, `phi_x` is in (-1, 1), so `|1 + phi_x|` is in (0, 2)
and the runaway branch does not exist.** The failure mode is the *absence* of
the bound, not the rule.

**And the far crops were invisible to training.** An overflowing crop gives a
float32 loss of `inf`, so the trainer's finite-loss guard skips the update
entirely: **~10 % of training rollouts contributed no gradient, at lr 1e-3,
3e-4 and 1e-4 alike.** The rule was never charged for exploding. That is why
lowering the learning rate changed the frequency and not the fact.

## The five departures, from the authors' code for OUR task

Our task is `trainers/geometric_graph.py` (`FixedTargetGAE`), the point-cloud
target run -- not `gae.py` or `dsystem.py`, whose defaults differ. Read from
the public repo `gengala/egnca`:

    setting                  theirs (geometric_graph)   ours        bears on
    ---------------------------------------------------------------------------
    tanh on phi_x output     has_coord_act=True         ABSENT      divergence
    phi_x output layer       weight.data.zero_(),       xavier      divergence
                             bias=False                 gain 1e-3,
                                                        with bias
    h update                 is_residual=True           ABSENT      --
    hidden activation        -act tanh                  SiLU        divergence
    node-feature norm        -nt pn (PairNorm)          ABSENT      divergence

**Three of those five bound something**, and we removed all three while keeping
the one unbounded channel (`d^2`) that drives the feedback.

**The optimiser settings we had were right.** `-lr 1e-3`, `-wd 1e-5`,
`-gcv 1.0`, `-s1 15 -s2 25`, `-ps 256`, `-std 0.5`, `has_attention=False` all
match what the port already had. So the `lr = 0.0003` override added during
stabilisation is a **departure invented to compensate for a missing published
component**, and it should be reverted rather than kept.

**One recorded open question is now closed.** The port note said `h' =
phi_h(h, m)` is implemented non-residually and that "this cannot be settled
from the note alone". It is settled: `-r` / `is_residual` defaults **True** in
this trainer, so the published update is residual.

## What the paper does and does not contain

Checked against arXiv:2102.09844 (Satorras et al., E(n)-GNN) read verbatim:

  * **The paper has none of the five.** Eq. (3) takes `||x_i - x_j||^2`
    unnormalised; eq. (4) uses the raw difference with `C = 1/(M-1)`; `phi_x`
    is specified as a two-layer MLP with a **linear** output. No normalisation
    of the difference, no bound on `phi_x`, no clamp, and **no discussion of
    coordinate explosion for EGNN at all.**
  * **The paper does report this exact failure for a sibling model.** Its
    Radial Field baseline -- also a scalar-weighted difference update -- needed
    a tanh, and the stated reason is instability *"causing this model to
    diverge"* in the dynamical-system experiment (Appendix C.1). So the failure
    we measured is a published failure mode of this family, with a published
    remedy, applied in the paper to a different model and in the authors' code
    to this one.
  * **The reference EGNN repo carries both options off by default**, with the
    docstrings *"We didn't use it in our paper"* for each -- `normalize`
    (`coord_diff / (sqrt(radial).detach() + eps)`) and `tanh`. Its `gcl.py`
    also carries `torch.clamp(trans, min=-100, max=100)` with the author's own
    comment that it is never activated.

**So "ported verbatim" was true of the paper and false of the method**, and the
distinction is the whole content of this note. The equations are four lines; the
working rule is the four lines plus a bound, a residual, a normaliser and an
activation choice, none of which are printed.

## The norm division is NOT taken, and that is the one judgement call

`normalize` exists in both repos -- `coord_diff / (sqrt(coord_radial).detach()
+ 1)` in Gala's `egnn.py` -- and would attack the mechanism at its root by
making the step scale-free. **It is off by default in every model class in
`models.py` and no trainer enables it.** It is therefore not part of the
published method and is not being taken. If the five published settings do not
hold the far crops, it is the registered next thing to try, and it is a change
to the rule.

## What this does not do

  * **It does not make the port's number better.** 8.77 -> 2.63 A at update 250
    stands as measured and stays the only quotable row until the corrected port
    runs. Nothing here is a new result, only a correct account of a failure.
  * **It does not touch any `g*` arm.** All 28 ran `AANCARule`, which has frames
    and no `phi_x` (the frame head and `rigid_relax`, which this rule does not have).
  * **It is read from code, not from a paper**, so it is evidence about what the
    authors ran rather than about what they wrote. Where the two disagree this
    note follows the code, because the code is what produced the published
    numbers.
  * **Single reading of a single repo at `main`**, not at a release tag. The
    repo's own bibtex gives TMLR 2024, but `main` may have moved since.

## Next

  1. **Revert `lr = 0.0003`.** It compensates for the missing tanh.
  2. **Add the five**, as removals of our departures, and re-run at the
     published settings.
  3. The ~10 % skip is then the test: if the bound works, the far crops stop
     overflowing and start contributing gradient, and `n` holds at 15.
  4. **Persistence past the training horizon** remains unmeasured, and their
     code evaluates it to **10,000-100,000 steps** against a 15-25 step
     training horizon -- so the gap is wider than the note estimated.

## Supports
- [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]
- [[the-port-must-reproduce-the-published-result-on-point-clouds-before-it-is-run-on-proteins]]
