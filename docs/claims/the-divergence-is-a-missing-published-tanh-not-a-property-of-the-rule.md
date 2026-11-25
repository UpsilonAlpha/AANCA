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

## MEASURED, same day: the five fix the divergence completely, and do not make it learn

Both arms from one scratch config, same seed, same pool, same crops, laptop
CPU, 1,000 updates. `skip/250` is the new permanent per-reason skip counter.

**Five on -- the port:**

    update  total  rmsd_dmg  (fit intact, start)   n  diverged  skip/250  of which inf
       250  7.353     2.921   (3.108,    8.774)   15         0         4              0
       500  7.861     2.952   (3.151,    8.774)   15         0         3              0
       750  8.762     3.264   (3.520,    8.774)   15         0         1              0
      1000 11.680     3.377   (3.450,    8.774)   15         0         2              0

**The four switchable ones off -- the paper's four equations:**

    update  total  rmsd_dmg  start   n  diverged  skip/250  of which inf
       250  7.925     2.913  8.774  15         0        13             12
       500  5.074     2.301  8.529  14         1        39             35
       750  4.787     2.291  7.203  11         4        38             37
      1000  3.418     2.006  6.748  10         5        62             61

**The prediction held exactly.** `n` holds at 15, `diverged` is 0 at every
evaluation, `start` stays flat at 8.774 so there is no survivorship, and
**0 of 1,000 training rollouts went non-finite** against **145 of 1,000** with
the four off. The run's ten skips are all `no_crop` or `no_window` --
properties of the crop, not of the rule. The ~10 % skip rate did not fall, it
went to zero.

**And the second table is the trap this project keeps walking into.** With the
settings off, `total` and `rmsd_dmg` look *better* at every row -- 2.006
against 3.377 at update 1,000. They are a mean over the ten crops that lived,
and `start` walking 8.774 -> 6.748 proves it, because the start state cannot
depend on the weights. The unstable arm appears to win by discarding its hard
cases.

**What the first table does not say.** With the five on the objective **does
not fall**: `total` 7.35 -> 11.68 and `rmsd_dmg` 2.92 -> 3.38 over 1,000
updates. So the divergence is fixed and *whether the rule learns the task is
not answered*. Two reasons not to read failure into it yet, and one reason not
to dismiss it:

  * **1,000 updates is 5 % of the configured 20,000**, and four evaluations are
    not a trend. A full-length run is the measurement.
  * **A rising *training* `total` is expected here.** The replay pool fills with
    rolled-out states, so the sampled states get progressively further from the
    crystal pose as training proceeds -- the objective is being asked a harder
    question each time. This is the Growing-NCA dynamic the pool exists for and
    not by itself evidence of anything.
  * **But held-out `rmsd_dmg` also rose, 2.92 -> 3.38**, and that one is scored
    on a fixed set of fifteen crops from the same start every time. It cannot
    be explained by the pool. **This is the open question, and it is the next
    one.**

### WITHDRAWN, same day: that rise does not replicate and was never a trend

The full-length run of the **same section at the same seed** gives, over the
same interval, `rmsd_dmg` **3.047 -> 2.641 -> 2.775** -- noise of the same
magnitude in the other direction. Four evaluations of a series whose spread is
about +/- 0.3 A cannot carry a direction, and reading one as if it could was
the error. **Nothing was wrong with the measurement; the inference was wrong.**

**And the seed does not fix the realisation in this trainer.** The 1,000-update
arm was predicted to reproduce `--section port` line for line and does not
(2.921 / 2.952 / 3.377 against 3.047 / 2.641 / 2.775), so changing `updates`
alone changes the stream. **Any two-arm comparison here must be checked for
that before a difference is attributed to the thing under test** -- which also
weakens the five-on/four-off contrast above to the extent that it rests on
small differences, though not the parts that rest on `diverged`, `n` and the
non-finite count, where the gap is 0 against 145 and categorical.

### SUPERSEDED 25 Nov: "the rule plateaus" was the same error a third time

The run finished all 20,000 updates and the block mean falls in **every**
block -- 2.702, 2.461, 2.422, 2.119, 2.079, 2.075, 2.009 A over 2-3,000-update
blocks -- with the spread narrowing 0.244 -> 0.118. So the "2.46-3.05 band with
no trend" below was a window of a falling series, read the same wrong way as
the withdrawn rise above it. **Three readings of this series were wrong before
one was right, all three by taking a handful of evaluations for a direction.
No row of it may be read outside a block mean.** The open question is
therefore not "why does it stop improving" but **why the remaining 0.7 A takes
18,000 updates when the first 6 A takes 2,000.**

### What 27 evaluations to update 6,750 do show

    update   250    500   1000   2000   3000   4000   5000   6000   6750
    rmsd_dmg 3.047  2.641 2.775  2.500  2.573  2.734  2.571  2.457  2.887
    n          15     15    15     15     15     15     15     15     15
    diverged    0      0     0      0      0      0      0      0      0

**The stability holds completely**: `n` 15 and `diverged` 0 at all 27, `start`
pinned at 8.774, so no row is a survivor mean. **And the rule plateaus** --
~2.6-3.0 A inside 250 updates, then a 2.46-3.05 band with no trend for 6,500
more.

**So the open question is "why does it stop improving after 250 updates",** not
why it degrades. A rule that gets most of the way in 1 % of its budget and then
sits there is a different failure from any this project has recorded.

**2.457 A at update 6,000 is below the 4.0 A absolute bar**, against its own
8.774 A start, fifteen crops, nothing diverged -- and **not comparable to any
arm**, because the pool is the laptop's, the crops are easier and the contact
graph is given.

**The 250-update row is also the honest version of the port's headline.**
8.774 -> 2.92 A on fifteen crops with nothing diverged and nothing skipped
replaces the old 8.77 -> 2.63 A, which was measured before the bound and
carried four overflowing crops by update 500.

## MEASURED to completion, 25 Nov: 2.17 A fit_intact, and the bar is passed

    update  total  pair_anchor  pair_window  rmsd_dmg  fit_intact   n  diverged
     20000  3.512        1.731        1.781     2.040       2.167  15         0

**80 evaluations, `n` 15 and `diverged` 0 at every one, `start` pinned at
8.774, and zero non-finite training rollouts in the whole run.** The bound
holds at full length. Against the pre-registered buckets in
[[the-port-must-reproduce-the-published-result-on-point-clouds-before-it-is-run-on-proteins]]
this is the `<= 4.0 A` bucket, passed for the first time in the project -- and
still not comparable to any arm, because the pool is the laptop's, the crops
are easier and the contact graph is given. Bunya job 33419979 is the
comparable re-run.

**And `pair_window` fell 3.28 -> 1.95 A^2**, the one quantity the record had as
having no progress (22.2 -> 23.38 over 10,000 updates in `g25`), because it was
the deliberately unweighted half of the restraint. Weighting the full pair set
put a gradient on it. The A^2 values are not comparable with `g25`'s --
different pool, different denominator.

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
