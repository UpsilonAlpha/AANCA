---
type: claim
status: open
evidence: ours
sources: [32449992, model.py]
checked: [32423296]
---

# The move head asks for five times the cap, so tanh kills its magnitude gradient

`step` reads **0.3000 on every one of twelve crops** on `g10_equiv_ceil`, which
is 100 % of the 0.3 A per-step cap. That looks like a clamp sitting on its
limit and is not: `_bounded` in `src/aanca/model.py` is a **tanh**, so it never
reaches the cap and saturating it means the head is asking for far more.

**Measured: 1.61 A per step, 5.4x the cap, with a magnitude gradient 9.58e-05
of its near-origin value** -- a ten-thousandfold attenuation (job 32449992).

The curve is invertible, so the raw request comes back from the printed number
without touching `model.py`:

    bounded = L * tanh(raw / L)        ->        raw = L * atanh(bounded / L)

    g10_equiv_ceil   step 0.3000  ->  raw >= 1.41 A  = >= 4.7x the cap
    g8_ceiling       step 0.0309  ->  raw    0.031 A =    0.1x the cap

and with it the derivative of the bounded norm with respect to the raw one.
**Differentiated in float32 rather than evaluated as `1 - tanh^2`, and the two
disagree**, which is the part that could only be found by running it:

    request    bounded norm    d|out|/d|in|
    0.031 A     0.0308901       9.9e-01     g8_ceiling -- healthy
    1.41  A     0.2999504       3.3e-04     g10's conservative lower bound
    2.0   A     0.2999991       6.4e-06     recovery already 0.2% off
    3.0   A     0.3000000      -7.5e-09     SIGN IS WRONG
    5.0   A     0.3000000       0.0         EXACTLY zero

**So the magnitude half of the move head is attenuated by at least three
thousandfold, in this arm and not in the architecture** -- `g8_ceiling` ran the
same head on the same task with the same cap and sits in the linear region --
**and past about ten times the cap it is not attenuated but dead.** In exact
arithmetic tanh is asymptotic and `1 - tanh^2` at ten times the cap is about
8e-9; float32 rounds the whole scaling factor to 1.0 and the gradient collapses
to exactly zero two orders of magnitude earlier, after passing through an
interval where its *sign* is wrong.

This was assumed otherwise and the test found it. The first version of
`tests/test_bounded_saturation.py` asserted that the bounded norm is always
below the cap, because tanh is asymptotic; it failed, because in float32
`_bounded` reaches and slightly exceeds the cap. The precision limit is
therefore measured rather than reasoned about, which also means **g10's
request has a lower bound and no upper one**: 0.3000 is consistent with 4.7x
the cap and equally with a hundred times it.

**The mean cannot have been dragged up by outliers**, which is why one number
carries this. `step` is a mean of norms and no single norm can exceed the cap,
so a mean of 0.29995 requires nearly every residue on nearly every step to be
saturated. An asymptotic bound makes the mean a worst case rather than a
typical one.

## This is a documented failure arriving at the other end of the same curve

`_bounded`'s own docstring records the failure at the **origin** -- a
zero-initialised head gives `limit * tanh(0) / 0 = 0 / eps = 0`, which
multiplies the output by zero and leaves the head with no gradient "for every
update, forever" -- and calls the branch guarding it load-bearing, with the
measured `0.000e+00` to prove it. The far end has the same shape and was
assumed benign: tanh was chosen so that "the gradient does not vanish for a
head that is asking for too much". Whether it vanishes anyway is an empirical
question about *how much* too much, and nobody had asked it.

Both ends of `_bounded` therefore destroy the gradient, one by construction at
zero and one by saturation at scale, and only the first was known.

## Why it explains the record rather than merely accompanying it

Over 120 steps at the cap a damaged window travels **36 A** while
[[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]
measures the need at 10.10 A mean. **Overshoot is therefore forced rather than
chosen**, and the rule cannot be taught to ask for less, because the gradient
that would teach it is the one tanh has flattened.

That connects to the bearing directly.
[[the-teacher-forced-rule-emits-moves-pointing-away-from-the-target]] already
established the mechanism on a random head -- a near-constant vector overshoots
over 120 steps, and once a residue is past its target "toward the target"
reverses and the late-step cosines go negative, which is why the null is -0.36
rather than zero. A saturated magnitude head is a near-constant vector *by
force*. So `align` near zero on an arm that saturates the cap is not only a
bearing failure; part of it is the signature of a walk that cannot slow down.

`net/path` 1.000 on every crop is the third leg of the same picture: each
damaged residue walks a **straight line at full speed** in a direction
uncorrelated with where it has to go.

## Why this is `open` when half of it is arithmetic

Check 1 flagged it as `supported` resting on nothing but our own calculation,
and **the flag is accepted rather than exempted**, although a case for an
exemption exists: the hierarchy that puts `experimental` above `ours` is about
claims concerning *proteins*, and the invertibility of tanh and the value of
`1 - tanh^2` are mathematics. Writing an exemption for my own node on the day I
wrote it is how a check stops being load-bearing, and this project has already
promoted four measurements that turned out to be defaults.

So the status reflects the weaker half. The claim is really two:

  * **the head asks for 5.4x the cap and its magnitude gradient is 9.58e-05 of
    its near-origin value** -- measured, and before it was measured the same
    thing was bounded at >= 4.7x and <= 3.3e-4 by arithmetic on four printed
    decimals; not in doubt either way;
  * **this is why the record looks as it does** -- interpretive, and what
    `open` refers to.

`tests/test_bounded_saturation.py` pins the first half, so it stays true if
`_bounded` changes. Promotion needs the second: job 32449794 for the request at
full precision, and `--truncate 60` (32449661) for whether the saturation is a
cause or a consequence.

## What is not claimed

The bearing half is **not** attenuated this way -- direction passes through
`_bounded` scaled, not flattened -- so this does not explain `align` on its
own, and it is not a competing explanation for
[[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]],
which concerns what a direction could mean rather than how long a step is.

**The request is now measured rather than bounded.** Job 32449992 inverted it
on full-precision norms and it resolved, below the float32 ceiling:

    the head is ASKING for 1.61 A per step = 5.4x the 0.3 A cap
    tanh gradient on the magnitude: 9.58e-05 of its near-origin value

So the attenuation is **ten thousandfold, not three**, and the heading
understates it by a factor of three. The `>= 4.7x` lower bound read off four
printed decimals was correct and conservative. Not renamed, because a node that
is cited under one name and silently answers to another is worse than a heading
that undersells its own number; the figure in the body is the one to quote.

Nothing here says the gradient being dead is *why* `rmsd_dmg` is 12.5 A. It
says the rule cannot be taught to ask for a shorter step, which is a different
and smaller statement.

Whether the saturation is a cause or a consequence is also open. A rule whose
bearing is uninformative has no reason to prefer a small step, so asking for
the maximum may be what an unaimable move looks like once trained, rather than
what prevents aiming. `--truncate 60` (job 32449661) bears on it: a wider
gradient window is the one change that lets a residue be differentiated through
its own arrival, so if the request falls there, saturation was downstream.

## Supports
- [[question-does-a-trained-rule-use-its-movement-budget]]
- [[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]
