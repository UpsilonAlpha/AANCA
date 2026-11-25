---
type: question
status: open
---

# The port has two legs and the project has only ever run the second: reproduce Gala on point clouds before changing the substrate

**Pre-registration for the single faithful port, written before any of it is
built.** The task is
[[the-task-is-to-realise-coordinates-for-a-given-contact-graph]] and the
instruction is fewer arms, so the thing registered here is a *replication*,
with the step the project has never taken put first.

## The two legs, and why conflating them is how you get twenty-eight arms

A "straightforward mapping of E(n)-GNCA and Growing NCA to proteins" is two
claims, not one:

    leg 1   our implementation reproduces the published result
            on the PUBLISHED substrate -- point clouds
    leg 2   that implementation transfers to proteins

**Every arm in this project has tested leg 2 while assuming leg 1.** Nothing in
the record shows this codebase reproducing
[[an-equivariant-graph-nca-morphs-random-coordinates-into-a-target-point-cloud]]
on anything. So a protein failure has never been able to distinguish "proteins
are harder" from "our rollout, loss or graph does not do what theirs does", and
with several settings changing at once in the port it would distinguish them
even less.

## AMENDED SAME DAY: leg 2 first, and leg 1 is the registered response to its failure

**Ulban, 23 Nov: "I don't see any reason we don't just go straight to
proteins."** Correct, and the ordering above was wrong. Leg 1 only ever pays
off in the failure branch: if the port works on proteins, reproducing a torus
proves nothing anyone needed. Running it first spends the one thing the
instruction "fewer arms" is protecting, which is attention, on a branch that
may not be taken. **So leg 2 runs first, and leg 1 is pre-registered as the
response to a leg-2 failure** -- designated now precisely so that a failure
produces the torus and not arm thirty. That is this note's real function and
it survives the reordering intact.

The title is therefore historical; the body's argument for *why* leg 1 is the
right failure response stands unchanged below.

**One claim of mine does hold up better than I expected, and it matters for the
failure branch.** `AANCARule.forward` takes generic graph tensors
(`node_features, hidden, edge_features, neighbours, edge_mask, envelope`), so
the rule is substrate-agnostic at its boundary; Gala's node features are all
ones; and the all-pairs loss has to be written for the port anyway. A
point-cloud leg 1 would therefore reuse the rollout loop, the graph builder,
the step cap, the truncation and the move head -- the machinery actually under
suspicion -- rather than being a parallel implementation that tests different
code. So it remains a *good* failure response, not merely the only one.

## Instrument leg 2 so a failure is diagnosable WITHOUT the torus

Since leg 1 no longer runs first, the port must carry the readings that would
otherwise have come from it:

  * **`align`, and the per-step bearing, not just its mean.** No native-edge
    arm has ever had `align` read. A mean hides "right then lost", which is
    the pattern a given edge set makes plausible for the first time.
  * **Stability past the training horizon.** The published rule holds its
    attractor 1000 steps past a 15-25 step horizon. We train on ~20 and infer
    and infer at 120, so the horizon gap is ours too. This is already measured
    here twice over and the port inherits both results:
    [[the-rule-has-no-attractor-past-its-training-horizon]], and
    [[the-pool-and-length-jitter-stop-the-divergence-past-the-training-horizon]]
    -- which is the replay pool doing in our hands what it does in Gala's, and
    is why the pool is a line of the port rather than an optional extra. Read
    the window at 20, 120 and 600 steps in the same run.
  * **Whether the weighted pair set actually determines the target --
    DISCHARGED 23 Nov, analytically, and it needs no run.** The condition is
    that the grounded graph be generically globally rigid; the efficiently
    solvable protein instance is Dong & Wu's, where **four already-placed,
    non-coplanar anchors fix an atom exactly**
    ([[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]).
    With the **full** pair set every one of the 12 window residues is
    constrained against all ~116 intact residues of a 128-residue crop, which
    is four non-coplanar anchors by a wide margin: the only way it could fail
    is anchors spanning fewer than three dimensions, and a 116-residue
    backbone fragment is not coplanar -- even an ideal helix puts its CAs on a
    cylinder, which is rank 3. The residual ambiguity is the global
    reflection, fixed by chirality and by the context being held at its
    crystal pose. **So the full pair set satisfies the precondition by
    construction, for every crop, and the old window-to-anchor-only set did
    too.** Rigidity was therefore never what was broken -- which is worth
    knowing, because it removes a candidate rather than confirming one, and it
    means the full pair set earns its place through `pair_window` (the one
    quantity that never improves,
    [[the-clamp-the-project-tuned-governs-two-percent-of-the-objective]])
    rather than through uniqueness of the target.

With those three in place, a leg-2 failure is already evidence about proteins
rather than about our plumbing, which was the whole reason leg 1 was going to
go first.

## Why the cap and the rollout length must move together

Recorded because they look like two independent lines of the port and are one.

The step cap is `max_translation = 0.3 A`, so a 12-step gradient window can
carry a residue at most `12 x 0.3 = 3.6 A`
([[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]),
while the damaged window starts a median **10.95 A** from where it belongs.
**The cap is what makes 120 steps necessary**: at 0.3 A a residue needs ~36
steps merely to cover the distance, so the rollout was lengthened to let a
capped rule arrive at all, and the gradient window then covered 10 % of it.

Remove the cap and the arithmetic inverts. An unbounded move head crossing
11 A in 20 steps needs ~0.55 A per step, which is inside what an MLP emits
without saturating -- and `g28` measured the head *asking* for only 0.04 A per
step with the tanh at 0.98 of its gradient near the origin, i.e. nowhere near
its ceiling
([[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]]). So a short
rollout is only coherent together with removing the cap, and removing the cap
is only safe together with a short rollout fully differentiated, which is
exactly the published combination: 15-25 steps, no cap, stability from weight
decay and gradient clipping rather than from a hard clamp.

**This is also why `g26_fullbptt` settles nothing about difference 3.** It
raised `truncate` to 120 over a 120-step rollout **with the cap still on** --
the longest, most expensive form of the wrong half -- and read 1.68 A worse.
Neither the short rollout nor the uncapped step has ever been run, together or
apart.

## Why leg 1 is the right failure response, and it is cheap: no protein data, 5K parameters, 256 nodes,
CPU. Their reported targets include a 2D grid and a 3D torus, with all-pairs
distance MSE, 15-25 step rollouts, a replay pool, no step cap, and random
Gaussian initial coordinates with **no anchors**. If this codebase cannot morph
a Gaussian cloud into a torus, no protein number it produces means anything,
and that is worth knowing before a GPU slot is spent rather than after.

**Leg 1 has a pass/fail that does not need inventing**: the authors say they
succeed on all 8 targets and hold the attractor for 1000 steps past a 15-25
step training horizon, including after damage. Reproducing *stability past the
horizon* is the sharp part, because it is the Growing-NCA persistence property
and it is what the replay pool exists for.

## The port, fixed now so it is not adjusted after a number appears

    native edge set            the task statement
    move_weight = 0            NO teacher -- the combination never yet run
    all-pairs distance loss    unclamped, FULL pair set (Ulban, 23 Nov)
    ~15-25 steps, FULL BPTT    shorter and fully differentiated
    replay pool                the persistence trick, as published
    no 0.3 A step cap
    small network              5K succeeded; ours is 229,588

Each line is a *departure of ours* being removed, not an idea being added. The
full pair set is the decided form of difference 1; the cap and the truncation
are difference 3, which no arm has ever touched together
([[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
tabulates all three).

## TWO LINES OF THE SPEC ARE WRONG, found by building it

**1. "small network -- 5K succeeded; ours is 229,588" cannot be done by
setting a width.** `AANCARule.__init__` exposes exactly two size knobs,
`hidden_dim` (96) and `message_dim` (128), and they bottom out at **17,552
parameters** -- 3.5x the published figure:

    hidden  message   params
        96      128   229,588   <- default, the number the record quotes
        32       64    86,868
         8       16    30,180
         1        2    17,552   <- the floor these two keys can reach

The residue is structural: four heads at `Linear(node_dim + hidden_dim, 64)`
with the 64 **hardcoded**, over a growth rule's `node_dim = 56`. Reaching ~5,000
needs the head width exposed or the featuriser narrowed, neither of which the
constructor has. **So this is the one line where "each line is a departure of
ours being removed, not an idea being added" fails** -- removing this departure
requires adding something to the model. It is deliberately not done, and it is
Ulban's call whether a 5 kparam port is worth a model change or whether 17,552
is close enough to call it small.

**2. Leg 1 is not as cheap as this note priced it, and I was wrong twice.** I
first said a point-cloud leg 1 was cheap, then conceded the cost and rested on
`AANCARule.forward` taking generic graph tensors. `forward` does -- but
`__init__` does **not**: `growth=True` silently widens `node_dim` from 32 to 56
by appending the closure block, and `build_rule` always passes `growth=True`. A
256-node point cloud has no closure structure, so leg 1 either carries 24 dead
channels or runs a **differently shaped rule from the one leg 2 runs** -- which
is exactly the "parallel implementation that tests different code" this note
says leg 1 avoids. Not fatal, and it is a line of work this note still prices
at zero.

**3. There is no port section in the config.** Everything the table lists is
now a key -- `native_edges`, `pairwise_window_weight`, `hidden_dim`,
`message_dim`, and the rest were already settings -- but assembling them into a
section is a step nobody has taken, and it waits on (1).

## Buckets for leg 2, on `rmsd_dmg` fitted on the intact context

The comparators already in the record, all `fit_intact`: doing nothing 16.28 A,
the geometric floor 10.05 A, the best arm `g25_equivariant` 8.10 A, and the
absolute bar **4.0 A**.

  * **<= 4.0 A -- the mapping works.** The bar is passed for the first time and
    the project's negative results were about its departures from the published
    method, not about neural cellular automata on proteins.
  * **4.0 to 8.10 A -- the port helps and the bar still fails.** Then ablate
    ONE line at a time from this baseline, which is the first time that
    discipline would be applied to something that works.
  * **8.10 to 10.05 A -- no better than the best arm, at or below plain
    geometry.** The combination does not rescue it.
  * **> 10.05 A -- worse than bond closure and sterics alone**, which is where
    `g26_fullbptt` landed.

## ANSWERED 25 Nov, on the laptop pool: the first bucket, and leg 1 does not run

**`rmsd_dmg` fit on the intact context reads 2.17 A** at update 20,000 of the
port, 15 crops, nothing diverged, from a start of 8.774 A
([[the-divergence-is-a-missing-published-tanh-not-a-property-of-the-rule]]).
That is the **`<= 4.0 A` -- the mapping works** bucket, passed for the first
time in this project.

**So leg 1 does not run.** This note pre-registered the point-cloud
replication as the response to a leg-2 *failure*; leg 2 did not fail, so the
torus is not owed and building it would be the arm-thirty this note exists to
prevent. That is the pre-registration doing its job in the branch nobody
expected.

**What the pass does NOT yet license.** The bucket's own text says the
project's negative results "were about its departures from the published
method, not about neural cellular automata on proteins" -- and that sentence is
a comparison, which this run cannot make. The laptop pool is 954 structures
against Bunya's 23,760, the start distance is 8.77 A against 12 to 16 A, and
the contact graph is given. ~~Bunya job 33419979 is the run that decides which
bucket this really is, and until it reports, the bucket is "passed on an
easier pool".~~ **It reported. See below.**

## THE BUCKET, DECIDED 25 Nov ON BUNYA: 2.53 A, the first one

**Job 33419979 ran the port to completion on the pool every comparator in this
project was trained on**: `COMPLETED`, exit `0:0`, 39:52 elapsed against a 24 h
budget, 80 evaluations, last `update` 20,000.

    fit_intact at 20,000      2.528 A      best single evaluation 2.477 A
    n                         15 at all 80 evaluations
    diverged                  0 at all 80
    start_rmsd_dmg            10.679728087770185 at all 80 -- identical to the
                              last digit, so NO row is a survivor mean
    nonfinite_loss skips      0 of 20,000

**Against this note's own buckets: doing nothing 16.28, the geometric floor
10.05, the best arm `g25_equivariant` 8.10, the bar 4.0 -- the port reads
2.528.** The `<= 4.0 A -- the mapping works` bucket, by a factor of 3.2 on the
best arm and 4.0 on plain geometry, with 5,328 parameters against 229,588.

**And the pool objection that voided every previous cross-arm comparison does
not apply to this run.** [[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]
voids a laptop-to-Bunya comparison because `available[:416]` is a different 416
proteins. This is Bunya's 23,760-structure pool, the same slice, the same crop
index meaning the same protein. That was the entire purpose of the job.

**So the first bucket's conclusion is now licensed, in the words registered for
it before the run**: the project's negative results "were about its departures
from the published method, not about neural cellular automata on proteins."

**Three things that still differ from `g25`, stated so the claim is not
overstated.** (1) The start is 10.68 A where `g25`'s was 13.13 A, a closer
start by 2.45 A -- but as improvement-against-own-start, the licensed form, the
port moves 8.15 A against `g25`'s 5.03 A, and 2.45 A of easier start does not
explain finishing 5.6 A better and 7.5 A inside the geometric floor. (2) The
contact graph is given, where `g25` had the live k-NN graph; under
[[the-task-is-to-realise-coordinates-for-a-given-contact-graph]] that is the
task statement, but it is a different task and the port is not a better rule at
`g25`'s. (3) 474 of 20,000 updates contributed no gradient -- `no_crop` 452,
`no_window` 22, `nonfinite_loss` **0** -- so it trained on 19,526 effective
updates and not one skip was the rule exploding.

**Leg 1 does not run, and this is the branch that decides it.** This note
registered the point-cloud replication as the response to a leg-2 *failure*.
Leg 2 passed on the comparator pool, so the torus is not owed and building it
would be the arm thirty this note exists to prevent.

**What is NOT measured, and is now the project's largest gap.** Persistence
past the training horizon -- the published rule's headline claim, which their
code evaluates to 10,000-100,000 steps against a 15-25 step training horizon.
This note listed it as the second of three instruments the port had to carry
and it was never built. With a working baseline it is finally a measurement on
something rather than on nothing.

**No arm, this one included, has been run at a second seed.**

`align` is read alongside, because it is the instrument the capacity question
turned on and **no native-edge arm has ever had it read**. Twist is a guard, not
a result.

## What a failure here can and cannot say -- the asymmetry, registered

**A pass licenses the ablation; a failure localises nothing.** Several settings
move at once by design, so a leg-2 failure cannot name which departure mattered.
That is the accepted price of a replication target, and it is acceptable *only
because leg 1 runs first*: with leg 1 passing, a leg-2 failure is evidence
about proteins, which is the interesting thing. Without leg 1, it is evidence
about nothing.

**It does not reopen the capacity question.**
[[the-architecture-can-place-residues-under-some-supervision]] is answered NO
for invariant perception over an error-derived neighbourhood, and the NO was
explicitly scoped to that edge set
([[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]]). A port that
supplies the edge set is a different architecture on a different task
statement, not a re-run of the arm that closed it, and it must not be presented
as reopening it.

**It is one arm.** If it fails, the answer is not a second port.

## Supports
- [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]
