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

**Leg 1 first, and it is cheap**: no protein data, 5K parameters, 256 nodes,
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
