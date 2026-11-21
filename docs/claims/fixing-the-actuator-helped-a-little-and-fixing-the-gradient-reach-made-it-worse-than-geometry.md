---
type: observation
status: supported
evidence: ours
sources: [33283263, 33283266, 33283602, 33277377, 33249861]
---

# Both known defects were fixed: the unsteerable rotation cost about 0.4 angstrom and full backpropagation made the rule WORSE than bond closure and sterics, so the arm chain ends here

`g25_equivariant` (33283263, exit 0, 378.6 min) and `g26_fullbptt` (33283266,
exit 0, 502.5 min), each one variable against `g24_frozen`. All numbers below are
on **the same 12 held-out crops and the same damage windows** -- confirmed, not
assumed: `train_growth.score` seeds its window generator `manual_seed(seed +
index)` with `seed = 0`, and `geometric_floor.py` does the same, and the floor
read its crop indices straight out of each checkpoint's `eval_indices`.

    settled, matched crops            fit_all  fit_intact   twist   margin vs floor
    untrained start                     14.76       16.28      --        --
    GEOMETRIC FLOOR (no network)         9.21       10.05      --        --
    g22_random   context free            8.41        9.02   12.8d      1.03
    g24_frozen   context pinned          7.98        8.49   10.3d      1.56
    g25_equivariant  + rotation fix      7.50        8.10    9.2d      1.95
    g26_fullbptt     + truncate 120      9.46       10.17   12.0d     -0.12
    bucket 0 PASS                          --        4.00      --        --

## Defect 2, the unsteerable rotation: a real but small gain

`lab_rotation = false` moves the rotation half of the emitted move into each
residue's own frame, removing the 45 % of it that an invariant encoder could not
represent. Placement improved 8.49 -> **8.10 A** and **twist fell to 9.2 deg, the
lowest in the record**. The two move together as the mechanism predicts: the
defect was in the *rotational* actuator, and orientation is what improved most.

**Per this arm's own pre-registration that 0.39 A falls in the 6.5-8.49 bucket:
"a real effect inside the noise of an unreplicated seed. Report direction only,
and SAY SO rather than quoting the margin."** So: the direction is right, twist
corroborates it, and the magnitude is not a result. Nothing here has an error bar
([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).

## Defect 1, the gradient reach: the fix made it worse, and refutes its own prediction

`truncate = 12 -> 120` gives BPTT the whole 120-step rollout instead of its last
tenth. Full backpropagation **fits** on the MI355X, at 3.01 s/update against
2.27. It made placement **1.68 A worse** -- 8.49 -> 10.17 A -- which puts it
**above the geometric floor of 10.05**, meaning this arm has **no signal at all**:
a network-free optimiser that only closes the chain and avoids clashes does as
well or better.

**This refutes the prediction it was built to test.**
[[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]] measured
the truncation window as 27 % of the journey and named it *the binding
constraint*, with the per-step cap having 3.6x headroom and binding on nothing.
It called itself *"pre-positioning rather than a result -- it says which rollout
hyperparameter to test first"*. It was tested first, and raising it hurt.

Mechanism unknown and offered as nothing more than a guess: 120 steps of a
recurrent rule is a long credit-assignment path, and gradients through it may be
badly conditioned in a way the truncated version avoids. **What is established is
only that `truncate = 120` is not the fix, not that gradient reach does not
matter** -- an intermediate value was not tried and this does not say one would
fail.

## The registration says to stop here, and it is being honoured

`g26`'s `>= 8.49` bucket reads: *"the defects were not what was holding it back.
This is the most informative outcome for the central claim and the one that
should end the arm chain rather than extend it."* And bucket 0's clause: *"if it
fails, we do not add terms and re-run."*

So **`g27_both` will not be run**, even though it is registered and wired. It is
the rotation fix plus the truncation change, and the truncation change is now
measured as harmful; running it would be chasing a number rather than testing a
claim.

**Where that leaves the central question.** The best configuration in the project
places a damaged 12-residue window at **8.10 A** where a correct answer is 1-2 A,
doing nothing is 16.28 A, and pure geometry with no learned anything is 10.05 A.
Three measured faults have now been fixed -- the translation half of the move
(7 Nov), the context drift (g24), the rotation half (g25) -- and the absolute bar
still fails by a factor of two. **A local per-residue rule with this objective
does not place a backbone, and the remaining gap is not explained by any known
defect.**

## Scope

  * **One seed per arm, no replication**, which is why the 0.39 A gain is
    reported as a direction. The 1.68 A loss is large enough to exceed any
    plausible seed noise in this record, where arm-to-arm differences have run
    0.4-1.1 A.
  * **`fape`, `rmsd` and `twist` are not comparable in absolute terms with
    g1-g22**, which ran the context free: 116 of 128 residues cannot move here,
    so a mean over residues is a different quantity. The comparisons above are
    against `g24_frozen`, which differs in exactly one setting each.
  * **The floor is one optimiser configuration** (Adam, lr 0.03, 1500 iters), so
    it is an upper bound on the true geometric floor. A better optimiser lowers
    it and shrinks every margin above.
  * **Bucket 2 was not computed.** Every arm from g24 on has twist in band and so
    could carry it -- the helical Ramachandran basin fraction, the direct test of
    learned secondary structure. It remains the one cheap measurement never made.

## No `refutes` edge to the gradient-window note, and why

One was written here first and removed, following the precedent in
[[the-rule-has-no-attractor-past-its-training-horizon]], which records doing the
same thing for the same reason: *"it was a miscategorisation rather than a
tension the validator should carry."*

[[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]] is
titled on a **measurement** -- the window must travel ~10-13 A and BPTT sees
3.6 A of it -- and that measurement is untouched by this arm. What this arm
refutes is an **inference stated inside** that note, that the truncation window
is therefore *the binding constraint*. Raising it hurt, so a constraint that
binds is not necessarily one whose removal helps.

A `refutes` edge would have made the validator demand that a true, experimental
measurement be demoted. The narrowing is recorded in that note's own body
instead, under its own heading, which is where a reader who cites it will see
it.

## Supports
- [[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]

## Requires
- [[freezing-the-context-improves-placement-twist-and-whole-structure-rmsd-together]]
