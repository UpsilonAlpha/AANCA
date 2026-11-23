---
type: observation
status: supported
evidence: ours
sources: [33283263, 33319941]
---

# FAPE is 2 % of the objective and flat, the unclamped pairwise term is 38-44 % of it, and placement improved five angstrom with no bearing

From `runs/g25_equivariant/history.json`, 40 held-out evaluations over **10,000
updates** -- the arm's own recorded curve, not a re-measurement. Weights
resolved through `config.resolve` rather than read off a banner:
`fape 0.1`, `pairwise 0.01`, `bond 0.122`, `clash 1.0`, `reach 0.086`,
`orientation 2.5`, `completion 5.23`, `move 0`, `truncate 12`, `steps 120`,
`freeze_context true`.

## Weighted contribution to the objective, first evaluation and last

    term                     u=250            u=10000        of total
    pairwise (anchor)        0.695            0.241           44 % -> 38 %
    orientation+completion   0.298            0.238           19 % -> 37 %
    bond                     0.518            0.125           32 % -> 20 %
    clash                    0.068            0.020            4 % ->  3 %
    FAPE                     0.0153           0.0136           1 % ->  2 %
    reach                    small            small
    TOTAL                    1.5938           0.6375

(`orientation+completion` is the residual after the terms the history records
individually; `completion` reads 0 throughout, so it is the orientation term.)

**The clamp this project has spent arms reasoning about governs one to two
percent of its objective, and that share does not move.** `fape` goes 0.0153 to
0.0136 across ten thousand updates -- flat, and two orders of magnitude below
the largest term. `the-fape-clamp-is-what-gates-the-geometry` is about where
the *gradient* sits rather than where the loss sits, and a flat small term can
still carry gradient, so this does not refute it. But any account of these arms
that treats FAPE as the placement signal is describing 2 % of the objective.

**The all-pairs loss is not missing, and difference 1 was mis-stated.**
[[placing-a-node-from-local-distances-needs-a-globally-rigid-neighbourhood]]
tabulates "ours: FAPE clamped at 10 A" against Gala's all-pairs MSE. In fact
the unclamped pairwise term is the **single largest contributor** at both ends
of training, at a weight of 0.01, because its raw magnitude is in A^2 and runs
24 to 70. What is missing is not the loss but its *pair set*: ours covers
window-to-anchor pairs only.

## The quantity that never improves is the one nobody weighted

    raw, A^2         u=250     u=10000    range over the run
    pair_anchor       69.5       24.12     monotone-ish fall, factor 2.9
    pair_window       22.2       23.38     21.4 to 29.3, no trend

**`pair_window` -- the window's own internal shape -- does not improve at all
over ten thousand updates.** It is the half deliberately left unweighted
([[the-objective-should-restrain-the-output-not-teach-the-placement]] and the
contracted-intermediate measurement behind it), and it is the only quantity
here with no progress. Under
[[the-task-is-to-realise-coordinates-for-a-given-contact-graph]] the decision
is to weight the full pair set, which makes this the first arm that would put a
gradient on the one thing that has never moved.

## Placement improved by five angstrom and the rule has no bearing

    rmsd_dmg (fit_intact)   13.13 at u=250  ->  8.10 at u=10000
    twist                   10.4            ->  8.8   (inside the band)
    align, measured today   0.141, inside the -0.10 to +0.25 null band

So the best arm **did** learn placement -- 13.13 to 8.10 A, against 16.28 A for
doing nothing -- while emitting per-step moves whose cosine with the ideal move
is indistinguishable from zero
([[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]] records
the `align` reading).

**That is not a contradiction, and resolving it matters more than either
number.** A rule can reduce a distance objective without any step pointing at
the destination: Gala's update is literally
`x_i += mean_j (x_i - x_j) * phi_x(m_ij)` -- a **distance-driven relaxation**,
where each node is pushed along existing neighbour-difference vectors by
learned scalars. Nothing in it aims. `align` asks whether the emitted move
points at where the residue belongs, which is a Growing-NCA question: on a grid
a cell knows its address and a bearing is the thing to learn.

**So `align` may be the wrong instrument for the task the project has now
chosen.** It is a correct measurement of aiming, and
[[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]] stands as
measured and scoped -- the architecture cannot be *taught to aim*. What this
observation questions is whether aiming is the capacity a realise-coordinates
rule needs, and the 5 A improvement with `align` at 0.141 is direct evidence
that it is not the only route.

## What this does NOT establish

  * **`pair_anchor` falling is close to a restatement of placement improving**,
    not independent evidence that the term *caused* it. Window-to-anchor
    distance error in A^2 and window displacement in A are near-monotone
    functions of each other, so "the big term fell" and "placement improved"
    are largely one observation. Nothing here separates cause from measurement.
  * **Loss share is not gradient share.** A term can be small in value and
    large in gradient, which is the whole content of the clamp argument.
    Establishing that FAPE's gradient is as negligible as its value needs the
    per-term gradient norms, which nothing has measured.
  * **One seed, one arm** ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).
  * **No floor margin is quoted.** `g25` crosses below 10.05 A between updates
    4250 and 5250, but the floor was measured on harder windows than the arms
    it is compared against, so the margin is not quotable
    ([[the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar]]).
  * It does not reach the 4.0 A bar. 8.10 A fails it by a factor of two.

## Supports
- [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]
- [[the-objective-should-restrain-the-output-not-teach-the-placement]]
