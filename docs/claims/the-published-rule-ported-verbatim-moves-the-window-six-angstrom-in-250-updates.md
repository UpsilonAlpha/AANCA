---
type: observation
status: open
evidence: ours
sources: [egnca.py, train_egnca.py]
---

# Gala's rule ported verbatim, 5,329 parameters, moves the window 8.77 to 2.63 angstrom in 250 updates -- and diverges by 500

**The first run of the port.** `src/aanca/egnca.py` implements the four
published lines
([[an-equivariant-graph-nca-morphs-random-coordinates-into-a-target-point-cloud]])
with no frames, no actuator between the network and the coordinates, and
**5,329 parameters** at Gala's stated widths of hidden 16 / message 32 --
6.6 % above the paper's rounded "5K", which is what those widths cost in the
standard block shape. Against `AANCARule`'s 229,588 and its 17,552 floor
([[the-port-must-reproduce-the-published-result-on-point-clouds-before-it-is-run-on-proteins]]).

**`status: open`, and the reason is in the third column.**

## The run, on the LAPTOP pool, CPU, 0.21 s/update

    update  total  anchor  window  rmsd_dmg  (fit_intact, start)   n  diverged
       250  6.075   2.960   3.115     2.501   (2.629,     8.774)   15         0
       500  4.240   1.769   2.471     2.194   (2.302,     7.203)   11         4
       750  3.769   1.676   2.093     2.140   (2.223,     5.402)    7         8

**The metric is the project's own, not a new one.** `fape.backbone_rmsd_on` is
*"CA RMSD read off `read`, superposed on `superpose`"*, and the port calls that
same function with the same two superpositions every arm uses. The carrier that
places a residue's atom cloud keeps its **native orientation**, which does not
flatter a CA-only metric because only the CA position is read.

## What is real here, and it is the first column against the third

**Within its own run: 8.77 A to 2.63 A `fit_intact` in 250 updates, on 15
crops, with zero divergences.** A 6.1 A improvement against its own start.
The comparator is `g25_equivariant`, the best arm in the record, which moved
13.13 to 8.10 A `fit_intact` -- **5.03 A, over 10,000 updates**
([[the-clamp-the-project-tuned-governs-two-percent-of-the-objective]]). So the
port's improvement is larger and roughly forty times faster, with 2.3 % of the
parameters.

**Improvement-against-own-start is the only form of that comparison which is
licensed**, and even it is weakened by the pool being easier.

## Why no absolute number here may be set beside any arm's

  * **Different pool, and this is a standing project rule.** This ran on the
    laptop's 954 structures; every arm ran on Bunya's 23,760, so
    `available[:416]` is a different 416 proteins and the same crop index is a
    different protein
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).
    The start distance shows it directly: **8.77 A here against 12 to 16 A on
    Bunya.** Easier crops. Nothing absolute crosses that gap.
  * **The contact graph is given.** `native_edges = true`, which under
    [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]] is the task
    statement rather than a leak -- but `g25` had the **live k-NN** graph, so it
    was doing the harder thing. The port is not a better rule at `g25`'s task;
    it is a rule at the task the project has since chosen.
  * **Survivorship from update 500 onward.** `start` falls 8.77 -> 7.20 -> 5.40
    and **the start state cannot depend on the weights**, so what moved is
    *which crops were averaged*: the hard ones went non-finite and left the
    mean. **Eight of fifteen are gone by 750.** The later two rows are
    therefore not an improving rule, and the `diverged` column exists so this
    cannot hide -- it was added before the table was read.

**So the 250-update row is the only one to quote, and only against its own
start.** Rows 500 and 750 are recorded to show the failure, not the progress.

## Next, in this order

  1. **Stabilise.** `lr = 1e-3`, `clip = 1.0`, `weight_decay = 1e-5` as
     configured; the published stabilisers are weight decay and gradient
     clipping and this rule has no step cap by design. Lower `lr` or tighten
     `clip` first. **This is making the published recipe run, not adding an
     arm** ([[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]
     on replicate-before-ablating).
  2. **Re-run on Bunya**, where every comparator lives.
  3. **Measure persistence past the training horizon**, which is the published
     rule's *headline* claim -- damage at t=24 then 1,000 further steps,
     *"can maintain the target state after any number of time steps t > 15"* --
     and which **nothing here tests.** The replay pool trains toward it; no
     readout measures it. Arguably the most important gap.

## Departures from the published method, recorded because they are real

  * **We have anchors and the published task has none.** Pattern formation
    starts every node from an isotropic Gaussian; regeneration pins the whole
    surviving context at its crystal pose and displaces twelve residues. A
    strictly easier start, and the published-result note's "no anchors at all"
    does not carry over.
  * **Not literally all-pairs.** `pairwise_distance_restraint` covers
    window x context and window x window and omits context x context, which a
    frozen context satisfies for free. The gradient content is the same; the
    A^2 values are a mean over a different denominator and are **not**
    comparable with a true all-pairs figure.
  * **Their Gaussian-noise half is not ported.** Half of each published batch
    is perturbed, partly globally and partly locally; every sample here carries
    the project's damage window instead. The replay pool is ported, this is not.
  * **`phi_x`'s output layer starts at gain 1e-3**, which the paper does not
    state. It is the reference E(n)-GNN initialisation and it is *necessary* at
    protein scale: a neighbour difference is ~10 A rather than a normalised
    cloud's unit scale, and at default `Linear` init the first step displaces a
    residue tens of Angstrom and a 20-step rollout overflows before training
    starts. An initialisation, not a cap -- a test shows the layer free to grow
    linearly over three decades (2.40 / 24.03 / 240.27 A).
  * **`h' = phi_h(h, m)` is implemented non-residually**, as the claim node
    records it. The reference E(n)-GNN has a residual option and Gala's code may
    use it; the paper as recorded does not say, and this cannot be settled from
    the note alone.
  * **No edge attributes.** `e_ij` is in the recorded equation but the
    pattern-formation task has none ("node features all 1"), so the port passes
    none and raises loudly if a rule built for them is called without them.

## Two inconsistencies this created, named so they are not found later

  * **`pairwise_window_weight` now has two different defaults in two
    trainers** -- 0.0 in `train_growth.py`, so every arm to g28 keeps the
    objective it ran under, and 1.0 in `train_egnca.py`, per the 23 Nov
    full-pair-set decision. Deliberate in both places and confusing in the
    pair.
  * **`FixedNeighbours`' docstring guarantee is vacuous for this rule.** It
    protects the distinction that only the *selection* is frozen while
    distances, directions and the envelope are recomputed. This rule has no
    edge features at all, so there is nothing to recompute beyond
    `||x_i - x_j||^2`, which it computes itself.

## Which project metrics are absent, and why that is deliberate

`twist` cannot exist for a frameless rule, being a bonded pair's relative
rotation. Also absent: `fape` (defined in per-residue frames), `bond` and the
peptide angles (N and C are never placed), `clash` and the chi metrics (no side
chains), `identity` and `recovery` (no identity head), `alive`, `completion`
and `reach` (no births). Every one of them would be handed the carrier's native
orientation and would read near-perfect. They are **absent from `EGNCATerms`,
not zeroed**, and a test asserts the absence so that adding one forces the
docstring's claim to be rewritten rather than quietly extended.

## Supports
- [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]
