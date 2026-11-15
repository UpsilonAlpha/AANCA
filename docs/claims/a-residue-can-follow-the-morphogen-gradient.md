---
type: observation
status: supported
evidence: ours
sources: [morphogen.py, model.py, rollout.py, graph.py, growth.py]
---

# A residue can read which way the concentration rises and move that way -- measured, and it needed the gradient handed to it

Ulban, 15 Nov: *"nodes should be able to 'see' the morphogen gradients (in this
case electrostatics) and follow them to self-organise into a protein - is that
the case?"* Mostly yes for position, no for orientation, and the "mostly" was
worth 0.25 of a cosine.

## The circuit, which already existed

Three pieces had to line up, and they did:

  * **`model.py:480`** builds each message as
    `message(cat([receiver, sender, edges]))`, and `state` includes the node
    block, so residue i reads **each neighbour's concentration**, not only its
    own.
  * **`graph.py` block 4, `local_direction`** is the unit vector to that
    neighbour **in residue i's own N-CA-C frame**. So a message carries a
    concentration and a bearing together, and the messages are summed -- the
    shape a discrete gradient has.
  * **`equivariant_move = true`** in `[base.inputs]` reads the frame head's
    translation in that same frame
    ([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]).
    Without it the output direction is a lab-frame vector an invariant encoder
    cannot represent, and no amount of gradient would help.

## Measured, because reading the code is not the same as it working

The untrained rule emits **exactly zero** -- the frame head is zero-initialised,
so the translation and its whole Jacobian are zero and no perturbation probe
measures anything. Two measurements instead.

**Expressivity.** Randomise the frame head's readout, then put a persistent
source on one neighbour of residue 40 and on another neighbour lying on the
opposite side (`cos = -1.000`):

    t, empty field  : +0.01604 +0.12167 +0.11259
    t, source at 38 : +0.01517 +0.03382 +0.15549
    t, source at 68 : +0.01324 +0.03444 +0.15467

So the field reaches the head -- it moved the translation by 0.079 A on average
across all residues -- but **which** neighbour carries it changed the move by
only **0.0022 A**. The amount dominated the bearing by a factor of 36.

**Learnability.** A synthetic chemotaxis task on 1UBQ: one residue secretes,
every other should emit a translation pointing at it, and the structure is
**identical in every example**, so the diffused field is the only input that
says where the source is. Cosine against the true local-frame direction, on
three sources never trained on, two seeds:

    update   concentration only   + the gradient block
       100         +0.652              +0.947
       200         +0.703              +0.947
       300         +0.715              +0.960
       600         +0.709              +0.960

    parameters     241,814              245,270

**Both halves of that table are the result.** 0.71 is far above the 0.00 a rule
that cannot read direction would sit at, so the message function really does
infer a gradient from `(c_j, u_ij)` pairs. And inferring it is much worse than
being told, for 3,456 parameters -- 5 % of a cosine per thousand parameters,
and the gap is already there at update 100.

## So the gradient is supplied, which is what Growing NCA does

A Growing NCA cell perceives `[identity, grad_x, grad_y]`, the gradients coming
from **fixed Sobel filters** rather than from anything the cell learns. That
choice is usually read as an efficiency; the table above says it is load-bearing.
`morphogen.field_gradient` is the graph analogue:

    sum_j w_ij (c_j - c_i) u_ij        w = envelope * edge_mask

in each residue's own frame, which costs one einsum because `u_ij` is the
`local_direction` block the edge features already carry. `FIELD_NODE_EXTRA`
4 -> 10. Invariance checked on real geometry rather than argued: rotate 1UBQ
globally and the block changes by **3.6e-08**, and the neighbour list is
unchanged.

Two choices stated because neither is forced. The sum is **unnormalised**, as
Sobel is -- dividing by the weight sum would make it a mean difference per
neighbour instead of a sum over the neighbourhood, and which is better is
untested. And the **envelope** weights it, so a neighbour at the cutoff fades
out rather than dropping, which keeps the block continuous as the graph
rewires.

## This is the right functional form for screened electrostatics, which is Ulban's framing

Worth writing down because it was not designed for it. The steady state of
`dc/dt = D grad^2 c - lambda c + s` satisfies `(D grad^2 - lambda) c = -s`,
whose Green's function is `exp(-r / xi) / (4 pi D r)` with `xi = sqrt(D /
lambda)`. That is the **Debye-Huckel screened Coulomb potential**, the same
functional form electrostatics takes in salt water. So:

  * The diffusing morphogen **is** a screened potential, and `DEGRADATION` is
    the screening length ([[the-emitted-chemical-diffuses-over-the-perception-graph]]).
    The measured per-hop decay on 1UBQ is 0.41 then 0.57 over hops averaging
    about 6 A, which is a screening length near **9 A** against the ~8 A Debye
    length of water at physiological salt. Coincidence, not derivation -- the
    constant was set from the graph's diameter -- but the order is right.
  * **`field_gradient` is the force, not the potential.** `grad phi` is what a
    charge responds to, and it is exactly what the block supplies, in the frame
    the translation is read in. "Follow the gradient" is one linear map.
  * **The rule can learn signed charge**, because the node block carries the
    residue one-hot and a pair of channels gives a signed potential: one
    positive, one negative, their difference the sign. Nothing is tabulated,
    which is the condition from
    [[the-clash-term-does-not-see-the-ramachandran-plot]].
  * **`EMIT_CHANNELS` is now 4, not 2** (15 Nov, Ulban's call). Two channels
    *was* the signed pair and therefore said the field is electrostatics. Four
    says the field is whatever the rule finds useful to secrete -- the signed
    pair still fits inside it, and so does a hydrophobic signal, which has no
    sign and so cannot share the pair. This is the Growing NCA position, where
    a cell's chemicals carry no assigned meaning at all. `FIELD_NODE_EXTRA`
    10 -> 20, parameters 245,270 -> 251,160, node_dim 76. Nothing asserts the
    rule will use all four; an unused channel decays to nothing, which is a
    cheaper failure than a channel the rule needed and did not have.

A graph is not R^3 and the discrete field has no `1/r`, so this is an analogy
with the right form rather than an implementation of Coulomb's law.

## Which makes this a candidate answer to the project's open bearing problem

The record already contains the other half of this experiment, and it is the
reason the result matters. `g10_equiv_ceil` ran the regeneration task **with**
`--equivariant-move` for 2,000 updates and its emitted move was **not aimed**:
align **-0.096** against a convention-matched null of **+0.206**, below even an
untrained equivariant head
([[the-equivariance-fix-did-not-release-a-bearing]]). The fix made aiming
*representable* and nothing started aiming.

The measurement above says why that is not a contradiction. Given a signal that
says which way to go, the same architecture reaches cosine 0.96 in a hundred
updates. So what g10 was missing was never expressivity -- it was a **bearing**,
and the record already said so in the FAPE node's words, "a magnitude and not a
bearing" ([[fape-can-teach-the-frame-head-which-way-to-move]], and the clamped
FAPE node for why a clamp flattens it). A diffused morphogen gradient is
exactly a quantity that carries a bearing: it is a vector, it is in the right
frame, and it is non-local in a way one message-passing hop is not.

**That is a hypothesis, not a result.** Nothing has trained the field on the
regeneration task, and the chemotaxis task handed the rule a source whose
position was the answer, which regeneration does not.

Two caveats on the chemotaxis number, kept because they bound it. The loss was
on the **normalised** translation, so it tested bearing alone and says nothing
about the magnitude pathology -- a trained head asking 5.4x the cap with its
magnitude gradient attenuated ten-thousandfold
([[the-move-heads-magnitude-gradient-is-attenuated-three-thousandfold]]); the
mean step here was 0.033 A, far inside the cap and nowhere near that regime.
And a single structure was used throughout, so the rule could not learn
anything structure-specific but also never had to generalise across folds.

## Three gaps, and they are the honest answer to the question

  1. **Orientation cannot follow anything.** `lab_rotation = true` in
     `[base.inputs]` and in **every arm ever run**: the translation is read in
     the residue's frame, the **rotation is still applied in the lab frame**.
     An invariant encoder cannot represent a lab-frame rotation axis, so the
     rule can express "move up the gradient" and cannot express "turn toward
     it". `g23_localrot` is the arm that would fix it and it has never run;
     the inexpressibility is measured at 0.68 A per step on a 1.50 A move.
     Position is half of a frame; secondary structure is mostly the other half.
  2. **No arm has ever had a field at all.** `morphogen = false` everywhere, so
     every number above is a probe, not a result from the regeneration task.
  3. **At initialisation the field is saturated and flat.** An untrained
     sigmoid emit head secretes about 0.5 *everywhere*, so after 20 steps on
     1UBQ the concentration is 0.8935-0.9038 and the gradient block reads
     |value| 0.0032 mean, 0.016 max. The rule has to learn to secrete
     **differentially** before the block carries anything, and nothing in the
     objective currently rewards that. A chemotaxis loss made it happen in 100
     updates; regeneration might not.

So: a node can see the gradient and follow it with its position, by a circuit
that is now measured rather than assumed, and it cannot yet turn toward it.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
