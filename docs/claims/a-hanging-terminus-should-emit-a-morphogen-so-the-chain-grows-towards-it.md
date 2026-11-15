---
type: claim
status: open
sources: [morphogen.py, rollout.py, growth.py, a-residue-can-follow-the-morphogen-gradient]
---

# A hanging terminus should emit a morphogen, and the frame reads its angles off the N-C gradient

Ulban's premise, 15 Nov: *"a hanging N terminus should probably emit a
morphogen so that the chain can 'grow towards' it"*, then the correction that
matters: *"it would mean that frame angles place themselves influenced by an
N-C gradient."*

**Built on 15 Nov. Nothing has trained with it.** The correction is the whole
design: this is not an attractor pulling residues towards a cut end, it is a
**local coordinate system** that a frame reads its angles against. That is
Wolpert's positional information rather than a tractor beam -- every cell reads
the same gradient and does something different with it -- and it dissolves two
of the four objections the first version of this node raised.

## What exists

  * `morphogen.terminus_emission(links, n)` -- 1.0 in channel 0 at every
    residue whose **N side** has no covalent neighbour, 1.0 in channel 1 at
    every residue whose **C side** has none. A given input, not a prediction.
  * `TERMINUS_CHANNELS = 2`, `LEARNED_EMIT_CHANNELS = 4`,
    `EMIT_CHANNELS = 6`. The polarity pair is an **addition** to the four
    learned channels rather than a tax on them, so the signed-charge and
    unsigned-hydrophobicity budget survives. `FIELD_NODE_EXTRA` 20 -> 30,
    node_dim 86, parameters 251,160 -> 256,920.
  * `morphogen.compose_emission` fixes the channel order -- structural first,
    secreted after -- so a field tensor means the same thing in every arm.
  * `RolloutConfig.terminus_source`, default **off** so an arm with it is
    comparable against one without. The source is **re-injected every step**,
    because it is a boundary condition and not an initial condition:
    degradation would erase it, and an end that stops hanging must stop
    secreting.

Measured end to end on 1UBQ, 20 steps: with the flag off the polarity channels
are exactly 0.0; with it on they span **0.0003 to 1.0000**, while the learned
channels sit at 0.907 -- saturated and flat, which is the initialisation
problem this was meant to route around, visible in the same tensor.

## Two sources give one axis, and the anti-alignment is the signal

Ulban, 15 Nov: *"two gradients should be anti-aligned, they're coming from
different termini - exactly french flag style."* Correct, and it corrects a
claim made here a few hours earlier. Measured on 1UBQ at steady state, both
gradients in each residue's own frame:

    cos(grad_nward, grad_cward)        -0.8030   mean over all 76 residues
    both gradients non-degenerate       76/76

They are anti-aligned, and **that is what an opposing pair is for.** The
French flag model is one coordinate along one body axis, read differently by
every cell according to its own identity, and the biology is a pair of
opposing gradients -- anterior and posterior -- whose *ratio* is the readout.
So the quantity to measure is not the angle between them but

    r = log(c_nward) - log(c_cward)

which is a signed position along the axis joining the two sources.

**What this retracts.** An earlier draft of this node said two sources were
meant to fix an *orientation*, and read the anti-alignment as that claim
failing. Two opposing gradients do not fix an orientation and were never going
to: they fix a **position along an axis**, which is both what the French flag
needs and what the next section measures. The 0.6 perpendicular component is
incidental, not the point.

## And it cannot turn yet, whatever the field says

`lab_rotation = true` in `[base.inputs]` and in every arm ever run: the
translation is read in the residue's own frame and the **rotation is still
applied in the lab frame**, which an invariant encoder cannot represent
([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]],
[[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]). A
perceivable reference axis does not help an output that is interpreted in the
wrong frame. **`g23_localrot` is a prerequisite for this premise, not an
alternative to it**, and the inexpressibility it removes is now measured at
0.68 A per step on a 1.50 A move -- 45 % of the move, larger in proportion
than the translation half the project has already fixed. Position can follow the field today; orientation cannot,
and the measured bearing failure ([[the-equivariance-fix-did-not-release-a-bearing]],
align -0.096 against a +0.206 null) is about orientation.

## It is a spatial coordinate, not a positional encoding, and the degeneracy belongs to the start state

A field with sources at the termini is **what the retired chain-transported
design was**, and that one was measured to be `decay ** index`
([[the-chain-transported-morphogen-is-a-positional-encoding]]). So the test is
whether `r` tracks **where a residue is in the folded body** or **what its
index is**. Against the residue index, and against `x`, the CA projected on the
unit vector from the first CA to the last, with `rho(x, index)` as the control:

    start state         rho(r, index)   rho(r, x)   rho(x, index)   end-to-end
    native                    -0.189      -0.972          +0.241       37.1 A
    12-residue window         -0.319      -0.925          +0.404       37.1 A
    24-residue window         +0.109      -0.970          -0.066       37.1 A
    helix baseline            -1.000      -1.000          +1.000      113.9 A
    random baseline           -0.934      -0.953          +0.871       49.4 A
    extended baseline         -1.000      -1.000          +1.000      273.2 A

**`rho(r, x)` is -0.92 to -1.00 in every row.** The field reports position
along the body axis, faithfully, whatever the geometry -- which is what a cell
can actually sense and is *not* what the retired module did, because that one
reported the index regardless of geometry.

What varies down the table is the **control**. On a folded structure the body
axis and the sequence axis are different things (`rho(x, index)` = 0.24), so a
spatial coordinate is not an index. On any torsion-walk baseline they are the
same thing (1.000), and the field reads as an index because *on that start
state position is the index*. The degeneracy is a property of an unfolded
chain, not of this function.

**So "don't use a helix baseline" is right as a direction and does not fix
it.** The random mode only moves the control from 1.000 to 0.871, because any
walk with ideal covalent geometry is extended -- 49 A end to end against the
native 37 A -- and every extended chain has this degeneracy. No choice of
torsions removes it.

**What removes it is the task.** Regeneration keeps the intact remainder at its
crystal pose, so the structure is folded from step 0 and the control sits at
0.40 and -0.07 for a 12- and a 24-residue window. The problem is confined to
whole-structure growth, which the record already holds to be harder than and
different from the goal
([[whole-structure-growth-is-a-harder-and-different-task-than-regeneration]]).

`tests/test_morphogen.py` pins the mechanism on a synthetic chain: on a
straight graph the readout is exactly monotone in the index, and adding one
spatial edge between the ends breaks it.

## Two objections the polarity framing dissolves

  1. **"A scalar source says *towards a terminus*, not towards *your*
     terminus."** Withdrawn. The field is not asked to identify a partner --
     N-ward and C-ward are globally meaningful, so a residue reads *which way
     the chain runs* rather than *where my partner is*. Partner identity was
     only ever needed by the attractor reading.
  2. **"A source attracts everything, not only its partner."** Withdrawn, and
     it was backwards. Every residue reading the same gradient is the
     mechanism, not a side effect: a morphogen gradient is positional
     information, and the French flag model is exactly one gradient read
     differently by every cell according to its own identity -- which the node
     block already carries.

## What is left

  * **The reach is unmeasured outside ubiquitin.** `DEGRADATION = 0.10` was set
    from a three-hop diameter on 1UBQ and gives a screening length near 9 A.
    A larger protein has a larger diameter.
  * **A gradient is a bearing, not a bond.** It gets ends near each other and
    cannot close them, so this is complementary to a relaxation step per birth
    rather than a replacement for it.
  * **Nothing has trained.** `morphogen = false` in every arm, and
    `terminus_source` defaults off, so every number above is a probe.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Requires
- [[a-residue-can-follow-the-morphogen-gradient]]
- [[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]
