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

## Two sources, because one fixes a bearing and two fix an orientation -- and that claim is weaker than it sounds

The argument for a *pair* of sources is that an orientation needs two
independent directions, where a bearing needs one. Measured on 1UBQ at steady
state, with both gradients taken in each residue's own frame:

    cos(grad_nward, grad_cward)        -0.8030   mean over all 76 residues
    both gradients non-degenerate       76/76
    |cos| < 0.90                        46/76
    |cos| < 0.99                        72/76

**So they are mostly anti-aligned, not independent.** One source points up the
chain and the other down it, so the pair spans a line plus a perpendicular
component of about 0.6 of the magnitude -- real, but a long way from a frame.
Forty-six of seventy-six residues have enough independence to span a plane.
The honest statement is that two sources give one strong axis and a weak second
direction, not a coordinate frame, and a third source would be needed before
"the frame reads its angles off the field" is literally true.

## And it cannot turn yet, whatever the field says

`lab_rotation = true` in `[base.inputs]` and in every arm ever run: the
translation is read in the residue's own frame and the **rotation is still
applied in the lab frame**, which an invariant encoder cannot represent
([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]],
[[the-rotation-half-of-the-equivariance-fix-is-still-switched-off]]). A
perceivable reference axis does not help an output that is interpreted in the
wrong frame. **`g16_localrot` is a prerequisite for this premise, not an
alternative to it.** Position can follow the field today; orientation cannot,
and the measured bearing failure ([[the-equivariance-fix-did-not-release-a-bearing]],
align -0.096 against a +0.206 null) is about orientation.

## The thing to be suspicious of, measured rather than argued

A field with sources at the termini is **exactly what the retired
chain-transported design was**, and that was measured to be `decay ** index` --
a positional encoding ([[the-chain-transported-morphogen-is-a-positional-encoding]]).
The defence is that this one diffuses over the perception graph, so
concentration falls with distance *through space*. Spearman correlation of each
polarity channel against the residue index, 1UBQ, 60 steps to steady state:

    start state        rho(nward, i)   rho(cward, i)   |grad| mean
    native fold            -0.2120         +0.1596        0.0832
    helix baseline         -1.0000         +1.0000        0.0604

**On the native fold it is not a coordinate; on the helix baseline it is
exactly one.** Rank correlation 1.00, which is the retired module's failure
reproduced in full.

That is a real finding and it cuts against the premise, because **a growth
rollout starts at the helix baseline**. So the field begins as a positional
encoding and stops being one only as the structure folds -- which is the "right
kind of wrong" the retired node argued for, now with numbers on both ends
rather than an assurance. Whether a rule that reads a coordinate early learns
to stop relying on it later is not something this measurement answers.
`tests/test_morphogen.py` pins both halves on a synthetic chain: rho is exactly
-1 and +1 on a straight graph, and adding one spatial edge between the ends
breaks monotonicity.

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
