---
type: thesis
status: unsupported
---

# One local update rule, applied per residue, can cofold a protein -- sequence and structure together -- with no global coordinator

**The root, and deliberately the strongest assertion the project makes.** It
was briefly demoted on 6 Nov in favour of the regeneration claim, which was the
wrong way round: argmap's `supports` is evidence-for rather than entailment, so
the strongest assertion belongs at the root and the achievable results feed it.
Regeneration succeeding would be strong evidence that a local rule can place
residues from local information, without being the whole claim -- so it
`supports` this, not the reverse
([[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]).

`unsupported`: no run has produced a backbone measurably closer to its native
than the conformation it started from. Four arms of 8,000 updates each ended at
their own baseline, and because they tested *this* claim directly rather than
the regeneration one, they are evidence against it and are attached below as
refutations.

What works and is not evidence for this: the self-placing architecture does
what it was built to do, and the rollout, data pipeline and optimiser are
demonstrably functioning. The geometry is the part that has never moved.

Its premises are listed below, and both of the hard ones are failing.

## The task is local cofolding, and why that is the interesting one

Stated 9 Nov, and the file was wrong until then -- it said *backbone*, which is
half the task.

**Inverse folding is done.** A local rule reads a structure and recovers its
sequence: **38.0 % on held-out structures**, converged, against a ~9 %
composition background. That is sequence given structure, and it is a result,
not a premise.

What is open is the **joint** direction: emit sequence *and* structure at once,
from local information only. That is the setting the diffusion models work in --
RFD4 and its relatives -- except that they get global information, a pairformer
with all-pairs attention mixing every residue with every other at every block.
This project asks whether the same job can be done with **one shared rule, one
message-passing hop, no global coordinator and no pooling**.

**The payoff is interpretability, and it is the reason the constraint is the
point rather than a handicap.** A single small local rule can be ablated
channel by channel, or run through an autoencoder, in a way a pairformer
cannot: there is one function, shared across every residue, with a hidden state
small enough to read. If a local rule cofolds at all, then what it learned
about structural interactions is *recoverable*. That makes `rmsd_dmg`
instrumental -- the number is how we know the rule works, not what we want from
it.

Two consequences the record has not been honouring:

  * **A ceiling that supplies interactions destroys the payoff, not just the
    claim.** `--native-edges` hands the rule the target contact graph. The
    target contact graph is precisely the structural-interaction information an
    ablation or an autoencoder is meant to discover. An arm that reads it can
    never report anything about interactions, because they were an input.
  * **Every arm since g9 has had the sequence half switched off.**
    `identity_weight = 0.0` is the config base, and g9 through g18 -- including
    the 6.69 A result of 9 Nov
    ([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]]) --
    all ran there. The last arm to supervise sequence and structure together was
    **g7_regen**, before the equivariance fix, and it froze
    ([[g7-regen-found-a-fixed-point-that-is-not-the-identity-map]]). So the
    project holds one solved half, one ceiling-bound half, and **no arm in the
    joint setting since the architecture was repaired.**

Cofolding is not the conjunction of two working halves. In the joint setting the
rule perceives its own wrong coordinates *and* its own uncommitted labels, and
the failure that only exists there -- sequence and structure co-adapting to a
mutually consistent wrong guess -- has never been given a chance to appear.

## Supported by
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Refuted by
- [[the-rule-keeps-whatever-conformation-it-is-given]]

## Requires
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
- [[fape-can-teach-the-frame-head-which-way-to-move]]
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
