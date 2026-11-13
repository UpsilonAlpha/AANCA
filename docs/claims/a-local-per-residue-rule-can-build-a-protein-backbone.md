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

## What the rule is conditioned on, and what the damage protocol is FOR

Ulban, 13 Nov, and the second half of this was not in the record at all.

**The conditioning target.** Each residue is conditioned on its neighbouring
residues -- **and eventually on ligands** -- the way ProteinMPNN is, and the
intent is for that conditioning to be *physically realistic* in the sense
`src/aanca/potts.py` already aims at rather than a generic learned embedding.
Ligand conditioning is the direction LigandMPNN took from ProteinMPNN (atomic
context over small molecules, nucleotides and metals, 50.4 % -> 63.3 % sequence
recovery near small molecules), and it is a stated destination here, not a
speculation. Nothing in the growth stack reads a non-protein atom today.

**Repeated damage and regrowth is an instrument, not just a training task.**
The record had regeneration as the near target because the surviving context
supplies the positional information a Growing NCA gets from its grid
([[in-regeneration-the-surviving-context-is-the-grid]]). That is still true and
it is only half the reason. The other half: a rule that can destroy and regrow
an arbitrary section **repeatedly** can be interrogated by doing exactly that --
regrow one loop a hundred times and the spread of the outcomes is a measurement
of that loop's conformational freedom, made with a local rule rather than
inferred from a global model's confidence.

That makes the unmeasured persistence question
([[the-growth-objective-supervises-a-fixed-final-step-while-the-track-that-converged-samples-one]])
much more than a convergence check. The dispersion of a settled window past the
training horizon **is** a per-residue flexibility readout, and flexibility is
the quantity that binding-affinity prediction is worst at. A pairformer can be
asked for an affinity; it cannot cheaply be asked "how many ways can this loop
sit". This architecture can, and that is a capability the interpretability
argument gets for free rather than a separate feature.

## How this differs from diffusion, which is the question the claim lives or dies on

Asked 9 Nov, and the answer is uncomfortably short: **an AANCA is a diffusion
model with three things removed.** The similarity is real -- both are iterative
refinement with weights shared across steps -- so the content of the thesis is
entirely in the removals.

    diffusion (RFD4, Chroma, RFdiffusion)   this rule
    ------------------------------------    ---------------------------------
    timestep is an INPUT to every cell      time exists; no cell can read it
    noise is the corruption                destroyed sections are
    all-pairs attention each step           one hop per step, every step
    one-step denoising at sampled noise     BPTT through a truncated rollout
    stops because the schedule ends         settles into a realistic basin

**Corrected by Ulban, 9 Nov, on two rows.** The first is not "no timestep":
*"There is kind of a time step, as the network needs to grow and settle, the
individual AANCAs just cannot know it."* Global time exists and the trajectory
has a direction; what is withheld is any cell's *access* to it, so the rule
cannot learn the index in place of the structure. And the corruption is not
noise: sections are **destroyed** and must regenerate, which is Growing NCA's
damage protocol rather than a noise schedule.

The last row was overstated as "must converge to a fixed point". *"We want
convergence to a realistic local minimum, doesn't necessarily need to be a fixed
point, as that is unrealistic / not physical."* Correct, and the overstatement is
load-bearing rather than cosmetic: **an exact fixed point is precisely the
property the degenerate solution has.** The identity map is a perfect fixed point
([[the-rule-keeps-whatever-conformation-it-is-given]]), so an objective that
rewards *stopping* rewards the thing four arms did. A bounded basin -- the
structure fluctuating within a band around a realistic conformation -- is both
physical and distinguishable from never having moved. `rollout.py`'s own
`sample_loss_step` docstring carries the same overstatement and should be read
the same way.

**This is also a measurement gap.** Every growth number is read at step 120 and
nowhere else, so the record cannot currently distinguish *settled in a basin*
from *frozen* from *will drift away by step 300*. E(n)-GNCA reports its attractor
maintained for 1000 steps past the training horizon; the equivalent here is to
roll past 120 and report the dispersion, which nothing does.

`src/aanca/model.py` states the first two as the claim itself: *"No step index,
no noise-level conditioning, no global pooling, no attention over the whole
protein. That absence is the scientific claim -- do not let a time embedding
creep in."*

**Each removal has a cost the record has already paid.**

  * **An unreadable timestep** is why the identity map is a seductive degenerate
    solution and why four arms froze at their own baselines
    ([[the-rule-keeps-whatever-conformation-it-is-given]]). A diffusion model
    cannot freeze: `t` forces progress whether the network has learned anything
    or not. We ask the rule to decide *for itself* that it has arrived -- and
    the repository already holds the mechanism for that, unused by this track
    ([[the-growth-objective-supervises-a-fixed-final-step-while-the-track-that-converged-samples-one]]).
  * **One hop per step, not one hop total.** Perception is refreshed every step
    from a graph rebuilt at the current coordinates, so a cell is never stale;
    what it is not is *deep* within a step. Two stacked hops would make a single
    application globally connected on a small protein -- degree 28-33 -- and
    nothing in the loss curve would reveal it, so the limit is the claim rather
    than a budget. This is why perception from wrong coordinates is the
    central problem
    ([[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]), and
    it is what `native_edges` was papering over.
  * **Training through the rollout** is why budget turned out to dominate every
    lever ([[the-lever-table-was-read-at-a-fifth-of-the-budget-the-arm-needed]])
    and why `truncate 12` is a liability: it cuts the gradient through the
    iteration that E(n)-GNCA's authors identify as the whole source of global
    reach ([[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]).

**Locality is not the same as a small receptive field, and conflating them
understates the claim.** The measured mean degree at the 10 A cutoff is 28-33,
so one hop reaches a quarter to a third of a small protein and two reach all of
it; a 120-step rollout crosses the molecule many times over. The converged rule's
fixed point therefore depends on everything. What is local is the *function*, not
its reach.

That is exactly why the interpretability payoff holds. A pairformer is many
large functions over all pairs, conditioned on a noise level. This is **one small
function of a neighbourhood, applied repeatedly** -- 229,588 parameters against
tens of millions -- so the thing to ablate or autoencode is small even though
what it computes is global. `docs/PLAN_V1.md` already concedes the weaker form
of the result: *"a rule landing close to a globally attentive diffusion model is
already the interesting result."*

## Supported by
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Refuted by
- [[the-rule-keeps-whatever-conformation-it-is-given]]

## Requires
- [[question-can-the-architecture-place-residues-under-any-supervision]]
- [[the-rule-can-move-a-residue-far-enough-to-reach-the-native]]
- [[fape-can-teach-the-frame-head-which-way-to-move]]
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
