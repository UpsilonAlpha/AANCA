---
type: claim
status: open
sources: [potts.py, backbone.py]
---

# Repeated destruction and regrowth of one window yields a local statistical potential, which is the thing a pairwise diffusion model cannot sweep

Ulban's programme, stated 13 Nov, and it is the first version of the
interpretability payoff that names a deliverable instead of an aspiration.

## The idea

ProteinEBM's move is that a diffusion model's score **is** an energy gradient,
so a generative model over structures read backwards is a statistical
potential: `E = -log p(structure)`.

Do that locally. Destroy a window, regrow it N times, and the empirical
distribution over the window's conformations -- conditioned on the context that
survived -- is `p(local conformation | environment)`. Its negative log is a
**local statistical potential**, and the potential is the product, not the
structure.

## Why a pairwise diffusion model cannot do this, stated carefully

Not "cannot produce a local conditional" -- it can, by inpainting with the rest
held fixed, which is what motif scaffolding does. The claim has to be narrower
and it is still strong:

**Every sample costs a full global denoising trajectory with all-pairs
attention at every step.** So one window's conditional is affordable and a
*dense sweep* of every window in a protein, at N samples each, is not. The
AANCA's sampling cost is local -- one window, one rollout, one shared rule over
a neighbourhood -- so the sweep is the thing it can do and the pairformer
cannot. The deliverable is a **map** of local potentials over a structure, and
the map is combinatorially out of reach for a global model.

That is the honest form of "in a way pairwise diffusion can't", and it is an
argument about cost scaling rather than about expressive power.

## Potts is the right summary, and the machinery is already here

A Potts model is `E = sum_i h_i(s_i) + sum_ij J_ij(s_i, s_j)` over discrete
states. Fit one to the regrowth ensemble and the potential arrives in exactly
the form the interpretability argument needs: **explicit fields and explicit
couplings**, ablatable term by term, rather than a hidden state to be probed.

**`src/aanca/potts.py` already implements this and is agnostic to what the
states are.** `PottsModel` carries `h` as `(n_res, n_types)` and `J` sparsely
as `(n_edges, n_types, n_types)` over an explicit edge list, with `energy`,
`delta_energy` and `edges_from_adjacency` beside it. It was built for sequence,
where `n_types` is 20 amino acids. Nothing in it requires that: **swap the
alphabet for K backbone conformational basins and the same code fits a
conformational Potts model.** `backbone.measure_backbone_torsions` returns
`(phi, psi, omega)` per residue, so the discretisation input exists too.

So the two pieces this needs are both built, for a different purpose.

## And it makes CDR pre-organisation a computable quantity

This is where the binder hypothesis becomes a prediction rather than a story.
**Pre-organisation is low conformational entropy of the local potential.** A
loop whose residues are strongly coupled has few accessible joint states; a
floppy loop has many. From a fitted Potts model both are readable:

    coupling strength  |J_ij|     how much neighbouring residues' states
                                  constrain each other -- pre-organisation
    field concentration h_i       how strongly one residue prefers one basin
    model entropy                 the number of joint states actually occupied

That is a per-loop rigidity number computed from a local rule, and it is the
quantity Ulban's hypothesis says separates binders that work and are
predictable from binders that do not.

## The objection that matters most: the current objective suppresses the ensemble

**A statistical potential needs a distribution, and this rule is trained to
produce a point.** FAPE plus restraints against a *single* native structure,
scored at one fixed step, pushes the regrowth distribution toward a delta
function. The better the arm gets at `rmsd_dmg`, the less of a distribution
there is to take the log of
([[question-does-regrowth-dispersion-measure-conformational-freedom]]).

So the missing Growing-NCA ingredients stop being housekeeping and become
load-bearing: the **sample pool** and a **sampled or convergence-determined
rollout length** are what make the output an ensemble rather than a regression
([[update-probability-is-a-result-affecting-setting-that-no-config-records]]).
And the ensemble has to come from varied **initial conditions**, with the
mask-only spread as the control -- fitting a Potts model to update-mask noise
would produce a confident and meaningless potential.

## The chain has three links and only one is being tested

Added 13 Nov, answering "how does this help achieve our objectives". The route
from the sample pool to a usable potential is **pool -> persistence ->
distribution -> potential**, and the arm that was just built tests the first
arrow only.

  1. **pool -> persistence.** `g21_pool`'s question. It can fail outright: if
     divergence is architectural -- the rule having no way to represent "I have
     arrived" -- then no training protocol fixes it, which is that arm's fourth
     bucket and says go architectural.
  2. **persistence -> a usable distribution.** *Not tested by that arm, and the
     pool may work against it.* A rule that holds its structure perfectly can
     have **zero** spread, and a delta function is as useless here as a
     diverging trajectory. Worse, the pool's guard keeps only end-states no
     worse than the baseline, which is selective pressure toward one good
     answer -- so persistence could be bought by collapsing the multiplicity it
     was wanted for. A third pre-registered bucket on the step-120 ensemble
     spread was added to catch exactly that.
  3. **distribution -> potential.** Still `-log` of optimiser outcomes rather
     than of a Boltzmann distribution, with the B-factor correlation as the only
     external check and that check untried.

So the pool is **necessary and not sufficient**, and its honest value is that it
is the cheapest way to learn whether divergence is a protocol problem or an
architectural one. Either answer redirects the project, which is what makes it
worth a slot.

**And it has a justification that does not depend on this programme at all.**
Every number this project has ever reported is read at step 120. If the rule
does not hold its structure, every `settled` value is a snapshot of a transient
-- the 7.78 A included
([[the-rule-has-no-attractor-past-its-training-horizon]]). Persistence is not
only a prerequisite for the potential; it is what would make the existing
numbers mean what they are reported as meaning. That reason survives even if
the potential programme is abandoned.

## And the honest limit on what "statistical potential" can mean here

It is a potential in the knowledge-based sense -- `-log` of an observed
frequency, which is how DOPE and DFIRE are built from PDB statistics -- and
**not** a free energy. The rule is trained to reproduce native structures, so
its regrowth distribution is a distribution over *optimiser outcomes*
conditioned on context, with no temperature and no guarantee of relating to
thermodynamics. Calling it a statistical potential is fair; calling it a free
energy is not, and the difference matters the moment anyone wants a Kd out of
it.

What would narrow the gap: training across many contexts, so the learned
conditional approximates `p(local conformation | environment)` over the data
rather than the idiosyncrasy of one target. That is what a pool buys, and it is
the same argument from the other direction.

## Why this is testable before the geometry is good

At `settled` 7.78 A the regrown windows are badly wrong
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]),
and a potential over wrong conformations is a wrong potential. **But the
rigidity readout only needs the *spread* to be informative, not the mean to be
right.** Whether the rule ranks flexible loops above rigid ones is a far weaker
requirement than whether it places them correctly, and it may already hold.

That is the argument for running the B-factor correlation now rather than after
accuracy improves, and it is the cheapest possible first test of this whole
programme.

## What would support this claim

  1. Ensemble regrowth with varied initial conditions, mask-only control, on
     `g20_orient`. Per-residue spread.
  2. Spearman of that spread against per-residue backbone B-factor -- the
     external anchor, and the first `experimental` evidence under the thesis.
  3. Discretise to Ramachandran basins and fit `potts.py` over the ensemble;
     check the couplings are structured rather than flat.
  4. Only then CDRs, against free-versus-bound antibody pairs where the
     conformational change on binding is known.

ProteinEBM's weights are on Bunya and are useful here in the **scorer** role --
Boltzmann reweighting of the ensemble at a sigma matched to our displacement --
which is the one way to put a temperature on this without training against it
([[question-can-proteinebm-be-reused-and-in-which-of-three-ways]]).

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]

## Requires
- [[question-does-regrowth-dispersion-measure-conformational-freedom]]
