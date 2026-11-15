# AANCA — handoff brief

AANCA = amino-acid neural cellular automaton. Read this file first. The dated
journal is `docs/JOURNAL.md`; the claims and their evidence are one note each in
`docs/claims/`, validated by `docs/argmap/argmap.py`.

---

## 0. Current state — 2026-11-14

### The question, rescoped 14 Nov

**Can a very simple shared local rule — one function, 229,588 parameters,
applied per residue in parallel with no global coordinator, no pooling and no
attention — regrow a damaged region of a protein from the context that
survived?**

That is the whole central argument. Not a design method, not a competitor to
diffusion: a study of whether a local update rule can learn enough real physics
from structure data to put destroyed atoms back where they belong.

### What this is explicitly not

Read against the RFD4-Proteina preprint on 14 Nov
([[what-an-aanca-can-learn-that-a-pairformer-cannot-and-the-reverse]] carries
the head-to-head). They are ~435M parameters across a flow model and an
autoencoder, a 147-class chemical vocabulary, seven benchmark tasks, and
wet-lab-validated binders at a 10.6 % clustered pass rate and 42-fold the
throughput of BindCraft. This is ~1/1900 the size and does a fraction of one of
their tasks, worse.

So "which designs better" is settled and was never the question. The question
is **what is the least information a rule needs**, and the useful framing is
that this rule is roughly RFD4's own inner recycling loop — they recycle at
every diffusion timestep, with a variable pass count — **with everything around
it deleted**. Theirs reads the pair track and the global time while it iterates;
ours reads one hop of a 10 A heavy-atom graph and nothing else. That is the
ablation, and that is the project.

Two things therefore stop being contributions and are recorded as such:

  * **Design performance.** Not attempted, and should not be claimed.
  * **Local-ensemble spread as a readout.** Published twice — PLACER
    (Anishchenko *et al.*, *PNAS* 2025) and RFD4's reproduction of it. The
    downstream programmes (a fitted Potts potential, CDR pre-organisation, the
    binder hypothesis) stay in `docs/claims/` as what regeneration would *buy*,
    and they are strictly downstream of the central claim. Nothing there is
    worth running while the central claim is unsettled.

### Where the central claim actually stands

Two tracks, and only one is finished.

    track                        state
    ---------------------------  -------------------------------------------
    inverse folding              DONE, 38.0 % held-out, converged, against a
                                 ~9 % composition background. The architecture
                                 works and can learn from local context.
    regeneration                 PARTIAL AND UNREPLICATED. See below.
    joint sequence + structure   NEVER RUN since the equivariance fix.

**13 Nov was the first capability rather than a bound.** `g20_orient` placed a
damaged 12-residue window at **7.78 A** from its 12.23 A baseline on 12 held-out
crops, with no teacher forcing and no target graph
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]).
Nine earlier arms without teacher forcing had sat at their own baselines to two
decimal places, so this is a real signal and not a readout artefact.

**And it does not persist.** The same rule read at 600 steps is at 16.4–16.9 A,
worse than the 12.23 A untrained baseline — there is no attractor past the
trained horizon ([[the-rule-has-no-attractor-past-its-training-horizon]]). Every
number this project reports is a snapshot of a transient at step 120.

### The three things that decide the central claim

In this order, and nothing downstream matters until they are answered.

  1. **Is 7.78 A real?** No arm in this project has been replicated at a second
     seed, because `seed` also chooses the held-out set, so the one replicate
     attempted measured different proteins
     ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]). This needs
     `eval_ids` as a config key pinning PDB ids rather than integer indices, and
     one run. It is the cheapest thing in the queue and it gates everything.
  2. **Does the rule hold what it builds?** `g21_pool` (32904055) is the
     cheapest test of whether divergence is a protocol problem or an
     architectural one — the rule having no way to represent "I have arrived".
     Pre-registered in three buckets, primary the 600-step reading.
  3. **Is 7.78 A better than the prior it starts from?** The damage operator is
     **not noise**: `baseline_coords(..., initial="helix", ...)` discards the
     window's native pose and regrows it in torsion space off the intact flanks
     at ideal (phi, psi, omega). So the start state is an informative prior, and
     the honest question is how much of the 4.45 A of progress is the rule and
     how much is reachable by relaxing an ideal helix against fixed flanks. That
     control has not been run and it should be.

The interpretability payoff — one small shared function that can be ablated
channel by channel, where a pairformer conditioned on a noise level cannot — is
why the answer is worth having. It is not evidence for the claim, and it does
not substitute for the three runs above.

Locality is not a small receptive field. At the measured mean degree of 28–33, a
120-step rollout crosses the molecule many times, so the fixed point depends on
everything. What is local is the **function**, not its reach.
### How it differs from diffusion

    diffusion (RFD4, Chroma, RFdiffusion)   this rule
    ------------------------------------    ---------------------------------
    timestep is an INPUT to every cell      time exists; no cell can read it
    noise is the corruption                 damage is a PRIOR, not noise:
                                            an ideal helix regrown off
                                            the intact flanks
    all-pairs attention each step           one hop per step, every step
    one-step denoising at sampled noise     BPTT through a truncated rollout
    stops because the schedule ends         has no schedule and, measured,
                                            does NOT settle -- 16.4-16.9 A
                                            at 600 steps

Not "no timestep": global time exists and the trajectory has a direction, but no
cell can read it, which is what stops the rule learning the index instead of the
structure. And **not** "must converge to a fixed point" — an exact fixed point is
the property the degenerate identity map has, so an objective that rewards
stopping rewards what four arms did. A bounded basin is physical and
distinguishable from never having moved.

### The arm details behind those numbers

Removing both rollout ceilings costs 1.09 A against `g17_long` at matched budget,
so 7.78 A with no ceiling sits against 6.69 A under two. `twist` settled at
11.24 against g19's 14.26, so the orientation term built on 5 Nov controls the
quantity it was built for. `g19_nomove`'s placement number is void by its own
pre-registration — twist left the 9-12 band — and stands as that control. The
12 held-out crops are the same ones every comparator used, verified from the
checkpoints.

`identity_weight = 0.0` is the config base, so every arm from g9 to g18 had the
sequence half switched off. The last joint arm is `g7_regen`, which predates the
equivariance fix and froze 1.5 A the wrong way. **The joint arm is the test of
the rescoped question that has not been run**: regrowing a window means
regrowing its sequence too, and nothing since the equivariance fix has asked for
both at once.

### What was removed, and why it matters

`native_edges` held the perception graph's neighbour selection at the native
edge set. **Removed 9 Nov**, and five arms went with it (g13, g15, g16, g17,
g18 — deleted from `configs/regeneration.toml`, reproducible at commit
`bd0c7f8`). The reason is not that it was a ceiling: who contacts whom *is* the
structural-interaction topology an ablation would be trying to recover, so
handing it over means the rule never learns it and there is nothing in the
weights to read back. `g14_pairwise` is the only historical arm the config can
still express.

### Next

Reordered 14 Nov by the rescope. The ordering rule is now: **anything that
decides whether the rule can regrow damage comes before anything that uses the
fact that it can.** `rmsd_dmg` is how we know the rule works — not the
deliverable, and not a number to chase on its own.

1. **The replicate arm, with `eval_ids` pinned.** The one run that turns the
   central claim from `open` to supported, and it is the cheapest thing in the
   queue. Note the rename from `eval_indices`: pinning integer **indices** only
   half fixes it, because the pool is `available[:412]` over whatever is on
   disk, so the same index is a different protein on a 954-structure laptop and
   a 23,760-structure Bunya — which is what voided the 10.77 A baseline. **Pin
   PDB ids.** Then move `seed` with the held-out set held fixed.
2. **Read `g21_pool` (32904055) against its three pre-registered buckets.**
   Primary is the 600-step reading from `scripts/regrowth_dispersion.py`;
   second is `settled` at step 120 within 1.0 A of g20's 7.78; third is the
   step-120 ensemble spread against g20. This is the test of whether divergence
   past the trained horizon is a protocol problem or an architectural one, and
   either answer redirects the project. *Blocked while Bunya login is
   disabled.*
3. **`g22_random` -- take the helix away.** Pre-registered 14 Nov on Ulban's
   objection that the model should learn to form helices and sheets rather than
   be handed one. Every arm from g1 to g21 ran `initial = "helix"`, so the
   damaged window has always *started* as an ideal helix and the rule has only
   ever been asked to move one -- which makes "it learns secondary structure"
   untestable. Measured cost of `random`: 3.35 A further out, FAPE 87.1 % live,
   bond 5.49 against 4.50, well inside the 36 A reach ceiling
   (`docs/claims/the-start-state-hands-the-rule-its-secondary-structure.md`).
   One variable against `g20_orient`. Two buckets, and the second -- what
   fraction of natively-helical window residues land in the helical
   Ramachandran basin from a uniform start -- is the result to lead with, not
   `rmsd_dmg`. **This replaces the helix-relaxation control**: from a random
   start there is no secondary structure to inherit, so the null needs no
   separate defence.
4. **The joint arm, `identity_weight` on.** Regrowing a window means regrowing
   its sequence, and no arm since the equivariance fix has asked for both at
   once, so the rescoped question has never actually been put to the rule in
   full. Also the only route to side chains in the growth track, which is where
   everything downstream needs them.
5. **The eval-gating fix, two parts**, once 32904055 is done: ungate the
   held-out score from the training step at `scripts/train_growth.py:1327`, and
   make the pool's two new failure paths fall back to a fresh baseline draw
   rather than `continue`
   (`docs/claims/a-skipped-update-silently-skips-its-evaluation.md`).
6. **Nothing in the objective forbids an impossible backbone, and nothing
   cheap will.** `clash` dropped the bonded residue pair whole, so the 15 atom
   pairs at or beyond `CLASH_MIN_SEPARATION` -- the ones phi and psi control --
   were unscored, and a single-residue (phi, psi) scan read flat across the
   whole plane. Narrowing the exemption is now available as `clash_bonded`,
   default off, and **it changes nothing measurable**: those pairs are four or
   more bonds apart precisely because they are far apart, and at Bondi
   heavy-atom radii they never reach contact -- a bonded pair on crystal
   ubiquitin reads exactly 0.0 under the switch at tolerance zero. The
   Ramachandran exclusions are carried largely by hydrogens this representation
   does not have. So a heavy-atom clash term cannot produce the plot at this
   resolution either way
   (`docs/claims/the-clash-term-does-not-see-the-ramachandran-plot.md`), and the
   phi/psi preference has to be **learned** -- which is `g22_random`'s second
   bucket, not a new term. No tabulated Ramachandran potential and no
   electrostatic term: both would supply the preference that bucket exists to
   detect.
7. **The `[environment]` block** in `snapshot()`. A fourth undiagnosed
   per-update rate: 3.2 s/update on mi300x against 0.85 on mi355x.

**Replaced 14-15 Nov, on Ulban's instruction:** the morphogen is now one
secreted chemical **diffusing** over the perception graph, with `diffuse`
taking `graph.neighbours` and `edge_mask` instead of `links`:

    field <- (1 - lambda) * (field + D * (neighbourhood mean - field))  soft-or  secretion
    DIFFUSION_RATE D = 0.5    DEGRADATION lambda = 0.10

Both chain-transported groups are gone; they converged to `decay ** index` and
were a positional encoding. `FIELD_NODE_EXTRA` 20 -> 4, one mechanism rather
than three, no sources and no alive mask required, 12,482 parameters. Verified
end to end on 1UBQ: 33 alive, 33 carrying, 0 outside alive, finite.

**The open choice is closed** -- Ulban chose real diffusion over the relay on 15
Nov, so the field has memory and a pulse travels: twelve hops out on a chain it
reads 3.6e-05 at step 40 with nothing secreting since step 0, where the relay
read exactly zero. Degradation is what keeps the steady state a gradient rather
than a saturated constant, and the chain reach is analytic --
`1/(1-lambda) = (1-D) + D(1/r + r)/2` gives `r = 0.519494` against 0.51949
measured.

**And `lambda` is set by the graph's diameter, not by taste**, which is the new
finding. A kNN graph over a compact protein is small-world: 1UBQ at `k=16` has
mean degree 15.5 and a **diameter of three hops**, so a reach longer than that
leaves a near-constant field. Hop-shell contrast `hop1/hop3`: **1.52x at
lambda=0.02**, **4.32x at 0.10**, **20.5x at 0.30** -- the first says nothing,
the third duplicates one message-passing hop, so 0.10. Unmeasured on anything
but ubiquitin, and a larger protein has a larger diameter.
(`docs/claims/the-emitted-chemical-diffuses-over-the-perception-graph.md`.)
`morphogen = false` in every arm, so nothing is waiting on it.

**And a residue can now see the gradient, not only the concentration (15
Nov).** Ulban asked whether nodes can see the morphogen gradient and follow it.
The circuit already existed -- a message carries the neighbour's concentration
(`model.py:480`) together with `local_direction`, the unit vector to it in the
residue's own frame, and `equivariant_move = true` reads the translation in
that same frame. Measured on a synthetic chemotaxis task on 1UBQ, where the
structure is identical in every example so the field is the only input that
says where the source is, held-out cosine on three unseen sources, two seeds:

    update   concentration only   + the gradient block
       100         +0.652              +0.947
       600         +0.709              +0.960
    parameters     241,814              245,270

So the message function can infer a gradient (0.71 against the 0.00 of a rule
that cannot read direction) and inferring it is much worse than being told.
`morphogen.field_gradient` now supplies `sum_j w_ij (c_j - c_i) u_ij` in each
residue's own frame -- Growing NCA's Sobel pair, on a graph -- for 3,456
parameters. `FIELD_NODE_EXTRA` 4 -> 10. Invariance checked on real geometry at
3.6e-08.

**The field now has four channels, not two (15 Nov).** Two was a *signed pair*
and so declared the field to be electrostatics; four declares nothing, which is
the Growing NCA position -- a signed pair still fits inside four, and so does a
hydrophobic signal, which has no sign and cannot share a pair. `EMIT_CHANNELS`
2 -> 4, `FIELD_NODE_EXTRA` 10 -> 20, parameters 245,270 -> 251,160, node_dim 76,
all 951 tests pass with the dimension guard unchanged because the tests assert
the relation rather than the number.

**And the field now has a job to do: a hanging terminus should be a source
(15 Nov, Ulban).** `growth.py` already listed loop closure as deferred -- *"two
segments that meet are not covalently closed ... left to the bond restraint"* --
and a bond restraint cannot say which way to walk to make a bond that does not
yet exist. A gradient can, which makes this **chemotactic closure** rather than
analytic (CCD/KIC). The chemotaxis probe was almost this experiment already: an
arbitrary source, 0.96 held-out cosine with the gradient block; make the source
a hanging end and it is the premise. It also fixes the initialisation problem
that gates the whole field, because a topologically placed source is
differential from step 0 while a learned one has to bootstrap out of a
saturated flat field. Recommendation is a **given** source, on the Growing NCA
precedent that a seed cell is handed its state. Four objections are written up,
of which two matter: a scalar source says "towards a terminus" and not
"towards *your* terminus" (two ends per run, ten to sixteen runs per crop), and
a source attracts every residue rather than only its partner, so the rule has
to learn to read the field only at a frontier.
(`docs/claims/a-hanging-terminus-should-emit-a-morphogen-so-the-chain-grows-towards-it.md`.)

**Argmap convention, corrected by Ulban 15 Nov: entries are premises, stated as
assertions.** Done. `the-backbone-should-be-grown-sequentially` was restated,
and the nine `question-*.md` nodes were renamed to premise form -- 34 files
rewritten, no broken wikilinks, finding count unchanged at 4. `type: question`
stays in the frontmatter, because that is what gates dependent claims at check
3; only the title and the slug changed.

**BUILT 15 Nov: the polarity source.** Ulban's correction is the design --
*"frame angles place themselves influenced by an N-C gradient"* -- so this is a
**local coordinate system a frame reads its angles against**, not an attractor
pulling residues at a cut end. `morphogen.terminus_emission` secretes 1.0 into
channel 0 at every residue whose N side has no covalent neighbour and channel 1
where the C side has none; `compose_emission` fixes the order, structural
first; `RolloutConfig.terminus_source` gates it, default **off** so arms stay
comparable, and the source is **re-injected every step** because it is a
boundary condition, not an initial one. `TERMINUS_CHANNELS = 2` is an addition
to `LEARNED_EMIT_CHANNELS = 4` rather than a tax on it, so `EMIT_CHANNELS = 6`,
`FIELD_NODE_EXTRA` 20 -> 30, node_dim 86, parameters 251,160 -> 256,920. On
1UBQ at 20 steps the polarity channels span 0.0003-1.0000 with the flag on and
are exactly zero with it off, while the learned channels sit at 0.907 --
saturated and flat, the initialisation problem visible in the same tensor.

**The field is a spatial coordinate, not a positional encoding, and the
earlier reading of this was wrong.** Ulban, 15 Nov: *"two gradients should be
anti-aligned, they're coming from different termini - exactly french flag
style."* Right, and it reframes everything. An opposing pair is not two
independent directions and was never going to fix an orientation; its **ratio**
is the readout, `r = log(c_nward) - log(c_cward)`, a signed position along the
axis joining the sources. So the test is whether `r` tracks the **body axis**
or the **residue index**, with `x` = CA projected on the first-to-last-CA unit
vector and `rho(x, index)` as the control:

    start state         rho(r, index)   rho(r, x)   rho(x, index)   end-to-end
    native                    -0.189      -0.972          +0.241       37.1 A
    12-residue window         -0.319      -0.925          +0.404       37.1 A
    24-residue window         +0.109      -0.970          -0.066       37.1 A
    helix baseline            -1.000      -1.000          +1.000      113.9 A
    random baseline           -0.934      -0.953          +0.871       49.4 A
    extended baseline         -1.000      -1.000          +1.000      273.2 A

**`rho(r, x)` is -0.92 to -1.00 in every row**, so the field faithfully
reports position along the body axis whatever the geometry -- which the retired
module did not, because it reported the index regardless of geometry. What
varies is the control: on a folded structure body axis and sequence axis are
different (0.241), on any torsion walk they are the same (1.000), and there the
field reads as an index because position *is* the index.

**So "don't use a helix baseline" is right as a direction and does not fix
it.** Random only moves the control 1.000 -> 0.871, because any walk with ideal
covalent geometry is extended (49 A against the native 37 A) and every extended
chain is degenerate. **The task is what fixes it**: regeneration keeps the
remainder at its crystal pose, so the control is 0.40 and -0.07 for a 12- and a
24-residue window. The problem is confined to whole-structure growth, already
held to be harder than and different from the goal.

**It also cannot turn yet.** `lab_rotation = true` everywhere means the
rotation is applied in the lab frame while perception is invariant, so a
perceivable reference axis does not help an output read in the wrong frame.
**`g16_localrot` is a prerequisite for this premise, not an alternative**, and
it has still never run.

Two earlier objections are withdrawn by the polarity framing: partner identity
is unnecessary because N-ward and C-ward are globally meaningful, and "a source
attracts everything" was backwards -- every residue reading one gradient is the
French flag model, which is the mechanism rather than a side effect.
(`docs/claims/a-hanging-terminus-should-emit-a-morphogen-so-the-chain-grows-towards-it.md`.)

**Why this matters beyond the morphogen.** `g10_equiv_ceil` ran regeneration
*with* the equivariance fix and its move was still not aimed: align **-0.096**
against a +0.206 null. The chemotaxis number says that was never an
expressivity limit -- given something that carries a bearing, this rule aims in
a hundred updates. A diffused morphogen gradient is such a quantity, which
makes the field a candidate answer to the standing "FAPE gives a magnitude and
not a bearing" problem rather than a side feature. **Hypothesis, not a
result:** no arm has trained with the field on.

**The gap that is now the clearest thing in the record.** `lab_rotation = true`
in `[base.inputs]` and in every arm ever run, so the **rotation is still
applied in the lab frame** while perception is invariant. The rule can express
"move up the gradient" and **cannot express "turn toward it"** -- position is
half of a frame and secondary structure is mostly the other half. `g16_localrot`
is the arm that fixes it and has never run; it was demoted by the rescope and
this promotes it.
(`docs/claims/a-residue-can-follow-the-morphogen-gradient.md`.)

**Demoted by the rescope**, and recorded as such rather than dropped:
ensemble-dispersion spread and the B-factor Spearman. A positive result there
is a **replication** of PLACER and RFD4's reproduction of it, not a finding, so
its value is as a validation check on the regrowth distribution — not as a
result. The fitted-Potts programme, CDR pre-organisation and the binder
hypothesis stay downstream of items 1–4 and are not worth a slot until those
answer. Also still demoted: driving `rmsd_dmg` lower on its own, and
`g16_localrot`.

### Environment

Bunya checkout `/scratch/user/uquadhik/AANCA` (a git clone; `git pull` to
update). `.venv_bunya`, torch **2.10.0+rocm7.0** over
`python/3.12.3-gcccore-13.3.0`. Job logs
`/scratch/user/uquadhik/AANCA/logs/<name>-<jobid>.{out,err}`; browser-bridge
diagnostics `/scratch/user/uquadhik/AANCA/.cc/`. Read
`binder-physics/docs/BUNYA_BROWSER_SHELL.md` before any cluster work.

---

## Historical framing — §1-§11 below were written 2026-10-27

They are kept for the prior-art search and the pitfalls, both still useful.
**Two things in them are now wrong**, and §0 above supersedes them: the project
is backbone-and-sequence cofolding, not side-chain repacking, and the objective
is a realistic basin rather than a fixed point.

---

## 1. What is being built, and why it is novel

A per-residue rule, applied repeatedly to a protein structure, that moves side chains into their
correct packing. Each residue sees only its local neighbourhood, outputs a change to its own
side-chain torsions, and the same rule runs again on the result. No global information pathway, no
step index as input, no noise-level conditioning.

**The novelty is not the architecture, it is the training objective.** Existing side-chain packers
(AttnPacker, DiffPack, FlowPacker) predict torsions in a fixed number of passes or a fixed diffusion
schedule. This trains for a **fixed point**: the rule must converge to the native packing and stay
there, from many different starting conditions, over a step count it was never told. That is the
Growing-NCA objective (regeneration from damage, stability past the trained horizon) applied to
protein geometry, and a prior-art search on 2026-10-27 found nobody has published it.

Nearest neighbours in the literature, and what they are missing:

- **Graph NCA** (arXiv 2110.14237) and **E(n)-equivariant Graph NCA** (arXiv 2301.10497, TMLR 2024) —
  the NCA machinery on arbitrary graphs, but not applied to proteins. Note some of their rules
  oscillate rather than converge; that failure mode is expected here too and is one of the things
  worth measuring.
- **GNNRefine, ATOMRefine** — all-atom protein refinement with graph networks, but trained as a
  one-shot or few-shot refinement, not for regeneration under damage.
- **AttnPacker / DiffPack / FlowPacker** — chi prediction from a residue graph. DiffPack receives
  the noise level as conditioning; this rule deliberately does not, which makes its denoising blind.
- **PottsMPNN** — the head-swap trick being borrowed: keep a ProteinMPNN-style encoder, replace the
  output head. Here the head emits torsion deltas instead of sequence logits.
- **ProteinEBM** — the "learned forcefield" goal met a different way, as an energy whose gradient is
  a diffusion score. Deliberately *not* equivariant; uses augmentation instead.

Full cited write-up with everything that was searched and ruled out:
**https://claude.ai/artifact/SBXsPoHiExPDNNYMki2Rka**

---

## 2. The model

- **Fixed native backbone.** Only side chains move. Backbone remodelling is a later escalation (§8).
- **Per-residue output: delta-chi for chi1..chi4.** Applied to the residue's current torsions, not
  predicted absolutely. Residues with fewer chis mask the unused outputs.
- **Encoder:** ProteinMPNN-style message passing. Invariant edge features, k-NN graph built from a
  10 Å heavy-atom cutoff plus sequence neighbours. Weights shared across steps.
- **Residue identity is an input, not an output.** One-hot node feature. This is a repacker; the
  sequence is known.
- **No pretrained weights — train from scratch.** Decided 2026-10-27, against the earlier plan.
  Reasoning: ProteinMPNN's encoder is sequence-blind and sees only backbone geometry (N, CA, C, O
  and a virtual CB), so it has never seen a side-chain atom — and side-chain coordinates are the
  entire mutable input to a repacker. Adding side-chain edge features puts the pretrained weights
  in front of an input distribution they were never trained on, so most of it retrains anyway.
  PottsMPNN's head swap worked because its new head read the *same* backbone-only input as the
  original decoder; ours does not. And unlike sequence design, repacking has effectively unlimited
  supervision — every PDB structure is its own label and the perturbations are synthetic — so
  pretraining buys less. AttnPacker and DiffPack both train from scratch. Keep the architecture,
  drop the weights. A loadable-encoder flag was offered as an option and is not yet decided.
- **Strict locality is a design goal, not a compromise.** Ulban's stated reason: that is how atoms
  see each other. Do not "fix" convergence problems by adding global attention.

### Training recipe (from the agreed plan)

- Rollouts of 20–50 steps, with the step count randomised per sample.
- Stochastic update mask (only a random subset of residues update each step) — the NCA asynchrony trick.
- Sample pool, as in Growing NCA: persist intermediate states back into the pool so the rule has to
  stay stable rather than just arrive once.
- Loss taken at a **random late step**, not the last one. This is what forces a fixed point rather
  than a schedule.
- Plus a **no-op term**: a native, undamaged input must produce zero update.

---

## 3. The two damage operators — keep them separate

**Geometric damage.** Perturb side-chain positions: randomise rotamers, or add noise to torsions.
Ground truth is the native packing, exactly known. The task is blind denoising. This is the operator
that produces the headline figure.

**Chemical damage.** Mutate a residue and repack its neighbourhood. There is no ground-truth
structure, so it cannot be trained with a reconstruction loss in the same way, and it is evaluated
against **measured ddG** instead.

They are different objectives with different supervision. Do not merge them into one loss term
without a deliberate decision.

---

## 4. Evaluation — three tiers

**Tier 1, repacking accuracy.** Chi MAE and side-chain RMSD on CASP13 and CASP14 targets.
Baselines: DiffPack, AttnPacker, FlowPacker, Rosetta. This is the table that shows the method is
not worse than existing packers.

**Tier 2, fixed-point behaviour — unpublished for proteins, and the actual contribution.**
- **Convergence:** does mean |delta-chi| decay with step count?
- **Stability:** run 500 steps on a rule trained to 50. Does it hold, drift, or oscillate?
- **Basin of attraction:** recovery accuracy as a function of perturbation magnitude.
  **This is the headline figure.** Nothing in the packing literature reports it.

**Tier 3, mutation response.** SKEMPI 2.0, ~7k measured ddG values. Candidate readouts:
- *Strain* — how far and how long the neighbourhood keeps moving after a mutation.
- *Conformational entropy*, if the head is made stochastic. This is RDE's signal.
- *ProteinEBM energy difference* on the converged structure.
Report per-complex Pearson and Spearman.
Baselines: RDE / RDE-Network (Luo et al. ICLR 2023, github.com/luost26/RDE-PPI), DiffAffinity
(NeurIPS 2023), EBM-DDG (KDD 2025, arXiv 2508.10629). Floors: FoldX, Flex ddG.

### The SKEMPI leakage trap — do not skip this

Bushuiev et al. 2024 (arXiv 2404.10457) measured **~56% test-interface leakage** in the PDB-code
split that the RDE paper used, and ~30% even at 30% sequence identity. Splitting by PDB code or by
sequence identity **does not work** on SKEMPI, because the same interface recurs across entries.

Split by **interface similarity** (iDist, threshold 0.04) or by **SKEMPI's own expert categories**
(0% measured leakage). Report both the clean number and the leaky one, so the comparison to published
baselines is honest in both directions.

---

## 5. Repository and code state

**Remote:** `https://github.com/UpsilonAlpha/AANCA.git`, owner UpsilonAlpha. Default branch `main`.
**As of 2026-10-27 13:33 UTC the remote is empty — zero commits, no branches.**

**Local clone on Ulban's laptop:** `C:\Users\uquadhik\Library\University\Year1\Structures\AANCA`
(next to `binder-physics`). Cloned with the laptop's own git credentials, which work. Git identity
there is `Ulban Adhikary <46319463+UpsilonAlpha@users.noreply.github.com>`, git 2.56.0.windows.1.

Two things to know about that laptop:
- ~~**There is no working Python on it.**~~ **Corrected 2026-10-28: it is a working run environment.**
  `python` does hit the Microsoft Store stub and `py` does not exist, but **`uv` is installed** at
  `C:\Users\uquadhik\.local\bin\uv.exe` and already had CPython 3.11.16 and 3.12.14 toolchains.
  `uv venv --python 3.11 .venv` then `uv pip install numpy biotite pytest` takes under a minute, and
  the whole Milestone 1 geometry layer was written and verified locally on it. Do not plan around
  the laptop being git-only.
- The clone sits **inside the `Library` git repo**, so `Library` will list `Structures/AANCA/` as
  untracked. Gitignore it there or make it a submodule.

**Superseded 2026-10-28.** The repo is no longer greenfield: commit `3c156eb` on `main` carries the
Milestone 1 geometry and data layer, passing its exactness bar (round trip accurate to 3.6e-14 A
against a 0.01 A target, on 193L / 3NIR / 1UBQ). See `README.md` for the result table, the float32
measurement trap, and what is still a stub. The earlier skeleton at `/mnt/project-files/aanca/` is
obsolete.

**Why nothing is pushed:** this Claude account has no GitHub account linked, so a cloud session
cannot clone or push. Either Ulban links it at https://claude.ai/connect-github (and installs the
Claude GitHub App on the AANCA repo specifically), or pushes go through the laptop's credentials.
**If you can get the GitHub link done, do — it removes a lot of friction.**

### Agreed stack

Python 3.11+, PyTorch, **biotite** for structure parsing. biotite 2.0.0 and numpy 2.5.3 install
cleanly in a cloud container (`pip install biotite numpy`, ~1–2 minutes; it exceeds a 60-second
timeout, so give it longer). `gemmi` is **not** available on Bunya — see §6.

---

## 6. Compute: Bunya

**Read `binder-physics/docs/BUNYA_BROWSER_SHELL.md` before touching the cluster.** 386 lines,
on Ulban's laptop at
`C:\Users\uquadhik\Library\University\Year1\Structures\binder-physics\docs\BUNYA_BROWSER_SHELL.md`.
It is authoritative and supersedes Part E of `docs/TOOL_USAGE.md`. Every rule in it was learned from
a failure, most of them twice. The essentials, so you know what you are walking into:

- **Access is a code-server pane** inside an interactive SLURM job, behind UQ's Open OnDemand proxy
  at `https://onbunya.rcc.uq.edu.au/rnode/<node>/<port>/`. **This is why a browser-capable session is
  needed at all.** Node and port change every session and the session can move mid-task — read them
  from the live tab every time, never from notes. Never sign in on Ulban's behalf.
- **Work lives in `/scratch/user/uquadhik/`**, not `~`. The home directory has a near-empty
  lookalike that wastes time. `~/Scratch/uquadhik` is a symlink, so `find` without `-L` silently
  finds nothing; prefer the real path.
- **Never scrape the terminal canvas.** Write output to a file, fetch it through
  `vscode-remote-resource` with `credentials:'include'`. Diagnostics go to
  `/scratch/user/uquadhik/<project>/.cc/`, never `/tmp` (node-local, dies with the session).
- **Typing is fragile:** keep each command under ~100 characters, send Return as its own key action,
  never a heredoc, and `cat` a generated file back before running it. A line that fails to submit
  silently swallows the next one.
- **GPU:** `-p gpu_rocm --qos=gpu --gres=gpu:mi355x:1 --account=a_imb_ccbcd`. **Max 4 concurrent GPU
  jobs**; a 5th sitting at `QOSMaxJobsPerUserLimit` is not an error. Ulban's preference is the
  mi355x. CPU stages belong on `-p general`, which is outside the GPU QOS and so runs concurrently.
- **Quota binds on inodes, not bytes** (`rquota`; `rcc-user-quota` does not exist). Count with
  `find <dir> | wc -l`, never `du`. **Check before any run that writes many small files** — an NCA
  checkpoint or rollout sweep is exactly that. Quota failure looks identical to mangled typing: the
  redirect fails and the log is never created. The tell is that a quota-failed redirect leaves an
  existing file *untouched* rather than truncating it.
- **A job's SLURM state is not a health signal.** Quota-killed jobs reported `COMPLETED 0:0`.
  Always read the `.err`, and make scripts end `RC=$?; echo RC=$RC; exit $RC`.
- **mmCIF:** three incompatible layouts in this project's own tooling (BindCraft, Boltz/BoltzGen,
  biotite). Parse the `_atom_site` header dynamically with an `auth_`→`label_` fallback, never index
  columns by position. Identify chains by length or sequence, never by letter.
- **git auth dies with the terminal, not the session.** `Repository not found` mid-session means the
  per-terminal credential socket went stale. **Open a new terminal.** Do not rewrite the remote to
  SSH or embed a token.
- **Exactly one agent drives the browser pane at a time.** `document.activeElement` is global to the
  page, so separate terminals do *not* isolate concurrent agents — a measured collision on 16 Oct
  merged two agents' commands into one file. Agents that only write code and push need no pane and
  can run in parallel; prefer that split.
- **A safety classifier intermittently refuses typed commands**, `browser_batch` more than single
  calls, and `sbatch`/`squeue` more than plain reads. Retry once; **do not rephrase to get around a
  refusal** — tell Ulban and hand over the exact command.

---

## 7. Data — none of it is local yet

Ulban has no local copies and asked for it to be pulled.

- **Training structures:** a PDB subset from RCSB. Needs a resolution and redundancy filter; the
  packing papers' usual protocol is a reasonable starting point.
- **Test sets:** CASP13 and CASP14 targets, to compare against published packer numbers.
- **SKEMPI 2.0** (~7k ddG), from its own site. Read §4's leakage warning before building any split.
- Known-good packer output for the geometry validation in §9 — Rosetta or an existing packer's
  repacked structures, to check the chi extraction and rebuild against.

---

## 8. Milestone plan

**Milestone 1 (agreed, in progress, nothing finished): the geometry and data layer, verified.**
Parse PDB, extract chi1–4 for all 20 residue types, rebuild side-chain coordinates from torsions on
the fixed backbone, and **prove the round-trip is exact** (< 0.01 Å) against a known-good packer.
Then the two damage operators, then the local graph builder. No model training until this passes.

The reason for the order: the prior-art write-up flags the **torsion-space lever arm** as the main
risk — dihedral errors compound along the chain, which is why RGN2 lost to frame-based methods and
why AF2 uses FAPE in local frames. An NCA makes it worse by applying the rule many times. A quiet
sign error or atom-ordering bug in the chi code would show up later as a model that will not
converge, and would be very expensive to find there.

Implementation note already settled: implement `set_chi` as a **rotation of the downstream atom set
about the chi bond axis**, applied to the residue's existing coordinates — not a rebuild from ideal
internal coordinates. That makes the round-trip exact by construction and preserves the crystal
bond lengths and angles. Two things to handle explicitly:
- **Proline is a ring.** Rotating chi1 breaks the CD–N closure. Treat PRO side chains as fixed, or
  use discrete ring-pucker states; do not let it into the free-torsion path silently.
- **Two-fold symmetric terminal chis** — ASP chi2, GLU chi3, PHE chi2, TYR chi2 — are equivalent
  under a 180° flip. Encode this and respect it in both the loss and the reported MAE, or the
  numbers will be wrong and not obviously so.

**Later escalations, in order:** loop remodelling (unfix 8–12 residues, let phi/psi move), then the
whole backbone with frame outputs rather than torsions.

**ProteinEBM add-on, three tiers if it is wanted:** (a) score the converged structure with
ProteinEBM-x, the low-noise variant; (b) use an energy difference as a ddG readout — but its
published protocol measures folding stability against a Ramachandran-random unfolded ensemble, so
take the bound/unbound decomposition from EBM-DDG instead; (c) an auxiliary energy gradient on the
last few rollout steps with a small lambda, checking it does not degrade repacking.

---

## 9. Read these next

1. **The prior-art write-up:** https://claude.ai/artifact/SBXsPoHiExPDNNYMki2Rka — the full cited
   search, including what was ruled out and the five pitfalls.
2. **`binder-physics/docs/BUNYA_BROWSER_SHELL.md`** on Ulban's laptop — before any cluster work.

Project memory in this project also holds `protein-nca-prior-art` and
`residue-nca-packing-experiment`, which are the condensed versions of §1–§4.

---

## 10. Open questions for Ulban

1. **GitHub link** — will he connect the Claude GitHub App to AANCA, or do all pushes go through the
   laptop's credentials?
2. **Loadable-encoder flag** — keep the option to load pretrained ProteinMPNN weights, or commit to
   from-scratch and drop the code path?
3. **Training-set protocol** — follow DiffPack's PDB filtering so the numbers are directly
   comparable, or pick our own?
4. **Tier-3 readout** — strain, stochastic-head entropy, or ProteinEBM energy? Affects whether the
   delta-chi head needs to be stochastic, which is a model decision, not an evaluation one.
5. **`/scratch/project_mnt/S0253`** — the group fileset has far more inode headroom than the personal
   scratch quota. It is shared, so using it needs his say-so.

---

## 11. Pitfalls from the prior-art search — worth re-reading before each stage

- **Torsion-space lever arm.** See §8. The main technical risk in the whole project.
- **The contact graph changes as coordinates move.** Freeze it and the rule reads the answer through
  the topology; rebuild it every step and the computation cost rises and training may destabilise.
  This is an open design decision, not a solved one.
- **Equivariance must be chosen deliberately.** ProteinEBM chose *against* it and used augmentation.
  Invariant edge features sidestep much of it. Do not inherit the choice by accident.
- **"Damage" in a covalent chain means perturbation, not ablation.** You cannot delete a residue the
  way a Growing-NCA deletes a voxel.
- **Local-only rules cannot fix global topology.** If the backbone is wrong, no amount of local
  repacking rescues it. This is fine for a fixed-backbone repacker and becomes a real constraint at
  the backbone-remodelling escalation.
- **Some Graph NCA rules oscillate instead of converging.** Expect it, measure it (§4 tier 2), and
  resist the urge to fix it with a global pathway.

---

## 12. Live on Bunya — added 2026-10-29

Checkout: `/scratch/user/uquadhik/AANCA` (a git clone; `git pull` to update).
Environment: `.venv_bunya`, torch **2.10.0+rocm7.0** over `python/3.12.3-gcccore-13.3.0`.
Diagnostics written for the browser bridge: `/scratch/user/uquadhik/.cc_aanca/`.
Job logs: `/scratch/user/uquadhik/AANCA/logs/<name>-<jobid>.{out,err}`.

