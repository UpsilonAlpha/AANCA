# AANCA — handoff brief

AANCA = amino-acid neural cellular automaton. Read this file first. The dated
journal is `docs/JOURNAL.md`; the claims and their evidence are one note each in
`docs/claims/`, validated by `docs/argmap/argmap.py`.

---

## 0. Current state — 2026-11-22

### The question, rescoped 14 Nov

**Can a very simple shared local rule — one function, 229,588 parameters,
applied per residue in parallel with no global coordinator, no pooling and no
attention — regrow a damaged region of a protein from the context that
survived?**

That is the whole central argument. Not a design method, not a competitor to
diffusion: a study of whether a local update rule can learn enough real physics
from structure data to put destroyed atoms back where they belong.

### IS THE FAILURE INHERENT TO THE ARCHITECTURE, OR IS IT THIS APPROACH? (23 Nov)

The central question a reader will ask, and the record answers it more clearly
than any single arm does. **On present evidence it is the approach, and the
reason is that the objective supplies a magnitude and never a bearing.**

  * **The architecture can aim.** Handed a diffused concentration gradient that
    says which way to go, the same rule reaches **cosine 0.96 in a hundred
    updates** ([[a-residue-can-follow-the-morphogen-gradient]]). Directed motion
    is representable and learnable by this architecture, measured.
  * **Nothing ever tells it which way.** `g10_equiv_ceil` **teacher-forced the
    exact ideal move at every step for 2,000 updates**, with **both halves of
    the move local**, and the emitted bearing was **-0.096 against a +0.206
    null** -- below an untrained head
    ([[the-equivariance-fix-did-not-release-a-bearing]]). The fix made aiming
    representable and nothing started aiming. Clamped FAPE is *"a magnitude and
    not a bearing"* ([[fape-can-teach-the-frame-head-which-way-to-move]]).
    **Read the caveat in experiment 1 below before citing this**: that align was
    read off a checkpoint whose twist had already blown up to 50 degrees, so it
    is not yet a clean capacity measurement.
  * **Today's mode collapse is what that predicts.** An objective that supplies
    a magnitude, with no term rewarding a torsion
    ([[the-clash-term-does-not-see-the-ramachandran-plot]]), gets a rule that
    converges on one average local conformation and sprays it regardless of
    context. Which is exactly what the cross-tabulation measured.

**What is NOT settled, and the two experiments that would settle it.** Neither
has been run, both are cheap, both are one variable from an existing arm:

  1. **A ceiling arm with a working actuator AND identity off** -- `g28_ceil_local`.
    **CORRECTED 23 Nov: an earlier version of this section said no ceiling arm
    had ever run with both actuator fixes. That was wrong.**
    [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]
    tabulates `g10_equiv_ceil` as *"local translation, local rotation"* -- both
    halves local, a sound actuator, teacher-forced for 2,000 updates.

    What confounds `g10` is not its actuator but **when its bearing was read**.
    Its twist runs 9.9 (update 250) -> 52.3 (1000) -> 50.2 (2000), and the
    align -0.096 was taken off the **final** checkpoint, i.e. off a rule whose
    geometry had already broken by a factor of five. The break belongs to the
    **identity head leaving its floor at update 1000** -- `g8_ceiling` has
    neither equivariance fix and breaks in the same row, while `g9_ceil_noid` at
    `identity_weight = 0` holds twist 9.1-9.2 for the full 2,000.

    So the untested cell is **both halves local AND identity off AND teacher
    forced**. `g10` had the first without the second; `g9_ceil_noid` had the
    second without the first. Nothing has had both. The current `base` already
    carries `move_weight = 1.0` and `identity_weight = 0.0`, so this arm is
    **one variable from base: `lab_rotation = false`.** ~70 min of GPU.

    If the rule can be taught to aim when handed the exact answer with a working
    actuator and nothing breaking it, every rung-1 failure in the record is about
    the objective. If it still cannot, the limitation is architectural --
    invariant perception plus a local move head cannot be taught to aim even when
    given the answer -- and that is a negative result about this class of model
    rather than about this loss.
  2. **The morphogen field trained on regeneration.** The gradient is a
    candidate bearing -- a vector, in the right frame, non-local in a way one
    hop is not. Nothing has trained it on this task, and the chemotaxis task
    that measured cosine 0.96 handed the rule a source whose position *was* the
    answer, which regeneration does not. Explicitly flagged in that node as a
    hypothesis.

**The one question that IS about the class rather than this instance.** An NCA's
power in its original setting comes from a grid that supplies position for free.
A backbone has no grid; the surviving context is the grid
([[in-regeneration-the-surviving-context-is-the-grid]]), and a destroyed
residue's correct position is a *global* geometric constraint that no single hop
reads. So the bearing cannot be perceived, it has to be **constructed** --
by diffusion, by more hops, or by iterative registration. **Whether a local rule
can construct it is the real question about AANCAs**, and it is unanswered.
"Can it regrow a helix" was never the hard part.

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

### BOTH KNOWN DEFECTS ARE NOW FIXED AND THE BAR STILL FAILS BY 2x (22 Nov)

Every number below is on **the same 12 held-out crops and the same damage
windows**, confirmed rather than assumed: `train_growth.score` seeds its window
generator `manual_seed(seed + index)` at `seed = 0` and `geometric_floor.py`
does the same, and the floor read its crop indices straight out of each
checkpoint's `eval_indices`.

    settled, matched crops            fit_all  fit_intact   twist   margin vs floor
    untrained start                     14.76       16.28      --        --
    GEOMETRIC FLOOR (no network)         9.21       10.05      --        --
    g22_random   context free            8.41        9.02   12.8d      1.03
    g24_frozen   context pinned          7.98        8.49   10.3d      1.56
    g25_equivariant  + rotation fix      7.50        8.10    9.2d      1.95
    g26_fullbptt     + truncate 120      9.46       10.17   12.0d     -0.12
    bucket 0 PASS                          --        4.00      --        --

**The rotation fix helped a little.** `lab_rotation = false` removes the 45 % of
the emitted rotation an invariant encoder could not represent. Placement 8.49 ->
8.10 A and **twist 9.2 deg, the lowest in the record** -- the two moving together
is the mechanism, since the defect was in the rotational actuator. But 0.39 A
lands in this arm's own pre-registered 6.5-8.49 bucket: *"report direction only,
and SAY SO rather than quoting the margin."* Honoured.

**The gradient-reach fix made it worse, and refuted its own prediction.**
`truncate = 12 -> 120` fits on the MI355X (3.01 s/update against 2.27) and cost
**1.68 A**: 8.49 -> 10.17 A, which is *above* the 10.05 A floor, so that arm has
**no signal at all**. The note that named truncation *the binding constraint*
called itself *"pre-positioning... it says which rollout hyperparameter to test
first"*; it was tested first and it hurt. Its arithmetic stands, its inference
does not -- a constraint that binds is not necessarily one whose removal helps.
Narrowed in its own body; no `refutes` edge, because the refutation bites an
inference and not the titled measurement.

**`g27_both` IS REGISTERED AND WILL NOT BE RUN.** g26's `>= 8.49` bucket reads
*"the defects were not what was holding it back... should end the arm chain
rather than extend it"*, and bucket 0's clause is *"if it fails, we do not add
terms and re-run."* g27 is the rotation fix plus the truncation change, and the
truncation change is now measured as harmful.

**So: four measured faults have been fixed** -- the translation half of the move
(7 Nov), the context drift (g24), the rotation half (g25), and the crop-selection
and superposition faults in the instruments -- **and the best configuration
places a damaged 12-residue window at 8.10 A where correct is 1-2 A, doing
nothing is 16.28 A and pure geometry is 10.05 A. A local per-residue rule with
this objective does not place a backbone, and the remaining gap is not explained
by any known defect.**
(`docs/claims/fixing-the-actuator-helped-a-little-and-fixing-the-gradient-reach-made-it-worse-than-geometry.md`.)

### Bucket 2, made 23 Nov: helix learned, sheet lost to plain geometry

`scripts/ramachandran_basins.py` on the three readable arms (33291294 g25,
33291295 g24, 33291297 g22), exit 0 each, on the twelve held-out crops each
checkpoint records in its own `eval_indices`, `eval_pdb_ids` verified. Four rows
over the same residues and the same boxes; **read `model` against `floor`**, the
network-free bond-closure-and-sterics optimiser, never against `start`.

    HELIX basin, natively-helical residues    -10d      +0d     +10d
    native                                   1.000    1.000    1.000
    start                                    0.036    0.034    0.065
    GEOMETRIC FLOOR                          0.055    0.068    0.065
    g22_random                               0.073    0.102    0.194
    g24_frozen                               0.018    0.034    0.097   <- CROSSES
    g25_equivariant                          0.218    0.237    0.306

    SHEET basin, natively-sheet residues      -10d      +0d     +10d
    GEOMETRIC FLOOR                          0.085    0.143    0.186
    g22_random                               0.021    0.125    0.136
    g24_frozen                               0.021    0.036    0.051
    g25_equivariant                          0.000    0.089    0.085

Denominators 55-62 helical, 47-59 sheet. The boxes are rectangles where the real
basins are contours, so **no absolute fraction here is "the share of residues in
the helical basin"** -- it belongs to these rectangles. What licenses the reading
is that all four rows share the same boxes, so box error largely cancels in the
difference.

**Helix: a real learned signal, in the best arm only.** `g25_equivariant` sits
3.4-4.7x above the geometric floor at all three widenings. That is the first
direct evidence the rule learned *local conformation* rather than only pushing
atoms toward a centroid -- a placement RMSD cannot show it, since 8.10 A is
compatible with either. `g22` is above the floor but weakly; `g24` crosses it and
nothing is quoted from it, so **the three-arm ordering does not track placement**.

**Sheet: at or below the floor in every arm, at every widening.** Plain geometry
puts *more* residues in the sheet basin than any trained rule does. The rule is
not merely failing to learn sheet -- it displaces residues out of a basin bond
closure and sterics alone would have found.

**The split is measured. The first mechanism offered for it was WRONG and is
withdrawn** -- it said a sheet is non-local because its partner strand is far
away *in sequence*. Sequence distance is not what a residue perceives:
`graph.py:327` selects neighbours by distance over all heavy atoms, and
`rollout.py:497` rebuilds the graph live every step from current coordinates.
Strands sit ~4.8 A apart, so a native sheet partner is already inside the k = 32
neighbourhood. **One hop reaches it, and "more hops" does not follow from this
measurement.**

Three candidates survive, and the first two are separated by one cheap
cross-tabulation of native basin against achieved basin (the torsions are
already computed):

  1. **One mode, over-applied** -- do natively-sheet residues land in the
     *helix* box? Then the rule learned a single local conformation and applies
     it everywhere.
  2. **The floor is trivially good at beta** -- does the floor also dump
     natively-*helical* residues into the sheet box? The beta region is the
     open, extended part of the plot, so a bond-closure-and-sterics optimiser
     tends there by default, and "below the floor on sheet" would say more
     about the floor than the rule.
  3. **Registration, not reach** -- a helix is built from residues inside the
     window, a sheet must be in register with a partner at a specific offset,
     which an 8.10 A placement error destroys even with the partner in view.
     Untouched by the correction, but untestable apart from the placement
     failure itself.

**The cross-tabulation settled 1 and 2** (33303677, 33303679, 33303681, exit 0).
Where natively-SHEET residues went when they missed the sheet box:

    natively-SHEET -> HELIX box       -10d     +0d    +10d
    start                            0.021   0.036   0.068
    GEOMETRIC FLOOR                  0.021   0.054   0.085
    g24_frozen                       0.043   0.036   0.068
    g22_random                       0.106   0.107   0.119
    g25_equivariant                  0.149   0.125   0.169

**Candidate 1 SUPPORTED.** The rule sends natively-sheet residues to the helix
box at 2-7x the floor's rate. Conditional on landing in any favoured region,
g25's helix share is 88/47/52 % against the floor's 17/25/30 %. And the
over-application ordering (g24 none < g22 < g25) is the SAME ordering as
helical success (g24 crosses < g22 0.102 < g25 0.237) -- two quantities, same
order, the same form of corroboration the rotation-fix arm rested on, and the
same weight: three points, one seed each.

**So the rule learned ONE local conformation and applies it regardless of what
the native conformation is.** That is the answer to "why does it not place a
backbone correctly" that no placement number could give.

**Candidate 2 weakened, not dead.** The refusal did not fire, so the floor
converged. It does discriminate (natively-sheet to sheet box 0.085/0.143/0.186
vs natively-helical 0.036/0.119/0.129, sign stable), so its sheet number is not
purely extended-default -- **but the margin narrows to 1.2-1.4x as the boxes
widen, and at both wider settings the floor puts natively-HELICAL residues in
the SHEET box more often than in the helix box.** Quote "every arm is below the
floor on sheet" with that attached.

**DISALLOWED dominates every row** -- model 0.68-0.83, floor 0.71-0.87, the
model better by almost nothing. **Most window residues in every row end at a
phi/psi in no favoured region at all**, and the helix signal lives inside the
minority that lands anywhere real.
(`docs/claims/the-rule-learned-helical-torsions-and-learned-nothing-about-sheet.md`.)

### How the bar came to be set, and the first failure (20 Nov)

The first arm in this project measured against **an absolute threshold fixed
before the run** and against **a measured floor**, from a neutral start with no
secondary structure handed to it. `g22_random` (33249861), 10,000 updates:

    on the same 12 held-out crops            fit_all   fit_intact
    g22_random start (untrained, update 250)   12.38        13.44
    GEOMETRIC FLOOR, random, mean               8.60         9.48
    g22_random SETTLED                          8.41         9.02
    geometric floor, best of 10 restarts        5.68         6.00
    bucket 0 PASS line                             --         4.00

**NO FLOOR-VERSUS-ARM MARGIN IN THIS RECORD IS CURRENTLY QUOTABLE** -- not the
0.46 A below, not `g24_frozen`'s apparent 0.99, not the 1.56 that 8.49 against
the corrected 10.05 would suggest. The floors were re-measured on 21 Nov after
a crop-selection fix (confirmed: the helix start now reads **12.23 A**, the
recorded baseline, to the digit):

    corrected floors              start fit_all/int    floor fit_all/int   best_int
    helix                             12.23 / 13.26       6.20 /  6.73       6.73
    random                            14.76 / 16.28       9.21 / 10.05       5.91

But the floor still picks its own crops, so the random floor's start is
**16.28 fit_intact against the arm's 13.44** -- harder windows, which inflates
the floor and flatters the rule. Before the crop fix the two agreed by accident
(13.09 vs 13.44), which hid it. `geometric_floor.py` is being given the
checkpoint-driven crop selection `context_drift.py` already has.

**And the number that makes this urgent**: the helix floor is **6.20 A fit_all**
while `g20_orient` -- the arm behind this project's headline, on a helix start --
settled at **7.78 A fit_all**. If that holds on matched crops, a network-free
optimiser beats the headline and the 7.78 was at or below the floor: no signal
rather than a small one. Different crops, so a hint, not a finding -- and a CPU
job to settle. The floor (`scripts/geometric_floor.py`) optimises the
window's frames against those two terms only -- no distance to native, no
torsion prior, no fragment library -- using the rule's own degrees of freedom,
so the two search the same structures and differ only in that one of them is
told where native is. It is a valid comparator and that was checked three ways:
same Bunya pool and held-out ids, same start state (11.92/13.09 against the
arm's own 12.38/13.44), and the chain closed to `bond` 0.0204 against the
trained rule's 0.629.

**And the rule is not selecting among valid completions.** The best of ten
random restarts of that same optimiser reaches **6.00 A**, so completions 3 A
better than the rule's output exist, satisfy the same constraints, and are
findable by chance in ten draws. That statement does not depend on where the bar
was put, which makes it the sharper one.

Bucket 0 survived the twist guard because the clause was narrowed *before* this
run to void only the g20 comparisons: a regime marker cannot invalidate a
measurement that is not a comparison. Without that narrowing this arm would have
joined `g19_nomove`, `g21_pool` and `g23_localrot` -- **four consecutive arms
with a placement number voided on twist** -- and the question would still be
open after a fifth 400-minute run.

The registration's own clause applies: *if it fails, we do not add terms and
re-run.* A local rule with this information, this objective and this much
training does not place a backbone.
(`docs/claims/the-rule-beats-bond-closure-and-sterics-by-half-an-angstrom-and-fails-the-absolute-bar.md`.)

### WHY it fails: three mechanisms, confirmed 21 Nov without spending a GPU hour

**1. The trained rule displaces the surviving context by 2.34 A.** Measured
directly on `runs/g22_random/best.pt`, 12 held-out crops, 120 steps
(`scripts/context_drift.py`, jobs 33277250 free / 33277251 frozen, both exit 0):

    g22_random best.pt      context N/CA/C drift   worst atom   O (reconstructed)
    context FREE (as run)               2.3402 A      16.020 A            2.6545
    context FROZEN                      0.0000 A       0.000 A            0.0392

**So the surviving context does not survive**, and the rule is scored against a
frame it is itself destroying -- which is the sharpest possible problem for a
claim about regeneration FROM SURVIVING CONTEXT. It happens because
`growth_objective` called `rollout` with no `frozen` argument; that was
deliberate (whole-structure FAPE was meant to teach "leave the context alone"),
and the measurement says the enforcement fails. `freeze_context` now exists and
drives the drift to exactly 0.0000, so **`[g24_frozen]`'s gate is open and its
second bucket is already satisfied**.

**This claim was inferred badly, retracted, and then confirmed, in one day.**
The morning's 1.20/2.83 A came from decomposing `rmsd` against `rmsd_dmg`, which
is an invalid instrument -- it reads 0.96 A on the frozen run where the true
drift is exactly zero. It was voided at midday on an **untrained** control that
read 0.0000, and that retraction was the second error: an untrained rule's move
heads emit ~0, so it cannot drift and free-versus-frozen is vacuously identical.
**A null that cannot exhibit the effect is not a null.** The direct measurement
on the trained checkpoint is what settled it.
(`docs/claims/the-rule-displaces-the-surviving-context-it-is-supposed-to-read.md`.)

**2. The gradient reaches about a quarter of the journey.** `truncate = 12`
against `steps = 120` in `[base.inputs]`, **unoverridden anywhere in g22's
lineage**. At the 0.3 A cap that is 3.6 A of visible travel against a 13.3 A
start displacement -- 27 %, and
[[the-gradient-window-is-a-third-of-the-journey-the-window-must-travel]]
measured this model-free and named it the binding constraint while the per-step
cap has 3.6x headroom.

**3. Half the move operator is a direction the rule cannot choose.** g22 ran
*"TRANSLATION ONLY: the rotation is left in the lab frame"*, and a lab-frame
rotation is unrepresentable by an invariant encoder: 0.68 A of error on a 1.50 A
move, **45 %**, larger than the translation half that was already fixed.

**Synthesis.** The rule learned the terms with dense local in-window gradients
and not placement: `bond` 4.278 -> 0.629, `clash` 0.0148, while placement landed
0.46 A past what bond closure and sterics alone force. Those two terms are
local, in-window and scored every step; placement has to be measured against a
frame the rule is moving (1), survive truncation (2), and be actuated through a
half-broken rotation (3). **So "this architecture cannot place a backbone" is
NOT established** -- what is established is that this *training setup* cannot,
and all three reasons now have measured magnitudes.
(`docs/claims/the-gradient-reaches-a-quarter-of-the-journey-and-half-the-move-operator-is-unsteerable.md`.)

**ORDER OF WORK.** The drift is measured and the gate is open, so
`[g24_frozen]` is the next arm: one variable against `g22_random`, same neutral
start, same absolute bar, and it changes a real 2.34 A. Then `truncate`, then
`lab_rotation = false` on a non-void arm. **The 4.0 A bar stays where it is and
the failed 9.02 A stands.** Note what g24 cannot settle: freezing could make
placement WORSE, since the rule currently relieves window strain by moving
anchors and would lose that -- which would mean the 9.02 A depended on being
allowed to corrupt the frame, and is a finding either way.

**And the methodological lesson, which cost two wrong claims in one day.** Both
errors were the same error: a superposition fitted on residues whose placement
was the question. It produced `rmsd_dmg` under-reporting placement by 1-2 A, and
then 1.7 A of context drift out of zero. Any number in this project that comes
from a fit over all residues should be assumed to be measuring the fit until
checked against a direct one.

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

**That rule did not persist, and 20 Nov found the protocol that does.**
`g20_orient` read at 600 steps is at 16.4–16.9 A, worse than the 12.23 A
untrained baseline ([[the-rule-has-no-attractor-past-its-training-horizon]]).
The same measurement on `g21_pool` — same six crops, same members, same
amplitudes, same `start bond` to two decimals — reads **8.5–9.1 A**, and the
step-120-to-600 change is **0.10 A** against g20's 9.15
([[the-pool-and-length-jitter-stop-the-divergence-past-the-training-horizon]]).
**So the divergence was a training-protocol fault, not an architectural one.**
Still 0 of 48 converged, so this is a bounded basin and not a fixed point, which
is the property this record wanted. Read it as **confounded**: g21's twist is
void, and the horizon-accuracy gate that was pre-registered to license this
comparison is void with it. Length jitter is the likelier of the two causes.

### The three things that decide the central claim

In this order, and nothing downstream matters until they are answered.

  1. **Is 7.78 A real?** No arm in this project has been replicated at a second
     seed, because `seed` also chooses the held-out set, so the one replicate
     attempted measured different proteins
     ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]). This needs
     `eval_ids` as a config key pinning PDB ids rather than integer indices, and
     one run. It is the cheapest thing in the queue and it gates everything.
  2. **Does the rule hold what it builds? ANSWERED YES on 20 Nov, by an arm
     whose own placement number is void.** `g21_pool`'s primary bucket came back
     at **8.5 A at 600 steps** against g20's 16.9 and a 12.23 baseline, so
     divergence is a protocol problem and not the rule having no way to
     represent "I have arrived"
     ([[the-pool-and-length-jitter-stop-the-divergence-past-the-training-horizon]]).
     What is now needed is an arm that holds its structure *and* can report what
     it built: the pool and the jitter with twist inside the band, which is
     question 1's replicate as well if it also pins `eval_ids`.
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
    stops because the schedule ends         has no schedule; with a pool and
                                            length jitter it holds 8.5 A to
                                            600 steps without converging --
                                            a bounded basin, 0/48 fixed points.
                                            Without them it diverges to 16.9

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
2. **`g21_pool` (32904055) is READ, its placement number is VOID, and the
   primary bucket still needs a job (20 Nov).** Completed, exit 0, 10,000
   updates, 444.6 min on bun144. Its void clause was *"if `twist` leaves the
   9-12 band"* and settled twist is **13.25** (14.2 then 12.3 on the last two
   held-out evaluations, out of band at 10 of 39, all from update 5500 on). So
   `settled` rmsd_dmg 8.41 A is void -- by exact precedent, since `g19_nomove`
   carried the same clause, settled at 14.26 and had its 8.39 voided while its
   twist reading stood.

   **The reason the void fired is the result.** g21 inherits
   `orientation_weight = 2.5` and adds only `pool_probability = 0.5` and
   `steps_min = 40`, and twist goes most of the way back to the no-term value:

       baseline    --                                 9.6      12.23
       g17_long    no orientation, 10k               14.3       6.69
       g19_nomove  no orientation, 10k              14.26  8.39 (void)
       g20_orient  orientation 2.5, 10k             11.24       7.78
       g21_pool    orientation 2.5 + pool + jitter  13.25  8.41 (void)

   g20's headline was that the term holds twist in the band, with g19 as the
   causal control. **The persistence protocol gives that back.** Guess at the
   mechanism, not more: the pool's guard keeps an end-state only if it beats
   that crop's fresh baseline and ranks by *accuracy*, so twist is not in the
   ranking and an entry can be better on placement and worse on orientation and
   still be kept. Two variables at once, so which of pool or jitter did it is
   unknown -- a recorded departure, not an accident.

   **And the primary bucket is now READ (job 33245103, exit 0, 20 Nov), and it
   is the first bucket -- the best of the four.** The 600-step
   `scripts/regrowth_dispersion.py` reading on `best.pt`, against g20's figures
   from the identical measurement:

       amp deg       g20 RMSF  g20 rmsd_dmg    g21 RMSF  g21 rmsd_dmg
             0          5.098         16.93       3.384          8.51
            15          5.139         16.35       4.085          8.72
            45          6.253         16.41       5.400          9.10

       untrained baseline                                       12.23
       g21 at step 120, the trained horizon (void)               8.41

   Pre-registered first bucket was `<= 9.0` at 600 steps: *"the rule holds its
   structure, the statistical-potential programme is unblocked."* It reads 8.51
   at the noise floor, and **the step-120-to-600 drift is 0.10 A against g20's
   9.15**. Like-for-like was checked and not assumed: same six crops, 8 members,
   same amplitudes, and `start bond` identical to two decimals in both runs.
   **0 of 48 converged in both**, so g21 is still moving at 600 steps -- it is
   moving *without going anywhere*, which is the bounded basin this record
   named as the physical property and distinguished from the identity map.

   **Third bucket does not fire**: settled spread ratios 0.66 / 0.79 / 0.86,
   nothing halved, so the pool did not buy persistence by collapsing the
   distribution. Two qualifications: the bucket asked for spread at step 120 and
   the script prints it at the cap, and g20's denominator *was divergence* by
   its own record, so this is a diverging spread against a bounded one and
   cannot be read as "multiplicity preserved".

   **Confounded, three ways, and the second is a knot rather than a caveat.**
   (i) The checkpoint's twist is void, as pre-registered, which makes this
   confounded and not void. (ii) **The second bucket was the gate on reading
   the first** -- *"more than an Angstrom and the price is too high to read the
   first bucket against g20 at all"* -- and g21's 8.41 is 0.63 A, inside
   tolerance, but void on the twist guard, so the gate cannot be read either
   way. The 8.5 A figure rests on the crops, members, amplitudes and start bond
   matching, not on a horizon-accuracy check, which does not exist. (iii) Of
   the two variables, **`steps_min = 40` is the likelier cause than the pool**:
   drawing the rollout length from [40, 120] removes the privileged step the
   loss is applied at, which is what horizon-dependence *is*. Stop writing this
   as "the pool".
   (`docs/claims/the-pool-and-length-jitter-stop-the-divergence-past-the-training-horizon.md`,
   `docs/claims/the-pool-and-length-jitter-undo-the-orientation-terms-hold-on-twist.md`.)
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

**It also cannot turn yet, and the rotation half is now MEASURED (15 Nov).**
Ulban asked for local rotation. It is one config key -- `lab_rotation = false`
-- and the code already implements both sides, so the work was to measure what
turning it on buys and to re-register the arm. Translation set to exactly zero
so only the emitted rotation moves an atom, then the structure turned 40 deg:

    form                               move     error    ratio
    lab frame (local=False)           1.5014   0.680 A    0.453
    translation local, rotation lab   1.5014   0.680 A    0.453
    both local                        1.5355   6.2e-08    0.000

**0.68 A of error on a 1.50 A move, 45 %, against the translation half's
27 %** -- so the half still switched on is the *bigger* one, which the record
did not know. The middle row is the current base and is identical to the
unfixed form: `equivariant_move` touches only the translation, and the two
halves are independent. Pinned as
`test_a_lab_frame_rotation_is_not_equivariant_either`.

## Why the objective is not an all-pairs distance loss (asked 23 Nov)

It partly is, and the part that is missing is missing on purpose.
`pairwise_distance_restraint` in the losses module is an **unclamped CA-CA
distance error**, written precisely because FAPE clamps at 10 A and `clamp_max`
is exactly flat above it, while the damaged window starts ~13.3 A out -- so
before a single update the placement signal is a constant. It is the published
automaton's all-pairs MSE *restricted to the pairs that carry placement
information*, and that restriction has three parts:

  * **Window-to-ANCHOR pairs are weighted.** One end sits at its crystal pose,
    so the distance to it constrains where the window goes.
  * **Window-to-window pairs are computed, printed every evaluation, and
    deliberately NOT weighted.** The reason is measured, not assumed: walk the
    window linearly from the baseline to its native pose on 1UBQ and the
    internal half **rises from 8.7 to 44.7 A^2 over the first four tenths**
    before falling, because a straight line between two poses passes through a
    contracted intermediate. Weighting it would charge the rule for taking the
    route to the answer. Reporting the two halves separately is also what makes
    "the window's internal shape tidied up while its placement never moved"
    visible at all -- the failure seven arms could not see.
  * **The default is `pairwise_weight = 0.0`**, so it is off unless an arm asks
    for it. `g14_pairwise` sets 0.01 and `g19` through `g27` inherit it;
    `g28_ceil_local` is base-rooted and therefore ran with it at **zero**.

So the answer is not "we are not using it". It is: the placement half is on at
0.01 in every recent arm, the internal half is off for a measured reason, and
the one arm that had it fully off is the teacher-forced capacity test, where
the ideal move was supplied directly instead.

**The live tension, recorded rather than resolved.** Gala's authors call the
all-pairs loss *necessary* to identify the target uniquely, because nodes "are
not aware of their global location". Our weighted half covers window-to-anchor
only. Raising the weight or widening the pair set cuts against
`docs/claims/the-objective-should-restrain-the-output-not-teach-the-placement.md`,
and the rigidity node is explicit that this tension is real rather than
settleable by preference: an under-determined target is not weaker supervision,
it is a different target.

---

**NOTHING IN FLIGHT as of 23 Nov 15:30. THE CAPACITY QUESTION IS ANSWERED NO,
AND THE ARM CHAIN ENDS HERE.**

`g28_ceil_local` (33308293, `train exit : 0`, 72.2 min) was the one unconfounded
cell: teacher forcing, local rotation, identity off, 2,000 updates. Read against
buckets pushed before the number existed, `align` = **+0.036, 95% CI
[-0.043, +0.114]** -- the whole interval inside the pre-registered null band of
-0.10 to +0.25. Readouts 33313562 (trained) and 33313576 (untrained, frame head
perturbed 0.05, same crops and convention, -0.141) on the update-2000 checkpoint.

**That is the architectural bucket.** Invariant perception plus a local move
head cannot be taught to aim even when handed the exact answer at every one of
2,000 steps. `docs/claims/the-architecture-can-place-residues-under-some-supervision.md`
is now `unsupported`, and the clause written into the pre-registration applies:
**do not add terms and re-run.** Full reading, guard, control, corroboration and
scope: `docs/claims/teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero.md`.

Three things to carry, because each could be misread:

  * **The guard's band condition was NOT met.** Twist never reached 9-12 at any
    evaluation (13.9, 22.1, 13.2, 15.3, 12.3, 17.2, 13.3, 18.9; minimum 12.3).
    No evaluation jumped 2x -- the largest ratio is 1.59 -- so the final
    checkpoint was readable under the pre-registered rule, and both readouts
    confirm they read `update: 2000`. But the band requirement was unsatisfied
    and is recorded as unsatisfied, not reinterpreted. It never ran away: it
    returns to 12.3-13.3 after every excursion, unlike `g10` which went 9.9 to
    52.3 and stayed at 50.2.
  * **Identity off is NOT sufficient for a quiet ceiling arm.** `g9_ceil_noid`
    held 9.1-9.2 with identity off; this arm oscillated 12.3-22.1 with identity
    equally off, and the difference between them is the local rotation. That
    does not refute the identity-jump node, whose claim is about a >2x jump,
    but the pre-registration's "identity off is what keeps a ceiling arm alive
    long enough to be read" is too strong. Alive, yes. Quiet, no.
  * **The readout's arrival warning does not reach `align`.** It flags that the
    gradient window is `truncate 12 x 0.3 A = 3.6 A` while the damaged window
    starts a median 10.95 A away, 10 of 12 crops beyond it, so those crops
    cannot be taught to *arrive*. That is why no `rmsd_dmg` here means anything.
    `align` is a per-step bearing and needs no arrival, which is exactly why the
    pre-registration made it the primary readout. **Raising `truncate` is not
    the escape hatch** -- pre-registered against two nodes at once.

Corroboration not needed for the bucket but worth knowing: `gained` is -0.02 A
(the window ends marginally FURTHER from native than it started), `net/path` is
0.996 (straight-line motion, not wandering), the per-step bearing is noise from
the first step rather than a decaying signal, and training moved the *variance*
(sd 0.266 to 0.138) while leaving the mean at zero -- a consistent small move
that points nowhere, asking for 0.04 A per step at 12.8 % of the cap with the
tanh unsaturated.

---

**SUPERSEDED, 23 Nov: 33308293 `g28_ceil_local`** on `gpu_rocm`/bun159, started
13:57:43 AEST. The resubmission of the arm 33304426 lost. Commit `408e813` confirmed on the cluster
by `git pull` before submitting, and the queue held nothing but the code-server
session, so one GPU slot and none of Ulban's own work crowded.

**Four confirmations in the header this time, not three.** `config :
configs/regeneration.toml [g28_ceil_local]`; the `CEILING CHECK: --move-weight 1`
banner; **no `TRANSLATION ONLY` line** and an `EQUIVARIANT MOVE` line in its
place, which is the positive confirmation that the rotation is local; and --
better than inferring it from the config -- the weights line reads `identity 0`
directly, so the one thing that kept `g9_ceil_noid` alive to 2,000 updates is
confirmed on this arm from its own output.

**Read it against the pre-registered buckets in
`docs/claims/the-capacity-test-has-never-been-run-with-a-working-actuator-and-no-identity-break.md`,
and check the twist guard FIRST to decide which checkpoint is readable.**

A lead this header hands over, worth one cheap check: it prints `covalent runs
per crop: mean 9.5 (range 1-19) over 8 crops` at startup, while the evaluation
block prints `segments 15.7`. The record has carried those two as a
disagreement between `growth.py`'s "about nine covalent runs" and the placement
node's "15.7 anchors". They are in the SAME log, which makes it likely they are
two different quantities measured at two different times rather than one
quantity measured inconsistently. Check the definitions before quoting either.

**DEAD, 23 Nov: 33304426 `g28_ceil_local` crashed at update ~250, `train exit : 1`.**
NOT on the science. `scripts/train_growth.py` had two variables named `pool` in
one function scope -- the structure list, and the Growing-NCA sample pool keyed
by `(structure index, crop seed)` assigned 220 lines later, which shadows it --
and the best-checkpoint writer's `[pool[i].pdb_id for i in eval_indices]`
therefore indexed a tuple-keyed dict with an integer. `KeyError: 180`. **That
line could never have succeeded on any machine**, so every run since it landed
on 21 Nov would have died at its first checkpoint, and **no checkpoint in this
project has ever carried `eval_pdb_ids`** -- the guard added to stop silent pool
mismatches has never fired once. All six probe jobs that consumed it
(33291294/95/97, 33303677/79/81) printed the "records no eval_pdb_ids" warning
and measured anyway, and the basin note wrongly said the check had passed.
Corrected there. Full account:
`docs/claims/the-held-out-id-guard-never-recorded-anything-and-killed-the-arm-it-was-meant-to-protect.md`.
The fix renames the list `catalogue` and resolves the ids **once before the
first update**, returning 1 if they will not resolve, so a run that cannot name
its held-out proteins refuses in a second rather than after a GPU hour; the
resume checkpoint records them too. **Not resubmitted until the fix is proven
end to end against a real checkpoint's contents** -- a green suite passed over
the broken line for two days because nothing asserted what a checkpoint
contains, and resubmitting on a compile check would be the same mistake twice.
The arm is unchanged and its pre-registration is **unread**: no `align` exists.
The 250-update row it did produce (`rmsd_dmg 11.87 A`, `fit_intact 12.88 A`,
twist 13.9d) is a 250-update row and is read against nothing.

Its confirmations held and carry over to the resubmission. 2,000 updates,
~75-100 min. Confirmed running the arm and not
the base: the header reads `[g28_ceil_local]`, carries the `CEILING CHECK:
--move-weight 1` banner, and **does NOT carry the "TRANSLATION ONLY: the
rotation is still applied in the lab frame" line** that every
`lab_rotation = true` arm prints -- its absence is the positive confirmation
that the rotation is local, the same check that confirmed `g23`.
`AANCA_ENTRY=scripts/train_growth.py` was exported and echoed to the log before
submission, because that is exactly how 33245088 died.
**Read it against the pre-registered buckets in
`docs/claims/the-capacity-test-has-never-been-run-with-a-working-actuator-and-no-identity-break.md`,
and check the twist guard FIRST to decide which checkpoint is readable.**

**NOTHING IN FLIGHT as of 21 Nov. All four jobs finished, exit 0.** The 4.0 A
test was read and **it FAILED**, which is the answer to the question and is
recorded directly below rather than in the arm list.

    33249861  gpu_rocm  g22_random                DONE  397.1 min  FAILS bucket 0
    33245111  gpu_rocm  g23_localrot              DONE  376.8 min  VOID on twist
    33249842  general   geometric floor, random   DONE  the comparator
    33249843  general   geometric floor, helix    DONE  BROKEN, see scope
    33245103  general   dispersion on g21         DONE, read above

`AANCA_ENTRY=scripts/train_growth.py` was set explicitly before submitting
33249861 and the echoed sbatch line ends `bunya_train.slurm
configs/regeneration.toml --section g22_random`, so it is the growth trainer
and not the packing default that killed 33245088. Checked rather than assumed,
because that is exactly how 33245088 died.

### The two corrections that came out of building the floor (20 Nov)

**1. `rmsd_dmg` is fitted on a superposition that INCLUDES the displaced
window.** `geometric = final.alive` (`growth.py:1912`) and a regeneration run is
100 % alive, so the twelve residues whose placement is the question are part of
what the measuring frame is fitted to, and the fit absorbs some of the error.
`backbone_rmsd_on`'s own docstring prescribes fitting on the surviving context;
`baseline_probe.py` has reported that all along as `fit_intact` while saying
*"fit_intact is the one that means what it says, and fit_all is the one every
arm has been scored by."* On start states, laptop pool, 12 crops:

    baseline     fit_all  fit_intact   diff
    helix          10.77       11.83   1.06
    random         14.12       15.84   1.73
    extended       21.30       23.74   2.44

Always in the direction that flatters the rule, and growing with displacement.
**The gap on a trained rollout had never been measured**, so g20's 7.78 and
g21's 8.41 are under-reports by an unknown amount of order 1-2 A. The trainer
now logs both columns (`GrowthOutcome.rmsd_damaged_intact`), and `g22_random`'s
bucket 0 is a line on **fit_intact**.

**2. `scripts/geometric_floor.py` -- the denominator every number here has been
missing.** It optimises the window against peptide-bond closure and steric
overlap ONLY, with no native information in the objective at all, using **the
rule's own degrees of freedom** (a free rigid move per window residue through
`rigid_relax`, not torsions) so that the optimiser and the rule search the same
set of structures and differ only in that one of them is told where native is.
Laptop pool, 3 crops, 1500 iters, chain closed (bond 0.0046 / 0.0328 / 0.0006):

    start              helix  fit_all 10.34     random  fit_all 11.03
    GEOMETRIC FLOOR    helix  fit_all  9.35     random  fit_all  9.62
    best of 3 random restarts                           fit_all  6.86

Two readings pointing opposite ways. **Against the floor the rule wins**: ~9.4 A
comes from closure and sterics alone against g20's 7.78. **Against the lower
bound it does not**: valid completions at 6.86 A are findable in three random
draws, so the rule sits well inside what the constraints permit and is not
selecting among completions well. Read the MEAN as the floor -- a native-blind
method cannot know which valid completion to emit -- and the best as the bound.

**NEITHER IS QUOTABLE YET, and the reason is the standing trap.** These are
laptop numbers on a 954-structure pool; g20's 7.78 is a Bunya number on 24,709,
where the same crop index is a different protein. That is what had the helix
baseline quoted as 10.77 for months when Bunya reads 12.23. Three crops, and
1P6F's floor came out *worse* than its own start. **33249842 and 33249843 are
the runs that settle it.**

`g23_localrot` at update 2480: twist oscillating **10.2-12.8** with the held-out
row at 2250 reading **11.5 deg**, so it is inside its 9-12 band and its void
condition has not fired. rmsd 2.83-3.38 A, alive 100 %, `rmsd_dmg` **10.78 A**
at update 2250 against the 12.23 baseline. 2.27 s/update, so ~4.7 h left. The
12.2d seen at update 806 was oscillation and not a trend, which is what the
early flag was uncertain about.

`g23_localrot` 33245111 is confirmed to be running the arm rather than the
base: g21's log header carried *"TRANSLATION ONLY: the rotation is left in the
lab frame"* and 33245111's does not, which is the flag taking effect observed
rather than assumed. One earlier submission, **33245088, died at 6 seconds with
exit 2** -- `AANCA_ENTRY` was unset so it ran the packing trainer and argparse
rejected the config arguments. No GPU time spent; the fix is
`export AANCA_ENTRY=scripts/train_growth.py` before `bunya_submit.sh`, which
the wrapper's own usage line states and a split command line dropped.

The dispersion job reproduces g20's measurement exactly -- 6 crops, 8 members,
amplitudes 0/15/45, 600-step cap, `--device cpu` -- and its header prints the
crop ids as `['3B82','4CQE','1JNP','2FAZ','8QTF','6HJ6']`, identical to g20's,
so the comparison is like for like and that was checked rather than assumed.
It reads `best.pt at update 9000`, the same selection rule g20's used.

**`g23_localrot` is registered and now running.** `g20_orient` with
`lab_rotation = false`, one setting, against g20_orient's 7.78 A at its
120-step horizon. The old `g16_localrot` registration is void: its comparator
was `g15_both`, which ran `native_edges`, a setting deleted from the schema on
9 Nov, so no section can express it. **No +- window is stated** -- the first
draft's +-0.34 A came from the g14-to-g15 gap and both arms are retired, no arm
has ever been replicated, so a reading near 7.78 is reported as "no difference
resolvable" and nothing stronger.

**Why it is safe to drop the flag now, which was not true at g10-g12.** The
reason it was kept is a four-arm table where the local conjugation trebled the
twist break's amplitude (16.6 and 17.6 against 50.2 and 53.9). That break
belongs to the identity term, and `identity_weight = 0.0` in this base, so
there is nothing left to amplify. If that attribution is wrong, the arm's
fourth bucket fires and the rotation-half claim is wrong whatever rmsd_dmg
does.

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

