# AANCA v1 — plan

**Agreed 2026-10-28.** Supersedes the model and evaluation sections of `HANDOFF.md` where they
disagree. `HANDOFF.md` remains the reference for prior art and for the Bunya cluster.

---

## 0. What success means

**Theoretically sound and experimentally accurate. Not necessarily unique.** Decided 2026-10-28,
and it is a deliberate correction to an earlier framing in this document that treated the
unpublished-ness of the fixed-point objective as the goal.

The consequence that matters: **if the learned rule turns out to behave like a machine-learned
interatomic potential, that is an interesting finding and gets reported as one.** It is not a
failure mode to be designed around. See §6a, which turns that question into two measurements rather
than a rhetorical position.

This also lowers the bar the work has to clear. It does not need to beat RDE or ProteinEBM. A local
update rule for side-chain packing, characterised honestly — accuracy, basin size, stability, and
whether its update field is conservative — is a result whichever way those numbers fall.

## 1. Scope: one model, one loss, one dataset, one split, four plots

v1 answers one question: **does a strictly local, step-unconditioned rule converge to the native
side-chain packing and stay there?**

That question needs only geometric damage on ordinary PDB structures. Everything that needs measured
ddG is cut, and cutting it removes most of the project's complexity at no cost to the claim —
because the unpublished-for-proteins result (tier 2 below) does not depend on it.

### In

- Fixed native backbone, side chains only.
- Geometric damage: von Mises noise on chis, up to fully randomised rotamers.
- Per-residue outputs: Δχ for chi1–4, plus a confidence κ per chi.
- Repacking accuracy, convergence, stability, basin of attraction.
- One cheap κ validation: does predicted κ track crystallographic B-factors?

### Out, and why

| Deferred | Reason |
|---|---|
| SKEMPI, ΔΔG, mutations | A separate piece of work. Brings the leakage split, the cross-task leak, and the readout choice with it. |
| `damage.mutate()` | Stays an explicit `NotImplementedError`. Its seeding rule interacts with the ddG readout. |
| ProteinEBM scoring | Post-hoc, addable to a trained model at any time. |
| Bond and angle *driving* | The degrees of freedom and their restraints are **built** (§2a); the heads are off in v1. |
| Per-residue energy head, variant (B) | See §3. |
| Gibbs sampling as a run mode | Falls out of the architecture for free once wanted. See §3. |
| Loop remodelling, backbone remodelling | The escalation ladder, untouched. |
| Hidden-state step-index probe | Worth running before publication, not before v1 works. |

None of these are rejected. Each adds a way for v1 to fail for reasons unrelated to the hypothesis.

**Binder design needs no extra machinery.** An interface is packing between two chains, and the data
layer is already multi-chain: `Protein` carries chain IDs, the graph has a `same_chain` edge feature,
and forced sequence-neighbour edges correctly refuse to bridge a chain break. v1 only needs
multi-chain entries in the training set, not new code.

---

## 2. Model

~220k parameters. Parameter count is not the constraint here; BPTT memory is (§4).

| Component | Shape |
|---|---|
| Per-residue state | 128 = 32 read from current geometry + 96 free hidden channels carried between steps |
| Edge features | 575 → 128 |
| Message | MLP over `[s_i, s_j, e_ij]`, 384 → 128 → 128, summed over neighbours weighted by the envelope |
| Update | Gated residual on the 96 hidden channels |
| Head A | 128 → 64 → 8, mean Δχ as sin/cos pairs |
| Head B | 128 → 64 → 4, log κ, **capped** |

### The decisions, and the reasons

**One message-passing hop per step.** Depth comes from time, not from stacked layers. The measured
reason: at a 10 Å all-heavy-atom cutoff, mean degree is 28.0 on 1UBQ (76 residues) and 33.4 on 193L
(129 residues), so **one hop already reaches 26–37% of a small protein and two hops reach all of
it**. Three ProteinMPNN-style layers per step would make a single application of the rule globally
connected on anything CASP-sized, and the loss curve would never show it.

Consequence to state honestly in the paper: the defensible claim is *"one message-passing hop per
update, no global pathway, no step conditioning"* — not *"each residue sees only a handful of
neighbours"*. Locality is also size-dependent: 28 neighbours is 37% of ubiquitin but under 10% of a
300-residue target. If tighter spatial locality is wanted, the lever is the **cutoff**, not the
layer count.

**κ is in v1 even though v1 barely uses it.** It is the only component that cannot be added to a
trained model without retraining, because it changes the loss from a regression to a likelihood. The
B-factor check (§5) makes it testable without any ddG data.

**κ is capped.** A likelihood loss can be reduced by claiming enormous confidence rather than by
getting the mean right. Cap `log κ` and watch whether it saturates.

**Bond lengths and angles are built but frozen.** See §2a — an earlier version of this plan argued
they would cost the exactness guarantee, which was wrong.

**Delta head first, energy head as a planned second arm.** Variant (B) — a per-residue von Mises
*energy* over its own chis, with the joint defined as the sum — is the more defensible object:
consistent conditionals, principled Gibbs, and Langevin for free. (A) ships first because it tests
the fixed-point objective with the fewest confounds, but (B) is **not** merely an alternative to
fall back on: per §6a it is the ablation that makes the comparison against a learned potential a
measurement rather than an argument. Build (A), then (B) on the same trunk.

**Gibbs, never Langevin.** Langevin needs ∇log p of a joint density; we never form a joint, only
local conditionals. Note that **an asynchronous update mask plus a stochastic head already *is*
random-scan Gibbs sampling** — the head emits p(Δχ_i | neighbourhood), which is the Gibbs
conditional, and updating a random subset is the random scan. So sampling is a run mode, not new
machinery. Caveat: independently trained local conditionals need not correspond to any consistent
joint, so Gibbs from them can depend on scan order. Measure that dependence before claiming to
sample from anything; severe dependence is the empirical argument for (B).

---

## 2a. Bond and angle control

A classical forcefield has stretching and bending terms as well as torsions, so a learned one needs
the scope for them. The degrees of freedom are built; the heads are off in v1.

**The exactness objection was wrong.** An earlier version of this plan said free bond geometry would
cost the Milestone 1 round-trip guarantee. It does not, provided the operations are *incremental*
rather than a rebuild:

- a bond stretch is a translation of the distal atoms **along** the bond axis. The near atom does not
  move and the far atom moves along the axis, so the axis direction is unchanged and `-delta`
  restores the coordinates exactly;
- an angle bend is a rotation of the distal atoms about the axis through the apex **perpendicular to
  the a-b-c plane**. Rotating about a plane's normal maps the plane to itself, so the normal is
  preserved and `-delta` undoes it exactly.

Measured: all three coordinates round-trip to ~3e-14 Å, and **inversion is order-independent** —
negating the deltas restores the structure whichever order the inverse runs in, across every
permutation. The reason is that each operation re-derives its axis from the coordinates it sees, and
every axis is pinned by atoms upstream of that operation's own moving set plus at most one atom that
stays on the axis or in the plane. That makes composing damage operators safe without tracking order.

**The clash objection stands, and is handled the way a forcefield handles it.** Unrestrained, bond
geometry is a physically wrong shortcut: deforming a bond relieves a steric clash more cheaply than
finding the right rotamer, which lowers the clash term, raises χ MAE, and is invisible in aggregate
metrics. So the heads are bounded *and* restrained:

- hard caps of 0.05 Å and 0.15 rad, a few times the real spread (bond lengths vary ~0.01–0.02 Å,
  angles ~2–3° at these energies);
- a harmonic restraint to the **crystal** values, not to an ideal-geometry table, so the structure's
  own geometry is preserved rather than pulled toward a library mean.

**Ring bonds are excluded**, derived rather than listed. A rigid translation of one side of a ring
cannot be absorbed by the closure. The test is whether the hinge atom is still reachable with itself
removed — checking only for backbone reachability is *not* enough, since an aromatic ring contains no
backbone and PHE's CG–CD1 would otherwise look stretchable. Result: PRO and GLY expose nothing,
PHE/TYR/TRP/HIS expose only CA–CB and CB–CG, and TYR additionally CZ–OH, which is exocyclic.

## 2b. Nonbonded fields as hormone gradients — the next design step

Electrostatics and van der Waals as *perception*, giving each residue a local reading of a
long-range field. This is the most promising way to get distance information into a strictly local
rule, and it is physically honest: a residue senses the local value of a global field without any
information pathway, exactly as a morphogen gradient works, and real atoms do feel electrostatics at
range.

**State the cost to the locality claim honestly.** Computing the potential at residue *i* sums over
all charges, so it *is* a global reduction. The claim weakens from "strictly local" to "no *learned*
global pathway; the only global information is a physically-derived field that real atoms also
sense." That is defensible, and arguably more interesting than pure locality, but it is a different
claim and a reviewer will notice the difference.

Not built yet, deliberately: it needs partial-charge and Lennard-Jones parameter tables, where a
wrong entry produces silently plausible garbage features. When it lands:

- compute from *current* coordinates every step, so the field moves with the side chains — that
  dynamic is the whole point of calling it a gradient;
- feed invariant readings (potential, field magnitude, field direction in the residue's own backbone
  frame, LJ energy), never raw vectors;
- keep it separate from the loss. vdW-as-feature is perception; vdW-as-penalty is supervision. Both
  are reasonable and they are different things.

## 3. Training

- **Rollouts of 30 steps**, step count randomised per sample.
- **Asynchronous update mask, p ≈ 0.5.** Only a random subset of residues updates each step.
- **Loss at a random late step**, not the last. This is what forces a fixed point rather than a
  schedule.
- **No-op term**: a native, undamaged input must produce zero update.
- **Von Mises NLL**, with the two-fold symmetric chis folded (ASP χ2, GLU χ3, PHE χ2, TYR χ2).
- **Sample pool**, as in Growing NCA: persist intermediate states back into the pool.
- **Random global rotations** throughout. Translation augmentation is meaningless in torsion space
  but costs nothing.

### Why asynchrony does not cost the fixed point

The no-op term forces every residue to output zero Δχ at the native structure, so at the fixed point
any subset of zeros is still zero. **The update mask randomises the path, not the attractor.** A
*sampled* head would randomise the attractor itself — at native, each residue draws from a
distribution with nonzero variance, so the structure never stops moving, mean |Δχ| decays only to a
noise floor, and "stability over 500 steps" becomes diffusion. Hence: distributional head,
deterministic rollout, sampling only as an explicit mode.

Asynchrony also has a *scientific* justification, not just a training one: a synchronous update is a
weak global information channel, since every cell implicitly knows it is step *t* at the same
instant. Removing the shared clock strengthens the strict-locality claim. The convergence benefit is
narrower — it damps *synchronised* oscillation modes, which is the Grattarola failure mode.

Note the interaction with §4: at p = 0.5 only half the side chains move per step, so graph churn per
step roughly halves. The measured churn figures are synchronous and are an upper bound.

---

## 4. Memory, which is the real constraint

**Measured, replacing an earlier estimate that was 12× too optimistic.** Run
`scripts/measure_rollout_memory.py`. It counts the activations torch retains for backward, via
`saved_tensors_hooks` — process RSS was tried first and is useless here, giving negative deltas that
contradicted their own per-step figures.

Retained activations per step, k=64, 10 Å:

| | 1UBQ (76 res) | 193L (129) | 4HHB (574, 4 chains) |
|---|---|---|---|
| baseline, float64 | 70.7 MB | 120.1 MB | 534.3 MB |
| `fixed_backbone`, float64 | 37.2 MB | 63.2 MB | 281.1 MB |
| `fixed_backbone` + float32 | **31.4 MB** | **53.3 MB** | **237.1 MB** |

Cost is linear in residue count at fixed k, so a 300-residue target is ~124 MB/step: **6.2 GB for a
50-step rollout, 3.7 GB for 30 steps — per sample.**

The earlier estimate said ~500 MB for the same thing. It counted only the edge-feature tensor and
forgot the network's own activations: the message MLP operates on `(R, K, 384)` and `(R, K, 128)`
tensors, every one of which is retained. Those dominate.

Two optimisations, both measured:

- **`fixed_backbone=True`** builds the 400-dimension backbone block under `no_grad`, worth 47%. It
  is exact rather than approximate — only side chains rotate, and the backbone is excluded from
  every chi's downstream atom set, so that block's gradient is identically zero. **Detaching it
  afterwards does not work**, which was measured: by then the tensor has been built inside the graph
  and saved for backward, and a later `.detach()` frees nothing. It has to not be computed under
  grad. This flag must flip when the backbone-remodelling escalation lands.
- **Featurise from `coords.float()`**, worth a further 16%. Torsion arithmetic stays float64, where
  the exactness guarantee lives; distances and RBF expansions do not need it.

**Consequence for the plan: truncated BPTT is mandatory, not optional.** At 3.7 GB per sample for a
300-residue protein at 30 steps, a batch of 8 is ~30 GB and a batch of 32 is out of reach. Backprop
through 8–16 steps and seed the rest from the sample pool.

Wall time on CPU is ~0.5 s/step at 574 residues, so these rollouts are GPU work, not laptop work.

Before any run that writes many files, check `rquota` — the Bunya quota binds on inodes, not bytes,
and a checkpoint or rollout sweep is exactly that pattern. See `BUNYA_BROWSER_SHELL.md`.

---

## 5. Data and split

**One split, and it is theirs, not ours.** Looked up 2026-10-28; this replaces an earlier
placeholder that said "follow DiffPack's protocol" without knowing what it was.

The protocol AttnPacker, DiffPack and FlowPacker all share:

| | |
|---|---|
| Training | **BC40** — PDB clustered at 40% sequence identity — on **AttnPacker's split**: 37,266 chains train, 1,500 validation. DiffPack follows it verbatim. |
| Test | **CASP13 (82 regular targets)** and **CASP14 (64 regular targets)**, native backbones. FlowPacker adds CASP15. |
| Leakage filter | No training structure within **≥40% sequence identity of any test target**. FlowPacker and the Invariant Point Message Passing paper use MMseqs2 `easy-search` for this; the tool differs between papers. |
| Resolution filter | **Not documented** in any paper or repo I could find. Unresolved — do not invent one, since an undocumented filter is a divergence we would be unable to defend. |

What is actually obtainable, which is the awkward part:

- **The test sets are.** AttnPacker's Zenodo record 7713779 holds the PDB files used to generate
  their published results. That is the way to match the benchmark exactly rather than re-deriving
  the CASP target lists.
- **BC40 and the split are not.** AttnPacker's repository publishes pre-trained models and result
  PDBs but neither the training set nor the split list. So the training pool has to be
  reconstructed, and that reconstruction is a known source of divergence to state in the write-up.
- Reconstruction route: **RCSB's own `clusters-by-entity-40.txt`**. BC40 derives from RCSB's 40%
  sequence clustering in the first place, so this is the same resource rather than a substitute —
  but RCSB has changed clustering tool over the years, so the clusters will not be bit-identical to
  the BC40 those papers used. `data.ClusterIndex` consumes that file.
- **Holdout exclusion is by cluster, not by entry.** Dropping only the CASP targets leaves their
  homologs in training. `data.split_by_cluster` removes whole groups and *refuses to run* on an
  empty holdout list, because silently training on everything is how an indefensible number gets
  produced.

### The leakage filter uses the same tool they did

The papers remove training chains within 40% sequence identity of any test target using MMseqs2.
MMseqs2 is not installed here and is awkward on Windows — but **RCSB runs it on their servers**, and
biotite exposes it as `rcsb.SequenceQuery(sequence, "protein", min_identity=0.4)`. So the filter is
done with the same tool rather than a reimplementation, and the comparison does not hinge on our
having got sequence search right.

`scripts/build_training_manifest.py` extracts each CASP target's sequence from the benchmark archive,
queries at 40%, and excludes every cluster the hits touch.

### Grouping: the conservative version collapses the pool

Two notions of "related entries" are needed, for two jobs with very different stakes.

An earlier version treated entries as related whenever they shared **any** 40% cluster, resolved by
union-find. Measured on 7,601 X-ray entries at 1.2 Å, that produced a single group of **6,682
entries — 88% of the pool**, because a promiscuous entity (a tag, a chaperone, a common lysozyme)
chains otherwise unrelated entries together and the merging cascades globally. It left 648 usable
representatives.

Grouping instead by each entry's **primary (lowest-numbered) entity** cut the largest group to 751
and raised the pool to 2,123 — 3.3× more data — with no transitivity.

What makes that safe is the separation of concerns:

| job | notion used | stakes |
|---|---|---|
| exclude test homologs | `clusters_of` — **every** cluster the entry touches, plus the sequence-identity filter above | high: this is test leakage |
| split train from validation | `group_of` — the primary entity's cluster only | low: contaminating validation costs calibration, not credibility |

Test leakage is caught by the sequence search, which finds a homologous chain whatever entity number
it carries. Cluster grouping only has to keep train and validation apart, and the primary entity is
enough for that. `group_size_histogram` is kept as a diagnostic because a largest-group number is the
cheapest possible signal that grouping has gone wrong — it is what caught this.

Include multi-chain entries so interfaces are in distribution (§1).

### The built manifest

`scripts/build_training_manifest.py`, then `scripts/audit_manifest.py` to verify:

| | |
|---|---|
| X-ray protein entries ≤2.5 Å | 161,997 |
| deduplicated to one per 40% cluster | 25,303 |
| largest cluster group | 2,143 |
| CASP target sequences queried | 210, 0 failures |
| entries within 40% of a target | 2,227, spanning 646 clusters |
| representatives removed | 593 |
| **train / validation** | **23,761 / 949** |

The published split is 37,266 training *chains*; ours is 23,761 *entries*, and entries hold several
chains, so the volume is comparable or larger — but not comparably composed, which is a divergence to
state rather than gloss.

**No size bound on the pool; samples are cropped instead.** Retained activations are linear in
residue count, so something has to bound a sample, and the obvious move is to filter the pool —
which was the plan, at 40–1000 residues. It was dropped. The filter removes 14% of entries and it
removes the *largest* ones, which are the multi-chain assemblies: exactly the interface-rich
structures this project cares about. `src/aanca/crop.py` bounds the sample rather than the entry,
taking a ball of `crop_size` residues around a random seed, so a 2,400-residue assembly contributes
a different region each time it is drawn. The nearest published precedent for a cap is AttnPacker's
600 residues, which applies to their non-native-backbone *test* set rather than to training, so
there is nothing here to reproduce either way.

**The boundary shell is excluded from the loss, and that exclusion is the whole correctness
argument.** A residue near the edge of a crop has had its neighbours deleted, so its perception is
wrong — not noisy, wrong — and the native torsion it would be asked to predict depends on contacts
it cannot see. Supervising it teaches the rule to guess packing from absent evidence: the gradient
is real and the signal is not. So every crop carries a mask of the residues whose perception is
*complete*, computed exactly rather than by a tuned margin — a residue is supervised iff every
residue within `cutoff` of it in the intact structure is still present, by minimum heavy-atom
distance, which is the same quantity the featuriser builds edges from. The shell stays in the graph
as context for the interior and is scored on nothing.

That cropping is lossless in the interior is testable, not arguable, and
`tests/test_crop.py::test_the_mask_is_exactly_the_set_whose_perception_changed` tests it: on 4HHB at
crop size 300, the residues whose features differ between crop and intact structure are **exactly**
the 137 unsupervised ones, and all 163 supervised residues are bit-identical. This is the one place
strict locality buys something a global model could not have.

**The cost is measured, and it says crop size is a memory budget rather than an optimum.** Over 40
validation entries above 700 residues (`scripts/crop_report.py`):

| crop size | supervised | effective residues | MB per 12-step window |
|---|---|---|---|
| 192 | 29.3% | 56 | 945 |
| 384 | 46.2% | 177 | 1,889 |
| 512 | 56.6% | 290 | 2,519 |
| 768 | 73.7% | 566 | 3,779 |
| 1024 | 85.2% | 872 | 5,038 |

The shell is one interaction cutoff deep whatever the crop size, so supervised residues per MB rises
monotonically — 0.059 at 192 against 0.173 at 1024. There is no sweet spot to find: bigger is
strictly better and only memory stops it. The 512 default should be raised to whatever the training
hardware holds.

### The split is stable under changes to the pool

Each *group* is assigned a side by hashing its id, not by a draw from a shuffle. This was paid for.
The first version took `randperm` over the sorted group list, which makes every group's side depend
on how many groups exist — so when the leakage filter was corrected and **one** group moved to
`excluded`, the permutation shifted and **945 of 988 validation entries changed**, requiring the
whole validation set to be downloaded again to evaluate against the same split. Hashing makes
membership depend only on `(group, seed)`, so the pool can grow, shrink or have its filter corrected
without disturbing an unrelated group. The price is that the realised fraction is binomial around
the target rather than exact: at 25,303 groups and 4% that is roughly ±31 groups.

A related failure in the same build is worth recording because it was nearly invisible. One of the
210 MMseqs2 queries hit a transient `RequestError`; the script logged a line and carried on, and the
holdout came back **one entry smaller** — a cluster that should have been banned was not, and the
only trace was console output nobody would diff. That one entry *was* a representative, so an entry
within 40% identity of a CASP target would have entered training. Queries now retry with backoff,
and a build whose search was incomplete unions in the stored list rather than writing a manifest
that looks clean without being clean.

Most CASP targets returned very few identity hits, which is the right signal: CASP targets are chosen
partly because they lack close PDB homologs, so thousands of hits per target would have meant the
filter was broken rather than working.

**The audit is part of the artifact.** The leakage guarantee costs 210 MMseqs2 queries to re-derive,
so the build persists the matched entries and banned clusters and `audit_manifest.py` checks them,
exiting non-zero on violation. Current state: 24,710 survivors, **0 touching a banned cluster**.

Getting that check right took two attempts, and the wrong version is the tempting one. An excluded
entry may touch *two* clusters — one banned, because it holds a chain homologous to a test target,
and one innocent. A survivor sharing only the innocent cluster is 40%-similar to *a chain of the
excluded entry*, which says nothing about its similarity to any test target. Testing "shares any
cluster with something excluded" reported **60 false leaks**; the correct condition is "touches a
banned cluster". `tests/test_data.py::test_sharing_a_cluster_with_an_excluded_entry_is_not_itself_leakage`
encodes the distinction.

Two standing divergences from BC40, neither about quality:

1. **The 2.5 Å resolution cutoff is ours.** No packing paper documents one, so it is a recorded
   parameter rather than a reproduction.
2. **We group by entry where BC40 counts chains**, so the pool is composed differently.

One caveat on local smoke-testing: by the burial definition in §6, ubiquitin is only 15.8% core and
65.8% surface, where 193L and 4HHB are 36–39% core. The small convenient structures are mostly the
*easy* population, so they are fine for correctness checks and useless as an accuracy signal.

---

## 6. Evaluation — four plots

1. **Does it pack at all?** χ MAE and side-chain RMSD on CASP13/CASP14, against DiffPack,
   AttnPacker, FlowPacker and Rosetta. Not winning is fine; a strictly local, step-unconditioned
   rule landing *close* to a globally attentive diffusion model is already the interesting result.
   Being far off is the kill signal.

   **Three numbers, theirs, in `metrics.py`:** χ MAE per chi index, **χ accuracy — the fraction
   within 20°**, and side-chain atom RMSD. The accuracy figure is their headline alongside MAE and
   was missing from an earlier version of this plan.

   **Stratified by burial, which was the most valuable thing to copy.** AttnPacker's convention, as
   cited by the later papers: **core** is ≥20 CB atoms within 10 Å, **surface** is ≤15. Note that is
   deliberately *not* a partition — 16–19 neighbours is neither, and is reported separately rather
   than silently reassigned. Core residues are the hard ones; a pooled number hides whether a method
   packs well or is merely good at solvent-exposed side chains that barely matter. GLY has no CB so
   the virtual one is used; the papers do not say what they do, and `include_glycine_cb` exists so
   the alternative can be checked.

   **Both metrics must respect chemical symmetry**, and this is not a detail. A carboxylate's two
   oxygens and an aromatic ring's two halves are indistinguishable, so a 180-degree flip is the same
   structure. Scoring it by atom name costs ~1.8 Å on the affected residues and inflates the
   whole-protein mean side-chain RMSD by 0.26–0.36 Å — comparable to the entire gap between
   competing methods, in the one comparison that has to be auditable. `sidechain_rmsd` minimises
   over `rc.symmetric_relabellings`; `chi_error` and `chi_mae_degrees` fold the periodic chis.
   LEU/VAL methyls are excluded, matching AlphaFold's table and therefore the baselines.

   Later escalation, already defined by their protocol: the **non-native backbone** variants (47
   CASP13 and 27 CASP14 targets, ≤600 residues, Cα RMSD < 2.5 Å from native). Worth knowing it
   exists so we do not invent a different robustness test.

### Metric validation — resolved, with one gap

Reverse-engineered from AttnPacker's released predictions by recomputing DiffPack's Table 1 and
sweeping candidate definitions. **The headline metrics now reproduce.** Four independent methods,
CASP13:

| | χ1 | χ2 | χ3 | χ4 | acc all | core | surf | RMSD all |
|---|---|---|---|---|---|---|---|---|
| SCWRL ours / pub | 28.45 / 27.64 | 29.30 / 28.97 | 47.57 / 49.75 | 57.30 / 61.54 | 55.1 / 56.2 | 70.8 / 71.3 | 43.7 / 43.4 | 0.936 / 0.934 |
| FASPR ours / pub | 27.80 / 27.04 | 28.90 / 28.41 | 48.30 / 50.30 | 57.28 / 60.89 | 55.3 / 56.4 | 69.6 / 70.3 | 44.1 / 43.6 | 0.907 / 0.910 |
| RosettaPacker | 26.59 / 25.88 | 28.58 / 28.25 | 46.23 / 48.13 | 55.22 / 59.82 | 57.5 / 58.6 | 74.7 / 75.3 | 44.3 / 35.7 | 0.870 / 0.872 |
| DLPacker | 22.89 / 22.18 | 27.88 / 27.00 | 50.82 / 51.22 | 68.79 / 70.04 | 57.5 / 58.8 | 73.2 / 73.9 | 45.3 / 45.4 | 0.787 / 0.772 |

Three definitions had to be recovered, none of them stated in any paper. They are encoded as
`metrics.Convention`, with `PUBLISHED` for comparability and `CHEMICAL` for the loss — conflating
the two would either make our numbers incomparable or put a reverse-engineered convention into the
training signal.

1. **Angle Accuracy is per *residue*, not per chi.** A residue counts once, and only when every chi
   it has is within 20°. Pooling per chi reads ~10 points high, which was the largest single
   discrepancy.
2. **RMSD includes CB and does *not* minimise over symmetry-equivalent labellings** — even though
   the angle error *is* folded. Inconsistent of them, but it takes all-RMSD from 0.15 Å high to
   0.003 Å (FASPR: 0.907 against 0.910).
3. **The symmetry set is AlphaFold's plus ASN χ2 and HIS χ2, but excluding GLN χ3.** Folding ASN/HIS
   χ2 takes χ2 from a 3.9° gap to 0.5°; folding GLN χ3 as well takes χ3 from a 2° agreement to a 5°
   disagreement across all four methods. Chemically odd — GLN's amide is as ambiguous as ASN's — so
   treat it as an observed convention of theirs, not a defensible rule.

**The remaining gap: burial-stratified RMSD.** Core RMSD is ~0.14 Å high (0.658 vs 0.502 for FASPR)
and surface ~0.10 Å high, while all-RMSD matches to 0.003 Å. That is only arithmetically possible if
their band *populations* differ from ours. Excluding glycine's virtual CB from the neighbour count
narrows it slightly (0.658 → 0.634) but does not close it. Untested candidates: neighbour counts
taken on the predicted rather than the native structure, a different radius, or backbone atoms
included in the count.

So: **χ MAE, Angle Accuracy (all/core/surface) and all-residue RMSD are safe to quote. Core and
surface RMSD are provisional.** Also unresolved: AttnPacker's own files need a different residue
alignment and still produce nonsense, which matters because it is the strongest baseline.

### How that validation was reached

AttnPacker's Zenodo release (record 7713779) ships the native test structures *and* every baseline's
predictions, so `metrics.py` can be checked against the published table rather than against itself.
`scripts/validate_benchmark.py` does that. Where it stands on CASP13 (DiffPack Table 1 as reference):

| | χ1 ours | χ1 published | χ2 gap | accuracy gap | RMSD gap |
|---|---|---|---|---|---|
| SCWRL | 28.45 | 27.64 | +3.4 | +10.5 pts | +0.150 Å |
| FASPR | 27.80 | 27.04 | +3.8 | +10.4 pts | +0.150 Å |
| RosettaPacker | 26.59 | 25.88 | +3.1 | +9.5 pts | +0.139 Å |
| DLPacker | 22.89 | 22.18 | +4.3 | +7.6 pts | +0.134 Å |

**χ1 reproduces to under 1° across four independent methods.** That validates chi extraction,
symmetry folding and the burial classification — four different programs' outputs would not all land
within 1° by accident.

The rest is off *systematically and in the same direction for every method*, which means a
definitional difference rather than a bug in any one place. Until it is identified, **our numbers
cannot be quoted against theirs.** Candidates, cheapest first:

1. **Is CB in the RMSD atom set?** We exclude it. CB is almost always correct, so including it would
   lower RMSD — consistent in sign with our being 0.13–0.15 Å high across the board.
2. **Which residues enter the denominator?** We score a residue when every side-chain atom is
   present. They may additionally exclude incomplete, terminal or non-rotameric residues. Our
   accuracy being high *and* RMSD being high is contradictory unless the populations differ, so this
   is the most likely single explanation.
3. **Does their accuracy fold symmetry?** Not folding would lower their accuracy. But it would also
   raise their χ2 MAE, and theirs is *lower* than ours, so this does not explain both.
4. **AttnPacker's own files need different residue alignment** — currently they match few residues
   and produce nonsense (RMSD 3.6 Å against a published 0.669 Å). Their numbering differs from the
   native files in a way the chain-blank fallback does not handle.

Two things this exercise already fixed, both of which would otherwise have produced plausible wrong
numbers forever: residues were being matched **by file position** across independently loaded files
(giving a 2× error), and SCWRL4's output leaves the B-factor column blank, which a strict parse
rejects outright (`load_protein(..., repair_columns=True)`).
2. **Does it converge?** Mean |Δχ| against step count. Does it decay, or oscillate as some of
   Grattarola's morphogenesis rules did?
3. **Does it hold?** 500 steps on a rule trained to 30. Measure drift from the step-30 structure.
4. **How big is the basin?** Recovery accuracy against perturbation magnitude. **This is the
   headline figure.** Nothing in the packing literature reports it.

Plus the κ check: convert predicted κ to a per-atom positional spread — circular variance
`1 − I₁(κ)/I₀(κ)` times the squared perpendicular distance from each downstream atom to its rotation
axis, which `CHI_DOWNSTREAM` already gives — and correlate with `B = 8π²⟨u²⟩` from the crystal
structure. Report Spearman, **per atom** rather than per residue, since the lever arm means one
high-κ χ1 inflates the prediction for its whole downstream set.

B-factors absorb refinement error, occupancy problems and resolution effects, so this is a
validation correlation and **never a loss** — training on them fits the refinement protocol.

## 6a. Characterisation: what kind of object did we learn?

Architecturally this is close to a machine-learned interatomic potential. NequIP, MACE and Allegro
are also strictly local, weight-shared across sites, invariant by feature design, and applied
iteratively to move atoms downhill; Allegro's selling point is literally strict locality, and
ProteinEBM is the protein-scale version. Adding nonbonded fields (§2b) and bond/angle control (§2a)
makes the *architecture* more forcefield-like, not less.

So the distinction cannot rest on architecture. It rests on two things: the update is learned
directly rather than derived as ∇E, and the supervision is the global basin structure rather than a
local derivative. Per §0, these are things to **measure and report**, not to defend.

### The curl test

An MLIP's update field is conservative by construction — it is the gradient of a scalar. A directly
learned update field need not be. Walk closed loops in torsion space and sum the update's component
along the loop tangent; nonzero circulation means the rule is not the gradient of *any* energy
function.

Report whichever way it falls:

- **Non-conservative.** The rule is doing something no potential can, and the Graph NCA oscillation
  failure mode becomes the expected risk rather than a surprise — limit cycles are what
  non-conservative fields do.
- **Approximately conservative.** We learned a local potential's gradient without ever forming the
  potential, and without needing force labels. That is a legitimate and useful result: MLIP training
  on proteins is bottlenecked on reference forces, and every PDB structure being its own label
  sidesteps that bottleneck entirely.

Cheap — a few hours on a trained model — but it needs a trained model, so it cannot run until after
§5 and a first training run.

### The energy-head ablation

Train the **same trunk** with a scalar energy head and descend its gradient. Same data, same
locality, same parameter count; only the objective differs. This is variant (B) from §2, promoted
from "deferred alternative" to a planned arm of the study.

Without it, "how does this differ from a learned potential?" gets answered by the reader. With it,
the answer is a number, and the comparison is clean because nothing but the objective changes.

### What AANCA gives that a potential does not

Worth stating plainly, since two of these are already built:

- κ is a per-site conformational uncertainty read off directly. A potential needs an ensemble.
- Rigid bond lengths and angles are exact by construction in torsion space. An MLIP must learn that
  bonds are stiff from data and spends capacity on it.
- No force labels are needed at all.

And what it does not give: free energies, thermodynamic ensembles, or kinetics. This is a
native-basin attractor finder, not a thermodynamic model, and the write-up should say so rather than
let a reader assume otherwise.

---

## 7. What is already built and verified

See `README.md` for the numbers. In short: chi extraction and exact modification for all 20 residue
types (round trip to 3.6e-14 Å against a 0.01 Å bar, cross-checked against rotamer statistics), the
geometric damage operators, the per-step perception graph with invariant features and a measured
`k=64` default, a differentiable Torch path pinned against that NumPy reference, the rule itself at
200,268 parameters, the von Mises loss with symmetry mixtures, the rollout with truncated BPTT and a
sample pool, the dataset pipeline with spatial cropping, and the published-metric conventions
reverse-engineered and reproduced against four packers. 482 tests.

The whole path from a downloaded coordinate file to a parameter update has been run end to end on
real structures (`scripts/smoke_train.py`). On all 949 validation entries, 926 parse (97.6%); the 23
failures are one class, nucleic-acid complexes that pass RCSB's polymer-monomer filter and yield
almost no protein, and they are rejected loudly. Entries run to 6,168 residues and crop to 512; 331
of 926 are large enough to crop, mean supervised fraction 87.7%. Loss is monotone in damage
severity: 1.13 at κ=100, 1.20 at κ=20, 1.34 at κ=5, 3.61 under uniform damage.

Two of my own diagnostics were wrong in the same direction and are worth recording, because both
would have been consulted exactly when they mattered. The smoke test summarised "loss first → last",
which compared a mild-noise sample against a uniform-damage one and read as a 3× regression when
nothing had regressed; loss is not comparable across damage levels, so it now groups by level. And
it printed the first five parse failures rather than counts per class, which let a whole class hide:
the 154 failures of an earlier run were reported as "all the size guards" because the five shown
were, while eight were a padding-width bug. `MAX_ATOMS` was hand-counted as 14, TRP's heavy-atom
count — correct except at a chain's C-terminus, which carries an extra `OXT`. Every structure whose
chain ended in TRP overflowed by one atom and vanished from the pool as a parse failure. The width is
now derived from the tables, and the rate of that bug rose with the multi-chain assemblies cropping
brought in, since every chain has its own terminus.

### The run is interruptible, because on a cluster that is the normal case

Walltime, pre-emption, a cancelled job: the question is never whether a run is interrupted, only
whether the interruption costs its history or its last few minutes. `src/aanca/checkpoint.py` saves
the whole training state -- weights, optimiser, the Python RNG, the torch generator, the global
torch RNG, the sample pool and its own RNG, the structure split and the history -- because saving
the weights alone would continue from the right parameters along a *different trajectory*, and
every diagnostic in §6 is read as a curve over updates. A curve assembled from differently-seeded
segments is not a curve.

Measured: stopped and resumed is bit-identical to straight through, at one thread. Two identical
invocations differ by ~6e-8 at the default thread count, because multithreaded reductions sum in
thread-completion order, so the regression test pins `--threads 1` -- the one condition under which
an inequality can only be a bookkeeping bug. Checkpoint writes are atomic, so a kill during the
save costs the new checkpoint rather than the old one.

A deliberate stop and an interruption are different events and exit differently (0 against 64), so
`scancel` is not followed by a resubmitted job while a walltime is. Resuming with a different crop
size, rollout length, truncation, seed, k or no-op weight is refused: that is two experiments, not
one run. Extending `--updates` is allowed, because a longer run is the same run.

**What v1 still needs is the result of a training run.** Every component works and the plumbing is
verified, but the first night on Bunya produced no training at all: the GPU arm could not see the
card (a ROCm 6.4 wheel on a gfx950 MI355X) and the CPU arm died after 22 minutes at a neighbour
search on non-finite coordinates. The second is the one that matters scientifically. A non-finite
update is now skipped rather than applied — `clip_grad_norm_` turns one NaN gradient into NaN
weights, permanently — and the sample pool refuses a non-finite state, since a state stored there
is drawn as a seed later and makes the failure outlive the update that caused it.

**The cause, found the same day, was not numerical.** Successive samples of one entry are
deliberately different crops of it, and the pool was keyed by structure id, so a state produced on
one crop was drawn as the seed for another. Every crop has the same shape, so the shape check in
the way could never fire, and the chi atom indices then addressed atoms that are not bonded. The
quantity that shows it is the largest distance any atom moves in a rollout, against the 11.5 Å
ceiling a π rotation on every chi of 4HHB produces: 45.8 Å before the fix, 3.46 Å after, same seed
and same structures. 52% of validation entries exceed the 384-residue crop, so roughly half of
every pool re-seed since the pool was written had been training on coordinates and a target that
described different residues. §4's "does it stay" half was the part being corrupted, which is also
why the shakedown's flat `|delta|` and rising drift should be re-read on the fixed code before
being treated as findings about the objective.

The shakedown's two early signals have now been re-measured on fixed code, 14,750 updates into
`v3_gpu`, and they did not survive unchanged. Mean `|delta|` is **no longer flat**: it peaks near
10°/step around step 7 and decays to about 3.1°/step by step 30, so the rollout is partially
contractive. Drift from a native start, on the other hand, is **worse than the signal that first
raised it** — between 1.08 and 5.61 A across 29 evaluations with no downward trend, against the
0.473 A that came out of the corrupted pool. The κ=100 column says it in the metric too: chi1 MAE
from a near-native start goes 4.46° untrained to 5.34° trained. The rule has learned to repack a
destroyed side chain and has got worse at leaving a good one alone.

So the answer to §4's question is, so far, **no**: the objective is producing a denoiser with a
nonzero noise floor rather than a map with an attractive fixed point. Held-out pooled chi1 MAE
falls 34.68° -> 24.7° by update 9,000 and is then flat for 5,500 updates, so the ceiling of this
architecture under this objective is reached in about 2.5 GPU-hours, not in the 200k-update budget.
The `--noop-weight 4` arm now has a measured justification rather than the contaminated one, and is
the first thing to read. If weighting the no-op term does not flatten drift, the problem is
structural and §4 needs rewriting rather than retuning. After that: nonbonded fields as perception
(§2b), which needs partial-charge and Lennard-Jones tables, and the characterisation in §6a, which
needs a trained model to characterise.
