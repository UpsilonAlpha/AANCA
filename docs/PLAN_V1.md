# AANCA v1 — plan

**Agreed 2026-10-28.** Supersedes the model and evaluation sections of `HANDOFF.md` where they
disagree. `HANDOFF.md` remains the reference for prior art and for the Bunya cluster.

---

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
| Free bond lengths and angles | See §3. |
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

**Bond lengths and angles stay rigid.** Not on principle — they vary by ~0.01–0.02 Å and ~2–3° at
the energies packing cares about, which is why Rosetta, AttnPacker, DiffPack and FlowPacker all fix
them. Making them free outputs hands the model a physically wrong way to relieve a steric clash:
squash a bond instead of finding the correct rotamer. The clash term falls, χ MAE rises, and the
failure is invisible in aggregate metrics. It would also cost the Milestone 1 exactness guarantee,
which holds precisely because every move is a rigid rotation about a bond.

**Delta head, not energy head.** Variant (B) — a per-residue von Mises *energy* over its own chis,
with the joint defined as the sum — is the more defensible object: consistent conditionals,
principled Gibbs, and Langevin for free. It is deferred because the unclaimed contribution is the
fixed-point objective, and (A) tests it with the fewest confounds. (B) also moves the work toward
"energy plus a sampler", which is what the prior-art write-up identifies as the field's existing
version of the idea.

**Gibbs, never Langevin.** Langevin needs ∇log p of a joint density; we never form a joint, only
local conditionals. Note that **an asynchronous update mask plus a stochastic head already *is*
random-scan Gibbs sampling** — the head emits p(Δχ_i | neighbourhood), which is the Gibbs
conditional, and updating a random subset is the random scan. So sampling is a run mode, not new
machinery. Caveat: independently trained local conditionals need not correspond to any consistent
joint, so Gibbs from them can depend on scan order. Measure that dependence before claiming to
sample from anything; severe dependence is the empirical argument for (B).

---

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

Keeping full edge features for a 300-residue protein at k=64 is ~44 MB/step in fp32, so ~2.2 GB for
a 50-step rollout before activations. **Caching the backbone block** — 400 of the 575 edge
dimensions, constant across a rollout because the backbone is fixed — drops the per-step tensor to
~10 MB, so ~500 MB. That is the difference between fitting on one MI355X and not.

Combine with **truncated BPTT**: backprop through 8–16 steps and seed the rest from the sample pool.

Before any run that writes many files, check `rquota` — the Bunya quota binds on inodes, not bytes,
and a checkpoint or rollout sweep is exactly that pattern. See `BUNYA_BROWSER_SHELL.md`.

---

## 5. Data and split

**One split.** The four-pool scheme belongs to the ddG work and is deferred with it.

- PDB subset, **following DiffPack's published filtering protocol**. Tier 1's job is "not worse than
  existing packers", and that claim is only auditable if the pipeline matches; any deviation is a
  free objection. Read the exact filters out of the paper rather than reconstructing them.
- Split **by 40% sequence-identity cluster, not by chain**.
- **CASP13 and CASP14 held out, together with their homologs.** Dropping only the targets leaks.
- Include multi-chain entries, so interfaces are in distribution (§1).
- A validation set of held-out clusters for early stopping and κ calibration.

---

## 6. Evaluation — four plots

1. **Does it pack at all?** χ MAE and side-chain RMSD on CASP13/CASP14, against DiffPack,
   AttnPacker, FlowPacker and Rosetta. Not winning is fine; a strictly local, step-unconditioned
   rule landing *close* to a globally attentive diffusion model is already the interesting result.
   Being far off is the kill signal.

   **Both metrics must respect chemical symmetry**, and this is not a detail. A carboxylate's two
   oxygens and an aromatic ring's two halves are indistinguishable, so a 180-degree flip is the same
   structure. Scoring it by atom name costs ~1.8 Å on the affected residues and inflates the
   whole-protein mean side-chain RMSD by 0.26–0.36 Å — comparable to the entire gap between
   competing methods, in the one comparison that has to be auditable. `sidechain_rmsd` minimises
   over `rc.symmetric_relabellings`; `chi_error` and `chi_mae_degrees` fold the periodic chis.
   LEU/VAL methyls are excluded, matching AlphaFold's table and therefore the baselines.
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

---

## 7. What is already built and verified

See `README.md` for the numbers. In short: chi extraction and exact modification for all 20 residue
types (round trip to 3.6e-14 Å against a 0.01 Å bar, cross-checked against rotamer statistics), the
geometric damage operators, and the per-step perception graph with invariant features and a measured
`k=64` default. 183 tests.

What v1 still needs: a differentiable torsion path in Torch, a Torch port of the featuriser, the
rule itself, the von Mises loss, the rollout with the sample pool, and the dataset pipeline.
