# AANCA — handoff brief

**Written 2026-10-27 by a project-thread session with no browser tools, for a session that has them.**

AANCA = amino-acid neural cellular automaton. A per-residue neural CA for protein side-chain
repacking. This document is meant to be sufficient on its own: read it, then read the two
reference documents named in §9, and you should be able to continue without re-deriving anything.

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

### What is live now (2026-10-29, ~15:05 AEST)

| job | what |
|---|---|
| 30944453 | `v3_gpu` chain, `--split train`, MI355X bun160, segment 1, **running** |
| 30966837 | `v3_noop4` chain, `--split train --noop-weight 4`, MI355X, segment 1, pending |

Both splits are downloaded (24,710 entries). **The GPU works**: torch 2.10.0+rocm7.0 reports
`gpu : True AMD Radeon Graphics` and the live chain holds **0.91 s/update**, so 200k updates is
about 2.1 days over three chained segments. Both arms are on the ROCm partition and the account's
GPU QOS allows four jobs, so the pair costs half the allowance.

**Both arms run the fixed code**: the cluster checkout is at `5764899`, the crop/pool fix.

The no-op arm was submitted to `-p general` first and moved. On 16 CPU cores the train split runs
at **11 s/update**, not the 3.56 measured on validation, which puts 200k updates at 25 days — the
two arms would never have been comparable at a matched update count, which is the only way an arm
means anything. Its 1,250 CPU updates are kept at `runs/v3_noop4_cpu` (STOP file in place) and the
arm restarted from zero on a card; losing 1,250 of 200,000 updates was the cheaper side of that
trade.

### First real training curve — 2026-10-29, v3_gpu at 14,750 updates

The first numbers in this project produced by code without the crop/pool bug in it. **Zero skipped
updates**: the non-finite guard has not fired once in 14,750 updates on the train split, against a
crash within a few hundred before the fix.

Held-out pooled chi1 MAE (36 cases over 12 structures, disjoint from training):

| updates | 0 | 500 | 1500 | 3500 | 5500 | 7500 | 9000 | 11000 | 13000 | 14500 |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled chi1 MAE (°) | 34.68 | 36.60 | 33.44 | 31.71 | 27.59 | 25.72 | **24.73** | 24.94 | 25.22 | 24.67 |

**It learns, and then it stops.** The curve falls from 34.68° to 24.7° by update 9,000 and is flat
for the 5,500 updates since — the last ten evaluations sit between 24.65 and 25.32 with no trend.
Whatever this architecture and objective can do, most of it is done by 9k updates, which is 2.5
hours on a card. That is worth knowing before anyone budgets 200k.

Split by damage level, untrained against update 14,500:

| start | chi1 MAE (°) | rotamer accuracy (%) |
|---|---|---|
| κ=0 (uniform random chi) | 88.90 → **59.25** | 4.7 → 13.8 |
| κ=20 (moderate) | 10.67 → 9.42 | 74.2 → 77.4 |
| κ=100 (near native) | 4.46 → **5.34** | 99.8 → 98.8 |

So the rule has learned to repack a destroyed side chain and has **got worse at leaving a good one
alone**. That is the fixed-point half of the objective failing, stated in the cleanest available
form. chi3 barely moves at all (30.77 → 29.7) while chi1 and chi2 improve by 10°, which is the
expected ordering — chi3 is the least determined by local environment — but it is not improving.

Two diagnostics say the same thing from the dynamics rather than the metrics.

- **Drift from a native start** (how far the rule moves a structure it should leave alone) is 0.000 Å
  untrained and then wanders between **1.08 and 5.61 Å with no downward trend** across 29
  evaluations; the last six are 1.076, 4.667, 3.047, 2.210, 1.161, 1.677. The shakedown's 0.473 Å
  came out of the corrupted pool and was not evidence, but the effect it pointed at is real, and
  larger, on fixed code. **This is now a measured justification for the `--noop-weight` arm**, which
  the earlier one was not.
- **Mean |delta| per step** is no longer flat, which the shakedown's was. It rises to a peak of
  ~10°/step around step 7 and decays monotonically to ~3.1°/step by step 30. The rollout is
  partially contractive but **settles on a floor rather than a fixed point** — it is still moving
  3° per step when the budget runs out.

Read together: the rule is behaving like a denoiser with a nonzero noise floor rather than a map
with an attractive fixed point. The no-op weight is the first knob to try, which is why the arm
exists; if 4× does not flatten drift, the next question is structural rather than a weight.

v2 was abandoned at update 2162 — it was training (best pooled chi1 30.47° against 34.68°
untrained) but on the mismatched crops described below, so its weights are not evidence of
anything. `runs/v2_*` still exist with STOP files in them; delete the directories when the disk
is wanted.

### The crop/pool mismatch — the actual bug, found 2026-10-29

Successive samples of one entry are deliberately different crops of it. The sample pool was keyed
by structure id alone, so a state produced on crop A was drawn as the seed for crop B of the same
entry; both are `crop_size` residues, so the shape check that was supposed to catch this could
not. Coordinates for one set of residues were then paired with another set's chi atom indices,
native target and topology.

Measured as the largest distance any atom moves in a rollout, same seed and same 24 structures:
**45.8 Å before, 3.46 Å after**. A π rotation on every chi of 4HHB moves an atom 11.5 Å, so 45.8
was not side-chain motion at all. 52% of validation entries exceed the 384-residue crop, so this
was about half of every pool re-seed since the pool was written.

Two things to carry forward. The pool now stores the crop's **seed residue** (a crop is a
deterministic function of it) and the caller rebuilds the sample on it, so the pool is consulted
*before* `dataset.sample`. And checkpoints written before this carry no seeds, so their pool
entries are dropped on load — a resume from an old checkpoint quietly starts with an empty pool.

### The first night: everything failed, and the two failures were unrelated

Recorded because the diagnoses are reusable and the symptoms were both misleading.

**The GPU arm never ran a line of project code.** On bun161, `rocminfo` prints
`amdgcn-amd-amdhsa--gfx950` and torch 2.9.1+rocm6.4 in the same job reports `is_available() False`
with an empty arch list, then raises `No HIP GPUs are available` at the first `.to(device)`. A ROCm
wheel carries code only for the architectures it was built for, and HIP does not report a mismatch
as a mismatch — it declines to enumerate the device, so the message is identical to having no card.
The account's own working MI355X environment (`/scratch/user/uquadhik/.venv_esm`) runs
torch 2.10.0+rocm7.0, which is now what `bunya_stage.sh` installs; `--gres` is pinned to
`gpu:mi355x:1` so the architecture is no longer whatever the queue happens to hand out.

**The CPU arm died after 22 minutes inside scipy**, at `cKDTree`: `data must be finite`. That is a
report from the scene of the symptom. Two amplifiers stand between it and the cause, and both are
now closed: `clip_grad_norm_` divides by a total norm that is itself NaN, so one non-finite
gradient writes NaN into every weight permanently; and the sample pool stores states the rule
produced, so a NaN written there is drawn as a *seed* later and the update that fails is not the
update that erred. Non-finite updates are now skipped and counted, the pool refuses a non-finite
state, and `--max-skipped` stops a run that is only skipping. **What produces the first NaN is
still open** — the guards turn it into a reported update number instead of a scipy traceback, which
is the next thing to read.

### Queued on 2026-10-29 at ~01:00 AEST (all three failed; see above)

| job | what | depends on |
|---|---|---|
| 30879935 | `bunya_gpu_check.slurm` — does the ROCm wheel run on gfx950 | — |
| 30879610 | fetch, validation split (949 entries) | — |
| 30879747 | fetch, train split (23,761 entries) | — |
| 30880356 | `v1_gpu` chain, defaults, 200k updates | afterok:30879610 |
| 30880357 | `v1_noop4` chain, `--noop-weight 4`, CPU | afterok:30879610 |

Both chains run on `--split validation` (the only split downloaded when they were queued), which
makes them a **development signal and never a benchmark result**; the train split is downloading
for the run that is.

The pair differs only in the no-op weight, which is the knob the shakedown pointed at: drift from a
native start rose 0.000 -> 0.473 A in 40 updates, i.e. the reconstruction term beating the term
that asks for zero output on an undamaged structure.

**Three things that will bite a later session.**

1. **CPU generation.** Bunya has a software tree per generation and the venv is built for `epyc5`.
   A job on an epyc3 node dies in two seconds with `Illegal instruction (core dumped)` before
   running any project code. `bunya_submit.sh` passes `--constraint` automatically; a bare `sbatch`
   of `bunya_fetch.slurm` needs `--constraint=$(cat .venv_bunya/arch.txt)`. A second architecture
   means a second venv (`AANCA_VENV=.venv_bunya_epyc3 bash scripts/bunya_stage.sh`).
2. **Stopping a chain is not `scancel`.** `touch runs/<tag>/STOP` first, or the segment exits 64 and
   the next one is submitted. Delete the file to resume.
3. **The OnDemand terminal can be dead while looking alive.** Keystrokes went nowhere and the
   screenshot still showed prompts; the fix was a new terminal from the command palette. Probe
   every terminal with a write-then-fetch before trusting it. A second way to lose a command:
   `computer{type}` with a trailing newline does **not** submit it in this xterm — the text piles
   up on one line and three commands concatenate into nonsense. Send an explicit `key: Return`.
   And check which terminal has focus: the pane may be showing one the user is running `grep` in,
   where your keystrokes become that process's stdin.
4. **`pip install torch` does not replace an existing torch**, so restaging against a different
   ROCm index prints "requirement already satisfied" and changes nothing. `bunya_stage.sh` now
   passes `--upgrade`; the symptom was a staging log printing the version it was meant to replace.
