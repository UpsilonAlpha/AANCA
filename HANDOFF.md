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

### What is live now (2026-10-31, ~04:05 AEST)

| job | what |
|---|---|
| 31148624 | `v4_gpu` — **COMPLETED**, 60,000 of 60,000, exit 0, 2026-10-30T22:50 |
| 31148638 | `v5_native` — **COMPLETED**, 60,000 of 60,000, exit 0, 2026-10-31T03:42 |
| 31182705 | `aanca-bench` — CASP13 on `runs/v5_native/best.pt`, 100 steps |

Both training arms are done. The QOS wait cleared the same evening they were
resubmitted: `v4_gpu` ran its last segment in 7:22, `v5_native` in 12:12.

Both were stopped with STOP files at 15:11 on 2026-10-30 and resubmitted on
the device neighbour path; the switch cost roughly half a day of queue,
because releasing two slots at once let the `s39fold` array — Ulban's own, on
the same GPU QOS — take them within two minutes. Switching one arm at a time
would have kept a slot held throughout and is the better sequence next time.

### The two arms, both at 60,000

**Not a matched comparison**: `v5_native` is a warm start from `v4_gpu`'s
weights at update 28,404, so it carries that training on top of its own
counter. Read it as "does native-in-pool move anything", not as an
equal-budget contrast.

| | chi1 k=0 | acc k=0 | chi1 k=100 | acc k=100 | pooled chi1 | drift (mean) |
|---|---|---|---|---|---|---|
| `v4_gpu` | 38.17 | 33.2 | 4.63 | 95.8 | 16.83 | 0.0896 A |
| `v5_native` | 33.47 | 36.3 | 4.27 | 99.5 | 15.20 | 0.0400 A |

The full final evals are in `docs/benchmarks/v5_final_eval.txt`.

Native-in-pool moves every column the right way, and the two it moves most are
the two it was aimed at: the near-native case is now essentially solved
(95.8 → 99.7 % accuracy) and the fixed point holds roughly three times tighter
(0.090 → 0.031 A mean drift). Supervising the fixed point as an ordinary
training case does what the no-op term never could.

What it does **not** fix is the deficit that matters. From fully randomised
chis the rule still lands at 33.5 deg chi1 and 36.3 % accuracy — far from
DiffPack's 15.35 deg on CASP13, and the gap is too large to be explained by
the different target sets. That is the open scientific question, and the
benchmark harness (§ `scripts/benchmark.py`) exists to put a number on it on
the published targets rather than on this held-out split.

### Tier 1: the CASP13 table — 2026-10-31

`runs/v5_native/best.pt` (update 59,500), 79 CASP13 targets, 100 steps, every
chi randomised. Full output in `docs/benchmarks/casp13_v5_native_bench.log`;
the baselines' rows are their published numbers, and
`docs/benchmarks/casp13_v5_native.log` is the run of
`scripts/validate_benchmark.py` that recovers them from their own released
structures, which is what makes the comparison a comparison.

| | chi1 | chi2 | chi3 | chi4 | acc all | core | surf | RMSD all |
|---|---|---|---|---|---|---|---|---|
| **AANCA (ours)** | **31.61** | **34.25** | **49.33** | **53.07** | **42.6** | **54.2** | **33.4** | **1.037** |
| scwrl | 27.64 | 28.97 | 49.75 | 61.54 | 56.2 | 71.3 | 43.4 | 0.934 |
| faspr | 27.04 | 28.41 | 50.30 | 60.89 | 56.4 | 70.3 | 43.6 | 0.910 |
| rosetta_packer | 25.88 | 28.25 | 48.13 | 59.82 | 58.6 | 75.3 | 35.7 | 0.872 |
| dl_packer | 22.18 | 27.00 | 51.22 | 70.04 | 58.8 | 73.9 | 45.4 | 0.772 |
| AttnPacker_PP | 18.92 | 23.17 | 44.89 | 58.98 | 62.1 | 73.7 | 47.6 | 0.669 |
| diffpack | 15.35 | 19.19 | 37.30 | 50.19 | 69.5 | 82.7 | 57.3 | 0.579 |

FlowPacker is kept out of the table on purpose: MLSB 2024 Table 1 is 20
CASP13 targets from the Prediction Center with per-chi accuracy, which is
neither this target set nor this definition. Its numbers, for reference:
chi MAE 16.44 / 24.38 / 42.82 / 52.16, RMSD 0.674 / 0.388 / 0.865.

**Audited columns only.** chi MAE, accuracy all/core/surface and RMSD all
reproduce from the baselines' own structures; RMSD core and surface do not,
by +0.10 to +0.17 A in the same direction for every method, so ours there is
not their measurement and must not be quoted as a contrast. AttnPacker's
re-scored row is not an audit either — its released structures label
equivalent methyls arbitrarily.

**Where we stand.** Last, on every audited column. chi1 is about twice
DiffPack's and accuracy is 42.6 % against 69.5 %. We are also behind SCWRL
and FASPR, which is the line that matters: a method that cannot beat a 2009
rotamer-library packer is not yet a packer.

**But the shape of the row is the interesting part.** Everyone else's error
climbs steeply from chi1 outwards — DiffPack 15.4 -> 50.2, SCWRL 27.6 ->
61.5 — because an error at chi1 swings everything distal to it. Ours is
nearly flat: 31.6 -> 53.1. We are *worst* at chi1 relative to the field and
roughly *competitive* at chi3 and chi4, beating SCWRL, FASPR and DLPacker on
both and every method except DiffPack on chi4. A local refiner that cannot
choose a chi1 well but does not compound once the well is fixed would look
exactly like this. It is the same reading the k ladder gives, from an
independent direction, on the published targets.

### What the chi1 deficit is made of — measured, 2026-11-01

`scripts/chi1_histogram.py` over the same 79 CASP13 targets, 17,979 residues
with a chi1. Full output in `docs/benchmarks/chi1_histogram_v5_native.log`.

| | predicted | native |
|---|---|---|
| distance to the nearest canonical well | 12.82 deg mean, **7.90 median** | 9.66 deg mean, **7.17 median** |
| predicted well is the native one | **77.3 %** | — (chance 33 %) |
| \|error\| when the well is right (13,903 res) | **10.93 deg**, 84.5 % within 20 | — |
| \|error\| when the well is wrong (4,076 res) | **102.15 deg** | — |

**The answer is "wrong well", and the averaging hypothesis is dead.**
Predictions sit on the rotamer wells, at a median 7.9 deg from one against
the natives' 7.2, and the predicted histogram has the same three-peak shape
as the native control. There is a modest excess of mass in the non-rotameric
valleys — it is the whole of the gap between the 12.82 deg mean and the 7.90
median — but it is a tail, not the distribution.

The failure is a **minority mode-selection error with clean consequences**:
on 22.7 % of residues the rule commits to a real rotamer that is not the
native one, and is then ~102 deg out, which is one well over. On the other
77.3 % it is an excellent refiner, 10.9 deg mean and 84.5 % within 20.

The decomposition reconstructs the published number exactly, which is the
check that it describes the same thing: 0.773 x 10.93 + 0.227 x 102.15 =
**31.6 deg**, against the 31.61 the CASP13 row reports.

**Two earlier readings in this file were wrong and are withdrawn.** The
arithmetic that argued for averaging — "if the 36.3 % correct were the
right-well fraction, a clean swap would give 80 deg MAE, so the failures must
average 50" — conflated two different quantities. The 36.3 / 42.6 % figure is
*per-residue accuracy over every chi*, the published definition, so a residue
with a perfect chi1 and a sloppy chi3 counts as wrong. The chi1 right-well
fraction is 77.3 %, not 42.6, and with the real number the clean-swap model
reproduces the MAE to two decimal places. The lesson generalises: do not do
arithmetic across two different denominators.

**Where the gap actually is.** Hold our within-well precision fixed at 10.93
deg and ask what wrong-well rate would give DiffPack's 15.35: about **4.8 %**.
So the whole of the chi1 deficit is one number — they get the well wrong
about one time in twenty and we get it wrong about one time in four — and
nothing about our refinement is the problem.

**What that rules in and out.**

- **Rotamer-library initialisation is not the fix.** The rule already lands
  on wells; handing it one does not tell it which.
- **Sampling from the predicted von Mises instead of taking mu** is no longer
  motivated as an antidote to averaging, because there is no averaging to
  cure. It stays interesting for a different reason: it makes best-of-N
  possible, and with 22.7 % wrong-well a scoring function over a handful of
  samples could recover much of the gap. That is a search strategy, not a fix
  to the rule.
- **Capacity and data move to the front.** Picking the right well from a
  randomised neighbourhood is a conditional-classification problem, and
  200,268 parameters trained on 400 structures is a thin model of it. This is
  now the cheapest hypothesis left standing.
- **Co-solving may still be the mechanism.** From randomised chis every
  neighbour is also wrong, so an early commitment propagates. Whether the
  22.7 % is cooperative — wrong-well residues clustered next to each other —
  is a measurement, not a guess.

**Open measurements, cheapest first.** Does the wrong-well rate move with
rollout length (100 vs 400 steps)? If not, it is capacity, not convergence.
Is the 22.7 % concentrated by burial, by residue type, or spatially clustered
with other wrong-well residues? And does a scaling probe — more structures,
same architecture — move it?

**Should the benchmark start from fully randomised chis?** Yes. It is what
DiffPack and AttnPacker report, and the measurement above shows the task is
being attempted honestly: the rule produces rotameric structures, it just
picks the wrong rotamer too often. Changing the initialisation now would hide
the one number that matters.

`v4_gpu`: `--split train --updates 60000 --eval-every 500 --steps 64 --steps-max 96 --lr-drop
30000 45000 --lr-gamma 0.3 --init-from runs/v3_gpu/last.pt`.
`v5_native`: the same, plus `--native-probability 0.2 --noop-weight 0`, from `v4_gpu`.

`v4_noop4` (30976062) was **stopped** at ~9,900 updates with a STOP file, once the probe showed
its extra term is vacuous as supervision and harmful as a gradient. Kept on disk at
`runs/v4_noop4` as the evidence.

Every generation until now has been the same two arms, differing in `--noop-weight` alone: `v1` on
the validation split (both died overnight, 30880356 / 30880357), `v2` on train (abandoned at
update 2162 as crop-contaminated, 30926053 / 30926033, resubmitted once by the `scancel` trap as
30944359 / 30944360), `v3` on train with the crop fix in (stopped at 20,000 updates with STOP
files, kept at `runs/v3_*`), `v4` with the learning-rate schedule and the longer rollout. **v5 is
the first generation where the arms differ in something other than that weight**, and the first
where the control is a run already in flight rather than a second submission.

### Joint packing and inverse folding — evaluated 2026-11-01

Ulban's proposal: add an amino-acid identity field to the cell state and let the
same local rule solve identity and conformation together, so packing informs
sequence design and sequence design informs packing. The prior-art artifact
("Protein NCA Prior Art", §"The stage to insert next, which is nearly free")
already names this as the next rung after repacking, in the parallel-update
form. What follows is what the chi1 measurement changes about it.

**The architectural claim holds, and it is stronger than it looks.** Node
features are already `residue one-hot (20) + chi sin/cos (8) + chi mask (4)`
(`graph.py:382`), and the docstring there says the quiet part out loud:
"Residue identity is an *input*: this is a repacker and the sequence is known."
The identity field exists; it is re-clamped from the native sequence every step.
Making it mutable is a change to *whether we write that channel back*, not a new
representation. The output head costs ~20x96 weights against 200,268 — under
1 %. Diffusion-over-frames has to design a joint noise process over a
categorical and a continuous manifold with a shared schedule (this is most of
what MultiFlow does); a per-residue state vector gets joint generation for
nothing. That asymmetry is a real argument for the architecture and belongs in
the paper.

**What is not free is the chi topology.** `chi_mask`, `CHI_ATOMS` and
`SIDECHAIN_BONDS` are all functions of the residue name, and
`apply_chi_deltas` rotates a fixed atom graph. Mutable identity means a mutable
atom graph and a mutable number of chis. That is the bulk of the work, and it is
a refactor rather than a head swap.

**The soft/hard dilemma, which has no free answer.**
- *Mean field* — carry a 20-simplex and pack all identities weighted by
  probability. This is exactly Koehl & Delarue's SCMF, which the artifact
  already flags as our true precedent, and it reintroduces SCMF's known failure
  (averaging washes out strong pairwise coupling) in identity space. Note the
  irony: we just **measured** that no such averaging happens in chi space. Adding
  a soft identity field invites it back through a different door.
- *Hard assign* each step and pack that residue. The rollout becomes Gibbs with
  random scan; BPTT needs straight-through or Gumbel-softmax. Standard, but it
  changes the training story from "differentiable fixed point" to "stochastic
  fixed point".
- *Decoupled read-out* — predict identity without feeding it back to geometry.
  Genuinely nearly free, but it is not the joint model: the promised benefit
  requires the feedback.

**Sequential N-to-C is the wrong update order.** A fixed sweep direction is a
bias the weights can exploit, it breaks the permutation symmetry that makes the
rule a field, and it makes the fixed point order-dependent. The stochastic
update mask already in the design (p~0.5 per residue per step) is Gibbs with
random scan and is strictly more general. It also protects the one thing the
CASP13 row shows we do well: our chi1-to-chi4 error is flat (31.6 to 53.1) where
everyone else's compounds, because all chis are solved at once. Sequentialising
re-imports the compounding the architecture currently avoids.

**The expected gain is a consistency gain, not an information gain.** The physics
behind the proposal is sound and has 25 years of precedent — fixed-backbone
design in Rosetta co-optimises identity and rotamer because whether a position
tolerates a TRP depends on whether the indole fits. But at design time there are
no native side chains to condition on. A side-chain-aware inverse folder
conditions on *its own* predicted side chains, which are a near-deterministic
function of the backbone and sequence it already has. So the model gains no new
information over ProteinMPNN; what it gains is the ability to reject an identity
whose rotamers cannot be packed. That is worth having, and it makes a falsifiable
prediction: **the improvement should concentrate at buried, tightly packed
positions and vanish on the surface.** Report it split that way or the claim is
untestable.

**Native sequence recovery is the wrong headline.** PottsMPNN deliberately traded
NSR down (~44 % against ProteinMPNN's ~58 %) for better refolding
self-consistency and ddG correlation. Beating 44 % on NSR could just mean we are
a worse-calibrated ProteinMPNN. Commit instead to (a) self-consistency — design,
fold, RMSD — and (b) **joint recovery**: sequence recovery and chi accuracy from
a jointly scrambled start. Nobody reports (b), because nobody's model does both
at once. That number is ours.

**Sequencing argument against doing this next.** Capacity is a live hypothesis
for the 22.7 % wrong-well rate. ProteinMPNN spends ~1.7 M parameters on sequence
alone; we have 200,268 on packing. Loading a second task onto the same trunk
before knowing whether the trunk is already saturated would confound both
results.

**Cheap probe that tests the premise first.** Freeze the v5 trunk, train a linear
96-to-20 head on the converged hidden state, and measure native identity
recovery. Well above background composition means side-chain-solving
representations already carry identity information and the joint model's premise
is confirmed before anything is built. At background, the premise is in doubt.
One CPU job, no training of the rule.

**The frozen-trunk identity probe is withdrawn as "nearly free" — it is
confounded.** The section above proposes it as the cheap go/no-go for the joint
packing-plus-inverse-folding idea: freeze the v5 trunk, train a linear 96-to-20
head on the converged hidden state, measure native sequence recovery. On trying
to write it, that is not a test of anything.

The rule's node features *are* `residue one-hot (20) + chi sin/cos (8) + chi
mask (4)`, and every hidden state is computed from
`cat([node_features, hidden])`. So residue i's hidden state has been handed
residue i's identity directly, at every step. A linear probe would recover it
near-perfectly by reading back its own input, and the number would mean
nothing. Worse, the leak does not close by zeroing the one-hot: the residue's
own side-chain *geometry* names it too -- a leucine has leucine's atoms, and
`chi_mask` alone distinguishes most types.

The cheapest honest version needs both halves masked: zero the subject's
one-hot **and** truncate its side chain to CB, for a random subset of
residues, then ask whether the environment lets the hidden state recover the
identity. That is masked-residue prediction, i.e. inverse folding, and it is
not a probe of the existing checkpoint -- the rule has never seen an
unlabelled, side-chain-less residue, so it would have to be *trained* with
masking for the result to mean anything. Side-chain truncation is still much
cheaper than the full mutable-identity refactor, since atoms only ever get
deleted and never added, so this is the right first rung. But it is a training
run, not an afternoon.

Recording this because the earlier estimate was told to Ulban as "one CPU job,
no training of the rule", and it is wrong.

### Is the 22.7 % a failure at all? — measured 2026-11-01/02, and it is

Ulban's objection to the framing above: if the rule is learning a forcefield,
then rolling downhill to the nearest well from a randomised start is *correct
behaviour*, and the wrong-well rate measures the basin structure of the energy
landscape rather than an error. The test is the one the prior-art artifact
called the part of this work that is ours ("Is it actually a fixed point?" —
convergence, stability, basin size), and `scripts/basin_probe.py` runs it.

#### Read this before the rest of the section

The entries below are in the order they were measured, across one night, and
**three of them were corrected by later ones.** The corrections are at the end,
under `stationary` and under the by-type table. If you read only the early
entries you will carry away two wrong conclusions. In short:

| what the section concludes | where |
| --- | --- |
| The benchmark's 22.7 % is a genuine wrong answer, not a basin artefact. The uniform start lies far outside the native basin, and at kappa 0 the starting conformation explains only ~8 % of the error. | `descent` |
| The rule is a fixed point: converged by step 50, stable to 1000 steps, no oscillation. The steps probe is answered flatly. | `converge` |
| But the landscape is **not** single-attractor. A narrow native basin, holding to roughly 12-24 deg of chi1 noise, sits inside a dominant basin everything further away falls into at 77-78 %. | `sweep` + `stationary` |
| The native structure **is** nearly a fixed point: 0.76 % of wells broken, 2.38 deg MAE. `--native-probability` worked. | `stationary` |
| The error is concentrated in the small polar types, but that shape belongs to the *problem*: SER is worst and PHE best for every published method too. Our own deficit is broadly uniform at 1.1-1.3x, so no single missing term explains it. | `breakdown` + by-type |
| Wrong wells cluster spatially, and it survives burial stratification. Correlational; `clamp_test` is the interventional form. | `breakdown` |
| The co-solving is real: varying only whether pinned neighbours hold *correct* conformations moves free-residue accuracy by 18 points. But the sign is backwards -- a native-frozen neighbourhood makes the free residues **worse**, because a spatially mixed state is off the training distribution. | `clamp` |
| The attractor is start-independent (modal vote over 8 seeds: +0.4 points), and unanimity across seeds is a free confidence signal: 83.3 % right where 8 agree, 47.0 % where they do not. | `bestof` |
| The rule already reads hydrogen-bond acceptor geometry. SER is under-determined, not invisible. | `hbond` |

**The three corrections, so they are findable rather than buried:**

1. *"The native structure is not a fixed point of the rule"* — written under
   `sweep`, from its kappa 16 row. **Wrong**, over-read; see Correction 1 under
   `stationary`.
2. *"One attractor per residue, reached from anywhere"* — written under
   `descent`'s "What this settles". **Too strong**; true at kappa 0 only. See
   Correction 2 under `stationary`.
3. *"Specifically weak on the rigid rings"* — commit 8c62a68, reading ratios
   against FASPR. **Mostly dissolves** against SCWRL. See the by-type entry.

A fourth correction, to the *previous* section rather than this one: the
frozen-trunk identity probe is withdrawn as confounded. That is written up
immediately above this heading.

**The reading does not hold.** The starting conformation explains about 8 % of
the wrong-well rate. Two of the four measurements are in
(`docs/benchmarks/basin_descent_v5_native.log`, 31329786, 14:39;
`basin_converge_v5_native.log`, 31329822, 12:35); `sweep` (31329787) and
`bestof` (31329821) were still running.

**descent — the start barely matters.** 17,979 residues, the same set and
protocol as the histogram, which it reproduces (77.3 % well-right).

|                                | measured | pure descent | pure classification |
| ------------------------------ | -------- | ------------ | ------------------- |
| start already in native well   | 33.1 %   | 33.1 %       | 33.1 %              |
| rule's answer right            | 77.3 %   | 33.1 %       | —                   |
| stayed in the well it started  | 34.7 %   | ~100 %       | 33.1 %              |

33.1 % is not a guess for the last column: under independence
P(stayed) = sum_w P(start=w)·P(pred=w), which is 33.1 % for these marginals.
The measured 34.7 % is **1.6 points** above it. The transition matrix says the
same thing — every row is the rule's own output distribution, whatever the
start:

      start \ predicted        g-       t      g+       n
      g-                   56.5%   30.7%   12.8%     5919
      t                    54.2%   31.4%   14.3%     5937
      g+                   53.0%   30.1%   16.9%     6123

Those rows average to g- 54.55 / t 30.73 / g+ 14.69, which is the histogram's
predicted-well distribution (54.6 / 30.7 / 14.7) to two decimals — an
independent cross-check that the two runs measure the same object. The residual
diagonal lift is about +2 points per row.

So the rule is **not** descending from where it was dropped; it recomputes the
answer from the environment. Right when the start was already right: 79.1 %.
Right when the start was wrong: 76.4 %. A perfect start would move the error
from 22.7 % to 20.9 %, so **92 % of the wrong-well errors are not attributable
to the starting conformation.**

**converge — it is a fixed point, and that is a real result.** 8 targets only
(1,367 residues, 83.7 % well-right), so the absolute numbers are not comparable
to the published row — an easier subset. The *shape* is the finding.

       steps   well right     chi1 MAE   drift from step 100
          10        50.4%        67.69                63.25
          50        82.7%        25.67                 4.44
         100        83.7%        24.66                 0.00
         200        83.9%        24.39                 2.38
        1000        83.8%        24.35                 4.73

The proposed |delta chi| peaks at 8.5 deg around step 10-20 — the hidden state
starts empty, so the rule is nearly silent at step 1 (2.6 deg), builds
perception, acts hardest, then settles — and decays to 0.18-0.44 deg from step
200 on, with a largest step-to-step change in the last 100 steps of 0.089 deg.
Not the oscillation some of Grattarola's morphogenesis rules show. Drift from
the step-100 structure is 4.73 deg after ten times the benchmark's rollout, and
**decelerating**: 0.024, 0.0065, 0.0022, 0.0014 deg/step over the four
intervals. Saturating, not diverging.

Two honest caveats. The residual update sits on a small **non-zero** floor
rather than reaching zero, and ticks up slightly after step 400 (0.181 to
0.435) — a slow wobble of a few tenths of a degree, worth re-running at 10,000
steps before the stability claim goes in a paper. And 8 targets is a thin base
for it.

**What this settles, and what it moves.**

- **The forcefield defence is closed.** ~~There is one attractor per residue and
  the rule reaches it from anywhere in the uniform start distribution.~~ That is a
  *better* dynamical claim than basin-dependence would have been, but it means a
  wrong well is a wrong answer, not an unlucky draw.
  **CORRECTED — see Correction 2 under `stationary`.** The struck sentence holds
  at kappa 0 only. There is a narrow native basin as well. The surviving part is
  the last clause: the benchmark's start is outside that basin, so its wrong
  wells are wrong answers.
- **The steps probe is answered in passing, and it is the flat answer.** 100
  against 400 steps moves well-right by 0.2 points. The benchmark is already run
  past convergence, so the deficit is not a convergence artefact — which, per
  the section above, leaves capacity and data at the front.
- **The clamp test is now the decisive experiment.** `descent` shows the rule
  ignores its own starting chi; it does *not* show whether it reads its
  neighbours' side chains. Those are different claims and only the clamp test
  separates them. If a half-native neighbourhood does not move the wrong-well
  rate, the rule is a per-residue marginal predictor and the co-solving is
  decorative — and note that 77.3 % is about what a backbone-dependent rotamer
  library gets from backbone alone.
  **ANSWERED, and not with either branch above** — see the `clamp` section.
  The rule does read its neighbours' side chains, strongly; what it cannot do
  is use them when they are correct and it is not.
- **A prediction for `bestof`, recorded before it lands.** If the attractor is
  environment-determined, independent seeds should largely agree: "any seed
  right" modestly above 77.3 % (the update-mask stream still differs between
  seeds) and unanimity high. A much larger figure — say 95 % — would contradict
  `descent` and one of the two would have to be wrong.
  **The prediction held**: 84.4 % and 83.5 %. See the `bestof` section.

**sweep — and here is the thing neither of the other two could see.** The rule
relaxes to its own answer *from both directions*, including downhill.

       kappa   start right   rule right    start MAE    rule MAE
       16.00         99.5%        97.7%       11.59        8.72
        8.00         98.0%        96.4%       16.59       10.36
        4.00         92.5%        92.8%       24.40       14.37
        2.00         78.2%        83.6%       38.38       24.57
        1.00         59.5%        78.6%       57.80       30.44
        0.50         46.9%        78.0%       72.76       30.99
        0.00         33.1%        77.3%       90.42       31.61

Read the two percentage columns against each other. Below kappa 2 the start
varies enormously — 78.2 % down to 33.1 % — and the answer barely moves: 83.6,
78.6, 78.0, 77.3. **The basin is as wide as the measurement can show.** That is
`descent` again as a dose-response curve, and it closes the initialisation
question for good: no starting distribution recovers the 22.7 %.

The new part is the top of the table. At kappa 16 the rule is handed a start
that is **99.5 %** right in well terms and returns **97.7 %** — it breaks wells
it was given. Same at kappa 8 (98.0 to 96.4). The crossover is near kappa 4.
~~So the native structure is **not a fixed point of the rule**~~, and `rule right`
is not a floor the rule climbs to but an attractor it relaxes onto from either
side.

> **CORRECTED — see Correction 1 under `stationary`.** From the *exact* crystal
> structure the rule breaks only 0.76 % of wells and moves chi1 by 2.38 deg, so
> native very nearly is a fixed point and `--native-probability` worked. The
> 1.8 points lost here is the edge of a narrow native basin being felt at
> 12 deg of noise, not a rule that cannot sit still. Everything below about
> `rule MAE` improving at every level still stands, and so does the training
> recommendation — it is just worth less than this entry claimed.

Note that `rule MAE` improves at every single level, 11.59 to 8.72 included.
Both are true and they are not in tension: the bulk of residues tighten toward
their well while a small tail is pushed across a boundary. An average over
angles cannot show that and the well-match rate can.

**This is a training finding, not a capacity one.** v5 set `--noop-weight 0`
and leaned on `--native-probability 0.2` to supervise stationarity, after
`scripts/noop_probe.py` showed the no-op term was vacuous — evaluated at step 0,
where the hidden state is empty and the rule proposes ~0.04 deg whatever it is
shown. The kappa 16 row says that substitution did not fully take. It is also
the cheapest thing on this list to act on, since it is a loss term rather than
an architecture.

**breakdown — where the 22.7 % lives, and it does cluster.** By burial, core
14.9 % wrong against non-core 26.3 %: the rule is *better* where there is more
environment to read, which is the direction co-solving predicts and also the
direction every packer reports. By type the split is steric against polar —
SER 43.7 %, GLU 33.5 %, ASP 28.5 % at one end; PHE 9.2 %, TYR 12.7 %, ILE
13.0 %, LEU 13.1 % at the other. SER and THR together are **24.3 % of all chi1
error from 16.4 % of the residues**; GLU, ASP, GLN and ASN are another 31.3 %.

The spatial statistic survives burial stratification:

      stratum     self wrong   self right   difference        n
      all              25.4%        20.0%        +5.5    17977
      core             18.6%        14.1%        +4.6     5741
      non-core         29.1%        24.1%        +5.1    12230

Wrong-well residues really do sit next to other wrong-well residues, and it is
not burial doing it. The identity rule's null is flat in all three strata
(-0.6, -0.2, +0.6), so the statistic is calibrated. **This is correlational.**
Shared local difficulty beyond burial — an ill-resolved surface loop, a region
of odd backbone geometry — would look identical. The clamp test is the
interventional form of exactly this question, which is what makes it the one
worth waiting for.

**hbond — my own hypothesis, refuted in the direction I did not predict.** The
obvious reading of SER 43.7 % was that a hydroxyl's chi1 is set by hydrogen
bonding and nothing in the feature set names a hydrogen bond. Counting the N
and O atoms of other residues within 3.5 A of the atom chi1 actually places
(`OG` for SER, `OG1` for THR) says otherwise: among SER and THR the error
**falls** as partners crowd in — 37.6 % at zero, 33.9 % at one, 30.5 % at two
or more. The rule is already reading acceptor geometry and more constraint
helps it. The control moves the other way, non-polar tips going 19.2 to 21.2 to
35.1 %, so this is not an artefact of the partner-count variable. SER is not a
hydrogen-bonding blind spot; it is small and under-determined, and its worst
cell is the 797 residues with no partner at all — 7.4 % of all chi1 error from
4.4 % of residues, where the crystal angle is partly arbitrary and partly set
by water we do not model.

**Which made the ownership question answerable with no rollout at all.**
`scripts/validate_benchmark.py --by-type` scores the baselines' own released
structures for chi1 well-match per residue type, on the same target
intersection and the same 17,979 residues, so the columns are comparable to
ours rather than adjacent to it
(`docs/benchmarks/baselines_by_type_casp13.log`).

      type        scwrl       faspr rosetta_pac   dl_packer AttnPacker_      n
      SER        39.1%       39.1%       37.8%       29.9%       26.9%     1553
      GLU        31.1%       29.6%       30.0%       24.5%       20.3%     1252
      ASP        22.6%       22.9%       22.7%       16.7%       13.7%     1209
      LEU        12.9%       11.9%       11.6%        9.2%        7.0%     2112
      TYR        10.8%        9.0%       11.3%        5.9%        4.8%      895
      PHE         8.4%        5.7%        7.8%        4.7%        3.1%      943
      all        20.1%       19.5%       18.9%       15.0%       14.6%    17979

**The shape is the field's, not ours.** SER is the worst type for every method,
PHE and TRP the best for every method. So the profile is a property of the
problem.

**And our deficit is not localised.** Against SCWRL, the weakest baseline, the
excess reconciles to +453 residues = +2.52 points against the observed +2.6,
and no type carries more than about a sixth of it: SER +71, ASP +71, VAL +57,
then a long tail. We are level with SCWRL on LEU (1.02x), GLN (1.04x), GLU
(1.08x) and PHE (1.10x), and we **beat** it on LYS (0.92x). The clearest
relative outliers are TRP (1.73x) and HIS (1.36x), but between them they are
+50 residues of the +453.

*An earlier commit message (8c62a68) overstated this.* It read the ratios
against FASPR, which is unusually strong on PHE (5.7 %) and TRP (7.3 %), and
concluded we are specifically weak on the rigid rings. Against SCWRL that
claim mostly dissolves — PHE is 1.10x. The robust statement is that the
deficit is broadly uniform, roughly 1.1-1.3x across most types, which argues
**against** a single missing physical term and for a general capability gap:
capacity, data, or search.

Note the asymmetry in the gaps, because it sets the nearer milestone. We are
+2.6 points from SCWRL and +8.1 from AttnPacker. The classical floor is within
reach; the learned ceiling is not close.

**A real implementation gap found while reading for this.** There is **no
clash term in the v1 loss**. `training_objective` is
`reconstruction + noop_weight * noop + restraint`, where `restraint` is the
bond/angle harmonic and only fires when `bond_geometry` is on. The only
occurrence of "clash" in `losses.py` is a comment explaining that bond
geometry must be restrained lest the rule relieve a clash term — one that was
never implemented. The agreed first-experiment recipe specified "chi-angle
error plus an all-atom clash term". This is not offered as the explanation of
anything above; it is a planned component that was omitted and should go in.


**stationary — and it corrects two things written above.** Started at the
crystal structure exactly, 79 targets, 100 steps
(`docs/benchmarks/basin_stationary_v5_native.log`):

      wells broken        :  0.76%  (136 residues)
      chi1 MAE from native:   2.38 deg
      within 20 deg       :  98.5%

      proposed |delta chi| per step from native, degrees
      step    1 :   0.097      step   20 :   0.054
      step    2 :   0.206      step   50 :   0.050
      step   10 :   0.086      step  100 :   0.042

**Correction 1. The native structure IS very nearly a fixed point.** The sweep
entry above says it is not, on the strength of the kappa 16 row, and that was
over-read. Handed the crystal structure the rule breaks 0.76 % of wells and
moves chi1 by 2.38 degrees — and its proposed update decays to 0.042 deg/step,
which is *quieter than at its own attractor from a random start* (0.18-0.44
deg/step in `converge`, on an easier 8-target subset, so compare the order of
magnitude and not the digits). So `--noop-weight 0` plus
`--native-probability 0.2` did substantially work, and the claim that the
substitution "did not fully take" was wrong. It took, to within 0.76 %.

**Correction 2. There is not one attractor per residue.** The entry above says
the rule has a single attractor it reaches from anywhere, which is what
`descent` shows *at kappa 0* and which I generalised too far. Put the three
measurements together instead:

- from the exact native structure, the rule holds it (0.76 % broken);
- at kappa 16 and 8, roughly 12-17 deg of chi1 noise, the output still tracks
  the start (99.5 to 97.7, 98.0 to 96.4) rather than relaxing to 77 %;
- at kappa 4 the two effects balance (92.5 to 92.8);
- at kappa 2 and below the output saturates at 77-78 % whatever the start was,
  and `descent` confirms directly that the predicted well is then nearly
  independent of the starting well.

That is a landscape with a **narrow native basin** — it holds out to something
like 12-24 degrees of perturbation — sitting inside a **dominant basin** that
everything further away falls into. The 1.8 points lost at kappa 16 against
only 0.76 % lost from exact native is the basin edge already being felt at
12 degrees: about a point of residues that the noise left in the right well
but displaced far enough that the rule relaxed them to its own answer instead
of back to the crystal one.

**So Ulban was partly right, in a more precise form than either of us put it.**
There is real basin structure. But the benchmark's uniform start lies far
outside the native basin, which is why the 22.7 % is still a wrong answer and
not a basin artefact — `descent` establishes that at kappa 0 directly, and
nothing here weakens it.

**What this re-opens, which the histogram entry had closed.** That entry says
rotamer-library initialisation is not the fix, because the rule already lands
on wells. That reasoning is still true and the conclusion no longer follows.
If a start inside the native basin is *kept and tightened* — and at kappa 16
the MAE improves from 11.59 to 8.72 — then initialising from something with
better well selection than ours and refining with the rule is worth measuring.
SCWRL is 79.9 % right against our 77.3 %, and our within-well precision is
good (10.93 deg when the well is right, 84.5 % within 20). Whether the
composite beats either part is an experiment, not an argument, and it needs
the baselines' structures as starting coordinates rather than a random draw —
which `aanca.casp.randomised_start` makes a one-argument change. It would be a
hybrid rather than a method, and it says nothing about whether the rule alone
is good. Worth knowing anyway.

### v6 answered the clamp test, and the run was stopped at its best — 2026-11-02

`--damage-fractions` worked, and the sign flipped. Clamp on
`runs/v6_mixed/best.pt` at update 2500 (31417064), against the warm-start
baseline in `docs/benchmarks/clamp_v5_native_last.log`:

| pinned | v6 native | v6 random | v6 diff | baseline diff |
|---|---|---|---|---|
| 0.00 | 77.7 % | — | — | 77.5 % unpinned |
| 0.25 | 77.6 % | 76.6 % | **+1.0** | **−0.9** |
| 0.50 | 72.1 % | *pending* | — | −11.5 (native arm 64.2 %) |
| 0.75 | *pending* | *pending* | — | −20.9 |

Correct neighbours now *help* the free residues instead of hurting them, which
is the thing the clamp test was built to detect and the entire premise of the
fix. The half-pinned native arm went 64.2 % → 72.1 %, **+7.9 points**. And no
regression on the global task, 77.7 % against 77.5 %, which is what the
duplicated 1.0 slot in the mixture was for.

**But read the 0.25 row with the replicate next to it.** 31422867, the
`--seed 1` run of the same baseline, gives 76.9 % on the 0.25 native arm where
seed 0 gave 75.7 % — a **1.2-point seed-to-seed spread**. v6's +1.0 against
−0.9 is a 1.9-point swing, which is not comfortably outside that. So 0.25 is
suggestive and 0.50 is decisive, and 0.75 — the worst failure and the
pre-registered target — was still running when the training was stopped. Both
CPU jobs were readable mid-flight only because `bunya_cpu.slurm` already
carried `python -u`; that is the first time that fix paid out.

**The run was stopped on purpose at update 7216**, user's call, because the
held-out metrics peaked early and gave some back. Warm-started from 15.20
pooled chi1 / 33.47 kappa-0 chi1 / 36.3 % kappa-0 accuracy; best at update 3000
with 14.47 / 31.25 / 40.2 %; by update 6500 back to 14.94 / 32.43 / 39.1 %.
Still ahead of where it started, behind its own peak. `best.pt` is from update
2500, so the clamp table above is the good checkpoint, not the drifted one.
Whether the give-back is eval noise at this cadence or the mixture trading the
global task against the mixed one is not established.

### v7: the clash term in an objective for the first time — 31481510

Submitted 2026-11-02, warm-started from `v6_mixed/last.pt` at update 7216:

    bash scripts/bunya_submit.sh gpu --tag v7_clash --split train \
      --updates 20000 --eval-every 500 --steps 64 --steps-max 96 \
      --lr 3e-5 --lr-drop 12000 --lr-gamma 0.3 \
      --native-probability 0.2 --noop-weight 0 \
      --damage-fractions 1.0 1.0 0.5 0.2 0.05 \
      --clash-weight 3.0 \
      --init-from runs/v6_mixed/last.pt

**The learning rate needed no adjustment this time, and that is worth knowing
rather than assuming.** v6 was launched at `--lr 3e-5` with its first drop at
12000 and only reached 7216, so it never dropped and was still training at
3e-5 when it stopped. Passing 3e-5 again continues at the same rate. The v5→v6
trap (a warm start restarts the schedule at update 0, so the drops re-apply and
the old `--lr` is the wrong number) does not bite here *because* the run was
stopped early — check where the previous run actually got to before reusing its
rate, rather than copying the flag.

**Why weight 3.0.** The term has a known physical scale: mean angstroms of van
der Waals overlap per supervised atom, which reads exactly 0.000 on a crystal
structure and 0.049 on 1UBQ with every chi randomised. A partly-resolved
rollout state should land around 0.01–0.03, so at weight 3 the term contributes
roughly 5–15 % of a loss whose magnitude is about 0.5–0.7. Visible, not
dominant. The progress line prints the term **unweighted** on purpose, so the
readout is direct: if it falls toward zero the weight was enough, and if it sits
flat the weight is too small rather than the term being useless.

**First attempt, 31481510, died on the first training update** with
`ValueError: residue 149 (GLU) has no CA`, raised by `steric_topology` — after
the warm start had loaded and the update-0 evaluation had run (pooled chi1
14.81, correctly reproducing v6's last state, so the warm start itself was
right). **A training residue need not have a backbone.** 1UBQ and 193L are
complete, so no test in `tests/test_steric.py` could produce the case, and the
rest of the pipeline already tolerates incomplete residues — that is what
`FeatureTopology.virtual_cb` and `fallback_idx` exist for. Raising made the
steric term stricter than the model it serves.

Fixed by falling back to N, C or CB, each carrying 2.5 Å of extra neighbour-list
slack **charged per residue** rather than absorbed into a wider global cutoff,
so an ordinary pair is still scored against the 17 Å bound. A residue with no
recognisable backbone atom at all anchors on its first real atom with 6.6 Å of
slack, which keeps the pair list a superset instead of silently lossy.

**The general lesson, and `scripts/steric_preflight.py` is the response.** Two
complete crystal structures cannot generate the training set's failure modes,
so a test suite built on them certifies nothing about the real data. Finding
those modes one per GPU queue cycle — queue, wait for a card, load the warm
start, run an eval, crash on one residue — is the thing to avoid. The preflight
walks a whole split on CPU and reports every problem at once: anything that
raises, every residue with no CA, every unassignable atom name, and what each
native structure actually scores (which must be 0.000, so anything above it is
a radius or an exemption wrong on chemistry the test structures lack).
Submitted as 31482841 alongside the v7 resubmission 31482842, rather than
gating on it, since another crash costs one update and the chain stops on exit
1 anyway.

**Two things to watch on the first eval.** Whether `s/update` moved — the
steric term adds an all-pairs distance computation over every residue pair
within 17 A, once per objective call rather than per step, and v6 ran at
2.82 s/update on a 384-residue crop. And whether the global task regresses:
`build_eval_set` is untouched, so the held-out curve is still directly
comparable to v5's and v6's.

### The identity library — 54 entries, and the Gumbel temperature is not the one you think

Built 2026-11-02, `src/aanca/identity.py`, 28 tests. One joint categorical over
(type, chi1 well), derived from `rotamers.WELL_NAMES` and `rc.num_chi` rather
than hardcoded: 17 types × 3 wells + 3 chi-less types × 1 = **54**.

**PRO is a chi-less type in this codebase**, which is easy to get wrong.
`residue_constants` detects its CA–CB bond as lying in a ring and puts it in
`RIGID_RESIDUES`, so `num_chi("PRO") == 0` and its `chi_mask` is all-False.
It gets one trivial entry, like ALA and GLY. Giving it three would waste two
indices the head can never populate; dropping it would shift every index above
it. Hence 54 and not 55 or 57, and there is a test asserting the chi-less set
is exactly `{ALA, GLY, PRO}`.

**chi1 only, deliberately.** The full chi tuple is `3**num_chi` per type, so
ARG and LYS are 81 each and a complete enumeration is ~400 entries, the large
majority sterically impossible and essentially unpopulated in any real training
set — the rare ones would be learned from a handful of examples apiece. The
selector's job is the type plus the gross chi1 branch that decides where the
side chain points; chi2 onward is the von Mises head's half of the
classification-plus-regression split. If it is ever deepened it should be
deepened *per type* (chi1 × chi2 for aromatics and branched residues), not
uniformly.

**The Gumbel-softmax temperature does not sharpen the choice.** This corrects
advice given earlier in the same discussion, where "anneal the temperature to
zero so it becomes iterated conditional modes" was said without distinguishing
two different temperatures. `argmax((logits + g) / τ)` **does not depend on τ**:
the committed label is a draw from `softmax(logits)` at every temperature, and
τ controls only how much of the returned vector's mass sits on the winner —
i.e. the bias and variance of the gradient estimator. Verified directly: with
one fixed seed, τ from 0.01 to 100 gives the *identical* argmax while the mean
max component goes 0.9934 → 0.0194.

So the two temperatures are separate knobs and both are needed. The
**Boltzmann** temperature divides the logits or energies before the softmax and
is what actually sharpens the decision toward the argmax — that is
`potts.conditional(..., temperature)`, and it is what "anneal to zero → ICM"
means. The **Gumbel** τ is only the relaxation. Two consequences: annealing τ
is not annealing the sampler, and **a falling `label_churn` curve must not be
attributed to a τ anneal**, because τ does not change the distribution of
choices at all. Mixing these up would produce a convergence curve that looks
like progress and is an artefact.

Also worth keeping: `type_nll` sums over a type's wells in log space with a
`(54, 20)` membership mask and `logsumexp`, not by exponentiating and
scatter-adding. A well whose probability underflows to zero in float32 would
make its type's log read `-inf`, and a correct type whose wells are all
individually small is exactly the case that loss exists to score.

### The Potts energy — built and validated 2026-11-02, `src/aanca/potts.py`

`E(s) = Σ h_i(s_i) + Σ J_ij(s_i, s_j)`, couplings stored sparsely over an
explicit edge list with `i < j` (a dense `(n_res, n_res, 20, 20)` is 1.4e10
entries at 300 residues). `energy`, `delta_energy`, `conditional`,
`metropolis_ratio`, `gibbs_sweep`, `metropolis_sweep`, `anneal`. `temperature`
*is* kT and 0 is legal, so a quench is exact ICM. Positions are passed in by
the caller, so `update_schedule.random_independent_set` can supply them without
this module importing it.

**The validation that matters.** Against brute-force Boltzmann enumeration on a
4-position, 3-type system (81 states):

| chain | sweeps | TV distance | iid floor | 
|---|---|---|---|
| Gibbs, kT=1 | 20 000 | **0.0104** | 0.0106 |
| Metropolis, proposal (0.7, 0.2, 0.1) | 40 000 | **0.0192** | 0.0075 |

Gibbs sits *at* the iid floor, so four sequential single-site updates fully
decorrelate this system. MH is 2.5× the floor, which is the rejection
autocorrelation. **And the test discriminates**: deleting the Hastings `q`
ratio — accepting on `exp(-ΔE/kT)` alone with an asymmetric proposal — gives
TV = **0.35**, eighteen times the threshold. So it really separates a correct
MH from Gibbs with extra steps, which was the whole point of running MH from a
deliberately biased proposal. `delta_energy` max error over 400 random moves on
a 30-residue, 20-type model: 9.5e-7, and the cancellation is in the *reference*.

**Three things wrong with the spec, two of them mine and real.**

*A temperature below the dtype's smallest normal is a quench, not a NaN.*
"Subtract the max before exponentiating" is not sufficient, and neither is a
`temperature == 0` guard. `temperature` arrives as a Python float, so 1e-300 is
genuinely positive and the guard does not fire — but torch casts it to float32
for the division, where it becomes exactly zero, and the result is 0/0 at the
best type and -inf at the rest. Every entry comes back NaN at a temperature
that looks fine. A test at 1e-6 and a test at exactly 0 **both pass**, because
the bug lives in the gap between them. Fixed by treating any temperature at or
below `finfo.tiny` as a quench, which is also the honest reading: a kT below the
dtype's resolution *is* zero here.

*`metropolis_ratio(ΔE, q_old, q_new, T)` is only correct for an independence
proposal.* The general Hastings correction is `q(old|new)/q(new|old)`; it
collapses to `q_old/q_new` only because a single-site proposal depends on the
*rest* of the sequence and not on the value currently at the moved position.
That holds for a fixed proposal and for a network head conditioned on
neighbours, so the signature stands — but a future proposal that reads its own
position's current value would make the function quietly wrong.

*"Annealing finds the global minimum" is not a property of the code.* It is a
stochastic optimiser, and the hit rate tracks the energy gap and nothing else:
gaps 0.83/1.72/1.34 → 16/16 starts, gap 0.62 → 11/16, gap 0.11 → 7/16. Four
schedule shapes including 800 sweeps did not fix the small-gap case and the
longer ones were sometimes worse. The honest statement is **"annealing finds
the minimum when the minimum is identifiable"**, and the small-gap behaviour is
the method being *right* rather than failing — two sequences 0.1 apart in energy
are nearly equally probable, so picking either is correct. Which is itself an
argument for finite-temperature sampling over quenching: at kT=1 a
near-degenerate pair should be equally populated, and that is information the
quench throws away.

### Three traps paid for on 2026-11-02

**`scancel` on a training job RESUBMITS it.** This nearly happened. SLURM sends
SIGTERM, `train.py`'s handler finishes the update in flight and exits
`ckpt.UNFINISHED` = 64, and `bunya_train.slurm` treats 64 as "checkpointed, not
finished" and submits the next segment — which is correct behaviour for a
walltime and exactly wrong for a deliberate stop. The clean stop is the **stop
file**: `touch runs/<tag>/STOP`. It is checked once per update, so it lands
within one step; train.py exits 0, the chain ends, and the log says so
explicitly. Observed: `stopped at update 7216 of 20000 by .../runs/v6_mixed/STOP`,
`train exit : 0`, no resubmission. The script's own comments say all of this
(`bunya_train.slurm:18-21, 141-152`) — the lesson is to read them before
reaching for `scancel`, not that anything was wrong.

**A bare `sbatch scripts/bunya_cpu.slurm` is a coin flip.** It carries no
`--constraint`, so the epyc5 venv can land on an epyc3 node and die in two
seconds with SIGILL — **exit 132, an empty `.out`, no traceback**. The CPU
generation trap was already recorded here for `bunya_fetch.slurm`; it applies
to every bare CPU submission, and `bunya_submit.sh gpu` is the only path that
passes the constraint for you. `steric_preflight.py` died this way on bun030
while two clamp jobs submitted identically in the same hour happened to land on
bun151 and ran fine, which is exactly why it had not shown up before. Submit
as:

    sbatch --constraint=$(cat .venv_bunya/arch.txt) scripts/bunya_cpu.slurm ...

A `#SBATCH` directive cannot read `arch.txt`, so this has to be passed at
submit time; the usage header of `bunya_cpu.slurm` now says so. Worth
suspecting retrospectively — an earlier CPU diagnostic that "produced nothing"
may have died this way rather than having had nothing to say.

**The Bunya checkout does not update itself, and a new flag fails at submit.**
`/scratch/user/uquadhik/AANCA` was sitting at `a2a6111` — two commits behind
where today started and eleven behind the clash term — so
`--clash-weight 3.0` would have been rejected by argparse after the job had
queued, waited for a card and started. Nothing warns about this: the submit
wrapper reads the local tree, and a stale tree is a *valid* tree. `git pull
--ff-only` on Bunya is a step in launching any run that uses code written since
the last run, and it is worth doing even when you think you know, because the
answer here was two commits older than expected.

Pulling mid-flight was safe in this case and is not safe in general: 31417064
and 31422867 were executing from that checkout. Python had already imported
everything, so changing the `.py` files under a running process does nothing,
and every new setting defaults to off — but a job that imports lazily, or a
pull that changes a default, would not be harmless.

**A full `clamp_test.py` does not fit in `bunya_cpu.slurm`'s 4-hour walltime.**
Measured on 31417064: four rows (0.00, 0.25 native, 0.25 random, 0.50 native)
in 3 h 11 min, so **about 48 minutes per row** at 79 CASP13 targets and 100
steps. The default `--fractions 0.25 0.5 0.75` is seven rows, which is ~5.6 h —
so the job is killed with the last two or three rows missing, and 0.75, the row
with the largest effect and the one worth measuring, is always the one lost.
31422867 on `bun153` ran at ~90 min/row, nearly twice as slow on identical
work, so node contention makes it worse and the budget cannot be assumed.

Use `--fractions` to ask for one fraction at a time: 0.00 plus one arm pair is
three rows, ~2.4 h, comfortably inside the walltime. Submitted that way as
31480041 (v6 `best.pt`) and 31480042 (v5 `last.pt --seed 1`), both at 0.75. The
earlier pair was left running rather than cancelled, since the 0.50 random row
they are working toward completes that comparison and costs nothing on
`general`.

Incidentally `runs/v6_mixed/best.pt` was still dated 11:34, i.e. update 2500,
after the run reached 7216 — so 4,700 further updates never produced a new
best by the quantity it tracks. That is independent support for having stopped
it, and a caution against the earlier note that update 3000 was the peak:
whatever `best.pt` scores on did not agree.

**A `Write` that says "updated" rather than "created" means the file already
existed.** `src/aanca/schedule.py` was already the learning-rate schedule
(`learning_rate(base, update, drops, gamma)`), and writing the new coupling-graph
module to that path silently destroyed it, along with `tests/test_schedule.py`.
Nothing caught it until the full suite ran, because the eight failures were in
`tests/test_train_resume.py` — which imports `learning_rate` via
`scripts/train.py:57` — and so looked unrelated to anything new. Both files were
restored verbatim from `cb3231c` (verified by an empty `git diff`) and the new
module became **`src/aanca/update_schedule.py`**, matching the config field and
the CLI flag. Check for an existing module before writing a new one, and read
the word the tool reports back.

### Inverse folding is wired and running — v8_if, 2026-11-02

`--identity-weight` turns on the head, the ripple and the blinding together.
Built tonight, 701 tests green, launched as **v8_if (31516313)**, 24 h
segments, chained, stopped with `touch runs/v8_if/STOP`.

**The head.** One joint categorical over the 54-entry (type, chi1 well)
library, not separate type and rotamer heads. Its neutral element is **keep**,
not uniform: a zeroed head plus a keep-current logit holds the present label,
which is the delta head's identity initialisation transposed to a discrete
label. Without it an untrained rule resamples the whole sequence from a
uniform distribution thirty times over and diverges before learning anything.

**The ripple.** The committed label goes forward as a straight-through one-hot
in `node_features`' 20-dim type block. That is its only pathway, and the
straight-through form is the only reason it is *trainable* rather than merely
observable: as a hard index the chain would stop at every commitment and the
loss could never reach the head that chose the label. Its gradient horizon is
`config.truncate` -- 12 steps by default -- because the rollout detaches at
truncation boundaries. Labels are gated by the same mask as the geometry, so
with `--update-schedule independent` the sequence is being decided by a
chromatic Gibbs sweep, which is what that schedule was built for.

**Blinding has to happen to the coordinates, and that is the trap.** Zeroing
the chi channels and the side-chain edge block is not enough. Neighbour
selection and the envelope are both *minimum heavy-atom* distances, so the
graph knows how far a side chain reaches even when no feature says so, and a
rule could narrow its own identity to TRP or ARG off either. `aanca.blind`
therefore moves every atom beyond CB onto its residue's CB. `tests/test_blind.py`
asserts the whole claim in one line -- randomise every chi and the neighbour
list, envelope, edge features and node features must not change by one bit --
with a companion test that the *unblinded* envelope does change, so the first
test is capable of failing.

**The keep strength is the one setting that needed measuring.** At the default
8.0 a masked-start run spends its early updates unlearning its own prior: step
1 draws a label from nothing, and from step 2 the keep term pins 98 % on that
guess. Measured on 8 validation structures: keep 8.0 gave identity NLL
6.59 -> 6.47 over 40 updates, still far above the uninformative ln 54 = 3.99.
Keep 2.0 gave **4.02 -> 3.76 over 80 updates** -- below the floor, i.e. the
head is reading real signal. v8_if runs at 2.0. A *native*-start run wants the
strong keep, because there the stability is the point.

### Silencing confirmed: correct neighbours tell the rule it is finished — 31515295

v5_native at 0.75 pinned, with the diagnostic columns, `--no-anchor`
(2026-11-02). The prediction was that the native arm's free residues would
barely move; they barely move.

```
  pinned      arm   free right    free MAE   free n   pinned drift    start   moved   stuck
   0.75   native        54.2%       58.50     4540     0.00e+00    33.5%   46.58   68.7%
   0.75   random        75.1%       34.34     4540            -    33.5%   89.17   35.2%

  pinned     native - random      moved n/r
   0.75             -20.9              0.52
```

`native - random` reproduces the original -20.9 exactly, so dropping the
anchor row perturbed nothing. Both arms report the same `start` (33.5 %), so
they are scoring the same residues from the same conformations. And then:

- **moved n/r = 0.52.** Given correct motionless neighbours the rule moves its
  free residues *half as far*.
- **stuck 68.7 % against 35.2 %.** Two thirds of the free residues never leave
  the well they were randomly started in, against one third when the frozen
  neighbours are wrong.
- The accuracy gap is therefore not "it solved them worse". From a 33.5 %
  floor the random arm reached 75.1 % and the native arm 54.2 %: it closed
  about a third of the distance, because it stopped.

**So the rule has learned "a native-looking neighbourhood means you are
done".** The no-op term supervises silence at the native fixed point, and in
training a neighbourhood only looks native when the residue itself is also
right -- those two conditions never came apart, so the rule reads the
neighbourhood and infers its own state. It is a *correct* inference about the
training distribution and a wrong one about the clamp.

This explains the whole table's shape, including the sign flip. At 0.25 pinned
the neighbourhood does not look converged enough to trigger it; at 0.75 it
does. And it explains why `--damage-fractions` helped (-20.9 -> -11.9) without
fixing it: damage is drawn uniformly per residue, so "my neighbours are native
and I am not" stays rare at any fraction.

**The next experiment follows directly, and it is cheap.** The no-op term is
applied to a whole undamaged structure. Make it per-residue instead: damage a
small fraction, then require silence on the *undamaged* residues and motion on
the damaged ones, in the same step. That severs the correlation the rule is
exploiting at its source, rather than diluting it. `ObjectiveTerms.noop`
already computes a masked mean, so the change is the mask.

Worth noting what this does *not* say: the rule may still read neighbour
conformations perfectly well and merely be gated by a bad stopping rule. The
clamp test cannot separate those, because a silenced rule and a blind one give
the same accuracy. A per-residue no-op would separate them.

### The 0.75 clamp row — the sign is still wrong, and halving it is the result

31480041, `runs/v6_mixed/best.pt` at update 2500, CASP13, 79 targets, 100
steps, `--fractions 0.75`. This is the pre-registered target row: the worst
failure in the v5 table (-20.9) and the one a 4 h walltime lost every time the
full seven-row sweep was run.

```
  pinned      arm   free right    free MAE   free n   pinned drift
   0.00        -        77.7%       31.50    17979              -
   0.75   native        63.4%       47.48     4540     0.00e+00
   0.75   random        75.3%       34.31     4540            -

  pinned     native - random    native - anchor
   0.75             -11.9              -14.3
```

`pinned drift` is exactly zero, so the pin held and the table is valid.

**-20.9 to -11.9.** The damage-fraction fix moved the worst row by nine
points and did not fix it: with three quarters of the structure pinned, the
rule still does *worse* when the frozen neighbours are correct than when they
are random, by twelve points. That is far outside the 1.2-point seed spread
measured at 0.25, so it is not noise. The whole-table picture is now a
monotone one — at 0.25 the sign has flipped positive (+1.0), at 0.50 it is
clearly positive (+7.9), at 0.75 it is still badly negative — which reads as
"the fix works to the extent that the clamped state still resembles something
the rule was trained on, and 0.75 pinned does not".

Note the 0.00 anchor is unchanged at 77.7 %, so none of this is bought with
global accuracy.

The seed-1 replicate of the v5 row (31480042) **did not produce it**: the log
ends after the 0.00 row with no `exit` line, i.e. killed on walltime. The
0.00 row alone costs most of the budget. A replicate of the 0.75 row needs
`--fractions 0.75` *without* the anchor, or a longer card.

### The steric preflight's inventory — the fallback is load-bearing, the radii are right

31484703 on bun158, `--limit 600 --seed 0` over the train split.

- 23,761 listed, **23,760 present on disk** — one structure in the manifest
  has no file. Harmless (every consumer re-roots through `on_disk`), worth
  knowing.
- 600 sampled, **585 checked, 15 raised** — and all fifteen are the dataset's
  own filters, not the steric code: `min_residues` 40, mostly nucleic-acid
  complexes that pass the manifest's polymer-monomer count with almost no
  protein (1AMT yields 33 standard amino acids). The training loader rejects
  these too, so they are not a steric finding at all.
- **13 residues with no CA** across 585 structures. The anchor fallback added
  after v7's crash is therefore load-bearing: without it v7 would have died
  again, on a different structure, several hours in.
- **0 unassignable atom names** in 585 structures. The element assignment
  covers the real data, not just 1UBQ and 193L.
- **7 of 585 structures score above zero**, worst 0.000430 A/atom. Three
  orders of magnitude below the 1.5 A tolerance's own scale, and consistent
  with crystal strain rather than a wrong radius or a missing exemption. The
  radii and the 1-5 exemption rule are correct on real chemistry.

### The clash term is nearly inactive at the rule's own converged states

Measured 2026-11-02 from v7 (31482842) at update 125, the first time the term
had ever been in an objective: **clash 0.0007** angstroms of overlap per atom
at tolerance 1.5, against a loss of −0.628. At `--clash-weight 3.0` that is
**0.3 % of the objective** — the term was effectively switched off. Relaunched
as `v7b_clash` (31484745) at weight 50, which puts it near 5 %. Cost: 2.82 →
**3.82 s/update**, about 35 %, for one all-pairs distance computation per
objective call.

**This is evidence against the clash hypothesis for the clamp failure, and it
should be read that way.** Two explanations predicted that table: the mixed
state being off-distribution, and the absence of a steric term. If the rule's
converged states carry 0.0007 angstroms of overlap per atom then steric
interpenetration is not what it is getting wrong — there is almost nothing for
a clash term to remove. The off-distribution explanation, which
`--damage-fractions` addresses and which already moved the clamp numbers
(+1.0 at a quarter pinned, +7.9 on the half-pinned native arm), looks like the
real one.

So the honest expectation for v7b is small. The term is still worth having —
it is a correctness constraint, it is the first component of an explicit
energy, and a folding objective cannot do without it — but it is not the
second half of the clamp fix, and the number says so before a run was spent
assuming it.

Caveat on the measurement: 0.0007 is the mean over supervised atoms at the
*final* rollout state, averaged over a mixture that is 40 % globally damaged.
It does not establish that the residual overlap is uncorrelated with the
errors, only that there is little of it.

**v7b confirmed it at weight 50** (31484745, update 6250): `clash 0.0004`
against `loss -0.462`, i.e. the weighted term is about 4 % of the objective
and the overlap it has to work with is 4e-4 A/atom. Not a controlled
comparison with v7's 0.0007 — that was update 125 at weight 3, and the
weighting changes the state the term is measured at — but the absolute level
is the point, and it is the same order either way. The global task is intact
(`drift from a native start: 0.0222 A mean`) and the step cost settled at
3.59 s/update.

### Growing a protein de novo, and why it is not diffusion, 2026-11-02

User's question: this sounds similar to diffusion, which is the state of the
art. It is not the same thing, and the difference is the part worth keeping.

**Three ways they differ, and only the third is a disadvantage.**

1. *What varies along the trajectory.* Diffusion varies the **noise** on a
   fixed node set -- every residue is present at step 0 and the trajectory is
   a prescribed schedule from noise to structure. Growth varies the
   **support**: which cells exist at all.
2. *Who decides the schedule.* Diffusion's is imposed and the network is told
   where it is -- DiffPack takes the noise level as conditioning. An NCA
   cannot be told: no step index, no noise level, which is this project's
   scientific claim (§1). So a growth NCA is not a diffusion model with a
   different schedule; it has to decide locally when a neighbour wakes up,
   which is a strictly harder problem and a different one.
3. *Where the distribution comes from.* Diffusion has a tractable objective
   and a sampling procedure with guarantees. Growth has neither. This is the
   same gap as "a fixed point is not an energy" (see the Potts section): a
   grown structure is a sample from whatever the rule happens to do, with no
   density attached. For design that matters less than designability; for
   folding ddG it matters completely, which is what the Potts and MH
   machinery is for.

**What growth buys that diffusion does not.** Variable length by construction
-- a diffusion model fixes N and needs length conditioning or padding.
O(surface) work per step while the structure is small instead of O(N). A model
that can **extend** an existing structure, which is the practically valuable
case (growing a binder onto a target, closing a loop, fusing a domain) and
which diffusion reaches only through inpainting. And one rule at every stage,
because there is no t to condition on.

**How it would be built here, concretely.**

- Pre-allocate N slots, as ``aanca.blind`` and the alive-mask discussion
  above require. Carry an alive mask ``a`` in ``{0,1}^N``; hidden state exists
  for every slot, coordinates only for the live ones.
- Seed one residue (or several, which is nucleation) at a reference frame.
- Each step: the live set runs the ordinary rule; every live cell also emits
  an **activation logit** for its dormant sequence-neighbours -- one new head,
  built exactly like the identity head; a newly woken residue is placed from
  its activating neighbour's frame plus ideal bond geometry and a predicted
  phi/psi. That is the one place a torsion is unavoidable and also the one
  place there is nothing downstream of it, so the lever arm is zero at birth.
  Relaxation then continues over everything alive, which is the "grow while
  relaxing" the user asked for.
- Loss on the completed state only, with the frontier excluded -- the prefix
  trap again. The clash term stops being nearly inactive here and becomes
  essential: growing into occupied space is *the* failure mode, and it is why
  that term was built now rather than later.
- The plumbing is already most of the way there. ``supervised`` is a
  per-residue gate on the loss, ``frozen`` is a per-residue gate on geometry,
  and ``step_once`` takes both. An alive mask is a third gate of the same
  shape.
- For refinement, EGNCA's coordinate update (relative-vector sums, no
  torsions, no lever arm) is the natural operator, and DGCA is the only prior
  art for a node-adding graph CA.

**What will bite, in order:** no handedness in torsion-space growth, so
mirror folds without an explicit chirality term; no energy, so "grown" says
nothing about whether the result is a protein; and the distribution problem
above, which is the one diffusion solves and this does not.

### Growth should be spatial, not sequential — and EGNCA says how, 2026-11-02

User's correction of the section below: sequential may not be the best approach;
grow instead over the distance-based ProteinMPNN graph, the way an NCA
originally works. Evaluated against the §1 prior art, in particular
**E(n)-equivariant Graph NCA** (arXiv 2301.10497, TMLR 2024). Nothing built.

**Why synthesis order is not folding order.** Three separate reasons, and they
matter because the N-to-C proposal borrows its authority from the ribosome.

1. *Synthesis imposes a causal filter, not an order.* A contact between residue
   10 and residue 290 cannot exist before 290 is made. That constrains which
   contacts are **available** — a downward-closed set of prefixes — and says
   nothing about which available contact forms next. The order within the
   available set is the landscape's business.
2. *The landscape orders foldons by native structure, not sequence index.*
   Cytochrome c is five cooperative units whose assembly order comes from
   "the same interactions that join the foldons in the native protein", earlier
   ones templating later ones. Nothing in that ordering is sequence position.
3. *The rate comparison decides whether the filter is even active.* The HemK
   N-terminal domain folds at ~5 ms⁻¹ while translation runs at ~3.6 aa/s —
   four orders of magnitude. The chain is therefore at quasi-equilibrium given
   the residues it has, not being placed one residue at a time. And EF-G's
   domain stays unfolded until the whole domain is synthesised and *then* folds
   from the C-terminus: folding waited for everything, then ran backwards.

So an N-to-C loop hard-codes a filter we do not need (we are given the whole
sequence; we are not transcribing in real time) and on top of that imposes an
order nature does not use. Keep chain connectivity, discard the order.

**The substrate pre-exists — that is what the original NCA actually does.**
Growing NCA does not create cells. The lattice is complete from step 0 and
every cell already has its neighbours; growth is an **alive mask** (alive if
any cell in the 3x3 has alpha > 0.1; dead cells have their state vector
zeroed) spreading from one central seed. Transplanted correctly, the protein
substrate is the **N residue slots**, which we know, with sequence edges
present from step 0 and *coordinates* absent. A dormant residue is live in the
graph and in hidden-state space and has no position; it acquires one when
activated.

That is also the resolution of the chicken-and-egg: a distance graph needs
coordinates and growth produces coordinates, so a residue cannot be placed by
the graph it is about to join. Under the alive mask it does not have to be —
it is placed by messages over edges that already exist, which is exactly how a
dead NCA cell comes alive from its living neighbours.

**And then growth need not be sequential at all, which is the real payoff.**
With an alive mask the order is chosen by the rule rather than by a loop: seed
anywhere, or at several places, and expand where the living cells are
confident. That is nucleation-condensation and foldon assembly, i.e. what the
biology does. It keeps the lever-arm relief — a residue on a *spatial*
frontier has few spatial neighbours, same as one on a sequence frontier — with
no claim that growth order equals sequence order. It is also **parallel**: a
spatial frontier activates O(surface) cells per step where a sequence prefix
activates exactly one, which removes the "300 serial placements" objection in
the section below.

**What EGNCA actually establishes, and what it does not.**

- *It is not a growth model, despite the name.* Its node set is fixed
  throughout and it never adds nodes; pattern formation starts from **random
  Gaussian coordinates on a given graph** with hidden features at 1, and there
  is no seed node. So it is relaxation-from-noise on a given topology — the
  "growing" half of Growing NCA is simply absent from the graph-NCA
  literature. The nearest thing is **DGCA** (Developmental Graph Cellular
  Automata), where nodes divide on their own and their neighbours' states, and
  which explicitly contrasts itself with graph-CA work "focused on
  self-organization rather than growth ... with a fixed number of nodes". DGCA
  is discrete structure only — no geometry. So node-addition growth on a
  *geometric* graph is genuinely open, and genuinely unsolved.
- *Rebuilding the graph every step is demonstrated.* Their Boids task takes
  the graph as "a fixed-radius nearest neighbourhood of the nodes at each time
  step". That is direct evidence on §11's open design decision ("the contact
  graph changes as coordinates move"): rebuild-per-step works in an
  equivariant NCA. Their caveat is error accumulation in chaotic dynamics.
- *Their coordinate update is a third option alongside torsions and frames.*
  EGNN: `x_i <- x_i + mean_j (x_i - x_j) phi(m_ij)` — coordinates move along
  weighted sums of relative position vectors. No torsions, no frames, **no
  lever arm at all**, and E(n)-equivariant by construction. It is also the
  only one of the three validated inside an NCA. The price is that the output
  is a free 3-vector per node, so bond lengths, angles and chirality are not
  guaranteed and need geometric terms — we have the restraint machinery and
  now the clash term — and it abandons our present invariant-features choice.
- *Their stability apparatus is ours already*: pool replay, replacing the
  highest-loss pool state with a fresh one, local and global perturbations,
  15-25 step BPTT, PairNorm/NodeNorm with tanh and skip connections. No formal
  guarantee; stable attractors reported past t = 15; exploding gradients
  reported as the main training difficulty.
- *One of their limitations does not bite us, which is a point in favour.*
  Their pattern loss needs a 1-to-1 correspondence between initial and target
  nodes, which makes it non-permutation-invariant and causes abrupt early
  dynamics; they propose optimal transport. For a protein the correspondence
  is free — residue *i* is residue *i*.

**The supervision trap survives the change, and we already own the fix.** A
growing spatial ball has the same ill-defined native as a growing sequence
prefix, and it is literally the same problem as the crop boundary shell:
`rollout.training_objective` excludes that shell because "its native torsions
depend on contacts that were cropped away and asking for them is unlearnable
rather than merely hard". A grown spatial ball **is** a crop, and
`RepackingDataset.load_crop(i, seed_residue=...)` already grows one from a
seed residue. The machinery for "spatially grown subset with a frontier"
exists in this repo.

**If and when this is built**, the shape is: alive mask over pre-allocated
residue slots, seeded spatially rather than at the N terminus, graph rebuilt
per step (EGNCA's Boids precedent), coordinates by EGNN-style relative-vector
updates rather than torsions, frontier shell excluded from supervision. It
sits downstream of the backbone escalation, which is downstream of inverse
folding.

### Growing the backbone N to C — what it fixes and what it breaks

User's suggestion, recorded 2026-11-02. Nothing built.

**It dissolves the lever arm by construction, which is the real attraction.**
The objection to backbone torsions is that rotating phi at residue *i* moves
every atom from *i* to the terminus. In a *growing* chain the newest residue
has nothing downstream of it, so placing its phi/psi is a bounded local
operation at the moment it is made. That is a genuinely different resolution
from the frame representation, which avoids the lever arm by not using torsions
at all.

**But it breaks the ripple, and that asymmetry is the thing to understand.**
The fix for single-pass identity decoding was to let committed residues revise
themselves in light of their neighbours. That does **not** transfer to backbone
torsions: re-deciding an identity at position *i* changes only position *i*'s
atoms, while re-deciding phi at *i* moves the entire C-terminal remainder. So
Gibbs-over-neighbourhoods generalises to sequence and fails for backbone
torsion, for exactly the reason growth works in the first place. Growth buys a
cheap forward pass and gives up refinement — which is the defect it was
supposed to cure.

The resolution is that growth and frames are not competing proposals:
**growth is an initialisation strategy, frames are the refinement
representation.** Rosetta reaches the same split from the other side, moving
fragments and then closing loops.

**The strongest version is growth and relaxation running concurrently** — new
cells appearing at the growing end while every placed cell keeps updating —
rather than grow-then-refine. That is what co-translational folding actually
is, it is what an NCA is naturally good at, and it is literally Growing Neural
CA, which starts from a single seed cell. Worth noting our present model is a
*fixed-size* CA, so this would make the architecture more faithful to its
namesake, not less.

**The biophysical motivation is weaker than it sounds, which is worth knowing
before leaning on it.** For elongation factor G, folding runs *counter* to
synthesis: a strictly sequential pathway initiating from the **C-terminus**
while translation proceeds N to C. So "N to C because that is how the ribosome
does it" conflates synthesis order with folding order, and they can be
opposite. Separately, structural asymmetry along sequences has been linked to
vectorial translation, with slow-folding sections toward the N terminus and
fast-folding toward the C.

**Prior art says autoregressive-over-angles is not where the field landed.**
PB-GPT generates backbones one residue at a time over internal angles and bond
lengths via codebook indices. More tellingly, the FoldingDiff work used an
autoregressive transformer over backbone angles as a *baseline* — i.e. the
diffusion approach was the contribution and the autoregressive one was the
thing it beat. Neither is a growth model in the co-translational sense; no
source was found for one.

**A supervision trap, and this codebase already contains the lesson.** The
native conformation of a *prefix* is not well defined: a 50-residue prefix of a
300-residue protein does not adopt its final conformation in isolation, because
it is stabilised by contacts that do not exist yet. Supervising prefixes
against the final crystal coordinates asks for something physically wrong.
`rollout.training_objective` already says this about crops — the boundary shell
is excluded because "its native torsions depend on contacts that were cropped
away and asking for them is unlearnable rather than merely hard." Same error
class, and the fix has the same shape: supervise only what has complete
perception, or only the final state.

Two smaller things: growth in torsion space can produce mirror-image folds, so
it needs a handedness term; and growth is inherently serial, so a 300-residue
protein costs 300 sequential placements per structure, which throws away the
parallelism that is an NCA's main computational appeal unless the concurrent
formulation above is used.

### After inverse folding: backbone degrees of freedom, folding, and folding ddG

User's direction, recorded 2026-11-02 for when inverse folding works. Hand the
rule control over backbone geometry with its neighbours as well, so the same
local update rule can fold and can score folding ddG, in the spirit of
ProteinEBM. Nothing here is built; the point is to record what is already in
place for it and the two things that are not what they look like.

**Already laid, and more than it seems.** `MOVABLE_BONDS` and `MOVABLE_ANGLES`
exist and are frozen rather than absent, so the degrees of freedom are
defined — though note they are *side-chain* internal coordinates, not backbone.
More importantly the two terms a folding objective cannot do without are now
written and tested: the steric term (`aanca.steric`) and the bond/angle
harmonic restraint. A folding model without steric exclusion is hopeless, and
the restraint is what stops geometry being deformed to relieve a clash. And
`FeatureTopology` already carries `frame_idx`, `frame_valid` and `has_frame`,
so the frame representation is in the graph as *input*.

**The lever arm is the real obstacle, and it is a difference in kind.**
Rotating a chi moves the downstream atoms of one side chain — a bounded, local
set, which is why a local update rule is coherent at all. Rotating phi at
residue *i* moves **every atom from *i* to the terminus**, so a 1 degree change
displaces distant atoms by many angstroms. A rule whose cells each nudged their
own phi/psi would produce enormous uncoordinated global motion, and the
"locality" the architecture rests on would be gone.

The literature's answer is not to give cells torsions. AlphaFold2's structure
module predicts a **rigid-body frame per residue** (translation and rotation in
SE(3)) and idealises local geometry; frames are independent, so there is no
lever arm — and AF2 predicts *torsions* only for side chains, which is exactly
our current setting. RFdiffusion, FrameDiff and Chroma operate on frames too,
and Rosetta's fragment insertion avoids the lever arm by moving fragments and
closing loops. So the step is **frames as an output**, not phi/psi, and the
frame fields above are already the input half of that.

**A code assumption that will silently go invalid.** `clash_loss` derives its
17 angstrom neighbour cutoff from a *fixed* backbone (CA positions plus a 6.6
angstrom side-chain reach), and it **drops backbone-backbone pairs** on the
grounds that those distances are constants no gradient can reach. Both become
wrong the moment the backbone moves: the pair list has to be rebuilt every
step rather than once, and backbone-backbone clashes are precisely what folding
must avoid. Neither failure raises — the term would just quietly stop scoring
the most important pairs. Whoever unfreezes the backbone has to change those
two things first.

**What the claim would have to be.** Fixed-backbone packing is scored against
crystal chis; folding is scored by TM-score, GDT and lDDT against the native
fold, and the comparison becomes AF2 and ESMFold. Competing on folding accuracy
is not plausible. What is plausible, and more interesting, is folding as a
**dynamics** claim — that a local update rule reaches the native fold as a fixed
point — which is a statement about self-organisation rather than accuracy, and
is the one thing this architecture could say that a single-pass predictor
cannot.

**And the ordering is right for a reason worth stating.** On a fixed backbone
every error is attributable, because the backbone is ground truth: that is what
makes the clamp test, the basin probes and the per-residue breakdowns
diagnostic. Unfreeze it and errors entangle — a wrong rotamer and a wrong
phi/psi produce the same lDDT drop — so all of that measuring power is spent at
once. Doing inverse folding first keeps it.

Folding ddG then composes with what is already built: the Potts energy gives an
evaluable scalar, MH at finite temperature gives an ensemble, and free-energy
differences come out of the ensemble rather than out of an entropic term. That
is the same chain of reasoning as above, extended from sequence to geometry.

### Inverse folding: the design settled on 2026-11-02, and why

Decided in discussion, not measured. Recorded because three of these are
decisions against an alternative that looked better, and the reasons will not
be recoverable from the code.

**The architecture is already fixed-backbone.** The rule only drives chis; the
backbone is input. So inverse folding is not a new setting, it is this setting
with identity unfrozen.

**The averaging problem — why identity must not be a soft simplex.** The
closest prior art to this architecture is SCMF (Koehl & Delarue 1994), which
reaches a fixed point by purely local field updates, exactly as we do. Its
failure is specific: the field residue *i* feels averages the pairwise energies
over neighbour *j*'s *distribution*, `E_eff(i,r) = E_self(i,r) + Σ_j Σ_s p_j(s)
E_pair(i,r; j,s)`. Steric exclusion is a hard pairwise constraint, so averaging
over a distribution turns "forbidden with one of *j*'s rotamers" into
"half-forbidden", and the converged field has minima that correspond to no
realisable joint structure. It is worst in tight cores, where the coupling is
strongest.

It does not bite us *now* because the cell state is a single concrete
conformation with actual coordinates, and messages carry realised geometry
rather than an average over options — the discrete, joint regime. That is why
the clamp test below could show an 18-point effect at all.

It would bite the moment identity is carried as a soft 20-way simplex, which is
the natural differentiable thing to do with an NCA output neuron. A cell that
is 40 % TRP and 30 % GLY broadcasts a field no residue exerts, and inter-type
steric differences are far larger than between rotamers of one type, so the
smearing is worse than in the case SCMF is already known to fail at. **So
identity is discrete in the forward pass**, sampled or argmaxed with a
straight-through or Gumbel-softmax gradient.

**The output is one joint categorical over (type, rotamer), not two heads.**
The cell *is* its position, so it never outputs that. What it needs is a
distribution over type-and-conformation jointly — which is precisely Rosetta's
rotamer set, where one move is "position *i* becomes rotamer *r* of type *t*".
Three things follow at once: discrete, so no mean-field smearing; joint, so
type and conformation cannot disagree; and **the atom-set problem disappears**,
because each library entry is a complete side chain with known atoms, so no
union-of-atoms mask (which would name the residue and leak the answer) and no
truncation to CB. The cost is the continuous output that AttnPacker and DiffPack
established as the advantage over binned methods, so the head is
classification *plus* regression: logits over (type, well) and a continuous
residual within the chosen well. The von Mises head is already the regression
half; only the selector above it is missing.

New convergence diagnostic required. `mean_abs_delta` decaying stops meaning
anything once a discrete label can flip between steps — the continuous part can
be settling while the labels churn. The replacement is **the fraction of
positions whose label changed this sweep, which must decay to zero**. It is the
same diagnostic the ripple needs, so it is one piece of machinery.

**PottsMPNN, corrected.** It does *not* replace the single pass. It uses
ProteinMPNN's featurisation and GNN encoder, trained to emit a Potts model
(fields and couplings) alongside single-site probabilities, building on
COORDinator; generation is still the autoregressive decoder, and the Potts
model scores variants and drives a local optimisation that sweeps in the
generation order until no favourable substitution remains. That is iterated
conditional modes at zero temperature, initialised from an autoregressive
sample — published precedent for the shape of the N→C-then-ripple scheme rather
than a competitor to it. Separately, "order-agnostic" belongs to ProteinMPNN
itself, which samples the decoding order from all permutations during training;
so no order is privileged, and the defect that survives is the narrower one that
*within* a sample, early positions commit and are never revised.

**A fixed point is not an energy, which is why the explicit energy is needed.**
An update map `x ← f(x)` descends a scalar only if its Jacobian is symmetric,
which a generic net's is not — so our converged state is the fixed point of a
field that is not curl-free, i.e. the minimum of nothing. The oscillation risk
is the observable symptom. **Decision: a Potts energy**, `E(s) = Σ h_i(s_i) +
Σ J_ij(s_i,s_j)`, with the parameters predicted from the backbone. Chosen over
an MD forcefield (enthalpy of one conformation, solvation-dominated, too stiff
to evaluate per move, and never fit for design — Rosetta's statistical hybrid
exists because pure physics underperforms here) and over a learned scalar
energy, for one structural reason: **a single-position move changes only the
terms involving *i*, so ΔE is O(neighbours)**, and that is exactly what both
Metropolis–Hastings acceptance and point-mutant ΔΔG need. Pairwise additivity
is an approximation and three-body packing is real — which is the division of
labour, since the AANCA supplies the joint geometry-aware refinement a Potts
model cannot express, and the Potts model supplies the evaluable scalar we
cannot produce.

**Entropy comes out of the ensemble, not out of a term.** With an enthalpic
energy and a valid MH chain at finite temperature, free-energy differences are
estimable from the sampled ensembles by standard means. Anneal to zero and that
is thrown away; run at finite *T* and it is free. This is the concrete reason to
pay for MH rather than stopping at annealed Gibbs. It also rehabilitates SCMF:
its mean-field marginals *are* a free-energy estimate, and side-chain
conformational entropy falls straight out of them. Averaging is wrong for
finding a structure and right for scoring one.

**Dropped: the fractional-position morphogen gradient.** Absolute index lets
the rule memorise "position 47 of this protein is LEU" — a shortcut that works
on the training set, generalises nowhere, shows up in no aggregate metric, and
is incoherent under cropping, since residue 47 of a crop is residue 203 of the
protein. The legitimate version is relative offset, and it is already in:
`torch_graph.py:538-541` one-hots the residue offset clamped to ±16 as an edge
feature, the same design as ProteinMPNN. If the gradient is ever wanted, the
form to add is capped distance to the nearest terminus computed from the full
chain, which is bounded and not a position index.

### The update schedule: a colouring cannot be fixed, so the set is drawn per step

Implemented 2026-11-02, `src/aanca/schedule.py`, default off.

Gibbs sampling requires the cells updated together to be conditionally
independent given the rest, and two residues in contact are the opposite of
that. The Bernoulli mask ticks half the protein at once, which is synchronous
mean field, and strongly coupled neighbours under synchronous updates oscillate
rather than converge.

The obvious fix — colour the interaction graph once, update one class per step
— does not work here, and the reason is worth keeping. A colouring has to be
proper for the graph message passing actually uses, and `torch_graph` selects
neighbours by **minimum heavy-atom distance**, which moves when a side chain
moves. Colouring a conservative superset is worse: a side-chain atom reaches
~6.6 Å from its CA, so "could ever fall inside the 10 Å cutoff" means CA–CA
within 23 Å, which on a 128-residue crop is every pair — the superset graph is
complete and wants one residue per step.

So the set is drawn **fresh per step**: a random greedy maximal independent set
of the current coupling graph. Exactly proper, because it is computed from the
geometry in hand; freshly random, so there is no fixed partition and no phase
for a cell to learn through its persistent hidden state; and no colour
bookkeeping, since one maximal independent set *is* one colour class.

The coupling graph is deliberately **not** the message-passing graph. Gibbs
independence is a property of the energy, not of the perception radius: two
residues 9 Å apart exchange messages but their side chains cannot touch, so
they are conditionally independent for packing and may tick together. That
distinction is worth a factor of 2.5, measured on native structures:

| coupling radius | mean degree | set size (1UBQ, 76 res) | colours per sweep |
|---|---|---|---|
| 5 Å (contact) | 9.5 | 13.6 (17.9 %) | 8–9 |
| 6 Å | 11.3 | 11.9 (15.7 %) | 9–11 |
| 8 Å | 19.8 | 7.9 (10.4 %) | 13–15 |
| 10 Å (perception) | 28.1 | 6.2 (8.2 %) | 20–21 |

193L (129 residues) at 5 Å: degree 10.6, sets 21.4 residues, 9–10 colours. At
10 Å the coupling graph's mean degree equals the k=64 message graph's edges per
residue exactly, 28.1 and 33.9, since both use the same distance measure.

**So an exact schedule is affordable**: 8–10 colours per sweep means a 30-step
rollout is about three sweeps. And the comparison that matters for the
oscillation worry: the Bernoulli mask at 0.5 ticks 38–64 residues per step
where independence allows 13–21, so the current schedule updates about three
times more residues than it may.

Not yet measured: whether the independent schedule helps *repacking*, which it
need not — the reconstruction task is not a sampler and the Bernoulli mask is
what v1–v6 were trained with. The default stays `bernoulli` and the stream is
bit-identical, so this is a switch to test rather than a change to absorb.

A trap worth recording for when the cell gets to choose anything about its own
turn: selection may depend on the current state only, never on the value being
proposed. A cell that decides it is its turn *because* of what it wants to
become destroys the chain's stationary distribution. Rosetta picks positions
uniformly at random for exactly this reason.

### The clamp test — the rule reads its neighbours, and reads them wrong

Measured 2026-11-02, 31346329, `docs/benchmarks/clamp_v5_native.log`. This is
the interventional form of the spatial-clustering question, and the answer is
neither of the two I had set up for.

A fraction of residues is held fixed through the whole rollout — they keep
ticking their hidden state, so neighbours are fed a vector the rule has seen
before, but their atoms never move (`rollout(..., frozen=)`). Two arms at each
fraction: the pinned residues held at their **native** conformations, or held
at an **independently randomised** draw. Both arms freeze the same residues and
score only the free ones, so the quieter-neighbourhood effect of pinning is
divided out and `native - random` varies one thing — whether the information in
the neighbourhood is correct.

      pinned     native - random
       0.25              +0.5
       0.50              -6.9
       0.75             -18.1

**Correct neighbours make it worse, by a lot.** With three quarters of the
protein pinned at crystal conformations the free residues score 57.1 %; pin the
same residues at random conformations instead and they score 75.2 %. The
`pinned drift` column is 0.00e+00 at every fraction, so the pin did not leak and
the table stands.

**What this settles, and what it does not.**

- **The co-solving is real, not decorative.** An 18-point swing from varying
  only the *content* of the neighbourhood is not compatible with a per-residue
  marginal predictor. The clamp test's null — "near zero at every fraction" —
  is decisively rejected. The question I wrote it to answer is answered: the
  rule reads its neighbours' side chains.
- **But the reading is miscalibrated, and in a way that names the cause.** A
  free residue displaced inside an otherwise-native neighbourhood is a state
  **the training distribution never contains**. `ChiDataset.sample` draws one
  concentration per sample and applies it to *every* chi (`data.py` 585-620,
  `von_mises_deltas` masking on `chi_mask`), so the curriculum is a mixture over
  *global* damage levels: all-damaged, or, at `--native-probability`, all-native.
  Nothing in it is spatially mixed. `stationary` already showed the all-native
  corner is fine (0.76 % wells broken). So the failure is specific to the
  mixed state, which is exactly the off-distribution prediction.
- **And that mixed state is the classical packing task.** "Repack this residue
  in a fixed native environment" is what a rotamer-library method with a
  pairwise energy does natively and what SCWRL is built on. Our rule cannot do
  it — 57.1 % against the 77.3 % it manages when everything is damaged
  together. That is the most basic form of the problem and we are *worse* at it
  than at the harder one.
- **It does not separate two mechanisms.** Off-distribution input and a missing
  clash term predict the same table. A dense correct neighbourhood is
  sterically tight; relieving that requires a response to overlap that was
  never in the loss (there is no clash term in v1 — `training_objective` is
  reconstruction + no-op + bond/angle restraint, and the only occurrence of
  "clash" in `losses.py` is a comment explaining why bond geometry must be
  restrained lest the rule relieve a term that does not exist). Both point at
  the same two fixes, which is why this is not worth a further probe before
  trying them.

**The fix this ranks first: per-residue damage.** Damage a random *fraction* of
residues per sample rather than every one of them, at a concentration drawn as
now. It is a change to `ChiDataset.sample` and nothing else; it costs no
compute; it makes the mixed state a training case; and it turns the clamp test
into a regression test with a pre-registered direction. The clash term is the
second, and the two are independent.

**One honest limit.** The clamp test varies neighbour *conformation* only. It
says nothing about whether the rule reads neighbour *identity*, which is the
question the joint model turns on, and nothing run so far addresses that.

### bestof — the prediction held, and unanimity is a free confidence signal

Measured 2026-11-02, 31329821, `docs/benchmarks/basin_bestof_v5_native.log`.
Eight independent starts over the same 17,979 residues.

      one seed, the published protocol :  77.5%  (77.3 to 77.7 across seeds)
      modal well across seeds          :  77.9%
      any seed right                   :  84.4%  (oracle)
      every seed right                 :  69.6%

      all 8 starts agree on the well   :  83.5% of residues
        and there the first seed right :  83.3%
        where they disagree, right     :  47.0%

**The prediction recorded before it ran was right**, which matters because it
was the one result that could have contradicted `descent`: "any seed right
modestly above 77.3 % and unanimity high" — 84.4 % and 83.5 %. The 95 % figure
that would have made `descent` and `sweep` untenable did not appear.

- **Voting buys nothing.** Modal over eight seeds is +0.4 points. The attractor
  really is mostly start-independent, so there is no free accuracy here and no
  case for an ensemble protocol.
- **Unanimity *is* a usable confidence estimate, and it is free.** 83.3 % right
  where eight seeds agree against 47.0 % where they do not. Nothing currently
  asks the rule for a confidence; this is a 36-point separation available from
  re-running it, with no energy function and no native answer needed. It is the
  obvious handle for the mutable-identity model, where a confidence ordering is
  what a sequential decoder needs.
- The oracle gap (77.5 to 84.4) bounds what *any* restart strategy could win at
  6.9 points, and nothing cheap gets near it: the modal vote captures 0.4 of
  those 6.9.

A 30-target, 4-seed replicate (31351486) is on scratch and agrees in shape. Its
headline reads 81.7 % because it scores a different, smaller target subset —
**do not compare that number to 77.5 %.**

### Why v4 looks the way it does

Three settings changed at once, which is usually bad practice and is right here: all three came
from the same reading of the Neural Cubes trainer (see `docs/PLAN_V1.md`), all three are
differences from prior art that works rather than guesses, and v3's curve is kept as the control
that any of them can be read against.

- **A decaying learning rate**, which this project had never had. Drops at 30,000 and 45,000, ×0.3
  each, mirroring the prior art's halves-and-three-quarters. v3's flat stretch from update 9,000
  is at least as consistent with a rate that had become too large as with a model that had
  finished learning, and there was no way to tell the two apart without a schedule.
- **A 64–96 step rollout**, resampled per update, against v3's fixed 20. Truncated BPTT keeps the
  memory flat, so this costs time and not memory.
- **Both arms warm-started from the same `runs/v3_gpu/last.pt`.** Not from their own v3
  counterparts: an identical starting point makes `--noop-weight` the only difference between the
  two arms at update 0, and v3_noop4 had barely started anyway. The warm start carries the weights,
  the optimiser moments and — the part worth having — the sample pool, which is the long-horizon
  memory the prior art says persistence comes from.

**60,000 updates, not 200,000.** Measured at **2.80–2.96 s/update** against v3's 0.91 at 20 steps,
so about 48 hours, or three chained segments per arm. Extending is allowed and will not move the
drops, which is exactly why they are absolute update numbers.

### The longer rollout paid off immediately — v4 at update 500

v3_gpu was at **20,000** updates when it was stopped, and v4's own first evaluation measures those
weights under v4's eval settings, so the two rows below are directly comparable. 500 updates at
64–96 steps:

| | v3 at 20,000 (v4's update 0) | v4_gpu at 500 | v4_noop4 at 500 |
|---|---|---|---|
| pooled chi1 MAE (°) | 23.17 | **22.44** | 22.61 |
| κ=100 chi1 MAE (°) | 5.01 | **4.76** | **4.62** |
| drift from native (Å) | 1.697 | 1.439 | 1.331 |
| mean &#124;delta&#124; at step 30 (°) | 2.0 | 1.2 | 1.4 |

**500 updates bought more than the previous 5,500 did.** The κ=100 row is the one that matters:
the rule had been getting *worse* at leaving a native structure alone (4.46 untrained, 5.34 at
14,500, 5.01 at 20,000) and is now going back the right way, and the per-step motion at the end of
the rollout has nearly halved. None of this is the learning-rate schedule, which does not drop
until update 30,000 — it is the rollout length alone, which says the fixed 20-step horizon was the
binding constraint and v3's plateau was not a property of the objective.

Too early to separate the arms: 22.44 against 22.61 at one evaluation is noise, and the no-op arm's
slightly lower drift is the right direction and nothing more yet.

### Nine thousand updates in — accuracy improves, persistence does not

Both arms are still inside segment 1 (`30976061`, `30976062`, bun161, 7.5 h, 2.88–2.92 s/update
holding). Nineteen evaluations each. The two halves of the objective have come apart:

| update | 0 | 1000 | 3000 | 5500 | 7500 | 8500 | 9000 |
|---|---|---|---|---|---|---|---|
| pooled chi1, v4_gpu | 23.17 | 22.17 | 21.90 | 21.34 | **20.07** | 21.72 | 22.13 |
| pooled chi1, v4_noop4 | 23.17 | 22.17 | 22.02 | 21.61 | 20.79 | **20.37** | 21.05 |
| drift (Å), v4_gpu | 1.697 | 1.193 | 1.364 | 0.995 | 1.497 | 1.302 | 1.728 |
| drift (Å), v4_noop4 | 1.697 | 1.173 | **0.805** | 1.173 | 1.647 | 2.213 | 2.859 |

**Pooled accuracy is still coming down**, roughly 3° over 9,000 updates, which is the prior art's
prediction holding. κ=100 fell from 5.01 to ~4.2 inside the first 1,000 updates and has been flat
and noisy since (4.1–4.7, with isolated spikes to 5.1–5.6) — the v3 regression is undone, and
nothing further has been bought there.

**Drift is going the wrong way, and the no-op arm is worse at it.** Both arms reached their best
hold near update 3,000–4,000 (0.81 Å noop4, 0.89 Å gpu) and have been climbing since; the noop4 arm
is now at 2.86 Å against the control's 1.73. That is the arm whose extra term exists to prevent
exactly this. The per-step motion says why: `noop_nll` pins the *first* step from native near zero
(both arms now start at 0.0–0.1°/step) and the rollout answers with a hump in the middle — the noop4
arm peaks at 6.2°/step around step 9 where the control peaks at 2.7. Weighting a single `step_once`
at the native structure buys step 1 and nothing after it, which is the one-step-against-thirty-step
mismatch flagged in §10, now with a number on it.

So the arms are no longer indistinguishable, but not in the direction the arm was motivated by:
pooled chi1 is a wash (20.07 vs 20.37 best, inside the oscillation), and on persistence the extra
term costs about a hundred per cent. Nothing to change mid-run — both arms go to 60,000 as planned,
the first learning-rate drop is at 30,000 (≈17 h away), and whether the drift climb survives the
drop is worth more than another weight sweep.

### The no-op term is supervising a step the rule is silent at — measured 2026-10-30

`scripts/noop_probe.py runs/v4_gpu/last.pt`, at update 9,700:

| | from native | from kappa 0 damage |
|---|---|---|
| mean &#124;mu&#124; at step 0 (°) | 0.041 | 0.091 |

Step 0 is the only step `noop_nll` ever sees — `training_objective` builds
`initial_state(native_coords, rule)` and takes one `step_once`. At that step the
hidden state is all zeros, and the measurement above says the rule proposes
nothing **whatever it is shown**: a structure 56° from native gets the same 0.09°
shrug as the native one. The term cannot tell the two apart, so it is not
supervising the fixed point; it is supervising a constant. That is why
`--noop-weight 4` changes nothing that matters, and why no other value would.

**And the fixed point is held anyway.** Over a 30-step rollout from native, mean
displacement is 0.016 Å and 0.013 Å on the two structures probed, with a
worst-atom excursion of 0.37 Å and 0.41 Å. The rule stays where it is put.

**So the previous entry's reading of the drift number was half wrong.**
`EvalResult.native_drift` is a max over atoms of a max over the 12 held-out
structures — the single most extreme coordinate in the set. Across all twelve,
at 30 steps, the control's worst atom ranges from 0.24 Å to 0.99 Å, so the one
number reported swings with whichever structure happens to flip a terminal chi.
As a *trend over updates* it is not readable, and the climb from 0.8 Å to 2.9 Å
should not have been read as one. The mid-rollout hump read off the convergence
curve belongs to the *damaged* rollouts, which is where motion is supposed to
be, so that was wrong too.

**But the gap between the arms is real.** Probed over all twelve at matched
updates (9,700 against 9,900), mean displacement at step 29:

| | control | noop weight 4 |
|---|---|---|
| mean over structures (Å) | 0.0143 | 0.0198 |
| worst structure (Å) | 0.990 | 1.609 |
| structures where this arm drifts more | 0 | **12 of 12** |

Twelve out of twelve is not a tail. The no-op term makes staying *worse*, and
the probe says why it can do that while supervising nothing: it is vacuous as a
statement about the fixed point, but it is not inert as a gradient. It still
pushes `mu` toward zero and `log_kappa` upward at step 0, on weights shared with
every other step, so what it actually buys is inflated confidence everywhere in
exchange for nothing — four times as much of it at weight 4.

What the probe leaves standing: pooled chi1 is still improving, and the real
deficit is not persistence but kappa 0 — 49–56° against a damaged input of
89.7°, with 20% accuracy. That is the number v1 has to move.

### v6: the mixed state as a training case — `--damage-fractions`

Built 2026-11-02, directly off the clamp test, and it is a **fine-tune rather
than a generation**: `--init-from runs/v5_native/last.pt` carries the weights,
the optimiser moments and the sample pool across, so nothing v5 learned is
discarded. Ulban's constraint, and the right one — the finding is a gap in the
training *distribution*, not in the architecture.

**The change.** `RepackingDataset` gains `damage_fractions`, a second axis of
the existing mixture: the share of residues the damage is applied to at all,
drawn per sample, the rest starting at their native conformations
(`aanca.data.damaged_residues`). A residue is damaged on all of its chis or
none — the unit is a side chain in the wrong place, not a chi2 disagreeing with
its own chi1 — and the subset is drawn independently per residue, which matches
the clamp test's own intervention. A spatially contiguous patch would also
train the mixed state but would let the rule key off the patch boundary; worth
trying as a variant, not first.

**Two details that would be bugs if they went the other way.**

1. `(1.0,)` — the default, and what every run up to v5 did — draws *nothing*.
   One option is not a choice, so no `randint` is spent on it and the random
   stream is byte-for-byte the pre-2026-11-02 one. Without that, v5's curve
   stops being reproducible and the warm start lands on a distribution it does
   not report. The precedent is `--native-probability`, which skips its own
   draw at probability 0 for exactly this reason. Tested.
2. `damage_fractions` is **pinned** in the checkpoint fingerprint, so a
   *resume* cannot silently change what the damaged half of the distribution
   is. It is deliberately absent from `train.WARM_START_MUST_MATCH`, which is
   what lets a warm start change it: the pool and the split still describe the
   same structures, which is all that list is defending.

Note that fraction 0 *is* a native start, so this subsumes
`--native-probability`. Both knobs are kept because the probability is pinned
in checkpoints already written.

**The run.** `v6_mixed`, from `v5_native/last.pt` at update 60,000:

    bash scripts/bunya_submit.sh gpu --tag v6_mixed --split train \
      --updates 20000 --eval-every 500 --steps 64 --steps-max 96 \
      --lr 3e-5 --lr-drop 12000 --lr-gamma 0.3 \
      --native-probability 0.2 --noop-weight 0 \
      --damage-fractions 1.0 1.0 0.5 0.2 0.05 \
      --init-from runs/v5_native/last.pt

- **The learning rate is the one thing to get right.** A warm start restarts at
  update 0, so the drop schedule is re-applied from scratch: passing v5's
  `--lr 3e-4` would raise the rate ~11x over what it was actually training at
  by update 60,000 (3e-4 after two 0.3 drops is 2.7e-5) and would undo the run
  rather than extend it. 3e-5 continues roughly where v5 left off.
- **1.0 is duplicated in the list on purpose**, so global damage holds 40 % of
  the fresh draws rather than 20 %. The published benchmark *is* the global
  task and must not regress to buy the mixed one. The in-run guard is free:
  `build_eval_set` is untouched, its three levels are still 100/20/0 applied
  globally, so the held-out curve stays directly comparable to v5's and a
  regression shows within a few hundred updates. Raise the 1.0 share if it
  does.
- **The readout is pre-registered**, which is the part worth protecting. Run
  `scripts/clamp_test.py` on the fine-tuned checkpoint and compare against
  `docs/benchmarks/clamp_v5_native.log`. The prediction: `native - random`
  moves from -18.1 at fraction 0.75 towards 0 or positive. If it does not move,
  the off-distribution explanation is wrong and the missing clash term is
  carrying the effect on its own, which is then the next thing to add.
- **A baseline on the exact weights.** The clamp and basin logs were measured
  on `best.pt` at update 59,500 (pooled chi1 MAE 14.69); `last.pt` is update
  60,000 (15.20, half an eval interval later and marginally worse). The warm
  start must use `last.pt`, because `best.pt` holds the weights alone — no
  optimiser, no pool, no args — and `--init-from` would refuse it. So
  `clamp_test` is also run on `last.pt`, to make the before/after exact rather
  than approximately exact.
  **It mattered more than it looked** (31355896,
  `docs/benchmarks/clamp_v5_native_last.log`). `last.pt` is *worse* on this
  axis than the measured `best.pt`, and not marginally:

        pinned      best.pt    last.pt
         0.25          +0.5       -0.9
         0.50          -6.9      -11.5
         0.75         -18.1      -20.9

  So the pre-fine-tune baseline is **-20.9**, not -18.1. Scoring the
  fine-tuned weights against the `best.pt` table would have credited the fix
  with about 3 points it did not earn. The two checkpoints are 500 updates
  apart and differ by 0.5 deg of pooled chi1 MAE, which is eval noise; that
  the clamp table moves 2.8 points over the same gap says this measurement is
  the more sensitive of the two, and that a single clamp run is not a tight
  number. Worth a seed replicate before any claim rests on a few points of it.

**Where it got to.** Segment 1 is 31355895 on an MI355X, 2.9 s/update, so
20,000 updates is about 16 h and fits one 24 h segment. Through update 3,000,
read from `runs/v6_mixed/history.json`:

      update   pooled chi1   acc all   kappa 0 chi1   kappa 0 acc
           0         15.20      73.3          33.47          36.3
         500         14.77      74.1          32.11          38.7
        1500         14.68      74.2          31.63          39.5
        2500         14.47      74.8          31.27          40.2
        3000         14.56      74.3          31.25          39.8

The update-0 row reproduces v5's own update-60,000 eval exactly (15.198),
which is the check that the warm start really loaded those weights and not
something else. **No regression on the global task** — the worry that bought
1.0 its duplicate slot in the mixture. kappa 0 is the benchmark's own
condition and it improves by 2.2 deg and 3.5 accuracy points, and pooled chi1
passes v5's all-time best of 14.69 by update 1,500. That is the fine-tune
paying for itself before the clamp test has even been re-run; whether it pays
on the thing it was *for* is 31417064, on `runs/v6_mixed/best.pt`.

**A trap this run walked into, now fixed.** `bunya_train.slurm` ran
`python scripts/train.py` without `-u`, so the `.out` sat two evals behind
`history.json` and looked for a few minutes like a stalled job. The `.out` is
the only window into a running segment and block buffering makes a healthy job
indistinguishable from a hung one -- the same lesson `bunya_cpu.slurm` already
carried, in the one place it had not been applied. `history.json` is written at


### v5: the fixed point as a training case — `--native-probability`

Decided 2026-10-30. The no-op term is dropped rather than relocated, and the
fixed point becomes an ordinary training case instead of an assertion:

* **`--native-probability`** (default 0.0). That fraction of fresh samples
  starts the rollout from the native structure rather than a damaged one. The
  reconstruction term then asks, at the same random late step it always uses,
  for the rule to *still* be at native after 64–96 steps — with a hidden state
  that has had time to say something, which is exactly what step 0 cannot
  offer. The end state is pooled like any other, so near-native states become
  future seeds and the horizon compounds. This is how the Neural Cubes runs get
  persistence: from the pool, with no extra term and no weight to tune.
* **`--noop-weight 0`** skips the term's forward pass entirely. The default
  stays 1.0 so the v4 runs still resume mid-chain.
* **`native_drift_mean`** and a per-structure breakdown are reported beside the
  max, and `native_starts` is tallied in the checkpoint and the history.
* `check_fingerprint` now takes the defaults. A key missing from a stored
  fingerprint is a setting that did not exist when it was written, so at its
  default the resume stands — otherwise pinning `native_probability` would have
  killed both v4 chains at their next segment boundary, hours after the commit.

What this does not address is the actual deficit, which is kappa 0: 49–56°
against an 89.7° damaged input, 20% accuracy. Persistence was never the thing
that was broken. v5 is worth running because the no-op term is measurably
harmful and the pool mechanism is what the prior art uses, not because it is
expected to move the headline.

### The tier-1 comparison, set up 2026-10-30

To run when `v5_native` finishes, and runnable on any checkpoint before then:

```bash
python scripts/validate_benchmark.py --casp 13     # our metrics vs their table
python scripts/benchmark.py runs/v5_native/best.pt --casp 13 --steps 100
```

The first is the check that makes the second mean anything: it scores the
baselines' *own* predicted structures from AttnPacker's Zenodo release with our
metrics and recovers their published table. FASPR comes out at 0.907 Å against
a published 0.910 over 93 CASP13 targets. The second randomises every chi —
the packing problem as the literature poses it — rolls the rule out, and prints
the row beside the published ones.

**What is comparable to what.** `aanca.casp.REFERENCE` is DiffPack (NeurIPS
2023) Tables 1 and 2: SCWRL4, FASPR, RosettaPacker, DLPacker, AttnPacker and
DiffPack, on the CASP13/CASP14 target sets in AttnPacker's release, with
accuracy defined per *residue* (every chi within 20°) and the core/surface split
at ≥20 / ≤15 CB neighbours within 10 Å. Our run uses the same targets — the
intersection over every baseline's directory, so nobody is averaged over a
different set — and the same `Convention`.

**FlowPacker is not in that table, on purpose.** Its accuracy columns are per
*chi*, not per residue, which reads about ten points high, and its test set is
20 / 34 / 45 targets for CASP13 / 14 / 15 pulled from the Prediction Center
rather than from AttnPacker's release. Its rows sit in
`aanca.casp.OTHER_PROTOCOLS` with the protocol attached and a test keeps them
out of the comparison. Two further warnings from the same source: FlowPacker's
own re-run of DiffPack ("DiffPack-fix", idealised bond geometry as input) does
not reproduce DiffPack's published numbers, and DiffPack's prose quotes 0.793 Å
for CASP14 all-residue RMSD where its Table 2 says 0.770. Nothing at the third
decimal is worth arguing about across papers.

**Which columns are auditable — chased down 2026-10-30.** Scoring all four
classical packers' own CASP13 structures with our metrics reproduces their
chi MAEs (within 1.3–4.6°), their accuracy in all three bands (within about a
point: FASPR core 69.6 against 70.3, surface 44.1 against 43.6, and the same
for SCWRL, RosettaPacker and DLPacker), and RMSD **all** (within 0.003–0.015
Å). It does not reproduce RMSD core or surface, which run +0.12 to +0.17 and
+0.10 to +0.13 Å high, the same direction for every method.

My first guess — the burial definition — is wrong, and the evidence against it
is that *accuracy* stratified by the very same masks reproduces. A sweep
confirms it: no CB-neighbour threshold gets near the published core RMSD, ours
bottoming out at 0.543 Å with a count of ≥30 over 174 residues against their
0.502 over presumably thousands; per-structure instead of per-residue averaging
moves core only to 0.607 and breaks `all`; CB inclusion and a minimum
side-chain atom count do nothing. So their RMSD-by-burial columns rest on
something other than the burial definition their own accuracy-by-burial columns
use, and nothing here can recover it.

**So the comparison quotes chi MAE, accuracy all/core/surface, and RMSD all.**
Not RMSD core or surface: there, ours is a number of our own sitting in a
column with their heading. Both scripts now print that caveat under the table.
Note what this costs — DiffPack's headline gain over AttnPacker is on surface
residues, and the surface *accuracy* column is auditable, so the claim can
still be engaged with, just not through RMSD.

**The AttnPacker alignment, fixed 2026-10-30 — and it was worse than a
failure.** AttnPacker rewrites the chain id (blank → `A`) *and* renumbers from
1, while keeping the sequence and its order. Against a native numbered 43–181
the keyed match found nothing, and the old fallback on residue number alone
then paired native residue 43 with their residue 43 — a residue 42 positions
away that happened to share a name. That is not a failure, it is a wrong
answer: 7,035 scored residues where every other method gives ~24,000, and
3.693 Å against a published 0.669.

`align_prediction` now tries an exact sequence match in order *before* the
number-only fallback. It cannot misalign, because it requires every residue
name to agree along the whole chain. AttnPacker now scores 89/89 structures and
23,842 residues at 0.858 Å, and no other method's numbers move.

**What that exposed next: the released AttnPacker structures label equivalent
methyls arbitrarily.** Their chi2 MAE is 35.18 against a published 23.17, and
it is one residue type:

| | as written | methyls resolved per residue |
|---|---|---|
| AttnPacker LEU chi2 | 85.6° | **12.9°** |
| AttnPacker VAL chi1 | 41.9° | **8.3°** |
| FASPR LEU chi2 | 19.7° | 16.0° |
| FASPR VAL chi1 | 18.7° | 10.6° |

LEU's CD1/CD2 and VAL's CG1/CG2 are chemically identical, so which is which
carries no information — and AttnPacker's files pick arbitrarily where every
other packer follows the native. 85.6° is not an error the method made.

**Decided 2026-10-30: leave the labels as written.** Resolving the ambiguity
would be a change to the *convention*, not to the reader, and the four
classical methods reproduce their published rows precisely without it; moving
the convention to rescue one method's re-scoring would put the other four at
risk for no gain, since their published rows are quotable as they stand.

The consequence, which must travel with any table: **AttnPacker's re-scored
numbers from this archive are not usable** — chi2 especially, and the RMSD
through it. Its *published* row in `aanca.casp.REFERENCE` is quotable like any
other, because that is their measurement rather than ours. What is lost is the
audit: of the five baselines, four are checked against their own structures and
AttnPacker is taken on trust.

**Where we stand going in.** v4_gpu at update 20,700, on the training split's
held-out crops rather than CASP: kappa 0 chi1 40.3°, pooled 17.6°. DiffPack is
at 15.35° on CASP13 and AttnPacker at 18.92°. Those are different target sets
and not yet comparable, which is the whole reason for the harness — but the gap
on the from-scratch case is the thing to watch, and it is still closing fast
(kappa 0 chi1 was 56.4° untrained, 49.1° at 8,000, 40.3° at 20,700).

### The run is not GPU-bound — profiled 2026-10-30

`sbatch scripts/bunya_profile.slurm`, job 31143249, MI355X, a 384-residue crop
exactly as training uses:

| part of one rollout step | ms | share |
|---|---|---|
| `select_neighbours` (CPU NumPy + transfers) | **29.72** | **80%** |
| `edge_features` | 1.19 | 3% |
| `rule` forward | 0.34 | 0.9% |
| `node_features` | 0.15 | 0.4% |
| `step_once`, all of it | 37.27 | |
| *the same geometry, done on the device* | *0.53* | *56× faster* |

The model is **one per cent** of a rollout step. Four-fifths of it is
[`torch_graph.py:301`](src/aanca/torch_graph.py:301), which does
`coords.detach().cpu().numpy()` and then a NumPy spatial search and some
Python set arithmetic — a device transfer, a synchronisation and a CPU search,
eighty times per update, none of which the card can help with.

The two estimates agree: 37.27 ms × 80 steps is 2.98 s against the 2.80 s/update
actually logged, and the 31.5 ms/step inferred from the v3-to-v4 rate change is
the same number from the other direction.

**Written and measured, job 31144401.** `torch_graph.select_neighbours_device`
does the same job without leaving the card — residue-pair minimum heavy-atom
distances by a double `scatter_reduce`, then one stable sort with forced
neighbours first:

| | ms |
|---|---|
| `select_neighbours` (NumPy) | 29.68 |
| **`select_neighbours_device`** | **5.62** |
| the same, sequence links hoisted out | 5.54 |
| geometry alone, no bookkeeping | 0.52 |

Exact agreement with the NumPy version on the structure profiled, and pinned
by four tests in `tests/test_torch_graph.py`. **A step would go from 33.6 ms to
9.6, and an update from 2.7 s to about 0.77** — so 60,000 updates in roughly 13
hours rather than 47. That is less than the 9 hours the lower bound suggested,
and the gap is the price of exactness: `cdist` is pinned to
`donot_use_mm_for_euclid_dist` because the matmul formulation is not exact, and
a pair within an ulp of the cutoff would then be admitted by one implementation
and refused by the other. Hoisting the sequence links out of the loop saves
nothing, so there is no reason to complicate the call.

One trap, caught only by the tests: combining "forced first" with "then by
distance" into one sort key needs float64. The ulp of float32 at the 1e6 offset
is 0.0625, so adding it rounded every distance into six-hundredths-of-an-
Angstrom buckets and reordered genuine neighbours — a wrong graph that looked
entirely plausible.

Two things not to do with this. Not mid-run: the neighbour path and the batch
size both move the random stream, and v5's whole purpose is a comparison with
v4_gpu that depends on them matching. And not as a correctness change —
`select_neighbours` is not wrong, it is in the wrong place, so the replacement
has to reproduce it exactly on a fixed structure before it goes anywhere near
a run. Batch size is the other lever and is untouched: one crop per update,
where the Neural Cubes trainer used eight.

### Two more traps, both paid for on 2026-10-29

1. **`AANCA_CHAIN=0` does not disable chaining.** It is the segment *counter*, not a switch, so a
   one-off job submitted that way still resubmits itself when cancelled. A short probe cancelled
   without a STOP file came straight back as a new job. For a genuinely one-off run pass
   **`AANCA_CHAIN_LIMIT=0`**, which makes the resubmission guard fire immediately.
2. **A file created during the session can 404 through `vscode-remote-resource` while existing
   perfectly well on disk.** `cat` showed the content; the fetch did not. Overwriting a file that
   had already been fetched once worked every time. So: keep one scratch file per session and
   write into it repeatedly, rather than a new name per command. This looks exactly like mangled
   typing and wasted several round trips before `rquota` ruled out the obvious explanation.

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

**A trap that cost two submissions (30966837).** The browser bridge wants short lines, so the
arguments went into a shell variable to keep the command under the length that types reliably —
and `--split train` was dropped in the rewrite. The arm ran for half an hour on the *validation*
split against a train-split control, which is not a one-variable pair and would have been read as
a no-op-weight effect. Nothing in SLURM or the training script can catch this: both argument lists
are valid. **Read the `args` line out of the job's own banner after every submission** — it is
printed for exactly this reason — rather than trusting the command you believe you typed. The
wrongly-split run is at `runs/v3_noop4_wrongsplit`.

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

---

### Growth is implemented — pre-allocated slots, an alive mask, a frontier — 2026-11-03

Built this session: `aanca/backbone.py`, `aanca/growth.py`, `aanca/fape.py`, three new heads on
the rule, the alive path through `step_once`, `growth_objective`, `scripts/growth_probe.py`, and
47 tests. Nothing on the packing path changed behaviour: `growth=None` in `RolloutConfig` touches
no new tensor, and the whole existing suite is still green.

**The design, and the one place it was forced.** Growing NCA does not create cells; the lattice is
complete from step 0 and growth is an *alive mask*. Transplanted here, the substrate is the `N`
residue slots, the sequence edges exist from step 0, and only the coordinates are absent — which
resolves the chicken-and-egg that kills the obvious design, since a residue cannot be placed by the
graph it is about to join. What is *forced*, not chosen: a dormant slot has no coordinates, so
"spatially adjacent to a live cell" is undefined for it, and its only adjacency is covalent. So the
alive set is a union of **sequence segments**, growth extends segments at either end, and several
seeds give nucleation and coalescence. One seed with `cward_only` is synthesis order, which is the
control rather than the model.

**The measurement that shaped it.** Regrowing 1UBQ's backbone from its own measured torsions
(`scripts/growth_probe.py`):

| residues grown | 1 | 2 | 5 | 10 | 20 | 40 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| measured bonds/angles | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **ideal** bonds/angles | 0.026 | 0.155 | 0.564 | **1.472** | 2.767 | 8.464 | 3.276 | **29.433** |

CA error in Angstrom. The measured row is the round trip and is the control on the placement
primitive — exact, so the NeRF is right. The ideal row is what a structure being grown de novo has
to use, because it has no measured geometry to read, and it is **the lever arm at the backbone**:
1.5 A by the tenth residue, tens of Angstrom down the chain. (Non-monotone, because accumulated
error is a random walk that can fold back; read the trend, not the digits.)

That is fatal for a torsion-space relaxation, and it is the reason for the architecture:

* **Birth is by torsion** — a newborn's backbone comes from its parent's frame plus ideal bond
  geometry and three predicted torsions (psi, omega of the parent and phi of the child, C-ward).
  This is the one place a torsion is unavoidable and the one place nothing is downstream, so the
  lever arm at birth is exactly zero.
* **Relaxation is not** — an alive residue relaxes by a bounded **rigid move of its own atoms**,
  axis-angle plus translation, which is EGNN's coordinate update in SE(3) form. No lever arm, and a
  global correction is a local parameter change. The chain is then held together by a
  `peptide_bond_restraint` (C-N length, omega planarity) instead of by construction, which is how
  AF2 and every structure diffusion model do it.
* **The rest of a newborn's atoms ride on a rigid transform** from its parked pose, so bond lengths
  and angles inside a residue stay exactly the crystal's — consistent with v1 freezing them, and it
  avoids needing ideal coordinates for twenty residue types. Only CA lands on the NeRF triad
  exactly; N and C follow the residue's own geometry, so the realised peptide bond is ideal plus
  that residue's deviation, **measured 0.007 A on 1UBQ**. The alternative moves the same
  discrepancy to the CB attachment; keeping it in the peptide bond is better because that is what
  the restraint supervises.

**Three heads, three jobs, and all three get gradient** (asserted in
`tests/test_growth.py::test_every_growth_head_receives_gradient`): `activation_head` (2 logits,
N-ward and C-ward), `birth_head` (3 torsions per direction as cos/sin pairs), `frame_head` (6, the
rigid move). The `birth_head` bias is set to phi −63, psi −43, omega 180, so **an untrained rule
grows an alpha helix**. That is the right analogue of the delta head's identity initialisation:
there is no "do nothing" for a newborn, so the neutral choice is the most populated region of the
Ramachandran plot rather than a random one.

**A dead gradient, found by asserting instead of assuming.** The frame head is zero-initialised and
its output is bounded by scaling the vector to keep the rotation axis intact — and
`limit * tanh(0) / 0` evaluates to `0 / eps = 0`, which multiplied the whole output by zero and
left the head with **exactly 0.000e+00 gradient, on every update, forever**. The factor's true
limit at the origin is 1, so that is what it is now told to be. Same class of bug as zeroing the
delta head's bias and getting `atan2(0, 0)`, and it would have looked like a head that had simply
learned nothing.

**The activation head's credit signal.** A newborn's coordinates are
`born * placed + (1 - born) * parked` with `born` a **straight-through Bernoulli**, so the gradient
reaching the activation logit is `(placed - parked) · dL/dx` — "waking this neighbour changed the
loss by this much". That is the same device as the identity head's straight-through one-hot and it
is what makes growth *order* trainable rather than merely sampled; a hard decision gives no
gradient at all.

**The degenerate optimum, found and closed.** FAPE is a mean over the residues that exist, so
growing one more residue can only add error: under FAPE alone the cheapest structure is the seed,
alone, forever, and nothing in the geometry objects. `completion_weight` is therefore load-bearing
rather than a nudge — a cross-entropy on the frontier's activation logits towards waking, evaluated
at the final state only, so it is exactly zero once the structure is complete. The two signals are
different and both are needed: this one says *grow*, the straight-through path says *that birth
helped*. Measured at initialisation the completion term is 2.13 against FAPE 0.56, and the
activation head's gradient is 4.8e2 against the birth head's 2.1 — so early training is dominated
by learning to grow at all, and `completion_weight` will need lowering once it does.

**FAPE, not a torsion loss, and taken rather than invented.** AF2 eq. 10, clamp 10 A, scale 10 A
(`aanca/fape.py`, pinned in a test so a tuning pass has to be deliberate). A hinge that rotates
half the protein leaves every bond, angle and torsion inside each residue untouched — so a torsion
loss cannot charge for a wrong fold, and a coordinate loss cannot be computed without choosing a
superposition first. `backbone_rmsd` is reported alongside because FAPE saturates at its clamp.

**The untrained baseline, 60 steps on 1UBQ, mean of 3 seeds:**

| seeds | mode | alive | bond | fape | CA rmsd |
|---|---|---|---|---|---|
| 1 | C-ward only | 6.1 % | 0.0001 | 0.365 | 2.33 |
| 1 | both ways | 8.3 % | 0.0022 | 0.470 | 3.43 |
| 2 | both ways | 18.9 % | 0.0012 | 0.658 | 5.24 |
| 4 | both ways | 34.6 % | **0.4828** | 0.741 | 5.60 |
| 8 | both ways | 61.8 % | **4.5100** | 0.716 | 5.18 |

**The loop-closure problem, predicted and now measured.** `bond` is the peptide restraint,
unweighted, and a crystal scores ~0.002. It is fine at one and two seeds and explodes at four and
eight — 4.51 at eight seeds. That is coalescence: two segments grown from different seeds meet in
the middle and nothing closes the gap between them. The restraint term is the *detector* and an
untrained rule has no way to act on it. So the honest statement of where growth stands is: single
nucleation works and the chain stays intact; multiple nucleation is the interesting model and its
open problem is loop closure, with a number attached.

**Still open, recorded rather than discovered later.** A grown structure is a sample from whatever
the rule happens to do, with no density attached, so "grown" still says nothing about whether the
result is a protein — the same gap as "a fixed point is not an energy", and what the Potts and MH
machinery is for. Nothing forbids a mirror fold, because the heads see only distances. Side chains
are deliberately out of `growth_objective`: `training_objective` already supervises chis, and
joining the two objectives should follow a measurement rather than a code layout.

### The no-op term is already gone, so the fix is the clamp — 2026-11-03

Correcting the previous session's prescription, which was wrong about its own premise. The
"per-residue no-op" it proposed cannot be built, because **the no-op term is switched off** —
`--noop-weight 0` since v5, measured vacuous by `scripts/noop_probe.py`, with
`--native-probability` carrying the fixed point instead. Worse, the per-residue part was never
missing: on a partially damaged sample the **reconstruction term is already per-residue**, asking
the damaged residues to move and the undamaged ones to stay, in the same step. `--damage-fractions`
has been doing exactly that since v6, which is why it helped (−20.9 → −11.9) — and the fact that it
did not *fix* the sign is the evidence that per-residue supervision was not the gap.

**What is actually missing is the immobility.** In training the undamaged residues sit at native
but are *free*, so they drift as the rule works. Under the clamp they are exactly native and do not
move at all. "My neighbours are correct **and they have stopped**" is the state the rule has never
seen, and uniform damage essentially cannot produce it: with 64 neighbours and damage fraction `f`,
an all-native shell around a damaged residue has probability `(1 − f)^64` — 4 % at `f` = 0.05 and
6e-7 at `f` = 0.2. Freezing the undamaged residues produces it on every sample.

Implemented as `--clamp-probability` (`training_objective(frozen=...)` → `rollout(frozen=...)`,
which `step_once` has supported since the clamp test was built; `Sample.damaged_mask` is read off
`deltas` so it cannot disagree with the damage that was applied). A frozen residue keeps ticking
and keeps being scored, so it is still charged for proposing a non-zero delta — the silence
supervision now happens in the presence of correct *immobile* neighbours, which is the whole point.
Pinned in the checkpoint fingerprint, absent from `WARM_START_MUST_MATCH`, so v9 warm-starts from
v6/v7 the way v6 did from v5. Progress lines print a `clamped` count so a run that is quietly
clamping nothing is visible.

**The cost, stated rather than hidden:** a run trained this way has seen the clamp test's
distribution, so **the clamp table stops being a held-out diagnostic for it**. The held-out claim
moves to global repacking accuracy, which must not regress. Report both or the result is
unreadable.

### Comparing v8_if with PottsMPNN — the numbers to report

Decided earlier and worth making concrete now that inverse folding runs. **Native sequence recovery
is not the headline.** PottsMPNN traded NSR down (~44 % against ProteinMPNN's ~58 %) for refolding
self-consistency and ddG correlation, so beating 44 % on NSR could just mean we are a
worse-calibrated ProteinMPNN. What to report, in order:

1. **Joint recovery** — sequence recovery *and* chi accuracy from a jointly scrambled start, on the
   held-out split. This is the number that is ours: nobody reports it because nobody's model decides
   both at once. PottsMPNN has no chi number to compare, which is exactly the point.
2. **NSR, reported anyway**, split core/surface, against PottsMPNN's ~44 % and ProteinMPNN's ~58 %
   on the same set. Not as a win condition — as the calibration check that says whether the identity
   head is in the right regime at all.
3. **Self-consistency** — design, fold, RMSD. The comparison PottsMPNN itself leads with.
4. **Label churn** — the fraction of positions whose label changed in a sweep, which must decay to
   zero. The sequence analogue of the `|delta|` convergence curve, and the diagnostic that says
   whether the ripple settles or oscillates; PottsMPNN's local optimisation sweeps "until no
   favourable substitution remains", which is the same condition stated discretely.
   `identity.label_churn` already computes it.

Sequencing: (1) and (2) need only `v8_if`'s checkpoint and the existing eval set, so they come
first. (3) needs a folding model and is the larger job. The set must be the same one the packing
benchmarks use, or none of it is comparable with our own earlier numbers either.

### Cartesian birth — a newborn placed from a frame, no torsion anywhere — 2026-11-03

`GrowthConfig.birth` and `AANCARule(birth=...)` select `"torsion"` (the NeRF, as built earlier) or
`"cartesian"`. Under Cartesian birth the parent emits a **frame** and the newborn's backbone is that
frame carried to a canonical ideal triad — no NeRF, no predicted torsions, and critically **nothing
about either residue's own internal geometry**, so the placement is type-independent. That is the
property the CB-level representation needs, and it arrives as a side effect rather than as extra
work.

**Two measurements set the parameterisation, and both overturned the obvious choice.**

*The CA–CA distance is not a free parameter.* It is **3.804 Å for every (phi, psi)** of a trans
peptide — fixed by the peptide's own bond geometry, not by the torsions. So a birth chooses the
*direction* of the child's CA and not its distance. The offset is therefore the reference offset
**rotated**, and a rotation preserves norm, so no head output however large can tear the peptide at
the moment of birth. Only relaxation can, which is what the restraint is for.

*The rotation residual has to be multiplicative.* Axis-angle is not a vector space. Measured from
the helical reference, C-ward:

| | helix | sheet | ppII | left-helix | bridge |
|---|---|---|---|---|---|
| additive axis-angle difference, rad | 0.00 | 4.60 | 3.36 | 2.77 | 0.30 |
| **geodesic rotation**, rad | 0.00 | 1.86 | 3.07 | 2.62 | 0.29 |
| offset direction, rad | 0.00 | 0.71 | 0.71 | 0.50 | 0.26 |

An additive residual would have to span 4.6 rad to reach beta sheet and is discontinuous near π; the
geodesic is 1.86. So the head's rotation output **composes onto** the reference rather than adding
to it. The same table sets the bounds: `MAX_BIRTH_DIRECTION = 1.0` rad covers every one of those
secondary structures with room to spare, while the child's **orientation is deliberately uncapped** —
polyproline II sits 3.07 rad from helix, nearly antipodal, so any cap tight enough to matter would
exclude real secondary structure, and a rotation is intrinsically bounded anyway.

**A second silent dead gradient, and it was worse than the first.**
`torch_geometry.rotation_matrices` normalises its axis. That is right for the chi path, where the
axis is a real bond and only the angle is zero, and **wrong when the vector itself is the learned
parameter**: at initialisation the axis is exactly zero, `axis / clamp(norm)` is zero, and `sin(0)`
multiplies whatever survives, so the map is the identity with no derivative. The Cartesian birth
head's total gradient was exactly `0.000e+00`, and so was **the rotation half of the frame head** —
which had been dead since the frame head was written, hidden because the translation half still
carried gradient and made the total look healthy. `backbone.rotation_from_vector` is the standard
stable Rodrigues form, `R = I + aK + bK²` with `a = sin t / t`, `b = (1 − cos t)/t²` built from the
**unnormalised** vector, whose coefficients are analytic at the identity (a → 1, b → ½). Measured:
5.68 against 0.00 for the same functional. Pinned in `tests/test_backbone.py`, with the old failure
asserted alongside so the test explains itself.

Both of these were found by asserting that every head receives gradient rather than by watching a
loss curve. Neither would have produced a symptom — a head that cannot learn looks exactly like a
head that has not learned yet.

**The two modes are initialised to the same birth, which is what makes the comparison a comparison.**
Both start at the helical reference. On a parent whose own internal geometry is ideal they agree to
float precision (3.8e-15, asserted). On a crystal parent they differ by **0.08 Å**, and that number
is accounted for rather than merely small: torsion birth reads the parent's measured N, CA and C
through the NeRF, Cartesian birth reads only the parent's frame, and residue 20 of 1UBQ has
N-CA-C = 109.84° against an ideal 111.00° — 1.16° at a 3.8 Å lever is 0.077 Å.

**Untrained baseline, 60 steps on 1UBQ, mean of 3 seeds:**

| birth | seeds | mode | alive | bond | fape | CA rmsd |
|---|---|---|---|---|---|---|
| torsion | 1 | C-ward only | 6.1 % | 0.0001 | 0.365 | 2.33 |
| torsion | 1 | both ways | 8.3 % | 0.0022 | 0.470 | 3.43 |
| torsion | 2 | both ways | 18.9 % | 0.0012 | 0.658 | 5.24 |
| torsion | 4 | both ways | 34.6 % | 0.4828 | 0.741 | 5.60 |
| torsion | 8 | both ways | 61.8 % | 4.5100 | 0.716 | 5.18 |
| cartesian | 1 | C-ward only | 6.1 % | 0.0050 | 0.361 | 2.27 |
| cartesian | 1 | both ways | 8.3 % | 0.0061 | 0.468 | 3.40 |
| cartesian | 2 | both ways | 18.9 % | 0.0057 | 0.653 | 5.19 |
| cartesian | 4 | both ways | 34.6 % | 0.4892 | 0.740 | 5.58 |
| cartesian | 8 | both ways | 61.8 % | 4.6282 | 0.715 | 5.14 |

`alive` is identical to the digit, as it must be — same activation head, same random stream. FAPE
and RMSD agree to 0.005 and 0.05, which is the 0.08 Å per-birth difference above accumulated and
then swamped by the helix both modes grow.

**The prediction was directionally right and quantitatively an overestimate, which is worth
recording as a miss.** It was that Cartesian birth would make the `bond` column worse *at every*
seed count, because nothing would be correct by construction any more. The direction holds — 0.0050
against 0.0001 at one seed, 0.0057 against 0.0012 at two — but the size is a factor of a few at
crystal scale, not a torn chain. The reason is that the parameterisation ended up keeping the one
invariant that genuinely is invariant, CA–CA = 3.804 Å, by construction. "Nothing correct by
construction" was the wrong summary of what Cartesian birth does.

**What this does not yet change.** A newborn's non-backbone atoms still ride on a rigid transform
from its parked pose, so it is still born as its native amino acid: the *placement* is now
type-independent but the *atom set* is not. Joining the identity head and growing at
backbone-and-CB resolution is the step that removes the leak, and it is now cheap, because nothing
in the placement path reads the child's type.

**The state-of-the-art comparison has moved and this is closer than I said.** An agent search
reports that the current Baker-lab/NVIDIA model, **RFD4-Proteina** (2026-10, project-site PDF, repo
`RosettaCommons/RFD4-Proteina`, weights on HuggingFace, ~0.4 B params), does **not** diffuse SE(3)
frames. Its state is **per-token anchor-atom coordinate (CA for amino acids) plus a 32-dim latent**,
generated jointly by **flow matching** with a stochastic TSR SDE at 60 steps, and it **co-designs
sequence** — the decoder emits all-atom structure and sequence together, with no ProteinMPNN handoff,
and sequence recovery against the final design rises from ~50 % to 97 % along the trajectory, i.e.
identity is still being decided mid-generation. Length is still fixed per sample (materialised from
a contig string, padded token count throughout) and the network is still explicitly time-conditioned
(separate 256-dim coordinate and latent time embeddings via adaLN), so the two differences that
matter to this project are unchanged. Also: RFdiffusion3 exists and I had missed it (bioRxiv
2025-09-18, all-atom, ~168 M params), and RFD4's own introduction says RFD3 "does not design
high-quality sequences for its generated structures". **Unverified:** that "RFD4" ever stands for
"RFdiffusion4" — nothing states it — and no bioRxiv/journal DOI was found for RFD4. Treat the
numbers as agent-reported until read from the PDF directly.

Two consequences for this project, stated now rather than acted on. First, "CA coordinate plus a
per-residue latent" is a *third* option alongside frames and torsions, and an AANCA already has the
latent — the 96-dim hidden state. The orientation would then be implicit in the latent rather than
explicit in a rotation, which is strictly more general and strictly harder to restrain. Second, the
field's own direction is towards joint sequence-structure generation with identity resolved
*during* the trajectory, which is an independent argument for joining the identity head to growth
next.

### Identity joined to growth — the sequence is grown too — 2026-11-03

`growth_objective(identity_weight=...)` with a rule built `AANCARule(growth=True, identity=True)`
and a `RolloutConfig(blind=True, growth=...)`. This is the step that makes growth de novo rather
than regrowth of a known chain, and most of it was already built — `blind.py` collapses side chains
onto CB, `identity.type_loss` marginalises the library over chi1 wells, `step_once` gates geometry
and labels through the same masks. What had to be added was three things, and each of them closes a
leak rather than adding a mechanism.

**A newborn arrives unlabelled.** `initial_identity` gained an `alive` argument that forces every
dormant slot to the all-zero unknown token *whatever* `identity_start` says — including `"native"`,
which is the one that would hand over the answer, and `"random"`, which is subtler: a random label
is still a commitment the rule never made. A newborn inherits its slot's row, so the first thing it
does once it can perceive anything is draw a label from its neighbourhood. That is the ripple.

**A blind growth rollout holds no side chains at all**, not merely none it can see. This was the
real leak and it is worth being explicit about why masking perception was not enough: a newborn's
non-backbone atoms arrive by rigid transform from its parked pose, so they are its *native* amino
acid's atoms sitting in the state — invisible to the graph under `blind`, and still read by the
clash term, by FAPE if it ever scored them, and by any later reader who forgets. A blind growth step
now collapses them onto CB in the **state**. Gated on growth and not on `blind` alone, because a
blind *packing* rollout is placing side chains and must keep them. Asserted two ways: collapsing the
grown structure again changes nothing, and — the headline — **randomising every native chi before
growth leaves the grown backbone bit-identical**, which is the growth analogue of
`tests/test_blind.py`.

**The loss is on the type marginal, not the joint label.** Growth runs at backbone-and-CB
resolution, so there is no side chain to carry a chi1, and supervising the joint (type, chi1 well)
entry would ask the rule to commit to a conformation the structure cannot express. `type_loss` was
already there for exactly this. The joint head is unchanged and its well half is simply unsupervised
during growth, which is the right division: packing supervises it already. The floor is **ln 20 =
3.00** rather than ln 54 = 3.99, and an untrained head measures 3.13.

**A structural consequence, found by asserting and worth knowing.** Under blind growth the **chi
head receives exactly zero gradient**, and that is correct rather than broken: with every
side-chain atom collapsed onto CB, the chi1 axis CA-CB passes through all of them, so a chi rotation
is the identity map. It confirms the representation really is backbone-and-CB. It also means
**growth and packing cannot share one rollout** — in the representation growth needs there is
nothing for the packing head to move — so composing them is a two-stage pipeline, not a joint
objective. That answers a question the previous section left open by putting it off.

**Untrained baseline, 60 steps on 1UBQ, Cartesian birth, mean of 3 seeds:**

```
background composition: 11.8% (LEU, the most frequent type)

  birth       seeds   mode         alive      bond      fape   CA rmsd  recovery
  cartesian     1   C-ward only    6.6%    0.0055    0.3263      1.79     16.7%
  cartesian     1   both ways     11.0%    0.0085    0.5037      3.97      9.4%
  cartesian     2   both ways     21.5%    0.0067    0.7168      6.26     13.5%
  cartesian     4   both ways     38.6%    0.0539    0.7468      6.10     10.2%
  cartesian     8   both ways     56.6%    3.8697    0.7206      5.22      7.0%
```

**Read the recovery column against the background, not against zero.** 1UBQ is 11.8 % leucine, so
"guess LEU everywhere" scores 11.8 %, and the untrained rule's 7–17 % is **at or below chance** —
which is what a zeroed identity head must give. The spread is small-sample noise: at 6.6 % alive
there are five residues, so one correct is 20 %. The probe prints the background for this reason;
without it the column reads as though an untrained rule were already doing something.

The number to beat, from the comparison plan recorded earlier: ProteinMPNN ~58 %, PottsMPNN ~44 %,
and whatever RFD4-Proteina reports. Those are on large held-out sets with a given backbone, so a
single-protein de novo number is not comparable to them yet — it is a plumbing baseline, and the
comparable run needs the held-out split.

**Still unbuilt, deliberately.** Nothing materialises a side chain from a committed label; that
needs either ideal coordinates per residue type, which this project does not have, or a second
packing pass over the grown backbone, which it does. The output of growth-with-identity is a
**backbone and a sequence** — which is exactly what RFdiffusion plus ProteinMPNN produce between
them, and what RFD4-Proteina produces directly — and placing the side chains afterwards is the
existing rule's job. The composition is the next measurement, and the finding above says it has to
be a pipeline.

### Coalescence, measured — and the bug the measurement found — 2026-11-03

`scripts/coalescence_probe.py`. Grow 1UBQ from several seeds, then look at every covalently bonded
pair that is alive at both ends and ask what its realised C-N distance is. A pair made by a *birth*
sits at the peptide bond's 1.33 Å. A pair where two independently grown segments merely became
sequence-adjacent has never been placed by anything.

```
 seeds   alive  segments  bonded  joined  junctions  worst gap  median gap
     1   8.3%       1.0     5.3     5.3        0.0          -           -
     2  18.9%       1.7    12.7    12.7        0.0          -           -
     4  34.6%       3.0    23.3    23.0        0.3        7.5         7.5
     8  61.8%       3.7    43.3    40.3        3.0       14.7         9.0
    16  92.1%       3.3    66.7    58.3        8.3       23.3         7.5
```

`joined` is pairs within 2 Å — real peptide bonds. `junctions` is pairs beyond it. **One and two
seeds produce no junctions at all**: every bond is real and the chain is intact, which is why the
restraint reads ~0.002 there. From four seeds up, junctions appear and their median gap is 7–9 Å,
which is where the restraint's 0.5 and 3.9 come from. So the restraint was never measuring poor
geometry; it was counting unclosed loops, and now the count is explicit.

**Why nothing closes them.** `frontier` computes `cward[i] = alive[i] & links[i] & ~alive[i+1]`, so
once `i+1` is alive from the *other* seed, `i` is no longer on any frontier and growth simply stops
there. The pair is adjacent in sequence, alive at both ends, and was never placed relative to each
other. The only tool for moving an already-alive residue is `rigid_relax`, bounded at 0.1 rad and
0.3 Å per step, and closing a 9 Å gap needs a **coordinated** motion of a whole segment that no
local rule produces: each residue's restraint gradient says "move toward your sequence neighbour",
which also pulls it away from its own segment-mates, who are pulling back. It is the classical loop
closure problem, which is a global constraint, and bounded local updates are bad at those.

**The diagnosis is actuation, not perception.** Residue `i` can already see where `i+1` is —
sequence neighbours are *forced* into each other's neighbour lists (`select_neighbours`'s
`include_sequence_neighbours`), both are alive, so the edge is real and carries the distance. The
rule knows the gap exists and how big it is. What it has no way to do is move a hundred residues
together. So the fix is not a feature; it is either closure-aware growth (bias births *toward* a
nearby frontier, so segments meet with a short gap that a few residues can bridge) or a dedicated
non-learned closure step at the end, which is what fragment assembly does.

**This matters more than a defect usually would**, because multi-seed growth was the whole argument:
nucleation and coalescence is the foldon-like, parallel, O(surface) story, and one seed growing
C-ward is synthesis order — the thing this project argues folding is not. If coalescence cannot be
made to work, multi-seed growth produces several fragments that never become one protein.

**The bug the number found.** The 16-seed worst gap first came out as **13,495 Å**, which is not a
geometry error but a parked residue — `PARK_SPACING` is 500 Å per slot. Refusing to report an
unexplained number turned up a real fault: a dormant slot can be woken from **both** sides in the
same step, which is exactly what closing a one-residue gap looks like. `give_birth` reads the
backbone once from the incoming coordinates, so the second placement's *source* triad was the
child's parked one while its atoms had already been moved by the first — and `rigid_onto_triad` then
subtracts a CA hundreds of Ångström away. Fixed by letting one side win and leaving the other as a
junction, which is the honest resolution: two segments meeting is not a birth, and pretending either
parent placed the child would hide the gap rather than report it. The worst gap is now 23.3 Å, a
real junction. Pinned in `tests/test_growth.py`.

Worth noting what did *not* catch it. Every growth test passed, the full suite passed, the untrained
probe's FAPE and RMSD looked ordinary, and the peptide restraint was already large at 8 and 16 seeds
for a legitimate reason — so the bug hid behind a number that was *supposed* to be big. It took
asking what a specific digit meant.

### Growth and packing cannot share one rollout — the three reasons, in order of depth

Recorded because the earlier section asserted this from one measurement and the measurement is the
weakest of the three arguments.

1. **Representation (contingent).** Growth needs a state with no side chains, or a newborn is its
   native amino acid. Packing needs side chains, because they are what it moves. Measured: under
   blind growth the chi head's gradient is exactly zero, because with every side-chain atom on CB
   the chi1 axis CA-CB passes through all of them and a rotation about an axis does not move points
   on it. **But this is dodgeable** — materialise an *idealised* side chain from the committed label
   instead of collapsing to CB, and there is no leak and something to pack. It needs ideal
   coordinates for twenty residue types, the table this project avoided by using rigid transfer. So
   "cannot" overstates it: cannot *yet*, for a reason that is a table.
2. **The atom set is a function of a label that is still moving (a real pathology).** Labels ripple
   throughout the rollout, so a residue's atom set is undetermined while its type is. The joint
   (type, chi1 well) library was designed so a committed entry has a definite atom set, so one could
   materialise on commitment and re-materialise on change — at the cost that every label flip
   discards the packing work done on the old side chain, and the chi head's gradient for a residue
   whose type later changes is noise. Not fatal, and not free.
3. **The chi target is unlearnable where growth is happening (the principled reason).** This project
   already encodes the principle: `topo.supervised_mask` exists to exclude residues whose perception
   is incomplete on a crop, because "its native torsions depend on contacts that were cropped away
   and asking for them is unlearnable rather than merely hard". A growing structure is that case at
   its worst — every residue near the frontier has a radically incomplete neighbourhood and stays
   that way until growth finishes. Supervising chis during growth would therefore supervise only the
   deep interior of the alive set: a small, biased subset that shrinks as the number of seeds rises.
   This argument survives both fixes above, and it is why the answer is a pipeline.

**The honest qualification.** All three are about a representation where side chains are *explicit
Cartesian atoms whose count depends on type*. RFD4-Proteina reportedly carries all-atom information
through its whole trajectory in a 32-dim per-token latent and decodes at the end, with sequence
recovery against the final design rising from ~50 % to 97 % along the way — so it does, in effect,
grow structure and sequence together. A latent has no type-dependent atom count, so reason 1
disappears and reason 2 weakens. Reason 3 does not: an incomplete neighbourhood is incomplete
whatever the state variable. That is the argument for a pipeline that does not depend on this
project's representation choices.

### Revising the "pipeline, not a joint objective" conclusion — 2026-11-03

The previous section's three reasons were challenged and two and a half of them do not hold. The
conclusion is withdrawn: **growth and packing can share one rollout.** What follows is what the
reasons were worth and what the cost actually is, because the revision matters more than the
original claim.

**"Bounded local updates are bad at global constraints" was wrong, and a paper in this project's own
reading list is the counterexample.** EGNCA (arXiv 2301.10497) does pattern formation on a graph
from *random Gaussian coordinates*, with purely local E(n)-equivariant updates and no seed node,
and converges to a target point cloud. That is bounded local updates satisfying a global geometric
constraint. Growing NCA's whole conceit is the same thing. The claim should never have been made in
that form.

What is actually missing is two specific things, and only one of them is information:

* **Information.** Every spatial neighbour is already in the graph, and sequence neighbours are
  *forced* in, so a residue at a junction already sees the gap and its size. What no feature carries
  is the **sequence distance to the nearest alive slot in each direction**. A frontier residue cannot
  distinguish "three slots left to span 9 Å" from "thirty", which is precisely the quantity closure
  needs. It is node-local and cheap -- clip and RBF-encode it, the way every other distance in this
  project is encoded.
* **Credit.** `truncate` is 12 steps and `max_translation` is 0.3 Å per step, so a coordinated
  segment translation closing a 9 Å gap needs about 30 steps of consistent motion by many residues
  -- two and a half truncation windows. The loss cannot credit a manoeuvre longer than its gradient
  horizon, however capable the rule is of performing it.

**So the real virtue of closure-aware growth is not more force, it is a shorter problem.** Aiming
births at a nearby frontier converts the global constraint into a local one: do not close a 9 Å gap,
never open one. A two-or-three-residue bridge fits inside the 12-step horizon, which is the same
argument as "birth has no lever arm because nothing is downstream yet" applied to time instead of
space.

**A newborn at the background amino-acid distribution is not a leak**, and the objection to it was
wrong. A draw from the global composition carries zero information about the native residue at that
position, and it buys something the all-zero unknown token does not: a **definite atom set
immediately**, which is what packing needs in order to have anything to move. The one real caveat is
that a pre-committed label fights the keep-current term -- at keep 8 it is held at 98 % -- and keep
2.0 was already chosen on measurement for masked starts, so this is a tuning matter and not a
principle.

**Flip, pack, flip, repack is the intended dynamics, not a pathology.** It is iterated conditional
modes, which is exactly what PottsMPNN's published local optimisation does ("sweeps until no
favourable substitution remains"), and what this project's own Potts plus Metropolis-Hastings
machinery exists for. The gradient-noise objection raised against it is weak: with final-state
supervision and a 12-step truncation, the gradients for packing work that a later flip discards are
mostly never computed. And the discard is cheaper than it looks if identity errors are
chemically local -- a LEU to ILE flip keeps nearly the same atom set. **That is checkable and should
be checked**: the confusion matrix of `v8_if`'s identity head, blocked by chemical class. If it is
block-diagonal, discarded packing is cheap and the objection evaporates entirely.

**The cost that does remain is engineering, and it is one purchase.** `TorsionTopology` is
documented as depending "only on the sequence, never the coordinates" and is resolved once into
index tensors; a flipping label makes `atom_mask`, `chi_mask`, `chi_atoms` and `downstream`
*label*-dependent. The tractable design is a gather: precompute those tables per library entry,
shape `(54, ...)`, and index them by the committed label each step. Masks are never differentiated,
so a gather costs nothing, and nothing has to be rebuilt. The same table purchase supplies the ideal
per-type coordinates needed to materialise a side chain for a newly chosen type -- so it is one
table, not two, and it is the same table that would let Cartesian birth place a side chain at all.

**Reason 3 was the principled one and it half holds -- as a design requirement, not a fact.** The
objection was that a growing structure has incomplete neighbourhoods where growth is happening, so
the chi target is unlearnable there. The answer given was that growth finishes quickly and chis
settle afterwards, and `loss_window = 0.5` already supervises only the back half of the rollout, so
if growth finishes in the first half the objection dissolves with no new machinery. Correct in
principle. Measured (`scripts/growth_rate_probe.py`, 1UBQ, 120 steps, mean of 3 seeds):

```
  bias  p(wake)  seeds  max_births  complete at  fraction  final alive
  -2.0     0.12      1           8         >120         -       14.5%
  -2.0     0.12      4           8         >120         -       57.5%
   0.0     0.50      1           8         >120         -       62.7%
   0.0     0.50      4           8   >120 (2/3)         -       98.7%
   2.0     0.88      1           8   >120 (1/3)         -       84.2%
   2.0     0.88      4           8           54       45%      100.0%
   2.0     0.88      4           2           50       41%      100.0%
   4.0     0.98      4           8           46       38%      100.0%
```

So: **at the current default growth never finishes at all** -- 14.5 % of 1UBQ in 120 steps at one
seed. At activation bias 2 to 4 with four seeds it completes at 38-46 % of the rollout, just inside
the loss window. `ACTIVATION_INIT = -2.0` is therefore wrong for a joint objective and should be
positive; -2 was chosen so an untrained rule would not wake the whole frontier at once, which is a
first-step concern and not a schedule.

**And the two problems are coupled, which is the finding here.** One seed never reliably finishes
even at p(wake) = 0.98, because a single segment's longer arm averages ~57 residues and the
asynchronous mask halves the birth rate, so it needs ~114 steps. More seeds means shorter arms means
faster completion. **So multi-seed growth is forced by the completion-time requirement** -- it is
not an optional elaboration of the interesting model. Which makes coalescence mandatory rather than
a feature to defer, and moves closure-aware growth from "next idea" to "prerequisite".

What does not dissolve: the *permanent* incompleteness of a crop's boundary shell, which is what
`supervised_mask` is for and is orthogonal to growth. And during the first half the chi head is
still *acting* on incomplete neighbourhoods even though it is not supervised there, so its early
output feeds the structure the back half is scored on -- noisy rather than unlearnable, and the
sample pool already trains on states the rule itself produced, which is the same situation.

### Closure-aware growth — the spec, with the reach bound measured — 2026-11-03

Three additions and one loss term. The loss term is the part that matters; the rest is what it needs
in order to have anything to act on.

**1. See the pending closure.** Partly there already. `edge_features`' `sequence_offset` block is a
33-dim one-hot of the residue-id difference clipped to ±16, so the rule can **already identify**
which spatial neighbour is its closure partner, and the backbone RBF block already gives the
distance and `relative_rotation` the orientation. Two things are genuinely missing:

* Beyond the 10 Å perception cutoff there is no edge at all, so a partner 20 Å away is invisible.
  Measured junction gaps are 7–9 Å median (inside the cutoff) but reach 23 Å (outside it). Fix: force
  the nearest-alive-in-sequence residue into the neighbour list, exactly the way ±1 sequence
  neighbours are already forced by `include_sequence_neighbours`. The `sequence_offset` one-hot then
  labels it for free.
* The **index gap** — how many dormant slots lie between me and the next alive one — is never
  directly available. It is inferable from a partner's sequence offset, but only when that partner is
  within the cutoff, which is the case that is already fine. Fix: a node feature, the index gap in
  each direction, clipped and RBF-encoded. It depends on no spatial quantity at all, which is the
  point.

**2. Budget it — the reachability bound, measured rather than assumed.** Max CA(i)–CA(i+g) distance
over two real proteins:

| g | 1 | 2 | 3 | 5 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| 1UBQ max, Å | 3.85 | 7.17 | 10.36 | 16.76 | 25.52 | 35.43 | 35.12 |
| 193L max, Å | 3.84 | 7.14 | 10.42 | 16.81 | 22.56 | 23.77 | 28.36 |
| per residue | 3.85 | 3.58 | 3.46 | 3.36 | ~3.0 | ~2.5 | ~2.0 |

The per-residue reach falls from 3.85 Å at one bond to ~3.4 Å by g = 3 and keeps falling, and the two
proteins agree to 0.06 Å up to g = 5. So the usable bound in the regime that matters is **about 3.4 Å
per spare slot**, and the headline consequence is sharp: **closing a 10 Å gap needs about three spare
slots.** The ±16 clip is not binding — g = 16 reaches 35 Å.

Against that, the junctions actually produced have **g = 0**: two segments adjacent in sequence, both
alive, nothing between them, and 7–9 Å apart. Zero budget against a 9 Å gap. Stopping one residue
earlier gives 3.85 Å of reach, two gives 7.2, three gives 10.4. So the failure is not that growth
aims badly, it is that growth **spends the entire budget before it needs it**.

**3. Aim it.** `MAX_BIRTH_DIRECTION` is 1.0 rad of swing around the helical reference and 0.72 rad
spans helix, sheet and polyproline II, so the freedom to steer already exists in the
parameterisation. Nothing is missing except a reason.

**4. The reason: a reachability restraint.** For each pending closure with index gap `g` and
end-to-end distance `d`, charge `relu(d − reach(g))`. Its properties are why this is the design and
not an extra term:

* Exactly zero for single-seed growth, and zero early in multi-seed growth — far apart but plenty of
  slots left.
* It bites the instant a birth commits to a geometry that **forecloses** closure, which makes it a
  *leading* indicator. The peptide restraint is a lagging one: it only fires once the junction has
  formed, by which point no local move can fix it.
* The gradient lands on precisely the two decisions that could have avoided the problem: the birth
  offset direction (aim) and the activation logit (do not wake yet, you are spending budget).
* It arrives within a step or two of the bad birth, well inside the 12-step truncation window. That
  is "convert the global constraint into a local one" made concrete, and it is the same move as
  "birth has no lever arm because nothing is downstream yet", applied to time.

This is the clash term's logic — charge for a geometry that forecloses an outcome rather than one
that is already wrong — which is a symmetry worth keeping in mind when weighting it.

**What stays hard.** The final bridge still has to be near-exact, and a local rule will get it
approximately, so expect residual strain at junctions; the honest standard is "comparable to a
crystal", since the steric preflight measured real structures at up to 0.000430 Å/atom of strain
rather than zero. And genuine unreachability remains possible: if the seeds fall such that the native
fold requires an approach direction growth has already committed to, only a coordinated segment
motion fixes it, and the restraint makes that rare rather than impossible.

**Alternatives considered.** An analytical closure step on each bridge at the end — exact, reliable,
non-learned, what fragment assembly does — is the fallback if the restraint underperforms, at the
cost of a step outside the NCA. Per-segment rigid moves are rejected: a segment is not a local
object, and the locality claim is the project. Single-seed growth is ruled out by the completion-time
measurement. Seed placement that avoids meetings does not scale: at 92 % alive there are ~8 junctions
regardless.

### Closure-aware growth is built — 2026-11-03

`scripts/reach_probe.py`, `aanca.growth.REACH` / `reach` / `pending_closures` /
`reachability_restraint` / `closure_separation` / `closure_features`, the closure block on the node
features, `reach_weight` in `growth_objective`, a `reach` and `n` column on the probe, and
`tests/test_closure.py`. 790 tests.

**The reach bound, measured properly.** 391 structures, ~188,000 pairs per separation, CA–CA within
a bonded run only — a gap that crosses a chain break or a numbering discontinuity is not a closure.
Taken at the **99.99th percentile, not the maximum**, and the reason is in the data: the maximum at
g = 1 is 4.32 Å, which is no peptide bond at all, since trans is 3.80 and cis 2.9. That pair is a
broken model or a chain break with contiguous numbering. The percentile still keeps the nineteen
longest of 188,000.

```
REACH = 0, 3.96, 7.44, 10.96, 14.37, 17.75, 21.09, 24.43, 27.74, 31.03,
        33.98, 37.18, 40.14, 43.00, 45.76, 48.56, 50.64, 53.33, 54.98, 57.24, 59.90
```

3.96 Å for one bond falling to 2.99 Å per residue by twenty, extrapolated beyond the table at the
last step's slope. Generous is the safe direction for a feasibility test — it never calls a closure
impossible that a real chain could have made. The earlier two-protein estimate of ~3.4 Å per slot
was roughly right but the max was contaminated; the headline stands, **a 10 Å gap needs about three
spare slots** (`REACH[3]` = 10.96).

**The restraint.** `relu(d − reach(g))` averaged over pending closures, where a pending closure is
two alive residues with nothing but dormant slots between them and an unbroken covalent path across.
Taking *consecutive* entries of the sorted alive set makes "nothing but dormant slots between" true
by construction, so only the covalent path has to be checked. A pair at separation 1 is included and
needs no special case: a bond made by a birth sits at 3.80 Å against `REACH[1]` = 3.96, so a real
peptide bond charges exactly nothing.

Measured on untrained growth, 60 steps on 1UBQ, mean of 3 seeds:

| seeds | alive | bond | **reach** | n closures | fape | CA rmsd |
|---|---|---|---|---|---|---|
| 1 (C-ward) | 6.1 % | 0.0050 | **0.0000** | 4 | 0.361 | 2.27 |
| 1 | 8.3 % | 0.0061 | **0.0000** | 5 | 0.468 | 3.40 |
| 2 | 18.9 % | 0.0057 | **0.0000** | 13 | 0.653 | 5.19 |
| 4 | 34.6 % | 0.4892 | **0.0728** | 25 | 0.740 | 5.58 |
| 8 | 61.8 % | 4.6282 | **0.4386** | 46 | 0.715 | 5.14 |

It is exactly 0.0000 at one and two seeds where the peptide restraint still reads 0.005–0.006 of
bond-length noise, so **it separates "intact chain" from "unclosable" where the peptide restraint
cannot**, and it is in interpretable Ångström of excess rather than squared distance. Zero on a
crystal structure and on any alive subset of one, asserted. **On by default**, unlike every other
optional term here, because multi-seed growth is required rather than optional and it costs nothing
when nothing is pending.

**The credit path, confirmed.** With only the reach term weighted, gradient reaches the activation
head, the birth head and the frame head. The activation head's gradient is the largest by orders of
magnitude — 3.3e4 against the birth head's 98 on one run — which is the term telling it loudly that
*waking a residue is what spends the budget*. That is the decision the peptide restraint can never
inform, because by the time it fires the budget is gone. Weighting will matter.

**The perception half: the count is a feature, the partner is not yet.** `closure_separation` gives
each residue the sequence distance to the nearest alive slot in each direction, one-hot over
`CLOSURE_BINS` for 24 node dimensions, so a growth rule's `node_dim` is 56 against a packing rule's
32 (applied automatically when `node_dim` is the default; an explicit value still wins, and
`CLOSURE_DIM == GROWTH_NODE_EXTRA` is asserted so the two modules cannot drift). This cannot be
inferred from the graph: the slots in between have no atoms and are not in it, so the only route is
reasoning over the *absence* of edges at the intervening offsets, which is what sum-aggregation
message passing does worst.

**What is deferred, and the number that says when it bites.** Forcing the closure partner into the
neighbour list the way ±1 sequence neighbours already are. The partner is visible when it is inside
the 10 Å cutoff, and the restraint fires at `d > reach(g) ≈ 3.5g`, so fires-and-visible holds for
g ≤ 3 — which is the regime where closure is still actionable. For g ≥ 4 the restraint fires only
beyond ~14 Å, where the partner is invisible, so the rule gets **gradient without perception**: it
can still act, through its own coordinates, but it learns a reflex rather than a policy. That is the
honest remaining gap, and the fix touches `select_neighbours`, its device twin and `build_graph`, so
it is its own change.

**One message fixed in passing.** A growth rollout on a packing rule used to die with "mat1 and mat2
shapes cannot be multiplied", because the widened node features hit the first linear layer before
the missing-growth-heads check. The check moved to the top of `step_once`.

### The junction gap is accumulated drift, r = 0.970 — 2026-11-03

`scripts/drift_probe.py`. The closure problem has been quoted at 7–9 Å of junction gap for several
sections now, as though that were its magnitude. It is not: it is a measure of how badly an
*untrained* rule places residues, and the correlation says so almost perfectly.

A parent–child pair sits at exactly 3.804 Å because that is what a birth sets, so the cross-segment
junctions are precisely the adjacent alive pairs that are *not* at 3.804. Over 8 runs at 6 seeds,
22 such junctions:

```
  junction CA-CA gap      mean   8.72 A  median   5.84  max  18.57
  sum of the two drifts   mean   6.28 A  median   3.32  max  18.05

  correlation between the gap and the summed drift: r = 0.970
  fraction of junctions already inside REACH[1] = 3.96 A: 31.8%
```

And the drift itself, as CA error against native by births from the nearest seed:

| births from seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| mean error, Å | 0.00 | 1.25 | 5.06 | 7.51 | 8.13 | 9.68 | 10.13 | 11.68 |

About 2 Å per birth for an untrained rule that grows alpha helices from its initialisation. Two
segments drift independently, so where they meet the discrepancy is the sum — which is what r = 0.970
is saying.

**So closure difficulty is a function of per-birth placement error, not a structural obstacle.**
Already at this accuracy **31.8 % of junctions are inside `REACH[1]` and therefore closable**, and
FAPE supervises placement directly. A rule at 0.3 Å per birth would drift ~1 Å by three hops and its
junctions would charge nothing at all. The reach restraint keeps its value, but its role is a
**backstop for when placement is poor**, not the primary mechanism — and that is a better
understanding of it than the one it was built under.

**Where closure *is* structural, and it is only here.** Supervised regrowth seeds at native
positions (`initial_state` parks everything else), so two segments share a reference frame and the
only discrepancy is error. De novo there are no native seed positions, so two independently placed
seeds have a genuinely undetermined 6-degree-of-freedom relative pose, which no amount of placement
accuracy fixes because there is no target to be accurate against. **That** is the gauge freedom, and
it is the real closure problem — confined to unconditional generation.

### Would an N-to-C sweep resolve it? Yes, and it is the wrong trade

Asked and worth recording with the numbers, because the answer is "completely, by construction".
One seed growing one direction is one segment, and one segment has no junctions at all: measured
`reach` exactly 0.0000 at one seed, where four seeds give 0.0728 and eight 0.4386. Three reasons not
to:

1. **It is synthesis order**, which is the thing this project argues folding is not — the causal
   filter, foldons ordered by native structure, and HemK NTD folding at ~5 ms⁻¹ against translation
   at ~3.6 aa/s. Adopting it is retreating to the control.
2. **It cannot finish.** From `scripts/growth_rate_probe.py`: one seed at p(wake) = 0.98 reaches only
   84 % in 120 steps, because a single segment's longer arm averages ~57 residues and the
   asynchronous mask halves the birth rate — ~114 steps for 76 residues, ~600 for a 300-residue
   crop, around twenty times the per-rollout cost. This is the O(surface) versus O(N) argument
   arriving as a wall-clock number.
3. **It is the least autonomous option available.** The order stops being a decision: it is imposed,
   and exactly one cell may act at a time, which is a sequential program rather than a cellular
   automaton. Mordvintsev's lizard does not grow head to tail.

**Three alternatives that keep autonomy, in order of how little they add.**

* **Train it.** r = 0.970 says accurate placement closes junctions by itself, FAPE already supervises
  placement, and the reach restraint is the backstop. Nothing new is needed for supervised regrowth.
* **One seed, bidirectional.** Zero junctions, because it is still one segment; no imposed direction,
  because the seed's position sets the order rather than the N terminus; and both arms advance at
  once, so half the steps of a unidirectional sweep. Already supported and measured at 0 junctions.
  Still O(N) steps.
* **Nucleation by an existing residue** — the fix for the de novo gauge freedom and the most
  autonomous addition. An earlier version of this paragraph called it "nucleation from a spatial
  neighbour", **which is incoherent and worth leaving corrected rather than deleted**: a dormant slot
  has no coordinates, so it has no spatial neighbours, and "spatially near a dormant slot" is
  undefined. That is the same chicken-and-egg that forced the alive-set-is-sequence-segments design
  in the first place — a residue cannot be placed by the graph it is about to join.

  The coherent version: the relation does not exist before the birth, the birth *creates* it. An
  alive residue places a newborn at a slot it chooses **by index**, at a free spatial offset from its
  own frame. Four differences from the mechanism as built, where `frontier` is
  `alive[:-1] & links & ~alive[1:]` and so licenses a birth on **covalent** adjacency alone:

  1. the choice of *which* slot — currently one or two options, `i ± 1`; then a range;
  2. the offset is not a peptide bond — CA–CA is currently pinned to 3.804 Å by construction, while
     a nucleation offset would be free within a cap;
  3. the newborn starts a segment rather than extending one;
  4. and the point: the new segment's pose is fixed relative to its placer, so segments are
     commensurable from birth and the gauge freedom disappears.

  **The trade, which the first version did not price.** This swaps a six-degree-of-freedom geometric
  indeterminacy for a *discrete* choice of index, and a wrong index is a topological error that
  relaxation cannot fix — unlike a wrong offset, which it can. The reach restraint does give gradient
  on whether the resulting closures are feasible, so it is not unsupervised, but it is a harder kind
  of error to be wrong about.

  It would also fix the concrete autonomy deficit that exists right now: `frontier` only extends
  existing segments, so the rule **cannot choose** how many segments there are. Seeds are a
  hyperparameter — `seed_alive` draws random slots and `park_coords` leaves them at their *native*
  coordinates, so even their relative poses come from the crystal rather than from any decision. The
  trade between completion speed, which wants many seeds, and closure difficulty, which wants few, is
  therefore set by hand where it should be learned, and the reach restraint already supplies exactly
  the pressure that would teach it, if the rule had the means to act on it.

**The step count decides whether any of this is needed, and it is arithmetic worth doing before
building.** One seed growing both ways has no gauge freedom at all and nothing to choose, so it is
the cheapest coherent de novo option — its only cost is step count. Births arrive at roughly
`2 × p_tick × p_wake ≈ 1` per step for one segment, so a 384-residue crop needs ~770 steps against
~50 at eight seeds (`N / (2k × 0.49)`). At ~13 ms a step with device neighbour selection that is
~10 s per rollout against ~0.7 s, and 60,000 updates turn 12 hours into 170. Early steps are cheaper
because few residues are alive — which is the O(surface) saving finally showing up as wall-clock — so
call it half. **Multi-seed growth is therefore not a nicety but the difference between a feasible run
and an infeasible one at realistic sizes**, which means de novo growth has to either nucleate or
accept the gauge freedom. Measure the real per-step cost on a growth rollout before choosing.

**One correction.** An analytical loop closure step was called "a step outside the NCA" earlier.
That is wrong on the locality axis: closing a three-residue bridge is a local solve over a seven-atom
span, not a global operation. What makes it un-NCA-like is that it is neither learned nor expressed
as a cell update. So if locality is the priority it is compatible; it is autonomy that it costs.

### Growth as a point cloud, not a chain of slots — the better version — 2026-11-03

The proposal, in its correct form and not the garbled one recorded two sections up: during training the
rule is already shown the CA and CB coordinates of its spatial neighbours; during growth it uses that
training to **add a node to the graph**. The newborn's position comes from what a real protein's
neighbourhood looks like — "my neighbourhood is sparse over there, there should be a residue there" —
rather than from a slot index or a peptide-bond budget.

**This dissolves the closure problem rather than solving it, which is why it is the better idea.**
With pre-allocated slots, residue *i* must bond to residue *i+1*: the partner is predetermined by
index, so two segments that drift apart have a specific unclosable gap, and that gap is what the reach
restraint charges for. In a cloud there is no predetermined partner. **Any** two segment ends may join
if they come within peptide range, so which ones join is decided by geometry rather than violated by
it, and the six-degree-of-freedom gauge freedom stops mattering because there is no correspondence to
be wrong about.

**It also recovers variable length, which the slot design silently gave up.** Growth's advertised
advantage over diffusion was that length is an outcome rather than an input — and then `seed_alive`
pre-allocates `N` slots, which fixes the length exactly as diffusion does. A cloud restores it.

**And it turns the missing distribution into a local one.** "Growth is a sample from whatever the rule
happens to do, with no density attached" has been the standing objection, matching "a fixed point is
not an energy". But "my neighbourhood should look like a real protein's" is a *local* statement over a
10 Å ball, learnable from data, and that is a Markov random field — which is exactly what the Potts
machinery was built for. The cloud formulation makes the distribution problem better, not worse.

**Perception is already in place.** `backbone_edge_block` is the full 5×5 N/CA/C/O/CB distance matrix,
RBF-encoded: 400 of the 575 edge dimensions. The rule has been shown precisely the coordinates the
proposal says it should be shown, on every structure it has ever trained on. What is missing is not
perception but a head that *predicts* them — the rule currently predicts chi deltas, frames, labels
and activation, and is never asked where a neighbour ought to be.

**The measurement, and why it answers a question the scheme does not have to ask.** The obvious worry
is sequence assignment: a density-placed residue arrives with a position and no index, so is the chain
recoverable from the cloud? `scripts/ca_shell_probe.py`, 291 structures:

```
  bonded CA-CA      n= 283066  min 2.75  median 3.81  max 4.32
  non-bonded CA-CA  n=1171242  min 3.08  1st pct 4.36
  non-bonded pairs closer than the longest bonded one: 10028 (8.6e-03 of them)

  CAs within 4.2 A of a residue:  1 -> 1.92%   2 -> 94.05%   3 -> 3.91%   4+ -> 0.10%
  95.97% have exactly one or two, which is what makes the chain a readable path
```

So the chain is *mostly* readable — 96 % of residues unambiguously — but not cleanly: the populations
overlap by 0.86 %, and 4 % of residues carry a third CA at peptide range. Reading a path off the cloud
would need more than a threshold.

**But the scheme never needs to read it back**, because the residue that creates a node knows it
created it. The covalent relation is *recorded at birth*, not inferred — so the chain is built
incrementally and sequence assignment is a non-problem. What the measurement does bound is a different
risk: a newborn placed at peptide range from its parent may land at peptide range from some *third*
residue, and nothing in the geometry distinguishes a real bond from that coincidence. The rate is the
4 % above, and it is the thing a clash-like term would have to suppress.

**What it costs: the loss loses its correspondence.** With slots, grown residue *i* is native residue
*i*, so FAPE and the identity term have a per-residue target for free. In a cloud there is no
correspondence, so a supervised loss needs an **assignment step** first. There is precedent rather
than novelty here: RFD4-Proteina scores motif-token correspondences with an auxiliary head and
resolves them by per-chain Hungarian assignment. For de novo there is no target anyway and the loss is
the local density above; it is supervised regrowth that would need the assignment.

**The cheap decisive next step, which needs no rollout at all.** Train a neighbour-placement head by
**masked-residue reconstruction on native structures**: delete one residue, show the rule the rest, ask
where the missing CA was. One forward pass per residue, no unrolling, no growth machinery — and
directly analogous to the masked-label task inverse folding already runs. The resulting error is the
number that says whether density-driven growth can work at all, because it is exactly the quantity a
newborn's placement inherits. Measure that before building the cloud representation, which is a large
refactor: `TorsionTopology` and `FeatureTopology` are both indexed by a fixed residue count with fixed
sequence ids.

### Threading a cloud works, and the peptide bond does it perfectly — 2026-11-03

`scripts/peptide_shell_probe.py`, 291 structures. The proposal was to thread the chain off a cloud
using the direction of the peptide bond rather than CA proximity, since C(i)→N(i+1) is both short and
*directed* and so fixes the N-to-C orientation a cloud has no indices for. It works, and it is not
close:

```
  bonded C(i)-N(i+1)   n= 141532  min 0.979  median 1.330  max 4.752
    percentiles        0.01% 1.262  0.1% 1.294  1% 1.311
                         99% 1.352  99.9% 1.366  99.99% 1.392
  every other C-N      n= 669759  min 2.509  1st pct 3.089

  directed C-to-N graph at 2 A:
    out-degree  0: 1.0107%  1: 98.9893%
    in-degree   0: 1.0107%  1: 98.9893%
  spurious edges found: 0 against 141532 real bonds
```

**99.98 % of real peptide bonds lie in [1.262, 1.392] Å** — a spread of 0.13 Å — and the nearest
non-bonded C–N anywhere in 670,000 candidate pairs is **2.509 Å**. (The 0.979 and 4.752 are broken
models, the same contamination `scripts/reach_probe.py` found at g = 1; the percentiles are the real
bond.) So the populations are separated by **1.12 Å**, a threshold anywhere in [1.5, 2.5] works, and
at 2.0 Å the directed graph has **zero** false edges and in- and out-degree never above one. It is
exactly a set of directed paths, which is exactly a set of chains.

**The number that matters for the design: ~0.6 Å of tolerance.** A bond sits at 1.33 and the
threshold sits at 2.0, so a newborn's N can be about 0.6 Å from where it belongs and the bond is
still unambiguously readable — up to 1.1 Å before it could be confused with anything. Against that,
`scripts/drift_probe.py` measured the untrained rule at **1.25 Å of error at one birth**. So training
has to roughly halve per-birth placement error for a cloud to thread itself, which is a realistic
target rather than a demanding one.

**A correction.** The previous section worried that 4 % of residues carry a third CA at peptide range,
so "nothing in the geometry distinguishes a real bond from that coincidence". At the C–N level that is
simply false: zero coincidences in 670,000 pairs. The 4 % is a CA-level artefact of a shell that
overlaps, and it does not transfer to the peptide bond. The mechanism is immune to the thing it was
charged with.

**It needs frames, not CA and CB** — which is already what Cartesian birth emits. A frame plus
`triad_from_frame` gives N, CA and C at ideal internal geometry, so the atoms the threading test needs
are produced by the existing birth path without any addition.

**The one distinction that remains: detecting a bond versus declaring one.** A birth already *declares*
its bond — the parent knows which residue it created and in which direction — so for parent-child
pairs threading is bookkeeping and free. Only a join between two independently grown segments would
have to be *discovered*, and discovering it by threshold requires the two ends to land within ~1.1 Å
of a correct peptide bond by accident.

The better route is to make the join an action rather than an inference: a frontier residue emits a
**claim** over its spatial neighbours — which it can see, unlike a dormant slot — for which one is its
C-ward partner, with "none, I will grow a new one" as an option. Then the restraint pulls the claimed
pair into bond geometry, and the C–N test becomes *verification* rather than discovery. That unifies
growth and closure into a single decision, **extend or connect**, which is a better shape than a
growth rule plus a separate closure mechanism, and it makes closure a positive action rather than a
failure mode. One-hot rather than multi-hot, since the measurement says out-degree is never above one.

**The refactor this implies.** `links` stops being topology and becomes *state*: the chain is a
declared structure maintained incrementally as residues are created and claims are made, rather than a
function of fixed residue ids. `sequence_links` is currently derived once from `chain_ids` and
`res_id_list`, and every growth function takes it as a given.

### The hole task — and the baseline that said the first version was wrong — 2026-11-03

`aanca/hole.py`, `scripts/train_hole.py`, `scripts/hole_baseline.py`, `tests/test_hole.py`. The
decisive measurement for growth as a point cloud, and the cheapest in this project: **no rollout.**
One message-passing hop, one head, one loss, 0.25–0.33 s an update on a CPU.

**The task.** Delete one residue from a native structure and ask the residues that could have seen it
to point at the hole. Well posed *because* one residue is missing: a head asked "where are your
neighbours" has nineteen right answers and no way to choose, while a head asked "where is the gap"
has one. Two mask modes — `hole` deletes one residue from an otherwise complete structure, which is
over-determined and an upper bound; `frontier` deletes a contiguous sequence stretch so the hole sits
on the boundary of a partial structure, which is what growth faces. The gap between them is the real
difficulty.

**The first version of the objective was measuring the wrong thing, and a baseline found it.** The
loss was an unweighted per-witness Huber and the reported metric was the mean per-witness error.
Training crawled from 6.93 to 6.50 Å over 2,662 updates, which looked like slow learning. It was not.
`scripts/hole_baseline.py` on 60 structures:

```
  mask           zero  anti-centroid  least-occupied
  hole         7.20 A        10.53 A         10.73 A
  frontier     7.04 A         8.96 A          9.55 A
```

**Every closed-form predictor scores worse than emitting zero.** That is not a weak baseline, it is a
broken metric. Witness distances are median 7.6 Å with only 30 % inside 6 Å, so the mean per-witness
error is dominated by how *far away the witnesses are*: predicting zero scores exactly the witness's
own distance, and any guess with magnitude in a slightly wrong direction scores worse. The objective
was paying the model to say nothing, and the far witnesses — which cannot locate the hole — were
setting the gradient.

**The rebuilt objective has three terms.**

* **The fused position**, which is what a generator would actually use: every witness's answer is
  mapped into the shared frame and combined under a **softmax** of the logits, and that single
  position is scored. A softmax rather than a sigmoid, because a softmax is a competition for weight
  and so lets a far witness abstain, while a sigmoid lets every witness assert itself independently.
* **A proximity-weighted per-witness term** for dense supervision, weighted by `1/d²`. That exponent
  is the Fisher argument rather than a taste: a witness localises the hole to an angular precision,
  which converts to a positional error proportional to its distance, so its precision goes as the
  inverse square.
* **Detection**, over every residue and not only the witnesses, because at growth time nothing tells a
  residue that a hole exists.

**The baseline is now the right one and it is an identity, not an estimate.** A zero offset and a flat
logit make the softmax uniform, so an untrained head's fused position **is** the witnesses' centroid —
verified to 1e-6 in `tests/test_hole.py`, and visible in the numbers (untrained fused 2.41 Å against
centroid 2.41 Å). So the centroid is the zero point:

| | centroid, Å | target, Å |
|---|---|---|
| surrounded hole | 2.51 | ~0.6 |
| at a frontier | 4.76 | ~0.6 |

The 0.6 Å target is the peptide bond's tolerance from `scripts/peptide_shell_probe.py` — the distance
within which a newborn's N still threads unambiguously. So the head has to improve on the centroid by
a factor of four surrounded and eight at a frontier, and `hole_v2` is running to find out whether it
can.

**What the first attempt cost and what it bought.** Two and a half thousand wasted updates, and the
lesson is the ordinary one stated precisely: the baseline should have been measured *before* the
trainer, not after it was already running. The thing that made it visible was not the loss curve —
which was moving, slowly, in the right direction — but a closed-form predictor scoring worse than
silence. A metric on which doing nothing beats every honest attempt is a metric that is measuring
something other than what it names.

**One leak checked rather than assumed.** `alive_topology` masks the hole out of `atom_mask`, but
`frame_atom_coords` reads `frame_idx` and not `atom_mask`, so the backbone/CB edge block does read the
hole's own coordinates for any pair that includes it — and ±1 sequence neighbours are *forced* into
the neighbour list, so such pairs always exist. Measured: moving the hole 25 Å changes **nothing** on
any alive residue, bit-identical, because the edge is zeroed at aggregation by
`edge_mask & alive[neighbours]`. Only the hole's own row moves, by 3e-7, and nothing reads it. The
same argument covers the growth path, which masks the same way.

### The clamp table is complete, and silencing is quantitative — 2026-11-03

All three jobs finished (exit 0). With `--no-anchor` at 0.75 pinned, every row from the same seed so
both arms score the same residues from the same starts:

| run | arm | free right | free MAE | start | moved | stuck | native − random | moved n/r |
|---|---|---|---|---|---|---|---|---|
| v5 (31515295) | native | 54.2 % | 58.50 | 33.5 % | 46.58 | 68.7 % | **−20.9** | **0.52** |
| v5 | random | 75.1 % | 34.34 | 33.5 % | 89.17 | 35.2 % | | |
| v6 (31515293) | native | 63.4 % | 47.48 | 33.5 % | 61.64 | 57.4 % | **−11.9** | **0.69** |
| v6 | random | 75.3 % | 34.31 | 33.5 % | 89.47 | 35.1 % | | |
| v6 seed 1 (31515294) | native | 63.0 % | 47.40 | 34.5 % | 59.07 | 59.3 % | **−12.7** | **0.67** |
| v6 seed 1 | random | 75.7 % | 33.49 | 34.5 % | 88.57 | 34.9 % | | |

Pinned drift is 0.00e+00 on every native arm, so no pin leaked and no row is void.

**The prediction held.** The previous session predicted v6's `moved n/r` would fall between v5's 0.52
and 1; it is 0.69 and 0.67. And the seed replicate puts the v6 deficit at **−12.3 ± 0.4**, so the
nine-point improvement over v5 is far outside seed noise — against the 1.2-point seed-to-seed spread
measured on the accuracy itself.

**The deficit is accounted for by the under-movement, numerically.** Taking the deficit against
`1 − moved n/r`: v5 is 20.9 over 0.48, which is 43.5 points per unit; v6 is 12.3 over 0.325, which is
37.8. Within 15 % of each other across two generations. The same holds on the stuck column: the
random arm leaves 35 % stuck in both runs — that is the rate at which a random start lands in the
right well by luck — and the native arm's *excess* over it is 33.5 points for v5 and 22.3 for v6,
tracking the deficit in the same proportion.

So the silencing account is not merely consistent with the table, it predicts its magnitude from a
movement statistic that has nothing to do with accuracy. **`--damage-fractions` bought exactly the
movement it restored and no more.** Whether `--clamp-probability` closes the rest is the next
measurement, and it has a quantitative target: driving `moved n/r` to 1 should take the deficit to
roughly zero.

### v7b finished: the clash term works and costs accuracy — 31484745

20,000 updates, 1,182 min on an MI355X, exit 0, `--clash-weight 50 --damage-fractions 1.0 1.0 0.5 0.2
0.05` warm-started from `v6_mixed/last.pt`. Held out after 20,000:

```
                   chi1   chi2   chi3   chi4  acc all    core    surf  rmsd all    core    surf
kappa 0           36.65  42.28  62.59  63.76     33.8    44.8    22.7     1.170   0.975   1.340
kappa 100          4.18   4.36   4.44   4.81     99.8    99.9    99.8     0.141   0.133   0.142
kappa 20           8.24   9.59   9.90  10.24     83.7    86.7    79.7     0.285   0.258   0.302
pooled            16.36  18.74  25.64  26.27     72.4    77.1    67.4     0.532   0.456   0.594
clash 0.0004 (a crystal structure scores 0.000)
drift from a native start: 0.0194 A mean, 2.883 A worst atom
|delta| per step: 0.9 2.3 2.6 ... 1.5 1.4 1.3 1.2 1.1 1.1
```

**The clash term does what it was built for.** 0.0004 Å of mean overlap per supervised atom against a
crystal's 0.000, at weight 50 — so the packing is physically valid, which is the claim the steric term
exists to support. And the `|delta|` curve rises to 2.8° and decays monotonically to 1.1° over 64
steps: converging, not oscillating, which is the tier-2 diagnostic.

**Do not compare these numbers with v5's.** v7b ran `--split train` with 400 structures and 12 held
out; the v5 figures in this document are from the validation split with a different held-out set. The
comparison that means something is v7b against **v6_mixed**, which shares the split and the pool by
construction (`split` is in `WARM_START_MUST_MATCH`) — and v6's final eval has not been collected yet.
That is the number to fetch before claiming the clash term is free or expensive. Noting the shape of
the worry honestly: a kappa-0 accuracy of 33.8 % is not obviously better than what v5 reported on its
own set, so "physical validity cost accuracy" is live and unresolved rather than ruled out.

### A device-only bug killed the first inverse-folding run — 31516313

v8_if took a GPU allocation and exited 1 **at update 0**, having run no updates at all:

```
RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
  identity.py:241  drawn = torch.multinomial(flat, 1, generator=generator)
```

A `torch.Generator` carries a device, and any draw that consumes one requires the generator's device
to match the tensor it draws from. That is in direct tension with this project's reproducibility
convention — stated at `rollout.update_mask`, every random choice is drawn on the **CPU** and moved,
so a run's stream is a property of its seed and not of its hardware — and the tension had never been
resolved anywhere, only avoided by accident.

**806 passing tests said nothing about it, and structurally could not.** Every test runs on CPU,
where the generator and the tensor are both CPU and the constraint holds for free. Nor did
`scripts/bunya_gpu_check.slurm`, which had passed on the same card: it answers whether the ROCm wheel
enumerates the GPU, not whether a training step runs on it.

Fixed in `aanca/rng.py`: `draw_multinomial` and `draw_uniform` draw on the generator's device and move
the *result*, which keeps the stream reproducible instead of raising. **Three call sites needed it**,
and only one had been reached: `identity.straight_through`'s multinomial (the crash),
`identity._gumbel_noise`'s uniform, which passed `device=logits.device` and would have failed
identically the moment a temperature was used, and `potts._sample`, which runs on CPU today and
carried the same latent fault.

**The systemic half is `scripts/device_check.py`**, which exercises ten generator-consuming paths --
including `training_objective` with identity on, and `growth_objective` with Cartesian birth -- on
whatever device it is handed, by calling the library rather than paraphrasing it. It is now the last
step of `bunya_gpu_check.slurm`, whose exit code it joins. Thirty seconds on a card against a queue
wait and a dead chain.

The lesson is narrower than "test on the target hardware" and more useful: **a convention that is
deliberately device-independent needs one place where that independence is implemented**, or every
call site re-decides it and the ones that are never exercised decide wrong.

### The clash term costs five points of accuracy — the matched comparison — 2026-11-03

The previous section left this open and said not to compare v7b with v5. The comparison that *is*
valid is v7b against **v6_mixed (31355895)**, which shares the split, the pool and the held-out set by
construction — `split`, `structures` and `eval_structures` are all in `WARM_START_MUST_MATCH`, and
v7b was warm-started from v6's own checkpoint.

| | kappa 0 chi1 | kappa 0 acc | pooled chi1 | pooled acc | pooled RMSD | clash |
|---|---|---|---|---|---|---|
| **v6_mixed**, 7,000 updates | 32.42° | **39.1 %** | 14.95° | 74.2 % | 0.496 Å | not measured |
| **v7b_clash**, 20,000 updates, weight 50 | 36.65° | **33.8 %** | 16.36° | 72.4 % | 0.532 Å | **0.0004 Å** |

**The steric term at weight 50 costs 5.3 points of kappa-0 accuracy** — the headline number, from a
fully randomised start — along with 4.2° of chi1 MAE and 0.036 Å of RMSD. What it buys is 0.0004 Å of
mean overlap per supervised atom against a crystal structure's 0.000, so the packing it produces is
physically valid where v6's was not measured to be.

**The direction of the comparison makes it stronger, not weaker.** v7b ran 13,000 updates *more* than
v6 and is worse on every accuracy column. And v6's own best checkpoint was at update 2,500, earlier
than the 7,000-update eval quoted here, so v6 at its best is better still than the row above. There is
no reading of this where the clash term is free.

**The missing data point is the weight.** v7 ran at `--clash-weight 3.0` and was never evaluated: job
31481510 exited 1 and 31482842 exited 0 without reaching an eval, so the weight-3 arm produced no
numbers at all. Weight 50 was chosen because the term was "nearly inactive" at 3 — but nearly
inactive is not the same as useless, and 0.0004 Å of residual overlap suggests the term is being
pushed far past the point where it has done its job. **The experiment is a weight sweep at 3, 10 and
50 from the same v6 checkpoint**, scoring kappa-0 accuracy against mean overlap, to find the knee. One
of those three points already exists.

Worth saying plainly what this does to the headline claim: a packing model that is 5 points less
accurate but produces no steric clashes may well be the better model, since a clashing side chain is
wrong in a way an accuracy metric does not charge for. But that is an argument for reporting both
numbers together and for finding the cheapest weight that clears the clash, not for taking 50 and the
accuracy loss as settled.

### The growth weights were wrong by two orders of magnitude, in both directions — 2026-11-03

`scripts/growth_balance_probe.py`. Measured before the first growth training run rather than after
it. Each term alone, every other weight zeroed, 1UBQ at 6 seeds over 30 steps, mean of 3:

```
  term            value   activation       birth       frame    identity
  fape           0.6558     1.83e+01    1.37e-01    1.07e+00    0.00e+00
  bond           0.7719     8.69e+02    1.97e+00    1.10e+01    0.00e+00
  reach          0.1676     2.26e+02    7.24e-01    1.88e+00    0.00e+00
  completion     2.1269     7.17e-01    0.00e+00    0.00e+00    0.00e+00
  identity       3.1214     0.00e+00    0.00e+00    0.00e+00    2.08e-01
```

All four weights were 1.0. Equalising each term's gradient against FAPE's gives **bond 0.022, reach
0.086, completion 27.2, identity 93.7** — so the peptide restraint was **47 times too strong** and
completion **27 times too weak**. Applied as the new defaults.

**This corrects a claim recorded earlier in this document.** The growth section said completion
dominated early training and would need lowering, because its *value* was 2.13 against FAPE's 0.56. A
term's value is not its influence: in gradient, completion is 0.72 against FAPE's 18.3 — it was 25
times too weak, the opposite of what was written. The 4.8e2 activation gradient quoted there belonged
to the **reach** term, which was on by default and which I had attributed to completion.

**A probe that could not measure what it named, caught the same way as the hole metric.** The first
version of this probe reported completion's gradient as *identical to FAPE's to three figures* —
because it was FAPE's. FAPE was the one term in `growth_objective` with no weight, so it could not be
switched off, every row measured "FAPE plus that term", and gradient norms do not decompose linearly
so the difference could not be taken afterwards either. `fape_weight` exists now for that reason
alone; nothing in training should ever set it to 0.

**A hypothesis tested and disproved, recorded so it is not re-tried.** The activation head's gradient
looked like an artefact of `PARK_SPACING`: the straight-through gate's derivative is
`placed − parked`, which is about 500 Å, so an arbitrary constant appeared to be setting the scale of
a learned decision. It is not. FAPE clamps at 10 Å and a clamp has **zero** gradient beyond it, so a
residue parked 500 Å away contributes nothing at all. Measured flat at 1.8e1 for spacings of 500, 100
and 20 Å.

**What the table shows beyond the weights.** Completion gives gradient to the activation head and
*exactly zero* to the others; identity only to the identity head. That is clean confirmation each term
is wired to the head it is about — and it would have shown a mis-wiring immediately. And the
activation head receives two orders of magnitude more gradient than the birth head from **every**
term, which is structural rather than a fault: a frontier residue decides whether to wake on every
step, while the birth head only matters on the steps where a birth actually happens. The consequence
to expect is that a growth run learns *when* to grow long before it learns *where* to put things.

**The limit, stated loudly.** These balance the terms at initialisation and nowhere else — three of
the four heads are zeroed and the structure is barely grown. The reach term reads 0.168 here and only
bites once a closure is foreclosed, so it is the one most likely underweighted; the bond term reads
0.772 and should shrink as the chain improves. Re-measure a few thousand updates in. And watch the
alive fraction on the first run: completion at 27.2 is a large change in the direction of growing
faster, and nothing yet says the other terms can keep up with it.

One cosmetic consequence: with identity at 93.7 the *total* is around 350 and its magnitude means
nothing. Read the per-term columns, not the total.

### Overnight state — what is running and what to read first — 2026-11-03

**On Bunya.** Two jobs, chained so they need no attention:

| job | what | state |
|---|---|---|
| **31736674** | `bunya_gpu_check.slurm`, now ending in `scripts/device_check.py` | queued behind the user's own `pmhccamp` array |
| **31736759** | `v8b_if`, `--dependency=afterok:31736674` | will start only if the check passes |

**If the check fails for any reason, 31736759 never runs** and sits in
`DependencyNeverSatisfied` indefinitely — the same state the stale `mm_h3_6b` job is in. That is the
intended behaviour and not a hang, but it means "v8b_if never started" should be read as "the check
failed", and the check's `.out` is the place to look.

**v8b_if corrects three things about the original v8_if (31516313)**, beyond the device bug that
killed it:

* `--noop-weight 0 --native-probability 0.2` were both absent, so it would have trained with the
  measured-vacuous no-op term at full weight and **no fixed-point supervision at all**. Every run
  since v5 has used those two.
* It omitted `--split`, so it would have trained on **validation** — which this document says is "a
  development signal, never a benchmark result", and which is the wrong split for the run that is
  meant to produce the PottsMPNN comparison.
* `--updates 120000` is about five days of GPU at v7b's 3.5 s/update. Reduced to **20000**, matching
  v7b's scale and one 24 h segment, because committing five days of a shared allocation to a
  configuration whose last attempt died at update 0 is not the right order to do things in. Extend
  if it looks right.

**Locally.** `hole_v2` is training the hole head on CPU, 12,000 updates, and it is the measurement
the cloud scheme turns on. At 9,000:

```
  mask          fused   centroid    gain  per-witness  witnesses   top-1
  hole         1.67 A     3.00 A   +1.33       6.62 A       16.2  54.0%
  frontier     1.71 A     4.54 A   +2.82       5.52 A        8.9  42.0%
```

**It is plateauing around 1.7 Å against the 0.6 Å the peptide bond needs** — a factor of about 2.8
short. Read `scripts/peptide_shell_probe.py` for where 0.6 comes from.

**And one prediction of mine is already wrong, in a useful direction.** I expected the frontier case
to be much harder, because a hole on the boundary of a partial structure has half the witnesses (8.9
against 16.2) and they are all on one side. Trained, it is **not** harder: 1.71 Å against 1.67 Å.
The witnesses' count is not what limits the answer. That matters for growth specifically, because
growth is *always* at a frontier — the case I thought would be the obstacle is free.

So the open question is no longer "can a frontier be handled" but "can one hop ever reach 0.6 Å",
and the answer looks like no. Which is what the `--hops` ablation is for: the architecture's claim is
"one hop per step, depth comes from time", so on a task with no time, hops *are* depth, and how many
this task needs estimates how many steps a growth rollout would need before a newborn could be placed
well. **Run `--hops 3` next, same budget, and compare.** If three hops reach 0.6 Å then growth needs
a few relaxation steps per birth, which is unsurprising and fine; if they do not, the local
neighbourhood does not determine a residue's position well enough for density-driven growth at
peptide-bond precision, whatever is built on top.

### A trap closed: the resume tests cannot run concurrently with themselves

Two full suites were running at once and *different* tests failed on each run — a warm start that
should have been refused reporting a missing file, two halves not matching. That reads as flakiness in
the resume logic, which is the one thing those tests exist to rule out, and it cost half an hour of
chasing a regression that was not one. Every one of them passes alone.

The cause: `tests/test_train_resume.py` runs the real training script and writes real checkpoints to
`runs/<tag>/`, then reads them back. Two copies share the tags, so each one's `pytest_pinned` run
deletes and rewrites the other's `last.pt` mid-test. Fixed with a per-process tag suffix and a
session-scoped fixture that removes only its own directories, so the tests are now safe to run
concurrently by construction.

Worth generalising: any test that writes to a fixed path outside `tmp_path` is a test that cannot be
run twice at once, and the failure it produces will look like a bug in the thing under test rather
than in the harness.

### The hole head reaches 0.95 Å, and the circle is why that number means something

`hole_v2` finished: 12,000 updates, 84.5 min on a CPU, no rollout.

```
  mask          fused   centroid    gain  per-witness  witnesses   top-1
  hole         1.96 A     3.00 A   +1.04       6.54 A       16.2  44.0%
  frontier     1.17 A     4.54 A   +3.37       5.25 A        8.9  38.0%
```

**Two predictions of mine were wrong, both in the useful direction.**

First, the frontier case is *not* harder — it is **better**, 1.17 Å against the surrounded case's
1.96 Å, consistently over the last three evals. I had expected it to be the real difficulty twice
over, on the grounds that a hole on the boundary of a partial structure has half the witnesses (8.9
against 16.2) and they are all on one side. The witness count is not what limits the answer.
Sequence-local context is: a frontier hole's witnesses are mostly its own chain neighbours, which
carry the 3.8 Å spacing and the Ramachandran constraint, while a surrounded hole's include many
spatially near but sequence-distant residues that say little about exactly where it goes. **That is
good news for growth specifically, because growth is always at a frontier and always extends from
sequence-local context.**

Second, I predicted the error would split sharply by secondary structure. It does not.

### The circle, which is the right denominator

A hole whose two covalent neighbours exist is not free in three dimensions. Its CA must sit 3.804 Å
from each, so it lies on a **circle** of radius `sqrt(3.804² − (d/2)²)` where `d` is the
CA(i−1)–CA(i+1) distance. Measured over 68,277 residues in 143 structures: **2.26 Å mean, 2.47 Å
median**. So reaching 0.6 Å is not about shrinking a spherical error — it is about knowing *where on
the circle*, which is what phi and psi determine.

`scripts/hole_by_structure.py`, binning by `d` (which separates the regular conformations without
needing DSSP — helix near 5.4, extended near 6.7):

| conformation | d range | fused error | circle r | ratio | n |
|---|---|---|---|---|---|
| alpha helix | 5.2–5.9 | **0.88 Å** | 2.61 Å | **2.98** | 172 |
| irregular | 5.9–6.4 | 0.93 Å | 2.19 Å | 2.36 | 39 |
| extended | 6.4–7.1 | 1.13 Å | 1.76 Å | 1.57 | 84 |
| near collinear | 7.1–7.6 | 0.99 Å | 1.21 Å | 1.23 | 14 |
| **all bonded** | | **0.95 Å** | | | 313 |
| *excluded: chain ends and breaks* | | *4.42 Å* | *none* | *—* | *35* |

**0.95 Å on any properly bonded position**, and the error is strikingly *uniform* across real
conformations — 0.88 to 1.13 Å. The prediction that structured regions would be far better than loops
is simply not borne out. What the error splits by is not conformation but **whether the hole has a
chain to be constrained by at all**: the 35 excluded holes average 4.42 Å, and at a tenth of the
sample they were dragging the aggregate from 0.95 to 1.30 Å.

So against the peptide bond's ~0.6 Å tolerance
(`scripts/peptide_shell_probe.py`), the head is short by a factor of about **1.6**, not the 2.8 the
unfiltered aggregate implied. Density-driven growth is closer to viable than the first numbers said.

### The remaining question, sharpened

The helix ratio of 2.98 is the informative one: on the conformation where phi and psi are most tightly
pinned, the head is three times tighter than chain geometry alone. On extended backbone it is 1.57 —
but the circle there is already small (1.76 Å), so there is less to win and the absolute error is
similar. The head is therefore not failing in any particular conformation; it is uniformly about a
factor of 1.6 from where it needs to be.

That makes `--hops` the right next experiment rather than a better loss or more data: the
architecture's claim is "one hop per step, depth comes from time", so on a task with no time, hops
*are* depth. **Run `--hops 3` at the same budget.** If it closes the 1.6× then a growth rollout needs
a few relaxation steps per birth, which is unsurprising and fine. If it does not, one hop's
neighbourhood genuinely does not determine a residue's position to peptide-bond precision, and growth
would need the geometry to settle over several steps before a newborn's bond could be trusted.

### A pattern worth naming

Four measurements today were wrong in a way that looked like a result, and all four were caught the
same way — by interrogating a number that did not make sense rather than by a test:

* the hole metric, where **every closed-form baseline beat the trained head**, because the unweighted
  per-witness error mostly measured how far the witnesses were;
* the growth balance probe, which reported **completion's gradient as identical to FAPE's**, because
  FAPE was the one unweighted term and so could not be switched off;
* the park-distance hypothesis, which was mine and which the measurement **disproved** — FAPE's clamp
  has zero gradient beyond 10 Å, so a residue parked 500 Å away contributes nothing;
* and this table's "very extended" bin, which read **3.70 Å and a ratio of 0.33** — apparently three
  times worse than chain geometry — because it was collecting chain breaks, where there is no circle
  and nothing to be measured against.

The common shape: a quantity whose name implies it isolates something, which does not. None of the
809 tests could have caught any of them, because each one computes exactly what it was written to
compute.

### Three device faults in one code path, each hidden behind the last — 2026-11-03

The hourly checkup found the GPU check (31736674) had run and **failed 8/10**, up from 6/10 after the
topology fix, with both remaining failures in the identity-carrying objectives — including the exact
path job 31516313 died on. v8b_if is chained `afterok`, so it would never have started.

The traceback, from the check's `.err` rather than guessed at:

```
  rollout.py:801      target = library_index(torsion.resnames, native_chi1)
  identity.py:156     wells, _distance = rotamers.nearest_well(chi1_degrees)
  rotamers.py:39      offsets = wrap_to_pi(torch.deg2rad(degrees.unsqueeze(-1) - centres))
RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu
```

`nearest_well` built the three rotamer-well centres **on the CPU** and subtracted them from a GPU
tensor. One line, one missing `device=`, and it takes down every path that assigns a rotamer well —
which is every path that touches identity.

**The structure of this is the finding, not the fix.** Three independent device faults sat in the same
call chain, and each one hid the next:

1. a CPU **generator** against GPU logits (`torch.multinomial`) — raised first, fixed in `aanca.rng`;
2. a CPU **topology** against GPU coordinates — raised next, and was `scripts/device_check.py`'s own
   bug rather than the library's;
3. a CPU **constant** against GPU input, here.

Each fix was necessary and none was sufficient, and after each one the natural report was "fixed,
confirmed on hardware" — which was true of the fault addressed and false of the path. **Two of the
three were only ever visible because something ran the real functions on a real card**; 809 CPU tests
could not see any of them, since on CPU every one of these pairs agrees for free.

**So the check now covers the eval path too**, through `build_eval_set` and `evaluate` rather than a
paraphrase of them. The reasoning is the same one that has now been right three times: a run that
clears all three faults above still dies at its **first eval**, 500 updates in rather than at update 0,
if the eval path carries a fourth. Also pre-filtered, because the first attempt at that check took the
first two entries alphabetically and got 10GZ — 12 amino acids against a `min_residues` of 40, since
RCSB's polymer-monomer count includes nucleotides. That was the dataset filtering correctly and the
check asking the wrong thing of it.

An audit of every other module-level tensor used in arithmetic found none left: `diagnostics` moves to
CPU deliberately, and `metrics`, `growth.closure_features` and `backbone.measured_geometry` all pass
`device=`. `nearest_well` was the last of them.

**Chain re-established after the rotamers fix.** v8b_if (31736759) could never have started, so it
was cancelled rather than left to sit in `DependencyNeverSatisfied`. New chain:

| job | what | depends on |
|---|---|---|
| **31739839** | `bunya_gpu_check.slurm`, now 11 paths including the eval path | — |
| **31739848** | `v8c_if`, same args as v8b_if | `afterok:31739839` |

Three GPU-check submissions to get a clean one is the cost of having had three stacked device faults,
and it is still the right trade: each check is ten minutes against a 24-hour segment that would have
died, and the second and third faults were only findable by running the real functions on the card.

### Depth is not the bottleneck: three hops buy 0.03 Å, and the reason is common-mode error

`hole_h3` finished — 12,000 updates, 66.9 min, `--hops 3`, everything else identical to `hole_v2`.
Like for like on bonded positions (`scripts/hole_by_structure.py`, same 60 structures, 6 repeats):

| | 1 hop | 3 hops |
|---|---|---|
| **all bonded** | **0.95 Å** | **0.92 Å** |
| alpha helix | 0.88 Å (ratio 2.98) | 0.86 Å (ratio 3.04) |
| irregular | 0.93 Å | 1.06 Å |
| extended | 1.13 Å | 0.96 Å |
| near collinear | 0.99 Å | 1.02 Å |
| **per-witness** | **6.54 / 5.25 Å** | **3.52 / 2.66 Å** |
| chain breaks (excluded) | 4.42 Å | 3.22 Å |
| training loss | ~8.2 | ~4.1 |

**So the answer to "can depth close the 1.6× gap" is no.** 0.92 Å against 0.95 Å is a 3 % improvement
where a factor of 1.6 is needed, and three hops cost 3× the message passing for it.

**But the per-witness error roughly halved, and that is the informative part.** Each individual
witness became much better at pointing at the hole — 6.54 → 3.52 Å — while the *fused* answer barely
moved. That can only mean the witnesses' residual errors are **largely correlated**: fusing sixteen
of them cannot average away a bias they share, and making each one individually sharper does not help
if they are all wrong in the same direction.

Which identifies what the remaining error *is*. The common-mode component is "where on the
Ramachandran circle" — and that is a property of the missing residue's **own** conformation, not of
any witness's view of it, so every witness is equally uncertain about it no matter how much
information reaches them. More hops move information further; they cannot create information nobody
has.

**The consequence for growth, and it is the useful one.** `hole_predictions` builds the graph once and
carries only the hidden state, which was deliberate: it isolates "how far does information have to
travel" from "how long does the geometry take to settle". The answer is that **information travel was
never the bottleneck — settling is.** So a newborn's placement cannot be made peptide-bond-accurate
from a frozen neighbourhood in any number of hops, and the thing that could fix it is a real rollout
where the structure *changes* between steps: the common-mode uncertainty is resolved by the
neighbourhood rearranging around the newborn, not by more message passing over a fixed graph.

That is a point in favour of the architecture rather than against it. The rule's claim is that depth
comes from time; this says the depth that matters is the kind that only time provides. It also means
the reach restraint and the relaxation steps are doing load-bearing work rather than cleanup — a
birth placed to 0.9 Å is *expected* to be wrong at the peptide bond, and the steps after it are what
make the bond.

**What would test it directly:** place a newborn, then run N relaxation steps with the *geometry
moving*, and measure its peptide bond against N. If the bond tightens from 0.9 Å towards 0.1 Å over a
handful of steps, growth works and the hole task was measuring the wrong instant. If it does not, the
newborn's position is never recoverable and birth needs to be more accurate than a neighbourhood can
make it.

### A liveness probe needs a control, or "queued" and "blind" look identical — 2026-11-04

The 00:17 checkup's first probe reported the GPU check as `still queued` and `v8c_if` as
`not started`. Both were false: the whole read path was down, and the helper that classified the
results treated an HTTP 404 and a `TypeError: Failed to fetch` the same way — so "the job log does not
exist yet" and "I cannot see anything at all" produced the same string.

**A control request for a file known to be absent is part of the fix, and on its own it is not
enough.** A working route answers 404 for a missing file; a dead route raises a network error for
*everything*. Measured: the real file, a second attempt at it, and a deliberately nonexistent path all
raised `Failed to fetch`, which is what established that the route rather than the files was gone.

**But the 01:17 round found the case that control misses, and it is the one this document already
warns about.** After the previous round navigated the read tab to the refresh link and it redirected
to sign-in, the tab sat on `iam.rcc.uq.edu.au`. The read idiom is relative to `location.pathname`, so
every fetch then resolved against the **IAM origin** — and both the real file and the control came
back a clean **HTTP 404**. No network error at all. Read only the real file's 404 and the conclusion
is "route fine, file missing"; read the control too and the conclusion is "route fine, both files
missing", which is just as wrong.

So the probe needs a third element, and it is the cheapest of the three: **check
`location.origin` against the one you expect.** The full test is

* `location.origin` is `https://onbunya.rcc.uq.edu.au` — otherwise you are reading the wrong site and
  nothing else in the probe means anything;
* a known file returns 200 with plausible content;
* a known-absent file returns 404 rather than a network error.

**And expiry is confirmed by navigation, not by a read.** The authoritative test is to point a tab at
the code-server URL and see where it lands: `iam.rcc.uq.edu.au` is expiry, code-server is a live
session whose read path you have broken yourself. Prefer a spare tab for that, per the "never navigate
the tab you read through" rule — though when the read tab is already stranded on the sign-in page it
is useless as a read path and navigating it costs nothing.

**What expiry does and does not stop.** Submitted SLURM jobs are separate jobs and are unaffected;
anything under `/scratch` survives. So 31739839 and 31739848 run or have run regardless, and the chain
resumes being readable the moment the user reconnects. What is blocked is only observation.

### One newborn in five would fail threading at 0.9 Å, and rotation is nearly free

`scripts/bond_sensitivity_probe.py`, 20 structures. The hole head's accuracy is a **CA** error and the
threading tolerance is a **C–N** distance, and the two had been compared directly as though they were
the same quantity. They are not: the bond runs from the parent's C to the child's **N**, which depends
on the child's orientation as well as its CA. So perturb a real residue's frame by a controlled amount
and remeasure the bond holding it to its parent. No model, no training — a property of the geometry,
so it bounds what either could achieve.

| CA shift | rotation | mean C–N | median | 90th | past the 2.0 Å cut |
|---|---|---|---|---|---|
| 0.0 | 0° | 1.33 Å | 1.33 | 1.35 | 0.0 % |
| 0.3 | 0° | 1.36 Å | 1.37 | 1.58 | 0.0 % |
| **0.6** | 0° | 1.42 Å | 1.46 | 1.85 | **0.0 %** |
| **0.9** | 0° | 1.54 Å | 1.61 | **2.12** | **20.4 %** |
| 1.5 | 0° | 1.90 Å | 2.01 | 2.69 | 50.4 % |
| 0.0 | 11° | 1.36 Å | 1.36 | 1.55 | 0.0 % |
| 0.9 | 11° | 1.56 Å | 1.62 | 2.16 | 21.8 % |

**Two corrections, in opposite directions.**

*Rotation is nearly free, so the gap I flagged was not one.* Eleven degrees about the residue's own CA
costs 0.03 Å of mean bond and sends nothing past the cut — because N sits only ~1.46 Å from CA, so even
a large rotation moves it a couple of tenths, much of that tangential to the bond. I had worried that
orientation error would amplify a CA error into a worse bond. It does the opposite: the bond is almost
entirely a function of where the CA is.

*But the statistic that matters is the tail, not the mean, and that makes 0.9 Å worse than it looked.*
At the hole head's 0.9 Å the **mean** bond is 1.54 Å, which reads as comfortably fine against an ideal
1.329 — while the 90th percentile is 2.12 Å and **20.4 % of bonds exceed the 2.0 Å cut and would not be
read as peptide bonds at all**. A random displacement is partly along the bond axis, where it is
absorbed, and partly perpendicular, where it is not; the mean hides that and the tail is the whole
story.

So "0.9 Å placement against a 0.6 Å target, short by 1.6×" understates it. The honest statement is
**one newborn in five is unthreadable**, and at 0.6 Å none are — which vindicates that target, though
for a different reason than it was derived: 0.6 Å is where the *tail* clears the cut, not where the
mean does.

**And this sharpens the hops conclusion rather than competing with it.** Three hops halved the
per-witness error and left the fused error alone, which says the residual is **common-mode** — every
witness wrong in the same direction. Common-mode error is exactly what produces a tail: it does not
average away over witnesses, so it survives into the 90th percentile while the mean looks healthy.
The two measurements are describing the same thing from different ends, and together they say the
remaining work is to remove a correlated bias, not to reduce an average.

### Inverse folding ran, and there was no held-out number to judge it by — 2026-11-04

Both jobs finished while the code-server session was expired.

**GPU check 31739839: 11/11 paths on cuda, exit 0.** All three stacked device faults are closed and
confirmed on the MI355X, including the eval path added after the third.

**v8c_if 31739848: completed.** 20,000 updates, 418.6 min, exit 0, empty `.err` —
`--split train --update-schedule independent --identity-weight 1.0 --identity-keep 2.0
--identity-start masked --noop-weight 0 --native-probability 0.2`, 212,035 parameters, from scratch.

**The identity head learned something real.** The `id` column fell **3.908 at update 50 → 2.243 at
20,000**, against the uninformative floor of **ln 54 = 3.99** that the run's own header prints. A
perplexity of e^2.243 ≈ 9.4 over 54 library entries. (A note in the hourly checkup prompt said this
column must fall below ln 20 = 3.00; that was wrong — ln 20 is the *type-marginal* floor used in
`growth_objective`, while `training_objective` scores the joint 54-entry label and floors at ln 54.)

**But that number is a training-set running mean, and there was no held-out identity measurement at
all.** `evaluate` called `initial_state(case.damaged, rule)` with no identity, so the eval never
carried labels, never exercised the head, and reported nothing about it. Every eval table v8c_if
printed is packing-only. With 400 structures seen about fifty times each under different crops, "this
backbone is structure X, whose residue 47 is LEU" is a memorisable fact — so a training-set NLL is
precisely the number that cannot settle the question it was being used to settle.

Fixed: the eval now carries an identity when the rule has the head and reports **joint NLL** (floor
ln 54 = 3.99), **type-marginal NLL** (floor ln 20 = 3.00) and **sequence recovery** — the last being
the quantity ProteinMPNN (~58 %) and PottsMPNN (~44 %) report, so the comparison plan recorded earlier
finally has a number to fill in. Verified on a 4-update rule: joint 4.085, type 3.150, recovery 4.5 %,
all sitting just *above* their floors and below background composition, which is what "learned
nothing yet" should look like.

**`label_churn` existed and was never called from anywhere.** It is wired in now, per step, through
`RolloutResult` and into the eval report. That matters because of the next finding.

### The rollout does not settle, and until now it could not be told why

v8c_if's `mean |delta| per step` runs **43° → 21° and plateaus**. v7b's decayed monotonically to
1.1°. A plateau is the oscillation risk this project has worried about since `docs/PLAN_V1.md` — "does
it decay, or oscillate as some Graph NCA rules do?" — and it is the first run where it is visible.

**And `mean |delta|` alone cannot diagnose it once a discrete label can flip between steps**, which
the design note has said from the start: the continuous part can be settling while the labels churn,
so the geometry is chasing a target that keeps moving. A plateau with *high* churn is the ripple
failing to converge; the same plateau with *low* churn is the geometry oscillating on its own. Those
need opposite fixes and were indistinguishable, because nothing called `label_churn`.

Measured on an untrained rule for calibration: churn sits at **~0.45 per step and does not decay**,
which is what a non-converging ripple looks like. Whether v8c_if's trained head churns that much is
the question its checkpoint can now answer, and the next run answers for itself.

**What is not a valid comparison, stated because it is tempting.** v8c_if's packing numbers — kappa-0
chi1 89.3°, accuracy 4.5 %, pooled 50.4° — look catastrophic against v7b's 36.7° and 33.8 %. They are
not comparable. `--identity-weight` truncates perception to **backbone and CB** (the header says so),
so this rule cannot see side chains at all while v7b can, and it trained from scratch for 20,000
updates where v7b inherited a chain going back to v1. The same applies to "drift from a native start"
of 0.514 Å: with `--identity-start masked` a native-start rollout is re-deciding the sequence, so its
side chains *should* move. Three of the standard diagnostics mean something different under an
identity run, and the honest position is that the only interpretable numbers from v8c_if are the
identity NLL trajectory and the `|delta|` plateau.

### Scoring a finished run's identity head — `scripts/score_identity.py`

Applies the now-instrumented `evaluate` to a checkpoint, so v8c_if's held-out identity number can be
had without spending another seven GPU-hours re-deriving it. Reports joint NLL (floor ln 54 = 3.99),
type-marginal NLL (floor ln 20 = 3.00), sequence recovery, and the label-churn curve.

**It reads the held-out indices from the checkpoint rather than reconstructing them**, and the reason
is a mistake the first version made and crashed on. `train.py` builds its dataset over the **whole**
available split, and `usable()` shuffles `range(len(dataset))` across that full list and keeps the
first that load — so a stored index is a position in the entire split. Truncating the pool to
`structures + eval_structures`, which looks like the obvious thing since that is how many the run
used, renumbers every index and scores a *different* set of structures, some of which the run trained
on. It raised an `IndexError` rather than quietly scoring the wrong twelve, which was luck rather than
design: had the pool been larger than the highest stored index, it would have returned a plausible
number for the wrong structures. That is the same class of error as the thing the script exists to
correct — a number measuring something other than what it is named — one level up.

`--weights` scores a different file's parameters against those indices, which `best.pt` needs: it
holds weights alone under `state_dict` and carries no split.

### Inverse folding generalises: 36.8 % held-out sequence recovery — 2026-11-04

`scripts/score_identity.py` on `runs/v8c_if/last.pt` (job 31750581, CPU, `general`), the run's own
twelve held-out structures and 36 cases, 64 steps:

```
identity: joint NLL 2.322 (floor ln 54 = 3.99), type NLL 1.977 (floor ln 20 = 3.00),
          sequence recovery 36.8%
label churn per step: 0.15 0.15 0.14 ... 0.12 0.12 0.12 0.12   (plateau, not decay)
```

**The memorisation worry is answered, and the answer is no.** The held-out joint NLL is **2.322**
against the training running mean's **2.243** — a gap of 0.08 nat. With 400 structures seen about
fifty times each under different crops there was room for "this backbone is structure X, whose
residue 47 is LEU", and the margin says that is not what happened.

**36.8 % sequence recovery on held-out structures.** Against a background of roughly 9 % — guess the
most frequent amino acid everywhere — and against PottsMPNN's ~44 % and ProteinMPNN's ~58 %. Not
directly comparable, since those are published on their own benchmarks and this is twelve structures
from our train split's held-out slice, but it is the first number this project has had in that
currency at all, and it is the right side of background by a factor of four.

**And label churn plateaus at 0.12 rather than decaying to zero.** Twelve per cent of positions flip
their committed label *every step*, indefinitely, after 20,000 updates of training. `identity.py`'s
own docstring called this exactly: "a plateau at a nonzero value is a limit cycle, not slow
convergence". So the discrete half of the ripple does not converge, and the sequence the rollout
"settles on" is not settled — it is a distribution being sampled. That matters for the design, because
the whole identity story rests on a discrete fixed point, and this says there isn't one.

**One discrepancy I have not resolved, flagged rather than explained away.** This scoring run's
`mean |delta| per step` decays to **0.0°**, while v8c_if's own final eval on the *same checkpoint*
printed 43.0° falling to 21.3°. A factor of several hundred at the first step. Candidate differences
are `--eval-steps` (its eval ran 30 steps, this ran 64), and that `train.py`'s `eval_config` passes
`cutoff` and `device_neighbours` explicitly where this script does not — none of which plausibly
accounts for 400×. Until it is accounted for, **the geometry-convergence question is open**, and the
earlier reading of that 21° plateau as "the rollout does not settle" should not be relied on. The
identity numbers above do not depend on it: they are read off the final state's logits, and the churn
curve is computed from the labels directly.

**Two bugs in the scoring script, both mine, both instructive.** Its first version truncated the
structure pool to `structures + eval_structures` and so renumbered every stored eval index — it
crashed rather than scoring the wrong twelve structures, which was luck. Its second read `steps` as
`stored.get("steps_max", stored.get("steps", 30))`, but `--steps-max` defaults to **0** meaning
"unset", so the stored 0 is present-and-falsy rather than absent and a default only fires on absence.
That gave a zero-step rollout, which never calls `step_once` — and so reported "the rule has no
identity head" when the head was built and correct, and printed the *damaged inputs* as though they
were predictions. The misleading part was the message, which would have been true had the run
genuinely lacked a head.

**v9_if submitted as 31754529**, warm-started from `v8c_if/last.pt` so the 20,000 updates already
spent are not thrown away, `--eval-every 1000` so the held-out identity trajectory and the churn curve
are recorded as it trains rather than reconstructed afterwards.

### Churn flips towards similar residues, but only weakly — 2026-11-04

`scripts/churn_composition.py` on `runs/v8c_if/last.pt` (job 31757240, 12 held-out structures, 64
steps, **24,111 flips**) answers the question directly. Every statistic is against a null drawn from
the head's *own* marginal over committed labels, which is what a flip would look like if the choice
ignored the position — the level alone means nothing, the gap is the claim.

| statistic | observed | null | ratio |
|---|---|---|---|
| flip stays in chemical class | 26.0 % | 18.0 % | **1.44** |
| mean \|side-chain size change\| | 2.16 | 2.59 | **0.84** |

**So yes, but modestly.** Both point the right way — same-class flips are 44 % more common than
chance and flips are 16 % smaller in size change — so the head is not flipping at random. But 26 %
in absolute terms means **three quarters of flips still cross chemical class**, which is not what
"choosing the residue with the best energy for how its atoms sit" would look like.

The top-flip list splits into two behaviours that the aggregate ratio hides:

* **The good case, and it is real.** ILE→VAL (424) and VAL→ILE (413) are the second and third most
  frequent flips: a near-degenerate aliphatic pair the backbone barely distinguishes, which is
  exactly the flat-surface sampling that churn *should* consist of. ALA→LEU and ALA→VAL likewise.
* **A charge reversal, which is the clearest counterexample.** GLU→LYS (312) and LYS→GLU (307) are a
  matched pair: same size, opposite charge. The head has apparently learned "this site wants a long
  charged residue" without learning *which sign*, and that is precisely the information the local
  environment is supposed to supply. The energy difference at a salt bridge is large and
  sign-specific, so this is not a flat region of the surface being sampled.

ALA sits at one end of **seven of the top twelve pairs**, which looks like a low-commitment default
the head bounces off. That reading is not yet earned: a residue the head commits often will head the
list for that reason alone. So the script now reports participation **per use** against the same
marginal, where only a ratio well above 1 is a hub; rerun queued.

### Growth had an objective and nothing that trained it

Every number this project held about growth came from probes on an **untrained** rule, because
`growth_objective` was complete with measured weights while `scripts/train.py` had no growth path.
`scripts/train_growth.py` closes that: one structure, one rollout, one backward pass per update, no
sample pool. `bunya_train.slurm` grew an `AANCA_ENTRY` hook — exported, not merely read, so a segment
that exits 64 cannot silently resubmit itself as a packing run.

Building it surfaced two faults, each of which read as a result.

**FAPE read exactly 0.000 with its RMSD reading `nan` beside it.** `supervised_mask` is the crop's
interior — residues whose 10 Å perception is complete — and `seed_alive` scatters seeds uniformly
over every slot, so at the small crops growth needs in order to finish inside a rollout **the two
sets did not intersect at all**. The loss was a mean over the empty set. The mask exists because a
truncated neighbourhood makes a chi, or an amino-acid identity, unlearnable: both are inferences
*from* the environment. A native CA position is not — it is exact regardless — and under growth every
residue has a truncated neighbourhood by construction, so that is the regime rather than an artefact
of cropping. Geometry now scores every alive residue and identity keeps the mask. Measured supervised
fraction, which is what sets the trainer's default crop of 128: **12.8 % at crop 48, 19.9 % at 64,
41.4 % at 96, 54.8 % at 128, 71.4 % at 192, 97.2 % at 384.**

**The sequence loss started at 7.6 nats against a floor of 3.0**, because `AANCARule`'s
`identity_keep` defaults to 8.0 while v8c_if reached 36.8 % recovery with 2.0. At initialisation the
growth type NLL reads **7.017 at keep 8.0, 4.234 at 4.0, 3.753 at 2.0, 3.659 at 0.0** (ln 20 = 3.00).
A keep of 8 is not merely a bad starting point: the head's logits have a standard deviation near 1,
so the bonus can never be overcome and the sequence **freezes on whichever label was first sampled at
random**. It bites harder here than in packing because a newborn arrives unlabelled and must decide,
and it would have shown up as "growth cannot learn sequence" rather than as a misconfigured constant.
The trainer defaults to 2.0 and exposes the flag.

Both were caught by refusing a number rather than by a test — a perfect FAPE sitting beside a `nan`,
and a loss above its own uninformative floor. That is now five or six times in this project that a
measurement computed exactly what it was written to compute and still said something false.

**g1 submitted as 31758141**, `--dependency=afterany:31754529` so it cannot take a second GPU slot
while v9_if holds one and the user's `pmhccamp` array is waiting on the 4-GPU QOS: `--split train
--updates 8000 --eval-every 500 --identity-weight 93.7 --seeds 4 --steps 40`. The first thing it
settles is whether `completion_weight` at 27.2 is enough to forbid the degenerate optimum — a low
RMSD at a low alive fraction is the rule declining to grow, which is why the trainer never prints
one without the other.

### v9_if at 3,250 updates, and the |delta| plateau does not reproduce

Held out at 3,000 updates: **sequence recovery 37.2 %**, type NLL 1.953, joint NLL 2.304, pooled chi1
34.68°, accuracy 59.5 %. Recovery is creeping (36.7 % at 2,000 → 37.2 % at 3,000) rather than
jumping, which is what a warm start from an already-converged head should look like.

**`label churn` plateaus at 0.12 again**, independently reproducing the figure the composition
analysis above was run on: 0.16 at step 1 decaying to 0.12 by step 17 and flat thereafter. Two runs
now agree there is no discrete fixed point.

**`mean |delta| per step` reads 0.1° flat across all thirty steps.** That is worth recording because
it does *not* reproduce v8c_if's 43° → 21° plateau on what are the same weights — v9_if is
warm-started from `v8c_if/last.pt`. So the plateau was a property of the configuration rather than of
the parameters, and the candidate explanation is the eval set: v9_if's table includes the kappa 100
and kappa 20 cases, which start near-native, while the quantity that plateaued was measured on a
rollout re-deciding the sequence from a masked start. **Stated as a candidate, not a conclusion** —
the discrepancy recorded earlier (my scoring run read 0.0° on the same checkpoint) is narrowed to a
configuration difference but not yet pinned to one, and the geometry-convergence question stays open
until it is.

### Three hops does not close the gap, so the limit is not depth — 2026-11-04

The hops ablation is finished and answered. `runs/hole_h3/best.pt` reached its full 12,000 updates at
`--hops 3`; `scripts/hole_by_structure.py` was then run on **both** checkpoints under the current code
so the comparison is like for like rather than against a number from an older script, over the same
313 covalently-bonded holes.

| hops | all | alpha helix | irregular | extended | near collinear | excluded cases |
|---|---|---|---|---|---|---|
| 1 (`hole_v2`) | **0.95 A** | 0.88 | 0.93 | 1.13 | 0.99 | 4.42 A |
| 3 (`hole_h3`) | **0.92 A** | 0.86 | 1.06 | 0.96 | 1.02 | 3.22 A |

**Tripling the depth bought 3 %.** Against the peptide bond's ~0.6 A tolerance the gap was 1.6x at
one hop and is 1.5x at three, so it is not closed. And the per-conformation numbers move in *both*
directions — helix and extended improve, irregular and near-collinear get worse — which is the
signature of noise rather than a systematic gain.

**So the limit is not depth**, which is the finding the ablation was built to deliver and it rules out
the cheaper of the two remedies: growth does **not** need merely a few more relaxation steps per
birth. This confirms the common-mode result recorded earlier at the structure-resolved level. Three
hops halved the *per-witness* error while leaving the *fused* error alone, meaning the residual is
correlated across witnesses; depth adds more views of the same neighbourhood and does not decorrelate
them, so the fused estimate cannot improve. The information that would fix a residue to peptide-bond
precision is not in the local neighbourhood at any depth.

The one place three hops does help is the cases with no circle bound — chain ends and holes not
bonded to both neighbours — which fall **4.42 A to 3.22 A**, a 27 % improvement. That is consistent
with the same reading: where the geometry leaves the position genuinely underdetermined, more context
helps; where two covalent neighbours already pin it to a circle, the remaining error is correlated and
depth is spent on nothing.

**What this does and does not say about threading.** 0.92 A is the error on a *CA position*, while the
threading margin measured earlier is on C-N: bonded lengths within [1.262, 1.392] A against a nearest
non-bonded C-N of 2.509 A anywhere in 670k pairs. The two are not the same quantity and one does not
convert into the other, but they are the same order — so threading from *predicted* positions is
marginal rather than safe, and whether the directed C->N graph stays a clean set of chains when its
inputs carry ~0.9 A of correlated error is **an open question this measurement does not settle.**
Threading was verified on native coordinates only.

### The churn lives at polar and charged positions, and ALA was not a hub

The per-use participation rerun (job 31758223, exit 0) **refutes the ALA-hub reading** that the raw
flip list invited. ALA does not appear in the ten most unstable residues at all: it headed the
frequency list purely because the head commits it often, which is exactly why the statistic was
divided by use before being believed.

What the table shows instead is sharper than the thing it ruled out. The ten most unstable labels per
use are:

| residue | ratio | class | | residue | ratio | class |
|---|---|---|---|---|---|---|
| ASN | 1.45 | polar | | ARG | 1.31 | positive |
| ASP | 1.42 | negative | | HIS | 1.30 | positive |
| LYS | 1.41 | positive | | THR | 1.28 | polar |
| GLN | 1.40 | polar | | CYS | 1.16 | special |
| GLU | 1.40 | negative | | | | |
| SER | 1.33 | polar | | | | |

**Every one is polar, charged or CYS. Not a single aliphatic or aromatic residue reaches 1.16** — so
LEU, ILE, VAL, PHE, MET and TRP are all more stable per use than the least stable polar residue.
Churn is not spread evenly over the sequence; it is concentrated at the polar and charged positions
while the hydrophobic positions hold.

This matches the prediction made when the label-flipping design was argued for — hydrophobic stays
hydrophobic, polar stays polar — but states it more usefully: the flipping is **localised to the
chemistry where the structural signal is weakest.** A buried hydrophobic position is strongly
determined by packing, since the cavity's size says which greasy side chain fits; a solvent-exposed
polar position is genuinely under-determined by local geometry, because many polar residues satisfy
the same site and the environment does not discriminate between them. On that reading the plateau at
0.12 is not the rule failing to converge everywhere, but converging on the core and remaining
undecided on the surface.

It also reframes the GLU<->LYS charge reversal recorded above. GLU (1.40) and LYS (1.41) are both top
five, so the 619 reversals are not a specific confusion between those two residues; they are one
visible case of a general instability at charged positions.

**The burial claim is an inference, not a measurement.** "Polar and charged" is being read as
"surface" from chemistry alone. Burial *is* measurable here -- the packing eval already splits
core/surf -- and this analysis does not split by it. Churn resolved by burial is the measurement that
would turn this into a result, and it is the obvious next one: if the churn ratio tracks solvent
exposure rather than chemical class, the reading above is confirmed; if a buried ASN churns as much
as an exposed one, it is not.

**v9_if at 1:43** continues to climb: sequence recovery 36.7 % at 2,000 -> 37.2 % at 3,000 -> **37.9 %**,
type NLL 1.930, churn still flat at 0.12. **g1 (31758141) is still correctly gated** behind it while
the `pmhccamp` array holds the 4-GPU QOS.

### Churn per step is confounded with burial, and the confound nearly passed as the result

The core/surface reading above was an inference from chemistry, so
`scripts/churn_calibration.py` was built to measure it: churn against *measured* burial, against the
head's own entropy and confidence, and against per-residue correctness, with AUC so the candidate
scores are comparable rather than merely all plausible. Burial uses `aanca.metrics`' existing
convention — AttnPacker's core at 20 or more CB neighbours within 10 Å, surface at 15 or fewer — which
is the same definition the packing eval's core/surf columns use.

**Why this is a confidence-score question and not a protein question.** pLDDT earns its keep because
its value predicts its own accuracy: a low number is a truthful warning. So whether churn correlates
with burial is a statement about proteins, while whether churn predicts **per-residue correctness**
is the statement about the model knowing what it does not know. Only the second makes it shippable.

**The probe nearly produced a false confirmation, on the smoke test.** Churn ordered core 0.103 <
intermediate 0.148 < surface 0.209 on a rule trained for *six updates* — which has learned nothing and
cannot have a calibrated confidence, so the ordering had to be mechanical. It was: **a residue that is
not asked to update cannot flip**, and under the `independent` schedule an independent set excludes
neighbours, so a residue with more neighbours is asked less often. The measured tick rate runs **core
0.137, intermediate 0.148, surface 0.227** — almost exactly the shape of the churn curve it was about
to explain.

So churn is now reported **per update** — the conditional "when this residue was asked, how often did
it change?" — with churn per step and the tick rate kept beside it as controls. Per update the
untrained gap narrows from 2.0x to 1.5x (0.538 core against 0.808 surface). Two controls are in the
AUC table for the same reason: CB neighbour count is a property of the target structure and not of the
model at all, and the update rate is a property of the scheduler. A score has to beat both.

Had this been run first on the trained checkpoint, it would have confirmed the core/surface reading
**for the wrong reason**, and the confirmation would have been quoted as a confidence score.

**Recovery by burial is now in `evaluate` itself**, not only in the probe, so every future run reports
it without a separate job: `core`, `surface`, on the same convention, with the report stating that
intermediate is neither so the two do not average back to the pooled figure. Sequence recovery is not
uniform over a protein — a buried position is pinned by the size of the cavity it must fill, while an
exposed polar position is satisfied by several residues at once — so a single pooled figure averages
two different problems and hides which one the rule is bad at. ProteinMPNN and PottsMPNN both report
the split, so this also makes that comparison like for like rather than pooled-against-stratified.

**Calibration job 31763318 is queued** (`PD (Resources)`, the general partition is busy) and has not
produced numbers yet. The claim above about what churn *predicts* is therefore still unmeasured on a
trained rule; only the confound is established.

### v9_if has stopped improving at 38 %, so more of this recipe will not help — 2026-11-04

At 9,500 updates (2:43 on the MI355X, `id` 2.108) the held-out identity numbers over the last four
evals read **37.9 %, 37.9 %, 37.9 %, 38.0 %** sequence recovery, with type NLL flat at 1.928-1.940 and
joint NLL at 2.277-2.284. The curve rose 36.7 -> 37.2 -> 37.9 and then stopped.

**That is a plateau, not slow progress.** Warm-starting from `v8c_if/last.pt` bought about 1.2 points
over its 36.8 % and has now converged, so the next gain in inverse folding has to come from a change
of recipe rather than more compute — which also means there is no reason to spend further GPU-hours
extending this run. The comparison set stands at PottsMPNN ~44 % and ProteinMPNN ~58 % on their own
benchmarks, so this is in kind but not yet near either.

`label churn` is unchanged at 0.12, now reproduced across the whole run.

**No `recovery by burial` line appears in this log, which is correct rather than a fault**: v9_if
started before that split was added to `evaluate`, so the first run to carry it will be g1 or
whatever follows. The burial breakdown for v8c_if's weights is what job 31763318 is computing.

**Calibration job 31763318 is running** (bun155, 18 min in, header written and no table yet). It is
slower than the composition job because it accumulates per-step entropy and confidence over a
384-residue crop as well as the flips, so it has produced no numbers to report.

### Churn and entropy predict correctness at AUC 0.77-0.82, but the head is over-confident

Job 31763318 finished (exit 0, 12 held-out structures, 64 steps, 3,015 supervised residues). It
settles the core/surface reading, and the answer is **partly yes, confirmed through a different
quantity than the one it was claimed with.**

#### The confound was real and large

| band | churn/upd | churn/step | ticks | entropy | confidence | recovery | CB nbrs | n |
|---|---|---|---|---|---|---|---|---|
| core | 0.663 | 0.085 | 0.131 | **1.662** | **0.470** | **31.9 %** | 23.4 | 1196 |
| intermediate | 0.734 | 0.108 | 0.148 | 1.926 | 0.381 | 24.4 % | 17.6 | 655 |
| surface | 0.756 | 0.167 | 0.220 | **1.950** | **0.362** | 28.1 % | 11.4 | 1164 |

`churn/step` runs 0.085 -> 0.167, a **1.96x** surface-to-core gradient, and the tick rate runs 0.131
-> 0.220, a **1.68x** gradient of its own. Per update the churn gradient collapses to **1.14x**. So
most of what looked like a confidence signal in churn-per-step was the scheduler declining to ask
buried residues, exactly as the smoke test warned.

**The reading survives, but on entropy rather than churn.** Entropy runs 1.662 on the core against
1.950 on the surface and confidence runs 0.470 against 0.362, both monotone. Neither is a count of
events that require a tick to occur, so neither is confounded the way a flip rate is. **The head
genuinely is more certain on the core.**

**One correction to the earlier reading: recovery is not monotone in burial.** It runs core 31.9 %,
**intermediate 24.4 %**, surface 28.1 % -- so core is best, but the surface *beats* the intermediate
band rather than being worst. At n = 655 the intermediate gap is about two standard errors, so it is
marginal rather than solid, but the clean "core good, surface bad" story is not what the data says and
should not be quoted as though it were.

#### As confidence scores, both work, and the entropy is the better one

| score | AUC |
|---|---|
| -entropy | **0.820** |
| confidence | 0.814 |
| -churn (per update) | 0.771 |
| -churn (per step) | 0.730 |
| update rate (ticks) | 0.547 *(control)* |
| CB neighbour count | 0.528 *(control)* |

Both controls sit near 0.5, which is what licenses reading the rest: the signal is not the scheduler
and not a property of the target structure. Recovery by churn quintile spans **73.0 % down to 12.9 %**
and by confidence quintile **8.6 % up to 76.9 %**. So a per-residue self-reported number does predict
its own accuracy, which is the property that makes pLDDT worth having.

**Churn is complementary rather than redundant, but only where it matters least and most.** Its AUC
within confidence terciles is 0.573, 0.524 and **0.788** -- so among residues the head is already
confident about, the flipping still separates right from wrong strongly. It catches the confidently-
wrong-but-unstable case that a single step's distribution does not.

#### The caveat that stops this being shippable yet

| predicted | observed | gap | n |
|---|---|---|---|
| 0.0-0.2 | 8.8 % | **-8.0 %** | 593 |
| 0.2-0.4 | 14.7 % | **-12.9 %** | 1313 |
| 0.4-0.6 | 31.6 % | **-17.2 %** | 512 |
| 0.6-0.8 | 50.0 % | **-18.9 %** | 202 |
| 0.8-1.0 | 91.4 % | **-5.3 %** | 395 |

**Every gap is negative: the head claims more than it delivers at every confidence level**, worst in
the middle, with an expected calibration error of **12.0 %**. pLDDT earns its keep by being *both*
well-ranked and well-calibrated, and this is the first and not the second. That is the benign version
of the problem -- ranking is the hard part and temperature or isotonic scaling fitted on a held-out
split is the standard remedy -- but the raw max-probability must not be quoted as a probability until
that is done.

#### A protocol difference, so 28.8 % is not a regression

This probe reports overall recovery **28.8 %** where the eval reports 36.8-38.0 % on the same weights.
Not a contradiction: the probe scores the **committed sampled label**, `state.identity.argmax`, because
churn is defined on what the rule commits, while the eval scores the head's **mode**,
`type_marginal(softmax(logits)).argmax`. Sampling is lossy against the mode, and the gap is the cost
of sampling rather than a drop in quality. Both are correct for their own question.

### Threading needs the placement error to be correlated, and no threshold substitutes

`scripts/threading_error_probe.py` tests the margin `peptide_shell_probe.py` measured on native
coordinates against the error the placement head actually makes. 26 structures, 13,425 residues; each
residue translated rigidly by a Gaussian field of controlled magnitude and correlation length, so its
internal N-CA-C geometry stays exact — **the optimistic case, and an upper bound.**

**Why correlation length is the axis and not an afterthought.** A peptide bond is a constraint between
two residues, so threading sees the error in |C(i) − N(i+1)|: a *difference* of two placement errors.
Independent placement at sigma carries sigma·sqrt(2) into that difference, and at the measured 0.92 Å
that is 1.3 Å against a native margin of only ~1.1 Å between the bonded range's top (1.392) and the
nearest non-bonded pair (2.509). So the absolute error is not the question; the relative error is.

At the measured **sigma = 0.92 Å**, as false edges / missed bonds / degree violations:

| threshold | L = 0 (independent) | L = 3.8 Å (one CA-CA step) | L = 8 Å | L = 16 Å |
|---|---|---|---|---|
| 2.0 Å | 1.5 % / **34.1 %** / 1.9 % | 0.1 % / 3.0 % / 0.1 % | 0 / 0 / 0 | 0 / 0 / 0 |
| 2.2 Å | 2.8 % / 24.3 % / 4.2 % | **0.3 % / 0.8 % / 0.5 %** | 0 / 0 / 0 | 0 / 0 / 0 |
| 2.4 Å | 5.2 % / 16.0 % / 8.8 % | 0.9 % / 0.1 % / 1.8 % | 0 / 0 / 0 | 0 / 0 / 0 |
| 2.5 Å | 6.7 % / 12.8 % / **11.6 %** | 1.7 % / 0.1 % / 3.3 % | 0.1 % / 0 / 0.2 % | 0 / 0 / 0 |

**Three results.**

**Independent placement at 0.92 Å destroys threading: a third of all peptide bonds go missing** at the
2.0 Å cut. The directed graph is not a set of chains; it is a set of fragments.

**No threshold rescues it.** Raising the cut trades misses for degree violations at roughly 1 : 1.5 —
at 2.5 Å there are still 12.8 % misses *and* 11.6 % of residues now carry an illegal in- or
out-degree. There is no operating point at which independent error threads, because the error exceeds
the margin and a cut cannot create information.

**Correlation over a single CA-CA step very nearly fixes it, and 2.2 Å is the better cut.** At L = 3.8
Å the 2.2 Å threshold gives 0.3 % false, 0.8 % missed, 0.5 % degree — against 0.1 / 3.0 / 0.1 at the
2.0 Å cut currently assumed. At L = 8 Å it is exact at every threshold tried.

**The failure mode is benign, which changes the remedy.** Misses dominate false edges by twenty to one
at the 2.0 Å cut, so a degraded graph fragments rather than mis-threads: the pieces keep their correct
N-to-C direction and nothing is wired to the wrong partner. A fragmented-but-correct graph can be
repaired by a second pass or a wider cut, where a scrambled one could not be.

#### What this does and does not license

The design is safe **iff** the placement residuals of adjacent newborns are correlated over at least a
few Å. The common-mode result is suggestive but **does not establish it**: three hops halving
per-witness error while leaving the fused error alone says the residual is correlated *across the
witnesses of one hole*, which is a statement about several predictors of one position. Whether *two
adjacent newborns* err in the same direction is a different statement and is unmeasured.

**That is the next measurement, and it is cheap**: delete two sequence-adjacent residues, predict both,
and measure the correlation between their error vectors. If it is high, threading works and the cut
should move to 2.2 Å. If it is near zero, Cartesian birth cannot thread unaided and needs either an
explicit bond-geometry term at birth or a relaxation pass that couples neighbours — which the hops
ablation already showed more depth alone will not provide.

### Correction: v9_if had not plateaued, and recovery is now 39.2 %

The entry above claiming v9_if "has stopped improving at 38 %, so more of this recipe will not help"
was **wrong, and the advice attached to it was wrong too.** The full per-eval trajectory, read from the
log at 17,000 of its 20,000 updates:

| updates | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k | 14k | 15k | 16k | 17k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recovery % | 36.1 | 36.7 | 37.2 | 37.3 | 37.9 | 37.9 | 37.9 | 37.9 | 38.0 | 37.6 | 37.6 | 38.2 | 38.2 | 38.4 | 38.2 | 38.8 | **39.2** |

The flat stretch was real — four consecutive evals at 37.9 %, then a dip to 37.6 % — but it **broke
out**, and the run has gained 1.2 points since the point at which it was declared finished. So "the
recipe is done" was a plateau called from inside one, which is the standard way to call one wrongly:
a four-eval flat is not evidence of convergence when the per-eval step is ~0.3 points and the noise is
about the same size. The honest reading is that recovery is still climbing slowly and 20,000 updates
will not be where it stops.

No action follows: `--updates 20000` is stored, the run is at 17,000 at 1.0 s/update, so it finishes on
its own in under an hour and **g1 starts then** — the dependency gate resolves itself.

**A second fault, caught before it was recorded.** The first extraction of that trajectory counted
`held out after N updates` and `sequence recovery X%` with two independent regexes and zipped them.
There are 17 blocks and **18** recovery lines in the log, so every pair was shifted by one and the
curve read 37.1, 36.1, 36.7, ... — plausible, monotone-ish, and wrong at every point, including a
fabricated 37.1 % first eval. Re-extracted by splitting into blocks and matching within each. Same
class as the other faults in this file: it computed exactly what it was written to compute.

### v9_if finished: 38.3 %, and both my readings of its curve were noise

**Done, exit 0, 334.7 min, 20,000 updates.** The full held-out recovery curve:

| 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k | 11k | 12k | 13k | 14k | 15k | 16k | 17k | 18k | 19k | 20k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 36.1 | 36.7 | 37.2 | 37.3 | 37.9 | 37.9 | 37.9 | 37.9 | 38.0 | 37.6 | 37.6 | 38.2 | 38.2 | 38.4 | 38.2 | 38.8 | 39.2 | 38.1 | 39.2 | **38.3** |

From 9,000 onward: **12 evals, mean 38.32 %, sd 0.52, range 37.6-39.2.**

**Both earlier calls were over-readings of that noise, in opposite directions.** The plateau was called
from four evals at 37.9; the retraction was called from 38.2 -> 38.8 -> 39.2. The per-eval sd is 0.52,
so neither window could support a trend, and the curve then went 39.2 -> 38.1 -> 39.2 -> 38.3. **The
conclusion that this recipe is finished near 38 % is correct and my original justification for it was
not** — it is supported by a twelve-eval flat with an error bar, not by a short run of equal numbers.
There is a possible residual drift (first six of those evals mean 38.03, last six 38.63, a gap of
about two standard errors) which is too marginal to act on.

Discipline for this file: quote 38.3 % final, or 38.3 +/- 0.5, and never a two-eval difference.

### Growth trains, and it stops at half a protein with the loss satisfied

**g1 (31758141) started the moment v9_if exited and is the first growth training this project has
run.** `--updates 8000 --eval-every 500 --identity-weight 93.7 --seeds 4 --steps 40`, 253,463
parameters, cartesian birth, crop 128, 0.55 s/update.

```
     125  total 296.291  fape 0.672  rmsd  6.19 A  alive 27.6%  rec 4.7%
     250  total 271.814  fape 0.784  rmsd 10.12 A  alive 47.9%  rec 7.1%
     500  total 272.768  fape 0.818  rmsd 10.20 A  alive 50.9%  rec 6.7%
     750  total 274.812  fape 0.775  rmsd 10.53 A  alive 49.0%  rec 6.0%

held out at 500:  total 275.654  fape 0.834  rmsd 10.68 A  alive 47.9%
                  bond 2.779  reach 0.136  closures 58.5  done 0.004
                  type NLL 2.931 (floor 2.996), recovery 7.5%
```

**It grows, which is the first-order question answered: `completion_weight` at 27.2 does defeat the
degenerate optimum.** The alive fraction climbs 27.6 % -> 47.9 % in 250 updates and does not collapse
back to the seeds.

**It then stops at about half the slots, and the loss does not object.** This is the finding, and it is
a design fault rather than a training artefact:

* `done` — the completion term — reads **0.004**, i.e. essentially satisfied, while **52 % of slots are
  dormant.** The term is a BCE over the *current frontier's* activation logits toward waking, with an
  explicit `else` branch returning exactly zero when the frontier is empty. So a structure that has
  stopped growing scores as well as a finished one: the term rewards **wanting** to grow, not **having
  grown**.
* `alive_fraction` is computed and reported and **appears nowhere in `total`** (verified: the sum is
  fape + bond + completion + identity + reach). Nothing in the objective charges for a dormant slot.

So the earlier claim in this file that `completion_weight` is "load-bearing" and closes the degenerate
optimum is **half right**: it starts growth, and it cannot finish it, because once the frontier is
eager or empty the term is blind to how much is missing.

**Geometry is bad and that part is expected this early**: rmsd 10.5 Å is a blob rather than a fold,
and `bond` at 2.779 against a crystal's ~0.002 says peptide bonds are not being formed. Note rmsd
*rises* as the alive fraction rises (6.19 -> 10.12 -> 10.68 Å), which is the newborn placement error
entering the average exactly as the threading probe predicts it would.

**Sequence has barely started**: type NLL 2.931 against the 2.996 floor, recovery 7.5 % against a
background near 5 %.

**The measurement that settles the mechanism, and it is one line**: log the frontier size next to
`done`. An empty frontier at 48 % alive would mean no alive residue is adjacent to a dormant slot,
which given the topology means the segments have ends that coincide — coalescence — and the remaining
dormant slots are unreachable. A large frontier with saturated logits would mean something else limits
births. The two have different fixes and the current log cannot tell them apart.

### g1 finished: growth learns the sequence and does not learn the geometry

**Done, exit 0, 8,000 updates, 74.6 min.** The first growth training this project has run, and the
result is a clean dissociation between its two tracks.

| update | fape | rmsd | alive | bond | closures | done | type NLL | recovery |
|---|---|---|---|---|---|---|---|---|
| 0.5k | 0.834 | 10.68 Å | 47.9 % | 2.779 | 58.5 | 0.004 | 2.931 | 7.5 % |
| 1.0k | 0.794 | 11.29 Å | 48.2 % | 2.686 | 58.9 | 0.001 | 2.932 | 6.2 % |
| 5.5k | 0.817 | 9.96 Å | 43.5 % | 2.062 | 52.8 | 0.206 | 2.486 | 22.3 % |
| 7.0k | 0.864 | 11.16 Å | 47.3 % | 3.648 | 57.7 | 0.000 | 2.448 | 21.0 % |
| 8.0k | 0.853 | 11.17 Å | 44.6 % | 3.585 | 54.2 | 0.000 | **2.431** | **22.7 %** |

**The sequence track works.** Type NLL falls **2.931 -> 2.431** against the ln 20 = 3.00 floor and
recovery rises **7.5 % -> 22.7 %** against a background near 5 %. So identity is learnable from a
growth rollout, on a backbone the rule is building itself, with newborns arriving unlabelled. That is
the part of the de novo claim that now has evidence behind it. It is well below the 38.3 % of the
packing-style runs, which is expected rather than disappointing: half the structure is missing and
there are no side chains to read.

**The geometry track does not move at all.** Over 8,000 updates FAPE goes 0.834 -> 0.853, RMSD stays
at 10-11 Å, and the alive fraction never exceeds 48.2 % and drifts *down* to 44.6 %. Eight thousand
updates of training bought no geometric improvement whatsoever.

**And the peptide restraint got worse: `bond` 2.779 -> 3.585**, against a crystal structure's ~0.002.
Nothing is holding the bonds together. `BOND_WEIGHT = 0.022` was measured to equalise its gradient
against FAPE's **at initialisation**, and the note recording it warned in writing that it "should
shrink as the chain improves" and to re-measure a few thousand updates in. The chain did not improve
and the weight was not re-measured, so the restraint has been 50x too weak to matter for most of the
run. **That is the first concrete thing to fix.**

**`done` is satisfied throughout** -- 0.004, 0.001, 0.000, with occasional excursions to 0.2 -- while
half the slots stay dormant, which is the stall recorded above. `scripts/growth_diagnose.py` and the
new `frontier` / `wake_probability` / `segments` / `unreachable` counters exist to say which of the
two mechanisms it is; job **31889896** is computing them on `runs/g1/last.pt`.

**Reading this against the threading probe.** RMSD at 10-11 Å is far outside the regime where
threading could work at all -- that probe showed independent error at 0.92 Å already loses a third of
the bonds -- so g1's output is not a threadable chain and was never going to be. The sequence result
stands on its own, but nothing about backbone construction is yet demonstrated.

### Growth is not stalled, it is rate-limited by the frontier — and my `unreachable` counter is near-tautological

`scripts/growth_diagnose.py` on `runs/g1/last.pt` (job 31889896, exit 0, 12 held-out structures):

```
    mean  alive 44.6%   frontier 1.4   wake p 1.000   segments 2.9   unreach 69.5   done 0.000
```

**`wake p` is 1.000 wherever a frontier exists.** The activation head is fully saturated: the rule
wants to grow at every opportunity it has. So the hypothesis recorded above -- that the loss is blind
to the shortfall and needs a dormant-slot penalty -- is **refuted**. A term charging for dormant slots
would push on an already-maximised quantity and change nothing. `done` reads 0.000 because the term is
*correct*: everything the rule can grow from, it is already trying to grow from.

**The limit is the frontier's size, which is 1.4 slots.** The arithmetic closes: a frontier of ~1.4
with a wake probability of 1.0 over 40 steps gives ~56 births, plus 4 seeds, against the 57 residues
actually alive. Growth was never stuck -- it ran at exactly the rate the frontier permitted and the
rollout ended. With 4 seeds growing both ways the frontier *should* be up to 8, and it is 1.4 because
`segments` fell from 4 to 2.9 (segments coalesced, consuming two growable ends each time) and because
the remaining ends sit where `links` is False.

**So the fix is not a new loss term.** It is more steps, more seeds, or a birth rule that is not
limited to one residue per segment end per step. `scripts/growth_rate_probe.py` exists to pick those
numbers and was run on an *untrained* rule; it should be re-run against this checkpoint, where the
wake probability is 1.0 rather than 0.12 and the arithmetic is completely different.

#### A correction to my own instrument

**`unreachable` is near-tautological and should not be quoted as evidence of anything.** It counts
dormant slots with no covalently linked alive neighbour -- but in a gap of length k, only the two
slots at its edges have one, so `unreachable` is always approximately `dormant - frontier` whatever the
state. Here: 71 dormant, 1.4 frontier, 69.5 unreachable. It is arithmetic, not a measurement, and it
does **not** mean those slots are permanently unreachable: growth is a wavefront, and an interior gap
slot becomes reachable as soon as the edge slot ahead of it is born. I introduced it as "the quantity
that makes a stall permanent rather than slow" and that description is wrong.

A hypothesis it caused me to form and then disproved: that the spatial crop destroys sequence
adjacency, so growth cannot propagate. Measured over six validation crops at size 128, the fraction of
present `links` is **0.92-1.00**, with 1-11 contiguous runs and longest runs of 46-124 residues. A
spatial ball mostly picks sequence neighbours, because proteins are locally compact. The crop is fine.

### g1 was given 40 steps for a job that needs about 90: growth completes when allowed to

The rewritten `scripts/growth_rate_probe.py` on 1UBQ (76 residues), 120 steps, untrained:

| bias | p(wake) | seeds | complete at | final | front@1 | front@mid | front@end | segs@end |
|---|---|---|---|---|---|---|---|---|
| −2.0 | 0.12 | 4 | >120 | 59.2 % | 7.0 | 5.5 | 4.5 | 2.5 |
| 0.0 | 0.50 | 4 | >120 (1/2) | 98.0 % | 7.0 | 2.5 | 0.5 | 1.0 |
| 2.0 | 0.88 | 4 | **62** | **100 %** | 7.0 | 1.0 | 0.0 | 1.0 |
| 4.0 | 0.98 | 4 | **50** | **100 %** | 7.0 | 0.0 | 0.0 | 1.0 |
| 4.0 | 0.98 | 1 | >120 (1/2) | 86.8 % | 2.0 | 1.0 | 0.5 | 1.0 |

**Growth does complete.** At a high wake probability with four seeds the structure is finished at
50-62 steps of 120, so it finishes well inside the loss window — which also settles the old question
this probe was written for, in the affirmative.

**And that explains g1 quantitatively rather than qualitatively.** 76 residues need ~50-62 steps at
four seeds. g1 ran **40 steps on a 128-residue crop**. Scaling by length, 128 residues need roughly
`128/76 x 55 = 93` steps, so g1 had **43 % of the steps required** — against the **44.6 %** alive
fraction it actually reached. The match is close enough that the shortfall needs no other explanation:
**g1 did not stall, and it was not mis-weighted. It was stopped less than half way.**

Scaling by length is an estimate, not a proof -- the frontier decays as segments coalesce, so the
relationship is not exactly linear -- but it is the right order and it lands on the observed number.

**A single seed never finishes**, topping out at 86.8 % even at p = 0.98, because one segment offers
only two growable ends. Multi-seed growth is therefore not an elaboration but a requirement, which
`growth_objective`'s docstring already asserted on different grounds (a single segment cannot finish
inside the loss window) and this now measures.

**The birth cap is not the lever and should stop being treated as one**: 8 against 2 changes almost
nothing, because the frontier never reaches either ceiling. `front@1` is 7.0 for four seeds -- one
less than the 8 available, since a seed landing at a chain end has one blocked side.

**So the next growth run is `--steps 120 --seeds 4` or better**, not a re-weighted objective. Job
**31890939** is running the same sweep on g1's trained weights, where the wake probability is 1.000
and the rate should match or beat the bias-4.0 row.

### Trained growth completes 1UBQ in 47 steps at four seeds, and the cap is definitively not the lever

Job 31890939, `runs/g1/last.pt` on 1UBQ, 120 steps, 3 draws:

| seeds | cap | complete at | final | front@1 | front@mid | segs@end |
|---|---|---|---|---|---|---|
| 1 | 8 | >120 (1/3) | 83.3 % | 2.0 | 1.0 | 1.0 |
| 2 | 8 | 86 | 100 % | 3.3 | 0.7 | 1.0 |
| 4 | 8 | **47** | 100 % | 6.7 | 0.0 | 1.0 |
| 8 | 8 | **33** | 100 % | 14.3 | 0.0 | 1.0 |
| 16 | 16 | **29** | 100 % | 25.7 | 0.0 | 1.0 |
| 4 | **2** | 47 | 100 % | 6.7 | 0.0 | 1.0 |
| 4 | **16** | 47 | 100 % | 6.7 | 0.0 | 1.0 |

**The trained rule grows faster than any untrained setting**: 47 steps against the best untrained row's
50 (bias 4.0). So training has improved growth *rate* even though it did not improve growth *geometry*
— consistent with the activation head learning (wake probability 1.000) while the birth and frame
heads did not.

**The cap is not the lever, now beyond argument**: 2, 8 and 16 all complete at exactly 47 steps with
four seeds. Any future discussion of `max_births` as a growth knob should stop here.

**Seeds are the lever, with diminishing returns**: 86 -> 47 -> 33 -> 29 steps at 2, 4, 8, 16 seeds. One
seed still fails (83.3 %, finishing 1 of 3 draws), so two is the minimum that works and four is where
the cost per seed stops paying.

**Revised arithmetic for g1.** 76 residues complete in 47 steps at four seeds, so a 128-residue crop
needs about `47 x 128/76 = 79` steps. g1 was given **40**, or 51 % of the requirement, against the
44.6 % it reached. (The earlier estimate of ~93 steps used the untrained bias-4.0 row; the trained
number is 47, not 50, so 79 is the better figure. Both say g1 had about half.)

**An open detail, recorded rather than guessed.** The mean birth rate at four seeds is `72/47 = 1.5`
residues per step, which is well below both the cap (8) and the step-1 frontier (6.7). Frontier
residues only give birth on steps where they tick, and `update_probability` is 0.5, which predicts
~3.35 per step — still roughly twice what is observed. So a third factor is limiting the rate and it
is not identified; the candidates are frontier decay through the run (only the endpoints are measured
here) and the birth path declining on steps where it is woken. Worth resolving before tuning steps
finely, since it is a factor of two on the step budget.

**The concrete recipe for the next growth run**: `--seeds 4 --steps 120` at crop 128, which leaves
50 % headroom over the 79-step estimate and keeps completion inside the loss window. Re-measure
`BOND_WEIGHT` on *that* run rather than on g1, since a weight fitted to a run that covers 45 % of the
structure would repeat the initialisation-only error.

### There was no missing factor: I compared a run's mean birth rate against its step-1 frontier

`scripts/birth_funnel_probe.py` traces frontier, gap-1 competition and actual births per step. At a
saturated wake probability (bias 4.0, p = 0.98, emulating g1's 1.000 without needing its weights),
four seeds on 1UBQ:

| step | 1 | 5 | 12 | 17 | 23 | 31 | 50 |
|---|---|---|---|---|---|---|---|
| alive | 9.2 % | 23.2 % | 46.1 % | 59.6 % | 75.0 % | 89.5 % | 99.6 % |
| frontier | 6.67 | 5.67 | 5.00 | 3.67 | 2.67 | 1.33 | 0.33 |
| births | 3.00 | 3.00 | 1.67 | 2.33 | 1.67 | 1.00 | 0.33 |

**The funnel closes with nothing left over: `births / (frontier x 0.5) = 0.969`.** So
`births = frontier x tick x wake` is the complete account, and there is no third limiter.

**The frontier decays monotonically through the run** -- 6.67 at step 1 to 0.33 at step 50, **mean
2.48** -- for the obvious reason that a filling structure has fewer dormant slots left. And
`2.48 x 0.5 = 1.24` expected births a step against **1.20** observed.

**So the factor of two was mine.** I took the observed mean birth rate (total births over total steps)
and compared it against the frontier *at step 1*, which overstates the run mean by about 2.7x. The
quantities were never comparable. `gap-1` competition, the mechanism I thought most likely, is
essentially zero throughout and explains nothing.

**This is the third time in this file that my own summary statistic was the fault** rather than the
system: the regex that zipped 17 held-out blocks against 18 recovery lines, the `unreachable` counter
that is `dormant - frontier` by construction, and now a rate compared against a non-mean. The pattern
is the same each time -- an aggregate quoted against a quantity measured at a different point.

**Consequence for the step budget, and it is a correction of method rather than of number.** Because
the frontier decays, completion time is *not* `N / (frontier_1 x 0.5)`; that formula would predict ~23
steps where the measurement says 51. The step budget has to be read off the rate probe empirically,
which is what it already measured: **47 steps for 76 residues at four seeds on trained weights.** The
recommendation of `--seeds 4 --steps 120` at crop 128 therefore stands unchanged, and now carries no
outstanding caveat. Job for confirmation on g1's actual weights is running.

**Confirmed on g1's actual weights** (job 31892241, exit 0, 1UBQ, 4 seeds, 60 steps, 3 draws): funnel
ratio **0.882**, mean frontier **2.72**, 72 births over 60 steps at 1.20 a step, `gap-1` exactly 0.00
at every step. So `2.72 x 0.5 = 1.36` expected against 1.20 observed. The bias-4.0 proxy read 0.969
and 2.48; both sit near 1 and the difference is within what three draws can resolve, so the account
holds on trained weights and not only on the proxy.

Two readings not to make from that log. All three draws reach 100 % by step 58, which does **not**
contradict the rate probe's "complete at 47" -- that figure is the mean of the per-draw completion
steps, while this one is the step by which the slowest draw finished. And the funnel ratio sitting a
little below the proxy's is not evidence of a residual limiter: it is 0.88 against 0.97 on three draws
each, which is noise, and chasing it would repeat exactly the error this entry resolves.

### Scratch is at 278 GB of a 300 GB quota, so there is 22 GB of headroom

Measured, after `rquota` finally returned rows (it prints a bare header on some nodes):

```
FileSystem     Used(GB) Limit    Files      Limit
home           38       50       626466     1048576
scratch        278      300      722456     1000000

Project fileset S0253    489      10240     226073    1000000
```

**Scratch is 93 % full with 22 GB left, and the inode count is at 72 %.** Home is at 76 %. The project
fileset S0253, by contrast, has used 489 GB of 10,240 — about 9.7 TB free. That asymmetry is the
important part: the constraint is the per-user scratch quota, not the allocation.

Where the 278 GB is, largest first: `channel` 41G, `glp1r_md` 41G, **`AANCA` 27G**, `hf` 25G, `rfd4`
23G, `boltzgen` 16G, `pmhc` 7.7G, `CSR` 6.1G, `ProtonPottsMPNN` 3.5G, `NeuralCubes` 3.1G, `BindCraft2`
3.0G, `envs` 1.4G.

Inside AANCA's 27G: `data` **14G**, `runs` 789M, `logs` 836K, `.cc` 93K — leaving ~12G that is the
Bunya virtualenvs and `.git`.

**What this project could give back, none of it done** (deleting checkpoints is irreversible and was
not asked for):

* **~700 MB of superseded packing runs** at ~60 MB each, v1 through v7b. Two are plainly redundant:
  `v3_noop4_wrongsplit` (60M, named for a split already known to be wrong) and `v3_noop4_cpu` (61M, a
  CPU duplicate of `v3_noop4`). Against 22 GB of headroom this is 3 % — worth doing, not a fix.
* The ~12 GB venv-and-git remainder, **if** the venvs are duplicated per CPU generation, which is
  plausible since `bunya_stage.sh` supports `AANCA_VENV=..._${HAVE}`. Recoverable without touching any
  data, and worth looking at before anything else because it is the largest single candidate.

**A growth run is not the problem**: g1's entire run directory is 3.9 MB, because checkpoints are
overwritten at each eval rather than accumulated, so g2 adds about 4 MB. No growth or packing run
threatens this quota; the 14 GB structure library and the environments do.

**g2 submitted as 31894090**, and the GPU queue was empty when it went in — `squeue -p gpu_rocm`
printed only its header, so the `pmhccamp` array has finished and nothing is being crowded.
`--tag g2 --steps 120 --seeds 4 --identity-weight 93.7 --init-from runs/g1/last.pt`, with
`BOND_WEIGHT` deliberately left at its default so the only change from g1 is the step budget.

### g2: more steps did help, and the real ceiling is the crop's chain breaks, not time

g2 (31894090) ran 1,375+ updates at `--steps 120 --seeds 4`, warm-started from g1, then **crashed**:

```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
  rng.py:68 draw_multinomial -> torch.multinomial
  identity.py:248 straight_through
```

Exit 1, so the chain stopped rather than retrying, which is the intended behaviour for a crash.

**Before dying it beat g1 on every track the step budget was supposed to fix:**

| | g1 (40 steps) | g2 (120 steps) |
|---|---|---|
| alive | 44.6 % | **55.5 %** (peaking 60.2 %) |
| recovery | 22.7 % | **24.3 %** |
| type NLL | 2.431 | **2.359** |
| rmsd | 11.17 Å | 12.60 Å |
| bond | 3.585 | 3.903 |

So the step budget was a real limit and raising it moved growth and sequence together. Geometry did
not improve -- RMSD is slightly worse, which is expected when more residues are alive to be scored.

**But the new diagnostics show the ceiling is topological, not temporal**, and this is the finding:

```
frontier 0.0   wake p n/a   segments 2.8   unreachable 57.0   done 0.000
```

**The frontier is empty with 44.5 % of slots still dormant.** Growth has not run out of time; it has
run out of places to grow. With `segments` at 2.8 from 4 seeds, the live segments' ends all sit where
`links` is False -- chain breaks and crop boundaries -- and `frontier` is gated on `links`, so a
segment cannot cross one. Growth is therefore confined to the contiguous covalent runs its seeds
happened to land in.

**This corrects an earlier entry in this file.** I measured link density at 0.92-1.00 over six crops
and concluded "the crop is fine". The density was right and the implication was wrong: 0.92 over 128
residues is about **nine breaks**, which fragments the crop into runs that growth cannot cross. 1UBQ
completes to 100 % in the rate probe precisely because it is one unbroken chain with `links` all true
-- so the structure I validated the step budget on is the one structure that cannot show this
failure. A probe on an unrepresentative input confirmed a budget and hid a ceiling.

**What follows.** Seeds must cover the runs: with ~9 runs per crop, 4 seeds can reach at most 4 of
them whatever the step count. Either seed per run rather than per structure, or let a dormant slot in
an unseeded run nucleate on its own. The first is a change to `seed_alive`; the second is closer to
"there should be another residue here" and is the mechanism originally wanted.

**The crash is unexplained and must be fixed before resubmitting.** A NaN reaching the identity
softmax most likely arrives through the coordinates -- `bond` was 3.903 and rising, so a diverging
peptide geometry producing a non-finite distance is the first place to look. `growth_objective` guards
`total` with `torch.isfinite` *after* the rollout, which is too late: the rollout itself samples a
label every step, so the guard cannot catch a NaN that appears mid-rollout.

### Housekeeping: the run deletion is authorised but blocked

Deleting the superseded packing runs was approved, and I **narrowed the list** rather than taking the
whole v1-v7b sweep, because two of those are load-bearing:

* **`v6_mixed` kept** -- the clash-weight sweep at 3/10/50 recorded earlier starts from this
  checkpoint, so deleting it would destroy an agreed experiment's starting point.
* **`v7b_clash` kept** -- the best packing rule (chi1 36.7°, 33.8 % accuracy). `growth_objective`'s
  docstring says side-chain placement is "the existing packing rule's job", so composing growth with
  packing needs these weights.

To delete (13 directories, ~551 MB): `v1_gpu v1_noop4 v2_gpu v2_noop4 v3_gpu v3_noop4 v3_noop4_cpu
v3_noop4_wrongsplit v4_gpu v4_noop4 v4_probe v5_native v7_clash`.

The `rm -rf` was refused by the agent's safety classifier, so it has **not** been done and must be run
by hand from `runs/`.

### A crop has about eleven covalent runs, so four seeds could never have finished

Measured by the growth trainer's new banner over six validation crops at size 128: **mean 11.2
covalent runs per crop, range 1-26.** That is the number g2's ceiling was made of. Four seeds reach
at most four runs, so the reachable share of the crop was fixed before the first step, and g2's 55.5 %
is close to what four of eleven runs covers. No step budget could have changed it.

Three fixes, all following from that measurement:

**`seed_alive` now seeds per run** when given `links`: every covalent run gets a seed before any run
gets a second, with the surplus spread at random rather than largest-run-first, which would bias
growth towards the runs easiest to fill. `growth_objective` passes `links`, so this is on by default.

**The trainer prints the run count and warns when seeds are short of it.** A configuration that cannot
reach the whole crop now says so in its own first ten lines instead of being diagnosed afterwards from
a flat alive fraction -- which is what cost g1 and g2 between them about two and a half GPU-hours.

**`count_runs` added** so the quantity has a name and can be asserted on.

### The NaN now names its own cause

g2's crash was `torch.multinomial`'s bare "probability tensor contains either `inf`, `nan` or element
< 0`, which does not say where the NaN came from -- and the two candidates need opposite fixes. The
rollout now checks the identity logits before sampling and raises with the step, the count, and
**whether the incoming coordinates were already non-finite**: if they were, the geometry diverged and
the restraints are at fault; if they were finite and only the logits are not, the identity head blew up
on its own. One crash is now enough to tell those apart, where before it was not enough to tell them
apart at all.

Deliberately not done: sanitising the distribution and carrying on. A NaN here means something
upstream is broken, and a run that quietly replaced it would keep training on a rule that had already
diverged -- the same class of mistake as a loss that reads 0.000 over an empty set.

### The balance probe's method does not transfer to trained weights, and the inf says so

Job 31896987, `growth_balance_probe.py --checkpoint runs/g1/last.pt`, 6 seeds, 30 steps, 3 draws:

```
  term            value  activation       birth       frame    identity
  fape           0.8865         inf         inf         inf    0.00e+00
  bond          11.8109         inf         inf         inf    0.00e+00
  reach          0.6697         inf         inf         inf    0.00e+00
  completion     0.1805         inf         inf         inf    0.00e+00
  identity       2.6129         inf         inf         inf         inf
```

**Every gradient norm is infinite and the bond term reads 11.81 against 0.772 at initialisation.**
The tempting reading is that g1's weights are divergent, which would make the g2 crash obvious and
would also condemn g3, since both warm-start from g1. **That reading is wrong, and the probe's own
construction is why.**

The probe isolates a term by **zeroing every other weight**, including the restraints. At
initialisation that is harmless, because an untrained rule barely moves the coordinates in 30 steps.
On a rule that has learned to move them, removing the peptide restraint removes the only thing
holding the chain together -- so the geometry diverges, the coordinates go non-finite, and every
gradient downstream is infinite. The `fape` row proves it: with FAPE as the *only* active term the
gradients are already infinite, and FAPE clamps at 10 Å so it cannot itself produce an infinity. The
non-finite value must already be in the coordinates.

So **the measurement is of the probe, not of the rule**: ablate-one-term-at-a-time is valid only while
the rule is too weak for the ablation to matter. That is a general trap for this kind of probe and it
is worth stating plainly, because the ablation design is what made the *original* measurement possible
(see the earlier note on `fape_weight` existing so FAPE could be switched off at all).

**What a trained balance measurement has to do instead**: keep every weight at its real training
value, run the real rollout once, and take each weighted term's gradient separately with
`retain_graph`. That measures each term's actual contribution during training rather than its
contribution in a counterfactual world with no restraints. Not yet built.

**Three things this does not say.** It does not show g1's weights are divergent -- g2 ran 1,375
updates from them with finite losses before crashing. It does not identify the g2 NaN, whose cause is
still open and whose next occurrence the new rollout guard will localise. And it does not tell us
whether `BOND_WEIGHT` is wrong at 8,000 updates, which was the question: that remains unmeasured.

### Adjacent newborns err 3.2 Å apart, so the point-cloud route cannot thread

`scripts/adjacent_error_probe.py` measures the condition
`scripts/threading_error_probe.py` proved necessary and left open: whether two
*adjacent* residues err in the same direction. Both holes of a pair are deleted at once, which is the
growth case -- predicting them one at a time from otherwise-complete structures would let each see the
other's true position and measure something strictly easier than growth meets.

115 adjacent pairs from 25 validation structures:

| | `hole_v2`, 1 hop | `hole_h3`, 3 hops |
|---|---|---|
| per-hole error, RMS \|e\| | 4.048 Å | **2.863 Å** |
| difference error, RMS \|e0−e1\| | 4.007 Å | **3.240 Å** |
| independent prediction, sigma·√2 | 5.724 Å | 4.050 Å |
| ratio (1 = independent, 0 = together) | 0.700 | 0.800 |
| mean cos(e0, e1) | 0.210 | 0.230 |

**A gap of two is a different problem from a gap of one.** The per-hole error is **2.863 Å** where
`hole_by_structure.py` reports 0.92 Å on bonded positions -- a factor of three. The reason is
structural rather than statistical: an isolated hole keeps **both** covalent neighbours, which pin it
to a circle of radius 2.26 Å, while in a gap of two each hole has only one, and the constraint that
made the 0.92 Å possible is simply gone. Every number this project has quoted about placement
precision was measured on the easier case, and growth faces gaps of many rather than one.

**The errors are only weakly correlated, so the common-mode result does not transfer.** Mean cosine
0.23 and a ratio of 0.80 -- adjacent holes do lean the same way, but barely. That is the distinction
worth keeping: three hops halving per-witness error while leaving the fused error alone said the
residual is correlated *across the witnesses of one hole*; it does not follow that two *different*
holes agree, and measured, they mostly do not.

**So threading fails on this route by a wide margin.** The difference error is what a peptide bond
sees, and at **3.24 Å** against a native margin of about 1.1 Å it is not marginal: the threading probe
loses 34 % of bonds at a relative error of 1.3 Å, and this is two and a half times that. Three hops
help (4.01 -> 3.24 Å) and nowhere near enough, consistent with depth not being the limit.

**What this does and does not close.** It closes **growth as a point cloud**, the variant the hole head
was built to test, where residues appear without known sequence indices and the directed C-to-N graph
must recover the threading. It does **not** bear on the growth rule that g1, g2 and g3 actually run:
`give_birth` is indexed by parent and direction -- `born_c[i]` means "i wakes i+1" -- so the sequence
relationship is carried explicitly and never inferred from geometry. Threading was only ever needed
for the index-free variant, and that is the variant this rules out.

### The term weights were fitted under full BPTT and are used under truncate 12

**This is the answer to "is `BOND_WEIGHT` wrong", and it is wrong for a different reason than
assumed.** Not because the chain improved and the restraint should have shrunk -- because the probe
that measured it backpropagated through a different number of steps than training does.

`RolloutConfig.truncate` is **12** in training (`train_growth.py`'s default). Both balance probes set
`truncate = steps`, i.e. **30**, so they measured gradients under full backprop-through-time while
training uses a 12-step window. Measured at training's truncate, 6 seeds, 30 steps, 3 draws, at
initialisation:

| term | weighted | activation grad | implied weight |
|---|---|---|---|
| fape | 0.6174 | **4.02** | 1.0 (reference) |
| bond | 0.0493 | **0.723** | 0.022 -> **0.122** |
| completion | 57.85 | **20.9** | 27.2 -> **5.23** |
| identity | 292.5 | 20.1 (identity head) | |
| reach | 0.0166 | 0.00 | |

**`bond` is 5.6x too weak and `completion` 5.2x too strong**, against weights explicitly chosen to
equalise each term's gradient with FAPE's. Both errors point the same way: the rule is paid to grow
and barely charged for the bonds it breaks doing it. That is precisely the observed behaviour --
`bond` climbing 2.779 -> 3.903 across g1 and g2 against a crystal's 0.002, with geometry flat while
the alive fraction moved.

**And the infinite gradients were the same fault, one step earlier.** Under `truncate = steps` the
gradients overflow to `inf`; under training's 12 they are finite. So g1's weights are not divergent,
the earlier `inf` table was the probe's configuration, and `--detect-anomaly` not firing was itself
the clue -- anomaly mode raises on NaN, not on overflow to infinity.

**Third instance of one pattern, now worth stating as a rule.** A probe whose configuration diverges
from training measures the probe: the ablation probe zeroing restraints a trained rule needs; the rate
probe validating a step budget on 1UBQ, the one structure with no chain breaks; and now both balance
probes backpropagating through 30 steps where training uses 12. **Any probe that builds its own
`RolloutConfig` must take the stored one's values, and the new `--truncate` default does.**

**What this does not settle.** These are initialisation numbers at the corrected truncate; the trained
measurement still needs a run (g1's earlier trained table is void, since it used truncate 30). And it
does **not** explain the g2 NaN: with training truncating at 12, exploding gradients through a 120-step
rollout are no longer the mechanism, so that cause is open again and one candidate fewer.

**Scoped the truncate fix rather than sweeping it.** Of the probes that build their own
`RolloutConfig`, only `growth_balance_probe.py` takes gradients -- every other one
(`growth_rate_probe`, `birth_funnel_probe`, `growth_probe`, `churn_calibration`,
`churn_composition`) runs under `no_grad`, where `truncate` controls graph detachment only and cannot
change a reported number. So the fault is in one file and the others need no change; "fixing" them
would be churn. `growth_balance_probe.py` now defaults to 12 and its `--truncate` help says never to
set it to `--steps`.

**The corrected weights are recorded beside the constants and the constants are unchanged.** 0.122 for
bond and 5.23 for completion are written into `aanca.growth`'s comment with the measurement that
produced them, but `BOND_WEIGHT` and `COMPLETION_WEIGHT` still read 0.022 and 27.2. Changing them
silently would make g1 and g2 incomparable with whatever runs next, and it is a change to the
scientific setup rather than a bug fix, so it waits on a decision.

### The crash: a finite loss with an infinite gradient, which clipping turns into NaN weights

**Root cause found, and the existing guard could never have caught it.** The chain, each link measured
rather than assumed:

1. **g3's rollout guard** reported `6912 of 6912 identity logits non-finite at step 0` with **incoming
   coordinates finite**, so the geometry was exonerated -- the suspect for two days.
2. **`runs/g1/last.pt` is clean**: 0 non-finite parameter tensors at update 8000. Not a corrupt
   checkpoint.
3. **`scripts/growth_nan.py`** hooked every leaf module and ran one step at 4 and 16 seeds. **Neither
   failed**, no module emitted a non-finite output, parked coordinates peaking at a harmless 6.4e4.
   So it is not the forward pass and not the seed count.
4. That left the training loop -- and corrected my own reading of the error. **`state.step` resets
   every update**, so "step 0" is the first step of *whichever* update died, not of update 1. g3 ran
   five minutes and never reached its first progress line at update 125, so it diverged somewhere in
   its first sixty updates.
5. **Measured directly**: give one entry of one gradient an infinity and clip to 1.0.
   `clip_grad_norm_` computes `total_norm = inf`, so the scale is `clip / inf = 0`, and `0 * inf =
   NaN`. The parameters holding that gradient go NaN on `step()`; if they are in the shared trunk,
   every logit downstream is NaN -- which is why it was 6912 of 6912 rather than a handful.

**The load-bearing fact is that the loss stayed finite.** In the test, `loss finite = True` while the
gradients were not, so `if not torch.isfinite(terms.total): continue` passes the update straight
through to the optimiser. The project has guarded the wrong quantity: the loss is checked and the
gradient, which is what actually touches the weights, is not.

Fixed in `train_growth.py`: the gradient is checked for finiteness before clipping, the update is
**skipped** rather than clamped -- an infinite gradient means the rollout produced something the
objective cannot price, and a clamped version of it is a fabricated direction -- and the count is
reported at the end as a share of updates, because a run that silently dropped a third of its updates
is not the run its tag claims.

**One correction to an earlier entry.** I wrote that truncation at 12 "removes exploding gradients
through a 120-step rollout as the mechanism". That was wrong in its conclusion: truncation bounds how
far gradients propagate, not their magnitude, and the gradients do still reach infinity -- g2 at 40
steps survived 1,375 updates while g3 at 120 died inside sixty, which is the rollout length showing
up exactly as an exploding-gradient story predicts.

**Still open**: why the gradient reaches infinity at all. The guard makes training survive it, which
is not the same as fixing it, and the skip count will now say how often it happens. The mis-weighted
objective is the obvious suspect -- `completion` 5.2x too strong and `bond` 5.6x too weak at training's
truncate -- but that is a hypothesis, and the weights have deliberately not been changed.

### g3 grows 98 % of the structure, and the objective cannot hold it together

**Per-run seeding works, and the ceiling was exactly what it was diagnosed to be.** g3 (31900114) is
running at `--seeds 16 --steps 120`:

| | g1 (4 seeds, 40 steps) | g2 (4 seeds, 120) | **g3 (16 seeds, 120)** |
|---|---|---|---|
| alive | 44.6 % | 55.5 % | **97.7 %** |
| unreachable slots | 69.5 | 57.0 | **3.0** |
| segments | 2.9 | 2.8 | 13.3 |
| rmsd | 11.17 Å | 12.60 Å | **14.07 Å** |
| bond | 3.585 | 3.903 | **33.053** |
| reach | — | 0.204 | **1.955** |
| recovery | 22.7 % | 24.3 % | **15.2 %** |
| type NLL | 2.431 | 2.359 | **2.659** |

**The topological ceiling is gone**: 44.6 % -> 97.7 % alive, with unreachable slots falling from 69.5
to 3.0. Seeding per covalent run was the whole fix, and the measurement that predicted it (mean 9.5
runs per crop against 4 seeds) was right.

**The gradient guard also works**: it fired once at update 120 -- "non-finite gradient with a finite
loss (268.013); skipped" -- and the run continued past 1,100 updates where g3's previous attempt died
inside sixty. One skip in 1,100 updates, so the pathology is rare rather than endemic.

**And growing the whole structure makes everything else worse.** `bond` goes **3.585 -> 33.053**
against a crystal's 0.002, which is not explained by there simply being more bonds to score: the alive
fraction rose 2.2x and the bond term rose 9x, so the error per bond worsened too. Sequence recovery
**fell** 24.3 % -> 15.2 % and type NLL rose 2.359 -> 2.659. RMSD is 14.07 Å, so the output is a
complete and thoroughly tangled chain.

**This is the mis-weighted objective doing exactly what the weights predict, now visible because the
structure is complete enough for it to show.** At training's truncate `completion` is 5.2x too strong
and `bond` 5.6x too weak -- the rule is paid to grow and barely charged for the bonds it breaks -- and
g1 and g2 could only ever express that at 45-55 % alive. At 98 % it is unmistakable. The earlier runs
were not evidence that the geometry track is unlearnable; they were evidence that it was never being
asked for.

**So the weight correction is no longer a hypothesis-driven tweak but the obvious next experiment**:
`BOND_WEIGHT` 0.022 -> 0.122 and `COMPLETION_WEIGHT` 27.2 -> 5.23. The constants are still unchanged,
deliberately, because changing them mid-experiment is a decision about the scientific setup rather
than a bug fix -- and g3 is now the clean baseline that correction would be measured against.

**g3 at 2,875 updates is flat on everything**, which settles the weight question as far as a run can:

| updates | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|
| alive | 97.7 % | 97.7 % | 97.7 % | 97.7 % | 97.7 % |
| rmsd | 13.60 Å | 14.07 | 14.24 | 14.27 | 13.90 |
| bond | 21.84 | 33.05 | 16.60 | 19.76 | 17.65 |
| reach | 1.200 | 1.955 | 0.979 | 1.224 | 1.043 |
| type NLL | 2.653 | 2.659 | 2.654 | 2.650 | 2.619 |
| recovery | 15.6 % | 15.2 % | 14.3 % | 15.2 % | 15.1 % |

Two thousand five hundred updates and **nothing moves**: coverage pinned, RMSD flat near 14 Å,
recovery flat near 15 %, type NLL creeping 0.03. The objective is converged and the structure it has
converged on is a tangle. So the flat geometry in g1 and g2 was not a coverage artefact either -- at
98 % coverage, with every residue present to be scored, the rule still does not improve.

**`bond` swings 16.6 to 33.1 between evals**, a factor of two on the same held-out set, which is worse
than merely large: the term is not just badly weighted but unstable, and a restraint that varies 2x
between measurements cannot be steering anything. It is also the one term whose gradient the balance
measurement says is 5.6x short.

**The skip count is still 1 in 2,875 updates**, so the gradient guard is not quietly discarding the
run -- the training it reports is the training that happened.

Nothing further will come from this configuration. The correction to measure against this baseline is
`BOND_WEIGHT` 0.022 -> 0.122 with `COMPLETION_WEIGHT` 27.2 -> 5.23, and it is still not applied.

### Weights corrected: bond 0.122, completion 5.23

Applied. Verified at training's truncate, 6 seeds / 30 steps / 3 draws at initialisation, the
activation gradients now read **fape 3.90, bond 3.87, completion 4.01** -- within 3 % of each other,
against 4.02 / 0.723 / 20.9 before. That is what the weights were always meant to do and had never
done, because the probe that fitted them backpropagated through 30 steps where training uses 12.

`REACH_WEIGHT` and the identity weight are deliberately unchanged. Reach's activation gradient is
exactly 0.00 at training's truncate -- it bites only once a closure is foreclosed, so there is nothing
to balance against at initialisation -- and the identity term's gradient lands on the identity head
rather than the activation head, so "matching FAPE" would compare different parameters. Rescaling
either on this measurement would be arithmetic without a meaning, and the identity track is the one
part of growth that already works (24.3 % recovery in g2).

**g3 is the baseline this is measured against**, and it is converged: 97.7 % alive, RMSD ~14 Å, bond
16.6-33.1, recovery 15.2 %, flat across five evals over 2,500 updates. The question g4 answers is
whether geometry improves once the objective actually charges for it. The prediction, stated before
the run so it can be wrong: `bond` should fall by roughly the weight ratio or better, the alive
fraction should drop below 97.7 % as growth stops being over-rewarded, and if RMSD does **not** improve
then the geometry track is a model-capacity problem rather than a weighting one -- which would be the
first real evidence for that, since every previous flat-geometry result is now explained by the
weights.

### Correction: g3 was not flat. I called it from the first 31 % of the run

g3 finished all 8,000 updates, exit 0, 5 skipped gradients (0.1 %). Its **final** held-out eval:

| | at 2,500 (where I called it flat) | at 8,000 (final) |
|---|---|---|
| rmsd | 13.90 Å | **12.78 Å** |
| bond | 17.65 | **16.33** |
| reach | 1.043 | **0.936** |
| type NLL | 2.619 | **2.533** |
| recovery | 15.1 % | **18.7 %** |
| alive | 97.7 % | 97.7 % |

**Everything improved.** Recovery rose 3.6 points, RMSD fell 1.1 Å, bond halved from its 33.05 peak,
type NLL fell 0.13. I wrote "2,500 updates and nothing moves" and "nothing further will come from this
configuration" from five evals spanning updates 500-2,500 -- **the first 31 % of an 8,000-update run.**

This is the same error as the v9_if plateau call, and worse, because after that one I wrote the rule
into this file: *quote the final value, never a two-eval difference.* I then broke it on a five-eval
window, which is a longer window and still far too short when the run had three quarters left. The
rule needs the stronger form: **do not characterise a run's trajectory until it has finished, or say
explicitly which fraction of it the claim covers.**

**The consequence is not just bookkeeping: the baseline g4 is measured against was wrong in my own
record.** Had this stood, g4 would have been compared against "flat at 15.2 % recovery, RMSD 14.27"
instead of the true 18.7 % and 12.78 Å, and a weight correction that did nothing could have looked
like a clear win.

**What survives the correction.** The weights really were mis-measured -- that is an independent
measurement of gradient norms, not an inference from the trajectory -- and `bond` at 16.33 against a
crystal's 0.002 is still four orders of magnitude out, with a 2x swing between evals. So g4 is still
the right experiment. But "g1 and g2's geometry was flat because the objective never asked for it" is
now weaker than stated: g3 shows the geometry track *does* move under the old weights, just slowly.

**Corrected baseline for g4 (31905384)**, which differs from g3 only in `BOND_WEIGHT` 0.122 and
`COMPLETION_WEIGHT` 5.23: rmsd **12.78 Å**, bond **16.33**, type NLL **2.533**, recovery **18.7 %**,
alive **97.7 %**, at 8,000 updates.

### g4 at 5,625 of 8,000: the bond restraint works, coverage did not drop, geometry undecided

**70 % of the run, and the claims below are scoped to that** -- g3 improved substantially between its
2,500th and 8,000th update, so nothing here is a statement about where g4 ends.

Matched against g3 at the same update counts (g4 differs only in `BOND_WEIGHT` 0.122 and
`COMPLETION_WEIGHT` 5.23):

| updates | bond g3 -> g4 | rmsd g3 -> g4 | recovery g3 -> g4 |
|---|---|---|---|
| 500 | 21.84 -> **7.58** | 13.60 -> 14.54 | 15.6 % -> 17.7 % |
| 1000 | 33.05 -> **6.15** | 14.07 -> 13.46 | 15.2 % -> 16.0 % |
| 1500 | 16.60 -> **6.29** | 14.24 -> 13.73 | 14.3 % -> 15.7 % |
| 2000 | 19.76 -> **7.22** | 14.27 -> 14.62 | 15.2 % -> 16.4 % |
| 2500 | 17.65 -> **8.17** | 13.90 -> 16.98 | 15.1 % -> 16.5 % |
| 5500 | — | 16.30 | 19.0 % |

**The restraint does what the weight says.** `bond` sits at 6.1-12.9 across eleven evals where g3 ran
16.6-33.1: lower at every matched point, by a factor of 2 to 5. A 5.6x weight increase buying a
2-4x reduction in the term is the restraint actually binding, and this is the clearest confirmation
that the balance measurement was describing something real rather than an artefact of the probe.
`reach` fell with it, 0.28-1.14 against g3's 0.94-1.96.

**One prediction already falsified.** I predicted the alive fraction would drop below 97.7 % once
growth stopped being over-rewarded. It has not: **97.7 % at all eleven evals**, identical to g3.
Cutting `completion` by 5.2x changed coverage not at all, which means coverage was never
completion-limited at 16 seeds -- the per-run seeding supplies enough fronts that the structure fills
regardless of how much the term pays. So `COMPLETION_WEIGHT` is now known to be doing nothing visible
at this seed count, and the 27.2 that looked load-bearing was not.

**RMSD is worse so far and I am not calling it.** g4 runs 13.46-18.02 against g3's 13.60-14.27, and it
rose through the middle of the run before falling back to 16.30. That is either the honest cost of a
restraint that now holds bonds at the expense of global placement, or the same mid-run excursion g3
had -- and I cannot tell which at 70 %. Recovery and type NLL are within noise of g3 throughout.

One gradient skipped in 5,625 updates, so the guard is not shaping the run.
