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

### Why k=0 might be this bad — hypotheses, not findings

The k ladder is itself the diagnostic. 99.5 % at k=100, 84.2 % at k=20,
36.3 % at k=0. A rule that were simply weak would degrade gradually, because
weakness is not conditional on the start. Instead it is near-perfect right up
to the point where the starting *rotamer well* stops being identifiable, and
then it falls off a cliff. That is the shape of a **mode-selection** failure,
not a refinement failure.

Two ways to fail at selection, and the summary statistics separate them. chi1
is roughly trimodal with wells 120 deg apart. If the 36.3 % that are correct
sit near 8 deg and every failure were a clean swap to a neighbouring well,
the MAE would be about 0.36 x 8 + 0.64 x 120 = **80 deg**. It is **33.5**, so
the failures average around **50 deg** — too far for a near miss, too close
for a neighbouring well. Fifty degrees is where you land if you split the
difference between two wells, which is what a von Mises NLL against the
native rewards when the answer is ambiguous: the loss is minimised in
expectation by moving toward the circular mean of the plausible answers, and
the circular mean of g- and t is an angle no side chain occupies. **Open
measurement:** the signed chi1 error histogram at k=0, and whether the
predictions land in rotameric regions at all. That decides it.

Structurally, a **deterministic local rule has no way to break a tie**. From
randomised chis every neighbour is also wrong, so the packing problem has to
be co-solved, and co-solving needs somebody to commit first. Diffusion gets
that from the noise in the reverse process; DiffPack gets more of it from
generating chi1 -> chi2 -> chi3 -> chi4 autoregressively, each level
conditioned on one already decided. This rule has neither, and applies the
mean of its own prediction simultaneously everywhere.

**Rule out the boring explanations first.** 400 training structures and
200,268 parameters, against DiffPack's tens of thousands of chains and
AttnPacker's ~200 M. It would be embarrassing to redesign the objective and
then find the gap was a scaling curve.

**Should the benchmark start from fully randomised chis?** Yes for the
headline number — it is what DiffPack and AttnPacker report, and a packer
that only works from a nearly-correct answer is a refiner. But uniform over
the torus is a choice DiffPack can afford because it is a sampler with a
learned prior. SCWRL and Rosetta never search continuous space from uniform;
they search a discrete rotamer library, which is to say they start from
modes. Initialising from a backbone-dependent rotamer draw is from-scratch by
any reasonable definition and squarely within prior art; if accuracy jumps
there, the diagnosis is confirmed. The deeper fix, if it is mode averaging,
is to **sample from the predicted von Mises during rollout instead of taking
mu**, optionally annealing kappa over steps — nearly free to implement, gives
the rule the noise it needs to break ties, and makes best-of-N available the
way the literature uses it.

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
