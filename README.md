# AANCA

An **a**mino-**a**cid **n**eural **c**ellular **a**utomaton for protein side-chain repacking.

A per-residue rule, applied repeatedly to a fixed native backbone, that moves side chains into
their correct packing. Each residue sees only its local neighbourhood, outputs a change to its own
side-chain torsions, and the same rule runs again on the result. No global information pathway, no
step index as input, no noise-level conditioning.

**The novelty is the training objective, not the architecture.** Existing packers (AttnPacker,
DiffPack, FlowPacker) predict torsions in a fixed number of passes or a fixed diffusion schedule.
This trains for a **fixed point**: the rule must converge to the native packing and stay there, from
many starting conditions, over a step count it was never told. That is the Growing-NCA objective
applied to protein geometry.

See [`HANDOFF.md`](HANDOFF.md) for the full design, the evaluation tiers, the prior-art position and
the open questions.

## Status

**Milestone 1 â€” the geometry and data layer â€” is passing its exactness bar.** No model yet.

| What | Where | State |
|---|---|---|
| Chi definitions, bond graphs, symmetries for all 20 types | `src/aanca/residue_constants.py` | done |
| Dihedral measurement and rigid bond rotation | `src/aanca/geometry.py` | done |
| Chi extraction and modification per residue | `src/aanca/torsions.py` | done |
| Structure loading, whole-protein chis, side-chain RMSD | `src/aanca/structure.py` | done |
| Geometric damage operators | `src/aanca/damage.py` | done |
| Local perception graph, rebuilt per step | `src/aanca/graph.py` | done |
| Differentiable torsion path (Torch) | `src/aanca/torch_geometry.py` | done |
| The rule â€” one hop, two heads | `src/aanca/model.py` | done, untrained |
| Von Mises likelihood + no-op term | `src/aanca/losses.py` | done |
| Torch port of the featuriser | `src/aanca/torch_graph.py` | done |
| Rollout with sample pool + async mask | â€” | **next** |
| Dataset pipeline | â€” | next |
| Chemical damage (mutation) | `src/aanca/damage.py` | deferred â€” not in v1 |

**v1 scope is in [`docs/PLAN_V1.md`](docs/PLAN_V1.md)** â€” read that before adding anything. The short
version: v1 answers *does a strictly local, step-unconditioned rule converge to the native packing
and stay there?*, which needs only geometric damage on PDB structures. SKEMPI, Î”Î”G, mutations,
ProteinEBM, free bond geometry and the energy-head variant are all deferred.

### The Milestone 1 result

Randomise every measurable chi in a structure, then set the native values back. The worst
coordinate deviation, on three crystal structures:

| entry | resolution | chis | damage RMSD | max deviation |
|---|---|---|---|---|
| 193L lysozyme | 1.33 Ã… | 216 | 3.04 Ã… | 3.55e-14 Ã… |
| 3NIR crambin | 0.48 Ã… | 53 | 2.17 Ã… | 7.11e-15 Ã… |
| 1UBQ ubiquitin | 1.80 Ã… | 151 | 2.72 Ã… | 2.13e-14 Ã… |

The bar in the plan was 0.01 Ã…; the result is double-precision noise, twelve orders under it. That
is because `set_chi` is implemented as a **rigid rotation of the downstream atom set about the chi
bond**, applied to the residue's existing coordinates â€” not a rebuild from ideal internal
coordinates. The round trip is therefore exact by construction, and the crystal's own bond lengths
and angles survive untouched.

Two things worth knowing about that table:

- **Measure deviations on `Protein.residues`, not `Protein.atoms.coord`.** biotite stores an
  `AtomArray` as float32, where a 1e-14 Ã… error rounds to a flattering exact zero. `residues` holds
  float64 and is the authoritative state; `atoms` is a mirror kept in sync for file output.
- **The round trip alone does not prove the convention is right** â€” a consistent sign error would
  also round-trip. The independent check is the chi1 rotamer distribution, which comes out
  gauche-minus 51.5% / trans 29.5% / gauche-plus 19.0%, matching the standard ordering. A flipped
  sign would swap the two gauche populations.

### Two things handled explicitly, because they are quiet failures

- **Proline is a ring.** Rotating its chi1 would break the CDâ€“N closure. PRO is detected as rigid by
  *walking its bond graph and finding the cycle*, not by being named, and carries an all-false chi
  mask. `tests/test_residue_constants.py` asserts it is the only such residue.
- **Two-fold symmetric terminal chis** â€” ASP Ï‡2, GLU Ï‡3, PHE Ï‡2, TYR Ï‡2 â€” are equivalent under a
  180Â° flip. `torsions.chi_error` and `damage.chi_mae_degrees` fold them. Using a plain difference
  instead costs up to 180Â° of error on those chis, and the resulting numbers look plausible.
  ASN/GLN/HIS are deliberately *not* folded: their terminal substituents differ.
- **Side-chain RMSD is minimised over symmetry-equivalent atom labellings.** The same chemistry
  means a carboxylate's two oxygens and an aromatic ring's two halves are indistinguishable, so
  comparing a flipped PHE to a reference *by atom name* scores a ~1.8 Ã… error for a structure that
  is physically identical. Measured: symmetry-aware RMSD drops the affected residues from 1.8 Ã… to
  0.08 Ã… and the whole-protein mean from 0.257 Ã… to 0.011 Ã… on 193L. Published packer RMSDs sit
  around 0.4â€“0.8 Ã…, so the unsymmetrised number would have inflated our tier-1 comparison by
  roughly the entire gap between competing methods.

  LEU's CD1/CD2 and VAL's CG1/CG2 are *not* swapped. Those methyls are chemically equivalent but
  IUPAC assigns them geometrically, and swapping them changes the defining chi by ~120Â° rather than
  180Â°, so it is not a periodicity. AlphaFold's table excludes them and the baselines follow it.

### The symmetry is chemical, not geometric

Worth knowing because it bounds what the above can achieve. Exact invariance under a 180Â° flip would
need ideal geometry â€” a carboxylate whose oxygens are related precisely by a rotation about CBâ€“CG, or
a ring whose own two-fold axis is precisely the CBâ€“CG bond. Real side chains depart from that, and
the lever arm to a distal atom amplifies it:

| group | summary deviation under a flip | symmetry-aware RMSD residual |
|---|---|---|
| ASP, GLU | 0.04â€“0.19 Ã… | â€” |
| PHE, TYR | up to 0.39 Ã… | â€” |
| all symmetric residues | â€” | 0.08â€“0.15 Ã… |

The flip moves atoms by ~2.4 Ã…, so both quantities are an order of magnitude smaller â€” but neither
is zero, and a test asserting exact invariance would be wrong. The residual is irreducible: the
flipped structure genuinely differs in coordinates while being the same molecule.

## The perception graph

Decided: **the contact graph is rebuilt every step, not frozen.** Freezing it from the native
structure would let the rule read the answer through the topology, and that topology does not exist
at inference time on a novel structure â€” so a frozen graph measures something the method cannot
actually do.

That decision has a consequence that is easy to miss. **The backbone is fixed in this milestone, so
a graph built from backbone and CB atoms alone would be identical at every step.** Per-step rebuild
only means anything because neighbour selection runs over *all* heavy atoms, side chains included.
Measured edge churn, as a fraction of the neighbour list that changes:

| entry | mild noise (Îº=20) | fully randomised |
|---|---|---|
| 193L | 2.7% | 14.3% |
| 3NIR | 1.6% | 6.7% |
| 1UBQ | 2.1% | 13.7% |

Real, then, but modest â€” which is useful to know before paying for a rebuild in the inner loop.

Two things handle the gradient discontinuity the rebuild introduces:

- **A cosine envelope** on every edge, C1-continuous at the cutoff, so a neighbour crossing it fades
  out instead of popping. Messages must be multiplied by `graph.envelope` or this is lost.
- **`k` large enough that the cutoff, not `k`, excludes neighbours.** The envelope only smooths an
  edge leaving through the *cutoff*; one dropped because `k` was full still jumps. ProteinMPNN's
  customary `k=32` is **not** enough here â€” an all-heavy-atom cutoff admits more neighbours than a
  CB-based one. Measured `k_binding_fraction`:

  | k | 16 | 24 | 32 | 48 | 64 |
  |---|---|---|---|---|---|
  | 193L | 98% | 81% | 48% | 8% | **0%** |
  | 1UBQ | 89% | 62% | 25% | 1% | **0%** |
  | 3NIR | 85% | 48% | 11% | 0% | **0%** |

  So the default is `k=64`, and `graph.stats.k_binding_fraction` is checked by a test so a future
  change to the cutoff cannot silently break the smoothness guarantee.

**Equivariance is chosen, not inherited.** Every edge feature is a distance, a relative orientation
`Ráµ¢áµ€ Râ±¼`, or a vector in a residue's own backbone frame â€” invariant features plus rotation
augmentation (`Protein.transformed`), which is ProteinEBM's choice rather than E(n)-GNCA's. The
tests assert invariance per feature block, and that assertion is not vacuous: the naive
global-coordinate version of the same feature shifts by 1.13 under a random rotation, against a 1e-8
tolerance.

575 edge dimensions, 32 node dimensions. 400 of the 575 are the backbone block, which is constant
across a rollout â€” a Torch port should cache it and recompute only the side-chain block.

`graph.py` is a NumPy reference implementation, part of the verified data layer; the model's Torch
version will be a port and should be checked against it. Neighbour selection is a hard argsort and
is not differentiable in either version â€” gradients flow through the envelope weight, not through
the choice of neighbour, which is why the envelope is there.

## The rule

200,268 parameters â€” about 0.3% of ProteinEBM. Parameter count is not the constraint here; BPTT
memory is, and `graph.py`'s backbone/side-chain block split is what keeps a 50-step rollout under
~500 MB instead of ~2.2 GB.

Three things in `model.py` are load-bearing and each has a test asserting it:

- **One message-passing hop per step.** Depth comes from time. `test_one_step_is_exactly_one_hop`
  perturbs a non-neighbour's features and asserts the target residue's update is bit-identical â€”
  that is the locality claim, checked rather than asserted in prose. A companion test perturbs an
  actual neighbour and requires the output to *change*, so the first can't pass vacuously.
- **Identity initialisation.** The delta head's final layer is zeroed with a bias giving
  `(cos, sin) = (1, 0)`, so an untrained rule outputs exactly zero and is the identity map. Zeroing
  the bias too would give `atan2(0, 0)`, whose gradient is undefined. A consequence worth knowing:
  on the first step no gradient reaches the trunk, only the head's own weights â€” that looks like a
  bug and has a test documenting it.
- **Îº is capped** at `log Îº â‰¤ 6` through a `tanh`, because a likelihood loss can be reduced by
  claiming confidence instead of being accurate.

The loss is a von Mises NLL. The two-fold symmetric chis are scored as a proper two-component
mixture `Â½p(x) + Â½p(x+Ï€)` rather than by folding the residual, which would compute `âˆ’log max(â€¦)`
instead of `âˆ’log mean(â€¦)` and bias Îº upward on exactly those chis.

`torch_geometry.py` is checked against the NumPy reference, and the Milestone 1 round-trip
exactness survives the port. Importing `aanca` does **not** import Torch â€” the data layer stays
usable without it.

### Where the gradient runs

`torch_graph.py` is the training featuriser, pinned against the NumPy reference block by block. The
split is deliberate:

- **Neighbour selection is not differentiable and cannot be** â€” it's an argsort over distances. It
  runs on detached coordinates through selection helpers *shared* with the NumPy path, so the two
  cannot pick different neighbours. A test asserts they agree exactly.
- **The envelope is differentiable, and is the gradient path for graph structure.** Once the
  neighbour list is fixed, each pair's minimum heavy-atom distance is recomputed *with* gradient, so
  moving a side chain changes how strongly its edges contribute. Without this the smooth cutoff
  would be decoration â€” and there's a test that the gradient actually reaches the coordinates
  through it.
- **The backbone block is cached.** 400 of 575 edge dimensions, constant across a rollout for a
  fixed neighbour list, with a test proving it doesn't move when side chains do.

One optimisation worth knowing about: `pair_min_distances` finds the argmin under `no_grad` over the
full `(R, K, 14, 14)` block, then measures only the winning atom pair *with* gradient. That's exact â€”
the derivative of a min is the derivative of its argmin â€” and it keeps the tensors retained for
backward at `(R, K, 3)` instead of `(R, K, 14, 14, 3)`, which is the difference between ~90 MB and
~1 MB per step once BPTT stores one per step.

## Setup

There is no system Python on the development laptop, but `uv` is installed, which is enough:

```bash
uv venv --python 3.11 .venv
uv pip install numpy biotite pytest
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CPU-only Torch deliberately: locally it is for correctness checks, and Bunya needs a ROCm build,
which is a different wheel entirely.

Then fetch the validation structures (small, ~1 MB total, gitignored) and run the checks:

```bash
python scripts/fetch_test_structures.py
python -m pytest
python scripts/validate_geometry.py
```

301 tests, ~15 s. The real-structure tests skip cleanly if `data/raw/` is empty, so the suite runs
without network access â€” the synthetic residues in `tests/synthetic.py` cover the maths, and the
downloaded entries cover PDB atom naming, altlocs and disordered side chains.

## Layout

```
src/aanca/
  residue_constants.py   chi tuples, bond graphs, symmetry flags; downstream
                         atom sets DERIVED from the bond graph, not listed
  geometry.py            dihedral, Rodrigues rotation, periodic angle maths
  torsions.py            Residue, measure_chis, set_chis, apply_chi_deltas
  structure.py           Protein, load_protein, sidechain_rmsd
  damage.py              geometric damage operators; mutation stub
  graph.py               per-step k-NN graph, invariant edge features, envelope
  torch_geometry.py      differentiable torsion path; static index topology
  model.py               the rule: one hop, delta + kappa heads
  losses.py              von Mises NLL, symmetry mixture, no-op term
  torch_graph.py         differentiable featuriser; shared selection helpers
scripts/
  fetch_test_structures.py
  validate_geometry.py   reports the round-trip table plus the rotamer check
  graph_report.py        reports graph shape, the k choice, and edge churn
tests/
  synthetic.py           deterministic residues built by forward kinematics
```

The sign convention is fixed in `geometry.py` and **asserted** in
`tests/test_geometry.py::test_rotating_downstream_atoms_increases_the_dihedral_by_theta`:
rotating by `+Î¸` about the directed axis `b â†’ c` adds `+Î¸` to the dihedral.

## Compute

Training runs on UQ's Bunya cluster. **Read `binder-physics/docs/BUNYA_BROWSER_SHELL.md` before
touching it** â€” access is a code-server pane inside an interactive SLURM job, the quota binds on
inodes rather than bytes, and a job's SLURM state is not a health signal. An NCA checkpoint or
rollout sweep writes exactly the many-small-files pattern that fills the inode quota, so check
`rquota` first.
