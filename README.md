# AANCA

An **a**mino-**a**cid **n**eural **c**ellular **a**utomaton for protein side-chain repacking.

A per-residue rule, applied repeatedly to a fixed native backbone, that moves side chains into their
correct packing. Each residue sees only its local neighbourhood, outputs a change to its own
side-chain torsions, and the same rule runs again on the result — one message-passing hop per step,
no global pathway, no step index as input, no noise-level conditioning.

The distinguishing choice is the training objective rather than the architecture. Existing packers
predict torsions in a fixed number of passes or along a fixed diffusion schedule. This trains for a
**fixed point**: the rule must converge to the native packing and stay there, from many starting
conditions, over a step count it was never told.

> **Status: pre-training.** The geometry layer, perception graph, rule, loss and rollout are built
> and tested. Nothing has been trained, so there are no accuracy results yet — every number below is
> a correctness or cost measurement, not a performance claim.

## What's built

| | Where | State |
|---|---|---|
| Chi extraction and exact modification, all 20 residue types | `residue_constants.py`, `torsions.py` | done |
| Structure loading, side-chain RMSD, rigid motion | `structure.py` | done |
| Geometric damage operators | `damage.py` | done |
| Per-step perception graph, invariant features | `graph.py` | done |
| Differentiable torsion path (Torch) | `torch_geometry.py` | done |
| Differentiable featuriser | `torch_graph.py` | done |
| The rule — one hop, Δχ + κ heads | `model.py` | done, untrained |
| Von Mises likelihood, no-op term, restraints | `losses.py` | done |
| Rollout: async mask, truncated BPTT, sample pool | `rollout.py` | done |
| Published benchmark metrics | `metrics.py` | done, partially validated |
| Bond/angle degrees of freedom | `torch_geometry.py` | built, disabled by default |
| Dataset assembly and splitting | `data.py` | done |
| Training data acquisition | — | **next** |
| Nonbonded fields as perception | — | next |

The design, with reasoning, is in [`docs/PLAN_V1.md`](docs/PLAN_V1.md). Read that before adding
anything — it records which options were rejected and why, including two where the original argument
turned out to be wrong.

## Verified properties

### The chi round trip is exact

Randomise every measurable chi, then set the native values back:

| entry | resolution | chis | damage RMSD | max deviation |
|---|---|---|---|---|
| 193L lysozyme | 1.33 Å | 216 | 3.04 Å | 3.55e-14 Å |
| 3NIR crambin | 0.48 Å | 53 | 2.17 Å | 7.11e-15 Å |
| 1UBQ ubiquitin | 1.80 Å | 151 | 2.72 Å | 2.13e-14 Å |

The target was 0.01 Å; the result is double-precision noise. `set_chi` is a rigid rotation of the
downstream atom set about the chi bond, applied to the residue's existing coordinates, not a rebuild
from ideal internal coordinates — so the inverse is exact by construction and the crystal's own bond
lengths and angles survive untouched. The same holds for bond stretches (translation along the bond
axis) and angle bends (rotation about the plane normal), and inversion is order-independent across
all three.

A round trip alone cannot catch a consistent sign error, so the convention is checked independently
against rotamer statistics: χ1 comes out gauche-minus 51.5% / trans 29.5% / gauche-plus 19.0%,
matching the standard ordering. A flipped sign would swap the two gauche populations.

### Locality is a test, not a claim

`test_one_step_is_exactly_one_hop` perturbs a non-neighbour's features and asserts the target
residue's update is bit-identical. A companion test perturbs an actual neighbour and requires the
output to change, so the first cannot pass by nothing mattering.

Worth stating honestly: at a 10 Å all-heavy-atom cutoff, measured mean degree is 28 on a 76-residue
protein, so **one hop already reaches a quarter to a third of a small protein**. The defensible claim
is "one message-passing hop per update, no global pathway, no step conditioning" — not "each residue
sees only a handful of neighbours". Locality is also size-dependent: 28 neighbours is 37% of
ubiquitin but under 10% of a 300-residue target.

### Equivariance is chosen, not inherited

Every edge feature is a distance, a relative orientation `Rᵢᵀ Rⱼ`, or a vector in a residue's own
backbone frame — invariant features plus rotation augmentation. The invariance tests are not vacuous:
the naive global-coordinate version of the same feature shifts by 1.13 under a random rotation,
against a 1e-8 tolerance.

### Chemical symmetry is handled in both directions

A carboxylate's two oxygens and an aromatic ring's two halves are indistinguishable, so a 180° flip
is the same structure. Side-chain RMSD therefore minimises over symmetry-equivalent atom labellings;
without that, a flipped PHE scores ~1.8 Å against a structure it is chemically identical to, and the
whole-protein mean inflates by 0.26–0.36 Å — about the entire gap between competing published
methods.

The symmetry is *chemical*, not geometric: exact invariance would need ideal geometry, and real side
chains deviate by up to 0.39 Å at the aromatic tips where the lever arm is longest. Tests assert the
measured bound rather than exactness.

LEU's CD1/CD2 and VAL's CG1/CG2 are deliberately **not** swapped — chemically equivalent, but IUPAC
assigns them geometrically and swapping changes the defining chi by ~120° rather than 180°.

### Rollout cost, measured

Activations retained for backward, counted via `saved_tensors_hooks` (process RSS was tried first
and is useless for this — it gave negative deltas that contradicted their own sums):

| per step, k=64 | 1UBQ (76 res) | 193L (129) | 4HHB (574, 4 chains) |
|---|---|---|---|
| baseline, float64 | 70.7 MB | 120.1 MB | 534.3 MB |
| `fixed_backbone`, float64 | 37.2 MB | 63.2 MB | 281.1 MB |
| `fixed_backbone` + float32 | **31.4 MB** | **53.3 MB** | **237.1 MB** |

Linear in residue count, so a 300-residue target is ~124 MB/step: 3.7 GB for 30 steps per sample.
An earlier estimate said ~500 MB and was 12× optimistic — it counted the edge-feature tensor and
forgot the network's own activations, which dominate. Truncated BPTT is therefore mandatory rather
than an optimisation.

`fixed_backbone=True` builds the 400-dimension backbone block under `no_grad`, worth 47%. It is
exact, not approximate: the backbone never moves, so that block's gradient is identically zero.
Detaching it *afterwards* saves nothing, which was measured — by then the tensor is already built
inside the graph.

## Benchmark reproduction — partial

`metrics.py` implements the metrics AttnPacker, DiffPack and FlowPacker all report: χ MAE per chi
index, χ accuracy within 20°, and side-chain atom RMSD, stratified by burial (core ≥20 CB atoms
within 10 Å, surface ≤15 — deliberately not a partition).

AttnPacker's benchmark release ships the native test structures *and* every baseline's predictions,
which allows a much stronger check than self-consistency: recompute their published table from their
own files. `scripts/validate_benchmark.py` does this. Current state on CASP13:

| χ1 MAE | ours | published | gap |
|---|---|---|---|
| SCWRL | 28.45 | 27.64 | +0.81 |
| FASPR | 27.80 | 27.04 | +0.76 |
| RosettaPacker | 26.59 | 25.88 | +0.71 |
| DLPacker | 22.89 | 22.18 | +0.71 |

**χ1 reproduces to under 1° across four independent methods**, which validates the chi extraction,
the symmetry folding and the burial classification. But χ2 runs 3–4° high, χ3/χ4 run low, accuracy
is ~10 points high and RMSD ~0.15 Å high — all systematically, in the same direction for every
method. That pattern means at least one definitional difference remains unidentified, so **the
metrics are not yet safe to quote against published numbers.** Candidates being checked: which
residues enter the denominator, whether CB is included in the RMSD atom set, and whether their
accuracy folds symmetry. AttnPacker's own files need a different residue alignment and currently
produce nonsense.

This is the kind of thing worth finding before a training run rather than after.

## Setup

Python 3.11+. No system Python is needed if `uv` is available:

```bash
uv venv --python 3.11 .venv
uv pip install numpy biotite pytest
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CPU-only Torch is deliberate: locally this is for correctness checks, and the training target needs
a different build anyway.

Fetch the small validation structures (~1 MB, gitignored) and run the checks:

```bash
python scripts/fetch_test_structures.py
python -m pytest
python scripts/validate_geometry.py
python scripts/graph_report.py
python scripts/measure_rollout_memory.py
```

401 tests, ~20 s. The real-structure tests skip cleanly when `data/raw/` is empty, so the suite runs
without network access — synthetic residues built by forward kinematics cover the maths, and the
downloaded entries cover PDB atom naming, altlocs and disordered side chains.

## Layout

```
src/aanca/
  residue_constants.py   chi tuples, bond graphs, symmetries, movable internal
                         coordinates; moving atom sets DERIVED from the bond
                         graph, not listed, so they cannot drift out of sync
  geometry.py            dihedral, Rodrigues rotation, periodic angle maths
  torsions.py            Residue, measure_chis, set_chis, apply_chi_deltas
  structure.py           Protein, load_protein, symmetry-aware RMSD
  damage.py              geometric damage operators
  graph.py               per-step k-NN graph, invariant edge features, envelope
  torch_geometry.py      differentiable torsion/bond/angle path
  torch_graph.py         differentiable featuriser
  model.py               the rule: one hop, delta + kappa heads
  losses.py              von Mises NLL, symmetry mixture, no-op, restraints
  rollout.py             async mask, truncated BPTT, sample pool, objective
  metrics.py             the published packing benchmark
  data.py                dataset assembly, cluster splitting, damage sampling
```

Importing `aanca` does **not** import Torch — the data layer stays usable without it.

The sign convention is fixed in `geometry.py` and asserted in
`tests/test_geometry.py::test_rotating_downstream_atoms_increases_the_dihedral_by_theta`: rotating
by `+θ` about the directed axis `b → c` adds `+θ` to the dihedral.

## Running it

Training runs on a GPU cluster, not locally — measured wall time is ~0.5 s/step on CPU at 574
residues. Local notes on cluster access live outside this repository.
