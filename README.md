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

**Milestone 1 — the geometry and data layer — is passing its exactness bar.** No model yet.

| What | Where | State |
|---|---|---|
| Chi definitions, bond graphs, symmetries for all 20 types | `src/aanca/residue_constants.py` | done |
| Dihedral measurement and rigid bond rotation | `src/aanca/geometry.py` | done |
| Chi extraction and modification per residue | `src/aanca/torsions.py` | done |
| Structure loading, whole-protein chis, side-chain RMSD | `src/aanca/structure.py` | done |
| Geometric damage operators | `src/aanca/damage.py` | done |
| Chemical damage (mutation) | `src/aanca/damage.py` | **stub** — pending the tier-3 readout decision |
| Local graph builder | — | next |
| The rule itself | — | not started |

### The Milestone 1 result

Randomise every measurable chi in a structure, then set the native values back. The worst
coordinate deviation, on three crystal structures:

| entry | resolution | chis | damage RMSD | max deviation |
|---|---|---|---|---|
| 193L lysozyme | 1.33 Å | 216 | 3.04 Å | 3.55e-14 Å |
| 3NIR crambin | 0.48 Å | 53 | 2.17 Å | 7.11e-15 Å |
| 1UBQ ubiquitin | 1.80 Å | 151 | 2.72 Å | 2.13e-14 Å |

The bar in the plan was 0.01 Å; the result is double-precision noise, twelve orders under it. That
is because `set_chi` is implemented as a **rigid rotation of the downstream atom set about the chi
bond**, applied to the residue's existing coordinates — not a rebuild from ideal internal
coordinates. The round trip is therefore exact by construction, and the crystal's own bond lengths
and angles survive untouched.

Two things worth knowing about that table:

- **Measure deviations on `Protein.residues`, not `Protein.atoms.coord`.** biotite stores an
  `AtomArray` as float32, where a 1e-14 Å error rounds to a flattering exact zero. `residues` holds
  float64 and is the authoritative state; `atoms` is a mirror kept in sync for file output.
- **The round trip alone does not prove the convention is right** — a consistent sign error would
  also round-trip. The independent check is the chi1 rotamer distribution, which comes out
  gauche-minus 51.5% / trans 29.5% / gauche-plus 19.0%, matching the standard ordering. A flipped
  sign would swap the two gauche populations.

### Two things handled explicitly, because they are quiet failures

- **Proline is a ring.** Rotating its chi1 would break the CD–N closure. PRO is detected as rigid by
  *walking its bond graph and finding the cycle*, not by being named, and carries an all-false chi
  mask. `tests/test_residue_constants.py` asserts it is the only such residue.
- **Two-fold symmetric terminal chis** — ASP χ2, GLU χ3, PHE χ2, TYR χ2 — are equivalent under a
  180° flip. `torsions.chi_error` and `damage.chi_mae_degrees` fold them. Using a plain difference
  instead costs up to 180° of error on those chis, and the resulting numbers look plausible.
  ASN/GLN/HIS are deliberately *not* folded: their terminal substituents differ.

## Setup

There is no system Python on the development laptop, but `uv` is installed, which is enough:

```bash
uv venv --python 3.11 .venv
uv pip install numpy biotite pytest
```

Then fetch the validation structures (small, ~1 MB total, gitignored) and run the checks:

```bash
python scripts/fetch_test_structures.py
python -m pytest
python scripts/validate_geometry.py
```

160 tests, ~1.5 s. The real-structure tests skip cleanly if `data/raw/` is empty, so the suite runs
without network access — the synthetic residues in `tests/synthetic.py` cover the maths, and the
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
scripts/
  fetch_test_structures.py
  validate_geometry.py   reports the table above plus the rotamer check
tests/
  synthetic.py           deterministic residues built by forward kinematics
```

The sign convention is fixed in `geometry.py` and **asserted** in
`tests/test_geometry.py::test_rotating_downstream_atoms_increases_the_dihedral_by_theta`:
rotating by `+θ` about the directed axis `b → c` adds `+θ` to the dihedral.

## Compute

Training runs on UQ's Bunya cluster. **Read `binder-physics/docs/BUNYA_BROWSER_SHELL.md` before
touching it** — access is a code-server pane inside an interactive SLURM job, the quota binds on
inodes rather than bytes, and a job's SLURM state is not a health signal. An NCA checkpoint or
rollout sweep writes exactly the many-small-files pattern that fills the inode quota, so check
`rquota` first.
