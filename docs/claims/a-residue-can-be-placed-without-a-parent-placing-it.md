---
type: claim
status: supported
sources: [32215014, 32241722]
---

# A residue can place itself, with no parent to inherit a pose from

The `--birth none` architecture. Every slot is alive from step 0 at a
constructed baseline conformation, one anchor per covalent run keeps its
crystal pose, and no residue inherits `rot_parent @ rot_reference @ rot_spin`
from anything. This replaced the whole birth-mode progression
(`torsion`/`cartesian`/`free`) as the way a pose gets assigned.

Held because the machinery works and is measured: the rollout keeps 100 % alive
with completion, reach and frontier at exactly zero, the activation head takes
exactly zero gradient, parameter count is 242,891 -- identical to `free`, so an
arm difference cannot be model size -- and the baseline has exact peptide
geometry by construction (bond term 0.0021 before any update). Fifteen test
cases in `tests/test_self_placing.py` pin it.

**This premise holds. It is the only one of the three that does.** What it does
not establish is that the rule then *moves* the residue anywhere, which is the
other two premises.

## Supported by
- [[the-baselines-are-17-45-and-36-73-angstrom-from-native]]
