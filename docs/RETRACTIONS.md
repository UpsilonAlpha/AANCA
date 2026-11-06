# Retracted measurements

Jobs and scripts whose output was **invalid**, so that any node citing one is
flagged by `docs/claims/check.py` (check 4) until someone records that they read
the correction. Format and the idea are borrowed from the pMHC binder session's
`binder-physics/docs/PMHC_RETRACTIONS.md`: a correction written only into a
later journal entry means a grep lands on the retracted half, so the retraction
has to be indexed where the citation is checked.

Acknowledge one by listing its id in a node's `checked:` field. That is an
assertion — "I read the correction" — not a way to silence the check.

## Invalid measurements

| id | what it claimed | why it is void |
| --- | --- | --- |
| **32240477** | per-term gradient norms on trained g5_ctrl weights | 1e+08 to 1e+11 on every row with every forward value finite — a diverged backward pass. Ran 1UBQ at 6 seeds / 30 steps against a run trained on crop-128 / 16 seeds / 120 steps. |
| **32240478** | the same, on g6_self | exit 1 with one line of output. Built a birth head the checkpoint does not have, and `load_state_dict` refused. |
| **32241768** | per-term gradients, g5_ctrl, after the `--crop` fix | still 5e+10. The crop was never the fault; `best.pt` carried no `args`, so seeds and steps silently stayed at the probe's defaults. |
| **32241769** | per-term gradients, g6_self, after the `--crop` fix | **finite, plausible and wrong** — reported `birth free` for a `--birth none` run, so the table describes a *grown* rollout for a self-placing arm. The dangerous one: it announced nothing. |
| **32421950** | the bearing (`align`, `net/path`) of the equivariant arm | **a frame mismatch in the diagnostic, not in the rule.** Under `--equivariant-move` the emitted move is a vector in each residue's own N-CA-C basis; `_bearing` compared it against a lab-frame difference of CA positions, so the cosine was between two different spaces. It read -0.676 with `net/path` exactly 1.000 and `step` identical to three digits across twelve crops -- too clean, which is the tell. Correcting the convention flips the sign on an untrained control (+0.206). `rmsd_dmg` and `clash` from this job are unaffected and stand. |
| **32405172** | `move`, the ceiling check's own loss | **partially void, and only for that column.** `evaluate()` never passed `move_weight`, and `growth_objective` computes the term only `if weight:`, so the held-out `move` was structurally absent rather than merely unprinted. Its `rmsd_dmg` (12.23 A, flat) is valid and is cited elsewhere; `move` from this job is not. |

## Claims measured away

These are not bad measurements but conclusions that later evidence killed. They
live as `void` nodes in `docs/claims/` carrying their own `## Refuted by`
edges, which is where the graph checks them; listed here so the two indices
agree.

| claim | killed by |
| --- | --- |
| the orientation term buys ~22 degrees of twist | crop-matched probing — the 90.2 deg comparator came from g4, a different crop sample, and the arm was the *worst* of three |
| the identity gradient starves the geometry | g6_noid (32241722), perfectly static with geometry as the entire objective |
| FAPE's clamp is what gates the geometry | g6_open (32359791), 17.53 A and twist nine degrees worse than its baseline |
| the four frozen arms measured the architecture's ceiling | they measured whole-structure reconstruction, which is not the project's task |

## Already marked

Nothing yet. Entries move here once every citing node has acknowledged them, so
the table above stays short enough to read.
