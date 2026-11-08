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
| **32423296** | the corrected bearing of `g10_equiv_ceil` *at 2000 updates* | **mislabelled, not miscomputed.** The job ran 01:05:19-01:10:55 and `runs/g10_equiv_ceil/last.pt` was overwritten at 01:10:13 by the run that owns it, 34 s before the job ended — so it read a complete but *intermediate* checkpoint, somewhere around update 1860-1922, and nothing in its output recorded which. Reported here as the arm's result: `align` +0.008, `rmsd_dmg` 12.52, per-crop -0.534 to +0.457. The arm's actual result is 32449992 (`align` -0.096, `rmsd_dmg` 12.85). The numbers are real measurements of an unrecorded intermediate, so they are void **as attributions to the arm** rather than wrong. `best.pt` carrying no args in a second costume; the readout now prints the checkpoint's update and mtime. |
| **32449992, 32449993, 32453500, 32453501** | "crops needing more than the gradient window", and the crop-297 hypothesis built on it | **a quantity labelled as the need and computed from the outcome.** `ceiling_readout.py` tested the truncation window against `r[3]`, the *final* `rmsd_dmg`, so "11 of 12 crops need more than 3.6 A" was the outcome distribution relabelled. The crop-297 story -- "the one crop that aims is the only one the gradient can follow to arrival" -- is therefore circular: window membership was decided by which crop *ended* closest, so of course the closest-ending crop was inside it. The readout now computes the start-state window RMSD with the same `ca_rmsd_on` and prints `need`, `rmsd_dmg` and `need - rmsd_dmg` side by side. `align`, `step`, `net/path`, `rmsd_dmg` and the agreement correlations from these jobs are unaffected and stand. |
| **32405172** | `move`, the ceiling check's own loss | **partially void, and only for that column.** `evaluate()` never passed `move_weight`, and `growth_objective` computes the term only `if weight:`, so the held-out `move` was structurally absent rather than merely unprinted. Its `rmsd_dmg` (12.23 A, flat) is valid and is cited elsewhere; `move` from this job is not. |

## Claims measured away

These are not bad measurements but conclusions that later evidence killed. They
live as `void` nodes in `docs/claims/` carrying their own `## Refuted by`
edges, which is where the graph checks them; listed here so the two indices
agree.

| claim | killed by |
| --- | --- |
| the regeneration baseline is 10.77 A, and every arm at 12.18-12.85 is worse than doing nothing | the same script on **Bunya's** pool (32462158), which gives **12.23 A**. 10.77 came from a laptop checkout holding 1,903 structures against Bunya's 24,709: both scripts take `available[:412]`, so a different set of files on disk means different proteins behind the same crop indices, and no arm ever saw the structures 10.77 was measured on. The arms are not worse than doing nothing -- they are *doing nothing*, sitting on their own baseline to three significant figures. Replaced by `the-regeneration-baseline-is-12-23-angstrom-and-every-arm-sits-on-it`. |
| the ~1000-update twist explosion is the conjugated rotation's, and `--lab-rotation` will show which half of the equivariance fix caused it | `g8_ceiling` (32405172), which has **neither** half of the fix and breaks at the same update, with `fape` and `bond` in the same row. Written into `growth.py`, `train_growth.py`, HANDOFF and a claim node from a two-arm comparison with no control in it — and g8's log was already on disk. The amplitude split (16-18 deg without a local rotation, 50-54 with) survives and is the narrower claim the split arm was built to make. The pre-registered decision rule assumed the fix was the cause in **both** of its branches, so it is void rather than answered; on the face of it the onset "follows the translation", which is the branch that would have been reported. |
| the orientation term buys ~22 degrees of twist | crop-matched probing — the 90.2 deg comparator came from g4, a different crop sample, and the arm was the *worst* of three |
| the identity gradient starves the geometry | g6_noid (32241722), perfectly static with geometry as the entire objective |
| FAPE's clamp is what gates the geometry | g6_open (32359791), 17.53 A and twist nine degrees worse than its baseline |
| placement and internal shape are being traded against each other, so the window-to-window weight is the indicated next arm | the transit geometry this project had already measured and pinned (`tests/test_pairwise_restraint.py`, 3 Nov): 16.88 is the window's value at **zero** displacement, and a window walking straight from baseline to native rises to **4.95x** its baseline before falling. The three arms sit at 1.47-1.85x at matched displacement -- below the shortest path, not above it -- and the ordering that looked like a trade is the ordering of how far each arm travelled. `window` is also an unsigned MSE, so it cannot separate collapse from stretch. The counter-measurement was never a node, so no `refutes` edge existed for check 2 to fire on; it is now `a-straight-line-between-two-poses-passes-through-a-contracted-intermediate`. Replaced by `placement-and-internal-shape-are-traded-against-each-other` (void). |
| the four frozen arms measured the architecture's ceiling | they measured whole-structure reconstruction, which is not the project's task |

## Already marked

Nothing yet. Entries move here once every citing node has acknowledged them, so
the table above stays short enough to read.
