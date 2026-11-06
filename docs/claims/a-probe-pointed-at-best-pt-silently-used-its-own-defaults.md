---
type: observation
status: supported
evidence: ours
sources: [32240477, 32240478, 32241768, 32241769]
---

# best.pt carried only the weights, so four probe runs silently measured the probe

The trainer wrote `vars(args)` into `last.pt` and nowhere else, so
`state.get("args")` on a `best.pt` returned an empty dict and every rollout
setting fell through to the probe's own default: 6 seeds where the run trained
on 16, 30 steps where it trained on 120, no `eval_indices` so the held-out
split was reshuffled, and birth `free` where the run is `none`.

Four invalid results from one cause:

    32240478   exit 1, one line: built a birth head the checkpoint lacks
    32240477   gradient norms 1e+08 to 1e+11, every forward value finite
    32241768   still 5e+10 after the --crop fix, because the crop was not the fault
    32241769   finite, plausible, and measuring a GROWN rollout for a
               self-placing run

The fourth is the dangerous shape: it announced nothing. Two fixes, because the
failure has two halves -- `best.pt` now carries `args`, `eval_indices` and
`update`; and a probe handed a checkpoint without arguments reads the sibling
`last.pt` or **exits 1**, because a default is indistinguishable from a
measurement once it is printed under a column heading.
`tests/test_probe_configuration.py` pins all three paths, including the refusal.

This is the fourth number in the project produced by a probe configured unlike
the run it read. The first three were each diagnosed as a one-off; four says the
class is structural.

## Supports
- [[a-collapsed-rules-twist-cannot-be-compared-across-crop-samples]]

## Refutes
- [[the-identity-gradient-starves-the-geometry]]
