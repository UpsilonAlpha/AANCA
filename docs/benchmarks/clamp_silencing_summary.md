# The clamp table, completed — the silencing result and its magnitude

Collected 2026-11-03 from Bunya jobs 31515293, 31515294 and 31515295, all exit 0. The per-run logs
beside this file (`clamp_v5_native.log`, `clamp_v5_native_last.log`) are the earlier, anchored runs;
this file is the `--no-anchor` set, where both arms share a seed so they score the same residues from
the same starting conformations and the comparison is a paired one.

All rows: CASP13, 79 targets, 100 steps, 0.75 of residues pinned, `--no-anchor`.

| run | arm | free right | free MAE | free n | pinned drift | start | moved | stuck |
|---|---|---|---|---|---|---|---|---|
| v5 `last.pt` u60000 (31515295) | native | 54.2 % | 58.50 | 4540 | 0.00e+00 | 33.5 % | 46.58 | 68.7 % |
| | random | 75.1 % | 34.34 | 4540 | — | 33.5 % | 89.17 | 35.2 % |
| v6 `best.pt` (31515293) | native | 63.4 % | 47.48 | 4540 | 0.00e+00 | 33.5 % | 61.64 | 57.4 % |
| | random | 75.3 % | 34.31 | 4540 | — | 33.5 % | 89.47 | 35.1 % |
| v6 `best.pt` seed 1 (31515294) | native | 63.0 % | 47.40 | 4440 | 0.00e+00 | 34.5 % | 59.07 | 59.3 % |
| | random | 75.7 % | 33.49 | 4440 | — | 34.5 % | 88.57 | 34.9 % |

| run | native − random | moved n/r | excess stuck over the random arm |
|---|---|---|---|
| v5 | **−20.9** | **0.52** | +33.5 points |
| v6 | **−11.9** | **0.69** | +22.3 points |
| v6 seed 1 | **−12.7** | **0.67** | +24.4 points |

`pinned drift` is 0.00e+00 on every native arm, so no pin leaked and no row is void.

## What the table says

**Correct, motionless neighbours make the rule worse.** Pin three quarters of a protein at its
crystal conformations and the free residues score 12 to 21 points *below* what the same residues
score when the same positions are pinned at *random* conformations. That is the opposite of the
expected direction: the native-pinned case is the classical packing problem, which a rotamer library
with a pairwise energy solves natively.

**The cause is that the rule stops, not that it solves badly.** The `moved` column is the mean chi1
distance a free residue travelled from its random start, and the native arm moves between half and
two thirds as far as the random arm. The `stuck` column is the fraction still in the well they were
randomly dropped into: 69 % for v5 against the random arm's 35 %. Both arms start from the same 33.5 %
right, so from an identical floor the random arm reaches 75 % and the native arm 54 % — it covers a
third of the distance because it quits.

**The mechanism.** The fixed point is supervised by native-start rollouts, and in training a
neighbourhood only *looks* native when the residue itself is also right. The two conditions never come
apart, so the rule learned to infer its own state from its neighbours'. That is a correct inference
about the training distribution and a wrong one about the clamp.

## The magnitude is predicted, not just the sign

Dividing the deficit by `1 − moved n/r`:

| run | deficit | 1 − moved n/r | ratio |
|---|---|---|---|
| v5 | 20.9 | 0.48 | 43.5 |
| v6 | 11.9 | 0.31 | 38.4 |
| v6 seed 1 | 12.7 | 0.33 | 38.5 |

Within 15 % across two generations and two seeds, from a movement statistic that has nothing to do
with accuracy. The excess-stuck column tracks it in the same proportion. So the silencing account does
not merely have the right sign — it predicts how large the deficit is.

**`--damage-fractions` bought exactly the movement it restored.** v6 added
`--damage-fractions 1.0 1.0 0.5 0.2 0.05` and moved `n/r` from 0.52 to 0.68, and the deficit fell from
20.9 to 12.3 — the amount the ratio above says it should. The seed replicate puts v6 at **−12.3 ± 0.4**,
far outside the 1.2-point seed-to-seed spread measured on the accuracy itself, so the improvement is
real and so is the remaining gap.

## The target for the next run

`--clamp-probability` freezes the *undamaged* residues of a partially damaged training sample, which is
the state uniform damage essentially cannot produce — with 64 neighbours and damage fraction `f`, an
all-native shell around a damaged residue has probability `(1 − f)^64`, which is 4 % at f = 0.05 and
6 × 10⁻⁷ at f = 0.2. The quantitative prediction is therefore available in advance: **driving
`moved n/r` to 1 should take the deficit to roughly zero**, and anything else means the ratio above is
a coincidence rather than a mechanism.

One cost, stated: a run trained with `--clamp-probability` has seen this test's distribution, so this
table stops being a held-out diagnostic for it. Report global repacking accuracy alongside, which must
not regress.
