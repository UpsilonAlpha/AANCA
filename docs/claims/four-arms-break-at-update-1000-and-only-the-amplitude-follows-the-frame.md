---
type: observation
status: supported
evidence: ours
sources: [32405172, 32420154, 32449660, 32449661]
checked: [32405172]
---

# All four frame arms break at update 1000, and the one with no equivariance breaks too

Held-out twist, degrees, across four arms that differ **only** in how the
emitted move is framed:

    update    g8      g11     g10     g12
      250     9.1     9.1     9.9     9.8
      500     9.1     9.0    10.0     9.5
      750     9.0     9.1    10.2    10.4
     1000    16.1    18.4    52.3    43.3        <- all four, same row
     1250    13.9    17.9    50.0    55.4
     1500    16.3    26.0    46.6    51.8
     1750    16.5    37.4    48.8    28.0
     2000    16.6    17.6    50.2    53.9

    g8_ceiling      lab translation,   lab rotation
    g11_split       local translation, lab rotation
    g10_equiv_ceil  local translation, local rotation
    g12_trunc       local translation, local rotation, --truncate 60

`g8_ceiling` has **neither half of the equivariance fix** and jumps 9.0 -> 16.1
at update 1000, with `fape` 0.146 -> 0.843 and `bond` 6.03 -> 7.60 in the same
row. So the ~1000-update transition is not a consequence of the fix.

**The amplitude, however, splits cleanly on the rotation half:**

    local rotation absent   g8  16.6 deg    g11  17.6 deg
    local rotation present  g10 50.2 deg    g12  53.9 deg

Threefold, two arms either side of one variable. The conjugated rotation does
not cause the transition; it makes the rule about three times more sensitive to
it.

**Nothing in the code is keyed to update 1000.** There is no learning-rate
schedule in the growth trainer (`src/aanca/schedule.py` exists but
`train_growth.py` never calls it, and Adam runs at a constant 3e-4),
`--fape-clamp-prob` is at its default of 1.0, and the only `% 1000` in the tree
is a logging line in a different trainer. So the coincidence is in the training
dynamics rather than in a switch, which is what
[[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]] is
about.

**32405172 is in `checked`**: that job's `move` column is void
(`evaluate()` never passed `move_weight`), and no `move` value from it is used
here. Its twist, bond, fape, identity NLL and `rmsd_dmg` are unaffected.

## Supports
- [[the-twist-jump-at-update-1000-is-the-identity-head-leaving-its-floor]]
