---
type: observation
status: supported
evidence: ours
sources: [33304426, 33291294, 33303677]
---

# The guard added to stop silent pool mismatches never recorded a single id, and killed the first arm that ran after it

`g28_ceil_local`, job 33304426, died nine minutes in with `train exit : 1`
after completing its 250-update held-out evaluation:

    File "scripts/train_growth.py", line 1473, in main
        "eval_pdb_ids": [pool[i].pdb_id for i in eval_indices],
    KeyError: 180

## The fault

Two variables named `pool` in one function scope.

  * `pool = available[:wanted]`, the **list of structures** `eval_indices`
    are offsets into.
  * `pool: dict[tuple[int, int | None], GrowthPoolEntry] = {}`, the
    **Growing-NCA sample pool**, assigned ~220 lines later and keyed by
    `(structure index, crop seed)` -- deliberately not by index alone,
    because `load_crop` re-crops on every visit.

The second shadows the first from its assignment onward, and the checkpoint
writer sits after it. So `pool[i]` looked up a tuple-keyed dict with a bare
integer. **No integer is ever a key of that dict**, which means this line
could not succeed on any machine, with any pool, at any update. `eval_indices`
is drawn from a *shuffled* `indices`, which is why the key was 180 and not 0 --
had it been 0 and the sample pool been keyed by index, the line would have
returned the wrong protein silently instead of raising.

## Two consequences, and the second is the one that matters

**One arm lost.** 33304426 is gone and `g28_ceil_local` has not been run. Its
pre-registration
([[the-capacity-test-has-never-been-run-with-a-working-actuator-and-no-identity-break]])
stands untouched and unread: no `align` number exists, and the arm was
confirmed to be the right one before it died, so the resubmission needs no
re-confirmation of the section, the teacher forcing or the local rotation.
The 250-update row it did produce -- `rmsd_dmg 11.87 A`, `fit_intact 12.88 A`,
twist 13.9d -- is a 250-update row and is not read against any bucket.

**Every checkpoint in the project lacks the field, so the guard never fired
once.** It was added on 21 Nov (`aa8226e`) specifically because
`scripts/geometric_floor.py` had been found resolving `eval_indices` against
the wrong pool, and the whole point was that a probe could then *refuse*
rather than measure. All six probe jobs that have consumed it instead printed

    !! this checkpoint records no eval_pdb_ids (written before 21 Nov), so
    the crop IDS are unverified

-- 33291294/95/97 (the basins) and 33303677/79/81 (the cross-tabulation) --
and measured anyway. The probes' fallback is honest and loud; what was wrong
was [[the-rule-learned-helical-torsions-and-learned-nothing-about-sheet]]
asserting the check had passed. Corrected there.

**This is not a reason to doubt those tables.** The probes ran on Bunya
against the same data dir the arms trained in, reconstructing each pool from
the arm's own recorded `args`, which is the condition under which an index
names the same protein. The defect is that the condition was assumed rather
than asserted -- the exact gap the field was introduced to close.

## The fix, and why the fix is in a different place than the bug

The list is renamed `catalogue`, which removes the collision. But
`eval_pdb_ids` is now resolved **once, before the first update**, rather than
inside the checkpoint writer, and a failure to resolve it returns 1 with a
message. A run that cannot name the proteins it held out should refuse in its
first second, when the only thing lost is the queue wait, instead of after a
GPU hour. The resume checkpoint records the ids as well as `best.pt`, so a
probe pointed at `last.pt` is as checkable as one pointed at `best.pt`.

## Scope

  * **The guard will only start working for runs submitted after this fix.**
    Every existing checkpoint still has no ids and still draws the warning.
    Nothing retroactively verifies the arms already measured, and no
    re-measurement would, since the field is written at training time.
  * **A green test suite did not catch this and would not have.** 25 config
    tests and the full suite passed over the broken line for two days,
    because nothing asserted what a checkpoint *contains*. A regression test
    on the checkpoint's contents is part of this fix.
  * **The guard covers one of the two trainers.** `scripts/train.py` records
    `eval_indices` and no `eval_pdb_ids` at all, and its structure list is not
    called `pool`, so it never had this bug and still has no ids. Both probes
    therefore fall back to the same warning for *every* `train.py`
    checkpoint, new or old, which means the warning cannot be read as "this
    checkpoint is stale" -- only as "this checkpoint has no ids". The growth
    arms all come from `train_growth.py`, so this does not touch them.
  * It says nothing about the architecture, the objective, or any number in
    the record.

## Verification of the fix

A one-update run of the real driver on 9 structures, held-out split
`[7, 5, 1]` after the seed-0 shuffle -- deliberately not in order, since an
in-order split would not exercise the fault -- writes both `best.pt` and
`last.pt` with `eval_pdb_ids = ['3Q5T', '4JDZ', '4DVT']`, equal to the ids
resolved independently from the manifest. The same config on the pre-fix code
dies exactly as the cluster did, `KeyError: 7` at the same line.
`tests/test_growth_checkpoint_ids.py` holds two tests, both of which fail on
the pre-fix code: one asserts the checkpoint's *contents* against
independently resolved ids (and that the split is not in order, so it cannot
silently stop testing anything), and one parses the trainer with `ast` and
asserts the structure list and the sample pool do not share a name. Suite:
966 passed.

## Supports
- [[the-capacity-test-has-never-been-run-with-a-working-actuator-and-no-identity-break]]
