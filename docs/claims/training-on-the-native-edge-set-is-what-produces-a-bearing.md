---
type: observation
status: supported
evidence: ours
sources: [33320676, 33320677, 33320678, 33319941, 33319943]
---

# Trained on the native edge set, the same architecture emits a bearing clear of the null: g17_long reads align 0.384, CI [+0.300, +0.468], every crop positive

**Status note: `supported` for the measurement, not for a mechanism.** The
numbers are what they are and the crops are shared; what no arm here can
support is that the edge set *alone* caused the difference, because nothing in
this project has ever been replicated and the arms differ in more than the edge
set. Read the comparison section before quoting the headline.

**This is the unambiguous branch the readout pre-registration asked for.**
Holding the native edge set at *measurement* time changed nothing
([[the-perception-graph-is-rebuilt-from-where-the-residue-wrongly-is]]),
and that was registered as uninterpretable because those weights had never
trained on edges they did not see. `g13`/`g15`/`g17`/`g18` **did** train on
them, so "never learned to use them" does not apply to them. Jobs
33320676/77/78 with `--native-edges`, exit 0 each, every log printing
`recorded settings: native_edges = True (retired; reproduced by
--native-edges)` -- the refusal guard reconciling correctly in production
rather than in a test.

## RETRACTED WITHIN THE HOUR: the edge set is not the variable, `lab_rotation` is

**The attribution in this note's title is withdrawn.** Three further readouts,
submitted before this was written and read after, settle it against me.
`g19_nomove` and `g20_orient` use the **live k-NN edge set**, no teacher
forcing and no native edges -- and read `align` **+0.293** and **+0.329**, with
every crop positive, indistinguishable from the native-edge arms. Resolving
every differing setting through `config.resolve`:

    arm              lab_rot  native  teacher  orient  upd     align
    g17_long          True     yes     yes      0     10000   +0.384
    g20_orient        True     no      no       2.5   10000   +0.329
    g13_natedge       True     yes     yes      0       750   +0.312
    g19_nomove        True     no      no       0      9250   +0.293
    g15_both          True     yes     yes      0      1750   +0.159
    ---------------------------------------------------------------
    g25_equivariant  FALSE     no      no       2.5   10000   +0.122
    g28_ceil_local   FALSE     no      yes      0      2000   +0.036

**Every arm with `lab_rotation = True` reads 0.159 to 0.384. Both arms with
`lab_rotation = False` read 0.036 and 0.122.** The edge set varies *within* the
upper group and makes no systematic difference; the teacher varies within it
too. The one setting that sorts the table is the rotation convention -- the
setting the project "fixed".

**The leading explanation is the instrument, and it is pre-registered.**
`align` is a cosine between the emitted move and the ideal one, and the readout
prints *"emitted move rotated into the lab frame for comparison"*. If that
rotation matches the convention of a `lab_rotation = True` run and not a
`lab_rotation = False` one, the split is an artefact of the measurement rather
than a property of the rules. Bucket 4 of
[[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]] names exactly
this -- *"re-check the readout's frame convention before reading anything into
it: that exact class of error has already cost this project a retraction"* --
and although no value here is below its -0.10 trigger, a systematic split on
precisely the rotation setting is the same warning arriving by a different
route. **Nothing in this note should be quoted until that convention is
re-derived.**

**The discriminating arm, predicted before it was read.** `g14_pairwise` is
`lab_rotation = True`, live edges, teacher-forced, 2,000 updates, `pairwise
0.01` -- which makes it `g28` with the rotation convention flipped and almost
nothing else (`g28` has `pairwise 0`). So:

  * if the rotation convention is the variable, `g14` reads roughly **0.15 to
    0.35**;
  * if teacher forcing on live edges is, `g14` reads roughly **0.04**, like
    `g28`.

Job 33321241 was submitted before this paragraph and is read below. Recorded
this way round so the prediction cannot be fitted to the answer.

**What survives of this note:** the five `align` numbers, the shared crop
indices, and the observation that no arm on this architecture has ever shown a
bearing clear of the null *with the local rotation*. What does not survive is
the claim that the native edge set is what produced one.

## align, with the spread, on the twelve held-out crops

    arm            edge set   teacher   upd      mean    95% CI            neg
    g17_long       NATIVE     yes      10000   +0.384  [+0.300, +0.468]    0/11
    g13_natedge    NATIVE     yes        750   +0.312  [+0.149, +0.475]    2/11
    g15_both       NATIVE     yes       1750   +0.159  [-0.026, +0.344]    4/11
    g25_equivariant  live     no       10000   +0.122  [-0.002, +0.246]    1/11
    g28_ceil_local   live     yes       2000   +0.036  [-0.043, +0.114]    6/12

(`upd` is the update each arm's `best.pt` was written at, not its length:
`g13` best-checkpointed at 750 of 2,000 and `g15` at 1,750.)

**`g17_long`'s whole confidence interval sits above the pre-registered null
band** of -0.10 to +0.25, and above the +0.206 convention-matched null. **Every
one of its crops is positive**, where `g28` has six of twelve negative. It is a
different distribution, not a shifted one.

Pre-registered bucket: **0.25-0.50, partial, direction only**
([[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]]). Its wording
is adopted as written -- report the direction and **say the magnitude is not a
result**. 0.384 is a third of the 0.96 the same architecture reaches when a
gradient is handed to it.

## The comparison that carries it is nearly one variable

`g17` against `g25` confounds the edge set with teacher forcing. **`g28` is the
control that breaks the confound**: teacher-forced, identity off, a *fixed*
actuator, and the live k-NN edge set -- and it reads 0.036 with half its crops
negative. So teacher forcing alone does not produce a bearing, and the
remaining difference that does is the edge set.

**It is not literally one variable, and the residue runs against the project's
own actuator story.** `g13`/`g15`/`g17` carry `--lab-rotation`, the *broken*
actuator
([[the-emitted-move-was-applied-in-the-lab-frame-and-perception-is-invariant]]),
plus 10,000 updates against `g28`'s 2,000 and `pairwise 0.01` against `g28`'s
0. **The arms with the broken actuator and the native edge set aim five to ten
times better than the arm with the fixed actuator and the live edge set.** That
does not refute the actuator node, which is about representability rather than
about this comparison, but it does say the actuator was not the binding
constraint it was taken for.

## The bearing grows with training, and only with native edges

`g15` and `g17` are the **same settings** at different lengths -- `g17` is
`g15` with `updates = 10000`. Along that one axis:

    g15, best at u1750    +0.159   CI crosses zero
    g17, best at u10000   +0.384   CI entirely above the null band

Whereas `g25` ran the **same 10,000 updates on the live edge set** and reads
+0.122 with its CI crossing zero. So length is not what produces the bearing;
length *plus* a neighbourhood that is not a function of the error is.

## What this does and does not settle

  * **It answers the edge-set question in the direction the hypothesis
    predicted**, and it does so on arms that trained with the edge set, which
    is what the no-retraining null could not do. The grid node's third gift --
    the determinate neighbourhood -- is the one that was missing and the one
    that matters
    ([[in-regeneration-the-surviving-context-is-the-grid]]).
  * **It does not reopen the capacity question, it confirms its scoping.**
    [[the-architecture-can-place-residues-under-some-supervision]] closed NO
    for *invariant perception over an error-derived neighbourhood*, and that
    scope was written into the result before this was measured. Change the
    neighbourhood and the architecture aims. The NO stands for what it says.
  * **Every number here is a ceiling twice over**: teacher-forced, and with the
    contact graph given. Under
    [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]] the given
    graph is the task statement rather than a leak, but the teacher is not.
    **No `rmsd_dmg` from these runs is a regeneration result**, and the
    readout's crops are resampled per run, so no placement number here
    compares with any other arm's.
  * **The five arms above share the same twelve crop indices**
    `[180, 167, 393, 264, 330, 193, 348, 297, 6, 319, 225, 129]`, verified
    from the checkpoints in
    [[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]],
    which is what makes a cross-arm `align` comparison legitimate at all.
  * **`g18_seed1` is NOT a replicate, and calling it one here was wrong.** It
    read `align` **-0.106** against `g15`'s +0.159 and I first recorded that as
    the project's first seed replicate overturning the result. It is not:
    `seed` does two jobs, seeding initialisation *and choosing the split*, so
    `g18`'s held-out set
    `[31, 45, 131, 138, 154, 164, 174, 200, 289, 294, 327, 388]` is **disjoint**
    from `g15`'s ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]],
    which is why its placement number was voided). So -0.106 is a measurement
    of *different proteins* and bounds nothing about `g15`, in either
    direction. It neither undermines nor supports anything here.
  * **So there is still no error bar, on anything.** No arm in this project has
    ever been replicated, `g17` included, and the only CIs available are
    within-arm across correlated crops -- which say nothing about seed
    variation. The `g13`-vs-`g15` gap (+0.312 against +0.159 at different best
    checkpoints, intervals overlapping) is the standing warning that
    differences at this scale may not survive one.
  * **`g17`'s twist is 14.3 degrees**, outside the project's 9-12 band, so its
    *placement* number carries a regime marker. By the clause pre-registered in
    [[teacher-forcing-a-sound-actuator-leaves-the-bearing-at-zero]] -- "a twist
    regime marker does not void this arm's `align`", because `align` is not a
    cross-regime placement comparison -- that does not reach the number read
    here. Stated because it would otherwise look unaddressed.
  * The CIs are normal-approximation over 11-12 correlated crops; residues
    within one window are not independent, so the intervals are optimistic.

## Supports
- [[in-regeneration-the-surviving-context-is-the-grid]]
- [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]
