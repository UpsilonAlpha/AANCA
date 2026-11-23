---
type: observation
status: supported
evidence: ours
sources: [33320676, 33320677, 33320678, 33319941, 33319943]
---

# Trained on the native edge set, the same architecture does learn a bearing: g17_long reads align 0.384, CI [+0.300, +0.468], every crop positive

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
  * **One seed each, and the replicate is still running.** `g18_seed1` is
    `g15` at `seed = 1` -- the only seed replicate this project has ever run
    ([[no-growth-arm-has-been-run-twice-so-no-difference-has-an-error-bar]]).
    Its `align` against `g15`'s +0.159 will be the first same-settings
    different-seed comparison in the record, and the `g13`-vs-`g15` gap
    (+0.312 against +0.159, overlapping intervals) is a warning that
    between-arm differences at this scale may be seed noise.
  * The CIs are normal-approximation over 11-12 correlated crops; residues
    within one window are not independent, so the intervals are optimistic.

## Supports
- [[in-regeneration-the-surviving-context-is-the-grid]]
- [[the-task-is-to-realise-coordinates-for-a-given-contact-graph]]
