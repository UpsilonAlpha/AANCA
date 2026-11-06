---
type: claim
status: supported
sources: [baseline_probe.py, steric.py]
---

# Every instrument here is anchored both ways except the one the thesis turns on

The pMHC binder session's generalisation, which it derived from the oracle panel
built here and then applied to its own arms, finding a ceiling-free one it had
read as two-sided for days: **ask any instrument two independent questions.**

  * **Convention** -- is it reading the right thing? Answerable only by two
    independently derived quantities that must agree, never by a planted
    signal, because a wrong-basis cosine of a planted displacement still moves
    in the right direction.
  * **Scale** -- is it anchored on *both* sides? Answerable only by a positive
    that genuinely does the thing, never by accumulating negatives.

Audited against this project's instruments, which mostly survive:

    instrument   floor (nothing)         ceiling (the thing)        two-sided?
    rmsd_dmg     10.77 A, model-free     0.00 A, the native        yes
    clash        0.00002, a crystal      0.092, a shoved window    yes
    bond         ~0.002, a crystal       ~4.5, an open join        yes
    twist        0 deg, a crystal        64.4 / 9.6, baselines     yes
    align        -0.4..+0.25, random     1.000, the oracle         yes, as of
                 heads                                             panel C
    net/path     0, a walk that turns    1.0, a straight line      by construction

**The exception is the success criterion itself, and it is the one that
matters.** `rmsd_dmg` is anchored at *do nothing* (10.77 A) and at *reproduce
the crystal* (0.00 A). It has **no anchor for a good non-native answer** --
which is precisely what the project is for: regrowing a binder interface "into
something that binds better than the original"
([[regrowing-a-designed-interface-can-produce-a-better-binder]]) would score
badly against the crystal and be the result wanted.

[[the-objective-should-restrain-the-output-not-teach-the-placement]] already
recorded that a structure which is valid, closed, clash-free and 10 A from the
crystal is indistinguishable by loss from the failure mode. What this adds is
that it is a **missing positive control**, not merely a measurement gap -- and
missing positives are the error that produces optimistic bias, which is exactly
what the audit above found in the other project: a design pool sitting entirely
above its only external anchor, recorded as a strength for days.

**This project's bias runs the other way, which is luck rather than virtue.**
Every arm has scored *worse* than the 10.77 A floor -- 12.18 to 12.67 -- so the
pool sits entirely on the wrong side of its only anchor and no optimistic
reading was available. Had an arm landed at 9 A the question "nine Angstrom
better than what, and worse than what?" would have had no answer.

## How to get the anchor

Borrowed from the same session's method, where a cross-deposition null
(0.340-0.490 A between unrelated depositions) was what made a 0.441 A
measurement interpretable: **measure what the same loop reads between two
depositions of the same protein.** If a 12-residue window differs by X A
between two crystal forms, then X is the scale at which "different but real"
lives, and an arm within X of native is as close as the crystal is reproducible.

Not built. It needs pairs of depositions of one protein in the 412-structure
pool, which requires sequence comparison to find and may come up empty. Recorded
as the concrete next step for the criterion rather than left as "we cannot tell
better from wrong".

## Supports
- [[the-objective-should-restrain-the-output-not-teach-the-placement]]
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
