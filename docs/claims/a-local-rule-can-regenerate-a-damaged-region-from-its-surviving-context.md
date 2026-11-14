---
type: claim
status: open
sources: [growth.py]
---

# A local per-residue rule can regenerate a damaged region of a protein from the context that survived

**The near-term goal, and the most useful thing this project could produce.**
Delete a part of a protein and have the rule fill it back in using only local
information from the intact remainder: that is the Growing-NCA objective --
regeneration from damage, stability past the trained horizon -- applied to
protein geometry, and a prior-art search on 2026-10-27 found it unpublished.
The novelty is the **objective**, not the architecture; existing packers
predict in a fixed number of passes or along a fixed diffusion schedule, and
none trains for a fixed point reached from many starting conditions over a step
count it was never told.

**Rescoped 14 Nov: this is the central argument the project is now testing.**
The thesis above it -- a local rule building a whole backbone -- stays at the
root, because this node lacks the grid's positional information rather than
lacking plausibility
([[a-local-per-residue-rule-can-build-a-protein-backbone]]). But the question
being *put to the rule* is this one: can a very simple shared local rule regrow
damaged structure from surviving context? Inverse folding already answered the
weaker question of whether the architecture can learn from local context at all
-- 38.0 % held-out, converged -- so regeneration is where the project either
has a result or does not. Design performance against RFD4-Proteina-class models
is explicitly not the comparison
([[what-an-aanca-can-learn-that-a-pairformer-cannot-and-the-reverse]]), and the
downstream programmes -- a fitted Potts potential, CDR pre-organisation,
binders -- are strictly downstream of this node and not worth a slot until it
settles.

**Trained, and it worked once.** `g20_orient` moved a 12-residue window from
the 12.23 A baseline to **7.78 A** on 12 held-out crops with **no native
information inside the rollout** -- no teacher forcing, no target graph
([[with-no-ceiling-at-all-the-rule-places-a-damaged-window-to-7-8-angstrom]]).
Nine earlier arms without teacher forcing had sat at their own baselines to two
decimal places, so the signal is real.

Three gaps keep it `open`, and they are specific rather than cautious.

  1. **Unreplicated.** No arm in this project has been replicated at a second
     seed, because `seed` also chooses the held-out set, so the one replicate
     attempted measured different proteins
     ([[question-has-any-arm-been-replicated-at-a-second-seed]]). One
     `eval_ids` config key and one run away.
  2. **It does not persist.** 16.4-16.9 A at 600 steps against a 12.23 A
     untrained baseline, so 7.78 A is a snapshot of a transient at step 120
     ([[the-rule-has-no-attractor-past-its-training-horizon]]).
  3. **The start state is a prior, not noise.** `baseline_coords(...,
     initial="helix", ...)` discards the window's native pose and regrows it in
     torsion space off the intact flanks at ideal (phi, psi, omega), so the
     rule begins from an informative guess. How much of the 4.45 A is the rule
     and how much is reachable by relaxing that helix against fixed flanks has
     **not been measured**, and it is the first thing a reviewer would ask.

`--damage N` destroys a contiguous window inside one covalent run and keeps
every other residue at its crystal pose. Measured on an untrained rule over
three held-out crops with a 12-residue window, against the reconstruction arms'
own held-out numbers:

    quantity        regeneration   reconstruction
    fape                0.186           0.836
    twist              10.7 deg        64.4 deg
    rmsd_dmg          10.76 A          --

fape and twist are means over pairs, and with the context intact most pairs are
already correct -- which is why those two numbers are **not** comparable
between the tasks, and why a regeneration run reports `rmsd_dmg`: the window
alone, superposed on the context so it measures placement rather than plausible
shape elsewhere.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
- [[regrowing-a-designed-interface-can-produce-a-better-binder]]
- [[a-peptide-motif-can-be-grown-a-scaffold-around-it]]

## Requires
- [[a-residue-can-be-placed-without-a-parent-placing-it]]
- [[the-damaged-region-must-start-within-reach-of-where-it-belongs]]
- [[question-is-7-8-angstrom-better-than-relaxing-the-helix-prior]]
