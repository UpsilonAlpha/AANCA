---
type: claim
status: open
sources: [growth.py, steric.py]
---

# The loss should restrain the output to be protein-like and let the rule discover placement itself

Ulban's framing, 7 Nov, and it decides the shape of every term rather than
just their weights: **the AANCA is the centrepiece, so the more supervision the
objective carries, the weaker the result.** A rule told where each residue goes
is a regression fitter that happens to be arranged as a cellular automaton.
Physically realistic restraints -- the ones that make an output actually look
like a protein -- are fine and loose. Redundant restraints are not.

**This killed a term that was half-built.** The reading of
[[the-reach-probes-split-the-three-frozen-arms]] is that FAPE supplies a
magnitude and not a bearing, and the obvious repair is to supervise the rule's
emitted move against the ideal one: the target pose is exactly computable at
every step, giving 128 residues x 12 live steps x 6 degrees of freedom of
signal against FAPE's one clamped scalar. It would probably have moved
`rmsd_dmg`. It is also teacher forcing, and it would have made the result
uninteresting in precisely the way this claim warns about. The plumbing was
reverted unused.

**It came back an hour later in a different role, and the distinction is the
whole point.** As the objective it kills the novelty. As **rung zero** -- a
positive control answering whether the architecture can place residues at all
-- it is what the information ladder requires before any looser result can be
read ([[question-can-the-architecture-place-residues-under-any-supervision]]).
It is labelled a diagnostic in its own docstring, in `step_once`, in the
`--move-weight` help text and in a banner line the run prints, because the one
way it does damage is by quietly becoming the objective.

## The audit, since "no redundant restraints" needs the list

    term          what it constrains              native?   verdict
    bond          C-N length, omega planarity      no        keep, physical
    reach         closure feasibility              no        keep, geometric
    clash         non-bonded atom overlap          no        MISSING -- add
    fape          placement onto the crystal       YES       keep, loose
    identity      residue type from neighbours     yes*      keep, drives trunk
    orientation   bonded-pair relative rotation    YES       already 0, leave
    completion    frontier activation              no        dead under `none`

`orientation` is the one that would have been redundant: a bonded pair's
relative rotation *is* (phi, psi, omega), so it overlaps `bond` on omega and
duplicates a Ramachandran restraint on phi/psi -- while being supervised
against the native, which is the thing to avoid. It has been weighted 0 in
every g5, g6 and g7 arm already, so there is nothing to remove; it stays as the
reported `twist` diagnostic.

**A Ramachandran restraint was considered and rejected as redundant.** The
Ramachandran plot is the empirical shadow of backbone sterics -- the disallowed
regions are disallowed because adjacent atoms collide -- so a proper clash term
subsumes its hard constraint. Adding both would double-count. What is lost is
the alpha-versus-beta basin preference, which is partly electrostatic rather
than steric; if conformations come out physically legal but implausible, that
is the gap to revisit, and it is a reason to add Ramachandran *then* rather
than now.

**`clash` landed on 7 Nov**, and finding it missing was the audit's real
result.
`steric.py` has an all-atom `clash_loss` used by the repacking objective, and
`growth_objective` references it in three comments and calls it nowhere. So
nothing currently stops a regenerated loop passing straight through the
protein. Growth runs at backbone-and-CB resolution, so the all-atom machinery
is the wrong tool -- it needs a TorsionTopology where growth carries a
FeatureTopology, and keeping two topologies in sync is a documented
silent-failure spot in this codebase (`alive_topology`). So `backbone_clash_loss` over N, CA, C, O, CB was written instead, at
tolerance 0.5 A for a reason that turned out to be physics rather than tuning
([[a-crystal-structures-clash-floor-is-its-hydrogen-bonds]]). Exemptions are at
residue level -- no residue against itself, and the whole 5x5 block across a
peptide bond -- and removing the second gives a constant 43x the real floor,
which is how this kind of term usually dies.

## Why loose FAPE rather than no FAPE

`--fape-weight` is now a flag; it was a hardcoded 1.0 in every run before
today and the comment beside it said "nothing should ever set it to 0".

It should not be zero, for a reason that is about the objective rather than the
measurement. **Regeneration is meant to find a conformation that works, not to
reproduce a crystal.** The application is regrowing a designed binder's
interface into something that binds *better* than the original
([[regrowing-a-designed-interface-can-produce-a-better-binder]]) -- which would
score badly against the native and be exactly the result wanted. So a term
pulling hard onto the crystal is working against the goal, while a loose one
keeps the output in the space of real protein geometry.

**The honest risk, recorded before the arm runs.** With FAPE loose and nothing
else supervising placement, the rule may close the loop into a conformation
that is physically valid and wrong, driving every restraint to zero while
`rmsd_dmg` sits at its 10.77 A baseline. That outcome is indistinguishable
*by loss* from success, and it is also indistinguishable from the "something
better than the original" the project is after. So `rmsd_dmg` has to stay
reported and must not become the pass/fail criterion -- and a conformation that
is valid, closed, clash-free and 10 A from the crystal needs a different
measurement to judge, which this map does not yet have.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]

## Supported by
- [[the-reach-probes-split-the-three-frozen-arms]]
