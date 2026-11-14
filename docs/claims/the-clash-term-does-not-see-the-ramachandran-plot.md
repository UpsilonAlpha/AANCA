---
type: observation
status: supported
evidence: ours
sources: [steric.py, ramachandran_clash_probe.py]
---

# The clash term exempts exactly the atom pairs that make the Ramachandran plot, so nothing in the objective forbids an impossible backbone

Ulban, 14 Nov: *"shouldn't we have ramachandran priors?"* Measured the same day,
and the answer is yes, because the argument that said otherwise is wrong.

## The argument this overturns

[[the-objective-should-restrain-the-output-not-teach-the-placement]] rejected a
Ramachandran restraint on 7 Nov as redundant:

> The Ramachandran plot is the empirical shadow of backbone sterics -- the
> disallowed regions are disallowed because adjacent atoms collide -- so a
> proper clash term subsumes its hard constraint.

The physics in that sentence is right. The claim that `backbone_clash_loss`
delivers it is not.

## One line of `steric.py` removes the shadow

    bonded = torch.nonzero(links[: n_res - 1], as_tuple=False).flatten()
    near[bonded, bonded + 1] = False

The **whole i/i+1 residue pair** is dropped -- all 25 atom pairs over N, CA, C,
O, CB -- wherever `links` says two residues are bonded. The docstring gives the
reason, and it is sound for *part* of the block: pairs one to three bonds across
the peptide bond have distances that psi and omega fix, and
`peptide_bond_restraint` already holds them.

But 11 of those 25 pairs are **four or more bonds apart**, which is
`CLASH_MIN_SEPARATION`, the project's own threshold for what counts as a steric
observation: O(i)-CB(i+1), CA(i)-CB(i+1), CB(i)-CA(i+1), CB(i)-CB(i+1),
N(i)-CA(i+1), O(i)-O(i+1), CA(i)-C(i+1), O(i)-C(i+1) and the rest. Those are
precisely the contacts whose distances **phi and psi control**, and they are the
ones that carve the Ramachandran plot's forbidden regions. Dropping the block
"in full" throws them out with the exempt ones.

So the term cannot be the shadow of Ramachandran: the pairs that cast the
shadow are not scored.

## Measured, and it is flat

`scripts/ramachandran_clash_probe.py --step 30 --structures 4 --damage 1
--scope all`. A **single** residue is damaged and rebuilt at a grid of fixed
(phi, psi) off its native neighbour, every other residue at its crystal pose,
and `backbone_clash_loss` read over the crop. That isolates one residue's own
conformation, which is what the Ramachandran plot is about.

Milli-angstroms of mean overlap per scored atom, psi across, phi down:

            -180  -150  -120   -90   -60   -30     0    30    60    90   120   150
     -180   12.1   8.0   4.9   2.9   2.5   2.4   3.4   3.4   5.8  11.3  16.6  15.4
     -120    9.5   6.2   3.9   2.6   2.4   2.6   3.0   5.5  10.7  15.3  15.5  13.4
      -60    6.8   4.5   2.8   2.4   2.4   2.6   4.5   6.6   8.8  10.4  11.2   9.1
        0    5.0   3.9   3.6   3.4   2.9   2.9   3.1   4.3   6.8   8.7   8.4   7.1
       60   10.3   9.4   6.5   3.9   2.7   2.4   3.4   4.9   6.2   6.4   6.5   8.4
      120   13.0   9.2   5.3   3.3   2.5   2.6   2.9   3.5   3.9   5.7  10.8  14.3

The crystal floor on the same crops is 2.59 milli-A. **The entire map lies
between 2.4 and 16.7**, so the worst conformation on the plane reads six
thousandths of an angstrom more overlap than a crystal structure. There is no
Ramachandran structure in it: `(phi, psi) = (0, 180)`, the most forbidden
conformation in protein stereochemistry, reads **3.4** -- *lower* than the
perfectly ordinary `(-180, 120)` at 16.6. The weak smooth bowl that is there,
lowest around psi -60 to -30 for every phi, is a reach artefact of the walk and
not a steric fact.

Favoured boxes mean 8.11 milli-A against 5.93 everywhere else, a ratio of
**0.73x** -- the allowed regions score *worse*, which is what no signal looks
like.

## The trap that nearly produced the opposite answer

A first pass set **all 12** residues of a damaged window to the same (phi, psi)
and scored the window alone. That gave a 42x separation and looked like a clean
confirmation. It is an artefact: a constant (phi, psi) is a regular helix of
some pitch, and the large values trace a ridge where that helix has about two
residues per turn and the chain folds back through itself. That is global
self-overlap at i/i+2 and beyond, not the local exclusion in question -- and at
a finer 30 degree grid the same scan inverts to 0.37x, because the ridge cuts
through the favoured boxes too.

**A poly-(phi, psi) scan cannot measure a per-residue constraint.** The probe
keeps `--damage` and `--scope` as flags so both readings are reproducible, and
`--damage 1 --scope all` is the one that means what it says.

## What follows, and what it is not

The honest fix is **narrower and more physical than a Ramachandran table**, and
it is the one the 7 Nov node already claimed was in place: exempt only the
i/i+1 atom pairs that are within `CLASH_MIN_SEPARATION` bonds and score the
remaining 11. Then the Ramachandran plot emerges from sterics, which is the
right shape for this project -- a tabulated (phi, psi) potential is an
engineered feature, and the preference here is to let the rule meet the physics
([[the-objective-should-restrain-the-output-not-teach-the-placement]]).

Two things this does **not** establish:

  * **That any arm has produced an impossible backbone.** Nothing forbids one,
    which is a different statement. The `twist` diagnostic is a mean over
    bonded-pair relative rotations and would not show it, and no arm's output
    torsions have ever been read against the plot. That measurement needs a
    checkpoint and Bunya is offline.
  * **That the alpha-versus-beta preference comes free.** It does not; it is
    partly electrostatic, the 7 Nov node says so, and no steric term supplies
    it. That part should be **learned**, and whether it is learned is bucket 2
    of `g22_random`
    ([[the-start-state-hands-the-rule-its-secondary-structure]]).

Changing the exemption set alters the objective of every future arm, so it is
pre-registered nowhere yet and is Ulban's call.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
