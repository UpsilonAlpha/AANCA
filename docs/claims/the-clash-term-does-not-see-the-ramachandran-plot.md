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

## One line of `steric.py` was the first suspect

    bonded = torch.nonzero(links[: n_res - 1], as_tuple=False).flatten()
    near[bonded, bonded + 1] = False

The **whole i/i+1 residue pair** is dropped -- all 25 atom pairs over N, CA, C,
O, CB -- wherever `links` says two residues are bonded. The docstring gives the
reason, and it is sound for *part* of the block: pairs one to three bonds across
the peptide bond have distances that psi and omega fix, and
`peptide_bond_restraint` already holds them.

But **15** of those 25 pairs are four or more bonds apart, which is
`CLASH_MIN_SEPARATION`, the project's own threshold for what counts as a steric
observation: O(i)-CB(i+1), CA(i)-CB(i+1), CB(i)-CA(i+1), CB(i)-CB(i+1),
N(i)-CA(i+1), O(i)-O(i+1), CA(i)-C(i+1), O(i)-C(i+1) and eight more --
enumerated in `GROWTH_DEPTH_TO_C` / `GROWTH_DEPTH_TO_N`, which count the
separation through the bond as `to_c[a] + 1 + to_n[b]` rather than by hand; an
earlier version of this node said 11, counted by hand, and was wrong. Those
are
precisely the contacts whose distances **phi and psi control**, and they are the
ones that carve the Ramachandran plot's forbidden regions. Dropping the block
"in full" throws them out with the exempt ones.

So the term cannot be the shadow of Ramachandran: the pairs that cast the
shadow are not scored. **That turned out to be true and not sufficient** --
scoring them changes nothing measurable, for the reason two sections down.

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

## Narrowing the exemption does not fix it either, and that is the real result

`backbone_clash_loss(..., bonded_pairs=True)` keeps the bonded residue pair and
exempts only the atom pairs under `CLASH_MIN_SEPARATION`, wired as the config
key `clash_bonded`, default `false` so g1-g22 reproduce. The 15 pairs are now in
the sum -- `tests/test_backbone_clash.py` proves it by collapsing a bonded pair,
where the switch charges and the default stays silent.

**It changes nothing measurable.** Rescanned with `--bonded`: still flat, 0.3 to
12.7 milli-A over the whole plane, ratio 0.79x, `(0, 180)` at 1.1 against an
ordinary `(-180, 120)` at 11.5. And on crystal ubiquitin a bonded pair reads
**exactly 0.0** under the switch at tolerance zero, not merely small.

That last number is the explanation. The pairs phi and psi control are four or
more bonds apart *because* they are far apart, and at Bondi heavy-atom radii
they never reach contact -- 2 x 1.52 = 3.04 A for two oxygens against a real
O(i)-O(i+1) distance well above it. The Ramachandran plot's exclusions are
carried largely by **hydrogens**, which this representation does not have, and
by effective radii tighter than heavy-atom vdW. So:

**A heavy-atom clash term cannot produce the Ramachandran plot at this
resolution, with the exemption or without it.** The 7 Nov argument was wrong in
a deeper way than the exemption: it is not that one line dropped the shadow, it
is that the representation does not cast it.

Two caveats on the scan, kept because they bound what it proves. A
**one-residue** window is placed by a walk from its N-ward neighbour and
[[the-damaged-region-must-start-within-reach-of-where-it-belongs]] records that
the C-ward join is deliberately left open, so the scanned residue has a dangling
bond on one side and some room that a closed backbone would not give it. A
closed-tripeptide sweep -- the experiment Ramachandran actually ran -- would
measure this without that slack. It is worth doing only to put a number on how
much of the flatness is the open join and how much is the missing hydrogens; it
will not change the conclusion, because the crystal-geometry reading of exactly
0.0 has no open join in it at all.

## So the phi/psi preference has to be learned, which is where this lands

Ulban, 14 Nov, on this and on electrostatics: *"electrostatic interactions
should be learned or through morphogen gradients"*, and *"we want a simple AANCA
model that can hopefully learn real physics"*. That settles the direction, and
the measurements above say it is the only direction available anyway.

  * **No tabulated (phi, psi) potential.** It would supply the
    alpha-versus-beta preference that `g22_random`'s second bucket exists to
    detect being learned
    ([[the-start-state-hands-the-rule-its-secondary-structure]]), so adding it
    would make the project's own result unmeasurable.
  * **No electrostatic term.** Same reason, and the preference is partly
    electrostatic.
  * **`clash_bonded` stays available and off.** It is a correction to a term
    that was provably blind on 15 pairs, worth having in the tree, and it buys
    no measured constraint -- so switching it on in an arm needs a reason this
    node does not supply.
  * **The route for electrostatics is the perception graph, not a new
    channel.** Backbone hydrogen bonding is through-space and short: i to i+4
    CA in a helix is about 6 A, inside the 10 A cutoff, so the partner is
    already an edge once the fold is roughly right and the rule can learn the
    interaction from one hop with nothing added. Morphogens transport along the
    **chain** rather than through space, by design -- that is the deficit they
    were built for -- so the emitted channels are a bootstrap for getting the
    fold roughly right, not a carrier for Coulomb. `morphogen = false` in every
    arm to date, and keeping it off is consistent with a simple rule.

So nothing in the objective forbids an impossible backbone, deliberately. The
question that replaces it is whether the rule produces one anyway, and that is a
measurement on output torsions, not a term.

## Supports
- [[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]
