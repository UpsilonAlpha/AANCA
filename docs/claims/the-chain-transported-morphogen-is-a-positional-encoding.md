---
type: observation
status: supported
evidence: ours
sources: [morphogen.py, growth.py]
---

# With nothing dormant the chain-transported field is exactly `decay ** index`, which is the positional encoding the module was written to avoid

Ulban, 14 Nov: *"they shouldn't transport along the chain -- the residue graph
is spatial."* Right, and the module's own argument is what makes it decisive.

## The defence was conditional, and the condition no longer holds

`_live_bonds` in `src/aanca/morphogen.py` states the test:

> A dormant slot does not conduct. That is the choice that makes the field a
> property of the structure grown so far rather than of the crop's sequence: if
> dormant slots conducted, the converged field would be a deterministic function
> of the residue index and the whole thing would be a positional encoding handed
> to the rule rather than a quantity the growing chain carries.

So the whole defence rests on `alive` varying. It does under `birth="cartesian"`,
where residues are born over the rollout. It does **not** under `birth="none"`,
which is what `configs/regeneration.toml` sets and what every arm since g9 has
run: *"Every slot alive, and the rollout starts from a baseline conformation"*
(`growth.py:1788`). With nothing dormant, `_live_bonds(alive, links) == links`
and every bond conducts from step 0.

## Measured, and it is exact

400 hops on a 12-residue single run, every slot alive, no emission:

    res  0  1.0000   1.0000   1.0000   1.0000
    res  1  0.9900   0.9500   0.8500   0.6000
    res  2  0.9801   0.9025   0.7225   0.3600
    res  5  0.9510   0.7738   0.4437   0.0778
    res 11  0.8953   0.5688   0.1673   0.0036

    max |field - decay ** index| = 1.2e-07

Floating point. The transported field **is** `decay ** i`, so by the module's
own criterion it is a positional encoding, in the one setting the project now
cares about. And `field_features` feeds its **logarithm**, which is
`-i * |log decay| / scale` -- a linear sequence coordinate, supplied at four
resolutions and mirrored from the other terminus. That is a positional encoding
in all but name, and the project's standing position is that no cell may read a
global coordinate
([[the-objective-should-restrain-the-output-not-teach-the-placement]] for the
analogous argument about the step index).

"Length-free" is true and is not the same as "not a coordinate": distance from a
terminus is a coordinate, merely unnormalised.

## What spatial diffusion would be, and it is simpler

The salvageable mechanism is the **emitted** group -- the rule chooses what to
secrete, it spreads, neighbours read it -- because that is emergent rather than
specified. But it currently also hops along `links` (`from_nward`, `from_cward`
over `bonds`), so it is chain-local too.

Diffusing over the **perception graph** instead means:

  * averaging a secreted scalar over the edges the rule already computes, so
    there is no second topology and no new neighbour list;
  * dropping `NWARD_DECAYS` and `CWARD_DECAYS` entirely -- 8 of the 10
    channels, `FIELD_CHANNELS` 10 -> 2 and `FIELD_FEATURE_DIM` 20 -> 4;
  * one mechanism rather than three, nothing tabulated, and a field over the
    geometry **as it currently is**, which is what a cell could actually sense.

That is a removal, which is the right shape for a rule that is meant to stay
simple.

## The bootstrap objection, and why it does not save chain transport

The module exists because the spatial graph is *wrong* early in a rollout: two
fronts drift apart, never come within the 10 A cutoff, and so never acquire the
edge that would correct them. A spatially diffused field is wrong in exactly the
same way.

But that is the right kind of wrong. It is a field over the structure as it
actually is, and it becomes right as the structure does. The chain field is
invariant to the structure being wrong **because it carries no information about
the structure at all** -- which is the same sentence as the measurement above.
Trading structural information for robustness gives a channel that is robust and
empty.

## Nothing is locked, and this is not queued

`morphogen = false` in `configs/regeneration.toml` and in every arm ever run, so
there is no recorded result to reproduce and the module can be **replaced**
rather than deprecated. It is also not in the queue: morphogens are machinery,
the central claim is regeneration and is unsettled
([[a-local-rule-can-regenerate-a-damaged-region-from-its-surviving-context]]),
and the per-residue route for a through-space interaction is the perception
graph with nothing added -- i to i+4 CA in a helix is about 6 A, inside the
cutoff, so a backbone hydrogen bond is already an edge once the fold is roughly
right ([[the-clash-term-does-not-see-the-ramachandran-plot]]).

**Built on 14 Nov, after this node was written.** Ulban asked for it directly,
so the spatial version exists:
[[the-emitted-chemical-diffuses-over-the-perception-graph]] carries the
arithmetic -- 20 node features down to 4 -- and the end-to-end check. What this
node records is why the old one went.

## Supports
- [[a-local-per-residue-rule-can-build-a-protein-backbone]]
