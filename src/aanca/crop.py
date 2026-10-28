"""Spatial cropping, and the honest accounting of what a crop costs.

A size bound on the training pool throws whole structures away, and it throws
away the *largest* ones -- the multi-chain assemblies, which are precisely the
interface-rich entries worth having. Cropping keeps them: take a ball of
residues around a seed and train on that, leaving the rest of the assembly on
disk for another sample.

A strictly local rule tolerates this unusually well, which is the whole reason
it is available to us. The rule's perception reaches one graph hop, bounded by
``cutoff``, so a residue in the interior of a crop sees exactly what it would
have seen in the intact structure. Nothing about its input is approximated.

**But a residue near the crop boundary is a different matter, and this is the
part that has to be got right.** Its neighbours were deleted, so its perception
is wrong -- not noisy, wrong -- and the native torsion it is asked to predict
depends on contacts it cannot see. Supervising it teaches the rule to guess
packing from absent evidence, which is unlearnable and actively harmful: the
gradient is real, the signal is not. So every crop carries a mask saying which
residues have **complete perception**, and only those are supervised. The
boundary shell stays in the structure as context for the interior and is scored
on nothing.

That mask is computed exactly rather than estimated by a radius. A residue is
supervised iff every residue within ``cutoff`` of it *in the intact structure*
is present in the crop, measured by minimum heavy-atom distance -- the same
quantity the featuriser uses to build edges. There is no approximation and no
tuned margin.

The cost is real and worth stating: the unsupervised shell is about
``cutoff``-deep, so a crop spends a sizeable fraction of its residues on
context. :func:`crop_report` measures that fraction on real structures instead
of leaving it to be assumed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from . import graph as np_graph
from .structure import Protein

#: Fallback when a residue has neither CB nor CA: its own heavy-atom centroid.
#: Only reachable for badly truncated residues, but a seed distance has to be
#: defined for every residue or the argsort is not a total order.
_CENTRE_ATOMS = ("CB", "CA")


def residue_centres(protein: Protein) -> np.ndarray:
    """``(n_res, 3)`` one point per residue: CB, else CA, else centroid.

    CB-first matches the burial counting in ``metrics.py``, so "near the seed"
    and "buried" are measured from the same point.
    """
    out = np.zeros((len(protein), 3))
    for i, res in enumerate(protein.residues):
        index = res.index
        for name in _CENTRE_ATOMS:
            if name in index:
                out[i] = res.coords[index[name]]
                break
        else:
            out[i] = res.coords.mean(axis=0)
    return out


def contact_sets(protein: Protein, cutoff: float = 10.0) -> list[set[int]]:
    """Residues within ``cutoff`` of each residue, by minimum heavy-atom distance.

    Includes bonded sequence neighbours regardless of distance, because the
    featuriser forces those edges: a residue whose bonded neighbour was cropped
    away has incomplete perception even if nothing else is missing.

    Built on the same helper the featuriser uses, so "what the rule can see" is
    one definition rather than two that can drift.
    """
    coords = np.concatenate([r.coords for r in protein.residues], axis=0)
    owner = np.repeat(
        np.arange(len(protein)), [len(r.atom_names) for r in protein.residues]
    )
    pairs, _dists = np_graph.residue_distances_from_atoms(
        coords, owner, len(protein), cutoff
    )
    sets: list[set[int]] = [set() for _ in range(len(protein))]
    for i, j in pairs:
        sets[int(i)].add(int(j))
        sets[int(j)].add(int(i))
    for i, forced in enumerate(
        np_graph.sequence_neighbour_sets(protein.chain_ids, protein.res_ids)
    ):
        sets[i] |= forced
    return sets


def crop_indices(
    protein: Protein, size: int, seed_residue: int, centres: np.ndarray | None = None
) -> list[int]:
    """The ``size`` residues nearest ``seed_residue``, returned ascending.

    A ball rather than a sequence window: packing is a spatial problem, and a
    window along the chain would cut away the contacts that matter while keeping
    distant ones that do not. It also crosses chains freely, which is the point
    -- an interface is only in the crop if both sides of it are.
    """
    if size < 1:
        raise ValueError(f"crop size must be at least 1, got {size}")
    if not 0 <= seed_residue < len(protein):
        raise IndexError(
            f"seed residue {seed_residue} out of range for {len(protein)} residues"
        )
    if centres is None:
        centres = residue_centres(protein)
    distance = np.linalg.norm(centres - centres[seed_residue], axis=-1)
    # Stable sort so an exact distance tie resolves by residue index rather
    # than by whatever the sort happens to do, which keeps a crop reproducible.
    nearest = np.argsort(distance, kind="stable")[:size]
    return sorted(int(i) for i in nearest)


@dataclass(frozen=True)
class Crop:
    """A cropped structure plus what may be supervised on it."""

    protein: Protein
    #: Indices into the *original* protein, ascending. Keeps a crop traceable
    #: back to the entry it came from.
    indices: list[int]
    #: ``(n_cropped,)`` bool: residues whose perception is complete, and so the
    #: only ones the loss may touch.
    supervised: np.ndarray
    seed_residue: int

    @property
    def supervised_fraction(self) -> float:
        return float(self.supervised.mean()) if len(self.supervised) else 0.0


def crop(
    protein: Protein,
    size: int,
    cutoff: float = 10.0,
    seed_residue: int | None = None,
    rng: random.Random | None = None,
) -> Crop:
    """Crop to ``size`` residues around a seed, masking the boundary shell.

    ``size >= len(protein)`` returns the structure untouched with everything
    supervised. That is not a special case for convenience: when nothing was
    removed, a truncated neighbourhood is the real molecular surface rather than
    an artefact of ours, and masking it would throw away the surface residues
    the benchmark specifically reports on.
    """
    if seed_residue is None:
        rng = rng or random
        seed_residue = rng.randrange(len(protein))

    if size >= len(protein):
        return Crop(
            protein=protein,
            indices=list(range(len(protein))),
            supervised=np.ones(len(protein), dtype=bool),
            seed_residue=seed_residue,
        )

    contacts = contact_sets(protein, cutoff)
    kept = crop_indices(protein, size, seed_residue)
    kept_set = set(kept)
    supervised = np.fromiter(
        (contacts[i] <= kept_set for i in kept), dtype=bool, count=len(kept)
    )
    return Crop(
        protein=protein.subset(kept),
        indices=kept,
        supervised=supervised,
        seed_residue=seed_residue,
    )


def crop_report(
    protein: Protein,
    sizes: tuple[int, ...] = (256, 384, 512, 768),
    cutoff: float = 10.0,
    seeds: int = 4,
    rng: random.Random | None = None,
) -> dict[int, float]:
    """Mean supervised fraction per crop size. The measurement, not an estimate.

    The boundary shell is roughly ``cutoff``-deep whatever the crop size, so the
    supervised fraction rises with size and the useful question -- what fraction
    of a crop's compute is spent on context -- has a number rather than an
    intuition. Averaged over random seeds because a crop centred in a domain
    core and one centred at a loose terminus are not alike.
    """
    rng = rng or random.Random(0)
    contacts = contact_sets(protein, cutoff)
    centres = residue_centres(protein)
    out: dict[int, float] = {}
    for size in sizes:
        fractions = []
        for _ in range(seeds):
            seed_residue = rng.randrange(len(protein))
            if size >= len(protein):
                fractions.append(1.0)
                continue
            kept = crop_indices(protein, size, seed_residue, centres)
            kept_set = set(kept)
            fractions.append(
                sum(contacts[i] <= kept_set for i in kept) / len(kept)
            )
        out[size] = sum(fractions) / len(fractions)
    return out
