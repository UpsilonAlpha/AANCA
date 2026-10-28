"""Dataset assembly: which structures, which split, and what damage.

Three decisions worth knowing about.

**This reconstructs BC40 rather than defining a new split.** The published
protocol (AttnPacker, followed verbatim by DiffPack) is BC40 -- PDB clustered at
40% sequence identity -- with 37,266 training and 1,500 validation chains. That
split list is *not* published: AttnPacker's repository releases pre-trained
models and result PDBs, not the training data. So the pool has to be rebuilt,
and this module rebuilds it from **RCSB's own** ``clusters-by-entity-40.txt``,
which is the resource BC40 derives from in the first place rather than a
substitute. RCSB has changed clustering tool over the years, so the clusters
will not be bit-identical to the BC40 those papers used; that divergence is real
and belongs in the write-up rather than being papered over.

Grouping here is **entry-level**, where BC40 is chain/entity-level, because the
loader works on whole entries. Union-find over entries sharing any cluster
deliberately over-merges, which costs training data and never leaks.

**Holdout exclusion is by cluster, not by entry.** Dropping the CASP targets
themselves leaks: their close homologs stay in training and the test is no
longer unseen. Any cluster containing a holdout entry is removed whole.

**A missing holdout list is an error, not an empty set.** Silently training on
everything because a file was absent is exactly the failure that produces a
good-looking number nobody can defend, so :func:`split_by_cluster` refuses
rather than defaults.

Batching note: each structure has its own topology, so a batch would need
padding across structures. v1 does not bother -- at the measured ~124 MB of
retained activations per step for a 300-residue target, single structures with
gradient accumulation is what fits anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from . import residue_constants as rc
from .structure import load_protein
from .torch_geometry import apply_chi_deltas
from .torch_graph import FeatureTopology, feature_topology

#: RCSB's precomputed clusters. The 40% file is the one the packing literature's
#: splits are usually built from.
CLUSTER_URL = (
    "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{identity}.txt"
)


@dataclass(frozen=True)
class StructureSpec:
    """One training structure: where it is and what it is related to."""

    pdb_id: str
    path: Path
    #: Representative id of the related-entry group this belongs to. Entries
    #: sharing any sequence cluster share a group, so a split by group leaves no
    #: sequence-similar pair straddling the boundary.
    group: str

    def as_json(self) -> dict:
        return {"pdb_id": self.pdb_id, "path": str(self.path), "group": self.group}


class ClusterIndex:
    """Entry-level relatedness, derived from RCSB's entity-level clusters.

    The published file lists one cluster per line as ``PDBID_ENTITYID`` tokens.
    Two *entries* are treated as related when they share any cluster, which is
    resolved by union-find. That deliberately over-merges: a promiscuous entity
    can chain unrelated entries into one group, which costs training data but
    never leaks. Over-merging is the safe direction, and
    :meth:`group_size_histogram` makes the cost visible rather than hidden.
    """

    def __init__(self, lines: list[str]) -> None:
        self._parent: dict[str, str] = {}
        for line in lines:
            entries = {token.split("_")[0].upper() for token in line.split() if token}
            if len(entries) < 2:
                for entry in entries:
                    self._parent.setdefault(entry, entry)
                continue
            iterator = iter(sorted(entries))
            first = next(iterator)
            self._parent.setdefault(first, first)
            for other in iterator:
                self._parent.setdefault(other, other)
                self._union(first, other)

    @classmethod
    def from_file(cls, path: str | Path) -> "ClusterIndex":
        with open(path, encoding="utf-8") as handle:
            return cls(handle.readlines())

    def _find(self, entry: str) -> str:
        root = entry
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[entry] != root:
            self._parent[entry], entry = root, self._parent[entry]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            # Keep the lexicographically smaller root, so group ids are stable
            # across runs rather than depending on insertion order.
            if rb < ra:
                ra, rb = rb, ra
            self._parent[rb] = ra

    def __len__(self) -> int:
        return len(self._parent)

    def group_of(self, pdb_id: str) -> str:
        """Group id, or the entry itself when RCSB lists no cluster for it.

        A singleton is the conservative reading: an unclustered entry is assumed
        related to nothing, which risks nothing because it cannot pull a
        homolog across the split.
        """
        key = pdb_id.upper()
        return self._find(key) if key in self._parent else key

    def group_size_histogram(self, pdb_ids: list[str]) -> dict[int, int]:
        counts: dict[str, int] = {}
        for pdb_id in pdb_ids:
            group = self.group_of(pdb_id)
            counts[group] = counts.get(group, 0) + 1
        histogram: dict[int, int] = {}
        for size in counts.values():
            histogram[size] = histogram.get(size, 0) + 1
        return dict(sorted(histogram.items()))


def split_by_cluster(
    specs: list[StructureSpec],
    holdout_ids: set[str],
    clusters: ClusterIndex,
    validation_fraction: float = 0.05,
    seed: int = 0,
    require_holdout: bool = True,
) -> dict[str, list[StructureSpec]]:
    """Split into ``train`` / ``validation`` / ``excluded``, never splitting a group.

    ``holdout_ids`` are the test entries (CASP targets). Every group containing
    one is moved to ``excluded`` in full, so no homolog of a test target reaches
    training. The test set itself is loaded separately -- this function's job is
    to make the *training* pool clean.
    """
    if require_holdout and not holdout_ids:
        raise ValueError(
            "holdout_ids is empty. Training without a holdout silently includes "
            "the test targets and their homologs; pass require_holdout=False "
            "only if that is genuinely intended."
        )
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction}")

    banned = {clusters.group_of(pdb_id) for pdb_id in holdout_ids}

    by_group: dict[str, list[StructureSpec]] = {}
    excluded: list[StructureSpec] = []
    for spec in specs:
        if spec.group in banned:
            excluded.append(spec)
        else:
            by_group.setdefault(spec.group, []).append(spec)

    # Shuffle groups, not structures: assigning structures independently would
    # put sequence-similar entries on both sides of the boundary.
    generator = torch.Generator().manual_seed(seed)
    groups = sorted(by_group)
    order = torch.randperm(len(groups), generator=generator).tolist()

    n_validation = int(round(len(groups) * validation_fraction))
    validation_groups = {groups[i] for i in order[:n_validation]}

    train: list[StructureSpec] = []
    validation: list[StructureSpec] = []
    for group in groups:
        target = validation if group in validation_groups else train
        target.extend(by_group[group])

    return {"train": train, "validation": validation, "excluded": excluded}


def write_manifest(path: str | Path, splits: dict[str, list[StructureSpec]]) -> None:
    payload = {name: [s.as_json() for s in specs] for name, specs in splits.items()}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(path: str | Path) -> dict[str, list[StructureSpec]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: [
            StructureSpec(pdb_id=e["pdb_id"], path=Path(e["path"]), group=e["group"])
            for e in entries
        ]
        for name, entries in payload.items()
    }


# -- damage, sampled in torch ----------------------------------------------


def von_mises_deltas(
    chi_mask: Tensor, concentration: float, generator: torch.Generator | None = None
) -> Tensor:
    """Circular noise on every real chi. Large concentration is mild damage.

    Sampling only -- no geometry. Apply with
    :func:`torch_geometry.apply_chi_deltas`, which keeps the perturbation an
    exactly invertible rigid rotation.
    """
    if concentration <= 0:
        return uniform_deltas(chi_mask, generator)
    distribution = torch.distributions.VonMises(
        loc=torch.zeros(chi_mask.shape, dtype=torch.float64),
        concentration=torch.full(chi_mask.shape, float(concentration), dtype=torch.float64),
    )
    noise = distribution.sample()
    return torch.where(chi_mask, noise, torch.zeros_like(noise))


def uniform_deltas(
    chi_mask: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Maximal damage: the chis carry no information about the native state."""
    draws = torch.rand(chi_mask.shape, dtype=torch.float64, generator=generator)
    noise = draws * 2 * torch.pi - torch.pi
    return torch.where(chi_mask, noise, torch.zeros_like(noise))


@dataclass
class Sample:
    pdb_id: str
    topology: FeatureTopology
    native: Tensor
    damaged: Tensor
    #: The deltas that produced ``damaged``. Negating them restores ``native``
    #: exactly, which makes the damage auditable rather than merely applied.
    deltas: Tensor
    concentration: float


class RepackingDataset:
    """Loads structures and yields damaged/native pairs.

    Topologies are cached because building one parses the structure and resolves
    every index table, which is far more expensive than sampling the damage.
    """

    def __init__(
        self,
        specs: list[StructureSpec],
        concentrations: tuple[float, ...] = (100.0, 20.0, 5.0, 0.0),
        cache_topologies: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        if not specs:
            raise ValueError("no structures given")
        if not concentrations:
            raise ValueError("no damage levels given")
        self.specs = specs
        self.concentrations = concentrations
        self.device = device
        self._cache: dict[str, tuple[FeatureTopology, Tensor]] = {}
        self._cache_topologies = cache_topologies

    def __len__(self) -> int:
        return len(self.specs)

    def load(self, index: int) -> tuple[FeatureTopology, Tensor]:
        spec = self.specs[index]
        if spec.pdb_id in self._cache:
            return self._cache[spec.pdb_id]
        topo, coords = feature_topology(load_protein(spec.path), device=self.device)
        if self._cache_topologies:
            self._cache[spec.pdb_id] = (topo, coords)
        return topo, coords

    def sample(
        self,
        index: int,
        concentration: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Sample:
        """Damage structure ``index``. ``concentration=None`` picks a level at random.

        The curriculum is a *mixture* rather than a schedule: the rule must
        recover from mild and severe damage alike, and it is never told which it
        got. Telling it would be noise-level conditioning, which is precisely
        what DiffPack has and this deliberately does not.
        """
        topo, native = self.load(index)
        if concentration is None:
            pick = torch.randint(
                len(self.concentrations), (1,), generator=generator
            ).item()
            concentration = self.concentrations[pick]

        mask = topo.torsion.chi_mask
        deltas = (
            uniform_deltas(mask, generator)
            if concentration <= 0
            else von_mises_deltas(mask, concentration, generator)
        )
        damaged = apply_chi_deltas(native, deltas, topo.torsion)
        return Sample(
            pdb_id=self.specs[index].pdb_id,
            topology=topo,
            native=native,
            damaged=damaged,
            deltas=deltas,
            concentration=float(concentration),
        )

    def iter_epoch(
        self, generator: torch.Generator | None = None, shuffle: bool = True
    ):
        order = (
            torch.randperm(len(self), generator=generator).tolist()
            if shuffle
            else range(len(self))
        )
        for index in order:
            yield self.sample(index, generator=generator)
