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
    """RCSB's entity-level 40% sequence clusters, read two different ways.

    The published file lists one cluster per line as ``PDBID_ENTITYID`` tokens.
    An entry has several entities, so there are two sensible notions of "related
    entries", and they are used for different jobs:

    :meth:`group_of` -- the cluster of the entry's **primary** (lowest-numbered)
    protein entity. Used to **split train from validation**. One cluster per
    entry, no transitivity.

    :meth:`clusters_of` -- **every** cluster the entry touches. Used to
    **exclude test homologs**, where over-caution is correct.

    An earlier version treated two entries as related whenever they shared *any*
    cluster, resolved by union-find. That was measured and it collapses: on 7,601
    X-ray entries at 1.2 A it produced a single group of **6,682 entries**, 88%
    of the pool, because a promiscuous entity -- a tag, a chaperone, a common
    lysozyme -- chains otherwise unrelated entries together, and the merging
    cascades globally. Grouping by the primary entity removes the cascade.

    What makes that safe is that the two jobs have very different stakes. Test
    leakage is handled separately and rigorously, by an MMseqs2 sequence-identity
    search against the actual test targets (see
    ``scripts/build_training_manifest.py``), which catches a homologous chain
    whatever entity number it carries. Cluster grouping only has to keep
    train and validation apart, and for that the primary entity is enough.
    """

    def __init__(self, lines: list[str]) -> None:
        #: ``PDBID`` -> the (entity number, cluster id) pairs it contains.
        #: Keyed by entry so lookups are O(1); scanning the entity table per
        #: query is 250k operations and far too slow at pool scale.
        self._entry_entities: dict[str, list[tuple[int, str]]] = {}

        for line in lines:
            tokens = [t.strip().upper() for t in line.split() if t.strip()]
            if not tokens:
                continue
            cluster = tokens[0]
            for token in tokens:
                entry, _, suffix = token.partition("_")
                number = int(suffix) if suffix.isdigit() else 10**9
                self._entry_entities.setdefault(entry, []).append((number, cluster))

        for entities in self._entry_entities.values():
            entities.sort()

    @classmethod
    def from_file(cls, path: str | Path) -> "ClusterIndex":
        with open(path, encoding="utf-8") as handle:
            return cls(handle.readlines())

    def __len__(self) -> int:
        return len(self._entry_entities)

    def group_of(self, pdb_id: str) -> str:
        """Cluster of the entry's primary entity; the entry id if unlisted.

        "Primary" is the lowest entity number present, which for a
        protein-of-interest plus accessories is the one that matters. An
        unlisted entry becomes its own group, which cannot pull anything across
        a split.
        """
        key = pdb_id.upper()
        entities = self._entry_entities.get(key)
        return entities[0][1] if entities else key

    def clusters_of(self, pdb_id: str) -> set[str]:
        """Every cluster this entry touches. For conservative exclusion."""
        return {cluster for _number, cluster in self._entry_entities.get(pdb_id.upper(), ())}

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

    # Exclusion uses EVERY cluster a holdout entry touches, not just its primary
    # one: a test target's homolog may sit at any entity number, and
    # over-caution is the right error here. Splitting train from validation uses
    # the primary cluster only (see ClusterIndex), because that job has far
    # lower stakes and the conservative version collapses the pool.
    banned: set[str] = set()
    for pdb_id in holdout_ids:
        banned |= clusters.clusters_of(pdb_id) or {pdb_id.upper()}

    by_group: dict[str, list[StructureSpec]] = {}
    excluded: list[StructureSpec] = []
    for spec in specs:
        if clusters.clusters_of(spec.pdb_id) & banned or spec.group in banned:
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
