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

**Size is bounded per sample, not per entry.** Retained activations are linear
in residue count, so something must bound a sample. Filtering large entries out
of the pool is the obvious answer and the wrong one: it discards the multi-chain
assemblies, which are the interface-rich structures. Instead a large entry
contributes a ball of residues around a random seed (``crop.py``), with the
boundary shell masked out of the loss because its perception is incomplete.

**The train/validation side of a group is a hash of its id, not a shuffle.**
Measured the hard way: with ``randperm`` over the sorted group list, every
group's side depends on how many groups exist, so correcting the leakage filter
by one group reassigned 945 of 988 validation entries and required the whole
validation set to be downloaded again. Hashing makes the split stable under any
change to the pool.

Batching note: each structure has its own topology, so a batch would need
padding across structures. v1 does not bother -- at the measured ~124 MB of
retained activations per step for a 300-residue target, single structures with
gradient accumulation is what fits anyway.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from . import residue_constants as rc
from .crop import crop
from .structure import Protein, load_protein
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

    # Assign each GROUP a side, never a structure: assigning structures
    # independently would put sequence-similar entries on both sides.
    #
    # The side is a hash of the group id, not a draw from a shuffle, and that
    # choice was paid for. The first version took ``randperm`` over the sorted
    # group list, which makes every group's side depend on how many groups there
    # are: when a rebuild moved one single group into ``excluded``, the
    # permutation shifted and **945 of 988 validation entries changed**, which
    # meant re-downloading the whole validation set to evaluate the same model.
    # Hashing the id makes membership depend only on ``(group, seed)``, so the
    # pool can grow, shrink, or have its leakage filter corrected without
    # disturbing a single unrelated group.
    #
    # The price is that the realised fraction is binomial around the target
    # rather than exact -- at 25,000 groups and 4% that is about +/- 31 groups,
    # which buys a great deal of stability for nothing that matters.
    groups = sorted(by_group)
    train: list[StructureSpec] = []
    validation: list[StructureSpec] = []
    for group in groups:
        side = validation if _is_validation(group, validation_fraction, seed) else train
        side.extend(by_group[group])

    return {"train": train, "validation": validation, "excluded": excluded}


def _is_validation(group: str, fraction: float, seed: int) -> bool:
    """Deterministic side for one group, independent of every other group."""
    if fraction <= 0.0:
        return False
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64 < fraction


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


#: Below this, Best-Fisher's constants degenerate (``b -> 0``, ``r -> inf``) and
#: the distribution is indistinguishable from uniform anyway.
_MIN_CONCENTRATION = 1e-6

#: Rejection sampling needs a bound. Best-Fisher accepts with probability above
#: 0.65 for every concentration, so reaching this many rounds means something is
#: wrong rather than merely unlucky.
_MAX_REJECTION_ROUNDS = 128


def von_mises_sample(
    shape: tuple[int, ...],
    concentration: float,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Von Mises draws about zero, honouring ``generator``.

    Best-Fisher (1979) rejection sampling, written out rather than delegated,
    because ``torch.distributions.VonMises.sample()`` **takes no generator** --
    its signature is ``(self, sample_shape)`` and it draws from the global RNG.

    The original version here passed a generator to it anyway. Python accepted
    the call, the argument went nowhere, and every von Mises damage draw in the
    project was unseeded: identical seeds gave different structures, while the
    uniform path beside it was correctly seeded and looked like proof that the
    whole thing was reproducible. Nothing failed, no test caught it, and it
    surfaced only when the held-out evaluation protocol demanded that damage be
    byte-identical across runs.
    """
    kappa = float(concentration)
    if kappa <= _MIN_CONCENTRATION:
        raise ValueError(f"concentration must be positive, got {concentration}")

    a = 1.0 + math.sqrt(1.0 + 4.0 * kappa * kappa)
    b = (a - math.sqrt(2.0 * a)) / (2.0 * kappa)
    r = (1.0 + b * b) / (2.0 * b)

    count = 1
    for dim in shape:
        count *= dim
    out = torch.zeros(count, dtype=dtype)
    todo = torch.arange(count)

    for _ in range(_MAX_REJECTION_ROUNDS):
        if todo.numel() == 0:
            break
        n = todo.numel()
        u1 = torch.rand(n, dtype=dtype, generator=generator)
        u2 = torch.rand(n, dtype=dtype, generator=generator)
        u3 = torch.rand(n, dtype=dtype, generator=generator)

        z = torch.cos(math.pi * u1)
        f = (1.0 + r * z) / (r + z)
        c = kappa * (r - f)

        # The second criterion involves log(c / u2), which is undefined for
        # c <= 0. Guard it rather than letting a NaN decide the comparison:
        # NaN >= 0 is False, so it would reject by accident rather than by
        # reasoning, and would hide a sign error if one ever appeared.
        positive = c > 0
        guarded = torch.where(positive, c, torch.ones_like(c))
        accept = (c * (2.0 - c) - u2 > 0) | (
            positive & (torch.log(guarded / u2) + 1.0 - c >= 0)
        )

        theta = torch.sign(u3 - 0.5) * torch.acos(f.clamp(-1.0, 1.0))
        out[todo[accept]] = theta[accept]
        todo = todo[~accept]
    else:
        raise RuntimeError(
            f"von Mises rejection sampling did not converge for kappa={kappa} "
            f"after {_MAX_REJECTION_ROUNDS} rounds, with {todo.numel()} draws "
            "outstanding -- the acceptance rate should never be this low"
        )

    return out.reshape(shape)


def von_mises_deltas(
    chi_mask: Tensor, concentration: float, generator: torch.Generator | None = None
) -> Tensor:
    """Circular noise on every real chi. Large concentration is mild damage.

    Sampling only -- no geometry. Apply with
    :func:`torch_geometry.apply_chi_deltas`, which keeps the perturbation an
    exactly invertible rigid rotation.
    """
    if concentration <= _MIN_CONCENTRATION:
        return uniform_deltas(chi_mask, generator)
    noise = von_mises_sample(tuple(chi_mask.shape), concentration, generator)
    noise = noise.to(chi_mask.device)
    return torch.where(chi_mask, noise, torch.zeros_like(noise))


def uniform_deltas(
    chi_mask: Tensor, generator: torch.Generator | None = None
) -> Tensor:
    """Maximal damage: the chis carry no information about the native state."""
    # Drawn on the CPU whatever the device, so one CPU ``generator`` keeps the
    # damage reproducible on a GPU run as well -- a device generator would make
    # the stream depend on where the job happened to land.
    draws = torch.rand(chi_mask.shape, dtype=torch.float64, generator=generator)
    noise = (draws * 2 * torch.pi - torch.pi).to(chi_mask.device)
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
    #: Residues in this sample, as indices into the full entry. ``None`` when
    #: the entry was taken whole.
    crop_indices: list[int] | None = None
    #: The residue the crop grew from, or ``None`` when nothing was cropped.
    #: A crop is a deterministic function of this, so it is the smallest thing
    #: that identifies *which problem instance* a sample is -- which matters
    #: because successive samples of one entry are deliberately different
    #: regions of it, and anything that stores a state for later reuse has to
    #: store this with it or it will pair coordinates with the wrong topology.
    crop_seed: int | None = None

    @property
    def supervised_fraction(self) -> float:
        """Fraction of residues the loss may touch. Below 1 only for a crop."""
        mask = self.topology.supervised_mask
        return float(mask.to(torch.float64).mean())


class RepackingDataset:
    """Loads structures, crops the large ones, and yields damaged/native pairs.

    **Size is handled by cropping, not by exclusion.** Retained activations are
    linear in residue count -- measured ~124 MB per step per 300 residues -- so
    something has to bound the sample, and the obvious move is a size filter on
    the pool. That filter throws away the biggest entries, which are the
    multi-chain assemblies: exactly the interface-rich structures worth keeping.
    So instead a large entry contributes a ball of ``crop_size`` residues around
    a randomly chosen seed, and ``crop.py`` masks the boundary shell out of the
    loss because its perception is incomplete. A different seed each epoch makes
    the crop augmentation rather than a one-off truncation.

    Topologies are cached for entries taken whole, since building one parses the
    structure and resolves every index table. A cropped entry cannot use that
    cache -- the crop differs per sample, which is the point -- so it pays the
    index-table cost each time, on ``crop_size`` residues rather than on the
    whole assembly.
    """

    def __init__(
        self,
        specs: list[StructureSpec],
        concentrations: tuple[float, ...] = (100.0, 20.0, 5.0, 0.0),
        cache_topologies: bool = True,
        device: torch.device | str = "cpu",
        crop_size: int | None = 512,
        cutoff: float = 10.0,
        max_residues: int | None = 1000,
        min_residues: int | None = 40,
        cache_limit: int = 256,
    ) -> None:
        if not specs:
            raise ValueError("no structures given")
        if not concentrations:
            raise ValueError("no damage levels given")
        if crop_size is not None and crop_size < 1:
            raise ValueError(f"crop_size must be positive or None, got {crop_size}")
        self.specs = specs
        self.concentrations = concentrations
        self.device = device
        #: Residues per sample for entries larger than this. ``None`` disables
        #: cropping, in which case ``max_residues`` is the only thing standing
        #: between a 3,500-residue assembly and a ~13 GB rollout.
        #:
        #: 512 is a memory budget, not an optimum, and that is the finding
        #: rather than a hedge. Measured over 40 validation entries above 700
        #: residues (``scripts/crop_report.py``), the share of a crop that is
        #: supervised -- the rest being the boundary shell that only provides
        #: context -- runs 29% at 192, 46% at 384, 57% at 512, 74% at 768 and
        #: 85% at 1024. The shell is one interaction cutoff deep whatever the
        #: crop size, so supervised residues per MB of retained activations
        #: *rises monotonically* with crop size: 0.059 at 192 against 0.173 at
        #: 1024. There is no efficiency sweet spot to find. Bigger is strictly
        #: better and the only thing stopping it is memory, so this number
        #: should be raised to whatever the training hardware will hold.
        self.crop_size = crop_size
        #: Interaction cutoff used to decide which cropped residues have
        #: complete perception. Must match the rollout's, or the mask is
        #: computed for a graph that is not the one being built.
        self.cutoff = cutoff
        #: A guard on what actually enters the rollout, checked *after* any
        #: crop. With cropping on it should never fire; with cropping off it is
        #: the only protection against an opaque mid-training OOM.
        #: ``None`` disables the check.
        self.max_residues = max_residues
        #: And a floor, which the manifest filter cannot provide on its own.
        #: RCSB's ``deposited_polymer_monomer_count`` counts *all* polymer
        #: monomers, nucleotides included, so a DNA-protein complex passes a
        #: 40-residue filter and then yields almost no amino acids once the
        #: loader keeps only the standard twenty. The smoke test found entries
        #: down to 2 residues this way. Applied to the entry, before cropping:
        #: a crop of a large entry is never short.
        self.min_residues = min_residues
        #: How many parsed entries and built topologies to keep. See
        #: :meth:`protein` for why this is bounded at all.
        self.cache_limit = cache_limit
        self._cache: dict[str, tuple[FeatureTopology, Tensor]] = {}
        self._proteins: dict[str, Protein] = {}
        self._cache_topologies = cache_topologies

    def __len__(self) -> int:
        return len(self.specs)

    def protein(self, index: int) -> Protein:
        """The parsed entry, uncropped. Cached; this is the expensive part.

        The cache is **bounded**, which it has to be: parsing is expensive
        enough to be worth avoiding, but a pool of 23,761 entries that includes
        4,000-residue assemblies cannot be held in memory, and an unbounded
        cache would simply grow until training died of it. Eviction is
        insertion-order and crude, which is the right amount of machinery for
        something whose only job is to stop an epoch re-parsing the same entry.
        """
        spec = self.specs[index]
        if spec.pdb_id in self._proteins:
            return self._proteins[spec.pdb_id]
        protein = load_protein(spec.path)
        if self.min_residues is not None and len(protein) < self.min_residues:
            raise ValueError(
                f"{spec.pdb_id} yields only {len(protein)} standard amino "
                f"acids, under the min_residues limit of {self.min_residues}. "
                f"RCSB's polymer-monomer count includes nucleotides, so a "
                f"nucleic-acid complex can pass the manifest filter and still "
                f"have almost no protein."
            )
        if self._cache_topologies:
            while len(self._proteins) >= self.cache_limit:
                self._proteins.pop(next(iter(self._proteins)))
            self._proteins[spec.pdb_id] = protein
        return protein

    def load(
        self,
        index: int,
        seed_residue: int | None = None,
        rng: random.Random | None = None,
    ) -> tuple[FeatureTopology, Tensor]:
        """Topology and native coordinates, cropped if the entry is large.

        The crop seed defaults to ``random.Random(index)``, so a bare
        ``load(index)`` is reproducible -- evaluation and the smoke test need
        the same crop twice. Pass ``rng`` to vary it, which is what training
        does.
        """
        topo, coords, _indices, _seed = self.load_crop(index, seed_residue, rng)
        return topo, coords

    def load_crop(
        self,
        index: int,
        seed_residue: int | None = None,
        rng: random.Random | None = None,
    ) -> tuple[FeatureTopology, Tensor, list[int] | None, int | None]:
        """As :meth:`load`, also returning which residues were kept and the seed.

        Both are ``None`` when the entry was taken whole. Separate from
        :meth:`load` so the common call site stays two-valued and nothing has
        to carry the crop around in mutable state on the dataset.

        The seed is returned because it is what makes a crop *repeatable*:
        passing it back as ``seed_residue`` reproduces this exact instance,
        which is the only way a caller holding a state from an earlier sample
        can pair it with the topology it belongs to.
        """
        spec = self.specs[index]
        if spec.pdb_id in self._cache:
            topo, coords = self._cache[spec.pdb_id]
            return topo, coords, None, None

        protein = self.protein(index)
        indices: list[int] | None = None
        used_seed: int | None = None
        supervised = None
        if self.crop_size is not None and len(protein) > self.crop_size:
            taken = crop(
                protein,
                self.crop_size,
                cutoff=self.cutoff,
                seed_residue=seed_residue,
                rng=rng or random.Random(index),
            )
            protein, indices, supervised = taken.protein, taken.indices, taken.supervised
            used_seed = taken.seed_residue

        topo, coords = feature_topology(
            protein, device=self.device, supervised=supervised
        )
        if self.max_residues is not None and topo.n_residues > self.max_residues:
            raise ValueError(
                f"{spec.pdb_id} has {topo.n_residues} residues after cropping, "
                f"over the max_residues limit of {self.max_residues}. Retained "
                f"activations are linear in residue count, so this would need "
                f"roughly {topo.n_residues * 0.41:.0f} MB per rollout step. "
                f"Lower crop_size or raise the limit deliberately."
            )
        if self._cache_topologies and indices is None:
            while len(self._cache) >= self.cache_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[spec.pdb_id] = (topo, coords)
        return topo, coords, indices, used_seed

    def sample(
        self,
        index: int,
        concentration: float | None = None,
        generator: torch.Generator | None = None,
        rng: random.Random | None = None,
        crop_seed: int | None = None,
    ) -> Sample:
        """Damage structure ``index``. ``concentration=None`` picks a level at random.

        The curriculum is a *mixture* rather than a schedule: the rule must
        recover from mild and severe damage alike, and it is never told which it
        got. Telling it would be noise-level conditioning, which is precisely
        what DiffPack has and this deliberately does not.

        ``rng`` draws the crop seed as well, so successive samples of a large
        entry see different regions of it. ``crop_seed`` overrides that and
        reproduces one particular region -- needed whenever the caller already
        holds coordinates for a specific crop, because the topology and the
        native target must describe the same residues as those coordinates.
        """
        topo, native, crop_indices, used_seed = self.load_crop(
            index, seed_residue=crop_seed, rng=rng
        )
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
            crop_indices=crop_indices,
            crop_seed=used_seed,
        )

    def iter_epoch(
        self,
        generator: torch.Generator | None = None,
        shuffle: bool = True,
        rng: random.Random | None = None,
    ):
        order = (
            torch.randperm(len(self), generator=generator).tolist()
            if shuffle
            else range(len(self))
        )
        for index in order:
            yield self.sample(index, generator=generator, rng=rng)
