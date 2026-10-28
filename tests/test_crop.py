"""Spatial cropping: the subset is a valid structure, and the interior is intact.

The claim cropping rests on is that a strictly local rule cannot tell a cropped
structure from the real one *in the crop's interior*. That is testable rather
than arguable, and :func:`test_supervised_residue_perceives_the_same_graph`
tests it directly: every feature the rule reads for a supervised residue is
bit-identical between the crop and the whole entry.

The matching claim is that the boundary is excluded from the loss. The exactness
of the mask is checked against an independent brute-force contact calculation,
and the loss is checked to be the one computed on the narrowed mask.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from aanca import residue_constants as rc
from aanca.crop import (
    contact_sets,
    crop,
    crop_indices,
    crop_report,
    residue_centres,
)
from aanca.losses import chi_nll
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig, training_objective
from aanca.structure import load_protein
from aanca.torch_geometry import measure_chis, wrap_to_pi
from aanca.torch_graph import build_graph, feature_topology

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
CUTOFF = 10.0


@pytest.fixture(scope="module")
def ubiquitin():
    return load_protein(DATA / "1UBQ.pdb")


@pytest.fixture(scope="module")
def haemoglobin():
    """Four chains, ~574 residues: large enough to crop and multi-chain."""
    return load_protein(DATA / "4HHB.pdb")


# -- Protein.subset --------------------------------------------------------


def test_subset_keeps_the_residues_asked_for(ubiquitin):
    kept = [0, 3, 4, 10]
    sub = ubiquitin.subset(kept)
    assert len(sub) == len(kept)
    assert [r.resname for r in sub.residues] == [
        ubiquitin.residues[i].resname for i in kept
    ]
    assert sub.res_ids == [ubiquitin.res_ids[i] for i in kept]


def test_subset_renumbers_atom_indices_against_the_new_array(ubiquitin):
    """Keeping the old numbering would make every coordinate write land on the
    wrong atom -- silently, since the shapes still agree."""
    sub = ubiquitin.subset([5, 6, 7])
    for idx, res in zip(sub.atom_indices, sub.residues):
        assert np.allclose(sub.atoms.coord[idx], res.coords, atol=1e-5)
        assert list(sub.atoms.atom_name[idx]) == list(res.atom_names)


def test_subset_survives_a_torsion_update(ubiquitin):
    """The real test of the renumbering: a chi update writes through ``atoms``."""
    sub = ubiquitin.subset([2, 3, 4, 5])
    deltas = np.zeros((len(sub), rc.MAX_CHI))
    deltas[:, 0] = 0.3
    moved = sub.apply_chi_deltas(deltas)
    for idx, res in zip(moved.atom_indices, moved.residues):
        assert np.allclose(moved.atoms.coord[idx], res.coords, atol=1e-5)


def test_subset_rejects_nonsense(ubiquitin):
    with pytest.raises(ValueError):
        ubiquitin.subset([])
    with pytest.raises(ValueError):
        ubiquitin.subset([1, 1])
    with pytest.raises(IndexError):
        ubiquitin.subset([len(ubiquitin)])
    with pytest.raises(IndexError):
        ubiquitin.subset([-1])


# -- crop selection --------------------------------------------------------


def test_crop_takes_the_nearest_residues_in_ascending_order(haemoglobin):
    kept = crop_indices(haemoglobin, 120, seed_residue=300)
    assert len(kept) == 120
    assert kept == sorted(kept)
    centres = residue_centres(haemoglobin)
    inside = np.linalg.norm(centres[kept] - centres[300], axis=-1).max()
    outside = np.delete(
        np.linalg.norm(centres - centres[300], axis=-1), kept
    ).min()
    assert inside <= outside


def test_crop_is_reproducible_for_a_given_seed(haemoglobin):
    a = crop(haemoglobin, 150, cutoff=CUTOFF, seed_residue=42)
    b = crop(haemoglobin, 150, cutoff=CUTOFF, seed_residue=42)
    assert a.indices == b.indices
    assert np.array_equal(a.supervised, b.supervised)


def test_different_seeds_give_different_crops(haemoglobin):
    """Which is what makes the crop augmentation rather than truncation."""
    a = crop(haemoglobin, 150, cutoff=CUTOFF, seed_residue=10)
    b = crop(haemoglobin, 150, cutoff=CUTOFF, seed_residue=400)
    assert a.indices != b.indices


def test_a_crop_larger_than_the_structure_is_a_no_op(ubiquitin):
    """And supervises everything: with nothing removed, a truncated
    neighbourhood is the real molecular surface, not an artefact of ours.
    Masking it would discard exactly the surface residues the benchmark
    reports on."""
    taken = crop(ubiquitin, len(ubiquitin) + 50, cutoff=CUTOFF, seed_residue=0)
    assert taken.indices == list(range(len(ubiquitin)))
    assert taken.supervised.all()
    assert taken.protein is ubiquitin


def test_crop_rejects_a_bad_size_or_seed(haemoglobin):
    with pytest.raises(ValueError):
        crop_indices(haemoglobin, 0, seed_residue=0)
    with pytest.raises(IndexError):
        crop_indices(haemoglobin, 10, seed_residue=len(haemoglobin))


# -- the supervision mask --------------------------------------------------


def _brute_force_contacts(protein, cutoff: float) -> list[set[int]]:
    """Every residue pair within ``cutoff`` by minimum heavy-atom distance.

    Written independently of ``crop.contact_sets``, which goes through the
    featuriser's KD-tree helper. O(n^2) in atoms, so only for small inputs.
    """
    out: list[set[int]] = [set() for _ in range(len(protein))]
    coords = [r.coords for r in protein.residues]
    for i in range(len(protein)):
        for j in range(i + 1, len(protein)):
            d = np.linalg.norm(
                coords[i][:, None, :] - coords[j][None, :, :], axis=-1
            ).min()
            if d <= cutoff:
                out[i].add(j)
                out[j].add(i)
    return out


def test_contact_sets_match_an_independent_calculation(ubiquitin):
    mine = contact_sets(ubiquitin, CUTOFF)
    reference = _brute_force_contacts(ubiquitin, CUTOFF)
    for i in range(len(ubiquitin)):
        # ``contact_sets`` adds bonded sequence neighbours regardless of
        # distance, because the featuriser forces those edges.
        assert reference[i] <= mine[i]
        assert mine[i] - reference[i] <= {i - 1, i + 1}


def test_supervised_is_exactly_complete_perception(ubiquitin):
    """No margin, no radius heuristic: a residue is supervised iff every
    residue it can see is still present."""
    taken = crop(ubiquitin, 40, cutoff=CUTOFF, seed_residue=20)
    contacts = _brute_force_contacts(ubiquitin, CUTOFF)
    kept = set(taken.indices)
    for position, original in enumerate(taken.indices):
        expected = contacts[original] <= kept
        if expected:
            assert taken.supervised[position], original
        # The converse holds up to the forced sequence neighbours, which
        # ``contact_sets`` adds and the brute-force reference does not.
        if not taken.supervised[position]:
            missing = (contacts[original] | {original - 1, original + 1}) - kept
            assert missing


def test_the_boundary_shell_is_a_real_cost(ubiquitin):
    """Not every residue in a crop can be supervised, and pretending otherwise
    is the failure this mask exists to prevent."""
    taken = crop(ubiquitin, 40, cutoff=CUTOFF, seed_residue=20)
    assert 0.0 < taken.supervised_fraction < 1.0


def test_supervised_fraction_rises_with_crop_size(haemoglobin):
    """The shell is ~cutoff-deep whatever the size, so a bigger crop spends a
    smaller share of itself on context. This is the number that decides
    ``crop_size``, so it is measured rather than assumed."""
    report = crop_report(
        haemoglobin, sizes=(128, 256, 512), cutoff=CUTOFF, seeds=3,
        rng=random.Random(0),
    )
    assert report[128] < report[256] < report[512]


# -- the claim cropping rests on -------------------------------------------


def test_supervised_residue_perceives_the_same_graph(haemoglobin):
    """A supervised residue's features are identical in the crop and the whole.

    This is the justification for cropping at all. One message-passing hop
    bounded by ``cutoff`` means a residue whose contacts are all present sees
    exactly what it would have seen in the intact assembly -- not an
    approximation of it. If this test fails, cropping is lossy in the interior
    and the whole approach is wrong rather than merely imperfect.
    """
    taken = crop(haemoglobin, 300, cutoff=CUTOFF, seed_residue=250)
    assert taken.supervised.any()

    whole_topo, whole_coords = feature_topology(haemoglobin)
    crop_topo, crop_coords = feature_topology(
        taken.protein, supervised=taken.supervised
    )
    whole = build_graph(whole_coords.to(torch.float32), whole_topo, k=64, cutoff=CUTOFF)
    cropped = build_graph(crop_coords.to(torch.float32), crop_topo, k=64, cutoff=CUTOFF)

    checked = 0
    for position, original in enumerate(taken.indices):
        if not taken.supervised[position]:
            continue
        assert torch.allclose(
            cropped.node_features[position], whole.node_features[original], atol=1e-5
        )
        # Same neighbours, in the same order, after mapping crop positions back
        # to entry positions.
        n_edges = int(cropped.edge_mask[position].sum())
        assert n_edges == int(whole.edge_mask[original].sum())
        mapped = [taken.indices[j] for j in cropped.neighbours[position, :n_edges]]
        assert mapped == whole.neighbours[original, :n_edges].tolist()
        assert torch.allclose(
            cropped.edge_features[position, :n_edges],
            whole.edge_features[original, :n_edges],
            atol=1e-4,
        )
        assert torch.allclose(
            cropped.envelope[position, :n_edges],
            whole.envelope[original, :n_edges],
            atol=1e-5,
        )
        checked += 1
    assert checked > 50, f"only {checked} supervised residues compared"


def test_the_mask_is_exactly_the_set_whose_perception_changed(haemoglobin):
    """The sharp version of both halves at once.

    The residues whose features differ between crop and whole are *precisely*
    the unsupervised ones -- not a subset (which would mean the loss touches
    corrupted inputs) and not a superset (which would mean the mask discards
    residues that were fine). Measured on 4HHB at crop size 300: 137 of 300
    residues differ, and all 137 are unsupervised.
    """
    taken = crop(haemoglobin, 300, cutoff=CUTOFF, seed_residue=250)
    whole_topo, whole_coords = feature_topology(haemoglobin)
    crop_topo, crop_coords = feature_topology(
        taken.protein, supervised=taken.supervised
    )
    whole = build_graph(whole_coords.to(torch.float32), whole_topo, k=64, cutoff=CUTOFF)
    cropped = build_graph(crop_coords.to(torch.float32), crop_topo, k=64, cutoff=CUTOFF)

    differing: set[int] = set()
    for position, original in enumerate(taken.indices):
        n_edges = int(cropped.edge_mask[position].sum())
        if n_edges != int(whole.edge_mask[original].sum()):
            differing.add(position)
            continue
        mapped = [taken.indices[j] for j in cropped.neighbours[position, :n_edges]]
        if mapped != whole.neighbours[original, :n_edges].tolist():
            differing.add(position)
            continue
        if not torch.allclose(
            cropped.edge_features[position, :n_edges],
            whole.edge_features[original, :n_edges],
            atol=1e-4,
        ):
            differing.add(position)

    unsupervised = {i for i, flag in enumerate(taken.supervised) if not flag}
    assert differing == unsupervised
    assert differing, "nothing differed, so the comparison proves nothing"


# -- the loss respects the mask --------------------------------------------


def _topology_with_mask(protein, supervised):
    return feature_topology(protein, supervised=supervised)


def test_objective_scores_only_supervised_residues(ubiquitin):
    """The reconstruction term equals the NLL computed on the narrowed mask.

    Checked by recomputing it from the rollout's own captured outputs, so this
    is the actual number the optimiser saw rather than a re-run.
    """
    supervised = np.zeros(len(ubiquitin), dtype=bool)
    supervised[: len(ubiquitin) // 2] = True
    topo, native = _topology_with_mask(ubiquitin, supervised)

    rule = AANCARule()
    config = RolloutConfig(steps=4, truncate=2, k=24, cutoff=CUTOFF)
    damaged = native.clone()
    terms, result = training_objective(
        rule,
        damaged,
        native,
        topo,
        config,
        generator=torch.Generator().manual_seed(0),
        rng=random.Random(0),
    )
    mu, log_kappa, seen = result.supervised
    torsion = topo.torsion
    target = wrap_to_pi(
        measure_chis(native, torsion) - measure_chis(seen, torsion)
    )
    narrowed = torsion.chi_mask & torch.as_tensor(supervised).unsqueeze(-1)
    expected = chi_nll(mu, log_kappa, target, narrowed, torsion.periodic)
    assert torch.allclose(terms.reconstruction, expected.detach(), atol=1e-6)

    # And it is genuinely narrower than the unmasked loss.
    whole = chi_nll(mu, log_kappa, target, torsion.chi_mask, torsion.periodic)
    assert not torch.allclose(expected, whole, atol=1e-4)


def test_objective_refuses_an_all_boundary_crop(ubiquitin):
    """A crop smaller than one interaction cutoff supervises nothing. Returning
    a zero loss would train on nothing while reporting success."""
    topo, native = _topology_with_mask(
        ubiquitin, np.zeros(len(ubiquitin), dtype=bool)
    )
    with pytest.raises(ValueError, match="complete perception"):
        training_objective(
            AANCARule(),
            native,
            native,
            topo,
            RolloutConfig(steps=2, truncate=2, k=16, cutoff=CUTOFF),
            generator=torch.Generator().manual_seed(0),
            rng=random.Random(0),
        )


def test_gradient_still_reaches_every_parameter_under_a_mask(ubiquitin):
    supervised = np.zeros(len(ubiquitin), dtype=bool)
    supervised[10:40] = True
    topo, native = _topology_with_mask(ubiquitin, supervised)
    rule = AANCARule()
    terms, _ = training_objective(
        rule,
        native,
        native,
        topo,
        RolloutConfig(steps=4, truncate=2, k=24, cutoff=CUTOFF),
        generator=torch.Generator().manual_seed(0),
        rng=random.Random(0),
    )
    terms.total.backward()
    missing = [name for name, p in rule.named_parameters() if p.grad is None]
    assert not missing


def test_mask_shape_is_checked(ubiquitin):
    """A mask built for the uncropped structure must not be accepted silently."""
    with pytest.raises(ValueError, match="supervised mask"):
        feature_topology(
            ubiquitin.subset(list(range(20))),
            supervised=np.ones(len(ubiquitin), dtype=bool),
        )


def test_topology_without_a_mask_supervises_everything(ubiquitin):
    topo, _ = feature_topology(ubiquitin)
    assert topo.supervised is None
    assert topo.supervised_mask.all()
    assert topo.supervised_mask.shape == (len(ubiquitin),)
