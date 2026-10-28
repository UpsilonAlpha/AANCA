"""Bond and angle control: exactly invertible, and ring-safe."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from aanca import residue_constants as rc
from aanca.structure import load_protein
from aanca.torch_geometry import (
    apply_angle_deltas,
    apply_bond_deltas,
    apply_internal_deltas,
    invert_internal_deltas,
    measure_angles,
    measure_bonds,
    measure_chis,
    topology_from_protein,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module", params=["193L", "1UBQ"])
def loaded(request):
    path = DATA / f"{request.param}.pdb"
    if not path.exists():
        pytest.skip(f"{path} missing; run scripts/fetch_test_structures.py")
    protein = load_protein(path)
    topo, coords = topology_from_protein(protein)
    return protein, topo, coords


def _rand(shape, scale, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g, dtype=torch.float64) - 0.5) * 2 * scale


# -- ring safety, which is why the topology is derived rather than listed ----


def test_ring_bonds_are_excluded():
    """An aromatic ring contains no backbone, so a backbone-reachability test
    alone misses it -- PHE's CG-CD1 would look stretchable. Only CA-CB and
    CB-CG, which are exocyclic, may be driven."""
    for res in ("PHE", "TYR", "TRP", "HIS"):
        drivable = {(a, b) for a, b, _ in rc.MOVABLE_BONDS[res]}
        assert ("CB", "CG") in drivable
        assert ("CA", "CB") in drivable
        assert ("CG", "CD1") not in drivable
        assert ("CG", "CD2") not in drivable
    # TYR's hydroxyl hangs off the ring, so it is drivable.
    assert ("CZ", "OH") in {(a, b) for a, b, _ in rc.MOVABLE_BONDS["TYR"]}


def test_proline_exposes_nothing():
    assert rc.MOVABLE_BONDS["PRO"] == ()
    assert rc.MOVABLE_ANGLES["PRO"] == ()


def test_glycine_exposes_nothing():
    assert rc.MOVABLE_BONDS["GLY"] == ()


def test_moving_sets_never_contain_backbone():
    for res in rc.AA3:
        for _a, _b, moving in rc.MOVABLE_BONDS[res]:
            assert not moving & rc.BACKBONE_ATOMS, res
        for _a, _b, _c, moving in rc.MOVABLE_ANGLES[res]:
            assert not moving & rc.BACKBONE_ATOMS, res


def test_downstream_atoms_rejects_a_non_bond():
    with pytest.raises(KeyError):
        rc.downstream_atoms("ALA", "CB", "CG")


# -- exact invertibility, the Milestone 1 guarantee extended ----------------


def test_bond_stretch_is_exactly_invertible(loaded):
    _protein, topo, coords = loaded
    deltas = _rand(topo.bond_mask.shape, 0.05, 0)
    stretched = apply_bond_deltas(coords, deltas, topo)
    assert (stretched - coords).abs().max() > 1e-3
    restored = apply_bond_deltas(stretched, -deltas, topo)
    assert float((restored - coords).abs().max()) < 1e-10


def test_angle_bend_is_exactly_invertible(loaded):
    _protein, topo, coords = loaded
    deltas = _rand(topo.angle_mask.shape, 0.15, 1)
    bent = apply_angle_deltas(coords, deltas, topo)
    assert (bent - coords).abs().max() > 1e-3
    restored = apply_angle_deltas(bent, -deltas, topo)
    assert float((restored - coords).abs().max()) < 1e-10


def test_all_three_coordinates_invert_together(loaded):
    """Chi, angle and bond do not commute, so the inverse must run in reverse."""
    _protein, topo, coords = loaded
    chi = _rand(topo.chi_mask.shape, 1.0, 2)
    angle = _rand(topo.angle_mask.shape, 0.1, 3)
    bond = _rand(topo.bond_mask.shape, 0.03, 4)

    moved = apply_internal_deltas(coords, topo, chi=chi, angle=angle, bond=bond)
    assert (moved - coords).abs().max() > 0.5
    restored = invert_internal_deltas(moved, topo, chi=chi, angle=angle, bond=bond)
    assert float((restored - coords).abs().max()) < 1e-9


def test_inversion_is_order_independent(loaded):
    """A stronger property than the one originally expected here.

    The three operators are not obviously commutative -- they are rotations and
    translations about different axes through different points, acting on
    overlapping atom sets. But negating the deltas restores the coordinates
    whichever order the inverse runs in, because each operation re-derives its
    axis from the coordinates it sees and every axis is pinned by atoms upstream
    of that operation's own moving set, plus at most one atom that stays on the
    axis or in the plane. So the axis is carried covariantly and re-derived
    identically on the way back.

    Worth pinning because it makes composing damage operators safe without
    tracking the order they were applied in.
    """
    import itertools

    _protein, topo, coords = loaded
    deltas = {
        "chi": _rand(topo.chi_mask.shape, 1.0, 5),
        "angle": _rand(topo.angle_mask.shape, 0.2, 6),
        "bond": _rand(topo.bond_mask.shape, 0.04, 7),
    }

    for sequence in itertools.permutations(deltas):
        moved = coords
        for name in sequence:
            moved = apply_internal_deltas(moved, topo, **{name: deltas[name]})
        assert (moved - coords).abs().max() > 0.5

        for inverse in (sequence, tuple(reversed(sequence))):
            back = moved
            for name in inverse:
                back = apply_internal_deltas(back, topo, **{name: -deltas[name]})
            deviation = float((back - coords).abs().max())
            assert deviation < 1e-9, f"{sequence} inverted as {inverse}: {deviation:.2e}"


# -- what the operators do and do not change --------------------------------


def test_bond_stretch_changes_only_its_own_length(loaded):
    _protein, topo, coords = loaded
    before = measure_bonds(coords, topo)
    deltas = torch.zeros_like(before)
    deltas[:, 0] = 0.03
    after = measure_bonds(apply_bond_deltas(coords, deltas, topo), topo)
    changed = (after - before).abs()
    assert torch.allclose(
        changed[topo.bond_mask[:, 0], 0],
        torch.full_like(changed[topo.bond_mask[:, 0], 0], 0.03),
        atol=1e-9,
    )


def test_bond_stretch_leaves_chis_alone(loaded):
    """A translation along the bond axis cannot rotate anything."""
    _protein, topo, coords = loaded
    before = measure_chis(coords, topo)
    stretched = apply_bond_deltas(coords, _rand(topo.bond_mask.shape, 0.04, 7), topo)
    after = measure_chis(stretched, topo)
    assert float((after - before).abs().max()) < 1e-8


def test_angle_bend_changes_the_angle_by_the_delta(loaded):
    _protein, topo, coords = loaded
    before = measure_angles(coords, topo)
    deltas = torch.zeros_like(before)
    deltas[:, 0] = 0.05
    after = measure_angles(apply_angle_deltas(coords, deltas, topo), topo)
    rows = topo.angle_mask[:, 0]
    # Sign depends on the plane normal's orientation, so compare magnitudes.
    assert torch.allclose(
        (after - before)[rows, 0].abs(),
        torch.full_like((after - before)[rows, 0], 0.05),
        atol=1e-8,
    )


def test_backbone_never_moves_under_bond_or_angle_changes(loaded):
    protein, topo, coords = loaded
    moved = apply_internal_deltas(
        coords,
        topo,
        angle=_rand(topo.angle_mask.shape, 0.2, 8),
        bond=_rand(topo.bond_mask.shape, 0.05, 9),
    )
    for i, res in enumerate(protein.residues):
        for name in ("N", "CA", "C", "O"):
            if name in res.index:
                j = res.index[name]
                assert torch.equal(moved[i, j], coords[i, j]), f"{name} moved"


def test_none_skips_the_coordinate_entirely(loaded):
    """The v1 path: bond and angle are None, so no work is done at all."""
    _protein, topo, coords = loaded
    out = apply_internal_deltas(coords, topo, chi=torch.zeros_like(topo.chi_mask, dtype=coords.dtype))
    assert torch.equal(out, coords)


def test_gradients_flow_through_bond_and_angle(loaded):
    _protein, topo, coords = loaded
    bond = torch.zeros(topo.bond_mask.shape, dtype=torch.float64, requires_grad=True)
    angle = torch.zeros(topo.angle_mask.shape, dtype=torch.float64, requires_grad=True)
    apply_internal_deltas(coords, topo, angle=angle, bond=bond).sum().backward()
    assert float(bond.grad[topo.bond_mask].abs().sum()) > 0.0
    assert float(angle.grad[topo.angle_mask].abs().sum()) > 0.0


def test_rejects_wrong_shapes(loaded):
    _protein, topo, coords = loaded
    with pytest.raises(ValueError):
        apply_bond_deltas(coords, torch.zeros(topo.n_residues, 1), topo)
    with pytest.raises(ValueError):
        apply_angle_deltas(coords, torch.zeros(topo.n_residues, 1), topo)
