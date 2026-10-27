"""Report the Milestone 1 numbers, and sanity-check chi1 against rotamer statistics.

    python scripts/validate_geometry.py

Two independent checks, because a self-consistent round trip can still be built
on a wrong convention:

1. Round-trip exactness -- randomise every chi, restore the native values, report
   the worst coordinate deviation. The milestone bar is 0.01 A.
2. The chi1 distribution. Side-chain chi1 is known to cluster at roughly -60
   (gauche-minus, the most populated), +60 (gauche-plus) and 180 (trans). If the
   atom order or the dihedral sign were wrong, the populations would swap or the
   peaks would shift, which the round trip alone would never notice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aanca import residue_constants as rc
from aanca.structure import load_protein, sidechain_rmsd

DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
ENTRIES = ("193L", "3NIR", "1UBQ")


def round_trip(path: Path) -> tuple[float, float, int]:
    protein = load_protein(path)
    native, mask = protein.chis()
    rng = np.random.default_rng(0)

    damaged = protein.apply_chi_deltas(
        np.where(mask, rng.uniform(-np.pi, np.pi, size=mask.shape), 0.0)
    )
    damaged_chis, _ = damaged.chis()
    restored = damaged.apply_chi_deltas(np.where(mask, native - damaged_chis, 0.0))

    # float64 residue coords, not atoms.coord: an AtomArray is float32, where a
    # 1e-14 A deviation rounds to a flattering exact zero.
    max_dev = max(
        float(np.abs(a.coords - b.coords).max())
        for a, b in zip(restored.residues, protein.residues)
    )
    damage_rmsd = float(np.nanmean(sidechain_rmsd(damaged, protein)))
    return max_dev, damage_rmsd, int(mask.sum())


def chi1_histogram() -> dict[str, np.ndarray]:
    bins = np.deg2rad([-180, -120, 0, 120, 180])
    counts = np.zeros(4)
    for pdb_id in ENTRIES:
        path = DATA / f"{pdb_id}.pdb"
        if not path.exists():
            continue
        protein = load_protein(path)
        chis, mask = protein.chis()
        vals = chis[:, 0][mask[:, 0]]
        counts += np.histogram(vals, bins=bins)[0]
    # The -180..-120 and 120..180 bins are both "trans"; fold them.
    trans = counts[0] + counts[3]
    return {
        "gauche-minus (~-60)": counts[1],
        "gauche-plus (~+60)": counts[2],
        "trans (~180)": np.array(trans),
    }


def main() -> int:
    print("Round-trip exactness (randomise every chi, restore natives)")
    print(f"{'entry':<8}{'chis':>7}{'damage RMSD':>14}{'max deviation':>16}")
    worst = 0.0
    for pdb_id in ENTRIES:
        path = DATA / f"{pdb_id}.pdb"
        if not path.exists():
            print(f"{pdb_id:<8}  missing -- run scripts/fetch_test_structures.py")
            continue
        max_dev, damage_rmsd, n_chi = round_trip(path)
        worst = max(worst, max_dev)
        print(f"{pdb_id:<8}{n_chi:>7}{damage_rmsd:>12.2f} A{max_dev:>14.2e} A")
    print(f"\nworst deviation {worst:.2e} A, bar 1.00e-02 A -- "
          f"{'PASS' if worst < 0.01 else 'FAIL'}")

    print("\nchi1 rotamer populations (expect gauche-minus most populated,")
    print("then trans, then gauche-plus -- the standard ordering)")
    hist = chi1_histogram()
    total = sum(float(v) for v in hist.values())
    for label, count in hist.items():
        frac = float(count) / total if total else 0.0
        print(f"  {label:<22}{int(count):>5}  {frac:>6.1%}")

    print("\nPer-type chi coverage")
    print(f"  rotatable types:  {sum(1 for r in rc.AA3 if rc.num_chi(r) > 0)}/20")
    print(f"  treated as rigid: {sorted(rc.RIGID_RESIDUES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
