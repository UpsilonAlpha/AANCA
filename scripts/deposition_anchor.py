"""The missing positive anchor: what a 12-residue window reads between two
independent depositions of the *same* protein.

**What this is for.** `rmsd_dmg` is anchored at *do nothing* (10.77 A) and at
*reproduce the crystal* (0.00 A) and has **no anchor for a good non-native
answer**, which is what the project is actually for -- regrowing a binder
interface into something that binds *better* would score badly against the
crystal and be the result wanted. That is a missing positive control, and
missing positives are the error that produces optimistic bias. See
``docs/claims/the-success-criterion-has-no-positive-anchor.md``.

If the same window differs by X A between two crystal forms of one sequence,
then **X is the scale at which "different but real" lives**, and an arm within
X of native is as close as the crystal itself is reproducible. Below X,
"wrong" and "a different valid conformation" are not separable by this metric
and no amount of training makes them so.

**It runs the project's own superposition code**, ``fape.ca_rmsd_on``, which is
the CA-level core of the ``rmsd_dmg`` used by every arm. An anchor computed by
different arithmetic from the quantity it anchors would compare two conventions
and call the difference a scale.

**Method, and what it deliberately excludes.**

  * Pairs come from RCSB's ``clusters-by-entity-100.txt``, so identity is theirs
    rather than from an alignment written here. ``scripts/deposition_pairs.py``
    finds and fetches them, and excludes AlphaFold / ModelArchive computed
    models -- including those would measure how far a *prediction* sits from a
    crystal and report it as crystallographic reproducibility.
  * Residues are matched on **author numbering within a chain**, not by
    position, because the two entries have different missing-residue sets. A
    positional match would silently compare residue i to residue i+3 wherever a
    loop is unmodelled, and
    ``docs/RETRACTIONS.md`` already records a one-residue numbering slip reading
    2.8-4.2 A in another project. Only residues present in both are used.
  * The **method of each entry is read from the mmCIF and reported**, because an
    X-ray/NMR pair measures the difference between two techniques and not the
    reproducibility of one. They are separated rather than pooled.
  * The window slides along the chain. Every window is superposed on **all
    matched residues outside it**, which is exactly what `rmsd_dmg` does with a
    damaged window against its surviving context.

Usage:

    python scripts/deposition_anchor.py
    python scripts/deposition_anchor.py --width 12 --stride 4
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import torch

from aanca.fape import ca_rmsd_on

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "data" / "pdb"
PAIRS = ROOT / "data" / "deposition_pairs"


def load_ca(path: Path) -> tuple[dict[tuple[str, int], torch.Tensor], str]:
    """``{(chain, author_seq_id): CA xyz}`` and the experimental method.

    Parsed straight from the mmCIF rather than through ``RepackingDataset``, so
    the anchor does not inherit the pipeline's cropping, filtering or chain
    selection. Only the first model is read.
    """
    import numpy as np
    from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile, get_structure

    # The project's own pool is stored as **BinaryCIF** (``data/pdb/*.bcif``)
    # while freshly fetched partners are text mmCIF, so the reader has to be
    # chosen by suffix. Reading a .bcif with the text parser raises
    # ``UnicodeDecodeError`` for every file, which is what the first run did.
    if path.suffix == ".bcif":
        cif = BinaryCIFFile.read(str(path))
    else:
        cif = CIFFile.read(str(path))
    block = cif.block
    method = ""
    if "exptl" in block:
        values = block["exptl"]["method"].as_array()
        method = " / ".join(sorted({str(v) for v in values}))

    # ``model=1`` so an NMR ensemble contributes one conformer rather than its
    # spread, which would otherwise enter the anchor as if it were
    # crystallographic disagreement.
    array = get_structure(cif, model=1)
    mask = (array.atom_name == "CA") & (array.element != "H")
    array = array[mask]

    out: dict[tuple[str, int], torch.Tensor] = {}
    seen: set[tuple[str, int]] = set()
    for chain, num, xyz in zip(array.chain_id, array.res_id, array.coord):
        key = (str(chain), int(num))
        # **Keep the first CA per (chain, residue) and drop the rest.** An
        # altloc or an insertion code can give a residue number two CA atoms,
        # and a dict would otherwise silently take whichever came last in the
        # file -- a different atom in each of the two entries being compared.
        if key in seen:
            continue
        seen.add(key)
        out[key] = torch.tensor(np.asarray(xyz), dtype=torch.float64)
    return out, method


def matched_chain(
    a: dict[tuple[str, int], torch.Tensor],
    b: dict[tuple[str, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The longest single chain the two entries share, by author numbering.

    One chain, because a window spanning two chains is two windows, and the
    anchor has to mean the same thing ``damage_window`` means -- a contiguous
    run inside one covalent chain.
    """
    by_chain: dict[str, list[int]] = collections.defaultdict(list)
    for (chain, num) in a:
        if (chain, num) in b:
            by_chain[chain].append(num)
    if not by_chain:
        return None
    chain = max(by_chain, key=lambda c: len(by_chain[c]))
    nums = sorted(by_chain[chain])
    if len(nums) < 30:
        return None
    first = torch.stack([a[(chain, n)] for n in nums])
    second = torch.stack([b[(chain, n)] for n in nums])
    return first, second


def global_rmsd(first: torch.Tensor, second: torch.Tensor) -> float:
    """CA RMSD over every matched residue, superposed on every matched residue.

    **The gate, and the anchor is meaningless without it.** A window value is
    only a statement about that window if the rest of the molecule actually
    superposes: the window RMSD is read after a fit to the context, so a fit
    that fails globally makes *every* window read large for reasons that have
    nothing to do with the window.

    The first run of this script showed exactly that and the pooled median hid
    it. Four of twenty-five X-ray pairs had a median window of 17-19 A **with
    their minimum also above 10 A**, which is not one flexible loop -- it is the
    whole correspondence being wrong, or two genuinely different conformational
    states of one sequence. Pooling them with the 0.08 A pairs gave a median of
    0.620 A with a 90th percentile of 14.5 A, which is two populations wearing
    one number.

    So this is the convention check for this instrument: a quantity derived
    independently of the window values that must agree with them before any of
    them mean anything.
    """
    everything = torch.ones(first.shape[0], dtype=torch.bool)
    value = ca_rmsd_on(first[everything], second[everything],
                       first[everything], second[everything])
    return float(value)


def windows(
    first: torch.Tensor, second: torch.Tensor, width: int, stride: int
) -> list[float]:
    n = first.shape[0]
    values = []
    for start in range(0, n - width + 1, stride):
        read = torch.zeros(n, dtype=torch.bool)
        read[start : start + width] = True
        fit = ~read
        if int(fit.sum()) < 3:
            continue
        value = ca_rmsd_on(first[fit], second[fit], first[read], second[read])
        if torch.isfinite(value):
            values.append(float(value))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    if not PAIRS.exists():
        print("no pairs yet: run scripts/deposition_pairs.py --fetch N")
        return 1

    held = {name.split(".")[0].upper(): PDB / name for name in
            __import__("os").listdir(PDB)}
    partners = sorted(PAIRS.glob("*.cif"))
    if not partners:
        print("no pairs yet: run scripts/deposition_pairs.py --fetch N")
        return 1

    # Re-derive which held structure each partner belongs with, from the same
    # cluster file, rather than trusting a filename convention.
    from deposition_pairs import candidates

    mapping = {}
    for a, rest in candidates(4000):
        for b in rest:
            mapping.setdefault(b, a)

    rows = []
    for path in partners:
        pid = path.stem.upper()
        other = mapping.get(pid)
        if other is None or other not in held:
            continue
        try:
            a_ca, a_method = load_ca(held[other])
            b_ca, b_method = load_ca(path)
        except Exception as error:  # noqa: BLE001
            print(f"  {other} <-> {pid}: unreadable ({type(error).__name__})")
            continue
        pair = matched_chain(a_ca, b_ca)
        if pair is None:
            continue
        values = windows(pair[0], pair[1], args.width, args.stride)
        if not values:
            continue
        rows.append((
            other, pid, a_method, b_method, len(pair[0]), values,
            global_rmsd(pair[0], pair[1]),
        ))

    if not rows:
        print("no measurable pairs")
        return 1

    xray = [r for r in rows if "DIFFRACTION" in (r[2] + r[3]).upper()
            and "NMR" not in (r[2] + r[3]).upper()]
    other = [r for r in rows if r not in xray]

    # **The gate.** 2 A is generous for two depositions of one sequence -- the
    # pairs that pass cluster at 0.1-0.8 A -- and anything above it is a failed
    # correspondence or a different conformational state, neither of which is
    # crystallographic reproducibility.
    GATE = 2.0
    fit_ok = [r for r in xray if r[6] <= GATE]
    fit_bad = [r for r in xray if r[6] > GATE]

    def report(group: list, label: str) -> list[float]:
        if not group:
            print(f"\n{label}: none")
            return []
        print(f"\n{label}: {len(group)} pairs")
        print(f"  {'held':>6} {'partner':>8} {'matched':>8} {'global':>7} "
              f"{'windows':>8} {'median':>7} {'min':>6} {'max':>7}")
        pooled: list[float] = []
        for a, b, _ma, _mb, n, values, whole in sorted(group):
            values_sorted = sorted(values)
            median = values_sorted[len(values_sorted) // 2]
            print(f"  {a:>6} {b:>8} {n:>8} {whole:>7.2f} {len(values):>8} "
                  f"{median:>7.3f} {min(values):>6.3f} {max(values):>7.3f}")
            pooled.extend(values)
        pooled.sort()
        def q(p: float) -> float:
            return pooled[min(len(pooled) - 1, int(p * len(pooled)))]
        print(f"  pooled {len(pooled)} windows: median {q(0.5):.3f} A, "
              f"10th {q(0.1):.3f}, 90th {q(0.9):.3f}, max {pooled[-1]:.3f}")
        return pooled

    print(f"\ncross-deposition anchor, {args.width}-residue window, "
          f"stride {args.stride}")
    print("superposed on all matched residues OUTSIDE the window, which is what")
    print("rmsd_dmg does with a damaged window against its surviving context.")
    print(f"`global` is the CA RMSD over ALL matched residues. A pair whose")
    print(f"global fit is worse than {GATE:g} A cannot say anything about one")
    print("window, because every window is read after a fit to the context.")
    pooled = report(fit_ok, f"X-ray, global fit <= {GATE:g} A (THE ANCHOR)")
    report(fit_bad, f"X-ray, global fit > {GATE:g} A (different state or bad "
                    f"correspondence)")
    report(other, "mixed or non-diffraction methods (NOT the anchor)")

    if pooled:
        median = pooled[len(pooled) // 2]
        p90 = pooled[min(len(pooled) - 1, int(0.9 * len(pooled)))]
        print(f"\n  *** THE ANCHOR: the same {args.width}-residue window reads")
        print(f"  *** {median:.2f} A (median) and {p90:.2f} A (90th) between")
        print(f"  *** two depositions of one protein whose global fit holds.")
        print("  *** So an arm within that of native is as close as the")
        print("  *** crystal is reproducible, and below it this metric cannot")
        print("  *** separate 'wrong' from 'a different valid conformation'.")
        print(f"  *** Against the 10.77 A model-free floor, the usable range")
        print(f"  *** of rmsd_dmg is {median:.2f} to 10.77 A -- every arm run")
        print("  *** so far has scored 12.18 to 12.85, outside it entirely.")
        print("  *** Read the 90th percentile too: the anchor is a")
        print("  *** DISTRIBUTION, and quoting only the median would repeat")
        print("  *** the mistake this script's own first run made.")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
