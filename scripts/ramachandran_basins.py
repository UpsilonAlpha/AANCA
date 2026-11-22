"""Does the regrown window land in the basin its NATIVE conformation occupies?

**RMSD says where the window ended up. It does not say whether anything local
was learned.** A 12-residue window pinned at both ends can score a respectable
placement RMSD while every residue in it is in a conformation no protein adopts.
So this asks the direct question instead: for the window residues whose
*crystal* (phi, psi) is helical, does the rule put them in the helical basin of
the Ramachandran plot? And the same for sheet. That is secondary structure as
the plot defines it, measured residue by residue on the damaged window only.

    python scripts/ramachandran_basins.py runs/g22_random/best.pt
    python scripts/ramachandran_basins.py            # untrained control

**Four rows, one set of crops, one set of damage windows, one residue set.**

  1. ``native``  -- the crystal structure's own torsions. **A SANITY CHECK, and
     a tautology up to a wiring error**: the residues counted as native-helical
     are exactly those the classifier calls helical on native coordinates, so
     this row must read 1.000 on both basins. If it does not, the torsion
     measurement or the masking is wrong -- the rows below are then measuring
     nothing and the script says so in its closing lines.
  2. ``start``   -- the untrained start state (``baseline_coords`` in the run's
     own recorded ``initial`` mode). The chance rate.
  3. ``floor``   -- ``geometric_floor.relax_to_geometry`` on that same start:
     peptide bond closure plus sterics, no network and no native information.
     **The comparator that matters.** If closure and sterics alone put
     native-helical residues in the helical basin, then the rule doing so is
     not evidence that the rule learned anything.
  4. ``model``   -- a ``--steps`` (default 120) rollout of the checkpoint.

**Both fractions are printed over their residue counts, always.** A window is 12
residues and a crop may contribute two sheet residues; a fraction over 3 is not
a measurement and the reader has to be able to see the denominator rather than
infer it.

WHY COARSE RECTANGLES ARE GOOD ENOUGH HERE, AND WHERE THEY ARE NOT

The basins are rectangles lifted from ``ramachandran_clash_probe.py``, and they
are this repository's approximations to the Lovell et al. 2003 favoured regions
rather than published boundaries. The helix box is wide in phi and clips the
bridge region; the sheet box contains polyproline II, so "sheet" here means the
beta/PPII region and not beta strand. The ``native`` row cannot catch any of
that, being 1.000 for *any* box by construction, and a +-10 degree sweep over a
rectangle cannot tell you the shape was wrong.

**What licenses the measurement anyway is that it is a CONTROLLED comparison.**
All four rows are classified by the same boxes on the same residues, so a box
that is too wide, too narrow or the wrong shape biases every row the same way
and the bias largely cancels in the row-to-row differences. So read the
DIFFERENCES -- model against floor above all -- and do not quote an absolute
fraction as "the share of residues in the helical basin", because that number
belongs to these boxes and not to the Ramachandran plot.

The comparison that survives this is the one the script exists for: does the
model beat the network-free floor on the same boxes? A wrong box cannot
manufacture that gap, and it cannot hide it either unless it is wrong enough to
put both rows at a ceiling or a floor -- which the printed denominators and the
sweep will show.

TORSIONS
``aanca.backbone.measure_backbone_torsions`` returns radians and **zero where
undefined** -- phi needs the covalent bond on the N side, psi the one on the C
side, so the two are defined on different subsets of the residues. Zero radians
is a real angle that sits inside no basin here but would still be counted as a
measurement, so a residue missing either angle is **excluded**, not scored. The
mask comes from the same ``links`` the measurement used (``growth_links``), and
because ``links`` is a property of the topology and not of the coordinates, the
excluded set is identical for all four rows -- the rows are compared on exactly
the same residues.

BASINS
Rectangular boxes in degrees, taken verbatim from the ``FAVOURED`` regions
already used by ``scripts/ramachandran_clash_probe.py`` in this repo, which
describes them as MolProbity-style coarse boxes over the favoured regions of
Lovell, Davis, Arendall, de Bakker, Word, Prisant, Richardson & Richardson
(2003), "Structure validation by Calpha geometry: phi,psi and Cbeta deviation",
*Proteins* **50**:437-450 -- the alpha-R region for helix and the beta/PPII
region (plus its psi wrap) for sheet. The exact numbers are printed in the
output header; they are not hidden and they are not re-derived here.

WHERE THE MISPLACED RESIDUES GO (the cross-tabulation)

The two-column table says how often a row lands a residue in the basin its
crystal conformation occupies. It is silent on the complement, and the
complement carries the diagnosis: a row that misses the sheet basin on 90% of
natively-sheet residues has either put them in the HELIX box or put them
somewhere else again, and those are different failures. So the same residues
are also printed as a 2x4 cross-tabulation -- native class (helix, sheet)
against where the row actually placed them -- at every widening, with the same
denominators as the table above. Two readings are intended, and the script
states both in its closing lines:

  (a) **When a row misplaces a natively-SHEET residue, where does it put it?**
      Three different answers, and they do not mean the same thing. In the
      HELIX box: the rule learned one conformational mode and over-applies it.
      In OTHER_FAV: it produced a real protein conformation, just the wrong one
      -- alpha-L is populated, and a residue there is not evidence of anything
      unlearned about locality. Only DISALLOWED, in no favoured box at all,
      supports "it learned nothing local and is simply off the plot", and even
      then the sweep has to agree, because a residue sitting a few degrees
      outside a box edge is counted disallowed by a rectangle that has no edge
      there in reality.
  (b) **Where does the FLOOR put natively-HELICAL residues?** If the floor dumps
      natively-helical residues into the sheet box too, then the floor's
      apparent success on sheet residues is just what chain closure does -- the
      beta region is the open, extended part of the Ramachandran plot -- and
      "the model is below the floor on sheet" is partly an artefact of the floor
      being trivially good there rather than of the rule being bad.

The four destinations are ``helix`` (the alpha-R box), ``sheet`` (the beta/PPII
boxes), ``other_fav`` (in neither of those but inside some other box of
``ramachandran_clash_probe.FAVOURED`` -- alpha-L today, imported from there and
widened by the same sweep offset so the sweep stays coherent) and
``disallowed`` (in no favoured box at all). First match wins in that order.

**Two things the printed block says out loud, because the numbers do not.**
First, reading (b) is a comparison of two small fractions and it is worthless
if the floor row's optimiser did not close the chain, so (b) REFUSES to state a
verdict when the achieved ``bond`` exceeds ``FLOOR_BOND_UNREADABLE`` rather
than reporting a one-residue difference as discrimination. Second, the native
class is re-derived at every widening, which is what keeps the ``native`` row a
tautology and therefore a wiring check -- but it also means **the three sweep
columns are not the same residue set** (a wider sheet box is a larger native
class), so the per-widening denominator is printed on the block and the columns
are not to be read as a trend over one population.

**The caveat discipline does not relax here.** These are the same rectangles,
so only differences between rows are quotable and never an absolute fraction.
The destinations are exclusive and exhaustive, so each native class's four
fractions sum to 1.000 by construction -- that sum is arithmetic, not a check.
The check is the ``native`` row, whose natively-helical residues must
cross-tabulate 1.000 into the helix box and 0.000 into the other three, for the
same tautological reason the two-column table must read 1.000.

**They are rectangles, not contours, and that is a real limitation.** The
alpha-R box is wide in phi and clips the bridge region; the sheet box contains
polyproline II as well as beta. The guard is the sensitivity sweep: the whole
table is printed with every box widened and narrowed by 10 degrees, the native
labelling moving with it. A conclusion that flips between the three is not a
conclusion and the closing lines say so.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aanca.backbone import measure_backbone_torsions  # noqa: E402
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.growth import (  # noqa: E402
    BASELINE_MODES,
    baseline_coords,
    damage_window,
    growth_links,
)
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import initial_state, rollout  # noqa: E402
from geometric_floor import relax_to_geometry  # noqa: E402
from ramachandran_clash_probe import FAVOURED  # noqa: E402
from regrowth_dispersion import rule_from_checkpoint  # noqa: E402

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"

#: ``(phi_lo, phi_hi, psi_lo, psi_hi)`` in degrees. See the module docstring:
#: these are ``ramachandran_clash_probe.FAVOURED``'s alpha-R entry and its two
#: beta entries, unchanged.
HELIX_BOXES = ((-160.0, -45.0, -75.0, -5.0),)
SHEET_BOXES = (
    (-180.0, -45.0, 100.0, 180.0),
    (-180.0, -45.0, -180.0, -160.0),
)
BASINS = {"helix": HELIX_BOXES, "sheet": SHEET_BOXES}
ROWS = ("native", "start", "floor", "model")
#: The FAVOURED boxes this script does not already name as a basin -- alpha-L
#: today. Taken by DIFFERENCE from ``ramachandran_clash_probe.FAVOURED`` rather
#: than retyped, so a region added there becomes ``other_fav`` here instead of
#: being silently counted as disallowed.
OTHER_FAVOURED = tuple(
    box for box in FAVOURED if box not in HELIX_BOXES + SHEET_BOXES
)
# The difference above is only meaningful while this script's own basin boxes
# ARE elements of FAVOURED. If a box here is edited, or the sibling's alpha-R
# or beta box moves even slightly, the difference stops removing it: the box
# reappears in OTHER_FAVOURED, and residues just outside this script's basin
# but inside the sibling's would be counted `other_fav` instead of
# `disallowed` -- a quiet shift in exactly the number reading (a) turns on.
# Refuse at import rather than measure, since nothing downstream could notice.
_unmatched = [box for box in HELIX_BOXES + SHEET_BOXES if box not in FAVOURED]
if _unmatched:
    raise SystemExit(
        "ramachandran_basins: these basin boxes are not elements of "
        f"ramachandran_clash_probe.FAVOURED: {_unmatched}. OTHER_FAVOURED is "
        "taken by difference from FAVOURED, so it can no longer separate "
        "'in another populated region' from 'in no favoured region at all', "
        "and reading (a) would be wrong rather than missing. Either restore "
        "the boxes or state the other-favoured set explicitly."
    )
#: The cross-tabulation's columns: where a row actually put the residue.
#: Exclusive (the boxes do not overlap in psi, and first match wins in this
#: order regardless) and exhaustive. ``disallowed`` is in NO favoured box at
#: all, which is a different statement from "not in the box I was asking
#: about" -- see the docstring.
DESTINATIONS = ("helix", "sheet", "other_fav", "disallowed")
#: Printed column heads for ``DESTINATIONS``.
DESTINATION_TEXT = {
    "helix": "helix box",
    "sheet": "sheet box",
    "other_fav": "other favoured",
    "disallowed": "disallowed",
}
#: Above this achieved ``bond``, reading (b) refuses to state a verdict.
#: ``peptide_bond_restraint`` is ``mean((C-N length - 1.329 A)**2)`` plus
#: ``mean(1 + cos omega)``, and that second half alone is 0 at planar trans,
#: ~1.0 for a uniformly random omega and 2.0 at cis. So 0.5 is already a
#: chain whose typical peptide unit is nowhere near planar trans, before the
#: length term is counted at all: deliberately loose, refusing only when the
#: chain is plainly open rather than merely imperfect.
FLOOR_BOND_UNREADABLE = 0.5


def widen(boxes, margin: float):
    """Grow (or with a negative margin shrink) every box by ``margin`` degrees.

    Clamped to the plot. A box shrunk past itself becomes empty, which is
    reported as an empty box rather than silently wrapping.
    """
    out = []
    for lo_phi, hi_phi, lo_psi, hi_psi in boxes:
        out.append((
            max(-180.0, lo_phi - margin), min(180.0, hi_phi + margin),
            max(-180.0, lo_psi - margin), min(180.0, hi_psi + margin),
        ))
    return tuple(out)


def in_boxes(phi: float, psi: float, boxes) -> bool:
    return any(
        lo_phi <= phi <= hi_phi and lo_psi <= psi <= hi_psi
        for lo_phi, hi_phi, lo_psi, hi_psi in boxes
    )


def box_text(boxes) -> str:
    return " or ".join(
        f"phi [{a:.0f},{b:.0f}] psi [{c:.0f},{d:.0f}]"
        for a, b, c, d in boxes
    )


def share_trio(cells) -> str:
    """The sweep's three fractions as text, in ``-margin, 0, +margin`` order.

    ``cells`` is the list of ``(hit, total)`` one widening at a time. An empty
    native class prints ``n/a`` rather than a nan, because there is nothing
    there to be a fraction of.
    """
    return ", ".join(
        f"{hit / total:.3f}" if total else "n/a" for hit, total in cells
    )


def share_values(cells) -> list[float]:
    """The same fractions as numbers, empty native classes dropped."""
    return [hit / total for hit, total in cells if total]


def defined_mask(links: torch.Tensor, n_res: int) -> torch.Tensor:
    """Residues with BOTH phi and psi defined.

    ``links`` is ``(n_res - 1,)`` and ``links[i]`` says ``i`` is bonded to
    ``i + 1``. ``measure_backbone_torsions`` writes psi at ``i`` and phi at
    ``i + 1`` for each set link, so phi is defined where the PREVIOUS link is
    set and psi where this one is. Everything else is the zero the function
    returns for "undefined", and zero is a real angle, so it is excluded here.
    """
    bits = links.bool()
    false = torch.zeros(1, dtype=torch.bool, device=bits.device)
    psi_ok = torch.cat([bits, false])
    phi_ok = torch.cat([false, bits])
    return phi_ok & psi_ok


def degrees(phi: torch.Tensor, psi: torch.Tensor):
    scale = 180.0 / math.pi
    return phi * scale, psi * scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="a trained best.pt. Its recorded args supply the pool, crop, "
             "window and start mode, and its `eval_indices` supply the crops, "
             "so the table describes the task that run was judged on. Omit it "
             "for the UNTRAINED CONTROL, where the `model` row is an untrained "
             "rule and is only a wiring check",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--eval-structures", type=int, default=12)
    parser.add_argument("--pool", type=int, default=400)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--damage", type=int, default=12)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initial", default="random", choices=list(BASELINE_MODES))
    parser.add_argument(
        "--floor-iters", type=int, default=600,
        help="Adam iterations for the network-free floor row. The achieved "
             "bond and clash are printed: a floor from an optimiser that did "
             "not close the chain is not a floor",
    )
    parser.add_argument("--floor-lr", type=float, default=0.05)
    parser.add_argument("--clash-tolerance", type=float, default=0.25)
    parser.add_argument("--clash-weight", type=float, default=1.0)
    parser.add_argument(
        "--margin", type=float, default=10.0,
        help="the sensitivity sweep: the table is printed at -margin, 0 and "
             "+margin degrees on every box edge",
    )
    parser.add_argument("--quiet-crops", action="store_true",
                        help="suppress the per-crop residue census")
    args = parser.parse_args()

    device = torch.device(args.device)

    # **The run's own recorded args, never this script's defaults.**
    # `geometric_floor.py` carries the mapping and the reason: the trainer's
    # `--structures` is the TRAINING pool size, which is this script's
    # `--pool`, while `--structures` here is how many held-out crops to
    # measure. `eval_indices` are positions in
    # `available[:pool + eval_structures]`, so a differently sized pool makes
    # the same integer name a different protein.
    blob = None
    rule = None
    recorded_indices: list[int] = []
    if args.checkpoint:
        blob = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
        rule, run_args, config = rule_from_checkpoint(blob, device)
        label = f"{args.checkpoint} at update {blob.get('update', '?')}"
        raw = dict(blob.get("args") or {})
        missing: list[str] = []
        for attribute, key in (
            ("split", "split"),
            ("pool", "structures"),
            ("eval_structures", "eval_structures"),
            ("seed", "seed"),
            ("crop_size", "crop_size"),
            ("initial", "initial"),
            ("damage", "damage"),
        ):
            if key not in raw:
                missing.append(key)
                continue
            value = raw[key]
            was = getattr(args, attribute)
            setattr(args, attribute, value)
            note = "" if was == value else f" (CLI/default was {was!r})"
            print(f"  from checkpoint args: {attribute} = {value!r}{note}")
        recorded_indices = [int(i) for i in (blob.get("eval_indices") or [])]
        if not recorded_indices:
            missing.append("eval_indices")
        if missing:
            print()
            print("  !! WARNING: checkpoint is MISSING "
                  f"{', '.join(sorted(set(missing)))}")
            print("  !! Those values come from this script's own "
                  "CLI/defaults instead.")
            if "eval_indices" in missing:
                print("  !! THE CROPS ARE A RESHUFFLE, NOT THE RUN'S HELD-OUT "
                      "SET. This table is not comparable to that run.")
            print("  !! Checkpoints written by scripts/train_growth.py carry "
                  "all of these; one that does not is from another trainer.")
        initial = args.initial
        config = dataclasses.replace(config, steps=args.steps)
        print()
    else:
        import train_growth as tg
        fallback = tg.build_schema().parse_args([])
        fallback.initial = args.initial
        fallback.steps = args.steps
        fallback.birth = "none"
        rule = AANCARule(growth=True, birth="none").to(device)
        config = tg.build_config(fallback, fallback.eval_activation)
        label = ("UNTRAINED CONTROL -- the `model` row is an untrained rule "
                 "whose move heads emit ~0, so it reads as the start state. "
                 "Wiring only; it says nothing about learning")
        initial = args.initial
    rule.eval()

    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[args.split]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = args.pool + args.eval_structures
    pool = available[:wanted] if len(available) >= wanted else available
    if not pool:
        print("no structures on disk for that split")
        return 1
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size, device=device,
        cache_limit=max(len(pool), 1),
    )
    if recorded_indices:
        indices = recorded_indices[: args.structures]
        provenance = "checkpoint eval_indices"
    else:
        order = list(range(len(pool)))
        random.Random(args.seed).shuffle(order)
        indices = order[: args.eval_structures][: args.structures]
        provenance = "reshuffle of this script's own pool slice"

    print(f"checkpoint : {label}")
    print(f"{len(pool)} structures on disk ({len(available)} available), "
          f"{len(indices)} crops at crop {args.crop_size}, window "
          f"{args.damage}, start {initial!r}, model steps {config.steps}")
    in_range = max(indices, default=-1) < len(pool)
    ids = [pool[i].pdb_id for i in indices] if in_range else "<out of range>"
    print(f"crops: {provenance}; indices {list(indices)}; ids {ids}")

    # **Assert the recorded pdb ids.** `eval_indices` are positions in a pool
    # whose size differs between machines -- hundreds on the laptop against
    # tens of thousands on Bunya -- so the same integer names a different
    # protein and that cannot be detected from the indices alone. Checkpoints
    # written before 21 Nov carry no `eval_pdb_ids`; for those this says the
    # ids are unverified rather than pretending to have checked.
    if blob is not None:
        recorded_ids = list(blob.get("eval_pdb_ids") or [])
        if not recorded_ids:
            print("  !! this checkpoint records no eval_pdb_ids (written "
                  "before 21 Nov), so the crop IDS are unverified -- the "
                  "indices resolve against THIS pool, which is only the run's "
                  "pool if this is the same machine and the same data dir")
        else:
            here = [pool[i].pdb_id for i in indices] if in_range else []
            want = recorded_ids[: len(here)]
            if here != want:
                print(f"  !! POOL MISMATCH: the checkpoint's eval_indices name "
                      f"{want} but resolve here to {here}. This pool is not the "
                      f"run's pool, so the table would be measured on different "
                      f"proteins. Refusing to measure.")
                return 1
            print(f"  eval_pdb_ids verified against the checkpoint: {here}")
    if not in_range:
        print("  !! WARNING: an index is past the end of this pool "
              f"({len(pool)} entries). The checkpoint was trained where more "
              f"structures were on disk; those crops CANNOT be measured here "
              f"and the comparison is void. Run this where the run ran.")
        return 1

    print()
    print(f"basins (degrees), from Lovell et al. 2003 favoured regions as "
          f"boxed in ramachandran_clash_probe.py:")
    for name, boxes in BASINS.items():
        print(f"  {name:>6}  {box_text(boxes)}")
    print(f"sensitivity: the same table at {-args.margin:+.0f}, 0 and "
          f"{args.margin:+.0f} degrees on every box edge")
    print("excluded: any window residue whose phi OR psi is undefined "
          "(measure_backbone_torsions returns 0 there, and 0 is a real angle)")
    print()

    # -- one pass over the crops, collecting torsions for all four rows ----
    #
    # **ONE rng advanced across the crops**, which is `train_growth.score`'s
    # convention and `geometric_floor.py`'s: the rng is what `crop()` draws the
    # seed residue from, so a freshly seeded stream per crop puts the window in
    # a different spatial ball of the same protein than every recorded number
    # was measured on.
    rng = random.Random(args.seed)
    per_crop: list[dict] = []
    bonds: list[float] = []
    clashes: list[float] = []
    skipped = 0
    for index in indices:
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            skipped += 1
            continue
        if topo.n_residues < args.damage:
            skipped += 1
            continue
        links = growth_links(topo)
        alive = torch.ones(topo.n_residues, dtype=torch.bool, device=device)
        generator = torch.Generator().manual_seed(args.seed + index)
        damaged = damage_window(
            topo.n_residues, links, args.damage, generator, device
        )
        if not bool(damaged.any()):
            skipped += 1
            continue

        scored = damaged & defined_mask(links, topo.n_residues)
        n_window = int(damaged.sum())
        n_scored = int(scored.sum())
        if n_scored == 0:
            skipped += 1
            continue

        start = baseline_coords(
            native, topo, links, ~damaged, initial, generator
        )
        floor, bond, clash = relax_to_geometry(
            start, topo, links, alive, damaged,
            iters=args.floor_iters, lr=args.floor_lr,
            tolerance=args.clash_tolerance, clash_weight=args.clash_weight,
        )
        bonds.append(bond)
        clashes.append(clash)
        with torch.no_grad():
            begin = initial_state(start, rule, alive=alive)
            result = rollout(
                rule, begin, topo, config,
                generator=torch.Generator().manual_seed(args.seed + index),
                record=False,
            )
            model = result.state.coords

        angles = {}
        for name, coords in (("native", native), ("start", start),
                             ("floor", floor), ("model", model)):
            phi, psi, _omega = measure_backbone_torsions(coords, topo, links)
            phi, psi = degrees(phi, psi)
            angles[name] = (phi[scored].tolist(), psi[scored].tolist())
        per_crop.append({
            "id": pool[index].pdb_id,
            "window": n_window,
            "scored": n_scored,
            "angles": angles,
        })
        if not args.quiet_crops:
            print(f"  {pool[index].pdb_id:>6}  window {n_window:>3} residues, "
                  f"{n_scored:>3} with phi and psi both defined, "
                  f"{n_window - n_scored:>2} excluded")

    if not per_crop:
        print("no crop produced a scorable window; nothing measured")
        return 1
    if skipped:
        print(f"  ({skipped} crops skipped: failed to load, too short, or no "
              f"window -- expected on this pool)")
    print()

    # -- the table, at each widening -------------------------------------
    verdicts: dict[str, list[tuple[float, float]]] = {"helix": [], "sheet": []}
    #: ``cross[row][native basin][destination]`` -> one ``(hit, total)`` per
    #: widening, in sweep order. ``DESTINATIONS`` is exclusive because the
    #: boxes do not overlap in psi; the first match wins if that ever changes.
    cross: dict = {
        row: {
            basin: {dest: [] for dest in DESTINATIONS}
            for basin in ("helix", "sheet")
        }
        for row in ROWS
    }
    native_ok = True
    for margin in (-args.margin, 0.0, args.margin):
        boxes = {n: widen(b, margin) for n, b in BASINS.items()}
        # The FAVOURED regions that are not a basin here, widened by the SAME
        # offset: a sweep that moved two boxes and left the third fixed would
        # shuffle residues between `other_fav` and `disallowed` for no reason
        # but the sweep itself.
        other_boxes = widen(OTHER_FAVOURED, margin)
        print(f"=== boxes {margin:+.0f} degrees " + "=" * 44)
        for name, b in boxes.items():
            print(f"    {name:>6}  {box_text(b)}")
        # Classify native once per widening: the residue SET moves with the
        # boxes, which is what keeps the native row a tautology and therefore
        # a check on the wiring at every setting.
        counts = {row: {"helix": [0, 0], "sheet": [0, 0]} for row in ROWS}
        placed = {
            row: {basin: dict.fromkeys(DESTINATIONS, 0)
                  for basin in ("helix", "sheet")}
            for row in ROWS
        }
        for crop in per_crop:
            n_phi, n_psi = crop["angles"]["native"]
            for position in range(len(n_phi)):
                for basin in ("helix", "sheet"):
                    if not in_boxes(n_phi[position], n_psi[position],
                                    boxes[basin]):
                        continue
                    for row in ROWS:
                        phi, psi = crop["angles"][row]
                        counts[row][basin][1] += 1
                        if in_boxes(phi[position], psi[position],
                                    boxes[basin]):
                            counts[row][basin][0] += 1
                        for where in ("helix", "sheet"):
                            if in_boxes(phi[position], psi[position],
                                        boxes[where]):
                                break
                        else:
                            where = ("other_fav" if in_boxes(
                                phi[position], psi[position], other_boxes
                            ) else "disallowed")
                        placed[row][basin][where] += 1
        print()
        print(f"    {'row':>8}  {'helix in helix basin':>28}  "
              f"{'sheet in sheet basin':>28}")
        for row in ROWS:
            cells = []
            for basin in ("helix", "sheet"):
                hit, total = counts[row][basin]
                frac = hit / total if total else float("nan")
                cells.append(f"{frac:5.3f}  ({hit:>4}/{total:<4})")
            print(f"    {row:>8}  {cells[0]:>28}  {cells[1]:>28}")

        # -- the cross-tabulation: not "did it hit its own basin" but
        # "where did it put them". Same residues, same denominators as
        # the two columns above; the three destinations are exclusive.
        print()
        print("    cross-tabulation -- where each row PUT the residues of a "
              "native class. Destinations are exclusive:")
        print(f"    other favoured = in no basin box but inside another "
              f"FAVOURED region of ramachandran_clash_probe")
        print(f"    ({len(OTHER_FAVOURED)} box(es), alpha-L: "
              f"{box_text(other_boxes)}); disallowed = in no favoured box at "
              f"all")
        print("    (a) a natively-SHEET residue in the HELIX box is one "
              "conformational mode over-applied; in OTHER")
        print("        FAVOURED it is a real conformation but the wrong one; "
              "only DISALLOWED is off the plot.")
        print("    (b) if the FLOOR puts natively-HELICAL residues in the "
              "SHEET box too, the sheet box is")
        print("        catching what chain closure does to an extended chain, "
              "not a basin the floor found.")
        print("    the native class is re-derived at THIS widening, so the "
              "denominator below is this widening's own:")
        print("    the three widenings are DIFFERENT RESIDUE SETS and their "
              "columns are not a trend over one population.")
        for basin in ("helix", "sheet"):
            print()
            n_native = counts["native"][basin][1]
            header = f"native {basin.upper()} (n={n_native}) ->"
            print(f"      {header:>24}  "
                  + "  ".join(f"{DESTINATION_TEXT[d]:>18}"
                              for d in DESTINATIONS))
            for row in ROWS:
                total = counts[row][basin][1]
                cells = []
                for dest in DESTINATIONS:
                    hit = placed[row][basin][dest]
                    frac = hit / total if total else float("nan")
                    cells.append(f"{frac:5.3f}  ({hit:>4}/{total:<4})")
                    cross[row][basin][dest].append((hit, total))
                print(f"      {row:>24}  "
                      + "  ".join(f"{cell:>18}" for cell in cells))
        # The native row's cross-tab is the same tautology as the table
        # above, seen from the side: a natively-helical residue is one the
        # classifier put in the helix box, so it must cross-tabulate 1.000
        # into that box and 0.000 into the other three.
        for basin in ("helix", "sheet"):
            total = counts["native"][basin][1]
            if total and placed["native"][basin][basin] != total:
                native_ok = False

        for basin in ("helix", "sheet"):
            floor_hit, floor_total = counts["floor"][basin]
            model_hit, model_total = counts["model"][basin]
            verdicts[basin].append((
                model_hit / model_total if model_total else float("nan"),
                floor_hit / floor_total if floor_total else float("nan"),
            ))
        for basin in ("helix", "sheet"):
            hit, total = counts["native"][basin]
            if total and hit != total:
                native_ok = False
        print()

    print(f"floor row achieved bond "
          f"{sum(bonds) / len(bonds):.4f}, clash "
          f"{sum(clashes) / len(clashes):.4f} "
          f"(Adam lr {args.floor_lr}, {args.floor_iters} iters). If bond is "
          f"not small the chain did not close and the floor row is not a "
          f"floor.")
    print()

    # -- closing lines ----------------------------------------------------
    if not native_ok:
        print("!! THE NATIVE ROW IS NOT 1.000. The residues counted as "
              "native-helical are by construction those the classifier calls")
        print("!! helical on native coordinates, so this row can only fail if "
              "the torsion measurement, the degree conversion or the")
        print("!! defined-torsion masking is wrong. NOTHING ELSE IN THE TABLE "
              "MEANS ANYTHING until that is fixed.")
        return 1
    print("The native row reads 1.000 on both basins, which is what it must: "
          "the classifier and the torsion measurement agree with")
    print("themselves. That is a wiring check, not a result.")
    print()
    print("Read `model` against `floor`, not against `start`. `floor` had no "
          "network and no native information -- only bond closure and")
    print("sterics -- so whatever fraction it reaches is the fraction that "
          "geometry alone forces, and only the excess over it can be")
    print("learning.")
    print()
    for basin in ("helix", "sheet"):
        pairs = [
            (model, floor) for model, floor in verdicts[basin]
            if model == model and floor == floor
        ]
        above = [1 for model, floor in pairs if model > floor]
        below = [1 for model, floor in pairs if model < floor]
        values = ", ".join(
            f"{model:.3f} vs {floor:.3f}" for model, floor in verdicts[basin]
        )
        # A verdict is only unstable if the comparison CROSSES: strictly above
        # at one widening and strictly below at another. "below, equal, below"
        # is not a flip -- the claim "the model beats the floor" is false at
        # every setting -- and calling it one would cry wolf on the one signal
        # this sweep exists to raise.
        if not pairs:
            print(f"  {basin:>6}: no residues in this basin at any widening; "
                  f"nothing to conclude")
        elif above and below:
            print(f"  {basin:>6}: the model/floor comparison CROSSES the floor "
                  f"across the widenings ({values}).")
            print(f"  {'':>6}  That is not a conclusion. It is a statement "
                  f"about where the box edge was drawn, and the basin")
            print(f"  {'':>6}  definition would have to be replaced with "
                  f"contours before anything is quoted from it.")
        elif not above and not below:
            print(f"  {basin:>6}: model equals the geometric floor at all "
                  f"three widenings ({values}) -- STABLE, and no excess")
        elif above and len(above) == len(pairs):
            print(f"  {basin:>6}: model is above the geometric floor at all "
                  f"three widenings ({values}) -- STABLE")
        elif below and len(below) == len(pairs):
            print(f"  {basin:>6}: model is below the geometric floor at all "
                  f"three widenings ({values}) -- STABLE, and that is the "
                  f"geometry winning")
        elif above:
            print(f"  {basin:>6}: model is at or above the geometric floor, "
                  f"above it at {len(above)} of {len(pairs)} widenings "
                  f"({values}) -- it is never beaten by geometry, but the "
                  f"excess is not present at every box")
        else:
            print(f"  {basin:>6}: model is at or below the geometric floor, "
                  f"below it at {len(below)} of {len(pairs)} widenings "
                  f"({values}) -- STABLE in the only direction that matters: "
                  f"it never exceeds the floor, so there is no excess to "
                  f"call learning")
    print()
    mean_bond = sum(bonds) / len(bonds)
    print("WHERE THE MISSES GO. The two columns above say how often a row hit "
          "the basin its crystal conformation")
    print("occupies. The cross-tabulation says where the rest went, and that "
          "is a different question with a")
    print("different answer. Fractions below are the sweep, "
          f"{-args.margin:+.0f}, 0, {args.margin:+.0f} degrees, in that order.")
    print()
    print("The denominators differ between the three because the native class "
          "is re-derived at each widening; the")
    print("columns are three different residue sets and only the dominant "
          "destination, not its value, is read here.")
    print()
    print("(a) A natively-SHEET residue the row did not put in the sheet box "
          "went to one of three places, and")
    print("    they do not mean the same thing -- HELIX is one mode "
          "over-applied, OTHER FAVOURED is a real")
    print("    conformation but the wrong one, and only DISALLOWED is in no "
          "favoured region of the plot at all:")
    for row in ("start", "floor", "model"):
        trios = {
            dest: cross[row]["sheet"][dest]
            for dest in ("helix", "other_fav", "disallowed")
        }
        print(f"  {row:>6}: HELIX box {share_trio(trios['helix'])}; "
              f"OTHER FAVOURED {share_trio(trios['other_fav'])}; "
              f"DISALLOWED {share_trio(trios['disallowed'])}")
        values = {d: share_values(c) for d, c in trios.items()}
        widenings = list(zip(values["helix"], values["other_fav"],
                             values["disallowed"]))
        if not widenings:
            print(f"  {'':>6}  no natively-sheet residues at any widening; "
                  f"nothing to read")
        elif all(d > h and d > o for h, o, d in widenings):
            print(f"  {'':>6}  DISALLOWED dominates at every widening: this "
                  f"row is not over-applying a helical mode and it is")
            print(f"  {'':>6}  not reaching another real conformation either, "
                  f"it is putting these residues where no protein")
            print(f"  {'':>6}  puts any residue. That is the one case that "
                  f"supports 'nothing local was learned' -- but these")
            print(f"  {'':>6}  are rectangles, so a residue a few degrees "
                  f"outside an edge counts here too.")
        elif all(h > o and h > d for h, o, d in widenings):
            print(f"  {'':>6}  the HELIX box dominates at every widening -- "
                  f"one conformational mode, over-applied to residues")
            print(f"  {'':>6}  whose crystal conformation is the other one. "
                  f"That is a learned mode used wrongly, and it is")
            print(f"  {'':>6}  the opposite of an absence of local structure.")
        elif all(o > h and o > d for h, o, d in widenings):
            print(f"  {'':>6}  OTHER FAVOURED dominates at every widening: "
                  f"these residues are in a populated region of the")
            print(f"  {'':>6}  plot, the wrong one. The row produced real "
                  f"protein conformations and chose badly among them;")
            print(f"  {'':>6}  nothing here says it is off the plot.")
        else:
            print(f"  {'':>6}  which destination dominates CHANGES across the "
                  f"widenings, so it is a statement about where the")
            print(f"  {'':>6}  box edge was drawn and not about the row. "
                  f"Nothing is quotable from it.")
        if row == "floor" and mean_bond > FLOOR_BOND_UNREADABLE:
            print(f"  {'':>6}  (and the floor's achieved bond is "
                  f"{mean_bond:.4f}: the chain did not close, so this row is "
                  f"not a floor and")
            print(f"  {'':>6}   the line above describes an unconverged "
                  f"optimiser, not geometry.)")
    print()
    print("(b) Where the FLOOR puts natively-HELICAL residues -- the question "
          "is whether its sheet number is")
    print("    a basin it found or just the open, extended part of the plot "
          "that chain closure lands on anyway:")
    floor_h2h = cross["floor"]["helix"]["helix"]
    floor_h2s = cross["floor"]["helix"]["sheet"]
    floor_h2o = cross["floor"]["helix"]["other_fav"]
    floor_h2d = cross["floor"]["helix"]["disallowed"]
    floor_s2s = cross["floor"]["sheet"]["sheet"]
    print(f"  {'floor':>6}: natively-HELICAL residues -> helix box "
          f"{share_trio(floor_h2h)}; sheet box {share_trio(floor_h2s)}; "
          f"other favoured {share_trio(floor_h2o)}; disallowed "
          f"{share_trio(floor_h2d)}")
    print(f"  {'':>6}  for comparison, the floor on natively-SHEET residues "
          f"-> sheet box {share_trio(floor_s2s)}")
    h2s_v = share_values(floor_h2s)
    s2s_v = share_values(floor_s2s)
    pairs = list(zip(h2s_v, s2s_v))
    # **The refusal.** (b) is a difference between two small fractions of a
    # few dozen residues, and if the floor's optimiser never closed the chain
    # then both fractions are properties of an unconverged Adam run and the
    # difference is noise. A one-residue gap stated as "it discriminates by
    # native conformation" is worse than no reading at all, so there is no
    # verdict here until `bond` says there is a floor to read.
    if mean_bond > FLOOR_BOND_UNREADABLE:
        print(f"  {'':>6}  !! NOT READABLE: the floor's achieved bond is "
              f"{mean_bond:.4f}, above the {FLOOR_BOND_UNREADABLE:.2f} at "
              f"which this")
        print(f"  {'':>6}  !! script will read (b) at all. The optimiser did "
              f"not close the chain, so the `floor` row is not a")
        print(f"  {'':>6}  !! floor and the two fractions above are properties "
              f"of an unconverged Adam run, not of geometry.")
        print(f"  {'':>6}  !! The difference between them would be noise and "
              f"NO VERDICT IS STATED. Raise --floor-iters or")
        print(f"  {'':>6}  !! --floor-lr and run it again.")
    elif not pairs:
        print(f"  {'':>6}  one of the two native classes is empty at every "
              f"widening; nothing to compare")
    elif all(h >= s for h, s in pairs):
        print(f"  {'':>6}  the floor puts natively-helical residues in the "
              f"SHEET box AT LEAST AS OFTEN as it puts natively-sheet")
        print(f"  {'':>6}  ones there, at every widening. The sheet box is "
              f"then catching what bond closure and sterics do to an")
        print(f"  {'':>6}  extended chain regardless of what the crystal "
              f"did, so the floor's sheet number is not evidence it")
        print(f"  {'':>6}  found the basin -- and 'the model is below the "
              f"floor on sheet' is partly the floor being trivially")
        print(f"  {'':>6}  good there rather than the rule being bad.")
    elif all(s > h for h, s in pairs):
        print(f"  {'':>6}  the floor puts natively-sheet residues in the "
              f"sheet box MORE often than natively-helical ones, at every")
        print(f"  {'':>6}  widening, so it does discriminate by native "
              f"conformation and its sheet number is not explained away")
        print(f"  {'':>6}  as chain closure landing on the extended region. "
              f"The sheet comparison stands as read above.")
    else:
        print(f"  {'':>6}  the comparison CROSSES across the widenings, so "
              f"whether the floor discriminates on sheet is a statement")
        print(f"  {'':>6}  about the box edge. Neither reading is quotable "
              f"until the boxes are replaced with contours.")
    print()
    print("Same discipline as the table: these are rectangles where the real "
          "basins are contours, so only the")
    print("DIFFERENCES between rows are quotable and no absolute fraction "
          "here is a share of a Ramachandran basin.")
    print()
    print("Check the denominators before quoting any of it: a fraction over a "
          "handful of residues is arithmetic, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
