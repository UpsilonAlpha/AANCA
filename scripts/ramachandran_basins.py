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
    native_ok = True
    for margin in (-args.margin, 0.0, args.margin):
        boxes = {n: widen(b, margin) for n, b in BASINS.items()}
        print(f"=== boxes {margin:+.0f} degrees " + "=" * 44)
        for name, b in boxes.items():
            print(f"    {name:>6}  {box_text(b)}")
        # Classify native once per widening: the residue SET moves with the
        # boxes, which is what keeps the native row a tautology and therefore
        # a check on the wiring at every setting.
        counts = {row: {"helix": [0, 0], "sheet": [0, 0]} for row in ROWS}
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
    print("Check the denominators before quoting any of it: a fraction over a "
          "handful of residues is arithmetic, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
