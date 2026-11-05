"""Is the fold wrong, or are the pieces right and arranged wrongly?

`scripts/lever_arm_probe.py` calibrated the per-birth orientation error that a
13 A structure would need if growth's local accuracy were the only thing wrong:
about **17 degrees**. That is far worse than any other local inference this rule
makes -- sheet parity separates at 96.1 % from one scalar, packing reaches 36.7
degrees of chi1 -- so the arithmetic does not support the explanation.

The alternative is that each grown piece is locally fine and the pieces are
globally misplaced relative to each other. Growth runs from sixteen seeds per
crop, a crop holds about nine covalent runs, and **nothing in local growth
constrains where a whole segment sits**: a segment can be rotated and
translated bodily with no local cost at all -- every bond, every angle and
every contact inside it is unchanged, and the residues that would object are in
a different segment, outside the 10 A cutoff, and therefore not even edges.

So this measures the same structure twice:

* **global** -- one superposition over every alive residue, which is what the
  training log's `rmsd` column reports.
* **per segment** -- each covalent run of grown residues superposed on its own
  native counterpart, independently. A segment that is internally correct
  scores low here however badly it is placed.

If per-segment error is 2-3 A while global is 13 A, the deficit is in the
arrangement of segments and neither free placement nor a chain field touches
it: both are about a residue's relation to its own neighbours, and a bodily
displaced segment has all of those right. If the two numbers are close, growth
really is locally wrong and the per-birth arithmetic above needs explaining.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aanca import checkpoint as ckpt  # noqa: E402
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.fape import backbone_rmsd  # noqa: E402
from aanca.growth import (  # noqa: E402
    GrowthConfig,
    growth_links,
    growth_objective,
    relative_frame_restraint,
)
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import RolloutConfig  # noqa: E402

PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"


def alive_runs(alive: torch.Tensor, links: torch.Tensor) -> list[list[int]]:
    """Maximal stretches of alive residues joined by covalent bonds.

    Not merely contiguous indices: a crop is a spatial ball, so two residues
    adjacent in index may be in different chains with no bond between them,
    and treating them as one segment would superpose two independently placed
    pieces together and report their relative error as internal error.
    """
    runs: list[list[int]] = []
    current: list[int] = []
    for i in range(alive.shape[0]):
        if not bool(alive[i]):
            if current:
                runs.append(current)
                current = []
            continue
        joined = current and i > 0 and bool(links[i - 1])
        if joined:
            current.append(i)
        else:
            if current:
                runs.append(current)
            current = [i]
    if current:
        runs.append(current)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--min-segment", type=int, default=4,
                        help="runs shorter than this cannot be superposed meaningfully")
    parser.add_argument(
        "--windows", default="4,8,16,32",
        help=(
            "also report RMSD over every contiguous covalent window of these "
            "lengths, superposed on its own native counterpart: the error as "
            "a function of the length it is measured over"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = ckpt.load(Path(args.checkpoint), map_location="cpu")
    stored = state.get("args") or {}
    weights = ckpt.rule_weights(state)

    # **Which heads the rule has is read from the weights, not from the stored
    # arguments.** g4's checkpoint carries an identity head while its stored
    # ``identity_weight`` reads 0, so trusting the arguments built a rule
    # without one and ``load_state_dict`` refused. The weights cannot
    # disagree with themselves.
    has_identity = "identity_head.0.weight" in weights
    has_field = "emit_head.0.weight" in weights
    has_birth_head = "birth_head.0.weight" in weights
    birth = str(stored.get("birth") or "cartesian")
    if not has_birth_head:
        birth = "free"
    elif birth == "free":
        # The stored argument says free and the weights say otherwise; the
        # weights win, and cartesian is the only other mode a growth run here
        # has used.
        birth = "cartesian"
    identity_weight = float(stored.get("identity_weight", 0.0) or 0.0)
    rule = AANCARule(
        growth=True,
        birth=birth,
        morphogen=has_field,
        activation_init=float(stored.get("activation_init", -2.0) or -2.0),
        identity=has_identity,
        identity_keep=float(stored.get("identity_keep", 2.0) or 2.0),
    ).to(device)
    rule.load_state_dict(weights)
    rule.eval()
    stored = dict(stored)
    stored["birth"] = birth

    # The identity *track* is on whenever the head exists: a blind rollout is
    # what the head was trained under, and running it sighted would feed it an
    # input distribution it never saw.
    if has_identity and not identity_weight:
        identity_weight = 93.7
    print(f"{args.checkpoint}: birth {birth}, field {has_field}, "
          f"identity head {has_identity} (weight {identity_weight})")

    # **The held-out structures, read from the checkpoint rather than chosen
    # here.** The trainer stores its own eval indices precisely so a later
    # probe cannot accidentally report training-set numbers; regenerating the
    # split from --seed would silently do that if any argument differed.
    # The same redirection ``on_disk`` does in the trainer: the manifest
    # records a .cif path while what is downloaded is .bcif, so filtering
    # without redirecting would drop every entry.
    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[args.split]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = int(stored.get("structures", 400)) + int(stored.get("eval_structures", 12))
    pool = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=args.crop_size, device=device,
        cache_limit=max(len(pool), 1),
    )
    indices = state.get("eval_indices")
    if indices:
        print(f"using the checkpoint's own {len(indices)} held-out indices")
    else:
        order = list(range(len(pool)))
        random.Random(int(stored.get("seed", 0))).shuffle(order)
        indices = order[: int(stored.get("eval_structures", 12))]
        print(f"checkpoint stored no eval indices; regenerated {len(indices)} "
              f"from its seed {stored.get('seed', 0)}")
    indices = list(indices)[: args.structures]

    config = RolloutConfig(
        steps=args.steps, truncate=99, k=int(stored.get("k", 32)),
        blind=bool(identity_weight),
        growth=GrowthConfig(
            seeds=args.seeds,
            birth=str(stored.get("birth", "cartesian")),
            placement_steps=int(stored.get("placement_steps", 0) or 0),
        ),
    )

    print(f"\n{'entry':>6} {'alive':>6} {'global':>9} {'per-seg':>9} "
          f"{'segs':>5} {'median':>7} {'largest':>8} {'twist':>7}")
    rows = []
    windows = [int(w) for w in args.windows.split(",") if w] if args.windows else []
    by_window: dict[int, list[float]] = {}
    rng = random.Random(args.seed)
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 12:
                continue
            _terms, result = growth_objective(
                rule, native, topo, config,
                generator=torch.Generator().manual_seed(args.seed + index),
                identity_weight=identity_weight,
            )
            alive = result.state.alive
            links = growth_links(topo)
            runs = [r for r in alive_runs(alive, links) if len(r) >= args.min_segment]
            if not runs:
                continue
            glob = float(backbone_rmsd(result.state.coords, native, topo, alive))
            # The per-bond rotation error, measured rather than inferred.
            # `lever_arm_probe.py` puts the error that would reproduce the
            # RMSD at every length scale at 15-17 degrees; this is the same
            # quantity read directly off the structure, so the inference can
            # be checked instead of trusted.
            _loss, twist = relative_frame_restraint(
                result.state.coords, native, topo, alive, links
            )
            # Weighted by residue count, so one four-residue run cannot
            # dominate a hundred-residue one. Squared then rooted, because
            # RMSDs do not average linearly.
            total_sq, total_n = 0.0, 0
            sizes = []
            for run in runs:
                mask = torch.zeros_like(alive)
                mask[torch.tensor(run, device=alive.device)] = True
                value = float(backbone_rmsd(result.state.coords, native, topo, mask))
                if value != value:  # nan from a run of fewer than three
                    continue
                total_sq += value * value * len(run)
                total_n += len(run)
                sizes.append(len(run))
            if not total_n:
                continue
            per_segment = (total_sq / total_n) ** 0.5
            # Every contiguous covalent window of each requested length,
            # superposed on its own native counterpart. Within a run only, so
            # a window never spans a chain break -- the same reason
            # ``alive_runs`` tests ``links`` rather than index adjacency.
            for length in windows:
                for run in runs:
                    for start in range(0, len(run) - length + 1):
                        piece = run[start : start + length]
                        mask = torch.zeros_like(alive)
                        mask[torch.tensor(piece, device=alive.device)] = True
                        value = float(
                            backbone_rmsd(result.state.coords, native, topo, mask)
                        )
                        if value == value:
                            by_window.setdefault(length, []).append(value)
            sizes.sort()
            rows.append((glob, per_segment, len(sizes), sizes[len(sizes) // 2],
                         sizes[-1], float(alive.float().mean()), twist))
            print(f"{index:>6} {rows[-1][5] * 100:>5.1f}% {glob:>7.2f} A "
                  f"{per_segment:>7.2f} A {len(sizes):>5} "
                  f"{sizes[len(sizes) // 2]:>7} {sizes[-1]:>8} "
                  f"{twist:>7.1f}")

    if not rows:
        print("nothing grew far enough to measure")
        return 1
    n = len(rows)
    glob = sum(r[0] for r in rows) / n
    seg = sum(r[1] for r in rows) / n
    print(f"\nover {n} held-out crops: global {glob:.2f} A, per segment "
          f"{seg:.2f} A, ratio {glob / max(seg, 1e-9):.2f}x")
    print(f"segments per crop {sum(r[2] for r in rows) / n:.1f}, "
          f"median length {sum(r[3] for r in rows) / n:.1f}, "
          f"largest {sum(r[4] for r in rows) / n:.1f}, "
          f"alive {100 * sum(r[5] for r in rows) / n:.1f}%")
    print(f"per-bond rotation error {sum(r[6] for r in rows) / n:.1f} degrees "
          "(exactly 0 on a crystal structure, about 75 on an untrained rule)")

    if by_window:
        # **The error as a function of the length it is measured over**, which
        # is what separates "every residue is slightly misplaced" from "the
        # chain drifts". A local error that does not accumulate gives a flat
        # per-residue column; drift gives one rising with length, and how fast
        # it rises is the drift rate.
        print(f"\n{'window':>7} {'rmsd':>8} {'per residue':>12} {'n':>7}")

        def rms(values):
            return (sum(v * v for v in values) / len(values)) ** 0.5

        for length in sorted(by_window):
            value = rms(by_window[length])
            print(f"{length:>7} {value:>6.2f} A {value / length:>10.3f} A "
                  f"{len(by_window[length]):>7}")
        shortest, longest = min(by_window), max(by_window)
        growth_ratio = rms(by_window[longest]) / max(rms(by_window[shortest]), 1e-9)
        length_ratio = longest / shortest
        print(
            f"\n{longest} residues carry {growth_ratio:.1f}x the error of "
            f"{shortest} over {length_ratio:.0f}x the length: "
            + (
                "steeper than linear, so error accumulates along the chain"
                if growth_ratio > length_ratio
                else "shallower than linear, so most of the error is local"
            )
        )

    print(
        "\nA large ratio means the pieces are better than the whole, and the "
        "deficit is in\nhow segments are placed relative to one another -- "
        "which a bodily rotation of a\nsegment costs nothing locally, since "
        "every bond and contact inside it is intact\nand the residues that "
        "would object are not even edges. A ratio near one means\ngrowth is "
        "locally wrong and the per-birth arithmetic in "
        "scripts/lever_arm_probe.py\nneeds explaining."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
