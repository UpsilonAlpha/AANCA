"""How far does one orientation error travel, and how much of it survives?

The lever arm is the reason free placement exists, and it was asserted twice in
this project before anyone measured it -- once as "Cartesian birth has no lever
arm anywhere" and once as "free placement makes learned error additive". Both
were wrong. This measures the thing instead.

Two quantities, because the argument has two halves:

**Propagation.** Inject a rotation of ``theta`` into one learned output at one
step, and measure how far each later residue moves. ``rot_child = rot_parent @
rot_reference @ rot_spin`` composes, so a residue ``m`` births downstream should
be displaced by roughly ``m * 3.804 * theta`` -- the arc a lever of that length
sweeps. At initialisation that prediction is exact and checkable, which is what
makes this probe trustworthy before it is interesting: run it on an untrained
Cartesian rule and the slope should come out near ``3.804 * theta`` per residue
with nothing else moving.

**Persistence.** The same injection, measured at the *end* of the rollout
rather than immediately. This is the half that distinguishes the two
architectures, and the propagation measurement alone does not. A birth rotation
is applied once and never revisited: nothing in the rollout re-decides where a
newborn was placed. A frame-head rotation is re-decided on every step the
residue ticks, from that residue's own neighbourhood. So the question is not
whether an orientation error propagates -- it does, under both modes, because
the reference pose is still ``rot_parent @ rot_reference`` -- but whether the
rule can see it afterwards and take it back.

**Persistence needs trained weights to mean anything.** Every head's output
layer is zero-initialised, so an untrained rule corrects nothing and persistence
is 100 % by construction. Pass ``--checkpoint`` with a trained ``best.pt``; the
untrained number is reported anyway, as the no-correction reference the trained
one is read against.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from aanca.backbone import backbone_atoms  # noqa: E402
from aanca import checkpoint as ckpt  # noqa: E402
from aanca.growth import GrowthConfig, growth_links  # noqa: E402
from aanca.model import AANCARule  # noqa: E402
from aanca.rollout import RolloutConfig, initial_state, rollout  # noqa: E402
from aanca.structure import load_protein  # noqa: E402
from aanca.torch_graph import feature_topology  # noqa: E402


def _perturbing_every(rule, mode: str, theta: float, seed: int):
    """Rotate one output by ``theta`` about a fresh random axis on every step.

    The single-injection measurement above answers "how far does one error
    travel", and the answer -- 0.43 A at twenty-eight residues for 0.05 rad --
    sounds reassuring until you notice that a rollout makes a hundred births
    and each one gets its own error. This is the quantity that actually matters:
    **what per-birth orientation accuracy does a given RMSD correspond to?**

    Random axes rather than a fixed one, because a fixed axis is a systematic
    bias and an untrained head's error is not: independent errors accumulate as
    the square root of their number while a bias accumulates linearly, and
    taking the pessimistic case here would overstate how precise a birth has
    to be.
    """
    original = rule.growth_outputs
    generator = torch.Generator().manual_seed(seed)

    def wrapped(node, hidden, rotation_limit=None, translation_limit=None):
        activation, birth, axis_angle, translation = original(
            node, hidden, rotation_limit, translation_limit
        )
        if mode == "birth":
            if birth is None:
                raise SystemExit(
                    "--inject birth needs a rule with a birth head; under "
                    "birth='free' the set of learned quantities that can "
                    "rotate a newborn at its birth is empty."
                )
            noise = torch.randn(
                birth.shape[0], 3, generator=generator, dtype=birth.dtype
            )
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            birth = birth.clone()
            birth[:, 1, 3:] = birth[:, 1, 3:] + theta * noise
        else:
            noise = torch.randn(
                axis_angle.shape[0], 3, generator=generator,
                dtype=axis_angle.dtype,
            )
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            axis_angle = axis_angle + theta * noise
        return activation, birth, axis_angle, translation

    rule.growth_outputs = wrapped
    return lambda: setattr(rule, "growth_outputs", original)


def _perturbing(rule, mode: str, step: int, residue: int, theta: float):
    """Wrap ``growth_outputs`` so one output is rotated at one step.

    The wrapper reads the rollout's step off a mutable cell rather than
    counting its own calls, because ``growth_outputs`` is called once per step
    and a miscount would silently move the injection somewhere else.
    """
    original = rule.growth_outputs
    cell = {"step": 0, "injected": False}

    def wrapped(node, hidden, rotation_limit=None, translation_limit=None):
        activation, birth, axis_angle, translation = original(
            node, hidden, rotation_limit, translation_limit
        )
        if cell["step"] == step:
            axis = torch.tensor([0.0, 0.0, theta], dtype=axis_angle.dtype)
            if mode == "birth":
                if birth is None:
                    raise SystemExit(
                        "--inject birth needs a rule with a birth head; this "
                        "rule was built with birth='free', where the set of "
                        "learned quantities that can rotate a newborn at its "
                        "birth is empty. That is the architectural claim, and "
                        "it is why this mode refuses rather than returning a "
                        "zero."
                    )
                # The C-ward spin, which is what sets the child's orientation.
                birth = birth.clone()
                birth[residue, 1, 3:] = birth[residue, 1, 3:] + axis
            else:
                axis_angle = axis_angle.clone()
                axis_angle[residue] = axis_angle[residue] + axis
            cell["injected"] = True
        cell["step"] += 1
        return activation, birth, axis_angle, translation

    rule.growth_outputs = wrapped
    return cell, lambda: setattr(rule, "growth_outputs", original)


def _grow(rule, coords, topo, config, seed_at: int, generator_seed: int):
    alive = torch.zeros(topo.n_residues, dtype=torch.bool)
    alive[seed_at] = True
    return rollout(
        rule,
        initial_state(coords, rule, alive=alive),
        topo,
        config,
        torch.Generator().manual_seed(generator_seed),
        record=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default=str(ROOT / "data" / "raw" / "1UBQ.pdb"))
    parser.add_argument("--birth", choices=("torsion", "cartesian", "free"),
                        default="cartesian")
    parser.add_argument("--inject", choices=("birth", "frame"), default="birth")
    parser.add_argument("--checkpoint", default=None,
                        help="a trained best.pt; persistence is meaningless without one")
    parser.add_argument("--theta", type=float, default=0.05,
                        help="radians injected, small enough that the response is linear")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--inject-step", type=int, default=6)
    parser.add_argument("--seed-at", type=int, default=10)
    parser.add_argument("--placement-steps", type=int, default=0)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument(
        "--sweep", default=None,
        help=(
            "comma-separated thetas; rotates the chosen output by each on "
            "EVERY step instead of once, and reports the RMSD that per-birth "
            "error alone produces"
        ),
    )
    args = parser.parse_args()

    topo, coords = feature_topology(load_protein(Path(args.structure)))
    links = growth_links(topo)
    if args.birth == "free" and args.placement_steps == 0:
        args.placement_steps = 8

    trained = None
    stored: dict = {}
    if args.checkpoint:
        state = ckpt.load(Path(args.checkpoint), map_location="cpu")
        stored = state.get("args") or {}
        trained = ckpt.rule_weights(state)
        # **The checkpoint decides the architecture, not the command line.** A
        # probe that read ``--birth`` while loading free-placement weights
        # would build a birth head the checkpoint has no entry for, and
        # ``load_state_dict`` would either refuse or -- worse, if anything were
        # ever non-strict -- leave it at its initialisation and report the
        # propagation of a head the trained rule does not have.
        for key in ("birth", "morphogen", "placement_steps"):
            if key in stored and stored[key] is not None:
                setattr(args, key.replace("placement_steps", "placement_steps"),
                        stored[key])
        print(f"architecture read from the checkpoint: birth {args.birth}, "
              f"morphogen {bool(stored.get('morphogen'))}, "
              f"placement window {args.placement_steps}")

    identity_weight = float(stored.get("identity_weight", 0.0) or 0.0)

    def build():
        # A high activation bias so growth is near-deterministic and a segment
        # of usable length exists by the injection step; the probe is about
        # where residues land, not about whether they are woken. A trained
        # rule keeps its own bias, because overriding it would be overriding
        # a trained parameter.
        rule = AANCARule(
            growth=True, birth=args.birth, activation_init=2.0,
            morphogen=bool(stored.get("morphogen")),
            identity=bool(identity_weight),
            identity_keep=float(stored.get("identity_keep", 2.0)),
            max_rotation=0.1, max_translation=0.3,
        )
        if trained is not None:
            rule.load_state_dict(trained)
        rule.eval()
        return rule

    config = RolloutConfig(
        steps=args.steps, truncate=99, k=32,
        growth=GrowthConfig(
            seeds=1, cward_only=True, birth=args.birth, activation="threshold",
            placement_steps=args.placement_steps,
        ),
    )

    if args.sweep:
        # The calibration. How accurate does one birth have to be for the
        # structure to come out at a given error? Growth is seeded once and
        # grown C-ward here, so this is a *segment's* accumulation and not a
        # whole crop's -- the multi-seed case adds junctions on top.
        print(f"{args.birth} birth, {args.inject} output rotated by theta "
              f"about a fresh random axis on every step, {args.steps} steps, "
              f"{args.draws} draws\n")
        print(f"{'theta rad':>10} {'theta deg':>10} {'rmsd':>9} "
              f"{'worst':>9} {'residues':>9}")
        for theta in [float(t) for t in args.sweep.split(",")]:
            shifts, counts = [], []
            for draw in range(args.draws):
                rule = build()
                with torch.no_grad():
                    clean = _grow(rule, coords, topo, config, args.seed_at, draw)
                restore = _perturbing_every(rule, args.inject, theta, draw + 991)
                with torch.no_grad():
                    dirty = _grow(rule, coords, topo, config, args.seed_at, draw)
                restore()
                shared = clean.state.alive & dirty.state.alive
                if not int(shared.sum()):
                    continue
                ca_clean = backbone_atoms(clean.state.coords, topo)[:, 1]
                ca_dirty = backbone_atoms(dirty.state.coords, topo)[:, 1]
                shift = (ca_dirty - ca_clean)[shared].norm(dim=-1)
                # Root mean square, not mean: it is the quantity the training
                # log reports, so the two can be compared without converting.
                shifts.append(float((shift**2).mean().sqrt()))
                counts.append(int(shared.sum()))
            if not shifts:
                continue
            print(f"{theta:>10.3f} {theta * 180 / 3.14159265:>10.1f} "
                  f"{sum(shifts) / len(shifts):>7.2f} A "
                  f"{max(shifts):>7.2f} A "
                  f"{sum(counts) / len(counts):>9.1f}")
        print("\nRead against the 12.8-14.3 A that g1-g4 never left. This is "
              "the per-birth\norientation error that alone would produce that "
              "much displacement, with every\nother part of the rule exact.")
        return 0

    print(f"{args.birth} birth, injecting {args.theta} rad into the "
          f"{args.inject} output at step {args.inject_step}, "
          f"{args.steps} steps, {args.draws} draws")
    # ``m * 3.804 * theta`` is what a straight lever of that length would
    # sweep, and it is a *ceiling* rather than a prediction: a rotation about
    # the growth axis displaces nothing, so only the perpendicular component
    # tells, and a helix curls, so an arc rotated about a point does not
    # displace linearly along its own length. Measured at 11-13 % of it. The
    # signature of a lever arm is therefore not the absolute rate but the
    # per-residue rate *rising* with distance, which the last line reports.
    print(f"a straight lever of this length would sweep "
          f"{3.804 * args.theta:.4f} A per residue; that is the ceiling, not "
          "the expectation\n")

    by_distance: dict[int, list[float]] = {}
    totals: list[tuple[float, float]] = []

    for draw in range(args.draws):
        rule = build()
        with torch.no_grad():
            clean = _grow(rule, coords, topo, config, args.seed_at, draw)
        cell, restore = _perturbing(
            rule, args.inject, args.inject_step, args.seed_at, args.theta
        )
        with torch.no_grad():
            dirty = _grow(rule, coords, topo, config, args.seed_at, draw)
        restore()
        if not cell["injected"]:
            raise SystemExit(
                f"the injection never fired: the rollout ran "
                f"{cell['step']} steps and step {args.inject_step} was not "
                "among them."
            )

        shared = clean.state.alive & dirty.state.alive
        ca_clean = backbone_atoms(clean.state.coords, topo)[:, 1]
        ca_dirty = backbone_atoms(dirty.state.coords, topo)[:, 1]
        shift = (ca_dirty - ca_clean).norm(dim=-1)
        for i in torch.nonzero(shared).flatten().tolist():
            distance = i - args.seed_at
            if distance <= 0:
                continue
            by_distance.setdefault(distance, []).append(float(shift[i]))
        far = shared.clone()
        far[: args.seed_at + 6] = False
        if int(far.sum()):
            totals.append((float(shift[far].mean()), float(shift[far].max())))

    print(f"{'distance':>9} {'mean shift':>11} {'per residue':>12} {'n':>4}")
    distances = sorted(by_distance)
    for distance in distances:
        values = by_distance[distance]
        mean = sum(values) / len(values)
        print(f"{distance:>9} {mean:>9.3f} A {mean / distance:>10.4f} A "
              f"{len(values):>4}")

    if len(distances) >= 4:
        # Slope through the origin, which is the right fit: zero distance is
        # zero displacement by definition, so an intercept would only absorb
        # the curvature a lever arm does not have.
        num = sum(
            d * (sum(by_distance[d]) / len(by_distance[d])) for d in distances
        )
        den = sum(d * d for d in distances)
        slope = num / den
        print(f"\nslope {slope:.4f} A per residue, "
              f"{slope / (3.804 * args.theta):.2f} of the straight-lever "
              "ceiling")
        # The lever arm's own signature. A *constant* per-residue error would
        # accumulate linearly and this ratio would be 1; anything above 1 is
        # an error being amplified by the length it is levered through.
        span = max(distances)
        near = [d for d in distances if d <= max(span // 3, 1)]
        far = [d for d in distances if d >= 2 * span // 3]

        def rate(group):
            values = [
                (sum(by_distance[d]) / len(by_distance[d])) / d for d in group
            ]
            return sum(values) / max(len(values), 1)

        if near and far:
            print(f"per-residue rate {rate(near):.4f} A near (to {max(near)}) "
                  f"against {rate(far):.4f} A far (from {min(far)}): "
                  f"amplification {rate(far) / max(rate(near), 1e-9):.2f}x")
    if totals:
        mean = sum(m for m, _ in totals) / len(totals)
        worst = max(w for _, w in totals)
        print(f"\nbeyond six residues: mean shift {mean:.3f} A, worst "
              f"{worst:.3f} A over {len(totals)} draws")
    if not args.checkpoint:
        print("\nUNTRAINED. Every head's output layer is zero-initialised, so "
              "nothing in this rollout\ncorrects anything and the numbers above "
              "are pure propagation -- the\nno-correction reference a trained "
              "run is read against, not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
