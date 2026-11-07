"""Read the ceiling check's own loss off a trained checkpoint, on the CPU.

`g8_ceiling` (32405172) ran its whole 2,000 updates with the held-out `move`
term structurally absent -- `evaluate()` never passed `move_weight`, and
`growth_objective` computes the term only `if weight:`. So the run cannot answer
the question it exists for, and the two readings need opposite fixes:

  * **`move` low** -- the rule learned the right per-step moves and they do not
    accumulate into placement. The suspect is then the rollout's composition:
    the bernoulli update schedule, rotating each residue about its own CA, or
    moves applied and undone.
  * **`move` high** -- the placement gradient never reached the frame head, and
    the architecture cannot place residues under any supervision.

Re-running the arm with working instrumentation costs 71 minutes of a GPU slot
and gives the trajectory. **This gives the endpoint for free**, off the
checkpoint the run already wrote, which is enough to pick the branch -- so run
this first and only spend the slot if the endpoint is ambiguous.

**And `move` alone cannot pick the branch, which is why this reports two more
numbers.** Measured on a local checkpoint: `move` tracks `rmsd_dmg` at a near
constant ratio (0.0206 against 1.11 A, 0.2269 against 11.12 A -- about 50x
throughout), because its position half is the same per-residue distance the
RMSD is, averaged over live steps and diluted by the intact residues. So "move
fell but rmsd_dmg did not" is close to impossible by construction, and a loss
value was the wrong instrument.

What separates the two readings is **direction**, not magnitude:

  * ``align`` -- the cosine between the emitted translation and the ideal one,
    over damaged residues and live steps. 1.0 is a perfect bearing, 0.0 is
    orthogonal, negative is away from the target. This is the number that says
    whether the rule knows *which way* to go, which is exactly what the reach
    probes left open for the reconstruction arms.
  * ``step`` -- the emitted translation's length against the 0.3 A cap, so a
    rule that knows the direction but asks for nothing is distinguishable from
    one that asks for the wrong thing.

    python scripts/ceiling_readout.py --checkpoint runs/g8_ceiling/last.pt
"""

from __future__ import annotations

import argparse
from datetime import datetime
import random
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca.data import RepackingDataset, read_manifest
from aanca.growth import GrowthConfig, growth_objective
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig

ROOT = Path(__file__).resolve().parent.parent
PDB = ROOT / "data" / "pdb"
MANIFEST = ROOT / "data" / "manifest.json"


def held_out(stored: dict, state: dict, extra: int = 0):
    splits = read_manifest(MANIFEST)
    available = []
    for spec in splits[str(stored.get("split") or "train")]:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                available.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    wanted = int(stored.get("structures", 400)) + int(
        stored.get("eval_structures", 12)
    )
    pool = available[:wanted] if len(available) >= wanted else available
    if not pool:
        raise SystemExit("no structures on disk for that split")
    dataset = RepackingDataset(
        pool, concentrations=(0.0,),
        crop_size=int(stored.get("crop_size") or 128),
        device=torch.device("cpu"), cache_limit=max(len(pool), 1),
    )
    indices = list(state.get("eval_indices") or [])
    if not indices:
        order = list(range(len(pool)))
        random.Random(int(stored.get("seed", 0))).shuffle(order)
        indices = order[: int(stored.get("eval_structures", 12))]
    if extra:
        # **These extra crops are TRAINING crops and are only ever valid for
        # an instrument-agreement question.** The held-out set is twelve by the
        # split, and twelve is not enough to settle whether `align` and
        # `rmsd_dmg` agree: on `g10_equiv_ceil` the correlation came back
        # r = -0.223 with a 95% interval of [-0.707, +0.402], which crosses
        # zero, so the check was underpowered rather than negative.
        #
        # Agreement between two instruments is a property of the instruments,
        # not of the split -- `align` is computed from emitted moves against
        # ideal directions and `rmsd_dmg` from final coordinates against the
        # native, so they are independently derived quantities that must agree
        # if both read the same reality. Asking that question on seen crops is
        # legitimate; quoting any resulting `rmsd_dmg` as performance is not,
        # which is why the readout prints the warning rather than trusting the
        # reader to remember.
        held = set(indices)
        rest = [i for i in range(len(pool)) if i not in held]
        random.Random(int(stored.get("seed", 0)) + 1).shuffle(rest)
        indices = indices + rest[:extra]
    return dataset, indices, len(pool)


def _bearing(result, native, topo, damage, stored, seed, index, local=False):
    """``(align, step, net_over_path, series, need)`` over the damaged window.

    **``need`` is the window's RMSD at the START state**, before a single step
    is taken, computed by the same ``ca_rmsd_on`` as ``rmsd_dmg``. It is the
    distance the gradient window has to cover, and it is here because the
    readout previously tested the truncation window against the FINAL
    ``rmsd_dmg`` -- which is the outcome, not the need. That made "crops
    needing more than the window" a relabelling of the outcome distribution,
    and made the one crop said to sit inside the window the crop that merely
    *ended* closest, which is true by construction. See
    ``docs/RETRACTIONS.md``.

    ``align`` is the cosine between the emitted translation and the direction
    the residue must travel to reach its native CA -- the quantity a loss value
    cannot give, since a rule can have the right magnitude and the wrong
    bearing.

    **``net_over_path`` and ``series`` exist because a correct bearing is not
    sufficient.** The pMHC binder session put it as the two-frames question
    pointed at time rather than space: if the bearing is right but *re-aimed*
    every step from a frame that is itself drifting, the walk can be correct at
    every step and still arrive nowhere. ``net_over_path`` is the window's net
    displacement over its total path length, so 1.0 is a straight line and
    values near 0 are a walk that spends its budget turning. ``series`` is the
    per-step cosine, because a mean hides the difference between "wrong
    throughout" and "right at first, then lost".

    That matters most for ``--equivariant-move``, which this measurement can
    embarrass: reading the output in the residue's own frame is what makes the
    same neighbourhood give the same *local* move, but if the residue's frame
    drifts across steps then a constant local output traces a curve in the lab.
    The fix removes one failure and can introduce the other, and only the path
    ratio distinguishes them.
    """
    from aanca.backbone import frame_from_triad
    from aanca.growth import backbone_atoms, damage_window, growth_links

    moves = getattr(result, "moves", [])
    nan = float("nan")
    if not moves:
        return nan, nan, nan, [], nan
    links = growth_links(topo)
    damaged = damage_window(
        topo.n_residues, links, damage,
        torch.Generator().manual_seed(seed + index), native.device,
    )
    if not bool(damaged.any()):
        return nan, nan, nan, [], nan
    ca_native = backbone_atoms(native, topo)[:, 1]

    cosines, lengths, series = [], [], []
    path = None
    first = None
    previous = None
    for _axis_angle, translation, coords, moving in moves:
        scored = moving & damaged
        ca_now = backbone_atoms(coords, topo)[:, 1]
        if first is None:
            first = ca_now
            path = torch.zeros_like(ca_now[:, 0])
        if previous is not None:
            path = path + (ca_now - previous).norm(dim=-1)
        previous = ca_now
        if not bool(scored.any()):
            continue
        ideal = (ca_native - ca_now)[scored]
        emitted = translation.to(coords.dtype)
        if local:
            # **Compare like with like.** Under `--equivariant-move` the rule's
            # output is a vector in each residue's own N-CA-C basis, and
            # `rigid_relax` maps it to the lab with `frame @ t` before applying
            # it. The ideal direction here is a lab-frame difference of CA
            # positions, so the emitted vector has to make the same trip or the
            # cosine is between two different spaces.
            #
            # Found by the numbers being too clean: align tight around -0.9
            # from the first step, `net/path` exactly 1.000, and `step`
            # identical to three digits across twelve structures. That is the
            # signature of a frame mismatch, which is the failure the pMHC
            # binder session warned about -- and I introduced it into the
            # diagnostic with the very fix the diagnostic was measuring.
            bb_all = backbone_atoms(coords, topo)
            frame = frame_from_triad(bb_all[:, 0], bb_all[:, 1], bb_all[:, 2])
            emitted = (frame @ emitted.unsqueeze(-1)).squeeze(-1)
        emitted = emitted[scored]
        ideal_unit = ideal / ideal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        emitted_norm = emitted.norm(dim=-1, keepdim=True)
        # **Skip a step that emitted nothing instead of scoring it zero.** A
        # cosine against a zero vector divides by the clamp and comes back 0.0,
        # which reads as "uncorrelated" when the truth is "no move was made".
        # That artefact was reported as evidence once already.
        if float(emitted_norm.mean()) < 1e-6:
            lengths.append(torch.zeros_like(emitted_norm.mean()))
            continue
        cosine = (ideal_unit * (emitted / emitted_norm.clamp_min(1e-8))).sum(-1)
        cosines.append(cosine.mean())
        series.append(round(float(cosine.mean()), 3))
        lengths.append(emitted.norm(dim=-1).mean())

    if not cosines:
        return nan, nan, nan, [], nan
    net = (previous - first).norm(dim=-1)[damaged].mean()
    walked = path[damaged].mean()
    # **The need: the window's RMSD before any step is taken.** The same
    # superposition as `rmsd_dmg` -- fit on the surviving context, read on the
    # damaged window -- but on `first`, the start state, so it is derived
    # independently of how the rollout turns out.
    from aanca.fape import ca_rmsd_on
    intact = ~damaged
    need = nan
    if int(intact.sum()) >= 3:
        need = float(ca_rmsd_on(
            first[intact], ca_native[intact],
            first[damaged], ca_native[damaged],
        ))
    return (
        float(torch.stack(cosines).mean()),
        float(torch.stack(lengths).mean()),
        float(net / walked.clamp_min(1e-9)),
        series,
        need,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--untrained", action="store_true",
        help="score a fresh rule instead, for the value these terms start at",
    )
    parser.add_argument(
        "--perturb", type=float, default=0.0,
        help=(
            "with --untrained, add noise of this scale to the frame head's "
            "last layer. **This is the control `--untrained` alone cannot "
            "be.** A fresh rule's frame head is zero-initialised, so it emits "
            "the zero vector and `align` comes back exactly 0.000 -- which is "
            "an artefact of taking a cosine against a zero norm, not a "
            "measurement that the bearing is random. Perturbing the head gives "
            "non-zero moves with no training, so the bearing is defined and "
            "the question 'does the lab-frame defect give an uncorrelated "
            "bearing independently of training?' can actually be asked. 0.05 "
            "reproduces the scale used in tests/test_move_equivariance.py"
        ),
    )
    parser.add_argument(
        "--extra-crops", type=int, default=0, metavar="N",
        help=(
            "add N TRAINING crops to the twelve held-out ones. **For the "
            "instrument-agreement question only.** `align` and `rmsd_dmg` are "
            "computed from different things -- emitted moves against ideal "
            "directions, and final coordinates against the native -- so they "
            "must agree if both read the same reality, and at k=12 that check "
            "is underpowered: r = -0.223, 95%% CI [-0.707, +0.402]. Agreement "
            "is a property of the instruments rather than of the split, so "
            "seen crops answer it. Any `rmsd_dmg` printed under this flag is "
            "NOT a held-out number and the readout says so on every run"
        ),
    )
    args = parser.parse_args()

    path = Path(args.checkpoint)
    state = ckpt.load(path, map_location="cpu")
    stored = state.get("args") or {}
    if not stored:
        sibling = path.with_name("last.pt")
        if sibling.exists():
            beside = ckpt.load(sibling, map_location="cpu")
            stored = beside.get("args") or {}
            if not state.get("eval_indices"):
                state["eval_indices"] = beside.get("eval_indices") or []
    if not stored:
        print(f"{path} stores no run arguments; point at the run's last.pt")
        return 1

    # **Say WHICH checkpoint this is, because `last.pt` moves under you.**
    # Job 32423296 read `runs/g10_equiv_ceil/last.pt` between 01:05 and 01:10
    # while the run that owns it was still training; the file was overwritten at
    # 01:10:13, 34 seconds before the job ended. Job 32449992 re-read the same
    # path at 11:52 and got materially different numbers -- align +0.008 against
    # -0.096, rmsd_dmg 12.52 against 12.85 -- which read as instrument
    # non-determinism for as long as it took to check the timestamps. It was two
    # different sets of weights with one filename.
    #
    # This is the `best.pt` failure again in a second costume: a probe pointed at
    # a file whose identity nothing in the output recorded, so a real
    # measurement of an unrecorded intermediate got reported as the arm's
    # result. Printing the update and the mtime makes a stale or mid-training
    # read visible in the log it produced, which is the only place anyone will
    # look later.
    update = state.get("update")
    try:
        stamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(
            timespec="seconds"
        )
    except OSError:
        stamp = "unknown"
    print(
        f"checkpoint : {path}\n"
        f"  update   : {update if update is not None else 'NOT RECORDED'}\n"
        f"  written  : {stamp}"
    )
    if update is None:
        print("  *** this checkpoint does not record its update, so the number")
        print("  *** below cannot be attributed to a point in the run.")

    weights = ckpt.rule_weights(state)
    dataset, indices, pool_size = held_out(stored, state, args.extra_crops)
    damage = int(stored.get("damage") or 0)
    if damage <= 0:
        print("this run is not a regeneration run; nothing to read")
        return 1

    config = RolloutConfig(
        steps=int(stored.get("steps") or 120),
        truncate=int(stored.get("truncate") or 12),
        k=int(stored.get("k") or 32),
        blind=bool(stored.get("blind", True)),
        growth=GrowthConfig(
            birth="none", initial=str(stored.get("initial") or "helix")
        ),
    )
    seed = int(stored.get("seed", 0))
    rows = []
    rng = random.Random(seed)
    for index in indices:
        try:
            topo, native, _c, _u = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 12:
            continue
        rule = AANCARule(
            growth=True, birth="none",
            identity="identity_head.0.weight" in weights,
            morphogen="emit_head.0.weight" in weights,
            identity_keep=float(stored.get("identity_keep", 2.0)),
        )
        if not args.untrained:
            rule.load_state_dict(weights)
        elif args.perturb:
            # Seeded per crop so the perturbation is a property of the run and
            # not of the order the crops happen to load in.
            generator = torch.Generator().manual_seed(seed + index)
            for parameter in rule.frame_head[-1].parameters():
                parameter.data.add_(
                    torch.randn(
                        parameter.shape, generator=generator
                    ) * args.perturb
                )
        rule.eval()
        with torch.no_grad():
            terms, result = growth_objective(
                rule, native, topo, config,
                generator=torch.Generator().manual_seed(seed + index),
                identity_weight=float(stored.get("identity_weight") or 0.0),
                identity_start=str(stored.get("identity_start", "masked")),
                damage=damage,
                move_weight=1.0,
                clash_weight=1.0,
                clash_tolerance=float(stored.get("clash_tolerance") or 0.5),
            )
        if terms.rmsd_damaged is None:
            continue
        align, step, ratio, series, need = _bearing(
            result, native, topo, damage, stored, seed, index,
            local=bool(stored.get("equivariant_move")),
        )
        rows.append((
            index,
            float(terms.move),
            float(terms.clash),
            float(terms.rmsd_damaged),
            align,
            step,
            ratio,
            series,
            need,
        ))

    if not rows:
        print("no crops produced a damaged window")
        return 1
    label = str(path)
    if args.untrained:
        label = (
            f"UNTRAINED, frame head perturbed by {args.perturb:g}"
            if args.perturb
            else "UNTRAINED (zero-initialised head: `align` is vacuous, see "
            "--perturb)"
        )
    frame_note = (
        "emitted move rotated into the lab frame for comparison "
        "(--equivariant-move run)"
        if stored.get("equivariant_move")
        else "emitted move is already a lab-frame vector"
    )
    if args.extra_crops:
        print(
            f"{label}, {len(rows)} crops of {pool_size} -- "
            f"12 HELD OUT PLUS {args.extra_crops} TRAINING CROPS"
        )
        print("  *** rmsd_dmg below is NOT a held-out number. This mode exists")
        print("  *** for the align/rmsd_dmg agreement check, which needs k the")
        print("  *** split cannot supply. Do not quote it as performance.")
    else:
        print(f"{label}, {len(rows)} held-out crops of {pool_size}, "
              f"{damage}-residue window, {config.steps} steps")
    print(f"{frame_note}\n")
    # `need` is the window's distance from native at the START state and
    # `rmsd_dmg` is where it ends, so `need - rmsd_dmg` is what the rollout
    # actually bought. Printing them side by side because a good `rmsd_dmg` on
    # a crop that started close is not the same result as the same number on a
    # crop that started far, and the two were conflated once already.
    print(f"{'crop':>6} {'move':>8} {'clash':>9} {'need':>7} {'rmsd_dmg':>9} "
          f"{'gained':>7} {'align':>7} {'step':>7} {'net/path':>9}")
    for index, move, clash, dmg, align, step, ratio, _s, need in rows:
        print(f"{index:>6} {move:>8.4f} {clash:>9.5f} {need:>7.2f} {dmg:>9.2f} "
              f"{need - dmg:>7.2f} {align:>7.3f} {step:>7.4f} {ratio:>9.3f}")
    n = len(rows)
    print(f"\n{'mean':>6} {sum(r[1] for r in rows)/n:>8.4f} "
          f"{sum(r[2] for r in rows)/n:>9.5f} {sum(r[8] for r in rows)/n:>7.2f} "
          f"{sum(r[3] for r in rows)/n:>9.2f} "
          f"{sum(r[8] - r[3] for r in rows)/n:>7.2f} "
          f"{sum(r[4] for r in rows)/n:>7.3f} {sum(r[5] for r in rows)/n:>7.4f} "
          f"{sum(r[6] for r in rows)/n:>9.3f}")
    cap = 0.3
    used = 100 * sum(r[5] for r in rows) / n / cap
    print(f"\n  step against the {cap:g} A cap: {used:.1f}% used")
    # **How much is the head actually ASKING for?** `step` is the move after
    # ``_bounded``, which is a tanh and not a clamp:
    #
    #     bounded = L * tanh(raw / L)
    #
    # so it is invertible, and the raw request the frame head emitted comes
    # back without touching ``model.py``:
    #
    #     raw = L * atanh(bounded / L)
    #
    # This matters because the two ends of that curve fail differently.
    # ``_bounded``'s own docstring records the failure at the origin -- a
    # zero-initialised head gets ``0/eps = 0`` and no gradient ever -- and the
    # branch there is called load-bearing for that reason. The far end has the
    # same shape: the derivative of the bounded norm with respect to the raw
    # one is ``1 - tanh^2(raw/L)``, so a head asking for several times the cap
    # sits in the flat tail and its MAGNITUDE gradient is attenuated by orders
    # of magnitude. The docstring says tanh was chosen so "the gradient does
    # not vanish for a head that is asking for too much"; whether it vanishes
    # anyway is an empirical question about how much too much, and this is the
    # number that answers it.
    #
    # **Precision ceiling, stated because the inversion has one, and it is
    # tighter than the algebra suggests.** Measured in float32 (and pinned in
    # ``tests/test_bounded_saturation.py``):
    #
    #     request    bounded norm    d|out|/d|in|
    #     0.031 A     0.0308901       9.9e-01     healthy
    #     1.41  A     0.2999504       3.3e-04
    #     2.0   A     0.2999991       6.4e-06     recovery already 0.2% off
    #     3.0   A     0.3000000       -7.5e-09    SIGN IS WRONG
    #     5.0   A     0.3000000        0.0        EXACTLY zero
    #
    # So in exact arithmetic tanh is asymptotic and the request is recoverable,
    # but float32 **reaches and slightly exceeds** the cap at around ten times
    # it, where the gradient is not merely attenuated but exactly zero -- the
    # same total loss of gradient ``_bounded``'s docstring documents at the
    # origin, two orders of magnitude earlier than ``1 - tanh^2`` predicts,
    # with a wrong-signed interval on the way. A saturated figure is therefore
    # a LOWER BOUND with no upper one, and is printed as such.
    import math as _math
    ratios = [min(r[5] / cap, 1.0 - 1e-7) for r in rows]
    raws = [cap * _math.atanh(v) for v in ratios]
    atten = [1.0 - _math.tanh(v / cap) ** 2 for v in raws]
    raw_mean = sum(raws) / n
    saturated = sum(1 for v in ratios if v >= 1.0 - 1e-7)
    bound = ">=" if saturated else "  "
    print(f"\n  the head is ASKING for {bound}{raw_mean:.2f} A per step, "
          f"which is {bound}{raw_mean / cap:.1f}x the {cap:g} A cap")
    print(f"  tanh gradient on the magnitude: {sum(atten) / n:.2e} "
          f"of its value near the origin")
    if saturated:
        print(f"  *** {saturated}/{n} crops saturate the float32 inversion, so")
        print("  *** the request is a LOWER bound with NO upper one -- it may")
        print("  *** be ten or a hundred times the cap and this cannot say.")
        print("  *** At that point the magnitude gradient is EXACTLY zero in")
        print("  *** float32, not merely small: the same total loss of")
        print("  *** gradient _bounded's docstring documents at the origin.")
    if raw_mean / cap > 3.0:
        print("  *** The magnitude head is in tanh's flat tail. This is the")
        print("  *** failure _bounded's docstring guards against at the ORIGIN,")
        print("  *** arriving at the other end: the smoothing that was chosen")
        print("  *** to keep the gradient alive for a head asking too much has")
        print("  *** saturated, so the rule cannot be taught to ask for LESS.")
        print(f"  *** Over {config.steps} steps at the cap the window travels")
        print(f"  *** {config.steps * cap:.1f} A, so overshoot is forced rather")
        print("  *** than chosen -- and a vector that overshoots turns its own")
        print("  *** bearing negative once it is past the target, which is how")
        print("  *** a saturated magnitude shows up in `align`.")

    if used > 99.0:
        # **Saturation is a result, not a formatting detail.** A uniform `step`
        # across crops is expected of a rule emitting a near-constant local
        # vector and is NOT a tell -- that was established on a fixed random
        # head, which gave 0.0394-0.0441 over twelve structures. Sitting at
        # 100 % of the cap is different: the rule asks for the largest move it
        # is allowed, on every step, on every crop. Read beside `align` near
        # zero and `net/path` near one, that says each damaged residue walks a
        # straight line at full speed in a direction uncorrelated with where it
        # has to go, which is a sharper statement than "it does not aim".
        print("  *** SATURATED: the rule asks for the whole cap every step.")

    window = config.truncate * cap
    # **`r[8]` is the START distance, not `r[3]` the final `rmsd_dmg`.** Using
    # the outcome here made this test a relabelling of the outcome
    # distribution, and made the crop said to sit inside the window the crop
    # that merely *ended* closest -- true by construction. See
    # docs/RETRACTIONS.md.
    needs = sorted(r[8] for r in rows if r[8] == r[8])
    beyond = sum(1 for v in needs if v > window)
    print(f"\n  gradient window = truncate {config.truncate} x {cap:g} A "
          f"= {window:.1f} A")
    if needs:
        print(f"  the damaged window STARTS {needs[len(needs) // 2]:.2f} A from "
              f"native (median of {len(needs)}), range {needs[0]:.2f} to "
              f"{needs[-1]:.2f}")
    print(f"  crops whose START distance exceeds the window: "
          f"{beyond}/{len(needs)}")
    print("  *** the window is what the gradient can see. A crop whose damaged")
    print("  *** region sits further from native than this cannot be taught to")
    print("  *** arrive, however good the bearing is, because arriving happens")
    print("  *** outside the segment the loss is differentiated through.")

    # **The agreement check, computed here rather than by hand.** `align` comes
    # from emitted moves against ideal directions; `rmsd_dmg` from final
    # coordinates against the native. Two independently derived quantities that
    # must agree if both read the same reality -- which is the only question
    # that tests an instrument's *convention*, as against a planted signal,
    # which tests sensitivity alone. Better aim should mean lower rmsd, so the
    # correlation should be clearly NEGATIVE.
    if n >= 6:
        import math
        a = [r[4] for r in rows]
        d = [r[3] for r in rows]

        def pearson(x, y):
            mx, my = sum(x) / len(x), sum(y) / len(y)
            sx = math.sqrt(sum((v - mx) ** 2 for v in x))
            sy = math.sqrt(sum((v - my) ** 2 for v in y))
            if sx == 0 or sy == 0:
                return float("nan")
            return sum(
                (x[i] - mx) * (y[i] - my) for i in range(len(x))
            ) / (sx * sy)

        r = pearson(a, d)
        print(f"\n  AGREEMENT  align vs rmsd_dmg : r = {r:+.3f}", end="")
        if n > 3 and -0.999 < r < 0.999:
            z = 0.5 * math.log((1 + r) / (1 - r))
            se = 1 / math.sqrt(n - 3)
            lo = math.tanh(z - 1.96 * se)
            hi = math.tanh(z + 1.96 * se)
            print(f"   95% CI [{lo:+.3f}, {hi:+.3f}]")
            if lo < 0 < hi:
                print("  *** the interval crosses zero, so this k cannot say")
                print("  *** whether the two instruments agree. Underpowered,")
                print("  *** which is NOT the same as disagreeing.")
                print("  *** --extra-crops raises k for this check.")
            elif hi < 0:
                print("  *** they agree: better bearing goes with lower rmsd,")
                print("  *** and the two were computed from different things.")
            else:
                print("  *** THEY DISAGREE IN SIGN. Better bearing going with")
                print("  *** WORSE rmsd means one of them is misreading; find")
                print("  *** out which before quoting either.")
        else:
            print()
    print("\n  per-step bearing, first crop (a mean hides 'right then lost'):")
    print("   ", rows[0][7][:14] or "(nothing emitted)")
    print(
        "\n  Read `align` first -- it is the one that picks the branch.\n"
        "    near 1.0  the rule knows the bearing, so a flat rmsd_dmg means\n"
        "              the moves are not accumulating: suspect the update\n"
        "              schedule, rotating about each residue's own CA, or\n"
        "              moves applied and undone.\n"
        "    near 0.0  the rule does not know which way to go even when every\n"
        "              step is supervised, and the capacity question answers\n"
        "              no.\n"
        "    negative  it is being driven away from the target, which no\n"
        "              amount of objective design fixes.\n"
        "  `step` against the 0.3 A cap separates 'wrong direction' from\n"
        "  'asking for nothing'. `move` is reported for continuity but is\n"
        "  roughly rmsd_dmg / 50 and settles nothing on its own."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
