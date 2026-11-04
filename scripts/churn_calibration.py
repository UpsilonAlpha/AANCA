"""Is churn a confidence score? Churn against burial, certainty and correctness.

    python scripts/churn_calibration.py runs/v8c_if/last.pt [--steps 64]

`scripts/churn_composition.py` found that every one of the ten most unstable
labels per use is polar, charged or CYS, and that no aliphatic or aromatic
residue reaches 1.16. The reading offered for that was **the rule converging on
the core and staying undecided on the surface** -- which would be realistic
behaviour rather than a fault, and would make churn a self-reported confidence
signal in the family of pLDDT and ipTM.

**That reading was an inference from chemistry, and this script is the
measurement it needs.** "Polar and charged" was being read as "exposed" without
anybody measuring exposure. Burial is measurable here -- ``aanca.metrics``
carries AttnPacker's convention, core at 20 or more CB neighbours within 10 A
and surface at 15 or fewer, which is the same definition the packing eval's
core/surf columns already use -- so the claim can simply be checked.

**What would make churn a confidence score, and what would not.** pLDDT is
useful because its value *predicts its own accuracy*: a low number is a
truthful warning. So the decisive question is not whether churn correlates with
burial, which is a statement about proteins, but whether churn **predicts
per-residue correctness**, which is a statement about the model knowing what it
does not know. Three things are therefore reported and they answer different
questions:

* **Churn and entropy by burial band.** Confirms or refutes the core/surface
  reading directly. This is the part that was asserted and never measured.
* **Recovery by churn quintile, and by confidence quintile.** The calibration
  curves. If recovery falls monotonically as churn rises, a residue's own
  flipping rate is a usable warning about its own label.
* **AUC for each, as a predictor of correctness.** One number per candidate
  score, so churn and the head's own certainty are comparable rather than
  merely both plausible. Computed from ranks (Mann-Whitney), which needs no
  threshold and no extra dependency.

**Churn and certainty are not independent, and the script says so rather than
reporting them as two findings.** An uncertain head flips more, so the two are
expected to agree; the question worth answering is whether churn carries
anything *beyond* confidence. The last table conditions on confidence and asks
whether churn still separates correct from incorrect inside a confidence band.
If it does not, churn is a more expensive restatement of the entropy and the
entropy is the score to ship.

A reliability table is included for confidence specifically, because that is
the pLDDT analogue proper: a max-probability is already on the scale of an
accuracy, so predicted and observed can be compared bin by bin instead of only
ranked.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from aanca import checkpoint as ckpt
from aanca import residue_constants as rc
from aanca.data import RepackingDataset, read_manifest
from aanca.identity import library_index, type_index, type_marginal
from aanca.metrics import cb_neighbour_counts, classify_burial
from aanca.model import AANCARule
from aanca.rollout import (
    RolloutConfig,
    initial_identity,
    initial_state,
    step_once,
    update_mask,
)
from aanca.torch_geometry import measure_chis
from aanca.update_schedule import coupling_graph, random_independent_set

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
PDB = ROOT / "data" / "pdb"

BANDS = ("core", "intermediate", "surface")


def on_disk(specs):
    out = []
    for spec in specs:
        for suffix in (".bcif", ".cif", ".pdb"):
            candidate = PDB / f"{spec.pdb_id}{suffix}"
            if candidate.exists():
                out.append(type(spec)(spec.pdb_id, candidate, spec.group))
                break
    return out


def auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    """P(score of a positive > score of a negative), ties counted as a half.

    The rank identity rather than a threshold sweep: with ``n_pos`` positives
    and ``n_neg`` negatives, the Mann-Whitney statistic over average ranks is
    exactly the probability above, so no operating point has to be chosen and
    no curve has to be integrated.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = score.argsort()
    ranks = torch.empty_like(score, dtype=torch.float64)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.float64)
    # Average the ranks inside each tied group, or a score that takes few
    # distinct values -- which churn does, being a count over a few steps --
    # would score differently depending on the sort's arbitrary order.
    unique, inverse = torch.unique(score, return_inverse=True)
    summed = torch.zeros(unique.numel(), dtype=torch.float64)
    counts = torch.zeros(unique.numel(), dtype=torch.float64)
    summed.scatter_add_(0, inverse, ranks)
    counts.scatter_add_(0, inverse, torch.ones_like(ranks))
    ranks = (summed / counts)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def quintile_table(
    score: torch.Tensor, correct: torch.Tensor, label: str, bins: int = 5
) -> str:
    """Observed recovery within equal-count bins of ``score``."""
    order = score.argsort()
    edges = [round(i * score.numel() / bins) for i in range(bins + 1)]
    lines = [
        f"    {label:<14}{'range':>16}{'recovery':>11}{'n':>7}"
    ]
    for i in range(bins):
        index = order[edges[i] : edges[i + 1]]
        if index.numel() == 0:
            continue
        chunk = score[index]
        lines.append(
            f"    {'bin ' + str(i + 1):<14}"
            f"{f'{float(chunk.min()):.3f}-{float(chunk.max()):.3f}':>16}"
            f"{float(correct[index].to(torch.float64).mean()):>10.1%}"
            f"{index.numel():>7}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--structures", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--window",
        type=float,
        default=0.5,
        help="fraction of the rollout, measured from the end, over which churn "
        "and entropy are averaged. The early steps are a transient from the "
        "masked start and describe the schedule rather than the converged "
        "behaviour, and churn's own curve shows it: 0.16 at step 1 settling to "
        "0.12 by step 17.",
    )
    args = parser.parse_args()

    state = ckpt.load(Path(args.checkpoint), map_location="cpu")
    stored = state.get("args") or {}
    if "eval_indices" not in state:
        print("no eval_indices in this checkpoint; pass the run's last.pt")
        return 1
    indices = list(state["eval_indices"])[: args.structures]

    split = stored.get("split", "validation")
    pool = on_disk(read_manifest(MANIFEST)[split])
    dataset = RepackingDataset(
        pool, concentrations=(0.0,), crop_size=int(stored.get("crop_size", 384)),
        device="cpu", cache_limit=max(len(pool), 1),
    )

    rule = AANCARule(
        identity=True, identity_keep=float(stored.get("identity_keep", 8.0))
    )
    weights = Path(args.weights) if args.weights else Path(args.checkpoint)
    rule.load_state_dict(ckpt.rule_weights(ckpt.load(weights, map_location="cpu")))
    rule.eval()

    config = RolloutConfig(
        steps=args.steps, truncate=args.steps, k=int(stored.get("k", 64)),
        blind=True,
        identity_start=str(stored.get("identity_start", "masked")),
        identity_sampling=str(stored.get("identity_sampling", "sample")),
        update_schedule=str(stored.get("update_schedule", "bernoulli")),
        coupling_radius=float(stored.get("coupling_radius", 5.0)),
    )

    first_scored = max(int(args.steps * (1.0 - args.window)), 1)
    churn_all, entropy_all, confidence_all = [], [], []
    correct_all, counts_all, band_all = [], [], []
    churn_step_all, ticks_all = [], []
    rng = random.Random(args.seed)

    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _c, _u = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            supervised = topo.supervised_mask
            if not bool(supervised.any()):
                continue

            native_chi1 = torch.rad2deg(measure_chis(native, topo.torsion)[:, 0])
            target = type_index(library_index(topo.torsion.resnames, native_chi1))
            counts = cb_neighbour_counts(native, topo)

            start = initial_identity(topo, native, config.identity_start)
            state_ = initial_state(native, rule, identity=start)
            previous = state_.identity.argmax(-1)
            generator = torch.Generator().manual_seed(args.seed)

            flips = torch.zeros(topo.n_residues, dtype=torch.float64)
            entropy = torch.zeros(topo.n_residues, dtype=torch.float64)
            confidence = torch.zeros(topo.n_residues, dtype=torch.float64)
            # **The confound this probe nearly reported as its result.** A
            # residue that does not tick cannot flip, so flips-per-step
            # measures how often a residue updates as much as how often it
            # changes its mind. Under the ``independent`` schedule those are
            # not independent of burial: an independent set excludes
            # neighbours, so a residue with more neighbours is excluded more
            # often and ticks less. Caught because churn ordered core <
            # intermediate < surface on a rule trained for six updates, which
            # has learned nothing and cannot have a calibrated confidence --
            # the ordering had to be mechanical. Ticks are counted so the
            # primary rate is **per update**, which is the conditional
            # "when this residue was asked, how often did it change?".
            ticks = torch.zeros(topo.n_residues, dtype=torch.float64)
            scored_steps = 0

            for step in range(1, args.steps + 1):
                if config.update_schedule == "independent":
                    mask = random_independent_set(
                        coupling_graph(state_.coords, topo, config.coupling_radius),
                        generator,
                    )
                else:
                    mask = update_mask(
                        topo.n_residues, config.update_probability,
                        state_.coords.device, generator,
                    )
                state_, _mu, _lk = step_once(
                    rule, state_, topo, config, mask, generator=generator
                )
                current = state_.identity.argmax(-1)
                if step >= first_scored:
                    flips += (current != previous).to(torch.float64)
                    ticks += mask.to(torch.float64)
                    marginal = type_marginal(torch.softmax(state_.identity_logits, -1))
                    marginal = marginal.clamp_min(1e-12)
                    entropy += -(marginal * marginal.log()).sum(-1).to(torch.float64)
                    confidence += marginal.max(-1).values.to(torch.float64)
                    scored_steps += 1
                previous = current

            if scored_steps == 0:
                continue
            final_type = type_index(state_.identity.argmax(-1))
            bands = classify_burial(counts)
            band = torch.zeros(topo.n_residues, dtype=torch.long)
            for i, name in enumerate(BANDS):
                band[bands[name]] = i

            keep = supervised
            churn_all.append((flips / ticks.clamp_min(1.0))[keep])
            churn_step_all.append((flips / scored_steps)[keep])
            ticks_all.append((ticks / scored_steps)[keep])
            entropy_all.append((entropy / scored_steps)[keep])
            confidence_all.append((confidence / scored_steps)[keep])
            correct_all.append((final_type == target)[keep])
            counts_all.append(counts[keep].to(torch.float64))
            band_all.append(band[keep])

    if not churn_all:
        print("no residues scored")
        return 1

    churn = torch.cat(churn_all)
    churn_step = torch.cat(churn_step_all)
    ticks = torch.cat(ticks_all)
    entropy = torch.cat(entropy_all)
    confidence = torch.cat(confidence_all)
    correct = torch.cat(correct_all)
    neighbours = torch.cat(counts_all)
    band = torch.cat(band_all)

    print(f"{weights}, {len(indices)} held-out structures, {args.steps} steps")
    print(
        f"churn and certainty averaged over the last {scored_steps} steps "
        f"(window {args.window:g}); {churn.numel()} supervised residues, "
        f"overall recovery {float(correct.to(torch.float64).mean()):.1%}\n"
    )

    # --- the claim that was asserted and never measured ---------------------
    print("  churn and certainty by burial (AttnPacker: core >=20 CB "
          "neighbours, surface <=15):")
    print(
        f"    {'band':<14}{'churn/upd':>10}{'churn/step':>11}{'ticks':>8}"
        f"{'entropy':>9}{'confid':>8}{'recovery':>10}{'CB nbrs':>9}{'n':>7}"
    )
    for i, name in enumerate(BANDS):
        mask = band == i
        if not bool(mask.any()):
            continue
        print(
            f"    {name:<14}{float(churn[mask].mean()):>10.3f}"
            f"{float(churn_step[mask].mean()):>11.3f}"
            f"{float(ticks[mask].mean()):>8.3f}"
            f"{float(entropy[mask].mean()):>9.3f}"
            f"{float(confidence[mask].mean()):>8.3f}"
            f"{float(correct[mask].to(torch.float64).mean()):>9.1%}"
            f"{float(neighbours[mask].mean()):>9.1f}{int(mask.sum()):>7}"
        )
    print(
        "\n    **Read churn/upd, not churn/step.** ``ticks`` is the fraction of\n"
        "    scored steps on which the residue was actually asked to update, and\n"
        "    a residue that is not asked cannot flip. Under the independent\n"
        "    schedule that fraction falls with burial by construction, since an\n"
        "    independent set excludes neighbours -- so churn/step is confounded\n"
        "    with burial before the model does anything. If ticks differs across\n"
        "    the bands and churn/upd is flat while churn/step is not, the\n"
        "    core/surface reading is an artefact of the schedule.\n\n"
        "    The reading predicts churn/upd and entropy lower on the core and\n"
        "    recovery higher. Flat bands mean the chemistry result had another\n"
        "    cause."
    )

    # --- is it a confidence score? ------------------------------------------
    print("\n  recovery by churn quintile (low churn first):")
    print(quintile_table(churn, correct, "churn"))
    print("\n  recovery by confidence quintile (low confidence first):")
    print(quintile_table(confidence, correct, "confidence"))

    print("\n  as predictors of per-residue correctness (AUC, 0.5 = useless):")
    print(f"    {'score':<24}{'AUC':>8}")
    for name, score in (
        ("-churn (per update)", -churn),
        ("-churn (per step)", -churn_step),
        ("-entropy", -entropy),
        ("confidence", confidence),
        ("update rate (ticks)", ticks),
        ("CB neighbours", neighbours),
    ):
        print(f"    {name:<24}{auc(score, correct):>8.3f}")
    print(
        "\n    Signs are set so higher is predicted-correct in every row, which\n"
        "    is why churn and entropy are negated. Two controls, and a score\n"
        "    has to beat both to be worth shipping: CB neighbours is a property\n"
        "    of the target structure and not of the model at all, and the update\n"
        "    rate is a property of the schedule. If churn/step scores well while\n"
        "    churn/upd and the update rate both score near 0.5, what looked like\n"
        "    a confidence signal was the scheduler."
    )

    # --- does churn add anything over the entropy? --------------------------
    print("\n  churn's AUC within confidence terciles (does it add anything?):")
    order = confidence.argsort()
    edges = [round(i * confidence.numel() / 3) for i in range(4)]
    print(f"    {'confidence tercile':<22}{'churn AUC':>11}{'recovery':>11}{'n':>7}")
    for i in range(3):
        index = order[edges[i] : edges[i + 1]]
        if index.numel() < 2:
            continue
        print(
            f"    {'tercile ' + str(i + 1):<22}"
            f"{auc(-churn[index], correct[index]):>11.3f}"
            f"{float(correct[index].to(torch.float64).mean()):>10.1%}"
            f"{index.numel():>7}"
        )
    print(
        "\n    An AUC near 0.5 in every tercile means churn is a restatement of\n"
        "    the entropy and the entropy is the cheaper score to ship. Staying\n"
        "    above 0.5 means the flipping carries information the head's own\n"
        "    distribution at one step does not."
    )

    # --- the pLDDT analogue proper ------------------------------------------
    print("\n  reliability of confidence (a max-probability is already on the\n"
          "  scale of an accuracy, so predicted and observed are comparable):")
    print(f"    {'predicted':<14}{'observed':>10}{'gap':>8}{'n':>7}")
    gap_total, gap_n = 0.0, 0
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        mask = (confidence >= low) & (confidence < low + 0.2)
        if not bool(mask.any()):
            continue
        predicted = float(confidence[mask].mean())
        observed = float(correct[mask].to(torch.float64).mean())
        print(
            f"    {f'{low:.1f}-{low + 0.2:.1f}':<14}{observed:>9.1%}"
            f"{observed - predicted:>+8.1%}{int(mask.sum()):>7}"
        )
        gap_total += abs(observed - predicted) * int(mask.sum())
        gap_n += int(mask.sum())
    if gap_n:
        print(f"\n    expected calibration error {gap_total / gap_n:.1%}")
    print(
        "    A positive gap is under-confidence and a negative gap is the\n"
        "    dangerous direction -- a head claiming more than it delivers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
