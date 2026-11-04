"""What does each growth term contribute, and to which head?

    python scripts/growth_balance_probe.py [--seeds 6] [--steps 30]

Written before the first growth training run rather than after it. The growth
objective has five terms -- FAPE, the peptide restraint, the reachability
restraint, completion, and identity -- and they are in wildly different units:
FAPE is clamped and divided by 10 A so it is order 1, the restraints are in
Angstrom and Angstrom squared, completion is a cross-entropy, identity is a
cross-entropy over twenty types. Nothing has ever checked that the weights put
them on comparable footing, and two observations at initialisation suggested
they do not: completion read 2.13 against FAPE's 0.56, and with the reach term
on, the activation head's gradient came out at 3.3e4 against the birth head's
98.

A term that dominates the gradient by two orders of magnitude is the only term
being optimised. That is worth twenty seconds here rather than a day of GPU.

**What this reports.** For each term alone, with every other weight zeroed: its
value, and the gradient norm it induces on each head. The gradient is the
quantity that matters -- a term can be numerically small and still dominate if
its gradient is large, which is exactly the case for a cross-entropy against a
saturated logit.

The recommendation at the end is the weight that would equalise each term's
gradient contribution against FAPE's, which is the term the whole objective is
really about.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aanca.growth import GrowthConfig, growth_objective
from aanca.model import AANCARule
from aanca.rollout import RolloutConfig
from aanca.structure import load_protein
from aanca.torch_graph import feature_topology

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

HEADS = ("activation_head", "birth_head", "frame_head", "identity_head")

#: Each term, and the keyword that switches it on alone.
TERMS = (
    ("fape", {"fape_weight": 1.0}),
    ("bond", {"bond_weight": 1.0}),
    ("reach", {"reach_weight": 1.0}),
    ("completion", {"completion_weight": 1.0}),
    ("identity", {"identity_weight": 1.0}),
)


def gradients(rule) -> dict[str, float]:
    out = {}
    for name in HEADS:
        head = getattr(rule, name, None)
        if head is None:
            out[name] = float("nan")
            continue
        total = 0.0
        for p in head.parameters():
            if p.grad is not None:
                total += float((p.grad**2).sum())
        out[name] = total**0.5
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default="1UBQ")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="measure the balance on these trained weights instead of at "
        "initialisation. **The reason this exists is written into the weights "
        "it produced.** The note recording BOND_WEIGHT, REACH_WEIGHT and "
        "COMPLETION_WEIGHT says in writing that they balance the terms 'at "
        "initialisation and nowhere else', that the bond term 'should shrink "
        "as the chain improves', and to re-measure a few thousand updates in. "
        "That was never done, and g1 and g2 then ran 9,000 updates between "
        "them with bond climbing 2.779 -> 3.903 against a crystal's 0.002 -- "
        "so the restraint was being weighted for a situation that had stopped "
        "being true.",
    )
    args = parser.parse_args()

    path = RAW / f"{args.pdb}.pdb"
    if not path.exists():
        print(f"{path} missing; run scripts/fetch_test_structures.py")
        return 1
    topo, native = feature_topology(load_protein(path))

    trained = None
    if args.checkpoint:
        from aanca import checkpoint as ckpt

        trained = ckpt.rule_weights(
            ckpt.load(Path(args.checkpoint), map_location="cpu")
        )

    print(
        f"{args.pdb}, {args.seeds} seeds, {args.steps} steps, "
        f"mean of {args.repeats} runs, every weight but one set to zero\n"
        f"rule: {args.checkpoint or 'at initialisation'}\n"
    )
    header = f"  {'term':<12}{'value':>9}" + "".join(f"{h.replace('_head',''):>12}" for h in HEADS)
    print(header)

    rows: dict[str, tuple[float, dict[str, float]]] = {}
    for name, kwargs in TERMS:
        values, grads = [], {h: [] for h in HEADS}
        for repeat in range(args.repeats):
            rule = AANCARule(
                growth=True, birth="cartesian", identity=True, identity_keep=2.0
            )
            if trained is not None:
                rule.load_state_dict(trained)
            config = RolloutConfig(
                steps=args.steps,
                truncate=args.steps,
                k=args.k,
                blind=True,
                growth=GrowthConfig(seeds=args.seeds, birth="cartesian"),
            )
            # Every weight to zero, then one back on. FAPE needs a weight for
            # this to be possible at all -- without ``fape_weight`` each row
            # measured "FAPE plus the term" and called it the term, and the
            # first version of this probe reported exactly that: completion's
            # gradient came out identical to FAPE's to three figures, because
            # it *was* FAPE's.
            off = {
                "fape_weight": 0.0,
                "bond_weight": 0.0,
                "reach_weight": 0.0,
                "completion_weight": 0.0,
                "identity_weight": 0.0,
            }
            # ``identity_weight`` has to be on for the identity track to exist
            # at all, so the FAPE-only case still carries labels; it is the
            # *loss* weight that is zeroed.
            off.update(kwargs)
            # The identity *track* has to exist for the labels to be carried at
            # all; it is the loss weight that is being zeroed, so the rollout
            # is identical in every row.
            carry = dict(off)
            if carry["identity_weight"] == 0.0:
                carry["identity_weight"] = 0.0
            terms, _result = growth_objective(
                rule, native, topo, config,
                torch.Generator().manual_seed(repeat),
                identity_start="masked",
                **off,
            )
            value = {
                "fape": terms.fape,
                "bond": terms.bond,
                "reach": terms.reach,
                "completion": terms.completion,
                "identity": terms.identity,
            }[name]
            values.append(float(value))
            rule.zero_grad()
            terms.total.backward()
            for h, g in gradients(rule).items():
                grads[h].append(g)
        mean_value = sum(values) / len(values)
        mean_grads = {h: sum(v) / len(v) for h, v in grads.items()}
        rows[name] = (mean_value, mean_grads)
        cells = "".join(f"{mean_grads[h]:>12.2e}" for h in HEADS)
        print(f"  {name:<12}{mean_value:>9.4f}{cells}")

    print("\n  values are the unweighted term; the rest are gradient norms.\n")

    # Equalise each term's total gradient against FAPE's.
    base = sum(rows["fape"][1][h] for h in HEADS if rows["fape"][1][h] == rows["fape"][1][h])
    print("  weight that would match FAPE's total gradient contribution:")
    for name, (_value, grads) in rows.items():
        if name == "fape":
            continue
        total = sum(g for g in grads.values() if g == g)
        if total <= 0:
            print(f"    {name:<12} no gradient at initialisation")
            continue
        print(f"    {name:<12}{base / total:>10.3g}")
    print(
        "\n  These are a starting point, not an answer: they balance the terms\n"
        "  at initialisation, where three of the four heads are zeroed and the\n"
        "  structure is barely grown. A term whose job is to bite later -- the\n"
        "  reach restraint is the clear case, since it reads exactly zero until\n"
        "  a closure is actually foreclosed -- will be underweighted by this\n"
        "  procedure. Re-measure a few thousand updates in."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
