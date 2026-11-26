"""Train the E(n)-GNCA denoiser with a GLOBAL FOLD-CLASS LABEL. A diagnostic arm.

    .venv/Scripts/python.exe scripts/train_conditioned.py configs/conditioned.toml --section cond
    .venv/Scripts/python.exe scripts/train_conditioned.py configs/conditioned.toml --list-sections

**Read the pre-registration before the table**:
``docs/claims/a-global-fold-class-label-tells-us-whether-option-2-failed-on-ambiguity-or-on-address.md``.
It is binding on everything here and it registers the expectation that **this
arm does not help**. Its value is that it says so cheaply and sends the next arm
to the frame head for a measured reason instead of a hopeful one.

**It cannot make option 2 true.** 5,328 parameters and 400 structures do not
become a protein generator because a label is added. What this arm can do is
discriminate two diagnoses of why
``docs/claims/a-local-rule-with-an-attractor-is-a-diffusion-model-without-a-schedule.md``
was refuted:

  * **H-ambiguity** -- one rule with no timestep and no conditioning was asked
    to map one Gaussian prior onto the whole distribution of folds, so the best
    available map satisfies the average pair constraint and nothing else.
    Narrow the target and the constraints become mutually satisfiable.
  * **H-address** -- the deficit is positional. A residue's only positional
    information is its sequence index, a 1-D address for a 3-D target, and **no
    amount of conditioning supplies an address**.

A fold class is identical for every residue in the structure, so it adds exactly
**zero** positional information. That is what makes it a discriminator rather
than another lever: under H-ambiguity the conditioning moves the pair loss and
the geometry; under H-address it changes the output distribution but not its
quality.

**A THIRD TRAINER, not a flag and not a key.** ``scripts/train_denoise.py``,
``scripts/train_egnca.py``, ``src/aanca/egnca.py`` and ``configs/egnca.toml``
stay bit-for-bit: both are quoted results, and ``runs/mvp/best.pt`` must stay
readable by ``scripts/prior_fixed_point.py`` and ``scripts/denoise_floor.py``,
which refuse a checkpoint missing any key their trainer's ``build_schema()``
defines. A key added to the denoiser's schema would break both probes against
the refuted MVP's own checkpoint. So this is a new trainer with a new schema,
and **everything it can reuse it imports by path** -- ``prior_coordinates``,
``run_to_fixed_point``, ``corrupt``, ``context_mask``, ``draw_sigma``,
``denoise_objective``, ``write_samples``, ``Sample``, ``DenoiseTerms``,
``skip_report``, ``on_disk`` -- so a difference in a result is the label and not
a sixth thing.

**Everything else is ``configs/denoise.toml``'s ``mvp`` section, identical**:
``context_fraction = 0.0``, sigma log-uniform [0.5, 30], no timestep, live k-NN,
15-25 step rollouts with full BPTT, replay pool 0.5, lr 1e-3, clip 1.0, weight
decay 1e-5, the widths, crop 128, ``k`` 24, cutoff 10.0, 20,000 updates,
evaluation every 250.

**THE LABEL.** ``generation_metrics.structure_class`` over P-SEA helix and
strand fractions: ``mainly-alpha``, ``mainly-beta``, ``alpha/beta``,
``irregular``. No network, no CATH download, no new dependency -- this project's
own validated code (1.000 per-residue agreement with biotite's ``annotate_sse``
on all four crystal structures in ``data/raw/``). **It is a P-SEA-derived
C-level analogue and must be named as ours**: no claim from this arm may say
"CATH class".

Two honest costs, registered before the arm ran:

  * **The label is computed on the CROP, not the chain.** A 128-residue crop of
    a larger protein can carry a different class from its parent, and the crop
    is what is supervised. A number from this arm is about crops.
  * **``irregular`` is rare.** The class counts over the training structures are
    printed once at startup, because a class with too few examples to train is a
    reported fact rather than a silent imbalance.

Where it enters is :mod:`aanca.conditioned`, which argues the three alternatives
it rules out. **Label dropout** at ``label_dropout`` replaces the class with a
learned unconditional row, so classifier-free guidance is available later
without retraining; **CFG is not implemented here** and a guidance scale is a
second axis.

**READOUT 0 IS TWO THINGS, AMENDED 26 NOV, AND THIS TRAINER PRINTS ONLY THE
SECOND.** As first registered it was the single ``helix a/b`` contrast -- the
P-SEA helix fraction of samples drawn under ``mainly-alpha`` against those drawn
under ``mainly-beta``, from the same noise seeds and the same crops -- with a
flat result declared a bug that blocked every other number. **That form could
not function**: there is a third state, the output carrying no secondary
structure at all, in which the gap is ``0.000 - 0.000`` whether the label is
live or dead, and that state is the EXPECTED one for this rule.

  * **0a, the instrument check, is the PAIRED PER-RESIDUE DISPLACEMENT** between
    samples differing in nothing but the label. It is not printed here -- it is
    computed from ``runs/<tag>/samples.npz`` by ``scripts/cond_readouts.py``,
    which is where the verdict is taken. A zero there is a bug and nothing else
    may be read until it is fixed.
  * **0b, ``helix a/b`` below, is a RESULT**, readable only once the samples
    carry any secondary structure. A flat 0b with a live 0a is not an instrument
    failure: it is H-address's prediction, that the conditioning arrives, moves
    the coordinates, and buys nothing.

Measured on the 20,000-update run: 0a was 1.80 A mean (2.85 A alpha-vs-beta) and
0b was ``+0.000`` with every sample produced ``irregular``.

``n`` and ``diverged`` come before every other column, this project's standing
rule, and ``sigma`` and ``start_rmsd`` are reported so they can be checked
single-valued.

**Exit code 64 means "checkpointed, not finished"**, as in
``scripts/train_denoise.py``: ``scripts/bunya_train.slurm`` resubmits its chain
only on 64. ``train_egnca.py`` and ``train_growth.py`` return 0 on that path,
which ended a chain in silence; that is not copied here and it is not fixed here
either.

**``history.json`` carries ``null``, never a bare ``NaN``.**
``scripts/train_denoise.py`` writes Python's ``float('nan')`` straight through
``json.dumps``, which emits a bare ``NaN`` token that is not valid JSON and
broke a reader. :func:`json_safe` converts every non-finite float to ``None``
before the dump.

Nothing in this docstring is an expected value.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aanca import checkpoint as ckpt  # noqa: E402
from aanca import config as run_config  # noqa: E402
from aanca.conditioned import (  # noqa: E402
    FOLD_CLASSES,
    LABEL_DIM,
    UNCONDITIONAL,
    ConditionedEGNCARule,
    class_index,
    class_name,
    conditioning_cost,
    effective_label,
)
from aanca.data import RepackingDataset, read_manifest  # noqa: E402
from aanca.egnca import ACTIVATIONS, NORM_TYPES, EGNCAPoolEntry  # noqa: E402
from aanca.egnca import ca_positions  # noqa: E402


def _module(name: str):
    """A ``scripts/`` file as a module -- ``scripts/`` is not a package.

    ``scripts/prior_fixed_point.py``'s pattern, and the registration in
    ``sys.modules`` before execution is load-bearing: ``train_denoise`` defines
    dataclasses, and ``@dataclass`` looks its own class's module up by name.
    """
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


#: The refuted MVP's trainer, **imported rather than copied**. Every piece of
#: the corruption, the rollout, the prior and the sample writer comes from here,
#: so this arm's numbers differ from the MVP's by the label alone.
denoise = _module("train_denoise")

#: The metric suite, for the label and for readout 0. ``structure_class`` is
#: *called*, never reimplemented: the thresholds live in one place and
#: ``tests/test_conditioned.py`` asserts the agreement by calling it.
metrics = _module("generation_metrics")

MANIFEST = denoise.MANIFEST
PDB = denoise.PDB
RUNS = denoise.RUNS
CHECKPOINTED_NOT_FINISHED = denoise.CHECKPOINTED_NOT_FINISHED
SKIP_REASONS = denoise.SKIP_REASONS
RG_PREFACTOR = denoise.RG_PREFACTOR
RG_EXPONENT = denoise.RG_EXPONENT
_SkipUpdate = denoise._SkipUpdate
skip_report = denoise.skip_report
Sample = denoise.Sample
DenoiseTerms = denoise.DenoiseTerms
denoise_objective = denoise.denoise_objective
#: Bound at module level, unlike ``context_mask`` and ``draw_sigma``, because
#: ``scripts/denoise_floor.py`` reaches for ``trainer.corrupt`` by name: the
#: floor is weight-free, so its only contact with a trainer is the corruption
#: and the held-out split. Rebinding it here is what lets that script measure
#: this arm's checkpoints with nothing but ``--trainer`` -- and it is the SAME
#: function object, so the conditioned arm's floor cannot drift from the MVP's.
corrupt = denoise.corrupt
prior_coordinates = denoise.prior_coordinates
run_to_fixed_point = denoise.run_to_fixed_point
on_disk = denoise.on_disk
resolve_device = denoise.resolve_device


# -- the label, computed on the crop ----------------------------------------


def crop_label(native_coords, topo) -> tuple[str, float, float]:
    """``(class, helix_fraction, strand_fraction)`` for one crop's native CA.

    **On the crop and not the chain**, which is the first of the two costs the
    pre-registration registers: the crop is what is supervised, so the crop is
    what is labelled, and a 128-residue window of a larger protein can carry a
    different class from its parent.

    ``generation_metrics.secondary_metrics`` does the work -- P-SEA, the two
    fractions, and ``structure_class`` over them. Nothing here reimplements a
    threshold.
    """
    ca = ca_positions(native_coords, topo).detach().cpu().to(torch.float64).numpy()
    found = metrics.secondary_metrics(np.ascontiguousarray(ca))
    return (
        str(found["class"]),
        float(found["helix_fraction"]),
        float(found["strand_fraction"]),
    )


class LabelCache:
    """``(structure index, crop seed) -> class index``, computed once.

    P-SEA on 128 residues is fast enough that correctness beats cleverness, so
    this is a plain dict with no eviction and no persistence: the key is the
    same ``(index, crop_seed)`` pair the replay pool uses, because
    ``load_crop`` re-crops on every visit and a label kept under the index alone
    would label a crop it was never computed on.
    """

    def __init__(self) -> None:
        self.by_key: dict[tuple[int, int | None], int] = {}
        self.counts: Counter[str] = Counter()

    def label_for(self, key, native_coords, topo) -> int:
        found = self.by_key.get(key)
        if found is None:
            name, _helix, _strand = crop_label(native_coords, topo)
            found = class_index(name)
            self.by_key[key] = found
            self.counts[name] += 1
        return found


def class_counts(dataset, indices, crop_size_seed: int) -> tuple[Counter, int]:
    """The class counts over the training structures, one crop each.

    Printed once at startup and never averaged into anything: **a class with too
    few examples to train is a reported fact rather than a silent imbalance**,
    which is the pre-registration's own wording. One crop per structure at a
    reproducible seed, so the counts are a property of the split and the seed
    rather than of when the run started.

    Returns the counts and how many structures could not be cropped at all --
    the second number is not a diagnostic either: a split that loses a quarter
    of itself to ``no_crop`` is training on 300 structures and not 400.
    """
    counts: Counter[str] = Counter()
    unreadable = 0
    rng = random.Random(crop_size_seed)
    for index in indices:
        try:
            topo, native, _crop, _seed = dataset.load_crop(index, rng=rng)
        except Exception:
            unreadable += 1
            continue
        if topo.n_residues < 3:
            unreadable += 1
            continue
        name, _helix, _strand = crop_label(native, topo)
        counts[name] += 1
    return counts, unreadable


def class_counts_report(counts: Counter, unreadable: int, total: int) -> str:
    """The startup line, every class named even at zero.

    Every class, including the ones with no examples: a missing column reads as
    a class that was never asked about, and ``irregular`` being absent is the
    single most likely way this arm's confusion table becomes unreadable.
    """
    scored = sum(counts.values())
    body = "  ".join(
        f"{name} {counts.get(name, 0)}"
        + (
            f" ({100.0 * counts.get(name, 0) / scored:.1f}%)"
            if scored
            else ""
        )
        for name in FOLD_CLASSES
    )
    out = f"class counts over {scored} of {total} training structures: {body}"
    if unreadable:
        out += f"  [{unreadable} could not be cropped]"
    thin = [name for name in FOLD_CLASSES if counts.get(name, 0) < 10]
    if thin:
        out += (
            "\n  TOO FEW TO TRAIN: "
            + ", ".join(thin)
            + " -- a reported fact and not a silent imbalance. The natural "
            "60-entry reference reads 35 alpha/beta, 14 mainly-beta, 10 "
            "mainly-alpha, 1 irregular, so this is expected rather than a bug, "
            "and the confusion table in readout 5 must be read with it."
        )
    return out


# -- the objective, which is the denoiser's with the label set ---------------


def conditioned_objective(
    rule: ConditionedEGNCARule,
    native_coords,
    topo,
    steps: int,
    label: int,
    label_dropout: float,
    dropout_draw: float | None = None,
    **kwargs,
):
    """``train_denoise.denoise_objective`` with the label set for the rollout.

    **The objective itself is not rewritten.** The label is state on the rule,
    so the imported corruption, rollout and loss run unchanged and the only
    difference from the MVP is which embedding row ``phi_m`` reads. That is what
    makes a comparison with ``configs/denoise.toml``'s ``mvp`` a comparison of
    one thing.

    ``dropout_draw`` is a uniform ``[0, 1)`` sample at training time and
    **None** at evaluation, where the condition is the question being asked.
    """
    row = effective_label(label, label_dropout, dropout_draw)
    with rule.conditioned_on(row):
        terms, state = denoise_objective(
            rule, native_coords, topo, steps=steps, **kwargs
        )
    return terms, state, row


# -- sampling: every class in turn, from the same noise seeds ----------------


@dataclass
class ConditionedSample:
    """One :class:`train_denoise.Sample` and the class it was conditioned on.

    Not a subclass, because ``Sample``'s fields have no defaults and a subclass
    adding one would have to default every field it inherits. The sample is held
    rather than flattened so :func:`train_denoise.write_samples` stays the one
    definition of the ``.npz`` layout.
    """

    sample: Sample
    #: The row actually conditioned on, **after dropout** -- ``UNCONDITIONAL``
    #: for every sample of the ``nolabel`` control, which is why readout 0 is
    #: flat there by construction and not by failure.
    label: int
    #: What the sample ACTUALLY came out as, by the same ``structure_class``
    #: that produced the label. Readout 5's confusion table is this against
    #: ``label``.
    produced: str = ""
    helix_fraction: float = float("nan")
    strand_fraction: float = float("nan")


def write_samples(path: Path, samples: list[ConditionedSample], update: int) -> None:
    """ONE ``runs/<tag>/samples.npz``, in the denoiser's own flat layout.

    The coordinates, offsets, steps, convergence, last move and radius of
    gyration are written by :func:`train_denoise.write_samples` itself -- the
    flat ``(total, 3)`` plus ``offsets`` layout is defined in exactly one place
    and cannot drift -- and this function then reopens that single file and adds
    the four conditioning columns. **One file, not one per sample and not two**:
    the user scratch quota on Bunya has ~313k inodes free.

    ``label`` is the row conditioned on and ``produced`` the class that came
    out, so readout 5's confusion table is computable from the ``.npz`` alone
    without rerunning the sampler.
    """
    denoise.write_samples(path, [c.sample for c in samples], update)
    with np.load(path) as blob:
        stored = {name: blob[name] for name in blob.files}
    stored["label"] = np.asarray([c.label for c in samples], dtype=np.int64)
    stored["conditioned_class"] = np.asarray(
        [class_name(c.label) for c in samples], dtype=object
    ).astype("U")
    stored["produced_class"] = np.asarray(
        [c.produced for c in samples], dtype=object
    ).astype("U")
    stored["helix_fraction"] = np.asarray(
        [c.helix_fraction for c in samples], dtype=np.float32
    )
    stored["strand_fraction"] = np.asarray(
        [c.strand_fraction for c in samples], dtype=np.float32
    )
    stored["fold_classes"] = np.asarray(FOLD_CLASSES, dtype=object).astype("U")
    np.savez_compressed(path, **stored)


def sample_batch(
    rule: ConditionedEGNCARule, dataset, indices, args, seed: int = 0
) -> list[ConditionedSample]:
    """``sample_count`` draws, **equal numbers conditioned on each class in turn**.

    **The noise seeds and the crops are shared across the classes**, so the
    ``mainly-alpha`` against ``mainly-beta`` contrast of readout 0 is paired:
    position ``p`` uses crop ``indices[p % len(indices)]`` and generator seed
    ``seed * 7_919 + p`` for *every* class, so a difference between two classes'
    helix fractions is the label and not the draw. Unpaired, with 2 draws per
    class, the contrast would be mostly which cloud each class happened to get.

    ``sample_count`` is rounded **down** to a multiple of the class count: an
    unequal split would make the contrast partly a difference in sample size,
    and silently sampling 3 alphas against 2 betas is exactly the kind of
    imbalance readout 0 cannot absorb.
    """
    rule.eval()
    per_class = args.sample_count // len(FOLD_CLASSES)
    out: list[ConditionedSample] = []
    for position in range(per_class):
        index = indices[position % len(indices)]
        rng = random.Random(seed * 7_919 + position)
        try:
            topo, native, _crop, _used = dataset.load_crop(index, rng=rng)
        except Exception:
            continue
        if topo.n_residues < 3:
            continue
        for label in range(len(FOLD_CLASSES)):
            generator = torch.Generator().manual_seed(seed * 7_919 + position)
            start = prior_coordinates(
                topo.n_residues,
                generator=generator,
                dtype=rule.dtype,
                device=native.device,
            )
            row = effective_label(label, args.label_dropout, None)
            with rule.conditioned_on(row):
                sample = run_to_fixed_point(
                    rule,
                    topo,
                    native,
                    start,
                    tolerance=args.sample_tolerance,
                    max_steps=args.sample_max_steps,
                    k=args.k,
                    cutoff=args.cutoff,
                    device_neighbours=args.device_neighbours,
                    native_edges=args.native_edges,
                )
            produced, helix, strand = "", float("nan"), float("nan")
            if bool(torch.isfinite(sample.ca).all()):
                trace = sample.ca.detach().cpu().to(torch.float64).numpy()
                found = metrics.secondary_metrics(np.ascontiguousarray(trace))
                produced = str(found["class"])
                helix = float(found["helix_fraction"])
                strand = float(found["strand_fraction"])
            out.append(
                ConditionedSample(
                    sample=sample,
                    label=row,
                    produced=produced,
                    helix_fraction=helix,
                    strand_fraction=strand,
                )
            )
    rule.train()
    return out


def helix_contrast(samples: list[ConditionedSample]) -> dict[str, float]:
    """READOUT 0b: mean P-SEA helix fraction under ``mainly-alpha`` vs ``mainly-beta``.

    **This is a result and not the instrument check** -- amended 26 Nov. A flat
    gap has two causes and this number cannot tell them apart: a conditioning
    that never arrives, and an output with no secondary structure for a label to
    shift. The second is this rule's ordinary state, so a flat gap here is read
    only against readout 0a, the paired per-residue displacement, which is
    computed by ``scripts/cond_readouts.py`` from the sample file.

    ``n_alpha`` and ``n_beta`` travel with the two means because a contrast over
    two samples each is not an error bar, and a mean over zero samples would
    otherwise print as ``nan`` with nothing saying why.
    """
    alpha = class_index("mainly-alpha")
    beta = class_index("mainly-beta")
    picked = {
        alpha: [
            c.helix_fraction
            for c in samples
            if c.label == alpha and c.helix_fraction == c.helix_fraction
        ],
        beta: [
            c.helix_fraction
            for c in samples
            if c.label == beta and c.helix_fraction == c.helix_fraction
        ],
    }
    mean = lambda rows: sum(rows) / len(rows) if rows else float("nan")  # noqa: E731
    return {
        "helix_alpha": mean(picked[alpha]),
        "helix_beta": mean(picked[beta]),
        "helix_gap": mean(picked[alpha]) - mean(picked[beta]),
        "n_alpha": float(len(picked[alpha])),
        "n_beta": float(len(picked[beta])),
    }


def confusion(samples: list[ConditionedSample]) -> dict[str, int]:
    """READOUT 5, as ``"<conditioned> -> <produced>": count``.

    Computed here as well as being recoverable from the ``.npz`` so the
    checkpoint's ``scores`` carry it: this is the one readout that could be a
    positive result on its own, because a rule that reliably produces more helix
    when asked for helix has learned something about secondary structure even if
    its bonds are wrong.
    """
    table: Counter[str] = Counter()
    for entry in samples:
        if entry.produced:
            table[f"{class_name(entry.label)} -> {entry.produced}"] += 1
    return dict(sorted(table.items()))


# -- the trainer -------------------------------------------------------------


def build_schema() -> argparse.ArgumentParser:
    """Every setting the trainer reads, with its type and its default.

    ``scripts/train_denoise.py``'s schema plus exactly two keys, and the rest
    identical **by value** rather than by import: the denoiser's schema is read
    back by ``scripts/prior_fixed_point.py`` and ``scripts/denoise_floor.py``
    against ``runs/mvp/best.pt``, and building this one on top of that object
    would make either probe's refusal depend on which trainer had been imported
    first. ``tests/test_conditioned.py`` asserts the overlap key for key, so the
    duplication cannot drift into a difference nobody declared.
    """
    p = argparse.ArgumentParser(add_help=False)

    p.add_argument("--split", default="train")
    p.add_argument("--structures", type=int, default=400)
    p.add_argument("--eval-structures", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)

    # The rule. Gala's widths, unchanged.
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--message-dim", type=int, default=32)

    # The five settings the paper omits, at the authors' defaults.
    p.add_argument("--has-coord-act", action="store_true", default=True)
    p.add_argument("--is-residual", action="store_true", default=True)
    p.add_argument("--activation", default="tanh", choices=tuple(ACTIVATIONS))
    p.add_argument("--norm-type", default="pn", choices=NORM_TYPES)

    # THE ONLY TWO NEW KEYS IN THIS ARM, and both change a result, so neither
    # is a flag (``src/aanca/config.py``'s rule).
    #
    # ``label_dim`` is the embedding width. 4 over five rows, which with the
    # widened ``phi_m`` input is ~148 parameters against 5,328 -- the smallness
    # is load-bearing for the framing, not a tuning choice.
    p.add_argument("--label-dim", type=int, default=LABEL_DIM)
    # The probability of replacing the class with the learned unconditional row,
    # so an unconditional pass exists and classifier-free guidance is available
    # later without retraining. 1.0 is the ``nolabel`` control: the class never
    # reaches the rule, in training or in evaluation.
    p.add_argument("--label-dropout", type=float, default=0.1)

    # The rollout, as published.
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--steps-min", type=int, default=15)

    # The task. Identical to the denoiser's MVP.
    p.add_argument("--context-fraction", type=float, default=0.0)
    p.add_argument("--sigma-min", type=float, default=0.5)
    p.add_argument("--sigma-max", type=float, default=30.0)

    # Perception. Live k-NN, as in the MVP.
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--device-neighbours", action="store_true")
    p.add_argument("--native-edges", action="store_true", default=False)

    # The objective.
    p.add_argument("--pairwise-weight", type=float, default=1.0)
    p.add_argument("--pairwise-window-weight", type=float, default=1.0)

    # Training.
    p.add_argument("--updates", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--pool-probability", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=250)

    # Sampling from the prior, equal numbers per class.
    p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--sample-tolerance", type=float, default=0.01)
    p.add_argument("--sample-max-steps", type=int, default=200)

    for dest in run_config.RUNTIME_ONLY:
        p.add_argument(f"--{dest.replace('_', '-')}", default=None)
    return p


#: The keys this schema adds to ``train_denoise.build_schema()``'s. Named so
#: ``tests/test_conditioned.py`` can assert the rest is identical key for key
#: and this arm cannot quietly change a setting the MVP comparison rests on.
NEW_KEYS = ("label_dim", "label_dropout")


def build_parser() -> argparse.ArgumentParser:
    """Only probes, paths and resources on the command line.

    Nothing here changes a result: ``--device`` names hardware,
    ``--max-seconds`` is the allocation, ``--stop-file`` and ``--resume`` are
    the Slurm chain's business, and ``--diff`` and ``--list-sections`` compute
    nothing.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--section", default=None)
    p.add_argument("--list-sections", action="store_true")
    p.add_argument("--diff", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--stop-file", default=None)
    p.add_argument("--resume", action="store_true")
    return p


def recorded_args(args: argparse.Namespace) -> dict:
    """What the run *is*, for the checkpoint: settings without the resources."""
    return {
        key: value
        for key, value in vars(args).items()
        if key not in run_config.RUNTIME_ONLY
    }


def build_rule(args) -> ConditionedEGNCARule:
    """The rule, built from the section. One call, so the keys cannot miss it."""
    return ConditionedEGNCARule(
        hidden_dim=args.hidden_dim,
        message_dim=args.message_dim,
        has_coord_act=args.has_coord_act,
        is_residual=args.is_residual,
        activation=args.activation,
        norm_type=args.norm_type,
        label_dim=args.label_dim,
    )


def objective_kwargs(args) -> dict:
    """Every setting that reaches :func:`conditioned_objective`.

    One dict for the training step and the held-out evaluation both, on
    ``train_denoise.objective_kwargs``' reason: this project's record carries a
    retraction from weights that reached training and not evaluation (job
    32405172), and there is no second place here for a key to go missing from.
    ``label_dropout`` is in it; the per-sample ``label`` and ``dropout_draw``
    are not, because they are properties of the crop and the update.
    """
    return dict(
        label_dropout=args.label_dropout,
        context_fraction=args.context_fraction,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        pairwise_weight=args.pairwise_weight,
        pairwise_window_weight=args.pairwise_window_weight,
        k=args.k,
        cutoff=args.cutoff,
        device_neighbours=args.device_neighbours,
        native_edges=args.native_edges,
    )


#: The averaged columns, the denoiser's own plus readout 0's five. ``n`` and
#: ``diverged`` are added by :func:`score` and come first in the printed line.
KEYS = denoise.KEYS


def json_safe(value):
    """Every non-finite float replaced by ``None``, recursively.

    **``scripts/train_denoise.py``'s one reproduced bug, not reproduced.** It
    dumps ``rmsd_fit_context`` -- which is ``nan`` for every sample at
    ``context_fraction = 0.0``, so every row of its ``history.json`` -- straight
    through ``json.dumps``, which emits a bare ``NaN`` token. That is not valid
    JSON by the spec and a strict parser refuses the whole file, which is how a
    reader of that history broke. ``null`` is the honest encoding of a quantity
    that does not exist, and it round-trips.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def score(rule, dataset, indices, args, labels: LabelCache, seed: int = 0) -> dict:
    """The held-out objective and its parts, averaged over crops that scored.

    ``train_denoise.score``'s structure, with the crop's own class set on the
    rule and **no dropout**: at evaluation the condition is the question being
    asked, so perturbing it would make the column a mixture of two models. The
    ``nolabel`` control is still unconditional here, because
    :func:`aanca.conditioned.effective_label` makes ``label_dropout >= 1.0``
    unconditional whether a draw is offered or not.

    ``n`` and ``diverged`` are part of the readout and not diagnostics: this
    rule is unbounded and its authors report exploding gradients, so a
    non-finite rollout is an ordinary outcome and a mean over the survivors
    printed without saying so would be the best-looking column in the log.
    """
    rule.eval()
    sums = {key: 0.0 for key in KEYS}
    counted = {key: 0 for key in KEYS}
    n = 0
    diverged = 0
    rng = random.Random(seed)
    with torch.no_grad():
        for index in indices:
            try:
                topo, native, _crop, crop_seed = dataset.load_crop(index, rng=rng)
            except Exception:
                continue
            if topo.n_residues < 3:
                continue
            label = labels.label_for((index, crop_seed), native, topo)
            generator = torch.Generator().manual_seed(seed + index)
            terms, _state, _row = conditioned_objective(
                rule,
                native,
                topo,
                steps=args.steps,
                label=label,
                generator=generator,
                dropout_draw=None,
                **objective_kwargs(args),
            )
            if terms.rmsd is None or not torch.isfinite(terms.total):
                diverged += 1
                continue
            for key, value in (
                ("total", float(terms.total)),
                ("pair_anchor", float(terms.pair_anchor)),
                ("pair_window", float(terms.pair_window)),
                ("sigma", terms.sigma),
                ("rmsd", terms.rmsd),
                ("rmsd_fit_context", terms.rmsd_fit_context),
                ("start_rmsd", terms.start_rmsd),
                ("move", terms.move),
            ):
                # ``rmsd_fit_context`` is nan whenever fewer than three residues
                # are held, which is every sample of this arm. Averaging nan in
                # would turn the whole row into nan, so it is counted out of its
                # own mean only -- and written as ``null``, never ``NaN``.
                if value is None or value != value:
                    continue
                sums[key] += value
                counted[key] += 1
            n += 1
    rule.train()
    tail = {"n": float(n), "diverged": float(diverged)}
    if not n:
        return {key: float("nan") for key in KEYS} | tail
    return {
        key: (sums[key] / counted[key] if counted[key] else float("nan"))
        for key in KEYS
    } | tail


def report(scores: dict, samples: list[ConditionedSample] | None) -> str:
    """One line. ``train_denoise.report``'s columns, then readout 0.

    ``n`` and ``diverged`` first, per this project's standing rule; ``sigma``
    and ``start`` present so they can be checked single-valued; then
    ``helix a/b``, which is the instrument check and is **read before every
    other column on the line**.
    """
    out = (
        f"n {int(scores['n'])}  diverged {int(scores['diverged'])}  "
        f"total {scores['total']:9.3f}  anchor {scores['pair_anchor']:9.3f}  "
        f"window {scores['pair_window']:9.3f}  sigma {scores['sigma']:6.3f}  "
        f"rmsd {scores['rmsd']:7.3f}  (fit_ctx {scores['rmsd_fit_context']:7.3f}, "
        f"start {scores['start_rmsd']:7.3f})  move {scores['move']:6.3f}"
    )
    if samples:
        reached = [c for c in samples if c.sample.converged]
        mean_steps = sum(c.sample.steps for c in samples) / len(samples)
        mean_rg = sum(c.sample.radius_of_gyration for c in samples) / len(samples)
        out += (
            f"  steps {mean_steps:6.1f}  conv {len(reached)}/{len(samples)}  "
            f"rg {mean_rg:6.2f}"
        )
        gap = helix_contrast(samples)
        out += (
            f"  helix a/b {gap['helix_alpha']:.3f}/{gap['helix_beta']:.3f} "
            f"(gap {gap['helix_gap']:+.3f}, n {int(gap['n_alpha'])}/"
            f"{int(gap['n_beta'])})"
        )
    return out


def main() -> int:
    runtime = build_parser().parse_args()
    schema = build_schema()

    data = run_config.load(runtime.config)
    if runtime.list_sections:
        for name in run_config.sections(data):
            print(name)
        return 0
    if not runtime.section:
        print(
            "error: --section is required. This config defines: "
            + (", ".join(run_config.sections(data)) or "(none)")
        )
        return 2
    run_config.check_base_is_complete(data, schema)
    args = run_config.to_namespace(
        schema, run_config.resolve(data, runtime.section), runtime.section, runtime
    )

    if runtime.diff:
        other = run_config.to_namespace(
            schema, run_config.resolve(data, runtime.diff), runtime.diff, runtime
        )
        rows = [
            (key, getattr(other, key), getattr(args, key))
            for key in sorted(vars(args))
            if key != "tag" and getattr(args, key) != getattr(other, key)
        ]
        print(f"{runtime.diff} -> {runtime.section}")
        for key, was, now in rows:
            print(f"  {key}: {was!r} -> {now!r}")
        print(f"  {len(rows)} setting(s) differ")
        return 0

    if args.steps_min > args.steps:
        print(
            f"steps_min {args.steps_min} exceeds steps {args.steps}: the "
            "rollout length is drawn uniformly from [steps_min, steps] and an "
            "empty range cannot be drawn from."
        )
        return 2
    if not 0.0 <= args.context_fraction <= 1.0:
        print(f"context_fraction must be in [0, 1], got {args.context_fraction}")
        return 2
    if args.sigma_min <= 0 or args.sigma_max < args.sigma_min:
        print(
            f"sigma band [{args.sigma_min}, {args.sigma_max}] is not a "
            "positive, ordered interval, and the draw is log-uniform."
        )
        return 2
    if not 0.0 <= args.label_dropout <= 1.0:
        print(f"label_dropout must be in [0, 1], got {args.label_dropout}")
        return 2
    if args.label_dim < 1:
        print(f"label_dim must be positive, got {args.label_dim}")
        return 2
    if args.sample_count and args.sample_count < len(FOLD_CLASSES):
        print(
            f"sample_count {args.sample_count} is below the {len(FOLD_CLASSES)} "
            "classes, so readout 0's paired alpha-versus-beta contrast cannot "
            "be drawn. Use 0 to turn sampling off or a multiple of "
            f"{len(FOLD_CLASSES)}."
        )
        return 2

    print(f"config     : {runtime.config} [{runtime.section}]")
    print(
        "E(n)-GNCA DENOISER + GLOBAL FOLD-CLASS LABEL. A DIAGNOSTIC arm, "
        "pre-registered at\n  docs/claims/a-global-fold-class-label-tells-us-"
        "whether-option-2-failed-on-ambiguity-or-on-address.md\n  with the "
        "expectation that it does NOT help. It cannot make option 2 true; it "
        "separates\n  H-ambiguity (too many targets) from H-address (no "
        "positional information), because a\n  fold class is the same number "
        "for every residue and so adds exactly ZERO address."
    )
    print(
        "  THE LABEL: generation_metrics.structure_class over P-SEA helix and "
        "strand fractions,\n  computed ON THE CROP (the crop is what is "
        "supervised, so a number here is about crops).\n  A P-SEA-derived "
        "C-level ANALOGUE and ours -- not a CATH assignment, and no claim from "
        "this\n  arm may say \"CATH class\". No network and no new dependency."
    )
    print(
        f"  WHERE IT ENTERS: a {args.label_dim}-wide learned embedding of the "
        "class in phi_m's INVARIANT input,\n  broadcast to every residue, at "
        "every step, beside ||x_i - x_j||^2 and the two hidden\n  states. Not "
        "the initial hidden state (which would confound preservation with "
        "conditioning)\n  and not added to h (norm_type = pn is PairNorm and "
        "would normalise a constant away)."
    )
    print(
        f"  LABEL DROPOUT {args.label_dropout:.0%}: that share of training "
        "passes read a learned UNCONDITIONAL row\n  instead, so "
        "classifier-free guidance is available later without retraining. CFG "
        "is NOT\n  implemented in this arm -- it needs two passes and a "
        "guidance scale, which is a second axis."
        + (
            "\n  THIS IS THE nolabel CONTROL: dropout 1.0, so the class NEVER "
            "reaches the rule, in\n  training or in evaluation. It should "
            "reproduce the MVP's behaviour up to the extra\n  parameters, and "
            "readout 0 is flat here BY CONSTRUCTION rather than by failure."
            if args.label_dropout >= 1.0
            else ""
        )
    )
    print(
        "  READOUT 0b, `helix a/b`: the P-SEA helix fraction of samples drawn "
        "under mainly-alpha\n  against mainly-beta, from the SAME noise seeds "
        "and crops. It is a RESULT, not the\n  instrument check, and it is "
        "readable only once the samples carry any secondary\n  structure at "
        "all -- a flat gap over an all-coil output says nothing about the "
        "label.\n  READOUT 0a, the instrument check, is the paired "
        "per-residue displacement between\n  samples differing only in the "
        "label; it is NOT printed here. Run\n  `scripts/cond_readouts.py "
        "--run runs/<tag>` for it, and read it before 0b."
    )
    print(
        "  Everything else is configs/denoise.toml's mvp, identical: "
        f"context_fraction {args.context_fraction:g}, sigma log-uniform "
        f"[{args.sigma_min:g}, {args.sigma_max:g}] A, NO timestep, "
        + ("LIVE" if not args.native_edges else "NATIVE")
        + " k-NN edges.\n  It places CA positions and produces NO residue "
        "orientation, so fape, twist, bond, clash\n  and identity are not "
        "reported. Read rmsd against start_rmsd at the same sigma."
    )

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    splits = read_manifest(MANIFEST)
    if args.split not in splits:
        print(f"no split {args.split!r}; available: {sorted(splits)}")
        return 1
    available = on_disk(splits[args.split])
    if not available:
        print(f"no coordinates in {PDB} -- run scripts/fetch_structures.py")
        return 1

    wanted = args.structures + args.eval_structures
    catalogue = available[:wanted] if len(available) >= wanted else available
    dataset = RepackingDataset(
        catalogue,
        concentrations=(0.0,),
        crop_size=args.crop_size,
        device=device,
        cache_limit=max(len(catalogue), 1),
    )
    indices = list(range(len(catalogue)))
    random.Random(args.seed).shuffle(indices)
    eval_indices = indices[: args.eval_structures]
    train_indices = indices[args.eval_structures :]
    if not train_indices:
        print("no training entries left after holding some out")
        return 1

    rule = build_rule(args).to(device)
    optimiser = torch.optim.Adam(
        rule.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    out_dir = RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_text = run_config.snapshot(
        args, args.tag, runtime.config, trainer="scripts/train_conditioned.py"
    )
    snapshot_path = out_dir / "config.toml"
    if (
        snapshot_path.exists()
        and snapshot_path.read_text(encoding="utf-8") != snapshot_text
    ):
        snapshot_path = out_dir / "config.resumed.toml"
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    samples_path = out_dir / "samples.npz"
    stop_file = Path(args.stop_file) if args.stop_file else out_dir / "STOP"

    start_update = 0
    best = float("inf")
    resumed_skips: dict = {}
    if args.resume and last_path.exists():
        state = ckpt.load(last_path, map_location=device)
        drifted = run_config.resume_drift(state.get("args") or {}, args)
        if drifted:
            detail = "\n".join(
                f"    {key}: checkpoint {was!r}, config now {now!r}"
                for key, (was, now) in drifted.items()
            )
            print(
                f"refusing to resume {last_path}: the config no longer "
                f"describes the run in it.\n{detail}\n"
                "Resuming would splice two experiments into one curve."
            )
            return 1
        rule.load_state_dict(state["rule"])
        optimiser.load_state_dict(state["optimiser"])
        start_update = int(state["update"])
        best = float(state.get("best", float("inf")))
        resumed_skips = state.get("skips") or {}
        ckpt.set_rng_state(state["rng"])
        ckpt.restore_python_rng(rng, state["python_rng"])
        if state.get("eval_indices"):
            eval_indices = list(state["eval_indices"])
            train_indices = [i for i in indices if i not in set(eval_indices)]
        print(f"resumed from {last_path} at update {start_update}")

    try:
        eval_pdb_ids = [catalogue[i].pdb_id for i in eval_indices]
    except (IndexError, KeyError, AttributeError) as problem:
        print(f"cannot resolve eval_indices against this catalogue: {problem!r}")
        return 1

    print(
        f"device {device}, split {args.split}: {len(train_indices)} training, "
        f"{len(eval_indices)} held out (disjoint)"
    )
    print(
        f"{rule.parameter_count()} parameters (hidden {rule.hidden_dim}, message "
        f"{rule.message_dim}), crop {args.crop_size}, k {args.k}, "
        + (
            f"{args.steps_min}-{args.steps} steps"
            if args.steps_min != args.steps
            else f"{args.steps} steps"
        )
        + ", full BPTT"
    )
    print(
        f"  the label costs {conditioning_cost(args.message_dim, args.label_dim)}"
        f" of those: {args.label_dim} * {args.message_dim} for phi_m's widened "
        f"input plus {len(FOLD_CLASSES) + 1} * {args.label_dim} for the "
        f"embedding, against the port's 5328.\n  The smallness is "
        "load-bearing: if this arm needed to GROW the rule to work, the \"one "
        "small\n  local rule\" claim is what would be traded away, and that "
        "trade gets argued before it is made."
    )
    print(
        "  the authors' own defaults for this task: coord_act "
        + ("tanh" if args.has_coord_act else "NONE (ours, and it diverged)")
        + f", h update {'residual' if args.is_residual else 'REPLACING (ours)'}"
        + f", hidden act {args.activation}, node norm {args.norm_type}"
        + ", phi_x output zero-initialised and unbiased."
        + (
            ""
            if (
                args.has_coord_act
                and args.is_residual
                and args.activation == "tanh"
                and args.norm_type == "pn"
            )
            else "\n  AT LEAST ONE IS OFF, so this is an ablation of the "
            "published rule and not the port's rule."
        )
    )
    print(
        f"weights: pairwise {args.pairwise_weight:g} (corrupted-to-context), "
        f"pairwise-window {args.pairwise_window_weight:g} "
        "(corrupted-to-corrupted) -- unclamped, A^2"
        + (
            "  FULL PAIR SET"
            if args.pairwise_weight and args.pairwise_window_weight
            else "  ONE HALF ONLY, which is not the published loss"
        )
    )
    print(
        f"sampling: {args.sample_count // len(FOLD_CLASSES)} draw(s) per class "
        f"per evaluation ({len(FOLD_CLASSES)} classes, "
        f"{args.sample_count // len(FOLD_CLASSES) * len(FOLD_CLASSES)} total) "
        f"from a Gaussian prior at rg = {RG_PREFACTOR:g} * N^{RG_EXPONENT:g} A, "
        "SAME noise seeds and crops across the classes so readout 0 is PAIRED; "
        f"run to a max per-step move below {args.sample_tolerance:g} A or "
        f"{args.sample_max_steps} steps; written to one {samples_path.name} "
        "with the class each sample was CONDITIONED on, so readout 5's "
        "confusion table is computable from the file alone"
        if args.sample_count
        else "sampling: OFF (sample_count 0), so this run measures denoising "
        "and not generation -- and readout 0, the instrument check, is not "
        "measured at all"
    )

    labels = LabelCache()
    counted_at = time.perf_counter()
    counts, unreadable = class_counts(dataset, train_indices, args.seed)
    print(
        class_counts_report(counts, unreadable, len(train_indices))
        + f"  [{time.perf_counter() - counted_at:.1f}s]"
    )

    started = time.perf_counter()
    skips = {reason: int(resumed_skips.get(reason, 0)) for reason in SKIP_REASONS}
    interval_skips = {reason: 0 for reason in SKIP_REASONS}
    interval_updates = 0
    #: How many of the interval's updates read the unconditional row instead of
    #: the crop's class. Reported because label dropout is the one setting that
    #: could make a flat readout 0 look like an inert embedding.
    interval_dropped = 0
    history_path = out_dir / "history.json"
    history: list[dict] = []
    if args.resume and history_path.exists():
        try:
            history = [
                row
                for row in json.loads(history_path.read_text(encoding="utf-8"))
                if row.get("update", 0) <= start_update
            ]
        except (OSError, ValueError):
            history = []

    pool: dict[tuple[int, int | None], EGNCAPoolEntry] = {}

    out_of_time = False
    stopped = False
    update = start_update
    for update in range(start_update + 1, args.updates + 1):
        try:
            resume: EGNCAPoolEntry | None = None
            key: tuple[int, int | None]
            if (
                args.pool_probability > 0
                and pool
                and rng.random() < args.pool_probability
            ):
                key = rng.choice(sorted(pool, key=lambda k: (k[0], k[1] or -1)))
                index, crop_seed = key
                try:
                    topo, native, _crop, _used = dataset.load_crop(
                        index, seed_residue=crop_seed, rng=rng
                    )
                except Exception:
                    pool.pop(key, None)
                else:
                    resume = pool[key]
                    if resume.ca.shape[0] != topo.n_residues:
                        pool.pop(key, None)
                        resume = None
            if resume is None:
                index = rng.choice(train_indices)
                try:
                    topo, native, _crop, crop_seed = dataset.load_crop(
                        index, rng=rng
                    )
                except Exception:
                    raise _SkipUpdate("no_crop")
                key = (index, crop_seed)
            if topo.n_residues < 3:
                raise _SkipUpdate("no_window")

            # **The label is a property of the crop, cached under the same
            # ``(index, crop_seed)`` key the pool uses.** A pool resume keeps
            # the class the state was built under, which is the pool entry's own
            # reason: a freshly drawn condition over resumed coordinates would
            # score a structure against a question it was never asked.
            label = labels.label_for(key, native, topo)

            generator = torch.Generator().manual_seed(
                args.seed * 1_000_003 + update
            )
            steps = (
                rng.randint(args.steps_min, args.steps)
                if args.steps_min != args.steps
                else args.steps
            )
            # Drawn from ``rng`` rather than from ``generator``, so the
            # corruption stream inside ``denoise_objective`` is bit-for-bit the
            # MVP's and the only difference between the two arms is the label.
            draw = rng.random()
            terms, state, row = conditioned_objective(
                rule,
                native,
                topo,
                steps=steps,
                label=label,
                dropout_draw=draw,
                generator=generator,
                resume_from=resume,
                **objective_kwargs(args),
            )
            if row == UNCONDITIONAL:
                interval_dropped += 1
            if terms.corrupted_mask is None or not bool(terms.corrupted_mask.any()):
                raise _SkipUpdate("no_window")
            if not torch.isfinite(terms.total):
                raise _SkipUpdate("nonfinite_loss")
            if not terms.total.requires_grad:
                raise _SkipUpdate("no_window")
            optimiser.zero_grad()
            terms.total.backward()
            # A finite loss does not imply a finite gradient, and
            # ``clip_grad_norm_`` turns one infinity into NaN parameters: the
            # total norm is inf, the scale is clip/inf = 0, and 0 * inf = NaN.
            if not all(
                torch.isfinite(p.grad).all()
                for p in rule.parameters()
                if p.grad is not None
            ):
                optimiser.zero_grad(set_to_none=True)
                seen = skips["nonfinite_gradient"] + 1
                if seen in (1, 10, 100) or seen % 500 == 0:
                    print(
                        f"  update {update}: non-finite gradient with a finite "
                        f"loss ({float(terms.total.detach()):.3f}); skipped. "
                        f"{seen} so far.",
                        flush=True,
                    )
                raise _SkipUpdate("nonfinite_gradient")
            torch.nn.utils.clip_grad_norm_(rule.parameters(), args.clip)
            optimiser.step()

            # The pool's bar is the sample's own corrupted start, so there is no
            # threshold to pick and it self-calibrates per protein and per noise
            # level. A state worse than not trying is not worth resuming.
            if args.pool_probability > 0:
                bar = (
                    resume.baseline_rmsd
                    if resume is not None
                    else terms.start_rmsd
                )
                reached = terms.rmsd
                if (
                    bar is not None
                    and reached is not None
                    and reached <= bar
                    and terms.corrupted_mask is not None
                ):
                    pool[key] = EGNCAPoolEntry(
                        ca=state.ca.detach().clone(),
                        hidden=state.hidden.detach().clone(),
                        damaged=terms.corrupted_mask.clone(),
                        baseline_rmsd=bar,
                    )
        except _SkipUpdate as skipped:
            skips[skipped.reason] += 1
            interval_skips[skipped.reason] += 1
        interval_updates += 1

        if update % args.eval_every == 0 or update == args.updates:
            scores = score(
                rule, dataset, eval_indices, args, labels, seed=args.seed
            )
            samples = (
                sample_batch(rule, dataset, eval_indices, args, seed=args.seed)
                if args.sample_count
                else []
            )
            if samples:
                write_samples(samples_path, samples, update)
                scores |= helix_contrast(samples)
            elapsed = time.perf_counter() - started
            print(
                f"update {update:7d}  {report(scores, samples)}  "
                f"{skip_report(interval_skips, interval_updates)}  "
                f"dropped {interval_dropped}/{interval_updates}  "
                f"{elapsed / max(update - start_update, 1):.2f}s/update",
                flush=True,
            )
            table = confusion(samples)
            if table:
                print(
                    "  readout 5, conditioned -> produced: "
                    + ", ".join(f"{pair} {count}" for pair, count in table.items()),
                    flush=True,
                )
            history.append({
                "update": update,
                **scores,
                "skips": dict(interval_skips),
                "skip_window": interval_updates,
                "label_dropped": interval_dropped,
                "sample_steps": [c.sample.steps for c in samples],
                "sample_converged": [bool(c.sample.converged) for c in samples],
                "sample_label": [class_name(c.label) for c in samples],
                "sample_produced": [c.produced for c in samples],
                "sample_helix_fraction": [c.helix_fraction for c in samples],
                "confusion": table,
            })
            interval_skips = {reason: 0 for reason in SKIP_REASONS}
            interval_updates = 0
            interval_dropped = 0
            # ``json_safe``, not a bare dump: see its docstring. A bare ``NaN``
            # is not valid JSON and broke a reader of the MVP's history.
            history_path.write_text(
                json.dumps(json_safe(history), indent=1), encoding="utf-8"
            )
            payload = {
                "rule": rule.state_dict(),
                "optimiser": optimiser.state_dict(),
                "update": update,
                "best": best,
                "args": recorded_args(args),
                "runtime": {
                    key: getattr(args, key)
                    for key in run_config.RUNTIME_ONLY
                    if hasattr(args, key)
                },
                "eval_indices": eval_indices,
                "eval_pdb_ids": eval_pdb_ids,
                "scores": scores,
                "class_counts": dict(counts),
                "confusion": table,
                "rng": ckpt.rng_state(),
                "python_rng": rng.getstate(),
                "skipped_updates": sum(skips.values()),
                "skips": dict(skips),
            }
            ckpt.save(last_path, payload)
            # Selection is on the held-out ``total``, the only scalar here that
            # prices the two halves of the restraint against each other.
            if scores["total"] == scores["total"] and scores["total"] < best:
                best = scores["total"]
                payload["best"] = best
                ckpt.save(best_path, payload)

        if stop_file.exists():
            print(f"stopping at update {update}: {stop_file} exists")
            stopped = True
            break
        if args.max_seconds and time.perf_counter() - started > float(
            args.max_seconds
        ):
            print(f"stopping at update {update}: out of time")
            out_of_time = True
            break

    if sum(skips.values()):
        print(
            f"{sum(skips.values())} update(s) contributed no gradient -- this "
            "run trained on fewer updates than its curve's x-axis says. By "
            "reason: "
            + ", ".join(f"{name} {skips[name]}" for name in SKIP_REASONS)
            + ". `nonfinite_loss` is the rule exploding and being excused for "
            "it; `no_window` is a property of the crop."
        )
    # **64, not 0.** ``scripts/bunya_train.slurm`` resubmits only on 64, and
    # ``train_egnca.py`` and ``train_growth.py`` return 0 here -- so a segment
    # that hit its walltime ended its chain without saying so. A stop file is a
    # deliberate stop and stays 0.
    if out_of_time and not stopped and update < args.updates:
        print(
            f"exiting {CHECKPOINTED_NOT_FINISHED} (checkpointed, not finished) "
            f"at update {update} of {args.updates}, so the chain resubmits."
        )
        return CHECKPOINTED_NOT_FINISHED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
