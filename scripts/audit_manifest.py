"""Audit the training manifest. Exits non-zero if any guarantee is violated.

    python scripts/audit_manifest.py

The test-leakage guarantee is the one claim in the data pipeline that cannot be
re-derived cheaply -- it costs 210 MMseqs2 queries against RCSB -- so it is
checked here against the lists the build persisted rather than taken on trust.

A note on getting this check right, because the obvious version is wrong. An
excluded entry may touch *two* clusters: one banned (it holds a chain homologous
to a test target) and one innocent. A surviving entry that shares only the
innocent cluster is **not** a leak -- its chain is 40%-similar to a chain of the
excluded entry, which says nothing about its similarity to any test target. The
check that matters is whether a survivor touches a **banned** cluster, not
whether it shares any cluster with something excluded. The first version of this
audit conflated the two and reported 60 false positives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aanca.data import ClusterIndex, read_manifest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"
META = ROOT / "data" / "manifest_meta.json"
CLUSTERS = ROOT / "data" / "external" / "rcsb" / "clusters-by-entity-40.txt"


def main() -> int:
    for path in (MANIFEST, META, CLUSTERS):
        if not path.exists():
            print(f"{path} missing -- run scripts/build_training_manifest.py")
            return 1

    splits = read_manifest(MANIFEST)
    meta = json.loads(META.read_text(encoding="utf-8"))
    clusters = ClusterIndex.from_file(CLUSTERS)

    print("manifest")
    for key in ("resolution_max", "candidates", "representatives", "seed"):
        print(f"  {key:<22}{meta.get(key)}")
    for name, specs in splits.items():
        print(f"  {name:<22}{len(specs)}")

    failures: list[str] = []

    if not meta.get("leakage_filter_applied"):
        failures.append(
            "leakage filter was NOT applied -- this manifest cannot back any "
            "reported number"
        )

    ids = {name: {s.pdb_id for s in specs} for name, specs in splits.items()}
    survivors = ids.get("train", set()) | ids.get("validation", set())

    for a, b in (("train", "validation"), ("train", "excluded"), ("validation", "excluded")):
        overlap = ids.get(a, set()) & ids.get(b, set())
        if overlap:
            failures.append(f"{a} and {b} share {len(overlap)} entries, e.g. {sorted(overlap)[:3]}")

    train_groups = {s.group for s in splits.get("train", [])}
    validation_groups = {s.group for s in splits.get("validation", [])}
    shared = train_groups & validation_groups
    if shared:
        failures.append(f"{len(shared)} cluster groups span train and validation")

    leaking = set(meta.get("leaking_entries", []))
    banned = set(meta.get("banned_clusters", []))
    if meta.get("leakage_filter_applied") and not banned:
        failures.append("leakage filter applied but no banned clusters recorded")

    direct = survivors & leaking
    if direct:
        failures.append(
            f"{len(direct)} surviving entries are themselves within 40% of a "
            f"test target, e.g. {sorted(direct)[:3]}"
        )

    # The check that actually matters: no survivor may touch a banned cluster.
    touching = [p for p in sorted(survivors) if clusters.clusters_of(p) & banned]
    if touching:
        failures.append(
            f"{len(touching)} surviving entries touch a banned cluster, "
            f"e.g. {touching[:3]}"
        )

    print(f"\nleakage: {len(leaking)} entries matched, {len(banned)} clusters banned")
    print(f"         {len(splits.get('excluded', []))} representatives removed")
    print(f"         {len(survivors)} survivors, {len(touching)} touching a banned cluster")

    if failures:
        print("\nFAILED")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nAll guarantees hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
