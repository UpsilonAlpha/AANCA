"""Fetch the published side-chain packing benchmark.

    python scripts/fetch_benchmark.py

Downloads ``scp_results.zip`` (185 MB) from AttnPacker's Zenodo release, record
7713779, and extracts it to ``data/external/attnpacker/``. That archive holds
the native CASP test structures **and** every baseline's predicted structures --
SCWRL4, FASPR, RosettaPacker, DLPacker, AttnPacker, plus AF2 and OmegaFold
backbones.

Having the baselines' own output is what makes
``scripts/validate_benchmark.py`` possible: our metrics can be checked against
their published table rather than merely against themselves.

The same record also holds ``AttnPackerPTM_V2.zip`` at 6.9 GB of pre-trained
weights. That is deliberately **not** downloaded -- this project trains from
scratch, so those weights are 6.9 GB of no use.

Archive handling: downloaded archives are untrusted input, so this extracts into
its own directory, refuses any entry with an absolute path or a ``..``
component, and executes nothing from it.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests

URL = "https://zenodo.org/api/records/7713779/files/scp_results.zip/content"
EXPECTED_BYTES = 185_280_925
DEST = Path(__file__).resolve().parent.parent / "data" / "external" / "attnpacker"


def download(archive: Path) -> None:
    if archive.exists() and archive.stat().st_size == EXPECTED_BYTES:
        print(f"{archive.name}: already present ({archive.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {EXPECTED_BYTES / 1e6:.0f} MB from Zenodo...")
    with requests.get(URL, stream=True, timeout=900) as response:
        response.raise_for_status()
        written = 0
        with open(archive, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
    print(f"{archive.name}: {written / 1e6:.1f} MB")


def extract(archive: Path, out: Path) -> int:
    with zipfile.ZipFile(archive) as zf:
        members = [
            name
            for name in zf.namelist()
            if not name.startswith("__MACOSX") and ".DS_Store" not in name
        ]
        unsafe = [
            name
            for name in members
            if name.startswith("/") or ".." in Path(name).parts
        ]
        if unsafe:
            raise ValueError(
                f"refusing to extract: {len(unsafe)} entries escape the target "
                f"directory, e.g. {unsafe[0]!r}"
            )
        zf.extractall(out, members=members)
    return len(members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-extract")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "scp_results.zip"
    download(archive)

    out = DEST / "extracted"
    if out.exists() and not args.force:
        print(f"{out} already extracted; pass --force to redo")
    else:
        count = extract(archive, out)
        print(f"extracted {count} entries to {out}")

    base = out / "standard_results"
    if base.is_dir():
        print("\nmethod directories:")
        for directory in sorted(p for p in base.iterdir() if p.is_dir()):
            counts = {
                sub.name: len(list(sub.glob("*.pdb")))
                for sub in sorted(directory.iterdir())
                if sub.is_dir()
            }
            print(f"  {directory.name:<24}{counts}")
        print("\n'casp_all' is the native-backbone set; 'non_native' uses")
        print("predicted backbones. Next: scripts/validate_benchmark.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
