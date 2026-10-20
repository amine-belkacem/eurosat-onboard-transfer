"""Download EuroSAT from the authors' Zenodo record and check its MD5.

Source: Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for
Land Use and Land Cover Classification", IEEE JSTARS 2019.
Zenodo record 7711810, MIT license.

Usage:
    python scripts/download_eurosat.py            # RGB version (~95 MB)
    python scripts/download_eurosat.py --ms       # 13-band version (~2.1 GB)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO = "https://zenodo.org/api/records/7711810/files/{name}/content"
ARCHIVES = {
    "rgb": ("EuroSAT_RGB.zip", "f46e308c4d50d4bf32fedad2d3d62f3b"),
    "ms": ("EuroSAT_MS.zip", "091174add3c8e680a49244acf185b9f0"),
}
CHUNK = 1 << 20


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        for block in iter(lambda: resp.read(CHUNK), b""):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing members that would land outside `target`."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (target / member).resolve().is_relative_to(target):
                raise RuntimeError(f"unsafe path in archive: {member}")
        zf.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ms", action="store_true", help="download the 13-band version")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    name, expected_md5 = ARCHIVES["ms" if args.ms else "rgb"]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / name

    if not archive.exists():
        print(f"Downloading {name} ...")
        download(ZENODO.format(name=name), archive)

    print("Checking MD5 ...")
    got = md5sum(archive)
    if got != expected_md5:
        print(f"ECHEC: MD5 {got}, attendu {expected_md5}. Delete {archive} and retry.")
        return 1

    extract_dir = args.data_dir / archive.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    print(f"Extracting to {extract_dir} ...")
    safe_extract(archive, extract_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
