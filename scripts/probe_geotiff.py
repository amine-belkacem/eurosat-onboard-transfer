"""Check that the EuroSAT multispectral GeoTIFFs carry their georeferencing.

The 13-band archive is 2.1 GB. To decide whether a geographic split is
possible (milestone 5) I only need a few files, so this script reads the zip's
central directory and the requested members over HTTP range requests, without
downloading the whole archive. It then prints the GeoTIFF tags of each file.

    python scripts/probe_geotiff.py
    python scripts/probe_geotiff.py --members AnnualCrop_1 Forest_1 River_7
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import urllib.request
import zipfile

MS_URL = "https://zenodo.org/api/records/7711810/files/EuroSAT_MS.zip/content"

# TIFF tags of interest: image size, bands, bits per sample, and the GeoTIFF tags.
TAG_NAMES = {
    256: "ImageWidth",
    257: "ImageLength",
    258: "BitsPerSample",
    277: "SamplesPerPixel",
    33550: "ModelPixelScale",
    33922: "ModelTiepoint",
    34737: "GeoAsciiParams",
}
# TIFF field types -> (struct code, size in bytes)
FIELD_TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 12: ("d", 8)}


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP range requests."""

    def __init__(self, url: str):
        super().__init__()
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:
            self.url = resp.geturl()  # follow the redirect once
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request) as resp:
            chunk = resp.read()
        buffer[:len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def read_tiff_tags(raw: bytes) -> dict[str, object]:
    """Tags of the first IFD of a little-endian TIFF, limited to TAG_NAMES."""
    if raw[:4] != b"II*\x00":
        raise ValueError("not a little-endian TIFF")
    ifd = struct.unpack_from("<I", raw, 4)[0]
    n_entries = struct.unpack_from("<H", raw, ifd)[0]
    tags = {}
    for i in range(n_entries):
        entry = ifd + 2 + 12 * i
        tag, ftype, count, value = struct.unpack_from("<HHII", raw, entry)
        if tag not in TAG_NAMES or ftype not in FIELD_TYPES:
            continue
        code, size = FIELD_TYPES[ftype]
        # Values that fit in 4 bytes are stored in the entry itself.
        offset = entry + 8 if count * size <= 4 else value
        if ftype == 2:
            tags[TAG_NAMES[tag]] = raw[offset:offset + count].decode("ascii", "replace").rstrip("\x00")
        else:
            values = struct.unpack_from(f"<{count}{code}", raw, offset)
            tags[TAG_NAMES[tag]] = values[0] if count == 1 else values
    return tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--members", nargs="+", default=["AnnualCrop_1", "AnnualCrop_2", "Forest_1"],
                        help="image names without extension")
    args = parser.parse_args()

    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(MS_URL), buffer_size=1 << 16))
    by_stem = {name.rsplit("/", 1)[-1][:-4]: name for name in archive.namelist() if name.endswith(".tif")}
    print(f"{len(by_stem)} GeoTIFFs in the archive")

    missing = [m for m in args.members if m not in by_stem]
    if missing:
        print(f"ECHEC: not in the archive: {missing}")
        return 1
    for stem in args.members:
        tags = read_tiff_tags(archive.read(by_stem[stem]))
        print(f"\n{by_stem[stem]}")
        for key, value in tags.items():
            if key == "BitsPerSample":
                value = f"{value[0]} bits x {len(value)} bands"
            print(f"  {key:16s} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
