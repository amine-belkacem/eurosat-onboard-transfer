"""TIFF tag reader of scripts/probe_geotiff.py, on a TIFF built byte by byte."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_geotiff.py"
spec = importlib.util.spec_from_file_location("probe_geotiff", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def check(got, expected, what=""):
    assert got == expected, f"ECHEC {what}: obtenu {got}, attendu {expected}"


def tiny_tiff() -> bytes:
    """Little-endian TIFF with two tags.

    bytes 0-7    header "II*\\0" + offset of the first IFD (8)
    bytes 8-9    number of entries (2)
    bytes 10-21  ImageWidth, SHORT, count 1, value 64 stored inline
    bytes 22-33  ModelPixelScale, DOUBLE, count 3, data at offset 38 (24 bytes do not fit inline)
    bytes 34-37  offset of the next IFD (0 = none)
    bytes 38-61  the three doubles 10.0, 10.0, 0.0
    """
    header = b"II*\x00" + struct.pack("<I", 8)
    ifd = struct.pack("<H", 2)
    ifd += struct.pack("<HHIH2x", 256, 3, 1, 64)
    ifd += struct.pack("<HHII", 33550, 12, 3, 38)
    ifd += struct.pack("<I", 0)
    data = struct.pack("<3d", 10.0, 10.0, 0.0)
    raw = header + ifd + data
    assert len(header + ifd) == 38
    return raw


def test_read_inline_and_offset_values():
    tags = probe.read_tiff_tags(tiny_tiff())
    check(tags["ImageWidth"], 64, "valeur stockee dans l'entree")
    check(tags["ModelPixelScale"], (10.0, 10.0, 0.0), "valeurs stockees a l'offset 38")


def test_rejects_big_endian():
    with pytest.raises(ValueError):
        probe.read_tiff_tags(b"MM\x00*" + bytes(60))
