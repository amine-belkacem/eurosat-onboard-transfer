"""Helpers for the inference benchmark on the Raspberry Pi 5 (also runs on a PC).

Kept free of PyTorch: the Pi only needs numpy, pandas, pillow and onnxruntime.
Everything that reads the hardware returns None when the file or command does
not exist, so the same code runs on Windows for a dry run.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
CPU_FREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
CPU_GOVERNOR = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
BOARD_MODEL = Path("/proc/device-tree/model")

# Bits of `vcgencmd get_throttled` (Raspberry Pi firmware documentation).
THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


def latency_stats(latencies_ms: list[float] | np.ndarray) -> dict:
    """Summary of per-inference latencies. p95 and p99 use linear interpolation (numpy default)."""
    x = np.asarray(latencies_ms, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two latencies")
    return {
        "n": int(x.size),
        "mean_ms": float(x.mean()),
        "std_ms": float(x.std(ddof=1)),
        "median_ms": float(np.median(x)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
    }


def decode_throttled(value: int) -> dict[str, bool]:
    return {name: bool(value >> bit & 1) for bit, name in THROTTLE_BITS.items()}


def parse_throttled(output: str) -> int:
    """'throttled=0x50005' -> 0x50005."""
    key, _, hexvalue = output.strip().partition("=")
    if key != "throttled" or not hexvalue:
        raise ValueError(f"unexpected vcgencmd output: {output!r}")
    return int(hexvalue, 16)


def read_throttled() -> int | None:
    if shutil.which("vcgencmd") is None:
        return None
    out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, check=False)
    return parse_throttled(out.stdout) if out.returncode == 0 else None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_temperature_c() -> float | None:
    milli = _read_int(THERMAL_ZONE)
    return None if milli is None else milli / 1000


def read_cpu_freq_mhz() -> float | None:
    khz = _read_int(CPU_FREQ)
    return None if khz is None else khz / 1000


def platform_info(threads: int) -> dict:
    import onnxruntime as ort

    def text(path: Path) -> str | None:
        try:
            return path.read_text().strip("\x00\n ")
        except OSError:
            return None

    return {
        "board": text(BOARD_MODEL) or platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "cpu_governor": text(CPU_GOVERNOR),
        "threads": threads,
    }


def run_is_valid(freq_samples_mhz: list[float | None], throttled_before: int | None,
                 throttled_after: int | None) -> tuple[bool, list[str]]:
    """A latency run counts only if the CPU clock never moved and no throttling happened.

    Returns (valid, reasons). Missing sensors (PC dry run) make the run
    "unchecked", reported as not valid with the reason.
    """
    reasons = []
    freqs = [f for f in freq_samples_mhz if f is not None]
    if not freqs:
        reasons.append("no CPU frequency readings")
    elif max(freqs) != min(freqs):
        reasons.append(f"CPU frequency changed during the run ({min(freqs):.0f}-{max(freqs):.0f} MHz)")
    if throttled_before is None or throttled_after is None:
        reasons.append("vcgencmd not available")
    else:
        now = decode_throttled(throttled_after)
        new_events = throttled_after & ~throttled_before
        active = [k for k, v in now.items() if v and k.endswith("_now")]
        if active:
            reasons.append(f"throttling active at the end: {active}")
        if new_events >> 16:
            reasons.append(f"new throttling events during the run: 0x{new_events:x}")
    return not reasons, reasons
