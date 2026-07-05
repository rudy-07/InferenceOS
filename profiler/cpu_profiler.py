"""
cpu_profiler.py
---------------
Detects CPU topology, clock speed, cache hierarchy, and ISA extension support.

Uses `cpuinfo` (py-cpuinfo) as the primary source, with `psutil` and
platform/ctypes fallbacks so the module is safe to import on any OS.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def _run_silent(*args: str) -> str:
    """Run a subprocess command, returning stdout or empty string on failure."""
    try:
        result = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_cpuinfo_data() -> dict[str, Any]:
    """
    Attempt to import py-cpuinfo and return its full info dict.
    Returns an empty dict if the package is not installed.
    """
    try:
        import cpuinfo  # type: ignore

        return cpuinfo.get_cpu_info()
    except ImportError:
        return {}


def _detect_isa_extensions(raw: dict[str, Any]) -> list[str]:
    """
    Extract ISA extensions relevant to llama.cpp from the cpuinfo flags list.
    Falls back to parsing /proc/cpuinfo on Linux if cpuinfo is unavailable.
    """
    # Extensions llama.cpp cares about (ordered roughly by importance)
    RELEVANT = {
        "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        "avx512_vnni", "avx2", "avx", "fma", "f16c",
        "sse4_1", "sse4_2", "ssse3", "sse3", "sse2",
        "neon",  # ARM
        "sve",   # ARM scalable
        "amx_bf16", "amx_int8",  # Intel AMX
    }

    flags: list[str] = []

    if "flags" in raw:
        flags = [f.lower() for f in raw["flags"]]
    elif platform.system() == "Linux":
        proc = _run_silent("grep", "-m1", "flags", "/proc/cpuinfo")
        if proc:
            flags = proc.split(":")[1].strip().split() if ":" in proc else []
    elif platform.system() == "Darwin":
        # macOS: use sysctl
        out = _run_silent("sysctl", "-a", "hw.optional")
        flags = [
            line.split(":")[0].split(".")[-1].lower()
            for line in out.splitlines()
            if line.endswith(": 1")
        ]

    return sorted(f for f in flags if f in RELEVANT)


def _get_cache_info() -> dict[str, Any]:
    """
    Returns L2 (per-core, KB) and L3 (total, MB) cache sizes.
    Uses py-cpuinfo if available, then OS-specific fallbacks.
    """
    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        l2_kb = info.get("l2_cache_size", 0)
        l3_mb = info.get("l3_cache_size", 0)

        # cpuinfo returns values as ints (bytes) or strings like "512 KB"
        def _to_kb(val: Any) -> float:
            if isinstance(val, int):
                return round(val / 1024, 1)
            if isinstance(val, str):
                val = val.upper().replace(" ", "")
                if "MB" in val:
                    return float(val.replace("MB", "")) * 1024
                if "KB" in val:
                    return float(val.replace("KB", ""))
                try:
                    return round(int(val) / 1024, 1)
                except ValueError:
                    return 0.0
            return 0.0

        return {
            "l2_cache_kb": _to_kb(l2_kb),
            "l3_cache_mb": round(_to_kb(l3_mb) / 1024, 1),
        }
    except ImportError:
        pass

    # Linux /sys fallback
    if platform.system() == "Linux":
        try:
            import os

            cache_base = "/sys/devices/system/cpu/cpu0/cache"
            caches: dict[str, float] = {}
            for entry in os.listdir(cache_base):
                level_file = f"{cache_base}/{entry}/level"
                size_file = f"{cache_base}/{entry}/size"
                type_file = f"{cache_base}/{entry}/type"
                if not os.path.exists(level_file):
                    continue
                with open(level_file) as f:
                    level = f.read().strip()
                with open(size_file) as f:
                    size_str = f.read().strip().upper()
                with open(type_file) as f:
                    cache_type = f.read().strip()
                size_kb = (
                    float(size_str.replace("K", "")) if "K" in size_str else 0.0
                )
                if level == "2" and cache_type in ("Unified", "Data"):
                    caches["l2_cache_kb"] = size_kb
                elif level == "3":
                    caches["l3_cache_mb"] = round(size_kb / 1024, 1)
            return caches
        except Exception:
            pass

    return {"l2_cache_kb": 0.0, "l3_cache_mb": 0.0}


def profile() -> dict[str, Any]:
    """
    Returns a structured dict describing the host CPU.
    Never raises — missing values default to 0 or empty list.
    """
    import psutil

    raw = _get_cpuinfo_data()
    cache = _get_cache_info()

    brand = raw.get("brand_raw", "")
    if not brand:
        brand = platform.processor() or "Unknown CPU"

    hz_info = raw.get("hz_advertised_friendly", "")
    base_freq_mhz: float = 0.0
    if hz_info:
        try:
            # e.g. "4.5000 GHz"
            val, unit = hz_info.split()
            multiplier = 1000.0 if unit.upper().startswith("G") else 1.0
            base_freq_mhz = round(float(val) * multiplier, 1)
        except (ValueError, AttributeError):
            pass

    # psutil is the most reliable cross-platform source for core counts
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1

    return {
        "brand": brand,
        "architecture": raw.get("arch", platform.machine()),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "base_freq_mhz": base_freq_mhz,
        "cache_l2_kb": cache.get("l2_cache_kb", 0.0),
        "cache_l3_mb": cache.get("l3_cache_mb", 0.0),
        "isa_extensions": _detect_isa_extensions(raw),
    }
