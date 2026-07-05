"""
apple_backend.py
----------------
Detects Apple Silicon (M-series) unified memory GPU via two strategies:

  Tier 1: system_profiler SPDisplaysDataType -json  (macOS built-in)
  Tier 2: Metal device enumeration via PyObjC / ctypes

On non-macOS systems this module immediately returns an empty list.
"""
from __future__ import annotations

import json
import platform
import subprocess
from typing import Any


def _parse_memory_str(s: str) -> int:
    """
    Convert strings like '16 GB', '8192 MB', '16384' → MB.
    Returns 0 on parse failure.
    """
    s = s.strip().upper()
    try:
        if "GB" in s:
            return round(float(s.replace("GB", "").strip()) * 1024)
        if "MB" in s:
            return round(float(s.replace("MB", "").strip()))
        # bare number (assume bytes if very large, else MB)
        val = float(s)
        return round(val / (1024 ** 2)) if val > 10_000 else round(val)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Tier 1 — system_profiler
# ---------------------------------------------------------------------------

def _probe_system_profiler() -> list[dict[str, Any]] | None:
    """
    Parses macOS system_profiler output for GPU/display adapter info.
    Works on both Intel Macs (discrete GPU) and Apple Silicon (integrated).
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        if not displays:
            return None

        gpus: list[dict[str, Any]] = []
        for i, adapter in enumerate(displays):
            name = adapter.get("sppci_model", adapter.get("_name", "Apple GPU"))

            # VRAM / unified memory key varies by chip generation
            vram_str = (
                adapter.get("spdisplays_vram", "")
                or adapter.get("spdisplays_vram_shared", "")
                or adapter.get("sppci_memory", "")
                or "0 MB"
            )
            vram_total_mb = _parse_memory_str(vram_str)

            vendor_raw = adapter.get("sppci_vendor", "").lower()
            vendor = "apple" if "apple" in vendor_raw else "intel" if "intel" in vendor_raw else "amd"
            backend = "metal"  # all current macOS GPUs support Metal

            gpus.append({
                "index": i,
                "vendor": vendor,
                "name": name,
                # Apple Silicon shares RAM with the GPU; free cannot be
                # accurately measured without Metal API calls.
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_total_mb,  # conservative: assume all free
                "vram_used_mb": 0,
                "driver_version": platform.mac_ver()[0],
                "compute_capability": "unknown",
                "backend_hint": backend,
            })

        return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


# ---------------------------------------------------------------------------
# Tier 2 — PyObjC Metal enumeration
# ---------------------------------------------------------------------------

def _probe_metal_pyobjc() -> list[dict[str, Any]] | None:
    """
    Use PyObjC + Metal framework to enumerate GPU devices.
    Returns None if PyObjC is not installed.
    """
    try:
        import Metal  # type: ignore  # part of pyobjc-framework-Metal

        devices = Metal.MTLCopyAllDevices()
        if not devices:
            return None

        gpus: list[dict[str, Any]] = []
        for i, device in enumerate(devices):
            name = str(device.name())
            # recommendedMaxWorkingSetSize is in bytes
            vram_bytes = device.recommendedMaxWorkingSetSize()
            vram_total_mb = round(vram_bytes / (1024 ** 2))

            gpus.append({
                "index": i,
                "vendor": "apple",
                "name": name,
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_total_mb,
                "vram_used_mb": 0,
                "driver_version": platform.mac_ver()[0],
                "compute_capability": "unknown",
                "backend_hint": "metal",
            })

        return gpus if gpus else None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect() -> list[dict[str, Any]]:
    """
    Returns a list of Apple/Metal GPU descriptors.
    Only runs on macOS; returns [] immediately on other platforms.
    """
    if platform.system() != "Darwin":
        return []

    for probe in (_probe_system_profiler, _probe_metal_pyobjc):
        result = probe()
        if result is not None:
            return result
    return []
