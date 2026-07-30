"""
apple_backend.py
----------------
Detects Apple Silicon (M-series) unified memory GPU via two strategies:
  Tier 1: system_profiler SPDisplaysDataType -json  (macOS built-in)
  Tier 2: Metal device enumeration via PyObjC / ctypes
"""
from __future__ import annotations

import json
import platform
import subprocess
from typing import Any, Dict, List, Optional


def _parse_memory_str(s: str) -> int:
    s = s.strip().upper()
    try:
        if "GB" in s:
            return round(float(s.replace("GB", "").strip()) * 1024)
        if "MB" in s:
            return round(float(s.replace("MB", "").strip()))
        val = float(s)
        return round(val / (1024 ** 2)) if val > 10_000 else round(val)
    except ValueError:
        return 0


def _probe_system_profiler() -> List[Dict[str, Any]] | None:
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

        gpus: List[Dict[str, Any]] = []
        for i, adapter in enumerate(displays):
            name = adapter.get("sppci_model", adapter.get("_name", "Apple GPU"))
            vram_str = (
                adapter.get("spdisplays_vram", "")
                or adapter.get("spdisplays_vram_shared", "")
                or adapter.get("sppci_memory", "")
                or "0 MB"
            )
            vram_total_mb = _parse_memory_str(vram_str)
            vendor_raw = adapter.get("sppci_vendor", "").lower()
            vendor = "apple" if "apple" in vendor_raw else "intel" if "intel" in vendor_raw else "amd"
            is_igpu = (vendor == "apple" or "integrated" in adapter.get("sppci_bus", "").lower())

            vram_bytes = vram_total_mb * 1024 * 1024

            gpus.append({
                "index": i,
                "vendor": vendor,
                "name": name,
                "model": name,
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_total_mb,
                "vram_used_mb": 0,
                "vram_total_bytes": vram_bytes,
                "vram_available_bytes": vram_bytes,
                "driver_version": platform.mac_ver()[0],
                "compute_capability": "Metal 3" if vendor == "apple" else "Metal 2",
                "backend_hint": "metal",
                "is_integrated": is_igpu,
                "shared_memory_bytes": vram_bytes if is_igpu else 0,
                "pcie_bandwidth_gbps": 0.0 if is_igpu else 16.0,
                "utilization": 0.0,
            })

        return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def _probe_metal_pyobjc() -> List[Dict[str, Any]] | None:
    try:
        import Metal  # type: ignore

        devices = Metal.MTLCopyAllDevices()
        if not devices:
            return None

        gpus: List[Dict[str, Any]] = []
        for i, device in enumerate(devices):
            name = str(device.name())
            vram_bytes = device.recommendedMaxWorkingSetSize()
            vram_total_mb = round(vram_bytes / (1024 ** 2))
            is_igpu = device.isLowPower() or "Apple" in name

            gpus.append({
                "index": i,
                "vendor": "apple",
                "name": name,
                "model": name,
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_total_mb,
                "vram_used_mb": 0,
                "vram_total_bytes": vram_bytes,
                "vram_available_bytes": vram_bytes,
                "driver_version": platform.mac_ver()[0],
                "compute_capability": "Metal 3",
                "backend_hint": "metal",
                "is_integrated": is_igpu,
                "shared_memory_bytes": vram_bytes if is_igpu else 0,
                "pcie_bandwidth_gbps": 0.0 if is_igpu else 16.0,
                "utilization": 0.0,
            })

        return gpus if gpus else None

    except Exception:
        return None


def detect() -> List[Dict[str, Any]]:
    if platform.system() != "Darwin":
        return []

    for probe in (_probe_system_profiler, _probe_metal_pyobjc):
        result = probe()
        if result is not None:
            return result
    return []
