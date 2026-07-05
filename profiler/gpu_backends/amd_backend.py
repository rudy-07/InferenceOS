"""
amd_backend.py
--------------
Detects AMD GPUs using two strategies:

  Tier 1: amdsmi Python bindings  — richest data (ROCm 5.6+)
  Tier 2: rocm-smi subprocess     — JSON output mode

Both are fully guarded.  On non-Linux systems (or systems without ROCm),
this module silently returns an empty list.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any


# ---------------------------------------------------------------------------
# Tier 1 — amdsmi Python bindings
# ---------------------------------------------------------------------------

def _probe_amdsmi() -> list[dict[str, Any]] | None:
    """
    amdsmi is the modern replacement for rsmi_bindings (ROCm 5.6+).
    The package is typically installed at /opt/rocm/lib/python3.x/site-packages/
    """
    try:
        import amdsmi  # type: ignore

        amdsmi.amdsmi_init()
        processors = amdsmi.amdsmi_get_processor_handles()
        gpus: list[dict[str, Any]] = []

        for i, handle in enumerate(processors):
            try:
                info = amdsmi.amdsmi_get_gpu_asic_info(handle)
                name = info.get("market_name", "AMD GPU")
            except Exception:
                name = "AMD GPU"

            try:
                vram_info = amdsmi.amdsmi_get_gpu_memory_total(
                    handle, amdsmi.AmdSmiMemoryType.VRAM
                )
                vram_total_mb = round(vram_info / (1024 ** 2))
            except Exception:
                vram_total_mb = 0

            try:
                vram_used = amdsmi.amdsmi_get_gpu_memory_usage(
                    handle, amdsmi.AmdSmiMemoryType.VRAM
                )
                vram_used_mb = round(vram_used / (1024 ** 2))
            except Exception:
                vram_used_mb = 0

            try:
                driver = amdsmi.amdsmi_get_gpu_driver_version(handle)
            except Exception:
                driver = "unknown"

            gpus.append({
                "index": i,
                "vendor": "amd",
                "name": name,
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": max(0, vram_total_mb - vram_used_mb),
                "vram_used_mb": vram_used_mb,
                "driver_version": str(driver),
                "compute_capability": "unknown",
                "backend_hint": "rocm",
            })

        amdsmi.amdsmi_shut_down()
        return gpus if gpus else None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier 2 — rocm-smi subprocess (JSON mode)
# ---------------------------------------------------------------------------

def _probe_rocm_smi() -> list[dict[str, Any]] | None:
    """
    rocm-smi --showallinfo --json  (ROCm < 5.6 style)
    Falls back to rocm-smi --showmeminfo vram --json for newer builds.
    """
    # ── Try the comprehensive JSON dump first ──
    try:
        result = subprocess.run(
            ["rocm-smi", "--showallinfo", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            gpus: list[dict[str, Any]] = []

            for key, gpu_data in data.items():
                # rocm-smi keys are like "card0", "card1", …
                if not key.startswith("card"):
                    continue
                idx = int(key.replace("card", ""))
                name = gpu_data.get("Card Series", gpu_data.get("Card model", "AMD GPU"))
                driver = gpu_data.get("Driver version", "unknown")

                # VRAM is reported as "X MB" strings
                def _parse_mb(s: str) -> int:
                    try:
                        return int(float(str(s).split()[0]))
                    except (ValueError, IndexError):
                        return 0

                vram_total_mb = _parse_mb(gpu_data.get("VRAM Total Memory (B)", "0"))
                if vram_total_mb == 0:
                    # older key format
                    vram_total_mb = _parse_mb(gpu_data.get("VRAM Total Used Memory (B)", "0"))

                # Some rocm-smi versions return bytes, not MB
                if vram_total_mb > 100_000:
                    vram_total_mb = round(vram_total_mb / (1024 ** 2))

                vram_used_mb = _parse_mb(gpu_data.get("VRAM Total Used Memory (B)", "0"))
                if vram_used_mb > 100_000:
                    vram_used_mb = round(vram_used_mb / (1024 ** 2))

                gpus.append({
                    "index": idx,
                    "vendor": "amd",
                    "name": name.strip(),
                    "vram_total_mb": vram_total_mb,
                    "vram_free_mb": max(0, vram_total_mb - vram_used_mb),
                    "vram_used_mb": vram_used_mb,
                    "driver_version": driver,
                    "compute_capability": "unknown",
                    "backend_hint": "rocm",
                })

            return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass

    # ── Minimal fallback: just check if rocm-smi exists and lists devices ──
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for i, line in enumerate(result.stdout.splitlines()):
                if "GPU" in line.upper() or "Radeon" in line:
                    gpus.append({
                        "index": i,
                        "vendor": "amd",
                        "name": line.strip(),
                        "vram_total_mb": 0,
                        "vram_free_mb": 0,
                        "vram_used_mb": 0,
                        "driver_version": "unknown",
                        "compute_capability": "unknown",
                        "backend_hint": "rocm",
                    })
            return gpus if gpus else None
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect() -> list[dict[str, Any]]:
    """
    Returns a list of AMD GPU descriptors (may be empty).
    Tries each tier in order, stopping at the first success.
    """
    for probe in (_probe_amdsmi, _probe_rocm_smi):
        result = probe()
        if result is not None:
            return result
    return []
