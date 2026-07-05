"""
intel_backend.py
----------------
Detects Intel discrete GPUs (Arc / Xe series) and integrated graphics.

Detection strategy:
  Tier 1: oneAPI Level Zero / Intel Extension for PyTorch  (optional libs)
  Tier 2: wmic path Win32_VideoController  (Windows)
  Tier 3: lshw -C display                 (Linux)
  Tier 4: system_profiler                 (macOS — handled by apple_backend)

VRAM figures for integrated GPUs are reported as 0 (dynamically shared).
"""
from __future__ import annotations

import platform
import subprocess
from typing import Any


# ---------------------------------------------------------------------------
# Tier 1 — Intel oneAPI (level_zero / intel_extension_for_pytorch)
# ---------------------------------------------------------------------------

def _probe_intel_oneapi() -> list[dict[str, Any]] | None:
    """
    level_zero is the low-level runtime for Intel GPUs on Linux and Windows.
    intel_extension_for_pytorch may expose an XPU device list.
    """
    # Try intel_extension_for_pytorch first
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
        import torch  # type: ignore

        if not torch.xpu.is_available():
            return None

        count = torch.xpu.device_count()
        gpus: list[dict[str, Any]] = []
        for i in range(count):
            props = torch.xpu.get_device_properties(i)
            total_mb = round(props.total_memory / (1024 ** 2))
            gpus.append({
                "index": i,
                "vendor": "intel",
                "name": props.name,
                "vram_total_mb": total_mb,
                "vram_free_mb": total_mb,  # XPU API lacks a free-memory query
                "vram_used_mb": 0,
                "driver_version": "unknown",
                "compute_capability": "unknown",
                "backend_hint": "vulkan",
            })
        return gpus if gpus else None
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Tier 2 — wmic (Windows)
# ---------------------------------------------------------------------------

def _probe_wmic() -> list[dict[str, Any]] | None:
    """
    Query Win32_VideoController via wmic or PowerShell Get-WmiObject.
    Returns Intel GPUs only; NVIDIA/AMD are handled by their own backends.
    """
    if platform.system() != "Windows":
        return None

    raw_lines: list[str] = []

    # ── Try wmic first ──
    try:
        result = subprocess.run(
            ["wmic", "path", "Win32_VideoController",
             "get", "Name,AdapterRAM,DriverVersion", "/format:csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            raw_lines = result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ── PowerShell fallback (wmic deprecated in Windows 11) ──
    if not raw_lines:
        try:
            ps_cmd = (
                "Get-WmiObject Win32_VideoController | "
                "Select-Object Name,AdapterRAM,DriverVersion | "
                "ConvertTo-Csv -NoTypeInformation"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                raw_lines = result.stdout.strip().splitlines()
        except Exception:
            return None

    # ── Parse CSV output ──
    gpus: list[dict[str, Any]] = []
    intel_idx = 0
    for line in raw_lines:
        parts = [p.strip().strip('"') for p in line.split(",")]
        # wmic CSV: Node,AdapterRAM,DriverVersion,Name   (column order varies)
        # PowerShell CSV header is always present as first line
        if not parts or parts[0].lower() in ("node", "name", ""):
            continue

        name = ""
        adapter_ram = 0
        driver = "unknown"

        # Heuristic: find the Intel name across all parts
        for p in parts:
            if "intel" in p.lower() or "arc" in p.lower() or "iris" in p.lower() or "uhd" in p.lower():
                name = p
            try:
                v = int(p)
                if v > 0 and adapter_ram == 0:
                    adapter_ram = v
            except ValueError:
                pass
            if "." in p and len(p.split(".")) >= 4:
                driver = p  # driver version strings like "31.0.101.4575"

        if not name:
            continue

        vram_mb = round(adapter_ram / (1024 ** 2)) if adapter_ram else 0

        gpus.append({
            "index": intel_idx,
            "vendor": "intel",
            "name": name,
            "vram_total_mb": vram_mb,
            "vram_free_mb": vram_mb,
            "vram_used_mb": 0,
            "driver_version": driver,
            "compute_capability": "unknown",
            "backend_hint": "vulkan",
        })
        intel_idx += 1

    return gpus if gpus else None


# ---------------------------------------------------------------------------
# Tier 3 — lshw (Linux)
# ---------------------------------------------------------------------------

def _probe_lshw() -> list[dict[str, Any]] | None:
    """
    Parse `lshw -C display` output for Intel graphics entries.
    Requires lshw to be installed (common on Debian/Ubuntu).
    """
    if platform.system() != "Linux":
        return None

    try:
        result = subprocess.run(
            ["lshw", "-C", "display", "-short"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None

        gpus: list[dict[str, Any]] = []
        idx = 0
        for line in result.stdout.splitlines():
            lower = line.lower()
            if "intel" in lower and ("display" in lower or "vga" in lower):
                # Extract description after the last column
                parts = line.split(maxsplit=3)
                name = parts[-1].strip() if len(parts) >= 4 else "Intel GPU"
                gpus.append({
                    "index": idx,
                    "vendor": "intel",
                    "name": name,
                    "vram_total_mb": 0,  # lshw rarely reports VRAM
                    "vram_free_mb": 0,
                    "vram_used_mb": 0,
                    "driver_version": "unknown",
                    "compute_capability": "unknown",
                    "backend_hint": "vulkan",
                })
                idx += 1

        return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect() -> list[dict[str, Any]]:
    """
    Returns a list of Intel GPU descriptors (may be empty).
    Note: Intel integrated graphics are only meaningful as a llama.cpp
    backend target via Vulkan/SYCL; this backend is a best-effort detection.
    """
    for probe in (_probe_intel_oneapi, _probe_wmic, _probe_lshw):
        result = probe()
        if result is not None:
            return result
    return []
