"""
intel_backend.py
----------------
Detects Intel discrete GPUs (Arc / Xe series) and integrated graphics.
"""
from __future__ import annotations

import platform
import subprocess
from typing import Any, Dict, List, Optional


def _is_intel_igpu(name: str) -> bool:
    name_lower = name.lower()
    if "arc" in name_lower and not ("arc graphics" in name_lower or "integrated" in name_lower):
        return False
    return any(k in name_lower for k in ("uhd", "iris", "hd graphics", "integrated", "graphics"))


def _probe_intel_oneapi() -> List[Dict[str, Any]] | None:
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
        import torch  # type: ignore

        if not torch.xpu.is_available():
            return None

        count = torch.xpu.device_count()
        gpus: List[Dict[str, Any]] = []
        for i in range(count):
            props = torch.xpu.get_device_properties(i)
            name = props.name
            total_mb = round(props.total_memory / (1024 ** 2))
            is_igpu = _is_intel_igpu(name)

            gpus.append({
                "index": i,
                "vendor": "intel",
                "name": name,
                "model": name,
                "vram_total_mb": total_mb,
                "vram_free_mb": total_mb,
                "vram_used_mb": 0,
                "vram_total_bytes": props.total_memory,
                "vram_available_bytes": props.total_memory,
                "driver_version": "unknown",
                "compute_capability": "oneAPI/XPU",
                "backend_hint": "vulkan",
                "is_integrated": is_igpu,
                "shared_memory_bytes": props.total_memory if is_igpu else 0,
                "pcie_bandwidth_gbps": 16.0 if not is_igpu else 0.0,
                "utilization": 0.0,
            })
        return gpus if gpus else None
    except Exception:
        pass

    return None


def _probe_wmic() -> List[Dict[str, Any]] | None:
    if platform.system() != "Windows":
        return None

    raw_lines: List[str] = []

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

    gpus: List[Dict[str, Any]] = []
    intel_idx = 0
    for line in raw_lines:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts or parts[0].lower() in ("node", "name", ""):
            continue

        name = ""
        adapter_ram = 0
        driver = "unknown"

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
                driver = p

        if not name:
            continue

        vram_mb = round(adapter_ram / (1024 ** 2)) if adapter_ram else 0
        is_igpu = _is_intel_igpu(name)
        vram_bytes = adapter_ram if adapter_ram else 0

        gpus.append({
            "index": intel_idx,
            "vendor": "intel",
            "name": name,
            "model": name,
            "vram_total_mb": vram_mb,
            "vram_free_mb": vram_mb,
            "vram_used_mb": 0,
            "vram_total_bytes": vram_bytes,
            "vram_available_bytes": vram_bytes,
            "driver_version": driver,
            "compute_capability": "unknown",
            "backend_hint": "vulkan",
            "is_integrated": is_igpu,
            "shared_memory_bytes": vram_bytes if is_igpu else 0,
            "pcie_bandwidth_gbps": 16.0 if not is_igpu else 0.0,
            "utilization": 0.0,
        })
        intel_idx += 1

    return gpus if gpus else None


def _probe_lshw() -> List[Dict[str, Any]] | None:
    if platform.system() != "Linux":
        return None

    try:
        result = subprocess.run(
            ["lshw", "-C", "display", "-short"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None

        gpus: List[Dict[str, Any]] = []
        idx = 0
        for line in result.stdout.splitlines():
            lower = line.lower()
            if "intel" in lower and ("display" in lower or "vga" in lower):
                parts = line.split(maxsplit=3)
                name = parts[-1].strip() if len(parts) >= 4 else "Intel GPU"
                is_igpu = _is_intel_igpu(name)

                gpus.append({
                    "index": idx,
                    "vendor": "intel",
                    "name": name,
                    "model": name,
                    "vram_total_mb": 0,
                    "vram_free_mb": 0,
                    "vram_used_mb": 0,
                    "vram_total_bytes": 0,
                    "vram_available_bytes": 0,
                    "driver_version": "unknown",
                    "compute_capability": "unknown",
                    "backend_hint": "vulkan",
                    "is_integrated": is_igpu,
                    "shared_memory_bytes": 0,
                    "pcie_bandwidth_gbps": 16.0 if not is_igpu else 0.0,
                    "utilization": 0.0,
                })
                idx += 1

        return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def detect() -> List[Dict[str, Any]]:
    for probe in (_probe_intel_oneapi, _probe_wmic, _probe_lshw):
        result = probe()
        if result is not None:
            return result
    return []
