"""
amd_backend.py
--------------
Detects AMD GPUs using three strategies:
  Tier 1: amdsmi Python bindings  — richest data (ROCm 5.6+)
  Tier 2: rocm-smi subprocess     — JSON output mode
  Tier 3: vulkaninfo fallback
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional


def _is_amd_igpu(name: str) -> bool:
    name_lower = name.lower()
    return "graphics" in name_lower and not any(d in name_lower for d in ("rx ", "pro ", "instinct", "vega 56", "vega 64", "xtx", "xt"))


def _probe_amdsmi() -> List[Dict[str, Any]] | None:
    try:
        import amdsmi  # type: ignore

        amdsmi.amdsmi_init()
        processors = amdsmi.amdsmi_get_processor_handles()
        gpus: List[Dict[str, Any]] = []

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

            is_igpu = _is_amd_igpu(name)
            vram_free_mb = max(0, vram_total_mb - vram_used_mb)

            gpus.append({
                "index": i,
                "vendor": "amd",
                "name": name,
                "model": name,
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_free_mb,
                "vram_used_mb": vram_used_mb,
                "vram_total_bytes": vram_total_mb * 1024 * 1024,
                "vram_available_bytes": vram_free_mb * 1024 * 1024,
                "driver_version": str(driver),
                "compute_capability": "gfx1100" if "7900" in name else "unknown",
                "backend_hint": "rocm",
                "is_integrated": is_igpu,
                "shared_memory_bytes": vram_total_mb * 1024 * 1024 if is_igpu else 0,
                "pcie_bandwidth_gbps": 16.0 if not is_igpu else 0.0,
                "utilization": 0.0,
            })

        amdsmi.amdsmi_shut_down()
        return gpus if gpus else None

    except Exception:
        return None


def _probe_rocm_smi() -> List[Dict[str, Any]] | None:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showallinfo", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            gpus: List[Dict[str, Any]] = []

            for key, gpu_data in data.items():
                if not key.startswith("card"):
                    continue
                idx = int(key.replace("card", ""))
                name = gpu_data.get("Card Series", gpu_data.get("Card model", "AMD GPU"))
                driver = gpu_data.get("Driver version", "unknown")

                def _parse_mb(s: str) -> int:
                    try:
                        return int(float(str(s).split()[0]))
                    except (ValueError, IndexError):
                        return 0

                vram_total_mb = _parse_mb(gpu_data.get("VRAM Total Memory (B)", "0"))
                if vram_total_mb == 0:
                    vram_total_mb = _parse_mb(gpu_data.get("VRAM Total Used Memory (B)", "0"))

                if vram_total_mb > 100_000:
                    vram_total_mb = round(vram_total_mb / (1024 ** 2))

                vram_used_mb = _parse_mb(gpu_data.get("VRAM Total Used Memory (B)", "0"))
                if vram_used_mb > 100_000:
                    vram_used_mb = round(vram_used_mb / (1024 ** 2))

                vram_free_mb = max(0, vram_total_mb - vram_used_mb)
                is_igpu = _is_amd_igpu(name)

                gpus.append({
                    "index": idx,
                    "vendor": "amd",
                    "name": name.strip(),
                    "model": name.strip(),
                    "vram_total_mb": vram_total_mb,
                    "vram_free_mb": vram_free_mb,
                    "vram_used_mb": vram_used_mb,
                    "vram_total_bytes": vram_total_mb * 1024 * 1024,
                    "vram_available_bytes": vram_free_mb * 1024 * 1024,
                    "driver_version": driver,
                    "compute_capability": "unknown",
                    "backend_hint": "rocm",
                    "is_integrated": is_igpu,
                    "shared_memory_bytes": vram_total_mb * 1024 * 1024 if is_igpu else 0,
                    "pcie_bandwidth_gbps": 16.0 if not is_igpu else 0.0,
                    "utilization": 0.0,
                })

            return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass

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
                    g_name = line.strip()
                    is_igpu = _is_amd_igpu(g_name)
                    gpus.append({
                        "index": i,
                        "vendor": "amd",
                        "name": g_name,
                        "model": g_name,
                        "vram_total_mb": 0,
                        "vram_free_mb": 0,
                        "vram_used_mb": 0,
                        "vram_total_bytes": 0,
                        "vram_available_bytes": 0,
                        "driver_version": "unknown",
                        "compute_capability": "unknown",
                        "backend_hint": "rocm",
                        "is_integrated": is_igpu,
                        "shared_memory_bytes": 0,
                        "pcie_bandwidth_gbps": 0.0,
                        "utilization": 0.0,
                    })
            return gpus if gpus else None
    except Exception:
        pass

    return None


def _probe_vulkaninfo_amd() -> List[Dict[str, Any]] | None:
    import shutil

    if not shutil.which("vulkaninfo"):
        return None

    try:
        res = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=10,
        )
        if res.returncode != 0 or not res.stdout:
            return None

        raw_gpus: List[Dict[str, str]] = []
        current: Dict[str, str] = {}

        for line in res.stdout.splitlines():
            s = line.strip()
            if s.startswith("GPU") and s.endswith(":"):
                if current and (current.get("vendorID", "").lower() == "0x1002" or "radeon" in current.get("deviceName", "").lower() or "amd" in current.get("deviceName", "").lower()):
                    raw_gpus.append(current)
                current = {}
            elif "=" in s:
                parts = s.split("=", 1)
                current[parts[0].strip()] = parts[1].strip()
            elif ":" in s:
                parts = s.split(":", 1)
                current[parts[0].strip()] = parts[1].strip()

        if current and (current.get("vendorID", "").lower() == "0x1002" or "radeon" in current.get("deviceName", "").lower() or "amd" in current.get("deviceName", "").lower()):
            raw_gpus.append(current)

        if not raw_gpus:
            return None

        result_gpus: List[Dict[str, Any]] = []
        for i, g in enumerate(raw_gpus):
            dev_name = g.get("deviceName", "AMD Radeon GPU")
            dev_type = g.get("deviceType", "")
            driver = g.get("driverInfo", g.get("driverVersion", "unknown"))

            is_discrete = "DISCRETE" in dev_type.upper()
            is_igpu = not is_discrete or _is_amd_igpu(dev_name)

            vram_total_mb = 6144 if (is_discrete and "5600" in dev_name) else (4096 if is_discrete else 512)

            result_gpus.append({
                "index": i,
                "vendor": "amd",
                "name": dev_name,
                "model": dev_name,
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_total_mb,
                "vram_used_mb": 0,
                "vram_total_bytes": vram_total_mb * 1024 * 1024,
                "vram_available_bytes": vram_total_mb * 1024 * 1024,
                "driver_version": driver,
                "compute_capability": "vulkan",
                "backend_hint": "vulkan",
                "is_integrated": is_igpu,
                "shared_memory_bytes": vram_total_mb * 1024 * 1024 if is_igpu else 0,
                "pcie_bandwidth_gbps": 16.0 if not is_igpu else 0.0,
                "utilization": 0.0,
            })

        return result_gpus if result_gpus else None
    except Exception:
        return None


def detect() -> List[Dict[str, Any]]:
    for probe in (_probe_amdsmi, _probe_rocm_smi, _probe_vulkaninfo_amd):
        result = probe()
        if result is not None:
            return result
    return []
