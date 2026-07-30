"""
nvidia_backend.py
-----------------
Detects NVIDIA GPUs using a three-tier strategy:
  Tier 1: pynvml (nvidia-ml-py)  — richest data, no subprocess
  Tier 2: nvidia-smi subprocess  — standard fallback
  Tier 3: GPUtil                 — last resort Python wrapper
"""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional


def _probe_pynvml() -> List[Dict[str, Any]] | None:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: List[Dict[str, Any]] = []

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode()
            except Exception:
                driver = "unknown"

            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                cc = f"{major}.{minor}"
            except Exception:
                cc = "unknown"

            try:
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
            except Exception:
                uuid = "unknown"

            try:
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util_rates.gpu)
            except Exception:
                gpu_util = 0.0

            try:
                gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                # Gen 3 ~1GB/s per lane, Gen 4 ~2GB/s per lane, Gen 5 ~4GB/s per lane
                pcie_bw = float(width * (2 ** (gen - 1)))
            except Exception:
                gen, width, pcie_bw = None, None, 16.0

            vram_total_mb = round(mem.total / (1024 ** 2))
            vram_free_mb = round(mem.free / (1024 ** 2))
            vram_used_mb = round(mem.used / (1024 ** 2))

            gpus.append({
                "index": i,
                "vendor": "nvidia",
                "name": name,
                "model": name,
                "uuid": uuid,
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_free_mb,
                "vram_used_mb": vram_used_mb,
                "vram_total_bytes": mem.total,
                "vram_available_bytes": mem.free,
                "driver_version": driver,
                "compute_capability": cc,
                "backend_hint": "cuda",
                "is_integrated": False,
                "shared_memory_bytes": 0,
                "pcie_gen": gen,
                "pcie_lanes": width,
                "pcie_bandwidth_gbps": pcie_bw,
                "utilization": gpu_util,
            })

        pynvml.nvmlShutdown()
        return gpus if gpus else None

    except Exception:
        return None


def _probe_nvidia_smi() -> List[Dict[str, Any]] | None:
    query = (
        "index,name,uuid,"
        "memory.total,memory.free,memory.used,"
        "driver_version,compute_cap,utilization.gpu"
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        gpus: List[Dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            idx, name, uuid, mem_total, mem_free, mem_used, driver, cc = parts[:8]
            gpu_util_str = parts[8] if len(parts) > 8 else "0"

            def safe_int(v: str) -> int:
                try:
                    return int(float(v))
                except ValueError:
                    return 0

            def safe_float(v: str) -> float:
                try:
                    return float(v)
                except ValueError:
                    return 0.0

            t_mb = safe_int(mem_total)
            f_mb = safe_int(mem_free)
            u_mb = safe_int(mem_used)

            gpus.append({
                "index": safe_int(idx),
                "vendor": "nvidia",
                "name": name,
                "model": name,
                "uuid": uuid,
                "vram_total_mb": t_mb,
                "vram_free_mb": f_mb,
                "vram_used_mb": u_mb,
                "vram_total_bytes": t_mb * 1024 * 1024,
                "vram_available_bytes": f_mb * 1024 * 1024,
                "driver_version": driver,
                "compute_capability": cc,
                "backend_hint": "cuda",
                "is_integrated": False,
                "shared_memory_bytes": 0,
                "pcie_gen": 4,
                "pcie_lanes": 16,
                "pcie_bandwidth_gbps": 31.5,
                "utilization": safe_float(gpu_util_str),
            })

        return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


def _probe_gputil() -> List[Dict[str, Any]] | None:
    try:
        import GPUtil  # type: ignore

        raw_gpus = GPUtil.getGPUs()
        if not raw_gpus:
            return None

        return [
            {
                "index": g.id,
                "vendor": "nvidia",
                "name": g.name,
                "model": g.name,
                "uuid": g.uuid,
                "vram_total_mb": round(g.memoryTotal),
                "vram_free_mb": round(g.memoryFree),
                "vram_used_mb": round(g.memoryUsed),
                "vram_total_bytes": int(g.memoryTotal * 1024 * 1024),
                "vram_available_bytes": int(g.memoryFree * 1024 * 1024),
                "driver_version": g.driver,
                "compute_capability": "unknown",
                "backend_hint": "cuda",
                "is_integrated": False,
                "shared_memory_bytes": 0,
                "pcie_bandwidth_gbps": 16.0,
                "utilization": float(g.load * 100.0),
            }
            for g in raw_gpus
        ]
    except Exception:
        return None


def detect() -> List[Dict[str, Any]]:
    for probe in (_probe_pynvml, _probe_nvidia_smi, _probe_gputil):
        result = probe()
        if result is not None:
            return result
    return []
