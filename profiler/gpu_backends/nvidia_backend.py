"""
nvidia_backend.py
-----------------
Detects NVIDIA GPUs using a three-tier strategy:

  Tier 1: pynvml (nvidia-ml-py)  — richest data, no subprocess
  Tier 2: nvidia-smi subprocess  — standard fallback
  Tier 3: GPUtil                 — last resort Python wrapper

All tiers are guarded.  If none succeed, returns an empty list.
"""
from __future__ import annotations

import subprocess
from typing import Any


# ---------------------------------------------------------------------------
# Tier 1 — pynvml
# ---------------------------------------------------------------------------

def _probe_pynvml() -> list[dict[str, Any]] | None:
    """
    Use pynvml (part of nvidia-ml-py / pynvml packages).
    Returns None if the library is not available or NVML init fails.
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[dict[str, Any]] = []

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

            gpus.append({
                "index": i,
                "vendor": "nvidia",
                "name": name,
                "uuid": uuid,
                "vram_total_mb": round(mem.total / (1024 ** 2)),
                "vram_free_mb": round(mem.free / (1024 ** 2)),
                "vram_used_mb": round(mem.used / (1024 ** 2)),
                "driver_version": driver,
                "compute_capability": cc,
                "backend_hint": "cuda",
            })

        pynvml.nvmlShutdown()
        return gpus if gpus else None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier 2 — nvidia-smi subprocess
# ---------------------------------------------------------------------------

def _probe_nvidia_smi() -> list[dict[str, Any]] | None:
    """
    Parse output of:
      nvidia-smi --query-gpu=... --format=csv,noheader,nounits
    Returns None if nvidia-smi is not in PATH or returns no data.
    """
    query = (
        "index,name,uuid,"
        "memory.total,memory.free,memory.used,"
        "driver_version,compute_cap"
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

        gpus: list[dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            idx, name, uuid, mem_total, mem_free, mem_used, driver, cc = parts[:8]

            def safe_int(v: str) -> int:
                try:
                    return int(float(v))
                except ValueError:
                    return 0

            gpus.append({
                "index": safe_int(idx),
                "vendor": "nvidia",
                "name": name,
                "uuid": uuid,
                "vram_total_mb": safe_int(mem_total),
                "vram_free_mb": safe_int(mem_free),
                "vram_used_mb": safe_int(mem_used),
                "driver_version": driver,
                "compute_capability": cc,
                "backend_hint": "cuda",
            })

        return gpus if gpus else None

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return None


# ---------------------------------------------------------------------------
# Tier 3 — GPUtil
# ---------------------------------------------------------------------------

def _probe_gputil() -> list[dict[str, Any]] | None:
    """
    Last resort: GPUtil Python package.
    Less detailed but widely installed.
    """
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
                "uuid": g.uuid,
                "vram_total_mb": round(g.memoryTotal),
                "vram_free_mb": round(g.memoryFree),
                "vram_used_mb": round(g.memoryUsed),
                "driver_version": g.driver,
                "compute_capability": "unknown",
                "backend_hint": "cuda",
            }
            for g in raw_gpus
        ]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect() -> list[dict[str, Any]]:
    """
    Returns a list of NVIDIA GPU descriptors (may be empty).
    Tries each tier in order, stopping at the first success.
    """
    for probe in (_probe_pynvml, _probe_nvidia_smi, _probe_gputil):
        result = probe()
        if result is not None:
            return result
    return []
