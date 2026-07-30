"""
memory_profiler.py
------------------
Reports system RAM and swap statistics using psutil, with memory bandwidth estimation.
"""
from __future__ import annotations

import platform
import subprocess
from typing import Any, Dict

from .resource_model import RAMResource


def _estimate_ram_bandwidth_gbps(total_gb: float) -> float:
    """
    Estimates host RAM memory bandwidth (GB/s) based on platform/architecture heuristics.
    DDR4 Dual-channel: ~35-45 GB/s
    DDR5 Dual-channel: ~60-90 GB/s
    Apple Silicon unified memory: ~100-800 GB/s depending on model
    """
    sys_name = platform.system()
    machine = platform.machine().lower()

    if sys_name == "Darwin" and ("arm" in machine or "aarch64" in machine):
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if "Max" in out or "Ultra" in out:
                return 400.0
            elif "Pro" in out:
                return 200.0
            return 100.0
        except Exception:
            return 100.0

    # Modern x86_64 desktop/server fallback heuristic
    if total_gb >= 64:
        return 85.0  # Likely DDR5 or quad-channel DDR4
    elif total_gb >= 16:
        return 45.0  # Standard dual-channel DDR4/DDR5
    elif total_gb > 0:
        return 25.0  # Single-channel or mobile RAM

    return 25.0


def get_ram_resource() -> RAMResource:
    """
    Returns total RAM, free RAM, available RAM, utilization, and bandwidth wrapped in a RAMResource model.
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        _gb = lambda b: round(b / (1024 ** 3), 2)

        total_bytes = vm.total
        free_bytes = getattr(vm, "free", vm.available)
        available_bytes = vm.available

        total_gb = _gb(total_bytes)
        free_gb = _gb(free_bytes)
        available_gb = _gb(available_bytes)
        used_gb = _gb(vm.used)
        percent_used = vm.percent

        bandwidth_gbps = _estimate_ram_bandwidth_gbps(total_gb)

        return RAMResource(
            capacity=float(total_bytes),
            available=float(available_bytes),
            bandwidth=bandwidth_gbps,
            latency=0.07,  # Standard RAM latency ~70ns (0.07 us)
            utilization=float(percent_used),
            total_bytes=total_bytes,
            free_bytes=free_bytes,
            available_bytes=available_bytes,
            total_gb=total_gb,
            free_gb=free_gb,
            available_gb=available_gb,
            used_gb=used_gb,
            swap_total_gb=_gb(sw.total),
            swap_used_gb=_gb(sw.used),
        )
    except Exception as exc:
        return RAMResource()


def profile() -> dict[str, Any]:
    """
    Backwards-compatible dictionary function returning RAM metrics.
    """
    res = get_ram_resource()
    d = res.to_dict()
    d["percent_used"] = res.utilization
    return d
