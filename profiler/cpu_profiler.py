"""
cpu_profiler.py
---------------
Detects CPU topology, clock speed, cache hierarchy, SIMD support, NUMA nodes,
and current CPU utilization.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List

from .resource_model import CPUResource


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
    except Exception:
        return {}


def _detect_isa_extensions(raw: dict[str, Any]) -> list[str]:
    """
    Extract ISA extensions relevant to llama.cpp from cpuinfo or OS sysctl/proc.
    """
    RELEVANT = {
        "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        "avx512_vnni", "avx2", "avx", "fma", "f16c",
        "sse4_1", "sse4_2", "ssse3", "sse3", "sse2",
        "neon", "sve", "amx_bf16", "amx_int8",
    }

    flags: list[str] = []

    if "flags" in raw:
        flags = [f.lower() for f in raw["flags"]]
    elif platform.system() == "Linux":
        proc = _run_silent("grep", "-m1", "flags", "/proc/cpuinfo")
        if proc:
            flags = proc.split(":")[1].strip().split() if ":" in proc else []
    elif platform.system() == "Darwin":
        out = _run_silent("sysctl", "-a", "hw.optional")
        flags = [
            line.split(":")[0].split(".")[-1].lower()
            for line in out.splitlines()
            if line.endswith(": 1")
        ]

    return sorted(f for f in flags if f in RELEVANT)


def _get_cache_info() -> dict[str, float]:
    """
    Returns L1 (per-core, KB), L2 (per-core, KB), and L3 (total, MB) cache sizes.
    """
    caches = {"l1_cache_kb": 0.0, "l2_cache_kb": 0.0, "l3_cache_mb": 0.0}
    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()

        def _to_kb(val: Any) -> float:
            if isinstance(val, (int, float)):
                return round(float(val) / 1024, 1)
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

        if "l1_data_cache_size" in info:
            caches["l1_cache_kb"] = _to_kb(info.get("l1_data_cache_size"))
        if "l2_cache_size" in info:
            caches["l2_cache_kb"] = _to_kb(info.get("l2_cache_size"))
        if "l3_cache_size" in info:
            caches["l3_cache_mb"] = round(_to_kb(info.get("l3_cache_size")) / 1024, 1)

    except Exception:
        pass

    # Linux /sys fallback
    if platform.system() == "Linux":
        try:
            cache_base = "/sys/devices/system/cpu/cpu0/cache"
            if os.path.exists(cache_base):
                for entry in os.listdir(cache_base):
                    level_file = f"{cache_base}/{entry}/level"
                    size_file = f"{cache_base}/{entry}/size"
                    if not os.path.exists(level_file):
                        continue
                    with open(level_file) as f:
                        level = f.read().strip()
                    with open(size_file) as f:
                        size_str = f.read().strip().upper()
                    size_kb = float(size_str.replace("K", "").replace("M", "000")) if "K" in size_str or "M" in size_str else 0.0
                    if level == "1" and caches["l1_cache_kb"] == 0.0:
                        caches["l1_cache_kb"] = size_kb
                    elif level == "2" and caches["l2_cache_kb"] == 0.0:
                        caches["l2_cache_kb"] = size_kb
                    elif level == "3" and caches["l3_cache_mb"] == 0.0:
                        caches["l3_cache_mb"] = round(size_kb / 1024, 1)
        except Exception:
            pass

    return caches


def _detect_numa_topology() -> tuple[int, List[Dict[str, Any]]]:
    """
    Detects NUMA topology (number of nodes and node mapping if available).
    """
    numa_nodes = 1
    numa_topology: List[Dict[str, Any]] = []

    sys_name = platform.system()
    if sys_name == "Linux":
        try:
            node_dir = "/sys/devices/system/node"
            if os.path.exists(node_dir):
                nodes = [d for d in os.listdir(node_dir) if d.startswith("node")]
                if nodes:
                    numa_nodes = len(nodes)
                    for n in nodes:
                        numa_topology.append({"node_id": n, "path": f"{node_dir}/{n}"})
        except Exception:
            pass

    elif sys_name == "Windows":
        try:
            node_count = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetNumaHighestNodeNumber(ctypes.byref(node_count)):
                numa_nodes = max(1, node_count.value + 1)
                for i in range(numa_nodes):
                    numa_topology.append({"node_id": i})
        except Exception:
            pass

    if not numa_topology:
        numa_topology = [{"node_id": 0}]

    return numa_nodes, numa_topology


def get_cpu_resource() -> CPUResource:
    """
    Returns CPU details wrapped in a CPUResource model.
    """
    import psutil

    raw = _get_cpuinfo_data()
    cache = _get_cache_info()
    numa_nodes, numa_topology = _detect_numa_topology()

    brand = raw.get("brand_raw", "")
    if not brand:
        brand = platform.processor() or "Unknown CPU"

    hz_info = raw.get("hz_advertised_friendly", "")
    base_freq_mhz: float = 0.0
    if hz_info:
        try:
            val, unit = hz_info.split()
            multiplier = 1000.0 if unit.upper().startswith("G") else 1.0
            base_freq_mhz = round(float(val) * multiplier, 1)
        except (ValueError, AttributeError):
            pass

    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1

    try:
        utilization = float(psutil.cpu_percent(interval=0.1))
    except Exception:
        utilization = 0.0

    capacity = float(logical_cores)
    available = max(0.0, capacity * (1.0 - (utilization / 100.0)))

    return CPUResource(
        capacity=capacity,
        available=round(available, 2),
        bandwidth=0.0,  # Memory bandwidth reported in RAMResource
        latency=0.0,
        utilization=utilization,
        brand=brand,
        architecture=raw.get("arch", platform.machine()),
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        base_freq_mhz=base_freq_mhz,
        cache_l1_kb=cache["l1_cache_kb"],
        cache_l2_kb=cache["l2_cache_kb"],
        cache_l3_mb=cache["l3_cache_mb"],
        isa_extensions=_detect_isa_extensions(raw),
        numa_nodes=numa_nodes,
        numa_topology=numa_topology,
    )


def profile() -> dict[str, Any]:
    """
    Backwards-compatible dictionary function returning CPU metrics.
    """
    res = get_cpu_resource()
    d = res.to_dict()
    # Add backward compatible key names
    d["cache_l1_kb"] = res.cache_l1_kb
    d["cache_l2_kb"] = res.cache_l2_kb
    d["cache_l3_mb"] = res.cache_l3_mb
    return d
