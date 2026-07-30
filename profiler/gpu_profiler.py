"""
gpu_profiler.py
---------------
Top-level GPU detection dispatcher.

Runs each vendor backend in defined priority order (NVIDIA, AMD, Apple, Intel)
and separates discrete GPUs from integrated GPUs (iGPUs).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .gpu_backends import amd_backend, apple_backend, intel_backend, nvidia_backend
from .resource_model import GPUResource

_BACKENDS = [
    nvidia_backend,
    amd_backend,
    apple_backend,
    intel_backend,
]


def _gpu_dict_to_resource(g: Dict[str, Any]) -> GPUResource:
    v_total = g.get("vram_total_bytes", g.get("vram_total_mb", 0) * 1024 * 1024)
    v_avail = g.get("vram_available_bytes", g.get("vram_free_mb", 0) * 1024 * 1024)
    bw = float(g.get("pcie_bandwidth_gbps", 16.0 if not g.get("is_integrated", False) else 100.0))
    util = float(g.get("utilization", 0.0))

    return GPUResource(
        capacity=float(v_total),
        available=float(v_avail),
        bandwidth=bw,
        latency=5.0,  # GPU host-device latency ~5us
        utilization=util,
        global_index=g.get("global_index", 0),
        vendor=g.get("vendor", "unknown"),
        model=g.get("model", g.get("name", "Unknown GPU")),
        uuid=g.get("uuid", ""),
        is_integrated=bool(g.get("is_integrated", False)),
        vram_total_bytes=v_total,
        vram_available_bytes=v_avail,
        vram_total_mb=g.get("vram_total_mb", 0),
        vram_free_mb=g.get("vram_free_mb", 0),
        vram_used_mb=g.get("vram_used_mb", 0),
        shared_memory_bytes=g.get("shared_memory_bytes", 0),
        compute_capability=g.get("compute_capability", "unknown"),
        backend_hint=g.get("backend_hint", "cpu"),
        pcie_gen=g.get("pcie_gen"),
        pcie_lanes=g.get("pcie_lanes"),
        driver_version=g.get("driver_version", "unknown"),
    )


def get_gpu_resources() -> Tuple[List[GPUResource], List[GPUResource]]:
    """
    Returns a tuple of (discrete_gpus, integrated_gpus) as GPUResource objects.
    """
    discrete_gpus: List[GPUResource] = []
    integrated_gpus: List[GPUResource] = []
    global_index = 0

    for backend in _BACKENDS:
        try:
            vendor_gpus = backend.detect()
        except Exception:
            vendor_gpus = []

        for gpu_dict in vendor_gpus:
            gpu_dict["global_index"] = global_index
            res = _gpu_dict_to_resource(gpu_dict)
            if res.is_integrated:
                integrated_gpus.append(res)
            else:
                discrete_gpus.append(res)
            global_index += 1

    return discrete_gpus, integrated_gpus


def profile() -> List[Dict[str, Any]]:
    """
    Backwards-compatible flat list of GPU descriptors.
    """
    discrete, igpus = get_gpu_resources()
    all_gpus = [g.to_dict() for g in discrete] + [g.to_dict() for g in igpus]
    # Add backward compatible 'name' key if missing
    for g in all_gpus:
        if "name" not in g:
            g["name"] = g.get("model", "Unknown GPU")
    return all_gpus
