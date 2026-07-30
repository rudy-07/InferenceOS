"""
hardware_profiler.py
--------------------
Main entry point for the InferenceOS hardware profiler.
Assembles the SystemResources model and exposes the public API functions:
  - getSystemResources() / get_system_resources()
  - getAvailableMemory() / get_available_memory()
  - getGPUs() / get_gpus()
  - getCPUs() / get_cpus()
  - estimateBandwidth() / estimate_bandwidth()
  - run_profiler()

CLI Usage:
    python -m profiler.hardware_profiler [--output hardware_profile.json] [--verbose]
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import cpu_profiler, gpu_profiler, interconnect_profiler, memory_profiler, storage_profiler
from .resource_model import (
    CPUResource,
    GPUResource,
    InterconnectResource,
    RAMResource,
    StorageResource,
    SystemResources,
)

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Quant selection and inference hint derivation
# ---------------------------------------------------------------------------

_QUANT_TABLE = [
    (24_000, "Q8_0"),
    (16_000, "Q6_K"),
    (12_000, "Q5_K_M"),
    (8_000,  "Q4_K_M"),
    (4_000,  "Q4_0"),
    (0,      "Q3_K_M"),
]

_CPU_QUANT_TABLE = [
    ("avx512f", "Q4_K_M"),
    ("avx2",    "Q4_0"),
    ("",        "Q4_0"),
]


def _select_quant(vram_mb: int) -> str:
    for threshold, quant in _QUANT_TABLE:
        if vram_mb >= threshold:
            return quant
    return "Q3_K_M"


def _cpu_quant(isa_extensions: list[str]) -> str:
    ext_set = set(isa_extensions)
    for required_ext, quant in _CPU_QUANT_TABLE:
        if not required_ext or required_ext in ext_set:
            return quant
    return "Q4_0"


def _derive_inference_hints(
    cpu_res: CPUResource,
    ram_res: RAMResource,
    discrete_gpus: List[GPUResource],
    igpus: List[GPUResource],
) -> dict[str, Any]:
    all_gpus = discrete_gpus + igpus

    if not all_gpus:
        return {
            "recommended_backend": "cpu",
            "recommended_quant": _cpu_quant(cpu_res.isa_extensions),
            "max_gpu_layers": 0,
            "parallelism_threads": cpu_res.logical_cores,
            "primary_gpu_index": None,
            "estimated_context_window": _estimate_cpu_ctx(ram_res.available_gb),
        }

    primary = max(all_gpus, key=lambda g: g.vram_total_mb)
    backend = primary.backend_hint
    vram_mb = primary.vram_total_mb
    vram_free_mb = primary.vram_free_mb

    backend_map = {
        "cuda": "cuda",
        "rocm": "rocm",
        "metal": "metal",
        "vulkan": "vulkan",
    }
    recommended_backend = backend_map.get(backend, "cpu")
    multi_gpu = len(all_gpus) > 1

    return {
        "recommended_backend": recommended_backend,
        "recommended_quant": _select_quant(vram_free_mb),
        "max_gpu_layers": -1 if vram_mb > 0 else 0,
        "parallelism_threads": cpu_res.logical_cores,
        "primary_gpu_index": primary.global_index,
        "multi_gpu": multi_gpu,
        "total_vram_mb": sum(g.vram_total_mb for g in all_gpus),
        "estimated_context_window": _estimate_gpu_ctx(vram_free_mb),
    }


def _estimate_cpu_ctx(available_gb: float) -> int:
    kv_budget_mb = available_gb * 1024 * 0.25
    ctx = int((kv_budget_mb / 0.5) * 256)
    return max(512, min(ctx, 131_072))


def _estimate_gpu_ctx(vram_free_mb: int) -> int:
    kv_budget_mb = vram_free_mb * 0.40
    ctx = int((kv_budget_mb / 0.5) * 256)
    return max(512, min(ctx, 131_072))


def _os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Main SystemResources builder
# ---------------------------------------------------------------------------

def get_system_resources(verbose: bool = False) -> SystemResources:
    """
    Discovers all host compute and memory resources and returns a SystemResources graph.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(f"[profiler] {msg}", file=sys.stderr)

    _log("Probing OS...")
    os_data = _os_info()

    _log("Probing CPU...")
    cpu_res = cpu_profiler.get_cpu_resource()

    _log("Probing RAM...")
    ram_res = memory_profiler.get_ram_resource()

    _log("Probing GPUs...")
    discrete_gpus, igpus = gpu_profiler.get_gpu_resources()

    _log("Probing Storage...")
    storage_resources = storage_profiler.get_storage_resources()

    _log("Probing Interconnects...")
    interconnect_resources = interconnect_profiler.get_interconnect_resources(discrete_gpus, ram_res.bandwidth)

    _log("Deriving inference hints...")
    hints = _derive_inference_hints(cpu_res, ram_res, discrete_gpus, igpus)

    return SystemResources(
        cpus=[cpu_res],
        gpus=discrete_gpus,
        igpus=igpus,
        ram=ram_res,
        storage=storage_resources,
        interconnects=interconnect_resources,
        os=os_data,
        schema_version=SCHEMA_VERSION,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        inference_hints=hints,
    )


# ---------------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------------

# Exact camelCase required by prompt
def getSystemResources() -> SystemResources:
    """Returns the unified SystemResources model."""
    return get_system_resources()


def getAvailableMemory() -> Dict[str, Any]:
    """Returns system memory and available memory metrics."""
    ram = memory_profiler.get_ram_resource()
    return ram.to_dict()


def getGPUs() -> List[Dict[str, Any]]:
    """Returns list of all GPUs (discrete and integrated)."""
    discrete, igpus = gpu_profiler.get_gpu_resources()
    return [g.to_dict() for g in discrete] + [g.to_dict() for g in igpus]


def getCPUs() -> List[Dict[str, Any]]:
    """Returns CPU resource specifications."""
    cpu = cpu_profiler.get_cpu_resource()
    return [cpu.to_dict()]


def estimateBandwidth() -> Dict[str, float]:
    """
    Returns estimated bandwidth for system memory, GPU interconnect, and storage.
    """
    ram = memory_profiler.get_ram_resource()
    discrete, igpus = gpu_profiler.get_gpu_resources()
    storage = storage_profiler.get_storage_resources()

    gpu_bw = max([g.bandwidth for g in discrete + igpus], default=0.0)
    storage_bw = max([s.bandwidth for s in storage], default=0.0)

    return {
        "ram_bandwidth_gbps": ram.bandwidth,
        "max_gpu_bandwidth_gbps": gpu_bw,
        "max_storage_bandwidth_gbps": storage_bw,
    }


# Pythonic snake_case aliases
get_available_memory = getAvailableMemory
get_gpus = getGPUs
get_cpus = getCPUs
estimate_bandwidth = estimateBandwidth


def run_profiler(verbose: bool = False) -> Dict[str, Any]:
    """
    Runs full profiler and returns serializable dict matching required output schema:
    {
      "cpu": {...},
      "ram": {...},
      "gpus": [...],
      "igpus": [...],
      "storage": [...]
    }
    """
    sys_res = get_system_resources(verbose=verbose)
    return sys_res.to_dict()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hardware_profiler",
        description="InferenceOS Hardware Profiler & Resource Manager",
    )
    parser.add_argument(
        "--output", "-o", default="hardware_profile.json",
        help="Path to write JSON profile (default: hardware_profile.json)",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="Print JSON profile to stdout",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress info to stderr",
    )
    parser.add_argument(
        "--no-file", action="store_true",
        help="Skip writing file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile_dict = run_profiler(verbose=args.verbose)
    json_str = json.dumps(profile_dict, indent=2, ensure_ascii=False)

    if not args.no_file:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")
        if args.verbose:
            print(f"[profiler] Profile written to: {output_path.resolve()}", file=sys.stderr)

    if args.stdout or args.no_file:
        print(json_str)


if __name__ == "__main__":
    main()
