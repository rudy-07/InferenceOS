"""
hardware_profiler.py
--------------------
Main entry point for the InferenceOS hardware profiler.

Usage (CLI):
    python -m profiler.hardware_profiler [--output hardware_profile.json] [--verbose]

Usage (API):
    from profiler.hardware_profiler import run_profiler
    profile = run_profiler()  # returns a dict
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any

from . import cpu_profiler, memory_profiler, gpu_profiler

# ── Schema version — bump when the output structure changes ──
SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Inference hint derivation
# ---------------------------------------------------------------------------

# Quant selection thresholds (VRAM of primary GPU, in MB)
_QUANT_TABLE = [
    # (min_vram_mb, quant_name)  — evaluated in descending order
    (24_000, "Q8_0"),
    (16_000, "Q6_K"),
    (12_000, "Q5_K_M"),
    (8_000,  "Q4_K_M"),
    (4_000,  "Q4_0"),
    (0,      "Q3_K_M"),   # CPU-only or very small VRAM
]

_CPU_QUANT_TABLE = [
    # Prefer quality over speed on CPU; AVX512 handles Q4_K_M well
    ("avx512f", "Q4_K_M"),
    ("avx2",    "Q4_0"),
    ("",        "Q4_0"),   # baseline
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
    cpu: dict[str, Any],
    memory: dict[str, Any],
    gpus: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Inspects the profiled data and returns a set of opinionated hints
    that downstream phases (build system, orchestrator) consume directly.
    """
    if not gpus:
        # ── CPU-only path ──
        return {
            "recommended_backend": "cpu",
            "recommended_quant": _cpu_quant(cpu.get("isa_extensions", [])),
            "max_gpu_layers": 0,
            "parallelism_threads": cpu.get("logical_cores", 4),
            "primary_gpu_index": None,
            "estimated_context_window": _estimate_cpu_ctx(memory),
        }

    # ── GPU path — pick the primary GPU (most VRAM) ──
    primary = max(gpus, key=lambda g: g.get("vram_total_mb", 0))
    vendor = primary.get("vendor", "unknown")
    backend = primary.get("backend_hint", "cpu")
    vram_mb = primary.get("vram_total_mb", 0)
    vram_free_mb = primary.get("vram_free_mb", vram_mb)

    backend_map = {
        "cuda":  "cuda",
        "rocm":  "rocm",
        "metal": "metal",
        "vulkan": "vulkan",
    }
    recommended_backend = backend_map.get(backend, "cpu")

    # If we have multiple GPUs, hint at multi-GPU support
    multi_gpu = len(gpus) > 1

    return {
        "recommended_backend": recommended_backend,
        "recommended_quant": _select_quant(vram_free_mb),
        # -1 means "offload all layers" (llama.cpp convention)
        "max_gpu_layers": -1 if vram_mb > 0 else 0,
        "parallelism_threads": cpu.get("logical_cores", 4),
        "primary_gpu_index": primary.get("global_index", 0),
        "multi_gpu": multi_gpu,
        "total_vram_mb": sum(g.get("vram_total_mb", 0) for g in gpus),
        "estimated_context_window": _estimate_gpu_ctx(vram_free_mb),
    }


def _estimate_cpu_ctx(memory: dict[str, Any]) -> int:
    """
    Rough context window estimate for CPU-only inference.
    KV cache for a 7B Q4 model: ~0.5 MB per 256 tokens.
    We allow up to 25% of available RAM for the KV cache.
    """
    available_gb = memory.get("available_gb", 4.0)
    kv_budget_mb = available_gb * 1024 * 0.25
    # Each 256 tokens ≈ 0.5 MB for a 7B model Q4
    ctx = int((kv_budget_mb / 0.5) * 256)
    # Clamp to sane limits
    return max(512, min(ctx, 131_072))


def _estimate_gpu_ctx(vram_free_mb: int) -> int:
    """
    Rough context window estimate based on free VRAM.
    KV cache for a 7B Q4 model: ~0.5 MB per 256 tokens.
    We allow up to 40% of free VRAM for the KV cache.
    """
    kv_budget_mb = vram_free_mb * 0.40
    ctx = int((kv_budget_mb / 0.5) * 256)
    return max(512, min(ctx, 131_072))


# ---------------------------------------------------------------------------
# OS info
# ---------------------------------------------------------------------------

def _os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Core profiler
# ---------------------------------------------------------------------------

def run_profiler(verbose: bool = False) -> dict[str, Any]:
    """
    Runs all sub-profilers and assembles the full hardware profile dict.

    Parameters
    ----------
    verbose : bool
        If True, prints progress messages to stderr.

    Returns
    -------
    dict
        The complete hardware profile (JSON-serialisable).
    """
    def _log(msg: str) -> None:
        if verbose:
            print(f"[profiler] {msg}", file=sys.stderr)

    _log("Probing OS …")
    os_data = _os_info()

    _log("Probing CPU …")
    cpu_data = cpu_profiler.profile()

    _log("Probing memory …")
    memory_data = memory_profiler.profile()

    _log("Probing GPUs …")
    gpu_data = gpu_profiler.profile()

    if verbose:
        gpu_names = [g.get("name", "?") for g in gpu_data]
        _log(f"  Found {len(gpu_data)} GPU(s): {gpu_names if gpu_names else 'none'}")

    _log("Deriving inference hints …")
    hints = _derive_inference_hints(cpu_data, memory_data, gpu_data)

    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "os": os_data,
        "cpu": cpu_data,
        "memory": memory_data,
        "gpus": gpu_data,
        "inference_hints": hints,
    }

    return profile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hardware_profiler",
        description=(
            "InferenceOS Hardware Profiler — "
            "detect system capabilities and emit a JSON configuration profile."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="hardware_profile.json",
        help="Path to write the JSON profile (default: hardware_profile.json)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the JSON profile to stdout",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information to stderr",
    )
    parser.add_argument(
        "--no-file",
        action="store_true",
        help="Skip writing to a file (useful with --stdout)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    profile = run_profiler(verbose=args.verbose)
    json_str = json.dumps(profile, indent=2, ensure_ascii=False)

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
