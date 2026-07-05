"""
gpu_profiler.py
---------------
Top-level GPU detection dispatcher.

Runs each vendor backend in a defined priority order and aggregates results.
A backend that fails (missing drivers, not applicable OS, etc.) silently
returns an empty list — it never raises.
"""
from __future__ import annotations

from typing import Any

from .gpu_backends import nvidia_backend, amd_backend, apple_backend, intel_backend


_BACKENDS = [
    nvidia_backend,
    amd_backend,
    apple_backend,
    intel_backend,
]


def profile() -> list[dict[str, Any]]:
    """
    Runs all GPU backends and returns a flat list of GPU descriptors,
    each tagged with a `vendor` and `backend_hint` field.

    The order within the list reflects detection priority (NVIDIA first,
    then AMD, Apple, Intel).  Index values are local to each vendor.
    """
    all_gpus: list[dict[str, Any]] = []
    global_index = 0

    for backend in _BACKENDS:
        try:
            vendor_gpus = backend.detect()
        except Exception:
            vendor_gpus = []

        for gpu in vendor_gpus:
            gpu["global_index"] = global_index
            all_gpus.append(gpu)
            global_index += 1

    return all_gpus
