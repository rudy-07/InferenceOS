"""
memory_profiler.py
------------------
Reports system RAM and swap statistics using psutil.

psutil is the most reliable cross-platform source for this data and
requires no elevated privileges.
"""
from __future__ import annotations

from typing import Any


def profile() -> dict[str, Any]:
    """
    Returns total RAM, currently available RAM, and swap totals in GiB.
    Values are rounded to two decimal places.
    Never raises.
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        _gb = lambda b: round(b / (1024 ** 3), 2)

        return {
            "total_gb": _gb(vm.total),
            "available_gb": _gb(vm.available),
            "used_gb": _gb(vm.used),
            "percent_used": vm.percent,
            "swap_total_gb": _gb(sw.total),
            "swap_used_gb": _gb(sw.used),
        }
    except Exception as exc:
        return {
            "total_gb": 0.0,
            "available_gb": 0.0,
            "used_gb": 0.0,
            "percent_used": 0.0,
            "swap_total_gb": 0.0,
            "swap_used_gb": 0.0,
            "error": str(exc),
        }
