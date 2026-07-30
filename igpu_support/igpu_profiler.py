"""
igpu_profiler.py
----------------
Integrated GPU profiling for InferenceOS Phase 7.

Reads iGPU information from the hardware profile (populated by Phase 1)
and computes an IgpuProfile that describes:
  - What hardware is present (vendor, VRAM, memory model)
  - Whether the iGPU shares system RAM (UMA) or has dedicated VRAM
  - Which Vulkan/Metal device index corresponds to the iGPU
  - An overall suitability score (0.0–1.0) for inference offload

Suitability score formula
--------------------------
Four components combine into a single 0–1 score:

  vram_score    = clamp(vram_mb / 2048, 0, 1)
                  Full score at 2 GB. Most Intel/AMD iGPUs have 512 MB–2 GB.

  bw_score      = clamp(effective_bw_gbps / 50, 0, 1)
                  Full score at 50 GB/s (Apple M-series range).
                  Shared-bus iGPUs are typically 25–50 GB/s.

  ram_headroom  = 1 - clamp(ram_utilization_pct / 100, 0, 1)
                  Penalises iGPU use when system RAM is already under pressure,
                  because shared-memory iGPUs draw from the same pool.

  backend_score = 1.0 if backend_hint in {"vulkan", "metal"}
                  0.3 if backend_hint == "opencl"
                  0.0 otherwise

  igpu_score = 0.30 * vram_score
             + 0.30 * bw_score
             + 0.25 * ram_headroom
             + 0.15 * backend_score

Vendor notes
------------
  AMD (GCN/RDNA iGPU):   shared VRAM from system RAM, Vulkan backend
  Intel (UHD/Xe):         shared VRAM from system RAM, Vulkan backend
  Apple (M-series):       unified memory, Metal backend — excellent suitability
  NVIDIA (MX series):     dedicated VRAM, CUDA — treated like a small dGPU

Minimum viable iGPU:
  vram_mb >= 256 AND backend_hint in {vulkan, metal, opencl}
  Otherwise detected=True but enabled=False (and score → 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Minimum VRAM (MB) below which we will not attempt iGPU offload at all.
_MIN_VIABLE_VRAM_MB = 256

# Bandwidth reference points for scoring (GB/s)
_BW_FULL_SCORE_GBPS = 50.0
_VRAM_FULL_SCORE_MB = 2048

# Weights
_W_VRAM    = 0.30
_W_BW      = 0.30
_W_HEADRM  = 0.25
_W_BACKEND = 0.15

# Backend scores
_BACKEND_SCORES: Dict[str, float] = {
    "vulkan": 1.0,
    "metal":  1.0,
    "cuda":   0.9,
    "rocm":   0.9,
    "opencl": 0.3,
    "cpu":    0.0,
}


@dataclass
class IgpuProfile:
    """
    Profile for the integrated GPU detected on the host system.

    Attributes
    ----------
    detected : bool
        True if at least one iGPU is present in ``hw_profile["igpus"]``.
    enabled : bool
        True if the iGPU meets minimum requirements (VRAM + backend).
        False when suitability_score would be near-zero anyway.
    vendor : str
        Vendor string: ``"amd"``, ``"intel"``, ``"apple"``, ``"nvidia"``,
        or ``"unknown"``.
    model : str
        Human-readable GPU model name.
    vram_mb : int
        Total VRAM in MB (may be shared from system RAM for AMD/Intel).
    vram_bytes : int
        Total VRAM in bytes.
    shared_memory_bytes : int
        Bytes of shared (UMA) system RAM used as VRAM. 0 for dedicated VRAM.
    is_shared_memory : bool
        True for AMD/Intel iGPUs that share system RAM as VRAM.
    backend_hint : str
        Preferred llama.cpp backend: ``"vulkan"``, ``"metal"``, ``"cuda"``,
        ``"opencl"``, or ``"cpu"``.
    vulkan_device_index : int
        Vulkan logical device index for this iGPU (from ``global_index``
        in the hardware profile).
    bandwidth_gbps : float
        Estimated memory bandwidth in GB/s.  For shared-memory iGPUs this
        is bounded by the system bus bandwidth read from the hardware profile.
    suitability_score : float
        Composite score in [0.0, 1.0].  Higher is better.
    warnings : List[str]
        Non-fatal informational messages about limitations.
    """
    detected: bool
    enabled: bool
    vendor: str
    model: str
    vram_mb: int
    vram_bytes: int
    shared_memory_bytes: int
    is_shared_memory: bool
    backend_hint: str
    vulkan_device_index: int
    bandwidth_gbps: float
    suitability_score: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return {
            "detected": self.detected,
            "enabled": self.enabled,
            "vendor": self.vendor,
            "model": self.model,
            "vram_mb": self.vram_mb,
            "vram_bytes": self.vram_bytes,
            "shared_memory_bytes": self.shared_memory_bytes,
            "is_shared_memory": self.is_shared_memory,
            "backend_hint": self.backend_hint,
            "vulkan_device_index": self.vulkan_device_index,
            "bandwidth_gbps": round(self.bandwidth_gbps, 2),
            "suitability_score": round(self.suitability_score, 4),
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    def is_apple_silicon(self) -> bool:
        """True for Apple unified-memory GPUs (highest suitability)."""
        return self.vendor == "apple"

    def has_vulkan(self) -> bool:
        """True if the iGPU backend is Vulkan."""
        return self.backend_hint == "vulkan"

    def has_metal(self) -> bool:
        """True if the iGPU backend is Metal (Apple)."""
        return self.backend_hint == "metal"


class IgpuProfiler:
    """
    Reads iGPU information from a hardware profile dict and produces an
    :class:`IgpuProfile`.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile produced by the Phase 1 profiler.
    """

    def __init__(self, hw_profile: Dict[str, Any]) -> None:
        self.hw_profile = hw_profile

    def profile(self) -> IgpuProfile:
        """
        Build and return an :class:`IgpuProfile` for the first detected iGPU.

        If no iGPU is found, returns a profile with ``detected=False`` and
        ``enabled=False``.
        """
        igpus: List[Dict[str, Any]] = self.hw_profile.get("igpus", [])

        if not igpus:
            return self._no_igpu()

        # Use the first iGPU (or the one with most VRAM if multiple exist)
        igpu_dict = max(igpus, key=lambda g: g.get("vram_total_mb", 0))

        vendor       = str(igpu_dict.get("vendor", "unknown")).lower()
        model        = str(igpu_dict.get("model", igpu_dict.get("name", "Unknown iGPU")))
        vram_mb      = int(igpu_dict.get("vram_total_mb", 0))
        vram_bytes   = int(igpu_dict.get("vram_total_bytes", vram_mb * 1024 * 1024))
        shared_bytes = int(igpu_dict.get("shared_memory_bytes", 0))
        backend_hint = str(igpu_dict.get("backend_hint", "cpu")).lower()
        vulkan_idx   = int(igpu_dict.get("global_index", 0))

        # Determine if shared memory (AMD/Intel UMA)
        is_shared = shared_bytes > 0 or vendor in ("amd", "intel")

        # Bandwidth: use profile value, fall back to system bus BW
        bw = float(igpu_dict.get("bandwidth", 0.0))
        if bw <= 0:
            bw = self._system_bus_bandwidth_gbps()

        warnings: List[str] = []

        # Minimum viability check
        if vram_mb < _MIN_VIABLE_VRAM_MB:
            warnings.append(
                f"iGPU VRAM is very low ({vram_mb} MB < {_MIN_VIABLE_VRAM_MB} MB minimum). "
                "iGPU offload disabled."
            )
            return IgpuProfile(
                detected=True, enabled=False,
                vendor=vendor, model=model,
                vram_mb=vram_mb, vram_bytes=vram_bytes,
                shared_memory_bytes=shared_bytes,
                is_shared_memory=is_shared,
                backend_hint=backend_hint,
                vulkan_device_index=vulkan_idx,
                bandwidth_gbps=bw,
                suitability_score=0.0,
                warnings=warnings,
            )

        if backend_hint not in ("vulkan", "metal", "cuda", "rocm", "opencl"):
            warnings.append(
                f"iGPU backend '{backend_hint}' is not supported for inference offload. "
                "Falling back to CPU."
            )
            return IgpuProfile(
                detected=True, enabled=False,
                vendor=vendor, model=model,
                vram_mb=vram_mb, vram_bytes=vram_bytes,
                shared_memory_bytes=shared_bytes,
                is_shared_memory=is_shared,
                backend_hint=backend_hint,
                vulkan_device_index=vulkan_idx,
                bandwidth_gbps=bw,
                suitability_score=0.0,
                warnings=warnings,
            )

        # Compute suitability score
        ram_util = self._ram_utilization_pct()
        score = self._compute_score(vram_mb, bw, ram_util, backend_hint)

        # Additional warnings
        if is_shared and ram_util > 85.0:
            warnings.append(
                f"System RAM utilisation is high ({ram_util:.0f}%). "
                "iGPU shared memory pool may be constrained."
            )
        if vram_mb < 512:
            warnings.append(
                f"iGPU has only {vram_mb} MB VRAM. Only very small workloads will be offloaded."
            )
        if bw < 10.0:
            warnings.append(
                f"iGPU estimated bandwidth is very low ({bw:.1f} GB/s). "
                "Compute-heavy layers will not be assigned to iGPU."
            )

        return IgpuProfile(
            detected=True,
            enabled=True,
            vendor=vendor,
            model=model,
            vram_mb=vram_mb,
            vram_bytes=vram_bytes,
            shared_memory_bytes=shared_bytes,
            is_shared_memory=is_shared,
            backend_hint=backend_hint,
            vulkan_device_index=vulkan_idx,
            bandwidth_gbps=bw,
            suitability_score=score,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_igpu() -> IgpuProfile:
        return IgpuProfile(
            detected=False, enabled=False,
            vendor="unknown", model="None",
            vram_mb=0, vram_bytes=0,
            shared_memory_bytes=0, is_shared_memory=False,
            backend_hint="cpu", vulkan_device_index=-1,
            bandwidth_gbps=0.0, suitability_score=0.0,
            warnings=["No integrated GPU detected."],
        )

    def _system_bus_bandwidth_gbps(self) -> float:
        """Read system memory bus bandwidth from interconnects or RAM profile."""
        interconnects = self.hw_profile.get("interconnects", [])
        for ic in interconnects:
            if str(ic.get("type", "")).lower() in ("systembus", "system_bus", "unified"):
                bw = float(ic.get("bandwidth", 0.0))
                if bw > 0:
                    return bw
        # Fall back to RAM bandwidth field
        ram = self.hw_profile.get("ram", self.hw_profile.get("memory", {}))
        bw = float(ram.get("bandwidth", 0.0))
        if bw > 0:
            return bw
        return 25.0  # conservative default (DDR4-3200 dual channel)

    def _ram_utilization_pct(self) -> float:
        """Return current RAM utilization as a percentage."""
        ram = self.hw_profile.get("ram", self.hw_profile.get("memory", {}))
        util = float(ram.get("utilization", 0.0))
        if util > 0:
            return util
        total = int(ram.get("total_bytes", 0))
        used_bytes = total - int(ram.get("available_bytes", ram.get("free_bytes", 0)))
        if total > 0:
            return (used_bytes / total) * 100.0
        return 50.0  # neutral default

    @staticmethod
    def _compute_score(
        vram_mb: int,
        bandwidth_gbps: float,
        ram_utilization_pct: float,
        backend_hint: str,
    ) -> float:
        """Compute composite suitability score in [0.0, 1.0]."""
        vram_score   = min(1.0, vram_mb / _VRAM_FULL_SCORE_MB)
        bw_score     = min(1.0, bandwidth_gbps / _BW_FULL_SCORE_GBPS)
        headroom     = 1.0 - min(1.0, ram_utilization_pct / 100.0)
        backend_sc   = _BACKEND_SCORES.get(backend_hint, 0.0)

        score = (
            _W_VRAM    * vram_score
            + _W_BW    * bw_score
            + _W_HEADRM * headroom
            + _W_BACKEND * backend_sc
        )
        return round(min(1.0, max(0.0, score)), 4)


def profile_igpu(hw_profile: Dict[str, Any]) -> IgpuProfile:
    """
    Module-level convenience wrapper around :class:`IgpuProfiler`.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile dict.

    Returns
    -------
    IgpuProfile
        Computed profile for the first detected iGPU, or a disabled profile
        if no iGPU is found.
    """
    return IgpuProfiler(hw_profile).profile()
