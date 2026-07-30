"""
contention_model.py
-------------------
Shared memory bus contention model for Phase 7 Integrated GPU Support.

AMD and Intel iGPUs use Unified Memory Architecture (UMA): they carve a
portion of system RAM as "VRAM" and access it over the same memory bus
that the CPU uses for normal system RAM accesses.

When the CPU is actively performing inference (loading tensors, computing
CPU-placed layers), the system memory bus is already under load. Placing
additional work on the iGPU at the same time creates *bus contention*:
both devices compete for the same bandwidth, which can be worse than
running everything on the CPU alone.

Contention model
----------------
  contention_factor = clamp(ram_utilization_pct / 100, 0, _MAX_CONTENTION)
  effective_igpu_bw = system_bus_bw × (1 - contention_factor)

Contention risk classification:
  effective_bw / system_bus_bw   →   risk
  ────────────────────────────────────────
  ≥ 0.70                         →   LOW      (> 70% BW available)
  ≥ 0.40                         →   MEDIUM   (40–70% BW available)
  < 0.40                         →   HIGH     (< 40% BW available)

For Apple Silicon (Metal, unified memory), contention is handled differently:
the memory subsystem is architecturally designed for CPU+GPU co-access.
We apply a reduced contention factor of 0.15 (fixed) for Apple.

Non-shared-memory iGPUs (NVIDIA MX) and discrete GPUs are not affected by
this contention model. They are modelled with contention_factor=0.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .igpu_profiler import IgpuProfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CONTENTION = 0.80    # Never model > 80% contention (OS always gets some BW)
_LOW_RISK_THRESHOLD  = 0.70   # effective / total ≥ this → LOW risk
_HIGH_RISK_THRESHOLD = 0.40   # effective / total < this → HIGH risk
_APPLE_CONTENTION    = 0.15   # Fixed contention for Apple unified memory


# ---------------------------------------------------------------------------
# ContentionEstimate
# ---------------------------------------------------------------------------

@dataclass
class ContentionEstimate:
    """
    Result of a contention analysis for an iGPU on the shared memory bus.

    Attributes
    ----------
    system_bus_bandwidth_gbps : float
        Raw system memory bus bandwidth from the hardware profile.
    ram_utilization_pct : float
        Current system RAM utilisation percentage (0–100).
    contention_factor : float
        Fraction of bus bandwidth lost to CPU/RAM contention (0.0–0.80).
    effective_igpu_bandwidth_gbps : float
        ``system_bus_bandwidth × (1 − contention_factor)``. This is what
        the iGPU can realistically use for its tensor accesses.
    contention_risk : str
        One of ``"LOW"``, ``"MEDIUM"``, or ``"HIGH"``.
    is_shared_bus : bool
        True if the iGPU shares the memory bus (AMD/Intel UMA).
        False for Apple unified memory and NVIDIA MX.
    """
    system_bus_bandwidth_gbps: float
    ram_utilization_pct: float
    contention_factor: float
    effective_igpu_bandwidth_gbps: float
    contention_risk: str
    is_shared_bus: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_bus_bandwidth_gbps": round(self.system_bus_bandwidth_gbps, 2),
            "ram_utilization_pct": round(self.ram_utilization_pct, 1),
            "contention_factor": round(self.contention_factor, 4),
            "effective_igpu_bandwidth_gbps": round(self.effective_igpu_bandwidth_gbps, 2),
            "contention_risk": self.contention_risk,
            "is_shared_bus": self.is_shared_bus,
        }


# ---------------------------------------------------------------------------
# ContentionModel
# ---------------------------------------------------------------------------

class ContentionModel:
    """
    Estimates shared memory bus contention for iGPU access.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile from Phase 1 profiler.
    """

    def __init__(self, hw_profile: Dict[str, Any]) -> None:
        self.hw_profile = hw_profile

    def estimate(self, igpu_profile: IgpuProfile) -> ContentionEstimate:
        """
        Compute a :class:`ContentionEstimate` for the given iGPU profile.

        Parameters
        ----------
        igpu_profile : IgpuProfile
            Pre-computed iGPU profile.

        Returns
        -------
        ContentionEstimate
            Contention analysis result.
        """
        sys_bw = self._system_bus_bandwidth_gbps()
        ram_util = self._ram_utilization_pct()
        is_shared = igpu_profile.is_shared_memory

        if not is_shared:
            # Non-shared bus: no contention (NVIDIA MX, Apple unified memory model)
            contention = _APPLE_CONTENTION if igpu_profile.vendor == "apple" else 0.0
        else:
            # UMA: contention scales with RAM utilisation
            contention = min(_MAX_CONTENTION, ram_util / 100.0)

        effective_bw = sys_bw * (1.0 - contention)

        # Risk classification based on effective/total ratio
        ratio = effective_bw / max(sys_bw, 1e-6)
        if ratio >= _LOW_RISK_THRESHOLD:
            risk = "LOW"
        elif ratio >= _HIGH_RISK_THRESHOLD:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return ContentionEstimate(
            system_bus_bandwidth_gbps=sys_bw,
            ram_utilization_pct=ram_util,
            contention_factor=round(contention, 4),
            effective_igpu_bandwidth_gbps=round(effective_bw, 2),
            contention_risk=risk,
            is_shared_bus=is_shared,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _system_bus_bandwidth_gbps(self) -> float:
        """Read system memory bus bandwidth."""
        interconnects = self.hw_profile.get("interconnects", [])
        for ic in interconnects:
            if str(ic.get("type", "")).lower() in ("systembus", "system_bus", "unified"):
                bw = float(ic.get("bandwidth", 0.0))
                if bw > 0:
                    return bw
        ram = self.hw_profile.get("ram", self.hw_profile.get("memory", {}))
        bw = float(ram.get("bandwidth", 0.0))
        return bw if bw > 0 else 25.0  # DDR4-3200 dual-channel default

    def _ram_utilization_pct(self) -> float:
        """Current RAM utilization percentage."""
        ram = self.hw_profile.get("ram", self.hw_profile.get("memory", {}))
        util = float(ram.get("utilization", 0.0))
        if util > 0:
            return util
        total = int(ram.get("total_bytes", 0))
        avail = int(ram.get("available_bytes", ram.get("free_bytes", 0)))
        if total > 0:
            return ((total - avail) / total) * 100.0
        return 50.0


def estimate_contention(
    hw_profile: Dict[str, Any],
    igpu_profile: IgpuProfile,
) -> ContentionEstimate:
    """
    Module-level convenience wrapper around :class:`ContentionModel`.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile.
    igpu_profile : IgpuProfile
        Pre-computed iGPU profile.

    Returns
    -------
    ContentionEstimate
    """
    return ContentionModel(hw_profile).estimate(igpu_profile)
