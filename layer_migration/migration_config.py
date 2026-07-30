"""
migration_config.py
-------------------
Configuration dataclass for the Phase 5 Dynamic Layer Migration system.

All thresholds, cooldown timings, and step sizes are consolidated here
so callers have a single place to tune migration behavior.

Design: hysteresis via dual thresholds
---------------------------------------
The dead-band between ``relief_threshold_pct`` (60%) and
``pressure_threshold_pct`` (85%) ensures that VRAM utilization floating
in the 60–85% range never triggers any migration — preventing oscillation.

  VRAM %    Action
  ──────    ──────
  > 95%     OOM-imminent → immediate emergency GPU→CPU migration
  > 85%     Pressure → GPU→CPU migration (after N consecutive samples)
  60–85%    Dead-band → no action
  < 60%     Relief → CPU→GPU restoration (after cooldown elapsed)

Cooldown timer
--------------
After *any* migration (up or down), a cooldown of ``cooldown_sec`` seconds
must elapse before the next migration is permitted. This is the second layer
of oscillation protection (the dead-band being the first).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MigrationConfig:
    """
    Complete configuration for the layer migration subsystem.

    Thresholds
    ----------
    pressure_threshold_pct : float
        VRAM utilization percentage above which downward migration is
        triggered. Default 85.0%.
    relief_threshold_pct : float
        VRAM utilization percentage below which upward restoration is
        allowed. Must be strictly less than ``pressure_threshold_pct``.
        Default 60.0%.
    oom_threshold_pct : float
        Emergency threshold. Above this, cooldown is bypassed and an
        immediate GPU→CPU migration fires. Default 95.0%.

    Hysteresis / Cooldown
    ---------------------
    cooldown_sec : float
        Minimum seconds that must elapse between any two migrations
        (in either direction). Prevents rapid oscillation. Default 30.0.
    consecutive_pressure_samples : int
        Number of consecutive above-threshold VRAM samples required
        before a downward migration fires. Filters transient spikes.
        Default 3.

    Step Sizes
    ----------
    layers_to_move_on_pressure : int
        Number of GPU transformer layers to move to CPU per pressure event.
        Default 4.
    layers_to_restore_on_relief : int
        Number of CPU layers to restore to GPU per relief event. Smaller
        than ``layers_to_move_on_pressure`` to favour conservative recovery.
        Default 2.
    min_gpu_layers : int
        Floor on GPU layer count. Migration will never reduce GPU layers
        below this value. Default 0 (full CPU fallback allowed).
    max_gpu_layers : int
        Ceiling on GPU layer count. Restoration will not exceed this.
        -1 = no ceiling (use plan total). Default -1.

    Polling
    -------
    monitor_interval_ms : float
        Interval between VRAM/RAM polls in milliseconds. Default 500.0.

    Feature Flags
    -------------
    enable_migration : bool
        Master switch. If False, no migrations fire regardless of pressure.
        Default True.
    enable_restoration : bool
        Allow restoring layers from CPU back to GPU on relief. If False,
        migrations are one-way (GPU→CPU only). Default True.
    enable_oom_protection : bool
        Allow emergency migration at ``oom_threshold_pct``. Default True.
    """

    # Thresholds
    pressure_threshold_pct: float = 85.0
    relief_threshold_pct: float = 60.0
    oom_threshold_pct: float = 95.0

    # Hysteresis
    cooldown_sec: float = 30.0
    consecutive_pressure_samples: int = 3

    # Step sizes
    layers_to_move_on_pressure: int = 4
    layers_to_restore_on_relief: int = 2
    min_gpu_layers: int = 0
    max_gpu_layers: int = -1

    # Polling
    monitor_interval_ms: float = 500.0

    # Feature flags
    enable_migration: bool = True
    enable_restoration: bool = True
    enable_oom_protection: bool = True

    def __post_init__(self) -> None:
        # Validate threshold ordering
        if self.relief_threshold_pct >= self.pressure_threshold_pct:
            raise ValueError(
                f"relief_threshold_pct ({self.relief_threshold_pct}) must be "
                f"strictly less than pressure_threshold_pct ({self.pressure_threshold_pct})."
            )
        if self.oom_threshold_pct <= self.pressure_threshold_pct:
            raise ValueError(
                f"oom_threshold_pct ({self.oom_threshold_pct}) must be "
                f"strictly greater than pressure_threshold_pct ({self.pressure_threshold_pct})."
            )
        if self.cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative.")
        if self.consecutive_pressure_samples < 1:
            raise ValueError("consecutive_pressure_samples must be >= 1.")
        if self.layers_to_move_on_pressure < 1:
            raise ValueError("layers_to_move_on_pressure must be >= 1.")
        if self.layers_to_restore_on_relief < 1:
            raise ValueError("layers_to_restore_on_relief must be >= 1.")
        if self.monitor_interval_ms < 50.0:
            raise ValueError("monitor_interval_ms must be >= 50 ms.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of all config values."""
        return {
            "pressure_threshold_pct": self.pressure_threshold_pct,
            "relief_threshold_pct": self.relief_threshold_pct,
            "oom_threshold_pct": self.oom_threshold_pct,
            "cooldown_sec": self.cooldown_sec,
            "consecutive_pressure_samples": self.consecutive_pressure_samples,
            "layers_to_move_on_pressure": self.layers_to_move_on_pressure,
            "layers_to_restore_on_relief": self.layers_to_restore_on_relief,
            "min_gpu_layers": self.min_gpu_layers,
            "max_gpu_layers": self.max_gpu_layers,
            "monitor_interval_ms": self.monitor_interval_ms,
            "enable_migration": self.enable_migration,
            "enable_restoration": self.enable_restoration,
            "enable_oom_protection": self.enable_oom_protection,
        }

    @classmethod
    def aggressive(cls) -> "MigrationConfig":
        """
        Preset for memory-constrained systems.
        Low threshold, small step sizes, short cooldown.
        """
        return cls(
            pressure_threshold_pct=75.0,
            relief_threshold_pct=50.0,
            cooldown_sec=10.0,
            layers_to_move_on_pressure=2,
            layers_to_restore_on_relief=1,
            consecutive_pressure_samples=2,
        )

    @classmethod
    def conservative(cls) -> "MigrationConfig":
        """
        Preset for systems with ample VRAM.
        High threshold, large step sizes, long cooldown.
        """
        return cls(
            pressure_threshold_pct=92.0,
            relief_threshold_pct=70.0,
            cooldown_sec=60.0,
            layers_to_move_on_pressure=8,
            layers_to_restore_on_relief=4,
            consecutive_pressure_samples=5,
        )
