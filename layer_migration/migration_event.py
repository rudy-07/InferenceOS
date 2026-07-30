"""
migration_event.py
------------------
Structured runtime events emitted by the Phase 5 Dynamic Layer Migration system.

Every migration action, memory pressure detection, and restoration is described
by a :class:`MigrationEvent`. Events are:

  - Fired synchronously into a ``threading.Queue`` for the coordinator
  - Passed to caller-registered ``on_migration_event`` callbacks immediately
  - Accumulated in ``MigrationCoordinator.migration_history`` for introspection

MigrationReason values
-----------------------
  PRESSURE_HIGH   VRAM crossed pressure_threshold_pct for N consecutive samples.
                  → triggers GPU→CPU layer migration.
  PRESSURE_RELIEF VRAM dropped below relief_threshold_pct after a cooldown.
                  → triggers CPU→GPU layer restoration.
  MANUAL          Caller called forceMigrateDown() / forceMigrateUp() directly.
  OOM_IMMINENT    VRAM > oom_threshold_pct (emergency — bypasses cooldown).
"""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


# ---------------------------------------------------------------------------
# MigrationReason
# ---------------------------------------------------------------------------

class MigrationReason(str, Enum):
    """Reason that triggered a migration event."""
    PRESSURE_HIGH   = "pressure_high"     # VRAM > pressure_threshold_pct
    PRESSURE_RELIEF = "pressure_relief"   # VRAM < relief_threshold_pct
    MANUAL          = "manual"            # Caller forced via API
    OOM_IMMINENT    = "oom_imminent"      # VRAM > oom_threshold_pct (emergency)

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# MigrationEvent
# ---------------------------------------------------------------------------

@dataclass
class MigrationEvent:
    """
    A structured event describing a single layer migration action.

    Attributes
    ----------
    event_id : str
        Unique UUID string for this event.
    timestamp : float
        ``time.perf_counter()`` value at event creation.
    reason : MigrationReason
        What triggered this migration.
    direction : str
        ``"gpu_to_cpu"`` for downward migration, ``"cpu_to_gpu"`` for
        restoration, or ``"none"`` if no actual layer change occurred.
    layers_before : int
        Number of GPU layers active *before* migration.
    layers_after : int
        Number of GPU layers active *after* migration.
    layers_moved : int
        Absolute difference ``|layers_after - layers_before|``.
    vram_usage_pct : float
        VRAM utilization (%) that triggered this event.
    migration_duration_ms : float
        Wall-clock time for the migration to complete in milliseconds.
        Zero until the migration is confirmed finished.
    success : bool
        True if the migration completed successfully.
    model_name : str
        Name of the model being served at the time of migration.
    backend : str
        Backend name (e.g. ``"vulkan"``, ``"cuda"``, ``"cpu"``).
    message : str
        Human-readable description for logging / UI display.
    """
    reason: MigrationReason
    direction: str
    layers_before: int
    layers_after: int
    vram_usage_pct: float
    model_name: str
    backend: str

    # Auto-filled
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.perf_counter)
    layers_moved: int = field(init=False)
    migration_duration_ms: float = 0.0
    success: bool = True
    message: str = ""

    def __post_init__(self) -> None:
        self.layers_moved = abs(self.layers_after - self.layers_before)
        if not self.message:
            self.message = self._build_message()

    def _build_message(self) -> str:
        """Build a human-readable description of this event."""
        direction_label = {
            "gpu_to_cpu": "GPU → CPU",
            "cpu_to_gpu": "CPU → GPU",
            "none": "no-op",
        }.get(self.direction, self.direction)

        reason_label = {
            MigrationReason.PRESSURE_HIGH: f"VRAM pressure ({self.vram_usage_pct:.1f}%)",
            MigrationReason.PRESSURE_RELIEF: f"VRAM relief ({self.vram_usage_pct:.1f}%)",
            MigrationReason.OOM_IMMINENT: f"OOM imminent ({self.vram_usage_pct:.1f}%)",
            MigrationReason.MANUAL: "manual request",
        }.get(self.reason, str(self.reason))

        return (
            f"[Migration] {direction_label}  |  "
            f"layers: {self.layers_before} → {self.layers_after} "
            f"(moved {self.layers_moved})  |  "
            f"reason: {reason_label}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of this event."""
        return {
            "event_id": self.event_id,
            "timestamp": round(self.timestamp, 6),
            "reason": str(self.reason),
            "direction": self.direction,
            "layers_before": self.layers_before,
            "layers_after": self.layers_after,
            "layers_moved": self.layers_moved,
            "vram_usage_pct": round(self.vram_usage_pct, 2),
            "migration_duration_ms": round(self.migration_duration_ms, 2),
            "success": self.success,
            "model_name": self.model_name,
            "backend": self.backend,
            "message": self.message,
        }
