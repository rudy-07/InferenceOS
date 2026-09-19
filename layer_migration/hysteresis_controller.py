"""
hysteresis_controller.py
-------------------------
State machine that decides whether a layer migration should fire, preventing
oscillation through dual thresholds and a mandatory cooldown timer.

State machine
-------------

         ┌──────────────────────────────────────────┐
         │                                          │
         ▼         N samples > pressure_threshold   │
       STABLE ─────────────────────────────────────► PRESSURE
         ▲                                          │
         │ cooldown elapsed                         │ fire_migration_down()
         │                                          ▼
      COOLDOWN ◄─────────────────────────── MIGRATING_DOWN
         │
         │ VRAM < relief_threshold AND enabled
         ▼
       RELIEF ──────────────────────────────────────► MIGRATING_UP
         ▲                                          │
         │ cooldown elapsed                         │ fire_migration_up()
         └──────────────────────────────────────────┘

OOM bypass
----------
When VRAM > oom_threshold_pct, the state machine is bypassed entirely:
``should_migrate_down_emergency()`` returns True regardless of current
state or cooldown. This guarantees that OOM conditions always trigger
an immediate emergency migration.

Hysteresis dead-band
--------------------
The gap between relief_threshold_pct (60%) and pressure_threshold_pct (85%)
acts as a dead-band. VRAM in this range does not advance any counters or
trigger any transitions, so steady-state operation near 70-75% is silent.

Thread safety
-------------
All state is protected by an internal ``threading.RLock``. The controller
is safe to call from the coordinator's background polling thread.
"""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Optional

from .memory_monitor import MemorySnapshot
from .migration_config import MigrationConfig


# ---------------------------------------------------------------------------
# HysteresisState
# ---------------------------------------------------------------------------

class HysteresisState(str, Enum):
    """Internal FSM state of the HysteresisController."""
    STABLE          = "STABLE"
    PRESSURE        = "PRESSURE"        # accumulating consecutive samples
    MIGRATING_DOWN  = "MIGRATING_DOWN"  # migration in progress
    COOLDOWN        = "COOLDOWN"        # waiting for cooldown to expire
    RELIEF          = "RELIEF"          # VRAM low enough to restore
    MIGRATING_UP    = "MIGRATING_UP"    # restoration in progress

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# HysteresisController
# ---------------------------------------------------------------------------

class HysteresisController:
    """
    Stateful hysteresis controller for VRAM-based layer migration decisions.

    Parameters
    ----------
    config : MigrationConfig
        Migration thresholds and cooldown settings.
    """

    def __init__(self, config: MigrationConfig) -> None:
        self._cfg = config
        self._lock = threading.RLock()

        # FSM state
        self._state: HysteresisState = HysteresisState.STABLE
        self._consecutive_pressure: int = 0
        self._last_migration_time: float = 0.0

    # ---------------------------------------------------------------------------
    # Primary decision methods (called from coordinator polling loop)
    # ---------------------------------------------------------------------------

    def should_migrate_down(self, snapshot: MemorySnapshot) -> bool:
        """
        Evaluate whether a downward (GPU→CPU) migration should fire.

        Increments the consecutive-pressure counter on high VRAM. Returns
        True only when the counter reaches ``config.consecutive_pressure_samples``
        *and* the cooldown has expired.

        Parameters
        ----------
        snapshot : MemorySnapshot
            Most recent memory reading.

        Returns
        -------
        bool
            True if a downward migration should be triggered now.
        """
        if not self._cfg.enable_migration:
            return False

        with self._lock:
            pct = snapshot.vram_used_pct

            # Dead-band: between relief and pressure thresholds → reset counter
            if pct <= self._cfg.pressure_threshold_pct:
                if pct < self._cfg.relief_threshold_pct:
                    # Relief zone — don't count as pressure
                    self._consecutive_pressure = 0
                elif self._state == HysteresisState.PRESSURE:
                    # Back in dead-band — reset
                    self._consecutive_pressure = 0
                    self._state = HysteresisState.STABLE
                return False

            # Above pressure threshold
            if self._state in (HysteresisState.MIGRATING_DOWN, HysteresisState.MIGRATING_UP):
                return False  # migration already in progress

            if self._is_in_cooldown():
                # Still cooling down; do NOT increment counter
                return False

            self._consecutive_pressure += 1
            if self._state != HysteresisState.PRESSURE:
                self._state = HysteresisState.PRESSURE

            if self._consecutive_pressure >= self._cfg.consecutive_pressure_samples:
                return True

            return False

    def should_migrate_down_emergency(self, snapshot: MemorySnapshot) -> bool:
        """
        Check for OOM-imminent condition (bypasses cooldown and counter).

        Returns True if VRAM > ``config.oom_threshold_pct`` and OOM
        protection is enabled, regardless of cooldown state.
        """
        if not self._cfg.enable_oom_protection:
            return False
        return snapshot.vram_used_pct >= self._cfg.oom_threshold_pct

    def should_migrate_up(self, snapshot: MemorySnapshot) -> bool:
        """
        Evaluate whether upward (CPU→GPU) restoration should fire.

        Returns True only when:
          - Restoration is enabled
          - VRAM is below ``relief_threshold_pct``
          - The cooldown timer has expired
          - Not currently mid-migration

        Parameters
        ----------
        snapshot : MemorySnapshot
            Most recent memory reading.

        Returns
        -------
        bool
            True if a restoration migration should be triggered now.
        """
        if not self._cfg.enable_restoration:
            return False
        if not self._cfg.enable_migration:
            return False

        with self._lock:
            if self._state in (HysteresisState.MIGRATING_DOWN, HysteresisState.MIGRATING_UP):
                return False

            if snapshot.vram_used_pct >= self._cfg.relief_threshold_pct:
                return False  # Still in dead-band or above

            if self._is_in_cooldown():
                return False

            # Relief only applies if a downward migration previously occurred or system was under pressure
            if self._last_migration_time == 0.0 and self._state == HysteresisState.STABLE:
                return False

            return True

    # ---------------------------------------------------------------------------
    # State update methods (called by coordinator after migration fires)
    # ---------------------------------------------------------------------------

    def record_migration_started(self, direction: str) -> None:
        """
        Notify the controller that a migration has been initiated.

        Parameters
        ----------
        direction : str
            ``"gpu_to_cpu"`` or ``"cpu_to_gpu"``.
        """
        with self._lock:
            if direction == "gpu_to_cpu":
                self._state = HysteresisState.MIGRATING_DOWN
            else:
                self._state = HysteresisState.MIGRATING_UP
            self._consecutive_pressure = 0

    def record_migration_completed(self) -> None:
        """Notify the controller that a migration has finished. Starts cooldown."""
        with self._lock:
            self._last_migration_time = time.perf_counter()
            self._state = HysteresisState.COOLDOWN
            self._consecutive_pressure = 0

    def force_reset(self) -> None:
        """Reset state to STABLE (used after manual forceMigrate calls)."""
        with self._lock:
            self._state = HysteresisState.STABLE
            self._consecutive_pressure = 0

    # ---------------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current FSM state name."""
        with self._lock:
            # Auto-transition out of COOLDOWN if timer expired
            if self._state == HysteresisState.COOLDOWN and not self._is_in_cooldown():
                self._state = HysteresisState.STABLE
            return str(self._state)

    def is_in_cooldown(self) -> bool:
        """
        Return True if the post-migration cooldown period is still active.
        Thread-safe public wrapper.
        """
        with self._lock:
            return self._is_in_cooldown()

    def cooldown_remaining_sec(self) -> float:
        """Seconds remaining in cooldown period (0.0 if not in cooldown)."""
        with self._lock:
            elapsed = time.perf_counter() - self._last_migration_time
            remaining = self._cfg.cooldown_sec - elapsed
            return max(0.0, remaining)

    def consecutive_pressure_count(self) -> int:
        """Current count of consecutive above-threshold pressure samples."""
        with self._lock:
            return self._consecutive_pressure

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _is_in_cooldown(self) -> bool:
        """Internal (non-locking) cooldown check. Must be called with lock held."""
        if self._last_migration_time == 0.0:
            return False
        elapsed = time.perf_counter() - self._last_migration_time
        in_cd = elapsed < self._cfg.cooldown_sec
        if not in_cd and self._state == HysteresisState.COOLDOWN:
            self._state = HysteresisState.STABLE
        return in_cd
