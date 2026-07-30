"""
migration_coordinator.py
------------------------
Orchestrates the full dynamic migration cycle for Phase 5.

The MigrationCoordinator ties together:
  - MemoryMonitor  — continuous VRAM/RAM background sampling
  - HysteresisController — throttled migration decisions
  - PlanMutator    — fast greedy plan mutations

It runs a single coordination loop in a daemon thread that:
  1. Reads the latest MemorySnapshot from the monitor
  2. Checks OOM-emergency condition (bypasses cooldown)
  3. Checks normal pressure/relief conditions via hysteresis controller
  4. Mutates the current plan (immutably) if migration is warranted
  5. Fires a MigrationEvent via the on_event callback (non-blocking)
  6. Notifies the engine via on_plan_changed callback (non-blocking)

Decoupling
----------
The coordinator is intentionally decoupled from subprocess lifecycle.
It only computes new plans and emits events. The MigrationEngine (facade)
decides whether and when to actually restart the llama.cpp subprocess in
response to plan changes.

Thread safety
-------------
``current_plan`` and ``migration_history`` are protected by a ``RLock``.
Callbacks (``on_plan_changed``, ``on_event``) are called from the coordinator
thread. Callers must ensure their callback implementations are thread-safe.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .hysteresis_controller import HysteresisController
from .memory_monitor import MemoryMonitor, MemorySnapshot
from .migration_config import MigrationConfig
from .migration_event import MigrationEvent, MigrationReason
from .plan_mutator import PlanMutator
from layer_placement.placement_plan import PlacementPlan


# Type aliases for callbacks
OnPlanChanged = Callable[[PlacementPlan, MigrationEvent], None]
OnEvent = Callable[[MigrationEvent], None]


class MigrationCoordinator:
    """
    Coordinates memory monitoring with plan mutation and event emission.

    Parameters
    ----------
    config : MigrationConfig
        Migration thresholds, cooldown, and step sizes.
    hw_profile : dict
        Hardware profile for memory monitoring setup.
    on_plan_changed : callable, optional
        Called whenever a migration produces a new PlacementPlan.
        Signature: ``(new_plan: PlacementPlan, event: MigrationEvent) -> None``.
        Called from the coordinator thread; must be thread-safe.
    on_event : callable, optional
        Called for every MigrationEvent (including non-plan-changing ones).
        Signature: ``(event: MigrationEvent) -> None``.
    model_name : str
        Model name annotated into MigrationEvents.
    backend : str
        Backend name annotated into MigrationEvents.
    """

    def __init__(
        self,
        config: MigrationConfig,
        hw_profile: Dict[str, Any],
        on_plan_changed: Optional[OnPlanChanged] = None,
        on_event: Optional[OnEvent] = None,
        model_name: str = "",
        backend: str = "unknown",
    ) -> None:
        self._cfg = config
        self._on_plan_changed = on_plan_changed
        self._on_event = on_event
        self._model_name = model_name
        self._backend = backend

        self._lock = threading.RLock()
        self._current_plan: Optional[PlacementPlan] = None
        self._migration_history: List[MigrationEvent] = []
        self._active = False
        self._thread: Optional[threading.Thread] = None

        # Sub-components
        self._monitor = MemoryMonitor(
            hw_profile=hw_profile,
            interval_ms=config.monitor_interval_ms,
        )
        self._hysteresis = HysteresisController(config)
        self._mutator = PlanMutator(
            min_gpu_layers=config.min_gpu_layers,
        )

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def start(self, initial_plan: PlacementPlan) -> None:
        """
        Start the coordinator with an initial placement plan.

        Starts the memory monitor and the coordination loop thread.

        Parameters
        ----------
        initial_plan : PlacementPlan
            The Phase 3 plan that is currently active.
        """
        with self._lock:
            self._current_plan = initial_plan
            self._active = True

        self._monitor.start()

        self._thread = threading.Thread(
            target=self._coordination_loop,
            daemon=True,
            name="InferenceOSMigrationCoordinator",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the coordinator and memory monitor."""
        self._active = False
        self._monitor.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    # ---------------------------------------------------------------------------
    # Manual migration API
    # ---------------------------------------------------------------------------

    def force_migrate(
        self,
        direction: str,
        n_layers: int,
    ) -> MigrationEvent:
        """
        Manually trigger a migration, bypassing threshold/cooldown checks.

        Parameters
        ----------
        direction : str
            ``"gpu_to_cpu"`` or ``"cpu_to_gpu"``.
        n_layers : int
            Number of layers to move.

        Returns
        -------
        MigrationEvent
            The event describing what was done.
        """
        with self._lock:
            if self._current_plan is None:
                raise RuntimeError("Coordinator not started. Call start() first.")

            plan_before = self._current_plan
            snapshot = self._monitor.get_latest() or self._monitor.capture_once()
            vram_pct = snapshot.vram_used_pct if snapshot else 0.0

            t_start = time.perf_counter()
            if direction == "gpu_to_cpu":
                new_plan = self._mutator.migrate_layers_to_cpu(plan_before, n_layers)
            else:
                new_plan = self._mutator.restore_layers_to_gpu(
                    plan_before, n_layers,
                    vram_available_bytes=snapshot.vram_free_bytes if snapshot else 0,
                )
            duration_ms = (time.perf_counter() - t_start) * 1000.0

            event = MigrationEvent(
                reason=MigrationReason.MANUAL,
                direction=direction,
                layers_before=plan_before.n_gpu_layers,
                layers_after=new_plan.n_gpu_layers,
                vram_usage_pct=vram_pct,
                model_name=self._model_name,
                backend=self._backend,
                migration_duration_ms=duration_ms,
            )

            self._current_plan = new_plan
            self._migration_history.append(event)
            self._hysteresis.force_reset()

        # Fire callbacks outside lock
        self._fire_event(event)
        self._fire_plan_changed(new_plan, event)
        return event

    # ---------------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------------

    @property
    def current_plan(self) -> Optional[PlacementPlan]:
        """The most recently active PlacementPlan."""
        with self._lock:
            return self._current_plan

    @property
    def migration_history(self) -> List[MigrationEvent]:
        """Read-only copy of all migration events so far."""
        with self._lock:
            return list(self._migration_history)

    @property
    def hysteresis_state(self) -> str:
        """Current state of the hysteresis FSM."""
        return self._hysteresis.state

    # ---------------------------------------------------------------------------
    # Coordination loop
    # ---------------------------------------------------------------------------

    def _coordination_loop(self) -> None:
        """Main loop running in daemon thread."""
        interval_sec = self._cfg.monitor_interval_ms / 1000.0

        while self._active:
            try:
                snapshot = self._monitor.get_latest()
                if snapshot is not None:
                    self._evaluate_snapshot(snapshot)
            except Exception:
                pass
            time.sleep(interval_sec)

    def _evaluate_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Evaluate a memory snapshot and fire migration if warranted."""
        with self._lock:
            if self._current_plan is None:
                return

            plan = self._current_plan

            # Priority 1: OOM-imminent (bypasses cooldown)
            if self._hysteresis.should_migrate_down_emergency(snapshot):
                n = self._cfg.layers_to_move_on_pressure * 2  # aggressive
                self._execute_migration("gpu_to_cpu", n, snapshot, MigrationReason.OOM_IMMINENT)
                return

            # Priority 2: Pressure → GPU-to-CPU
            if self._hysteresis.should_migrate_down(snapshot):
                n = self._cfg.layers_to_move_on_pressure
                self._execute_migration("gpu_to_cpu", n, snapshot, MigrationReason.PRESSURE_HIGH)
                return

            # Priority 3: Relief → CPU-to-GPU restoration
            if self._hysteresis.should_migrate_up(snapshot):
                n = self._cfg.layers_to_restore_on_relief
                self._execute_migration("cpu_to_gpu", n, snapshot, MigrationReason.PRESSURE_RELIEF)
                return

    def _execute_migration(
        self,
        direction: str,
        n_layers: int,
        snapshot: MemorySnapshot,
        reason: MigrationReason,
    ) -> None:
        """
        Execute a migration, record it, and fire callbacks.
        Must be called with ``self._lock`` held.
        """
        plan_before = self._current_plan
        if plan_before is None:
            return

        self._hysteresis.record_migration_started(direction)

        t_start = time.perf_counter()
        if direction == "gpu_to_cpu":
            new_plan = self._mutator.migrate_layers_to_cpu(plan_before, n_layers)
        else:
            new_plan = self._mutator.restore_layers_to_gpu(
                plan_before, n_layers,
                vram_available_bytes=snapshot.vram_free_bytes,
            )
        duration_ms = (time.perf_counter() - t_start) * 1000.0

        event = MigrationEvent(
            reason=reason,
            direction=direction,
            layers_before=plan_before.n_gpu_layers,
            layers_after=new_plan.n_gpu_layers,
            vram_usage_pct=snapshot.vram_used_pct,
            model_name=self._model_name,
            backend=self._backend,
            migration_duration_ms=duration_ms,
        )

        self._hysteresis.record_migration_completed()
        self._current_plan = new_plan
        self._migration_history.append(event)

        # Fire callbacks (outside lock section ends here — but we're inside lock;
        # callbacks must be non-blocking to avoid deadlock)
        self._fire_event(event)
        self._fire_plan_changed(new_plan, event)

    # ---------------------------------------------------------------------------
    # Callback helpers
    # ---------------------------------------------------------------------------

    def _fire_event(self, event: MigrationEvent) -> None:
        """Call on_event callback safely (exceptions are swallowed)."""
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass

    def _fire_plan_changed(self, plan: PlacementPlan, event: MigrationEvent) -> None:
        """Call on_plan_changed callback safely."""
        if self._on_plan_changed is not None:
            try:
                self._on_plan_changed(plan, event)
            except Exception:
                pass
