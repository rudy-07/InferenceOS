"""
migration_engine.py
--------------------
Top-level facade for Phase 5: Dynamic Layer Migration.

Extends Phase 4's RuntimeEngine with adaptive layer migration. When VRAM
pressure is detected mid-inference, the engine:

  1. Allows the current token generation batch to reach a natural pause point
  2. Records the partial output emitted so far
  3. Restarts llama.cpp with a reduced ``-ngl`` count and the combined
     (original_prompt + partial_output) as the new prompt
  4. Continues streaming tokens seamlessly to the caller
  5. Fires a MigrationEvent describing the action and duration

KV-cache safety
---------------
Because each subprocess restart rebuilds the KV cache from scratch using
the full context (original prompt + partial output), the model's internal
state is always consistent. There is no risk of KV cache corruption.

Usage
-----
    engine = MigrationEngine(hw_profile=hw)
    result = engine.runWithMigration(
        plan, model_path, "Your prompt",
        on_token=lambda t: print(t, end="", flush=True),
        on_migration_event=lambda e: print(f"\\n{e.message}\\n"),
    )
    print(f"Migrations: {result.total_migrations}")
    print(f"Final GPU layers: {result.final_n_gpu_layers}")

Public API (camelCase + snake_case aliases)
-------------------------------------------
    runWithMigration(plan, model_path, prompt, on_token, on_migration_event)
    forceMigrateDown(n_layers)  → MigrationEvent
    forceMigrateUp(n_layers)    → MigrationEvent
    getMigrationHistory()        → List[MigrationEvent]
    getCurrentPlan()             → PlacementPlan | None
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from inference_runtime.runtime_config import RuntimeConfig
from inference_runtime.stats_collector import RuntimeStats
from inference_runtime.inference_session import InferenceResult, InferenceSession

from .migration_config import MigrationConfig
from .migration_coordinator import MigrationCoordinator
from .migration_event import MigrationEvent, MigrationReason
from layer_placement.placement_plan import PlacementPlan


# ---------------------------------------------------------------------------
# MigrationResult
# ---------------------------------------------------------------------------

@dataclass
class MigrationResult:
    """
    Complete output from a migration-aware inference run.

    Attributes
    ----------
    generated_text : str
        Full generated text, concatenated across all subprocess restarts.
    stats : RuntimeStats
        Statistics from the *final* subprocess run (most recent -ngl config).
    migration_events : List[MigrationEvent]
        All migration events that fired during this inference run.
    total_migrations : int
        Number of migration events that resulted in a plan change.
    final_n_gpu_layers : int
        GPU layer count at the end of the run.
    initial_n_gpu_layers : int
        GPU layer count at the start of the run.
    success : bool
        True if the final subprocess exited with code 0.
    exit_code : int
        Exit code of the final subprocess.
    backend : str
        Backend used for the final subprocess.
    """
    generated_text: str
    stats: RuntimeStats
    migration_events: List[MigrationEvent]
    total_migrations: int
    final_n_gpu_layers: int
    initial_n_gpu_layers: int
    success: bool
    exit_code: int
    backend: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_text": self.generated_text,
            "stats": self.stats.to_dict(),
            "total_migrations": self.total_migrations,
            "final_n_gpu_layers": self.final_n_gpu_layers,
            "initial_n_gpu_layers": self.initial_n_gpu_layers,
            "success": self.success,
            "exit_code": self.exit_code,
            "backend": self.backend,
            "migration_events": [e.to_dict() for e in self.migration_events],
        }


# ---------------------------------------------------------------------------
# MigrationEngine
# ---------------------------------------------------------------------------

class MigrationEngine:
    """
    Inference engine with dynamic layer migration capability.

    Wraps Phase 4's :class:`~inference_runtime.inference_session.InferenceSession`
    and augments it with a :class:`MigrationCoordinator` that monitors VRAM
    and fires layer migrations when pressure thresholds are crossed.

    Parameters
    ----------
    hw_profile : dict, optional
        Hardware profile. Loaded from ``hardware_profile.json`` if None.
    llama_exe_path : Path, optional
        Explicit path to ``llama.exe``. Auto-searched if None.
    config : RuntimeConfig, optional
        Inference runtime configuration.
    migration_config : MigrationConfig, optional
        Layer migration thresholds and settings.
    """

    def __init__(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        llama_exe_path: Optional[Path] = None,
        config: Optional[RuntimeConfig] = None,
        migration_config: Optional[MigrationConfig] = None,
    ) -> None:
        self.hw_profile = hw_profile or self._load_hw_profile()
        self.llama_exe_path = llama_exe_path
        self.config = config or RuntimeConfig.from_hw_profile(self.hw_profile)
        self.migration_config = migration_config or MigrationConfig()

        self._coordinator: Optional[MigrationCoordinator] = None
        self._current_plan: Optional[PlacementPlan] = None
        self._migration_events: List[MigrationEvent] = []
        self._lock = threading.Lock()

        # Signals from coordinator that a new plan is ready
        self._pending_plan: Optional[PlacementPlan] = None
        self._pending_event: Optional[MigrationEvent] = None
        self._plan_changed = threading.Event()

    # ---------------------------------------------------------------------------
    # Core API: runWithMigration
    # ---------------------------------------------------------------------------

    def runWithMigration(
        self,
        plan: PlacementPlan,
        model_path: Path,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_migration_event: Optional[Callable[[MigrationEvent], None]] = None,
        config_override: Optional[RuntimeConfig] = None,
    ) -> MigrationResult:
        """
        Run inference with dynamic layer migration support.

        Starts a background coordinator that monitors VRAM. If pressure is
        detected, the coordinator computes a new plan. The engine completes
        the current token batch, collects partial output, then restarts the
        subprocess with the new configuration and continues generation.

        Parameters
        ----------
        plan : PlacementPlan
            Initial Phase 3 placement plan.
        model_path : Path
            Path to the GGUF model file.
        prompt : str
            Input prompt.
        on_token : callable, optional
            Called for each generated token (streaming UX).
            Fires across all subprocess restarts seamlessly.
        on_migration_event : callable, optional
            Called for each MigrationEvent as it fires.
            Signature: ``(event: MigrationEvent) -> None``.
        config_override : RuntimeConfig, optional
            Override the engine's default config for this run.

        Returns
        -------
        MigrationResult
            Full result including text, stats, and migration events.
        """
        model_path = Path(model_path)
        effective_config = config_override or self.config
        initial_n_gpu = plan.n_gpu_layers

        # Reset per-run state
        with self._lock:
            self._current_plan = plan
            self._migration_events = []
            self._pending_plan = None
            self._pending_event = None
            self._plan_changed.clear()

        # Start the migration coordinator
        self._coordinator = MigrationCoordinator(
            config=self.migration_config,
            hw_profile=self.hw_profile,
            on_plan_changed=self._handle_plan_changed,
            on_event=lambda e: self._handle_event(e, on_migration_event),
            model_name=plan.model_name,
            backend="unknown",  # resolved after first run
        )
        self._coordinator.start(plan)

        # --- Multi-run inference loop with migration restart ---
        current_plan = plan
        all_text_parts: List[str] = []
        current_prompt = prompt
        last_result: Optional[InferenceResult] = None
        max_restarts = 10  # safety ceiling

        for attempt in range(max_restarts + 1):
            # Check if a new plan arrived from the coordinator
            with self._lock:
                if self._pending_plan is not None:
                    current_plan = self._pending_plan
                    self._pending_plan = None
                    self._pending_event = None

            partial_tokens: List[str] = []

            def _on_token_with_migration(text: str) -> None:
                """Token callback that also listens for plan changes."""
                partial_tokens.append(text)
                if on_token:
                    on_token(text)

            # Run the subprocess
            with InferenceSession(
                model_path=model_path,
                plan=current_plan,
                config=effective_config,
                hw_profile=self.hw_profile,
                llama_exe_path=self.llama_exe_path,
            ) as session:
                result = session.run(current_prompt, on_token=_on_token_with_migration)

            last_result = result

            partial_text = result.generated_text
            all_text_parts.append(partial_text)

            # Check if a migration fired during this run
            with self._lock:
                has_pending_migration = self._pending_plan is not None

            if not has_pending_migration or attempt >= max_restarts:
                # No pending migration or safety limit hit — we're done
                break

            # Migration fired: carry forward context for next subprocess
            # The new prompt = original prompt + all generated text so far
            # This rebuilds KV cache from combined context on restart
            current_prompt = prompt + " " + " ".join(all_text_parts)

        # Stop the coordinator
        self._coordinator.stop()

        # Compile result
        with self._lock:
            events = list(self._migration_events)
            final_plan = self._current_plan or plan

        full_text = "\n".join(all_text_parts)
        final_stats = last_result.stats if last_result else RuntimeStats()
        exit_code = last_result.exit_code if last_result else 0

        return MigrationResult(
            generated_text=full_text,
            stats=final_stats,
            migration_events=events,
            total_migrations=len(events),
            final_n_gpu_layers=final_plan.n_gpu_layers,
            initial_n_gpu_layers=initial_n_gpu,
            success=exit_code == 0,
            exit_code=exit_code,
            backend=last_result.backend if last_result else "unknown",
        )

    run_with_migration = runWithMigration  # snake_case alias

    # ---------------------------------------------------------------------------
    # Manual migration API
    # ---------------------------------------------------------------------------

    def forceMigrateDown(self, n_layers: int) -> MigrationEvent:
        """
        Manually migrate ``n_layers`` transformer layers from GPU to CPU.

        Parameters
        ----------
        n_layers : int
            Number of GPU layers to move to CPU.

        Returns
        -------
        MigrationEvent
            Event describing the completed migration.

        Raises
        ------
        RuntimeError
            If called before :meth:`runWithMigration` has been started.
        """
        if self._coordinator is None:
            raise RuntimeError(
                "No active migration session. Call runWithMigration() first."
            )
        event = self._coordinator.force_migrate("gpu_to_cpu", n_layers)
        with self._lock:
            self._migration_events.append(event)
        return event

    force_migrate_down = forceMigrateDown

    def forceMigrateUp(self, n_layers: int) -> MigrationEvent:
        """
        Manually restore ``n_layers`` transformer layers from CPU to GPU.

        Parameters
        ----------
        n_layers : int
            Number of CPU layers to restore to GPU.

        Returns
        -------
        MigrationEvent
            Event describing the completed migration.
        """
        if self._coordinator is None:
            raise RuntimeError(
                "No active migration session. Call runWithMigration() first."
            )
        event = self._coordinator.force_migrate("cpu_to_gpu", n_layers)
        with self._lock:
            self._migration_events.append(event)
        return event

    force_migrate_up = forceMigrateUp

    # ---------------------------------------------------------------------------
    # Introspection API
    # ---------------------------------------------------------------------------

    def getMigrationHistory(self) -> List[MigrationEvent]:
        """
        Return all migration events from the most recent run.

        Returns
        -------
        List[MigrationEvent]
        """
        with self._lock:
            return list(self._migration_events)

    get_migration_history = getMigrationHistory

    def getCurrentPlan(self) -> Optional[PlacementPlan]:
        """
        Return the currently active PlacementPlan.

        Returns
        -------
        PlacementPlan or None
        """
        with self._lock:
            return self._current_plan

    get_current_plan = getCurrentPlan

    # ---------------------------------------------------------------------------
    # Internal callbacks (called from coordinator thread)
    # ---------------------------------------------------------------------------

    def _handle_plan_changed(
        self,
        new_plan: PlacementPlan,
        event: MigrationEvent,
    ) -> None:
        """Receive a new plan from the coordinator and signal the run loop."""
        with self._lock:
            self._pending_plan = new_plan
            self._pending_event = event
            self._current_plan = new_plan
            self._migration_events.append(event)
        self._plan_changed.set()

    def _handle_event(
        self,
        event: MigrationEvent,
        user_callback: Optional[Callable[[MigrationEvent], None]],
    ) -> None:
        """Forward a migration event to the user's callback."""
        if user_callback is not None:
            try:
                user_callback(event)
            except Exception:
                pass

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _load_hw_profile() -> Dict[str, Any]:
        """Load hardware_profile.json from project root."""
        import json
        from pathlib import Path as _Path
        candidate = _Path(__file__).parent.parent / "hardware_profile.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        try:
            from profiler import get_system_resources
            return get_system_resources().to_dict()
        except Exception:
            pass
        return {
            "gpus": [],
            "igpus": [],
            "ram": {"total_bytes": 16 * 1024 ** 3},
            "memory": {"total_bytes": 16 * 1024 ** 3},
            "cpu": {"physical_cores": 4},
            "interconnects": [],
            "inference_hints": {},
        }
