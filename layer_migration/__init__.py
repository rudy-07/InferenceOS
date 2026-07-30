"""
InferenceOS Layer Migration — Phase 5 Public API

Exposes the complete dynamic layer migration interface:

    MigrationEngine        — top-level facade (runWithMigration, forceMigrate...)
    MigrationResult        — output: text + stats + events across all restarts
    MigrationCoordinator   — VRAM monitor + hysteresis + plan mutation orchestrator
    MigrationConfig        — thresholds, cooldown, step sizes
    MigrationEvent         — structured runtime event (reason, layers, duration)
    MigrationReason        — enum: PRESSURE_HIGH, PRESSURE_RELIEF, MANUAL, OOM_IMMINENT
    MemoryMonitor          — background VRAM/RAM polling daemon
    MemorySnapshot         — point-in-time memory reading
    HysteresisController   — FSM-based oscillation prevention
    PlanMutator            — fast greedy PlacementPlan mutation

Usage
-----
    from layer_migration import MigrationEngine, MigrationConfig

    cfg = MigrationConfig(pressure_threshold_pct=85.0, cooldown_sec=30.0)
    engine = MigrationEngine(migration_config=cfg)

    result = engine.runWithMigration(
        plan, model_path, "Your prompt",
        on_token=lambda t: print(t, end="", flush=True),
        on_migration_event=lambda e: print(f"\\n{e.message}"),
    )
    print(f"Migrations: {result.total_migrations}")
    print(f"GPU layers: {result.initial_n_gpu_layers} → {result.final_n_gpu_layers}")
"""
from __future__ import annotations

from .hysteresis_controller import HysteresisController, HysteresisState
from .memory_monitor import MemoryMonitor, MemorySnapshot
from .migration_config import MigrationConfig
from .migration_coordinator import MigrationCoordinator
from .migration_engine import MigrationEngine, MigrationResult
from .migration_event import MigrationEvent, MigrationReason
from .plan_mutator import PlanMutator

__all__ = [
    # Facades
    "MigrationEngine",
    # Data types
    "MigrationResult",
    "MigrationEvent",
    "MigrationReason",
    "MemorySnapshot",
    # Configuration
    "MigrationConfig",
    # Components
    "MigrationCoordinator",
    "MemoryMonitor",
    "HysteresisController",
    "HysteresisState",
    "PlanMutator",
]
