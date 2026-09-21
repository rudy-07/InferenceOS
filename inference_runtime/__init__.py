"""
InferenceOS Execution Runtime — Phase 4 Public API

Exposes the complete execution runtime interface:

    RuntimeEngine         — top-level facade (executePlan, benchmark, stats)
    InferenceSession      — single-run session with streaming support
    InferenceResult       — output: generated text + RuntimeStats
    BenchmarkResult       — multi-run aggregate statistics
    RuntimeStats          — per-run timing, utilization, latency breakdown
    RuntimeConfig         — all tunable execution parameters
    BackendInfo           — resolved backend (cuda/vulkan/metal/cpu)
    BackendSelector       — hardware profile → backend detection
    ArgumentBuilder       — PlacementPlan + config → CLI args
    StatsCollector        — OS sampling + stderr parsing

Usage
-----
    from inference_runtime import RuntimeEngine, RuntimeConfig
    from layer_placement import PlacementEngine

    engine = RuntimeEngine()
    result = engine.executePlan(plan, model_path, "Your prompt here")
    print(result.generated_text)
    print(result.stats.eval_tps, "tok/s")
"""
from __future__ import annotations

from .argument_builder import ArgumentBuilder
from .backend_selector import BackendInfo, BackendSelector, detect_backend
from .inference_session import InferenceResult, InferenceSession
from .multiformat_engine import MultiFormatRuntimeEngine
from .process_manager import ProcessManager
from .runtime_config import RuntimeConfig
from .runtime_engine import BenchmarkResult, RuntimeEngine
from .stats_collector import RuntimeStats, StatsCollector

__all__ = [
    # Facades
    "RuntimeEngine",
    "MultiFormatRuntimeEngine",
    "InferenceSession",
    # Data types
    "InferenceResult",
    "BenchmarkResult",
    "RuntimeStats",
    # Configuration
    "RuntimeConfig",
    "BackendInfo",
    # Components
    "BackendSelector",
    "ArgumentBuilder",
    "ProcessManager",
    "StatsCollector",
    # Helpers
    "detect_backend",
    # Phase 8 — import from async_scheduler directly:
    # from async_scheduler import AsyncRuntimeEngine, SchedulerConfig, ...
]
