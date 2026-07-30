"""
InferenceOS Async Execution Scheduler — Phase 8 Public API

Exposes the complete asynchronous scheduling interface:

    AsyncRuntimeEngine    — Phase 8 facade (submit, submitBatch,
                            executePlan, getSchedulerMetrics,
                            getBenchmarkComparison)
    PipelineCoordinator   — 3-stage Prepare→Transfer→Compute pipeline
    SchedulerConfig       — all tunable scheduler parameters
    SchedulerMetrics      — aggregate performance metrics
    SchedulerReport       — human-readable report generator
    WorkerPool            — managed thread pool
    TaskQueue             — priority-aware bounded queue
    TaskFuture            — awaitable result handle
    Priority              — task priority enum (CRITICAL/HIGH/NORMAL/LOW)
    PrefetchManager       — model segment prefetch manager
    CudaStreamManager     — CUDA/Vulkan env-var stream manager
    MetricsAccumulator    — per-request timing accumulator
    StageTimings          — per-request stage timing record

Quick start
-----------
    from async_scheduler import AsyncRuntimeEngine, SchedulerConfig
    from layer_placement import PlacementEngine

    engine = AsyncRuntimeEngine(hw_profile=hw)
    engine.start()

    # Non-blocking single request:
    future = engine.submit(plan, model_path, "Explain quantum entanglement.")
    result = future.result(timeout=120)
    print(result.generated_text)

    # Batch (concurrent):
    futures = engine.submitBatch([
        {"plan": plan, "model_path": model_path, "prompt": "Question 1"},
        {"plan": plan, "model_path": model_path, "prompt": "Question 2"},
    ])
    results = [f.result() for f in futures]

    # Benchmark comparison:
    comparison = engine.getBenchmarkComparison(plan, model_path, n_runs=3)
    print(comparison["report"])

    engine.stop()
"""
from __future__ import annotations

from .async_runtime_engine import AsyncRuntimeEngine
from .cuda_stream_manager import CudaStreamManager, StreamStats, VirtualStream
from .pipeline_coordinator import PipelineCoordinator
from .prefetch_manager import PinnedBuffer, PrefetchManager, PrefetchResult
from .scheduler_config import SchedulerConfig
from .scheduler_metrics import (
    MetricsAccumulator,
    SchedulerMetrics,
    SchedulerReport,
    StageTimings,
)
from .task_queue import (
    Priority,
    QueueFullError,
    TaskCancelledError,
    TaskFailedError,
    TaskFuture,
    TaskQueue,
    TaskStats,
)
from .worker_pool import PoolStats, WorkerPool

__all__ = [
    # Primary facades
    "AsyncRuntimeEngine",
    "PipelineCoordinator",
    # Configuration
    "SchedulerConfig",
    # Metrics
    "SchedulerMetrics",
    "SchedulerReport",
    "MetricsAccumulator",
    "StageTimings",
    # Task queue primitives
    "TaskQueue",
    "TaskFuture",
    "TaskStats",
    "Priority",
    # Worker pool
    "WorkerPool",
    "PoolStats",
    # Prefetch
    "PrefetchManager",
    "PrefetchResult",
    "PinnedBuffer",
    # CUDA / Vulkan streams
    "CudaStreamManager",
    "StreamStats",
    "VirtualStream",
    # Exceptions
    "QueueFullError",
    "TaskCancelledError",
    "TaskFailedError",
]
