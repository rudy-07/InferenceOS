"""
async_runtime_engine.py
------------------------
Phase 8 AsyncRuntimeEngine — async-scheduled extension of Phase 4 RuntimeEngine.

This is the primary public facade for Phase 8. It wraps the Phase 4
RuntimeEngine and adds:

  1. **Non-blocking submit()**: fire-and-forget inference request that
     returns a TaskFuture immediately.

  2. **Batch submission (submitBatch())**: schedule N requests concurrently,
     up to max_concurrent_requests limit enforced by the PipelineCoordinator.

  3. **3-stage pipeline**: each request passes through Prepare → Transfer →
     Compute via the PipelineCoordinator, hiding PCIe latency behind CPU prep.

  4. **CUDA env-var hints**: CudaStreamManager injects async kernel scheduling
     flags into the llama.cpp subprocess environment.

  5. **Prefetch**: PrefetchManager pre-reads boundary GGUF segments into
     pinned buffers before GPU needs them.

  6. **Scheduler metrics**: getSchedulerMetrics() returns aggregate stats
     including gpu_idle_pct, pcie_latency_hidden_ms, bubble_rate_pct.

  7. **Benchmark comparison**: getBenchmarkComparison() runs both synchronous
     (Phase 4) and async (Phase 8) paths and reports the improvement.

Backward compatibility
----------------------
AsyncRuntimeEngine is a drop-in extension of RuntimeEngine. Callers that
only use executePlan() / benchmark() see no behaviour change if
enable_async_scheduler=False in the RuntimeConfig (default).

When enable_async_scheduler=True (set in RuntimeConfig or via constructor),
executePlan() is routed through the pipeline coordinator.

Example
-------
    from async_scheduler import AsyncRuntimeEngine
    from inference_runtime import RuntimeConfig

    engine = AsyncRuntimeEngine(hw_profile=hw)
    engine.start()

    # Non-blocking:
    future = engine.submit(plan, model_path, "Hello")
    result = future.result(timeout=120)

    # Blocking (drop-in replacement for RuntimeEngine.executePlan):
    result = engine.executePlan(plan, model_path, "Hello", use_async=True)

    print(engine.getSchedulerMetrics().to_dict())
    engine.stop()
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from inference_runtime.inference_session import InferenceSession
from inference_runtime.runtime_config import RuntimeConfig
from inference_runtime.runtime_engine import BenchmarkResult, RuntimeEngine
from inference_runtime.stats_collector import RuntimeStats
from layer_placement.placement_plan import PlacementPlan

from .pipeline_coordinator import PipelineCoordinator
from .scheduler_config import SchedulerConfig
from .scheduler_metrics import SchedulerMetrics, SchedulerReport
from .task_queue import TaskFuture


# ---------------------------------------------------------------------------
# AsyncRuntimeEngine
# ---------------------------------------------------------------------------

class AsyncRuntimeEngine(RuntimeEngine):
    """
    Phase 8 extension of RuntimeEngine with async pipelining.

    Parameters
    ----------
    hw_profile : dict, optional
        Hardware profile. Auto-loaded from hardware_profile.json if None.
    llama_exe_path : Path, optional
        Path to llama.exe. Auto-detected if None.
    config : RuntimeConfig, optional
        Default runtime configuration.
    scheduler_config : SchedulerConfig, optional
        Scheduler configuration. Auto-built from hw_profile if None.
    """

    def __init__(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        llama_exe_path: Optional[Path] = None,
        config: Optional[RuntimeConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ) -> None:
        super().__init__(
            hw_profile=hw_profile,
            llama_exe_path=llama_exe_path,
            config=config,
        )

        self.scheduler_config = (
            scheduler_config
            or SchedulerConfig.from_hw_profile(self.hw_profile)
        )

        # Build session factory (closes over engine state)
        def _session_factory(model_path, plan, cfg, hw):
            return InferenceSession(
                model_path=model_path,
                plan=plan,
                config=cfg,
                hw_profile=hw,
                llama_exe_path=self.llama_exe_path,
            )

        self._coordinator = PipelineCoordinator(
            config=self.scheduler_config,
            hw_profile=self.hw_profile,
            session_factory=_session_factory,
        )

        self._started = False
        self._sequential_tps_baseline: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "AsyncRuntimeEngine":
        """Start the pipeline coordinator and worker pools."""
        if not self._started:
            self._coordinator.start()
            self._started = True
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully stop all worker pools."""
        if self._started:
            self._coordinator.stop(timeout=timeout)
            self._started = False

    def __enter__(self) -> "AsyncRuntimeEngine":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Core API: executePlan (override to route through pipeline)
    # ------------------------------------------------------------------

    def executePlan(
        self,
        plan: PlacementPlan,
        model_path: Path,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        config_override: Optional[RuntimeConfig] = None,
        use_async: bool = True,
    ) -> Any:
        """
        Execute a placement plan.

        When ``use_async=True`` and the scheduler is started, routes through
        the 3-stage async pipeline. Otherwise falls back to Phase 4 execution.

        Parameters
        ----------
        plan, model_path, prompt, on_token, config_override
            Same as RuntimeEngine.executePlan().
        use_async : bool
            Route through async pipeline. Default True.

        Returns
        -------
        InferenceResult
        """
        if use_async and self._started:
            effective_config = config_override or self.config
            result = self._coordinator.run(
                plan=plan,
                model_path=Path(model_path),
                prompt=prompt,
                config=effective_config,
                on_token=on_token,
                timeout=effective_config.process_timeout_sec,
            )
            self._last_result = result
            self._last_stats = getattr(result, "stats", None)
            return result

        # Fallback: synchronous Phase 4 path
        return super().executePlan(plan, model_path, prompt, on_token, config_override)

    execute_plan = executePlan

    # ------------------------------------------------------------------
    # New API: submit (non-blocking)
    # ------------------------------------------------------------------

    def submit(
        self,
        plan: PlacementPlan,
        model_path: Path,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        config_override: Optional[RuntimeConfig] = None,
    ) -> TaskFuture:
        """
        Submit an inference request non-blocking.

        Starts the scheduler if not already started.

        Parameters
        ----------
        plan : PlacementPlan
            Phase 3 placement plan.
        model_path : Path
            GGUF model file path.
        prompt : str
            Input prompt.
        on_token : callable, optional
            Streaming token callback.
        config_override : RuntimeConfig, optional
            Per-request config override.

        Returns
        -------
        TaskFuture[InferenceResult]
            Future that resolves to an InferenceResult.
        """
        if not self._started:
            self.start()

        effective_config = config_override or self.config
        return self._coordinator.submit(
            plan=plan,
            model_path=Path(model_path),
            prompt=prompt,
            config=effective_config,
            on_token=on_token,
        )

    # ------------------------------------------------------------------
    # New API: submitBatch
    # ------------------------------------------------------------------

    def submitBatch(
        self,
        requests: List[Dict[str, Any]],
        config_override: Optional[RuntimeConfig] = None,
    ) -> List[TaskFuture]:
        """
        Submit multiple inference requests concurrently.

        Requests are submitted as fast as the queue allows (backpressure
        applies at max_queue_depth). Each request runs through the full
        3-stage pipeline.

        Parameters
        ----------
        requests : list of dict
            Each dict must contain:
              ``plan``       : PlacementPlan
              ``model_path`` : Path
              ``prompt``     : str
            Optional keys:
              ``on_token``   : callable

        config_override : RuntimeConfig, optional
            Applied to all requests in the batch.

        Returns
        -------
        list of TaskFuture[InferenceResult]
            One future per request, in submission order.
        """
        if not self._started:
            self.start()

        futures: List[TaskFuture] = []
        for req in requests:
            future = self.submit(
                plan=req["plan"],
                model_path=req["model_path"],
                prompt=req["prompt"],
                on_token=req.get("on_token"),
                config_override=config_override,
            )
            futures.append(future)
        return futures

    submit_batch = submitBatch  # snake_case alias

    # ------------------------------------------------------------------
    # New API: getSchedulerMetrics
    # ------------------------------------------------------------------

    def getSchedulerMetrics(self) -> SchedulerMetrics:
        """
        Return aggregate scheduler performance metrics.

        Returns
        -------
        SchedulerMetrics
            Includes gpu_idle_pct, pcie_latency_hidden_ms, bubble_rate_pct,
            speedup_estimate, mean_tokens_per_sec.
        """
        return self._coordinator.get_metrics(
            sequential_tps=self._sequential_tps_baseline,
        )

    get_scheduler_metrics = getSchedulerMetrics  # snake_case alias

    def getSchedulerReport(self) -> SchedulerReport:
        """Return a :class:`SchedulerReport` for printing."""
        return SchedulerReport(
            metrics=self.getSchedulerMetrics(),
            sequential_tps=self._sequential_tps_baseline,
        )

    get_scheduler_report = getSchedulerReport

    # ------------------------------------------------------------------
    # New API: getBenchmarkComparison
    # ------------------------------------------------------------------

    def getBenchmarkComparison(
        self,
        plan: PlacementPlan,
        model_path: Path,
        n_runs: int = 3,
        warmup_runs: int = 1,
        prompt: str = (
            "The transformer architecture introduced self-attention, which allows "
            "models to weigh the relevance of different parts of the input when "
            "generating each output token."
        ),
    ) -> Dict[str, Any]:
        """
        Run both synchronous (Phase 4) and async (Phase 8) paths and compare.

        Performs ``n_runs`` runs in each mode and returns a comparison dict.

        Parameters
        ----------
        plan : PlacementPlan
            Placement plan to benchmark.
        model_path : Path
            GGUF model file path.
        n_runs : int
            Number of measured runs per mode. Default 3.
        warmup_runs : int
            Warmup runs before measurement. Default 1.
        prompt : str
            Benchmark prompt.

        Returns
        -------
        dict
            Keys: ``sequential``, ``async``, ``comparison``, ``report``.
        """
        model_path = Path(model_path)
        print("\n[Phase 8 Benchmark Comparison]")

        # --- Sequential baseline (Phase 4) ---
        print("  Running sequential baseline (Phase 4)...")
        seq_result = self.benchmark(
            plan=plan,
            model_path=model_path,
            n_runs=n_runs,
            warmup_runs=warmup_runs,
            prompt=prompt,
            config_override=self.config,
        )
        seq_tps = seq_result.mean_eval_tps
        self._sequential_tps_baseline = seq_tps

        # --- Async (Phase 8) ---
        print("  Running async pipeline (Phase 8)...")
        if not self._started:
            self.start()

        # Warmup
        for _ in range(warmup_runs):
            try:
                self._coordinator.run(plan, model_path, prompt, self.config, timeout=300.0)
            except Exception:
                pass

        # Measured runs
        async_tps_list: List[float] = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            try:
                result = self._coordinator.run(plan, model_path, prompt, self.config, timeout=300.0)
                elapsed = time.perf_counter() - t0
                tps = getattr(getattr(result, "stats", None), "eval_tps", 0.0)
                async_tps_list.append(tps)
                print(f"    async run {i + 1}/{n_runs}: {tps:.1f} tok/s ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"    async run {i + 1}/{n_runs}: FAILED ({exc})")

        import statistics as _stats
        async_mean = _stats.mean(async_tps_list) if async_tps_list else 0.0

        metrics = self.getSchedulerMetrics()
        report = SchedulerReport(metrics, sequential_tps=seq_tps)

        improvement_pct = (async_mean / max(seq_tps, 1e-9) - 1.0) * 100.0

        print()
        print(report.comparison_table())

        return {
            "sequential": seq_result.to_dict(),
            "async_mean_tps": round(async_mean, 2),
            "sequential_mean_tps": round(seq_tps, 2),
            "improvement_pct": round(improvement_pct, 2),
            "scheduler_metrics": metrics.to_dict(),
            "report": report.full_report(),
        }

    get_benchmark_comparison = getBenchmarkComparison  # snake_case alias

    # ------------------------------------------------------------------
    # Pool diagnostics
    # ------------------------------------------------------------------

    def getPoolStats(self) -> Dict[str, Any]:
        """Return live worker pool statistics."""
        return self._coordinator.pool_stats()

    get_pool_stats = getPoolStats
