"""
pipeline_coordinator.py
-----------------------
3-stage producer/consumer pipeline for Phase 8 Async Execution Scheduler.

Pipeline design
---------------
Three stages execute concurrently using worker pools:

  Stage 1: PREPARE  (CPU pool)
  ─────────────────────────────
  - Schedule prefetch for the NEXT request's boundary segments
  - Record prepare timing
  - Produce a PrepareResult → Stage 2

  Stage 2: TRANSFER  (transfer pool, single thread)
  ──────────────────────────────────────────────────
  - Execute the prefetch that Stage 1 scheduled
  - Emit CUDA/Vulkan env-vars to the stream manager
  - Produce a TransferResult → Stage 3

  Stage 3: COMPUTE  (CPU pool, high priority)
  ─────────────────────────────────────────────
  - Launch llama.cpp (InferenceSession.run)
  - Record compute timing
  - Return InferenceResult to caller via TaskFuture

Pipeline overlap
----------------
Stages are pipelined across *consecutive requests*:

  Request N:   [PREPARE] ──────── [TRANSFER] ──────── [COMPUTE]
  Request N+1:             [PREPARE] ──────── [TRANSFER] ──────── [COMPUTE]

While Request N is in COMPUTE (GPU busy), Request N+1's PREPARE and TRANSFER
are already in flight. This hides the PCIe and CPU preparation latency.

Single-request mode
-------------------
Even for single requests, the pipeline still provides benefit:
- Prefetch runs concurrently with the previous request's compute
- CUDA env-vars are applied before subprocess launch
- Stage timing is recorded for diagnostics

Usage
-----
    coordinator = PipelineCoordinator(config, hw_profile)
    coordinator.start()

    result = coordinator.run(plan, model_path, prompt)
    # or non-blocking:
    future = coordinator.submit(plan, model_path, prompt)
    result = future.result(timeout=120)

    metrics = coordinator.get_metrics()
    coordinator.stop()
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .cuda_stream_manager import CudaStreamManager, VirtualStream
from .prefetch_manager import PrefetchManager, PrefetchResult
from .scheduler_config import SchedulerConfig
from .scheduler_metrics import MetricsAccumulator, SchedulerMetrics, StageTimings
from .task_queue import Priority, TaskFuture
from .worker_pool import WorkerPool


# ---------------------------------------------------------------------------
# Stage result types (internal handoff objects)
# ---------------------------------------------------------------------------

@dataclass
class _PrepareResult:
    request_id: int
    plan: Any          # PlacementPlan
    model_path: Path
    prompt: str
    config: Any        # RuntimeConfig
    on_token: Optional[Callable]
    prefetch_result: Optional[PrefetchResult]
    prepare_ms: float


@dataclass
class _TransferResult:
    request_id: int
    plan: Any
    model_path: Path
    prompt: str
    config: Any
    on_token: Optional[Callable]
    prefetch_result: Optional[PrefetchResult]
    env_vars: Dict[str, str]
    prepare_ms: float
    transfer_ms: float


# ---------------------------------------------------------------------------
# PipelineCoordinator
# ---------------------------------------------------------------------------

class PipelineCoordinator:
    """
    3-stage async pipeline: Prepare → Transfer → Compute.

    Parameters
    ----------
    config : SchedulerConfig
        Scheduler configuration.
    hw_profile : dict
        Hardware profile for backend and prefetch decisions.
    session_factory : callable
        Callable that returns a new :class:`InferenceSession` given
        ``(model_path, plan, config, hw_profile)``.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        hw_profile: Dict[str, Any],
        session_factory: Callable,
    ) -> None:
        self.config = config
        self.hw_profile = hw_profile
        self._session_factory = session_factory

        # Worker pools
        self._cpu_pool = WorkerPool.cpu_pool(
            n_workers=config.n_cpu_workers,
            max_queue_depth=config.max_queue_depth,
        )
        self._transfer_pool = WorkerPool.transfer_pool(
            max_queue_depth=config.max_queue_depth,
        )

        # Prefetch manager
        self._prefetch = PrefetchManager(
            pinned_buffer_size_mb=config.pinned_buffer_size_mb,
            n_buffers=max(2, config.prefetch_lookahead + 2),
        )

        # CUDA stream manager (initialized lazily, after first plan is known)
        self._stream_manager: Optional[CudaStreamManager] = None

        # Metrics
        self._accumulator = MetricsAccumulator()
        self._lock = threading.Lock()
        self._request_counter = 0
        self._running = False

        # PCIe bandwidth from hw_profile
        self._pcie_bw_gbps = self._read_pcie_bw()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "PipelineCoordinator":
        """Start all worker pools."""
        self._cpu_pool.start()
        self._transfer_pool.start()
        self._running = True
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully stop all worker pools."""
        self._running = False
        self._cpu_pool.stop(timeout=timeout)
        self._transfer_pool.stop(timeout=timeout)
        self._prefetch.release_all()

    def __enter__(self) -> "PipelineCoordinator":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        plan: Any,
        model_path: Path,
        prompt: str,
        config: Any,
        on_token: Optional[Callable[[str], None]] = None,
        timeout: float = 600.0,
    ) -> Any:
        """
        Execute one inference request through the full pipeline.

        Blocks until the compute stage completes.

        Parameters
        ----------
        plan : PlacementPlan
            Placement plan from Phase 3.
        model_path : Path
            Path to GGUF model file.
        prompt : str
            Input prompt.
        config : RuntimeConfig
            Runtime configuration.
        on_token : callable, optional
            Streaming token callback.
        timeout : float
            Maximum seconds to wait for completion.

        Returns
        -------
        InferenceResult
            Generated text and statistics.
        """
        future = self.submit(plan, model_path, prompt, config, on_token)
        return future.result(timeout=timeout)

    def submit(
        self,
        plan: Any,
        model_path: Path,
        prompt: str,
        config: Any,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> TaskFuture:
        """
        Submit an inference request non-blocking.

        Returns a :class:`TaskFuture` that resolves to an
        :class:`InferenceResult` when the compute stage completes.
        """
        with self._lock:
            req_id = self._request_counter
            self._request_counter += 1

        # Ensure stream manager is initialized
        if self._stream_manager is None:
            backend = (
                config.force_backend
                or self.hw_profile.get("inference_hints", {}).get("recommended_backend", "cpu")
            )
            self._stream_manager = CudaStreamManager(
                backend=backend,
                hw_profile=self.hw_profile,
                plan=plan,
                pcie_bandwidth_gbps=self._pcie_bw_gbps,
            )
        else:
            self._stream_manager.update_plan(plan)

        # Submit Stage 1 (PREPARE) to cpu_pool
        return self._cpu_pool.submit(
            self._stage_prepare,
            req_id=req_id,
            plan=plan,
            model_path=Path(model_path),
            prompt=prompt,
            config=config,
            on_token=on_token,
            priority=Priority.NORMAL,
        )

    def get_metrics(
        self,
        sequential_tps: float = 0.0,
    ) -> SchedulerMetrics:
        """
        Compute and return aggregate scheduler metrics.

        Parameters
        ----------
        sequential_tps : float
            Baseline tokens/sec from synchronous execution (for speedup calc).

        Returns
        -------
        SchedulerMetrics
        """
        return self._accumulator.compute(
            sequential_tps=sequential_tps,
            pcie_bw_gbps=self._pcie_bw_gbps,
            boundary_crossings=0,
        )

    def pool_stats(self) -> Dict[str, Any]:
        """Return status of all worker pools."""
        return {
            "cpu_pool": self._cpu_pool.stats.to_dict(),
            "transfer_pool": self._transfer_pool.stats.to_dict(),
            "prefetch": self._prefetch.get_stats(),
        }

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_prepare(
        self,
        req_id: int,
        plan: Any,
        model_path: Path,
        prompt: str,
        config: Any,
        on_token: Optional[Callable],
    ) -> Any:
        """
        Stage 1: CPU PREPARE.
        Schedule prefetch for boundary segments, record timing.
        """
        t_start = time.perf_counter()

        # Schedule prefetch (the actual I/O happens in Stage 2)
        prefetch_result = None
        if self.config.enable_prefetch:
            try:
                # Just identify boundary segments here; actual read in Stage 2
                from layer_placement.placement_plan import PlacementPlan
                if isinstance(plan, PlacementPlan):
                    # Identify boundaries without reading file
                    prefetch_result = self._prefetch.prefetch(plan, model_path=None)
            except Exception:
                pass

        prepare_ms = (time.perf_counter() - t_start) * 1000.0

        prepare_result = _PrepareResult(
            request_id=req_id,
            plan=plan,
            model_path=model_path,
            prompt=prompt,
            config=config,
            on_token=on_token,
            prefetch_result=prefetch_result,
            prepare_ms=prepare_ms,
        )

        # Submit Stage 2 (TRANSFER) to transfer pool and wait for it inline
        # so Stage 3 runs on the cpu_pool (not transfer pool)
        return self._run_stage_transfer(prepare_result)

    def _run_stage_transfer(self, prep: _PrepareResult) -> Any:
        """
        Stage 2: TRANSFER — execute prefetch I/O, emit env-var hints.
        This runs on the transfer pool's single thread to avoid bus contention.
        """
        future = self._transfer_pool.submit(
            self._stage_transfer,
            prep,
            priority=Priority.HIGH,
        )
        transfer_result = future.result(timeout=120.0)
        return self._stage_compute(transfer_result)

    def _stage_transfer(self, prep: _PrepareResult) -> _TransferResult:
        """
        Stage 2 body (runs on transfer thread).
        """
        t_start = time.perf_counter()

        # Dispatch TRANSFER virtual stream
        if self._stream_manager is not None:
            self._stream_manager.dispatch(VirtualStream.TRANSFER, "prefetch_boundary_segments")

        # Execute actual prefetch I/O (read model file into pinned buffers)
        prefetch_result = prep.prefetch_result
        if self.config.enable_prefetch and prep.model_path.exists():
            try:
                from layer_placement.placement_plan import PlacementPlan
                if isinstance(prep.plan, PlacementPlan):
                    prefetch_result = self._prefetch.prefetch(prep.plan, prep.model_path)
            except Exception:
                pass

        # Collect env-vars from CUDA stream manager
        env_vars: Dict[str, str] = {}
        if self._stream_manager is not None:
            env_vars = self._stream_manager.env_vars

        transfer_ms = (time.perf_counter() - t_start) * 1000.0

        return _TransferResult(
            request_id=prep.request_id,
            plan=prep.plan,
            model_path=prep.model_path,
            prompt=prep.prompt,
            config=prep.config,
            on_token=prep.on_token,
            prefetch_result=prefetch_result,
            env_vars=env_vars,
            prepare_ms=prep.prepare_ms,
            transfer_ms=transfer_ms,
        )

    def _stage_compute(self, xfr: _TransferResult) -> Any:
        """
        Stage 3: COMPUTE — launch llama.cpp and return InferenceResult.
        """
        t_start = time.perf_counter()

        if self._stream_manager is not None:
            self._stream_manager.dispatch(VirtualStream.COMPUTE, "llama_cpp_inference")

        # Apply prefetch hint: if prefetch recommended no-mmap, set it
        effective_config = xfr.config
        if (
            xfr.prefetch_result is not None
            and xfr.prefetch_result.recommended_no_mmap
            and hasattr(effective_config, "use_mmap")
        ):
            # Create a modified config copy with use_mmap=False
            import copy as _copy
            effective_config = _copy.copy(xfr.config)
            effective_config.use_mmap = False

        # Inject CUDA env-vars into a force_backend-aware config clone
        # The ProcessManager reads env_vars from BackendInfo.env_vars;
        # we patch the hw_profile hint so BackendSelector picks them up.
        hw_profile = dict(self.hw_profile)
        if xfr.env_vars:
            existing = dict(hw_profile.get("inference_hints", {}))
            existing.setdefault("extra_env_vars", {}).update(xfr.env_vars)
            hw_profile["inference_hints"] = existing

        # Launch inference via the session factory
        try:
            session = self._session_factory(
                xfr.model_path,
                xfr.plan,
                effective_config,
                hw_profile,
            )
            result = session.run(xfr.prompt, on_token=xfr.on_token)
        except Exception as exc:
            compute_ms = (time.perf_counter() - t_start) * 1000.0
            self._record_timings(xfr, compute_ms, eval_tps=0.0, tokens=0)
            raise

        compute_ms = (time.perf_counter() - t_start) * 1000.0

        # Dispatch SAMPLING stream for post-processing
        if self._stream_manager is not None:
            self._stream_manager.dispatch(VirtualStream.SAMPLING, "result_postprocess")

        # Record timings
        tps = getattr(getattr(result, "stats", None), "eval_tps", 0.0)
        tokens = getattr(getattr(result, "stats", None), "tokens_generated", 0)
        self._record_timings(xfr, compute_ms, tps, tokens)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_timings(
        self,
        xfr: _TransferResult,
        compute_ms: float,
        eval_tps: float,
        tokens: int,
    ) -> None:
        """Record stage timings into the metrics accumulator."""
        total_ms = xfr.prepare_ms + xfr.transfer_ms + compute_ms
        timings = StageTimings(
            request_id=xfr.request_id,
            prepare_ms=xfr.prepare_ms,
            transfer_ms=xfr.transfer_ms,
            compute_ms=compute_ms,
            total_ms=total_ms,
            # Idle = time transfer thread had to wait for the cpu_pool prepare result
            # (in this implementation = 0 since prepare runs first and blocks)
            prepare_idle_ms=0.0,
            transfer_idle_ms=0.0,
            compute_idle_ms=0.0,
            tokens_generated=tokens,
            eval_tps=eval_tps,
        )
        self._accumulator.record(timings)

    def _read_pcie_bw(self) -> float:
        for ic in self.hw_profile.get("interconnects", []):
            if str(ic.get("type", "")).upper() == "PCIE":
                return float(ic.get("bandwidth", 16.0))
        return 16.0
