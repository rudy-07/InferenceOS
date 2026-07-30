"""
runtime_engine.py
-----------------
Top-level facade for the Phase 4 Inference Execution Runtime.

Phase 8 note: when ``config.enable_async_scheduler=True``, executePlan()
automatically routes through the :class:`AsyncRuntimeEngine` pipeline.
Callers can also instantiate AsyncRuntimeEngine directly for full Phase 8
control (submit, submitBatch, getBenchmarkComparison, getSchedulerMetrics).

Public API (camelCase + snake_case aliases):

    executePlan(plan, model_path, prompt, on_token)  -> InferenceResult
    benchmark(plan, model_path, n_runs, ...)          -> BenchmarkResult
    getRuntimeStats()                                  -> RuntimeStats | None
    getLastResult()                                    -> InferenceResult | None

This engine is the single entry point that future orchestration layers use
to run inference. It loads hardware configuration, auto-detects the llama.exe
binary, wires up the InferenceSession, and aggregates multi-run benchmarks.

Example
-------
    from inference_runtime import RuntimeEngine
    from layer_placement import PlacementEngine, ModelDescriptor
    from orchestrator.gguf_parser import read_gguf_metadata
    from pathlib import Path

    hw = json.load(open("hardware_profile.json"))
    meta = read_gguf_metadata(Path("models/model.gguf"))
    model = ModelDescriptor.from_gguf_metadata(meta, ...)
    plan = PlacementEngine().generatePlacementPlan(model)

    engine = RuntimeEngine(hw_profile=hw)
    result = engine.executePlan(plan, Path("models/model.gguf"), "Hello!")
    print(result.generated_text)

    bench = engine.benchmark(plan, Path("models/model.gguf"), n_runs=3)
    print(f"{bench.mean_eval_tps:.1f} tok/s")
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .inference_session import InferenceResult, InferenceSession
from .runtime_config import RuntimeConfig
from .stats_collector import RuntimeStats
from layer_placement.placement_plan import PlacementPlan


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """
    Aggregate statistics across multiple benchmark runs.

    Attributes
    ----------
    runs : List[RuntimeStats]
        Per-run statistics (warmup runs excluded).
    mean_eval_tps : float
        Mean generation speed across all non-warmup runs (tokens/sec).
    std_eval_tps : float
        Standard deviation of generation speed.
    mean_prompt_tps : float
        Mean prompt processing speed (tokens/sec).
    peak_eval_tps : float
        Best single-run generation speed.
    min_eval_tps : float
        Worst single-run generation speed.
    p95_latency_ms : float
        Average of per-run p95 per-token latencies.
    avg_gpu_util_pct : float
        Average GPU utilization across all non-warmup runs.
    avg_cpu_util_pct : float
        Average CPU utilization across all non-warmup runs.
    total_pipeline_stalls : int
        Total pipeline stall sample count across all non-warmup runs.
    plan_summary : str
        One-line summary from :meth:`PlacementPlan.summary`.
    backend : str
        Backend used during benchmarking.
    n_runs : int
        Number of measured runs (excluding warmup).
    warmup_runs : int
        Number of warmup runs performed.
    """
    runs: List[RuntimeStats] = field(default_factory=list)
    mean_eval_tps: float = 0.0
    std_eval_tps: float = 0.0
    mean_prompt_tps: float = 0.0
    peak_eval_tps: float = 0.0
    min_eval_tps: float = 0.0
    p95_latency_ms: float = 0.0
    avg_gpu_util_pct: float = 0.0
    avg_cpu_util_pct: float = 0.0
    total_pipeline_stalls: int = 0
    plan_summary: str = ""
    backend: str = "unknown"
    n_runs: int = 0
    warmup_runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_eval_tps": round(self.mean_eval_tps, 2),
            "std_eval_tps": round(self.std_eval_tps, 2),
            "mean_prompt_tps": round(self.mean_prompt_tps, 2),
            "peak_eval_tps": round(self.peak_eval_tps, 2),
            "min_eval_tps": round(self.min_eval_tps, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "avg_gpu_util_pct": round(self.avg_gpu_util_pct, 1),
            "avg_cpu_util_pct": round(self.avg_cpu_util_pct, 1),
            "total_pipeline_stalls": self.total_pipeline_stalls,
            "plan_summary": self.plan_summary,
            "backend": self.backend,
            "n_runs": self.n_runs,
            "warmup_runs": self.warmup_runs,
        }

    def report(self) -> str:
        """Generate a multi-line human-readable benchmark report."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║       InferenceOS  ·  Phase 4 Benchmark Results              ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  Plan:               {self.plan_summary}",
            f"  Backend:            {self.backend.upper()}",
            f"  Measured Runs:      {self.n_runs}  (warmup: {self.warmup_runs})",
            "",
            "THROUGHPUT",
            "──────────────────────────────────────────────────────────────",
            f"  Mean Generation:    {self.mean_eval_tps:.2f} tok/s  (±{self.std_eval_tps:.2f})",
            f"  Peak Generation:    {self.peak_eval_tps:.2f} tok/s",
            f"  Min Generation:     {self.min_eval_tps:.2f} tok/s",
            f"  Mean Prompt Proc:   {self.mean_prompt_tps:.2f} tok/s",
            "",
            "LATENCY",
            "──────────────────────────────────────────────────────────────",
            f"  p95 per-token:      {self.p95_latency_ms:.2f} ms",
            "",
            "UTILIZATION",
            "──────────────────────────────────────────────────────────────",
            f"  Avg GPU:            {self.avg_gpu_util_pct:.1f}%",
            f"  Avg CPU:            {self.avg_cpu_util_pct:.1f}%",
            f"  Pipeline Stalls:    {self.total_pipeline_stalls}",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RuntimeEngine
# ---------------------------------------------------------------------------

class RuntimeEngine:
    """
    Top-level facade for InferenceOS Phase 4 execution runtime.

    Manages session creation, benchmark orchestration, and statistics
    aggregation. Designed to be instantiated once and reused across
    multiple inference calls.

    Parameters
    ----------
    hw_profile : dict, optional
        Hardware profile. If None, loads from ``hardware_profile.json`` in
        the project root.
    llama_exe_path : Path, optional
        Explicit path to llama.exe. If None, auto-searched in ``build/bin/``.
    config : RuntimeConfig, optional
        Default runtime configuration. Can be overridden per call.

    Examples
    --------
    >>> engine = RuntimeEngine()
    >>> result = engine.executePlan(plan, model_path, "Hello world")
    >>> print(result.stats.eval_tps, "tok/s")
    >>>
    >>> bench = engine.benchmark(plan, model_path, n_runs=3)
    >>> print(bench.report())
    """

    def __init__(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        llama_exe_path: Optional[Path] = None,
        config: Optional[RuntimeConfig] = None,
    ) -> None:
        self.hw_profile = hw_profile or self._load_hw_profile()
        self.llama_exe_path = llama_exe_path  # None = auto-detected by InferenceSession
        self.config = config or RuntimeConfig.from_hw_profile(self.hw_profile)

        self._last_result: Optional[InferenceResult] = None
        self._last_stats: Optional[RuntimeStats] = None

    # ---------------------------------------------------------------------------
    # Core API 1: executePlan
    # ---------------------------------------------------------------------------

    def executePlan(
        self,
        plan: PlacementPlan,
        model_path: Path,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        config_override: Optional[RuntimeConfig] = None,
    ) -> InferenceResult:
        """
        Execute a placement plan against a model and return the result.

        Creates a fresh :class:`InferenceSession` for this call. The session
        is automatically closed after the call completes.

        Parameters
        ----------
        plan : PlacementPlan
            Phase 3 placement plan (from :class:`PlacementEngine`).
        model_path : Path
            Path to the GGUF model file.
        prompt : str
            Input prompt text.
        on_token : callable, optional
            Called for each generated token. Enables streaming UX.
            Signature: ``(token_text: str) -> None``.
        config_override : RuntimeConfig, optional
            Override the engine's default config for this call only.

        Returns
        -------
        InferenceResult
            Complete result including generated text and statistics.
        """
        effective_config = config_override or self.config

        with InferenceSession(
            model_path=Path(model_path),
            plan=plan,
            config=effective_config,
            hw_profile=self.hw_profile,
            llama_exe_path=self.llama_exe_path,
        ) as session:
            result = session.run(prompt, on_token=on_token)

        self._last_result = result
        self._last_stats = result.stats
        return result

    # ---------------------------------------------------------------------------
    # Core API 1b: profileRun (Phase 9)
    # ---------------------------------------------------------------------------

    def profileRun(
        self,
        plan: PlacementPlan,
        model_path: Path,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        config_override: Optional[RuntimeConfig] = None,
        vram_free_mb: float = 0.0,
    ) -> Any:
        """
        Execute a placement plan with deep Phase 9 runtime profiling.

        Returns a tuple of (InferenceResult, ProfilerResult).
        """
        from runtime_profiler import ProfilerEngine
        effective_config = config_override or self.config

        profiler = ProfilerEngine(
            hw_profile=self.hw_profile,
            sample_interval_ms=effective_config.profiler_sample_interval_ms,
        )

        model_name = getattr(plan, "model_name", "model")
        backend = getattr(self._last_result, "backend", "cpu") if hasattr(self, "_last_result") else "cpu"

        # Intercept token streaming callback to record arrival timestamps
        def _wrapped_on_token(token_str: str) -> None:
            if hasattr(self, "_active_collector") and self._active_collector:
                self._active_collector.record_token()
            if on_token:
                on_token(token_str)

        with profiler.profile_session(
            plan=plan,
            model_name=model_name,
            backend=backend,
            context_length=getattr(plan, "context_length", 4096),
            vram_free_mb=vram_free_mb,
        ) as collector:
            self._active_collector = collector
            try:
                result = self.executePlan(
                    plan=plan,
                    model_path=model_path,
                    prompt=prompt,
                    on_token=_wrapped_on_token,
                    config_override=effective_config,
                )
                if result and result.stats and hasattr(collector, "set_stderr_info"):
                    collector.set_stderr_info({
                        "prompt_eval_tokens": getattr(result.stats, "prompt_tokens", 0),
                        "eval_tokens": getattr(result.stats, "tokens_generated", 0),
                        "prompt_eval_ms": getattr(result.stats, "prompt_eval_ms", 0.0),
                        "eval_ms": getattr(result.stats, "eval_ms", 0.0),
                        "prompt_eval_tps": getattr(result.stats, "prompt_eval_tps", 0.0),
                        "eval_tps": getattr(result.stats, "eval_tps", 0.0),
                    })
            finally:
                self._active_collector = None

        prof_result = profiler.get_last_result()
        return result, prof_result

    profile_run = profileRun  # snake_case alias

    execute_plan = executePlan  # snake_case alias

    def _maybe_async_execute(
        self,
        plan: PlacementPlan,
        model_path: Path,
        prompt: str,
        on_token: Optional[Callable[[str], None]],
        config_override: Optional[RuntimeConfig],
    ) -> Optional[Any]:
        """
        If enable_async_scheduler is True in the effective config, route
        through AsyncRuntimeEngine. Returns None to indicate the caller
        should fall back to the synchronous path.
        """
        effective_config = config_override or self.config
        if not effective_config.enable_async_scheduler:
            return None
        try:
            from async_scheduler import AsyncRuntimeEngine
            from async_scheduler.scheduler_config import SchedulerConfig
            sc = SchedulerConfig.from_hw_profile(
                self.hw_profile,
                n_cpu_workers=effective_config.scheduler_n_cpu_workers,
                max_queue_depth=effective_config.scheduler_queue_depth,
                enable_prefetch=effective_config.scheduler_prefetch,
                benchmark_mode=effective_config.scheduler_benchmark_mode,
            )
            async_engine = AsyncRuntimeEngine(
                hw_profile=self.hw_profile,
                llama_exe_path=self.llama_exe_path,
                config=effective_config,
                scheduler_config=sc,
            )
            with async_engine:
                return async_engine.executePlan(
                    plan, model_path, prompt, on_token,
                    config_override=effective_config,
                    use_async=True,
                )
        except Exception:
            # Any error in async path → fall back to synchronous
            return None

    # ---------------------------------------------------------------------------
    # Core API 2: benchmark
    # ---------------------------------------------------------------------------

    def benchmark(
        self,
        plan: PlacementPlan,
        model_path: Path,
        n_runs: int = 3,
        warmup_runs: int = 1,
        prompt: str = (
            "The transformer architecture introduced self-attention, which allows "
            "models to weigh the relevance of different parts of the input when "
            "generating each output token. This mechanism"
        ),
        config_override: Optional[RuntimeConfig] = None,
        on_run_complete: Optional[Callable[[int, RuntimeStats], None]] = None,
    ) -> BenchmarkResult:
        """
        Run multiple inference passes and aggregate performance statistics.

        Warmup runs are performed first (to heat up caches and JIT paths)
        and are excluded from the reported statistics.

        Parameters
        ----------
        plan : PlacementPlan
            Phase 3 placement plan.
        model_path : Path
            Path to the GGUF model file.
        n_runs : int
            Number of measured benchmark runs. Default 3.
        warmup_runs : int
            Number of warmup runs (not counted in stats). Default 1.
        prompt : str
            Prompt for all benchmark runs. Should be representative of
            typical usage length.
        config_override : RuntimeConfig, optional
            Override default config for all benchmark runs.
        on_run_complete : callable, optional
            Callback fired after each measured run.
            Signature: ``(run_index: int, stats: RuntimeStats) -> None``.

        Returns
        -------
        BenchmarkResult
            Aggregated statistics across all measured runs.
        """
        effective_config = config_override or self.config
        model_path = Path(model_path)
        all_stats: List[RuntimeStats] = []
        backend_name = "unknown"

        total_runs = warmup_runs + n_runs

        for i in range(total_runs):
            is_warmup = i < warmup_runs
            run_label = f"warmup {i + 1}" if is_warmup else f"run {i - warmup_runs + 1}/{n_runs}"
            print(f"  [InferenceOS Benchmark] {run_label}...", end="", flush=True)

            t_start = time.perf_counter()
            result = self.executePlan(
                plan=plan,
                model_path=model_path,
                prompt=prompt,
                config_override=effective_config,
            )
            t_end = time.perf_counter()

            backend_name = result.backend
            elapsed = (t_end - t_start)

            if is_warmup:
                print(f"  (warmup, {elapsed:.1f}s)")
                continue

            print(f"  {result.stats.eval_tps:.1f} tok/s  ({elapsed:.1f}s)")
            all_stats.append(result.stats)

            if on_run_complete is not None:
                try:
                    on_run_complete(i - warmup_runs, result.stats)
                except Exception:
                    pass

        return self._aggregate_benchmark(
            runs=all_stats,
            plan=plan,
            backend=backend_name,
            n_measured=n_runs,
            n_warmup=warmup_runs,
        )

    # ---------------------------------------------------------------------------
    # Core API 3: statistics accessors
    # ---------------------------------------------------------------------------

    def getRuntimeStats(self) -> Optional[RuntimeStats]:
        """
        Return the :class:`RuntimeStats` from the most recent execution.

        Returns
        -------
        RuntimeStats or None
            None if no execution has been performed yet.
        """
        return self._last_stats

    get_runtime_stats = getRuntimeStats  # snake_case alias

    def getLastResult(self) -> Optional[InferenceResult]:
        """
        Return the full :class:`InferenceResult` from the most recent execution.

        Returns
        -------
        InferenceResult or None
        """
        return self._last_result

    get_last_result = getLastResult  # snake_case alias

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _aggregate_benchmark(
        runs: List[RuntimeStats],
        plan: PlacementPlan,
        backend: str,
        n_measured: int,
        n_warmup: int,
    ) -> BenchmarkResult:
        """Aggregate a list of per-run RuntimeStats into a BenchmarkResult."""
        if not runs:
            return BenchmarkResult(
                plan_summary=plan.summary(),
                backend=backend,
                n_runs=0,
                warmup_runs=n_warmup,
            )

        eval_tps_list = [s.eval_tps for s in runs]
        prompt_tps_list = [s.prompt_eval_tps for s in runs]
        gpu_util_list = [s.avg_gpu_util_pct for s in runs]
        cpu_util_list = [s.avg_cpu_util_pct for s in runs]
        p95_list = [s.p95_latency_ms for s in runs]

        mean_tps = statistics.mean(eval_tps_list)
        std_tps = statistics.stdev(eval_tps_list) if len(eval_tps_list) > 1 else 0.0

        return BenchmarkResult(
            runs=runs,
            mean_eval_tps=mean_tps,
            std_eval_tps=std_tps,
            mean_prompt_tps=statistics.mean(prompt_tps_list),
            peak_eval_tps=max(eval_tps_list),
            min_eval_tps=min(eval_tps_list),
            p95_latency_ms=statistics.mean(p95_list) if p95_list else 0.0,
            avg_gpu_util_pct=statistics.mean(gpu_util_list),
            avg_cpu_util_pct=statistics.mean(cpu_util_list),
            total_pipeline_stalls=sum(s.pipeline_stalls for s in runs),
            plan_summary=plan.summary(),
            backend=backend,
            n_runs=n_measured,
            warmup_runs=n_warmup,
        )

    @staticmethod
    def _load_hw_profile() -> Dict[str, Any]:
        """Load hardware_profile.json from project root."""
        candidate = Path(__file__).parent.parent / "hardware_profile.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback: try live profiler
        try:
            from profiler import get_system_resources
            return get_system_resources().to_dict()
        except Exception:
            pass

        # Minimal safe defaults
        return {
            "gpus": [],
            "igpus": [],
            "ram": {"total_bytes": 16 * 1024 ** 3, "available_gb": 8.0},
            "memory": {"total_bytes": 16 * 1024 ** 3, "available_gb": 8.0},
            "cpu": {"logical_cores": 4, "physical_cores": 4, "base_freq_mhz": 2000.0},
            "interconnects": [],
            "inference_hints": {},
        }
