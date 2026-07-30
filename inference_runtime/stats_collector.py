"""
stats_collector.py
-------------------
Two-track runtime statistics collection for Phase 4:

Track A — llama.cpp stderr parsing
    Extracts the structured timing summary that llama.cpp always writes to
    stderr at the end of every run:
      llama_perf_context_print: prompt eval time = X ms / N tokens (...)
      llama_perf_context_print:        eval time = X ms / N runs   (...)

Track B — OS-level background sampling
    A daemon thread polls CPU and GPU utilization at a configurable interval
    using ``psutil`` (CPU) and ``GPUtil`` (GPU, optional). Calculates:
      - Average and peak CPU/GPU utilization during the run
      - Pipeline stall count: samples where CPU > 95% AND GPU < 10%
        (indicates the GPU is starved waiting for CPU-side transfers)

Per-token latency
    The :class:`ProcessManager` feeds token arrival timestamps into a
    shared list, from which this module computes p50/p95/p99 percentiles.
"""
from __future__ import annotations

import re
import statistics
import threading
import time
from dataclasses import dataclass, field
import sys
from typing import Any, Dict, List, Optional


class WindowsGpuMonitor:
    """Queries Windows PDH Performance Counters for native GPU 3D engine utilization."""
    def __init__(self) -> None:
        self._supported = False
        self.hQuery = None
        self.hCounter = None
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                self.pdh = ctypes.windll.pdh
                self.hQuery = wintypes.HANDLE()
                self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.hQuery))
                self.hCounter = wintypes.HANDLE()
                status = self.pdh.PdhAddEnglishCounterW(
                    self.hQuery,
                    r"\GPU Engine(*engtype_3D*)\Utilization Percentage",
                    0,
                    ctypes.byref(self.hCounter),
                )
                if status == 0:
                    self.pdh.PdhCollectQueryData(self.hQuery)
                    self._supported = True
            except Exception:
                self._supported = False

    def sample(self) -> float:
        if not self._supported:
            return -1.0
        try:
            import ctypes
            from ctypes import wintypes
            self.pdh.PdhCollectQueryData(self.hQuery)
            dwBufferSize = wintypes.DWORD(0)
            dwItemCount = wintypes.DWORD(0)
            status = self.pdh.PdhGetFormattedCounterArrayW(
                self.hCounter, 0x00000200, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), None
            )
            if dwBufferSize.value > 0:
                buf = ctypes.create_string_buffer(dwBufferSize.value)
                status = self.pdh.PdhGetFormattedCounterArrayW(
                    self.hCounter, 0x00000200, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), buf
                )
                if status == 0:
                    class PDH_FMT_COUNTERVALUE(ctypes.Structure):
                        _fields_ = [('CStatus', wintypes.DWORD), ('doubleValue', ctypes.c_double)]

                    class PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
                        _fields_ = [('szName', wintypes.LPCWSTR), ('FmtValue', PDH_FMT_COUNTERVALUE)]

                    items = ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM))
                    total_util = sum(items[i].FmtValue.doubleValue for i in range(dwItemCount.value))
                    return min(100.0, max(0.0, total_util))
        except Exception:
            pass
        return -1.0

    def close(self) -> None:
        if self._supported and self.hQuery:
            try:
                self.pdh.PdhCloseQuery(self.hQuery)
            except Exception:
                pass
            self._supported = False


# ---------------------------------------------------------------------------
# RuntimeStats output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RuntimeStats:
    """
    Complete runtime statistics snapshot for a single inference run.

    Throughput
    ----------
    prompt_eval_tps : float
        Tokens per second during prompt processing (prefill phase).
    eval_tps : float
        Tokens per second during autoregressive generation.
    tokens_generated : int
        Number of tokens actually generated.

    Timing (ms)
    -----------
    prompt_eval_ms : float
        Wall time for prompt processing in milliseconds.
    eval_ms : float
        Wall time for token generation in milliseconds.
    total_wall_ms : float
        Total subprocess wall-clock time in milliseconds.

    Per-token latency
    -----------------
    per_token_latency_ms : List[float]
        Measured inter-token arrival times in milliseconds.
        Empty if async streaming was disabled.
    p50_latency_ms : float
        Median per-token latency.
    p95_latency_ms : float
        95th percentile per-token latency.
    p99_latency_ms : float
        99th percentile per-token latency.

    OS utilization (time-averaged)
    -------------------------------
    avg_cpu_util_pct : float
        Mean CPU utilization (%) across all OS samples during the run.
    avg_gpu_util_pct : float
        Mean GPU utilization (%) across all OS samples.
    peak_cpu_util_pct : float
        Maximum CPU utilization observed during the run.
    peak_gpu_util_pct : float
        Maximum GPU utilization observed.

    Pipeline analysis
    -----------------
    estimated_transfer_ms : float
        Model-estimated time spent on PCIe transfers (from Phase 3 cost model).
    pipeline_stalls : int
        Number of OS samples where CPU > 95% AND GPU < 10% simultaneously.
        Indicates the GPU is idle waiting for CPU-side data transfers.
    compute_time_ms : float
        eval_ms - estimated_transfer_ms (GPU active time estimate).

    Session metadata
    ----------------
    n_gpu_layers : int
    context_length : int
    backend : str
    model_name : str
    is_feasible : bool
    """
    # Throughput
    prompt_eval_tps: float = 0.0
    eval_tps: float = 0.0
    tokens_generated: int = 0

    # Timing
    prompt_eval_ms: float = 0.0
    eval_ms: float = 0.0
    total_wall_ms: float = 0.0

    # Per-token latency
    per_token_latency_ms: List[float] = field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # OS utilization
    avg_cpu_util_pct: float = 0.0
    avg_gpu_util_pct: float = 0.0
    peak_cpu_util_pct: float = 0.0
    peak_gpu_util_pct: float = 0.0

    # Pipeline analysis
    estimated_transfer_ms: float = 0.0
    pipeline_stalls: int = 0
    compute_time_ms: float = 0.0

    # Session metadata
    n_gpu_layers: int = 0
    context_length: int = 0
    backend: str = "unknown"
    model_name: str = ""
    is_feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of all statistics."""
        return {
            "throughput": {
                "prompt_eval_tps": round(self.prompt_eval_tps, 2),
                "eval_tps": round(self.eval_tps, 2),
                "tokens_generated": self.tokens_generated,
            },
            "timing_ms": {
                "prompt_eval_ms": round(self.prompt_eval_ms, 2),
                "eval_ms": round(self.eval_ms, 2),
                "total_wall_ms": round(self.total_wall_ms, 2),
            },
            "latency_ms": {
                "p50": round(self.p50_latency_ms, 3),
                "p95": round(self.p95_latency_ms, 3),
                "p99": round(self.p99_latency_ms, 3),
                "sample_count": len(self.per_token_latency_ms),
            },
            "utilization_pct": {
                "avg_cpu": round(self.avg_cpu_util_pct, 1),
                "avg_gpu": round(self.avg_gpu_util_pct, 1),
                "peak_cpu": round(self.peak_cpu_util_pct, 1),
                "peak_gpu": round(self.peak_gpu_util_pct, 1),
            },
            "pipeline": {
                "estimated_transfer_ms": round(self.estimated_transfer_ms, 2),
                "pipeline_stalls": self.pipeline_stalls,
                "compute_time_ms": round(self.compute_time_ms, 2),
            },
            "session": {
                "n_gpu_layers": self.n_gpu_layers,
                "context_length": self.context_length,
                "backend": self.backend,
                "model_name": self.model_name,
                "is_feasible": self.is_feasible,
            },
        }

    def summary_line(self) -> str:
        """One-line human-readable summary."""
        return (
            f"{self.model_name}  |  {self.eval_tps:.1f} tok/s  |  "
            f"GPU {self.avg_gpu_util_pct:.0f}%  CPU {self.avg_cpu_util_pct:.0f}%  |  "
            f"stalls={self.pipeline_stalls}  |  backend={self.backend}"
        )


# ---------------------------------------------------------------------------
# StatsCollector
# ---------------------------------------------------------------------------

class StatsCollector:
    """
    Collects runtime statistics via two parallel tracks: llama.cpp stderr
    parsing and OS-level background sampling.

    Parameters
    ----------
    sample_interval_ms : float
        Interval between OS utilization polls in milliseconds. Default 250.
    n_gpu_layers : int
        GPU layer count for metadata annotation.
    context_length : int
        Context window for metadata annotation.
    backend : str
        Backend name for metadata annotation.
    model_name : str
        Model name for metadata annotation.
    boundary_crossings : int
        From PlacementPlan; used to estimate PCIe transfer time.
    pcie_bandwidth_gbps : float
        PCIe bandwidth from hardware profile for transfer estimation.
    """

    # Regex patterns for llama.cpp timing output
    _RE_PROMPT_EVAL = re.compile(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?"
        r"([\d.]+)\s*tokens per second",
        re.IGNORECASE,
    )
    _RE_EVAL = re.compile(
        r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens).*?"
        r"([\d.]+)\s*tokens per second",
        re.IGNORECASE,
    )
    # Alternative single-line format
    _RE_EVAL_ALT = re.compile(
        r"^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(
        self,
        sample_interval_ms: float = 250.0,
        n_gpu_layers: int = 0,
        context_length: int = 4096,
        backend: str = "unknown",
        model_name: str = "",
        boundary_crossings: int = 0,
        pcie_bandwidth_gbps: float = 16.0,
        is_feasible: bool = True,
    ) -> None:
        self.sample_interval_ms = sample_interval_ms
        self.n_gpu_layers = n_gpu_layers
        self.context_length = context_length
        self.backend = backend
        self.model_name = model_name
        self.boundary_crossings = boundary_crossings
        self.pcie_bandwidth_gbps = pcie_bandwidth_gbps
        self.is_feasible = is_feasible

        # OS sample storage (filled by background thread)
        self._cpu_samples: List[float] = []
        self._gpu_samples: List[float] = []
        self._stall_count: int = 0
        self._sampling_active: bool = False
        self._sample_thread: Optional[threading.Thread] = None

        # Per-token timestamps (filled by ProcessManager callbacks)
        self._token_timestamps: List[float] = []

        # Wall-clock timing
        self._wall_start: float = 0.0
        self._wall_end: float = 0.0

    # ---------------------------------------------------------------------------
    # OS sampling thread
    # ---------------------------------------------------------------------------

    def start_sampling(self) -> None:
        """Start the background OS utilization sampling thread."""
        self._sampling_active = True
        self._wall_start = time.perf_counter()
        self._sample_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="InferenceOSStatsSampler",
        )
        self._sample_thread.start()

    def stop_sampling(self) -> None:
        """Stop the background sampling thread and record wall-clock end."""
        self._sampling_active = False
        self._wall_end = time.perf_counter()
        if self._sample_thread and self._sample_thread.is_alive():
            self._sample_thread.join(timeout=2.0)

    def record_token(self) -> None:
        """Record the arrival timestamp of a generated token."""
        self._token_timestamps.append(time.perf_counter())

    def _sampling_loop(self) -> None:
        """Background loop: poll CPU + GPU utilization every interval."""
        interval_sec = self.sample_interval_ms / 1000.0

        # Try to import optional GPU monitoring library
        _gputil_available = False
        try:
            import GPUtil  # type: ignore
            _gputil_available = True
        except ImportError:
            pass

        try:
            import psutil
            _psutil_available = True
        except ImportError:
            _psutil_available = False

        win_gpu_mon = WindowsGpuMonitor()
        try:
            while self._sampling_active:
                cpu_pct = 0.0
                gpu_pct = 0.0

                if _psutil_available:
                    try:
                        import psutil
                        cpu_pct = psutil.cpu_percent(interval=None)
                    except Exception:
                        pass

                # 1. Native Windows Performance Counter (PDH) for AMD/NVIDIA/Intel
                val = win_gpu_mon.sample()
                if val >= 0.0:
                    gpu_pct = val
                elif _gputil_available:
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_pct = gpus[0].load * 100.0
                    except Exception:
                        pass

                self._cpu_samples.append(cpu_pct)
                self._gpu_samples.append(gpu_pct)

                # Detect pipeline stall: CPU saturated, GPU idle
                if cpu_pct > 95.0 and gpu_pct < 10.0:
                    self._stall_count += 1

                time.sleep(interval_sec)
        finally:
            win_gpu_mon.close()

    # ---------------------------------------------------------------------------
    # Track A: stderr parsing
    # ---------------------------------------------------------------------------

    _RE_BRACKET_TIMING = re.compile(
        r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
        re.IGNORECASE,
    )

    def parse_stderr(self, stderr: str) -> dict:
        """
        Parse llama.cpp timing output from stderr/stdout text.

        Extracts prompt eval and generation eval timing lines.

        Parameters
        ----------
        stderr : str
            Full stderr/stdout output from the llama.exe subprocess.

        Returns
        -------
        dict
            Parsed timing values: ``prompt_eval_ms``, ``prompt_eval_tokens``,
            ``prompt_eval_tps``, ``eval_ms``, ``eval_tokens``, ``eval_tps``.
        """
        result = {
            "prompt_eval_ms": 0.0,
            "prompt_eval_tokens": 0,
            "prompt_eval_tps": 0.0,
            "eval_ms": 0.0,
            "eval_tokens": 0,
            "eval_tps": 0.0,
        }

        for line in stderr.splitlines():
            # Bracket timing format: [ Prompt: 129.1 t/s | Generation: 50.4 t/s ]
            m_br = self._RE_BRACKET_TIMING.search(line)
            if m_br:
                result["prompt_eval_tps"] = float(m_br.group(1))
                result["eval_tps"] = float(m_br.group(2))
                continue

            # Prompt eval line
            m = self._RE_PROMPT_EVAL.search(line)
            if m:
                result["prompt_eval_ms"] = float(m.group(1))
                result["prompt_eval_tokens"] = int(m.group(2))
                result["prompt_eval_tps"] = float(m.group(3))
                continue

            # Skip lines that contain "prompt" and "eval" together
            if "prompt" in line.lower():
                continue

            # Generation eval line
            m = self._RE_EVAL.search(line)
            if m:
                result["eval_ms"] = float(m.group(1))
                result["eval_tokens"] = int(m.group(2))
                result["eval_tps"] = float(m.group(3))
                continue

            # Fallback alt pattern (some llama.cpp versions)
            m = self._RE_EVAL_ALT.search(line)
            if m and result["eval_ms"] == 0.0:
                result["eval_ms"] = float(m.group(1))
                result["eval_tokens"] = int(m.group(2))
                if result["eval_ms"] > 0 and result["eval_tokens"] > 0:
                    result["eval_tps"] = (result["eval_tokens"] / result["eval_ms"]) * 1000.0

        return result

    # ---------------------------------------------------------------------------
    # Finalization: build RuntimeStats
    # ---------------------------------------------------------------------------

    def finalize(self, stderr: str, total_wall_ms: Optional[float] = None) -> RuntimeStats:
        """
        Build the final :class:`RuntimeStats` by merging Track A (stderr)
        and Track B (OS samples) with per-token latency data.

        Parameters
        ----------
        stderr : str
            Full stderr from the subprocess.
        total_wall_ms : float, optional
            Total wall-clock time in milliseconds. If None, derived from
            internal timestamps.

        Returns
        -------
        RuntimeStats
            Fully populated statistics object.
        """
        # Track A: parse llama.cpp output
        parsed = self.parse_stderr(stderr)

        if parsed["eval_tokens"] == 0 and len(self._token_timestamps) > 0:
            parsed["eval_tokens"] = len(self._token_timestamps)

        if parsed["eval_tps"] > 0 and parsed["eval_ms"] == 0.0 and parsed["eval_tokens"] > 0:
            parsed["eval_ms"] = (parsed["eval_tokens"] / parsed["eval_tps"]) * 1000.0
        elif parsed["eval_ms"] == 0.0 and total_wall_ms and parsed["eval_tokens"] > 0:
            parsed["eval_ms"] = total_wall_ms
            if parsed["eval_tps"] == 0.0:
                parsed["eval_tps"] = (parsed["eval_tokens"] / total_wall_ms) * 1000.0

        # Wall time
        if total_wall_ms is None:
            total_wall_ms = (self._wall_end - self._wall_start) * 1000.0

        # Track B: OS utilization averages
        cpu_samples = self._cpu_samples or [0.0]
        gpu_samples = self._gpu_samples or [0.0]

        avg_cpu = statistics.mean(cpu_samples)
        avg_gpu = statistics.mean(gpu_samples)
        peak_cpu = max(cpu_samples)
        peak_gpu = max(gpu_samples)

        # Per-token latency distribution
        latency_ms: List[float] = []
        if len(self._token_timestamps) >= 2:
            latency_ms = [
                (self._token_timestamps[i] - self._token_timestamps[i - 1]) * 1000.0
                for i in range(1, len(self._token_timestamps))
            ]

        p50, p95, p99 = self._percentiles(latency_ms)

        # Transfer time estimate (PCIe boundary crossings × activation transfer time)
        # Activation size ≈ 8192 × 2 bytes (fp16, typical large model hidden dim)
        activation_bytes = 8192 * 2
        pcie_bps = self.pcie_bandwidth_gbps * 1024 ** 3
        estimated_transfer_ms = (
            self.boundary_crossings * activation_bytes / max(pcie_bps, 1)
        ) * 1000.0

        eval_ms = parsed["eval_ms"]
        compute_time_ms = max(0.0, eval_ms - estimated_transfer_ms)

        return RuntimeStats(
            prompt_eval_tps=parsed["prompt_eval_tps"],
            eval_tps=parsed["eval_tps"],
            tokens_generated=parsed["eval_tokens"],
            prompt_eval_ms=parsed["prompt_eval_ms"],
            eval_ms=eval_ms,
            total_wall_ms=total_wall_ms,
            per_token_latency_ms=latency_ms,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_cpu_util_pct=avg_cpu,
            avg_gpu_util_pct=avg_gpu,
            peak_cpu_util_pct=peak_cpu,
            peak_gpu_util_pct=peak_gpu,
            estimated_transfer_ms=estimated_transfer_ms,
            pipeline_stalls=self._stall_count,
            compute_time_ms=compute_time_ms,
            n_gpu_layers=self.n_gpu_layers,
            context_length=self.context_length,
            backend=self.backend,
            model_name=self.model_name,
            is_feasible=self.is_feasible,
        )

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _percentiles(values: List[float]) -> tuple:
        """Return (p50, p95, p99) for a list of float values."""
        if not values:
            return (0.0, 0.0, 0.0)
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def _pct(p: float) -> float:
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

        return (_pct(50), _pct(95), _pct(99))
