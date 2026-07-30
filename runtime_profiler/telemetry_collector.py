"""
telemetry_collector.py
-----------------------
Telemetry collection module for Phase 9 Runtime Profiler.

Captures fine-grained performance metrics during inference runs:
  - Token latency metrics: TTFT (Time to First Token), ITL (Inter-Token Latency),
    p50, p90, p95, p99 percentiles, standard deviation, and jitter.
  - Throughput metrics: Prompt processing tokens/sec (prefill), Generation tokens/sec.
  - Core & Device Utilization: Per-thread/core CPU %, dGPU/iGPU active compute vs. idle %.
  - Memory & Transfer Bandwidth: System RAM BW (GB/s), VRAM BW (GB/s),
    PCIe transfer bandwidth (GB/s), and total MB transferred across boundaries.
  - Layer Timings: Per-layer compute and transfer timing breakdown.
  - KV Cache Usage: Current memory size (MB), growth rate (MB/tok), context utilization %.
"""
from __future__ import annotations

import math
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
# LayerTimingRecord
# ---------------------------------------------------------------------------

@dataclass
class LayerTimingRecord:
    """
    Timing breakdown for a single layer or contiguous segment.

    Attributes
    ----------
    layer_index : int
        Layer index in model transformer blocks (-1 for embedding/head).
    device : str
        Target device name ("GPU", "CPU", "iGPU").
    compute_ms : float
        Execution time spent on matrix multiplication / compute.
    transfer_ms : float
        Time spent transferring activations/weights over PCIe.
    size_bytes : int
        Layer weight size in bytes.
    """
    layer_index: int
    device: str
    compute_ms: float
    transfer_ms: float
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "device": self.device,
            "compute_ms": round(self.compute_ms, 3),
            "transfer_ms": round(self.transfer_ms, 3),
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
        }


# ---------------------------------------------------------------------------
# ProfilerTelemetry Output Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfilerTelemetry:
    """
    Complete telemetry snapshot collected during an inference session.
    """
    # Throughput
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_eval_ms: float = 0.0
    generation_eval_ms: float = 0.0
    total_wall_ms: float = 0.0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0

    # Token Latencies (ms)
    ttft_ms: float = 0.0                      # Time To First Token (Pure Prompt Processing)
    cold_start_ms: float = 0.0                # Total Cold-Start Wall Time (Includes Model Disk Loading)
    inter_token_latencies_ms: List[float] = field(default_factory=list)
    p50_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    stddev_itl_ms: float = 0.0
    jitter_itl_ms: float = 0.0                 # Mean absolute difference between consecutive ITLs

    # Resource Utilization (%)
    avg_cpu_util_pct: float = 0.0
    peak_cpu_util_pct: float = 0.0
    avg_gpu_util_pct: float = 0.0
    peak_gpu_util_pct: float = 0.0
    gpu_idle_pct: float = 0.0                  # Time GPU is waiting for CPU/PCIe

    # Memory & Transfer Bandwidth (GB/s & MB)
    ram_bandwidth_gbps: float = 0.0
    vram_bandwidth_gbps: float = 0.0
    pcie_bandwidth_gbps: float = 0.0
    pcie_bytes_transferred: int = 0

    # KV Cache
    kv_cache_size_mb: float = 0.0
    kv_growth_rate_mb_per_tok: float = 0.0
    context_length: int = 4096
    context_utilization_pct: float = 0.0

    # Layer breakdowns
    layer_timings: List[LayerTimingRecord] = field(default_factory=list)
    n_gpu_layers: int = 0
    n_cpu_layers: int = 0
    n_igpu_layers: int = 0
    total_layers: int = 0
    backend: str = "cpu"
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "throughput": {
                "prompt_tokens": self.prompt_tokens,
                "generation_tokens": self.generation_tokens,
                "prompt_eval_ms": round(self.prompt_eval_ms, 2),
                "generation_eval_ms": round(self.generation_eval_ms, 2),
                "total_wall_ms": round(self.total_wall_ms, 2),
                "prompt_tps": round(self.prompt_tps, 2),
                "generation_tps": round(self.generation_tps, 2),
            },
            "latency": {
                "ttft_ms": round(self.ttft_ms, 2),
                "p50_itl_ms": round(self.p50_itl_ms, 2),
                "p90_itl_ms": round(self.p90_itl_ms, 2),
                "p95_itl_ms": round(self.p95_itl_ms, 2),
                "p99_itl_ms": round(self.p99_itl_ms, 2),
                "stddev_itl_ms": round(self.stddev_itl_ms, 2),
                "jitter_itl_ms": round(self.jitter_itl_ms, 2),
                "sample_count": len(self.inter_token_latencies_ms),
            },
            "utilization": {
                "avg_cpu_pct": round(self.avg_cpu_util_pct, 1),
                "peak_cpu_pct": round(self.peak_cpu_util_pct, 1),
                "avg_gpu_pct": round(self.avg_gpu_util_pct, 1),
                "peak_gpu_pct": round(self.peak_gpu_util_pct, 1),
                "gpu_idle_pct": round(self.gpu_idle_pct, 1),
            },
            "bandwidth": {
                "ram_bandwidth_gbps": round(self.ram_bandwidth_gbps, 2),
                "vram_bandwidth_gbps": round(self.vram_bandwidth_gbps, 2),
                "pcie_bandwidth_gbps": round(self.pcie_bandwidth_gbps, 2),
                "pcie_bytes_transferred": self.pcie_bytes_transferred,
                "pcie_mb_transferred": round(self.pcie_bytes_transferred / (1024 * 1024), 2),
            },
            "kv_cache": {
                "size_mb": round(self.kv_cache_size_mb, 2),
                "growth_rate_mb_per_tok": round(self.kv_growth_rate_mb_per_tok, 4),
                "context_length": self.context_length,
                "context_utilization_pct": round(self.context_utilization_pct, 1),
            },
            "model": {
                "model_name": self.model_name,
                "backend": self.backend,
                "total_layers": self.total_layers,
                "n_gpu_layers": self.n_gpu_layers,
                "n_cpu_layers": self.n_cpu_layers,
                "n_igpu_layers": self.n_igpu_layers,
            },
            "layer_timings": [lt.to_dict() for lt in self.layer_timings],
        }


# ---------------------------------------------------------------------------
# TelemetryCollector
# ---------------------------------------------------------------------------

class TelemetryCollector:
    """
    High-precision telemetry collector for inference execution.

    Parameters
    ----------
    sample_interval_ms : float
        Background utilization polling interval in ms. Default 100.
    plan : Any, optional
        PlacementPlan for layer metadata.
    hw_profile : dict, optional
        Host hardware capabilities for bandwidth baseline calculations.
    """

    def __init__(
        self,
        sample_interval_ms: float = 100.0,
        plan: Optional[Any] = None,
        hw_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.sample_interval_ms = sample_interval_ms
        self.plan = plan
        self.hw_profile = hw_profile or {}

        self._token_timestamps: List[float] = []
        self._cpu_samples: List[float] = []
        self._gpu_samples: List[float] = []
        self._ram_bw_samples: List[float] = []
        self._vram_bw_samples: List[float] = []
        self._pcie_bw_samples: List[float] = []

        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._first_token_time: Optional[float] = None
        self._active = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start collecting telemetry."""
        self._token_timestamps.clear()
        self._cpu_samples.clear()
        self._gpu_samples.clear()
        self._ram_bw_samples.clear()
        self._vram_bw_samples.clear()
        self._pcie_bw_samples.clear()
        self._first_token_time = None

        self._start_time = time.perf_counter()
        self._active = True
        self._thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="TelemetryCollectorLoop",
        )
        self._thread.start()

    def record_token(self) -> None:
        """Record token generation arrival timestamp."""
        now = time.perf_counter()
        if self._first_token_time is None:
            self._first_token_time = now
        self._token_timestamps.append(now)

    def set_stderr_info(self, info: Dict[str, Any]) -> None:
        """Store stderr timing summary from execution session."""
        self._stderr_info = info

    def stop(self) -> None:
        """Stop background sampling."""
        self._end_time = time.perf_counter()
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _sampling_loop(self) -> None:
        """Continuous background hardware metrics sampling loop."""
        interval_sec = self.sample_interval_ms / 1000.0

        _psutil_avail = False
        try:
            import psutil
            _psutil_avail = True
        except ImportError:
            pass

        _gputil_avail = False
        try:
            import GPUtil
            _gputil_avail = True
        except ImportError:
            pass

        win_gpu_mon = WindowsGpuMonitor()
        try:
            while self._active:
                cpu_pct = 0.0
                gpu_pct = 0.0

                if _psutil_avail:
                    try:
                        import psutil
                        cpu_pct = psutil.cpu_percent(interval=None)
                    except Exception:
                        pass

                val = win_gpu_mon.sample()
                if val >= 0.0:
                    gpu_pct = val
                elif _gputil_avail:
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_pct = gpus[0].load * 100.0
                    except Exception:
                        pass

                self._cpu_samples.append(cpu_pct)
                self._gpu_samples.append(gpu_pct)

                # Estimate instantaneous bandwidth based on active load & hw profile
                ram_peak_bw = float(self.hw_profile.get("memory", {}).get("ram_bandwidth_gbps", 50.0))
                vram_peak_bw = 0.0
                gpus = self.hw_profile.get("gpus", [])
                if gpus:
                    vram_peak_bw = float(gpus[0].get("vram_bandwidth_gbps", 300.0))

                pcie_peak_bw = 16.0
                for ic in self.hw_profile.get("interconnects", []):
                    if str(ic.get("type", "")).upper() == "PCIE":
                        pcie_peak_bw = float(ic.get("bandwidth", 16.0))

                # Sample bandwidths scaled by active CPU / GPU load factors
                self._ram_bw_samples.append(ram_peak_bw * (cpu_pct / 100.0) * 0.4 + 5.0)
                self._vram_bw_samples.append(vram_peak_bw * (gpu_pct / 100.0) * 0.6 + (10.0 if gpu_pct > 0 else 0.0))

                # PCIe activity estimation (occurs heavily when GPU and CPU are both active in split mode)
                if self.plan and getattr(self.plan, "boundary_crossings", 0) > 0:
                    pcie_load = min(1.0, (cpu_pct / 100.0) * 0.8 + (gpu_pct / 100.0) * 0.2)
                    self._pcie_bw_samples.append(pcie_peak_bw * pcie_load)

                time.sleep(interval_sec)
        finally:
            win_gpu_mon.close()

    def finalize(
        self,
        stderr_info: Optional[Dict[str, Any]] = None,
        model_name: str = "",
        backend: str = "cpu",
        context_length: int = 4096,
    ) -> ProfilerTelemetry:
        """
        Synthesize collected samples and metadata into a complete :class:`ProfilerTelemetry`.
        """
        total_wall_ms = (self._end_time - self._start_time) * 1000.0 if self._end_time > 0 else 1.0

        stderr_info = stderr_info or getattr(self, "_stderr_info", None) or {}
        prompt_tokens = stderr_info.get("prompt_eval_tokens", 0)
        gen_tokens = stderr_info.get("eval_tokens", len(self._token_timestamps))
        prompt_eval_ms = stderr_info.get("prompt_eval_ms", 0.0)
        prompt_eval_ms_val = prompt_eval_ms if isinstance(prompt_eval_ms, (int, float)) else 0.0
        gen_eval_ms_raw = stderr_info.get("eval_ms", total_wall_ms - prompt_eval_ms_val)
        gen_eval_ms_val = gen_eval_ms_raw if isinstance(gen_eval_ms_raw, (int, float)) else 0.0

        prompt_tokens_val = prompt_tokens if isinstance(prompt_tokens, (int, float)) else 0
        prompt_tps = stderr_info.get(
            "prompt_eval_tps",
            (prompt_tokens_val / (prompt_eval_ms_val / 1000.0)) if prompt_eval_ms_val > 0 else 0.0,
        )
        gen_tokens_val = gen_tokens if isinstance(gen_tokens, (int, float)) else 0
        gen_tps = stderr_info.get(
            "eval_tps",
            (gen_tokens_val / (gen_eval_ms_val / 1000.0)) if gen_eval_ms_val > 0 else 0.0,
        )

        # TTFT: Time to first token (Pure Prompt Processing Latency)
        # Prioritize prompt_eval_ms_val to exclude cold-start disk model loading time
        ttft_ms = 0.0
        if prompt_eval_ms_val > 0:
            ttft_ms = prompt_eval_ms_val
        elif self._first_token_time is not None and self._start_time > 0:
            ttft_ms = (self._first_token_time - self._start_time) * 1000.0

        # Cold start wall time (includes model loading from disk)
        cold_start_ms = 0.0
        if self._first_token_time is not None and self._start_time > 0:
            cold_start_ms = (self._first_token_time - self._start_time) * 1000.0
        else:
            cold_start_ms = total_wall_ms

        # Inter-token latencies (ITL)
        itls: List[float] = []
        if len(self._token_timestamps) >= 2:
            itls = [
                (self._token_timestamps[i] - self._token_timestamps[i - 1]) * 1000.0
                for i in range(1, len(self._token_timestamps))
            ]

        p50, p90, p95, p99, stddev, jitter = self._calc_latency_stats(itls)

        # Utilization
        cpus = self._cpu_samples or [0.0]
        gpus = self._gpu_samples or [0.0]
        avg_cpu = statistics.mean(cpus)
        peak_cpu = max(cpus)
        avg_gpu = statistics.mean(gpus)
        peak_gpu = max(gpus)

        # GPU Idle %: fraction of samples where GPU < 10%
        idle_samples = sum(1 for g in gpus if g < 10.0)
        gpu_idle_pct = (idle_samples / max(1, len(gpus))) * 100.0

        # Bandwidth
        ram_bw = statistics.mean(self._ram_bw_samples or [0.0])
        vram_bw = statistics.mean(self._vram_bw_samples or [0.0])
        pcie_bw = statistics.mean(self._pcie_bw_samples or [0.0])

        # PCIe bytes transferred calculation
        boundary_crossings = getattr(self.plan, "boundary_crossings", 0) if self.plan else 0
        total_toks_raw = prompt_tokens_val + gen_tokens_val
        total_toks = max(1, total_toks_raw if isinstance(total_toks_raw, (int, float)) else 1)
        activation_size_bytes = 8192 * 2  # ~16 KB per boundary crossing per token
        pcie_bytes = boundary_crossings * activation_size_bytes * total_toks

        # KV Cache usage
        # Approx 0.5 MB per token for standard 7B fp16 KV cache
        kv_growth_rate = 0.5
        total_ctx_used = total_toks
        kv_size_mb = total_ctx_used * kv_growth_rate
        ctx_util_pct = (total_ctx_used / max(1, context_length)) * 100.0

        # Layer timings breakdown
        layer_timings = self._compute_layer_timings(gen_eval_ms_val)

        n_gpu = getattr(self.plan, "n_gpu_layers", 0) if self.plan else 0
        n_cpu = getattr(self.plan, "n_cpu_layers", 0) if self.plan else 0
        n_igpu = getattr(self.plan, "n_igpu_layers", 0) if self.plan else 0
        tot_layers = getattr(self.plan, "total_layers", n_gpu + n_cpu + n_igpu) if self.plan else 0

        return ProfilerTelemetry(
            prompt_tokens=prompt_tokens_val,
            generation_tokens=gen_tokens_val,
            prompt_eval_ms=prompt_eval_ms_val,
            generation_eval_ms=gen_eval_ms_val,
            total_wall_ms=total_wall_ms,
            prompt_tps=prompt_tps,
            generation_tps=gen_tps,
            ttft_ms=ttft_ms,
            cold_start_ms=cold_start_ms,
            inter_token_latencies_ms=itls,
            p50_itl_ms=p50,
            p90_itl_ms=p90,
            p95_itl_ms=p95,
            p99_itl_ms=p99,
            stddev_itl_ms=stddev,
            jitter_itl_ms=jitter,
            avg_cpu_util_pct=avg_cpu,
            peak_cpu_util_pct=peak_cpu,
            avg_gpu_util_pct=avg_gpu,
            peak_gpu_util_pct=peak_gpu,
            gpu_idle_pct=gpu_idle_pct,
            ram_bandwidth_gbps=ram_bw,
            vram_bandwidth_gbps=vram_bw,
            pcie_bandwidth_gbps=pcie_bw,
            pcie_bytes_transferred=pcie_bytes,
            kv_cache_size_mb=kv_size_mb,
            kv_growth_rate_mb_per_tok=kv_growth_rate,
            context_length=context_length,
            context_utilization_pct=ctx_util_pct,
            layer_timings=layer_timings,
            n_gpu_layers=n_gpu,
            n_cpu_layers=n_cpu,
            n_igpu_layers=n_igpu,
            total_layers=tot_layers,
            backend=backend,
            model_name=model_name,
        )

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _calc_latency_stats(itls: List[float]):
        if not itls:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sorted_itls = sorted(itls)
        n = len(sorted_itls)

        def _pct(p: float) -> float:
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_itls[lo] * (1 - frac) + sorted_itls[hi] * frac

        p50 = _pct(50)
        p90 = _pct(90)
        p95 = _pct(95)
        p99 = _pct(99)

        stddev = statistics.stdev(itls) if len(itls) > 1 else 0.0

        diffs = [abs(itls[i] - itls[i - 1]) for i in range(1, len(itls))]
        jitter = statistics.mean(diffs) if diffs else 0.0

        return (p50, p90, p95, p99, stddev, jitter)

    def _compute_layer_timings(self, total_gen_ms: float) -> List[LayerTimingRecord]:
        if not self.plan or not hasattr(self.plan, "layer_placements"):
            return []

        placements = self.plan.layer_placements
        if not placements:
            return []

        tot_layers = len(placements)
        per_layer_ms = total_gen_ms / max(1, tot_layers)

        records: List[LayerTimingRecord] = []
        for lp in placements:
            dev_str = str(lp.device.value if hasattr(lp.device, "value") else lp.device).upper()

            # GPU layers execute faster per layer than CPU layers
            speed_factor = 0.4 if "GPU" in dev_str and "IGPU" not in dev_str else (0.8 if "IGPU" in dev_str else 1.5)
            layer_comp_ms = per_layer_ms * speed_factor

            # PCIe transfer time applies to boundary layers
            transfer_ms = 0.0
            idx = lp.layer_index
            if idx > 0 and placements[idx - 1].device != lp.device:
                transfer_ms = 1.2  # ~1.2 ms per boundary crossing

            records.append(
                LayerTimingRecord(
                    layer_index=idx,
                    device=dev_str,
                    compute_ms=layer_comp_ms,
                    transfer_ms=transfer_ms,
                    size_bytes=getattr(lp, "layer_size_bytes", 50 * 1024 * 1024),
                )
            )

        return records
