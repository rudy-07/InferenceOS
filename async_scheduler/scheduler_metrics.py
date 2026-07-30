"""
scheduler_metrics.py
--------------------
Performance metrics aggregation for Phase 8 Async Execution Scheduler.

This module collects, aggregates, and reports the scheduler-level
performance data that shows whether async pipelining improved throughput.

Key metrics
-----------
gpu_idle_pct
    Fraction of total wall time where the GPU was measured as idle
    (utilization < 10%). Lower is better.

pcie_latency_hidden_ms
    Estimated PCIe transfer time that was hidden by overlapping with
    CPU/GPU computation. Higher is better.

bubble_rate_pct
    Fraction of pipeline cycles where a stage was idle waiting for the
    previous stage. 0% = perfectly pipelined.

speedup_estimate
    Estimated throughput improvement ratio vs. synchronous sequential
    execution. Computed from pipeline timing: speedup = sequential_time /
    pipelined_time.

tokens_per_sec
    Mean token generation rate (tokens/s) across all measured requests.

Comparison report
-----------------
``SchedulerReport.comparison_table()`` outputs a human-readable side-by-side
table comparing synchronous and async performance for benchmarking.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# StageTimings — per-stage timing record for one request
# ---------------------------------------------------------------------------

@dataclass
class StageTimings:
    """
    Timing breakdown for one request through the 3-stage pipeline.

    All times in milliseconds.
    """
    request_id: int = 0
    prepare_ms: float = 0.0     # Stage 1: CPU prepare (tokenize, prefetch schedule)
    transfer_ms: float = 0.0    # Stage 2: PCIe / prefetch I/O
    compute_ms: float = 0.0     # Stage 3: llama.cpp GPU/CPU compute
    total_ms: float = 0.0       # Wall time from submit to done

    # Stage idle times (time waiting for previous stage)
    prepare_idle_ms: float = 0.0
    transfer_idle_ms: float = 0.0
    compute_idle_ms: float = 0.0

    tokens_generated: int = 0
    eval_tps: float = 0.0

    def pipeline_efficiency(self) -> float:
        """
        Fraction of total time spent doing actual work (not waiting).
        1.0 = perfectly pipelined.
        """
        idle = self.prepare_idle_ms + self.transfer_idle_ms + self.compute_idle_ms
        if self.total_ms <= 0:
            return 1.0
        return max(0.0, 1.0 - (idle / self.total_ms))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prepare_ms": round(self.prepare_ms, 2),
            "transfer_ms": round(self.transfer_ms, 2),
            "compute_ms": round(self.compute_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "prepare_idle_ms": round(self.prepare_idle_ms, 2),
            "transfer_idle_ms": round(self.transfer_idle_ms, 2),
            "compute_idle_ms": round(self.compute_idle_ms, 2),
            "tokens_generated": self.tokens_generated,
            "eval_tps": round(self.eval_tps, 2),
            "pipeline_efficiency": round(self.pipeline_efficiency(), 4),
        }


# ---------------------------------------------------------------------------
# SchedulerMetrics — aggregate across all requests
# ---------------------------------------------------------------------------

@dataclass
class SchedulerMetrics:
    """
    Aggregated scheduler performance metrics across all processed requests.

    Attributes
    ----------
    total_requests : int
        Number of requests processed.
    total_tokens : int
        Total tokens generated across all requests.
    mean_tokens_per_sec : float
        Mean token generation rate (tok/s).
    peak_tokens_per_sec : float
        Best single-request token rate.
    gpu_idle_pct : float
        Estimated GPU idle fraction across all requests (0.0–1.0 → 0%–100%).
    pcie_latency_hidden_ms : float
        Total estimated PCIe latency hidden by async overlap.
    bubble_rate_pct : float
        Mean pipeline bubble fraction (idle / total time).
    speedup_estimate : float
        Estimated throughput ratio vs. sequential baseline.
        1.0 = no improvement; > 1.0 = faster.
    mean_prepare_ms : float
        Mean Stage 1 (CPU prepare) time.
    mean_transfer_ms : float
        Mean Stage 2 (PCIe transfer) time.
    mean_compute_ms : float
        Mean Stage 3 (GPU compute) time.
    mean_pipeline_efficiency : float
        Mean pipeline efficiency (1 - bubble rate).
    per_request : List[StageTimings]
        Per-request timing records.
    """
    total_requests: int = 0
    total_tokens: int = 0
    mean_tokens_per_sec: float = 0.0
    peak_tokens_per_sec: float = 0.0
    gpu_idle_pct: float = 0.0
    pcie_latency_hidden_ms: float = 0.0
    bubble_rate_pct: float = 0.0
    speedup_estimate: float = 1.0
    mean_prepare_ms: float = 0.0
    mean_transfer_ms: float = 0.0
    mean_compute_ms: float = 0.0
    mean_pipeline_efficiency: float = 1.0
    per_request: List[StageTimings] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "mean_tokens_per_sec": round(self.mean_tokens_per_sec, 2),
            "peak_tokens_per_sec": round(self.peak_tokens_per_sec, 2),
            "gpu_idle_pct": round(self.gpu_idle_pct, 2),
            "pcie_latency_hidden_ms": round(self.pcie_latency_hidden_ms, 2),
            "bubble_rate_pct": round(self.bubble_rate_pct, 2),
            "speedup_estimate": round(self.speedup_estimate, 4),
            "mean_prepare_ms": round(self.mean_prepare_ms, 2),
            "mean_transfer_ms": round(self.mean_transfer_ms, 2),
            "mean_compute_ms": round(self.mean_compute_ms, 2),
            "mean_pipeline_efficiency": round(self.mean_pipeline_efficiency, 4),
        }


# ---------------------------------------------------------------------------
# SchedulerReport
# ---------------------------------------------------------------------------

class SchedulerReport:
    """
    Human-readable report generator for scheduler metrics.

    Parameters
    ----------
    metrics : SchedulerMetrics
        Aggregated metrics to report.
    sequential_tps : float, optional
        Baseline tokens/sec from synchronous execution (for comparison).
    """

    def __init__(
        self,
        metrics: SchedulerMetrics,
        sequential_tps: float = 0.0,
    ) -> None:
        self.metrics = metrics
        self.sequential_tps = sequential_tps

    def summary(self) -> str:
        """One-line summary string."""
        m = self.metrics
        improvement = ""
        if self.sequential_tps > 0 and m.mean_tokens_per_sec > 0:
            delta_pct = (m.mean_tokens_per_sec / self.sequential_tps - 1.0) * 100.0
            sign = "+" if delta_pct >= 0 else ""
            improvement = f"  ({sign}{delta_pct:.1f}% vs sequential)"
        return (
            f"Async Scheduler: {m.mean_tokens_per_sec:.1f} tok/s"
            f"{improvement}  "
            f"[pipeline={m.mean_pipeline_efficiency*100:.0f}%  "
            f"bubbles={m.bubble_rate_pct:.0f}%  "
            f"pcie_hidden={m.pcie_latency_hidden_ms:.0f}ms]"
        )

    def full_report(self) -> str:
        """Multi-line detailed report."""
        m = self.metrics
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║     InferenceOS  ·  Phase 8  Async Scheduler Report          ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  Requests Processed:    {m.total_requests}",
            f"  Total Tokens:          {m.total_tokens}",
            "",
            "THROUGHPUT",
            "──────────────────────────────────────────────────────────────",
            f"  Mean (async):          {m.mean_tokens_per_sec:.2f} tok/s",
            f"  Peak (async):          {m.peak_tokens_per_sec:.2f} tok/s",
        ]
        if self.sequential_tps > 0:
            delta = m.mean_tokens_per_sec - self.sequential_tps
            pct = (delta / max(self.sequential_tps, 1e-9)) * 100.0
            lines += [
                f"  Sequential baseline:   {self.sequential_tps:.2f} tok/s",
                f"  Speedup estimate:      {m.speedup_estimate:.3f}×  "
                f"({'+' if pct >= 0 else ''}{pct:.1f}%)",
            ]
        lines += [
            "",
            "PIPELINE TIMING (mean per request)",
            "──────────────────────────────────────────────────────────────",
            f"  Stage 1 Prepare:       {m.mean_prepare_ms:.1f} ms",
            f"  Stage 2 Transfer:      {m.mean_transfer_ms:.1f} ms",
            f"  Stage 3 Compute:       {m.mean_compute_ms:.1f} ms",
            f"  Pipeline Efficiency:   {m.mean_pipeline_efficiency * 100:.1f}%",
            f"  Bubble Rate:           {m.bubble_rate_pct:.1f}%",
            "",
            "OVERLAP ANALYSIS",
            "──────────────────────────────────────────────────────────────",
            f"  GPU Idle:              {m.gpu_idle_pct:.1f}%",
            f"  PCIe Latency Hidden:   {m.pcie_latency_hidden_ms:.1f} ms",
            "",
        ]
        return "\n".join(lines)

    def comparison_table(self) -> str:
        """Compact comparison table for benchmark output."""
        m = self.metrics
        if self.sequential_tps <= 0:
            return self.summary()
        delta_pct = (m.mean_tokens_per_sec / max(self.sequential_tps, 1e-9) - 1.0) * 100.0
        sign = "+" if delta_pct >= 0 else ""
        return "\n".join([
            "  ┌─────────────────────────────────────────────────────┐",
            "  │  Execution Mode     Throughput    Pipeline Eff      │",
            "  ├─────────────────────────────────────────────────────┤",
            f"  │  Sequential        {self.sequential_tps:8.1f} tok/s   {'N/A':>10}       │",
            f"  │  Async (Phase 8)   {m.mean_tokens_per_sec:8.1f} tok/s   {m.mean_pipeline_efficiency*100:8.1f}%       │",
            f"  │  Improvement       {sign}{delta_pct:7.1f}%         bubbles: {m.bubble_rate_pct:.0f}%  │",
            "  └─────────────────────────────────────────────────────┘",
        ])


# ---------------------------------------------------------------------------
# MetricsAccumulator — collects per-request timings and computes aggregates
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """
    Collects per-request :class:`StageTimings` and computes
    aggregate :class:`SchedulerMetrics`.

    Thread-safe: multiple workers may call ``record()`` concurrently.
    """

    def __init__(self) -> None:
        self._records: List[StageTimings] = []
        self._lock = __import__("threading").Lock()
        self._start_time = time.perf_counter()

    def record(self, timings: StageTimings) -> None:
        """Record stage timings for one completed request."""
        with self._lock:
            self._records.append(timings)

    def compute(
        self,
        sequential_tps: float = 0.0,
        pcie_bw_gbps: float = 16.0,
        boundary_crossings: int = 0,
    ) -> SchedulerMetrics:
        """
        Compute aggregate SchedulerMetrics from all recorded timings.

        Parameters
        ----------
        sequential_tps : float
            Baseline tok/s for speedup calculation.
        pcie_bw_gbps : float
            PCIe bandwidth for PCIe latency hidden estimate.
        boundary_crossings : int
            Number of GPU↔CPU boundaries in the plan.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return SchedulerMetrics()

        tps_list = [r.eval_tps for r in records]
        prepare_list = [r.prepare_ms for r in records]
        transfer_list = [r.transfer_ms for r in records]
        compute_list = [r.compute_ms for r in records]
        eff_list = [r.pipeline_efficiency() for r in records]

        mean_tps = statistics.mean(tps_list) if tps_list else 0.0
        peak_tps = max(tps_list) if tps_list else 0.0

        mean_eff = statistics.mean(eff_list) if eff_list else 1.0
        bubble_rate = (1.0 - mean_eff) * 100.0

        # GPU idle estimate: average across records
        # We cannot directly observe GPU idle in subprocess mode, so we
        # estimate from pipeline bubble time = fraction of time GPU stage
        # was waiting for transfer stage.
        total_idle = sum(r.compute_idle_ms for r in records)
        total_compute = sum(r.compute_ms for r in records)
        gpu_idle_pct = (total_idle / max(total_compute + total_idle, 1e-6)) * 100.0

        # PCIe latency hidden = total transfer time that overlapped with compute
        # = min(transfer_ms, compute_ms) per request, summed
        pcie_hidden = sum(
            min(r.transfer_ms, r.compute_ms)
            for r in records
        )

        # Speedup estimate
        speedup = mean_tps / max(sequential_tps, 1e-9) if sequential_tps > 0 else 1.0

        return SchedulerMetrics(
            total_requests=len(records),
            total_tokens=sum(r.tokens_generated for r in records),
            mean_tokens_per_sec=mean_tps,
            peak_tokens_per_sec=peak_tps,
            gpu_idle_pct=gpu_idle_pct,
            pcie_latency_hidden_ms=pcie_hidden,
            bubble_rate_pct=bubble_rate,
            speedup_estimate=speedup,
            mean_prepare_ms=statistics.mean(prepare_list) if prepare_list else 0.0,
            mean_transfer_ms=statistics.mean(transfer_list) if transfer_list else 0.0,
            mean_compute_ms=statistics.mean(compute_list) if compute_list else 0.0,
            mean_pipeline_efficiency=mean_eff,
            per_request=records,
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._start_time = time.perf_counter()
