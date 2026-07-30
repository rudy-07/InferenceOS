"""
performance.py
--------------
Performance Knowledge Base for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .interfaces import ExecutionRecord


@dataclass
class PerformanceSummary:
    total_runs: int = 0
    avg_tps: float = 0.0
    best_tps: float = 0.0
    worst_tps: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_latency_ms: float = 0.0
    avg_utilization_pct: float = 0.0
    regression_detected: bool = False


class PerformanceKnowledgeBase:
    """
    Maintains aggregated runtime performance statistics and regression tracking.
    """

    def __init__(self) -> None:
        self.summary = PerformanceSummary()

    def update_from_records(self, records: List[ExecutionRecord]) -> None:
        if not records:
            return

        valid = [r for r in records if r.success and r.eval_tps > 0]
        if not valid:
            return

        n = len(valid)
        tps_list = [r.eval_tps for r in valid]
        ttft_list = [r.ttft_ms for r in valid]
        lat_list = [r.latency_ms for r in valid]
        util_list = [r.gpu_utilization_pct for r in valid]

        self.summary = PerformanceSummary(
            total_runs=n,
            avg_tps=round(sum(tps_list) / n, 2),
            best_tps=round(max(tps_list), 2),
            worst_tps=round(min(tps_list), 2),
            avg_ttft_ms=round(sum(ttft_list) / n, 1),
            avg_latency_ms=round(sum(lat_list) / n, 1),
            avg_utilization_pct=round(sum(util_list) / n, 1),
            regression_detected=False,
        )

    def get_summary(self) -> PerformanceSummary:
        return self.summary
