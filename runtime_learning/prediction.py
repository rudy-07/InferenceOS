"""
prediction.py
-------------
Performance prediction, regression detection, and self-correction engine for Runtime Learning in InferenceOS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .history import ExecutionRecord

logger = logging.getLogger("InferenceOS.RuntimeLearning.Predictor")


@dataclass
class RegressionReport:
    """Report detailing a detected performance regression."""
    is_regression: bool
    drop_percentage: float
    previous_avg_tps: float
    recent_avg_tps: float
    message: str


class PerformancePredictor:
    """
    Performance prediction and regression detection engine.
    """

    @staticmethod
    def predict_metrics(records: List[ExecutionRecord]) -> Dict[str, float]:
        """
        Calculate weighted moving averages with outlier rejection.
        """
        if not records:
            return {
                "expected_prompt_tps": 0.0,
                "expected_eval_tps": 0.0,
                "expected_ttft_ms": 0.0,
                "expected_vram_mb": 0.0,
                "expected_ram_mb": 0.0,
                "expected_gpu_util": 0.0,
                "expected_cpu_util": 0.0,
            }

        # Filter out extreme outliers (bottom 5% / top 5% if sample >= 10)
        valid_eval = sorted([r.eval_tps for r in records if r.eval_tps > 0])
        if len(valid_eval) >= 10:
            trim = max(1, int(len(valid_eval) * 0.05))
            valid_eval = valid_eval[trim:-trim]

        eval_tps = sum(valid_eval) / max(1, len(valid_eval)) if valid_eval else 0.0

        prompt_vals = [r.prompt_tps for r in records if r.prompt_tps > 0]
        prompt_tps = sum(prompt_vals) / max(1, len(prompt_vals)) if prompt_vals else 0.0

        ttft_vals = [r.ttft_ms for r in records if r.ttft_ms > 0]
        ttft_ms = sum(ttft_vals) / max(1, len(ttft_vals)) if ttft_vals else 0.0

        vram_vals = [r.vram_used_mb for r in records if r.vram_used_mb > 0]
        vram_mb = sum(vram_vals) / max(1, len(vram_vals)) if vram_vals else 0.0

        ram_vals = [r.ram_used_mb for r in records if r.ram_used_mb > 0]
        ram_mb = sum(ram_vals) / max(1, len(ram_vals)) if ram_vals else 0.0

        gpu_util = sum(r.gpu_utilization_pct for r in records) / len(records)
        cpu_util = sum(r.cpu_utilization_pct for r in records) / len(records)

        return {
            "expected_prompt_tps": round(prompt_tps, 2),
            "expected_eval_tps": round(eval_tps, 2),
            "expected_ttft_ms": round(ttft_ms, 2),
            "expected_vram_mb": round(vram_mb, 2),
            "expected_ram_mb": round(ram_mb, 2),
            "expected_gpu_util": round(gpu_util, 1),
            "expected_cpu_util": round(cpu_util, 1),
        }

    @staticmethod
    def detect_regression(
        records: List[ExecutionRecord],
        threshold_pct: float = 10.0,
    ) -> RegressionReport:
        """
        Detect throughput regressions between recent runs and historical baseline.
        """
        if len(records) < 6:
            return RegressionReport(
                is_regression=False,
                drop_percentage=0.0,
                previous_avg_tps=0.0,
                recent_avg_tps=0.0,
                message="Insufficient records to evaluate performance regression.",
            )

        # Sort records chronologically (oldest first)
        sorted_records = sorted(records, key=lambda r: r.timestamp)
        split_idx = int(len(sorted_records) * 0.70)
        older_records = sorted_records[:split_idx]
        recent_records = sorted_records[split_idx:]

        prev_tps = sum(r.eval_tps for r in older_records if r.eval_tps > 0) / max(1, len(older_records))
        recent_tps = sum(r.eval_tps for r in recent_records if r.eval_tps > 0) / max(1, len(recent_records))

        if prev_tps <= 0:
            return RegressionReport(False, 0.0, prev_tps, recent_tps, "Baseline TPS is zero.")

        drop_pct = ((prev_tps - recent_tps) / prev_tps) * 100.0

        if drop_pct >= threshold_pct:
            msg = (
                f"Performance regression detected: Recent TPS ({recent_tps:.1f}) is {drop_pct:.1f}% "
                f"lower than baseline ({prev_tps:.1f} TPS). Possible driver/backend change."
            )
            return RegressionReport(
                is_regression=True,
                drop_percentage=round(drop_pct, 1),
                previous_avg_tps=round(prev_tps, 2),
                recent_avg_tps=round(recent_tps, 2),
                message=msg,
            )

        return RegressionReport(
            is_regression=False,
            drop_percentage=round(max(0.0, drop_pct), 1),
            previous_avg_tps=round(prev_tps, 2),
            recent_avg_tps=round(recent_tps, 2),
            message="Performance remains within expected historical parameters.",
        )
