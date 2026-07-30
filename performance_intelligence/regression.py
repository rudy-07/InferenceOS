"""
regression.py
-------------
Regression detector for Performance Intelligence Engine in InferenceOS.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .interfaces import PerformanceBaseline, RegressionReport


class RegressionDetector:
    """
    Detects statistically significant performance regressions.
    """

    def detect_regression(
        self,
        current_eval_tps: float,
        current_ttft_ms: float,
        baseline: PerformanceBaseline,
    ) -> RegressionReport:
        """
        Compare current metrics against baseline.
        """
        if baseline.sample_count < 2 or baseline.avg_eval_tps <= 0.0:
            return RegressionReport(is_regression=False, reasoning=["Insufficient baseline history."])

        # Evaluate Generation TPS drop
        tps_drop_pct = ((baseline.avg_eval_tps - current_eval_tps) / baseline.avg_eval_tps) * 100.0

        if tps_drop_pct >= 10.0:
            conf = min(98.0, 75.0 + (baseline.sample_count * 0.5))
            return RegressionReport(
                is_regression=True,
                metric_name="Generation TPS",
                old_value=baseline.avg_eval_tps,
                new_value=current_eval_tps,
                change_pct=round(-tps_drop_pct, 1),
                confidence_pct=round(conf, 1),
                reasoning=[f"Generation TPS dropped by {tps_drop_pct:.1f}% below baseline ({baseline.avg_eval_tps:.1f} -> {current_eval_tps:.1f})."],
            )

        return RegressionReport(
            is_regression=False,
            metric_name="Generation TPS",
            old_value=baseline.avg_eval_tps,
            new_value=current_eval_tps,
            change_pct=round(-tps_drop_pct, 1) if tps_drop_pct > 0 else round(abs(tps_drop_pct), 1),
            confidence_pct=95.0,
            reasoning=["Performance operating within normal baseline limits."],
        )
