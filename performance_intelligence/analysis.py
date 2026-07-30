"""
analysis.py
-----------
Root Cause Analyzer for Performance Intelligence Engine in InferenceOS.
"""
from __future__ import annotations

from typing import Any, Dict

from .interfaces import RegressionReport, RootCauseAnalysis


class RootCauseAnalyzer:
    """
    Determines explanations for performance regressions.
    """

    def analyze_cause(self, regression: RegressionReport, health_status: str = "Good") -> RootCauseAnalysis:
        if not regression.is_regression:
            return RootCauseAnalysis(
                regression_detected=False,
                probable_cause="None",
                explanation="Performance is operating within baseline boundaries.",
                severity="INFO",
            )

        if health_status == "Critical":
            return RootCauseAnalysis(
                regression_detected=True,
                probable_cause="GPU thermal throttling or high memory pressure",
                explanation="Regression caused by thermal throttling or critical memory pressure.",
                severity="CRITICAL",
            )

        return RootCauseAnalysis(
            regression_detected=True,
            probable_cause="GPU driver update or runtime configuration change",
            explanation="GPU driver updated or scheduler configuration changed.",
            severity="WARNING",
        )
