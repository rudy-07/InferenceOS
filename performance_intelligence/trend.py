"""
trend.py
--------
Trend analyzer for Performance Intelligence Engine in InferenceOS.
"""
from __future__ import annotations

from typing import Any, Dict, List


class TrendAnalyzer:
    """
    Analyzes historical trajectories to detect daily, weekly, and long-term performance trends.
    """

    def analyze_trend(self, history: List[Dict[str, Any]]) -> str:
        if len(history) < 3:
            return "Stable"

        recent = history[-3:]
        tps_vals = [r.get("eval_tps", 50.0) for r in recent]

        if tps_vals[-1] > tps_vals[0] * 1.05:
            return "Improving"
        elif tps_vals[-1] < tps_vals[0] * 0.95:
            return "Degrading"
        return "Stable"
