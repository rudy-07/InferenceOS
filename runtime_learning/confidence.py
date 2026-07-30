"""
confidence.py
-------------
Confidence scoring engine for Runtime Learning in InferenceOS.
"""
from __future__ import annotations

import math
from typing import List

from .history import ExecutionRecord


class ConfidenceEngine:
    """
    Computes statistical confidence scores (0.0 to 1.0) for learned recommendations.
    """

    @staticmethod
    def compute_confidence(
        sample_count: int,
        success_count: int,
        failure_count: int,
        records: List[ExecutionRecord],
    ) -> float:
        """
        Calculate confidence score based on sample count, success rate, and throughput variance.
        """
        if sample_count <= 0:
            return 0.0

        # 1. Sample Size Weight (0.0 to 0.70)
        # 1 execution = ~0.10, 5 = ~0.39, 15 = ~0.78, 30+ = ~0.95
        size_factor = 1.0 - math.exp(-sample_count / 12.0)

        # 2. Success Ratio Weight (0.0 to 1.0 multiplier)
        total_runs = success_count + failure_count
        success_ratio = (success_count / max(1, total_runs)) if total_runs > 0 else 1.0

        # 3. Variance Stability Weight (0.80 to 1.00)
        tps_values = [r.eval_tps for r in records if r.eval_tps > 0]
        variance_factor = 1.0
        if len(tps_values) >= 3:
            mean_tps = sum(tps_values) / len(tps_values)
            variance = sum((x - mean_tps) ** 2 for x in tps_values) / len(tps_values)
            std_dev = math.sqrt(variance)
            cv = (std_dev / max(0.1, mean_tps))
            variance_factor = max(0.60, min(1.0, 1.0 - cv))

        raw_confidence = size_factor * success_ratio * variance_factor
        return max(0.0, min(0.99, round(raw_confidence, 3)))
