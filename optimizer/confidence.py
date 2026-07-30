"""
confidence.py
-------------
Optimization confidence scoring engine for Automatic Performance Optimizer in InferenceOS.
"""
from __future__ import annotations


class OptimizationConfidenceEngine:
    """
    Computes profile confidence metrics based on validation passes and historical execution feedback.
    """

    def compute_confidence(self, validation_passed: bool, execution_count: int = 1) -> float:
        if not validation_passed:
            return 0.0

        base = 90.0
        increment = min(8.0, execution_count * 0.5)
        return round(base + increment, 1)
