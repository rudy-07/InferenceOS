"""
recommendation.py
-----------------
Performance recommender for Performance Intelligence Engine in InferenceOS.
"""
from __future__ import annotations

from .interfaces import PIERecommendation, RegressionReport, RootCauseAnalysis


class PerformanceRecommender:
    """
    Generates explainable recommendations based on regression analysis.
    """

    def generate_recommendation(
        self,
        regression: RegressionReport,
        root_cause: RootCauseAnalysis,
    ) -> PIERecommendation:
        if not regression.is_regression:
            return PIERecommendation(
                action_name="Maintain Strategy",
                reasoning="Current optimization remains optimal.",
                confidence_pct=95.0,
            )

        if "driver" in root_cause.probable_cause.lower():
            return PIERecommendation(
                action_name="Run Adaptive Runtime Optimizer",
                reasoning="Run Automatic Performance Optimizer to re-calibrate profile following driver change.",
                confidence_pct=96.0,
            )

        return PIERecommendation(
            action_name="Run Automatic Performance Optimizer",
            reasoning="Re-optimize runtime configuration to recover baseline TPS.",
            confidence_pct=90.0,
        )
