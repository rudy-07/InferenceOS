"""
query.py
--------
Knowledge Query Engine for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .hardware import HardwareKnowledgeBase
from .interfaces import KnowledgeRecommendation
from .models import ModelKnowledgeBase
from .performance import PerformanceKnowledgeBase
from .recommendation import KnowledgeRecommendationEngine


class KnowledgeQueryEngine:
    """
    Decoupled query API for accessing learned runtime knowledge.
    """

    def __init__(
        self,
        hw_kb: HardwareKnowledgeBase,
        model_kb: ModelKnowledgeBase,
        perf_kb: PerformanceKnowledgeBase,
        recommender: KnowledgeRecommendationEngine,
    ) -> None:
        self.hw_kb = hw_kb
        self.model_kb = model_kb
        self.perf_kb = perf_kb
        self.recommender = recommender

    def get_best_placement(self, model_name: str, gpu_name: str = "GPU") -> Tuple[int, float]:
        """Returns (best_gpu_layers, confidence_pct)."""
        mod_prof = self.model_kb.get_profile(model_name)
        if mod_prof.total_executions > 0:
            return mod_prof.best_placement_layers, mod_prof.confidence_pct
        hw_prof = self.hw_kb.get_profile(gpu_name)
        return hw_prof.best_placement_layers, hw_prof.confidence_pct

    def get_recommended_microbatch(self, model_name: str, gpu_name: str = "GPU") -> Tuple[int, float]:
        """Returns (best_microbatch, confidence_pct)."""
        mod_prof = self.model_kb.get_profile(model_name)
        if mod_prof.total_executions > 0:
            return mod_prof.best_microbatch, mod_prof.confidence_pct
        hw_prof = self.hw_kb.get_profile(gpu_name)
        return hw_prof.best_microbatch, hw_prof.confidence_pct

    def get_recommended_context(self, model_name: str, gpu_name: str = "GPU") -> Tuple[int, float]:
        """Returns (best_context, confidence_pct)."""
        mod_prof = self.model_kb.get_profile(model_name)
        if mod_prof.total_executions > 0:
            return mod_prof.best_context, mod_prof.confidence_pct
        hw_prof = self.hw_kb.get_profile(gpu_name)
        return hw_prof.best_context, hw_prof.confidence_pct

    def get_expected_tps(self, model_name: str) -> float:
        mod_prof = self.model_kb.get_profile(model_name)
        return mod_prof.best_tps if mod_prof.total_executions > 0 else 45.0

    def get_expected_memory_gb(self, model_name: str) -> float:
        mod_prof = self.model_kb.get_profile(model_name)
        return mod_prof.avg_memory_gb if mod_prof.total_executions > 0 else 4.0
