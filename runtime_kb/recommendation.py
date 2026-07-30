"""
recommendation.py
-----------------
Knowledge Recommendation Engine for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .database import HistoricalExecutionDatabase
from .hardware import HardwareKnowledgeBase
from .interfaces import KnowledgeRecommendation
from .models import ModelKnowledgeBase


class KnowledgeRecommendationEngine:
    """
    Computes high-level recommendations backed by confidence metrics.
    """

    def generate_recommendations(
        self,
        gpu_name: str = "GPU",
        model_name: str = "Qwen3-4B",
        db: Optional[HistoricalExecutionDatabase] = None,
        hw_kb: Optional[HardwareKnowledgeBase] = None,
        model_kb: Optional[ModelKnowledgeBase] = None,
    ) -> List[KnowledgeRecommendation]:
        """Generate structured recommendations."""
        records = db.get_all_records() if db else []
        count = len(records)

        hw_prof = hw_kb.get_profile(gpu_name) if hw_kb else None
        mod_prof = model_kb.get_profile(model_name) if model_kb else None

        mb_val = mod_prof.best_microbatch if mod_prof and mod_prof.total_executions > 0 else (hw_prof.best_microbatch if hw_prof else 512)
        layers_val = mod_prof.best_placement_layers if mod_prof and mod_prof.total_executions > 0 else (hw_prof.best_placement_layers if hw_prof else 32)
        ctx_val = mod_prof.best_context if mod_prof and mod_prof.total_executions > 0 else (hw_prof.best_context if hw_prof else 16384)

        conf = min(98.0, 50.0 + (count * 0.35)) if count > 0 else 50.0

        return [
            KnowledgeRecommendation(
                item_name="best_microbatch",
                recommended_value=mb_val,
                confidence_pct=round(conf, 1),
                execution_count=count,
                reasoning=f"Based on {count} successful executions.",
            ),
            KnowledgeRecommendation(
                item_name="best_placement_layers",
                recommended_value=f"{layers_val} GPU",
                confidence_pct=round(conf, 1),
                execution_count=count,
                reasoning=f"Based on {count} successful executions.",
            ),
            KnowledgeRecommendation(
                item_name="best_context",
                recommended_value=ctx_val,
                confidence_pct=round(conf, 1),
                execution_count=count,
                reasoning=f"Based on {count} successful executions.",
            ),
        ]
