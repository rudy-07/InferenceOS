"""
models.py
---------
Model Knowledge Base for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .interfaces import ExecutionRecord


@dataclass
class ModelKnowledgeProfile:
    model_name: str = "Qwen3-4B"
    total_executions: int = 0
    best_tps: float = 50.0
    avg_memory_gb: float = 4.5
    best_microbatch: int = 512
    best_context: int = 16384
    best_placement_layers: int = 32
    confidence_pct: float = 80.0


class ModelKnowledgeBase:
    """
    Maintains persistent knowledge for every model architecture.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, ModelKnowledgeProfile] = {}

    def update_from_records(self, records: List[ExecutionRecord]) -> None:
        """Aggregate execution records into model profiles."""
        if not records:
            return

        grouped: Dict[str, List[ExecutionRecord]] = {}
        for r in records:
            key = r.model_fp.model_name
            grouped.setdefault(key, []).append(r)

        for key, recs in grouped.items():
            valid = [r for r in recs if r.success]
            if not valid:
                continue

            n = len(valid)
            best_rec = max(valid, key=lambda r: r.eval_tps)
            avg_mem = sum(r.memory_used_mb for r in valid) / float(n) / 1024.0

            conf = min(98.0, 50.0 + (n * 0.35))

            self._profiles[key] = ModelKnowledgeProfile(
                model_name=key,
                total_executions=n,
                best_tps=round(best_rec.eval_tps, 1),
                avg_memory_gb=round(avg_mem, 2),
                best_microbatch=best_rec.microbatch_size,
                best_context=max(r.context_length for r in valid),
                best_placement_layers=best_rec.gpu_layers,
                confidence_pct=round(conf, 1),
            )

    def get_profile(self, model_name: str) -> ModelKnowledgeProfile:
        return self._profiles.get(model_name, ModelKnowledgeProfile(model_name=model_name))
