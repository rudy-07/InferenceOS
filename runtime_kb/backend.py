"""
backend.py
----------
Backend Knowledge Base for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .interfaces import ExecutionRecord


@dataclass
class BackendKnowledgeProfile:
    backend_name: str = "vulkan"
    total_executions: int = 0
    avg_tps: float = 40.0
    stability_pct: float = 99.0
    recommended_microbatch: int = 512


class BackendKnowledgeBase:
    """
    Maintains persistent knowledge about inference backends (CUDA, Vulkan, Metal, ROCm, CPU).
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, BackendKnowledgeProfile] = {}

    def update_from_records(self, records: List[ExecutionRecord]) -> None:
        if not records:
            return

        grouped: Dict[str, List[ExecutionRecord]] = {}
        for r in records:
            key = r.backend.lower()
            grouped.setdefault(key, []).append(r)

        for key, recs in grouped.items():
            n = len(recs)
            succ = [r for r in recs if r.success]
            stab = (len(succ) / max(1, n)) * 100.0
            avg_tps = (sum(r.eval_tps for r in succ) / float(len(succ))) if succ else 0.0

            best_mb = max(succ, key=lambda r: r.eval_tps).microbatch_size if succ else 512

            self._profiles[key] = BackendKnowledgeProfile(
                backend_name=key,
                total_executions=n,
                avg_tps=round(avg_tps, 1),
                stability_pct=round(stab, 1),
                recommended_microbatch=best_mb,
            )

    def get_profile(self, backend_name: str) -> BackendKnowledgeProfile:
        return self._profiles.get(backend_name.lower(), BackendKnowledgeProfile(backend_name=backend_name))
