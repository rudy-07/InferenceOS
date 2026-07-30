"""
scheduler.py
------------
Scheduler Knowledge Base for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .interfaces import ExecutionRecord


@dataclass
class SchedulerKnowledgeSummary:
    total_scheduled: int = 0
    success_rate_pct: float = 100.0
    best_microbatch: int = 512
    best_context: int = 8192
    best_placement_layers: int = 32
    avg_confidence_pct: float = 90.0


class SchedulerKnowledgeBase:
    """
    Maintains historical knowledge about scheduler decisions and effectiveness.
    """

    def __init__(self) -> None:
        self.summary = SchedulerKnowledgeSummary()

    def update_from_records(self, records: List[ExecutionRecord]) -> None:
        if not records:
            return

        n = len(records)
        succ = [r for r in records if r.success]
        rate = (len(succ) / float(n)) * 100.0

        if succ:
            best_r = max(succ, key=lambda r: r.eval_tps)
            mb = best_r.microbatch_size
            ctx = max(r.context_length for r in succ)
            layers = best_r.gpu_layers
        else:
            mb = 512
            ctx = 8192
            layers = 32

        conf = min(98.0, 50.0 + (len(succ) * 0.30))

        self.summary = SchedulerKnowledgeSummary(
            total_scheduled=n,
            success_rate_pct=round(rate, 1),
            best_microbatch=mb,
            best_context=ctx,
            best_placement_layers=layers,
            avg_confidence_pct=round(conf, 1),
        )

    def get_summary(self) -> SchedulerKnowledgeSummary:
        return self.summary
