"""
statistics.py
-------------
Knowledge aggregator for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from typing import List

from .backend import BackendKnowledgeBase
from .database import HistoricalExecutionDatabase
from .hardware import HardwareKnowledgeBase
from .models import ModelKnowledgeBase
from .performance import PerformanceKnowledgeBase
from .scheduler import SchedulerKnowledgeBase


class KnowledgeAggregator:
    """
    Aggregates raw historical execution records into specialized knowledge bases.
    """

    def aggregate_all(
        self,
        db: HistoricalExecutionDatabase,
        hw_kb: HardwareKnowledgeBase,
        model_kb: ModelKnowledgeBase,
        backend_kb: BackendKnowledgeBase,
        perf_kb: PerformanceKnowledgeBase,
        sched_kb: SchedulerKnowledgeBase,
    ) -> None:
        """Process raw records and update all knowledge bases."""
        records = db.get_all_records()
        if not records:
            return

        hw_kb.update_from_records(records)
        model_kb.update_from_records(records)
        backend_kb.update_from_records(records)
        perf_kb.update_from_records(records)
        sched_kb.update_from_records(records)
