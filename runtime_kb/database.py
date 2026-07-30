"""
database.py
-----------
Historical Execution Database for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .interfaces import ExecutionRecord
from .storage import RKBStorageEngine


class HistoricalExecutionDatabase:
    """
    Append-only database recording every inference execution.
    """

    def __init__(self, storage: Optional[RKBStorageEngine] = None) -> None:
        self.storage = storage or RKBStorageEngine()
        self._memory_cache: List[ExecutionRecord] = []
        self._load_cache()

    def _load_cache(self) -> None:
        raw_list = self.storage.fetch_all_executions()
        self._memory_cache.clear()
        for d in raw_list:
            # Reconstruct memory cache
            pass

    def record_execution(self, record: ExecutionRecord) -> None:
        """Add execution record to persistent database."""
        self._memory_cache.append(record)
        self.storage.insert_execution(record.to_dict())

    def get_all_records(self) -> List[ExecutionRecord]:
        return list(self._memory_cache)

    def count(self) -> int:
        return len(self._memory_cache)

    def clear(self) -> None:
        self._memory_cache.clear()
        self.storage.clear()
