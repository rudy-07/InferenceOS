"""
history.py
----------
Optimization history store for Automatic Performance Optimizer in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interfaces import OptimizationProfile


@dataclass
class OptimizationHistoryEntry:
    profile: OptimizationProfile
    event_type: str  # "Created", "Refined", "Reoptimized"
    timestamp: float = field(default_factory=time.time)


class OptimizationHistoryStore:
    """
    Maintains historical trajectory of optimization runs.
    """

    def __init__(self) -> None:
        self._entries: List[OptimizationHistoryEntry] = []

    def record_event(self, profile: OptimizationProfile, event_type: str = "Created") -> None:
        self._entries.append(OptimizationHistoryEntry(profile=profile, event_type=event_type))

    def get_history(self) -> List[OptimizationHistoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
