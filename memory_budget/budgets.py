"""
budgets.py
-----------
Budget evolution store for Memory Budget Manager in InferenceOS.
"""
from __future__ import annotations

import time
from typing import List, Optional

from .interfaces import BudgetDecision


class BudgetStore:
    """
    Maintains historical evolution of memory budget decisions.
    """

    def __init__(self, max_history: int = 500) -> None:
        self.max_history = max_history
        self._history: List[BudgetDecision] = []

    def record_decision(self, decision: BudgetDecision) -> None:
        self._history.append(decision)
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def get_latest(self) -> Optional[BudgetDecision]:
        return self._history[-1] if self._history else None

    def get_history(self) -> List[BudgetDecision]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
