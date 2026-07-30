"""
manager.py
----------
Adaptive Memory Budget Manager main facade for InferenceOS.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .budgets import BudgetStore
from .interfaces import BudgetConfig, BudgetDecision, ComponentBudgets
from .planner import BudgetPlanner

logger = logging.getLogger("InferenceOS.MemoryBudgetManager")


class MemoryBudgetManager:
    """
    Central facade responsible for dynamic component memory budget allocation.
    """

    def __init__(self, config: Optional[BudgetConfig] = None, verbose: bool = False) -> None:
        self.config = config or BudgetConfig()
        if verbose:
            self.config.verbose = True

        self.planner = BudgetPlanner()
        self.store = BudgetStore()

    def compute_budgets(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        health_status: str = "Good",
        pressure_level: str = "Low",
        learning_rec: Optional[Any] = None,
        config_override: Optional[BudgetConfig] = None,
    ) -> BudgetDecision:
        """
        Compute dynamic component memory budgets for an inference session.

        Returns
        -------
        BudgetDecision
            Object containing total VRAM budget, component allocations, active policy, and reasoning.
        """
        cfg = config_override or self.config
        hw = hw_profile or {}
        model = model_metadata or {}

        gpus = hw.get("gpus", [])
        total_vram_mb = sum(float(g.get("vram_total_mb", 8192.0)) for g in gpus) if gpus else 8192.0
        free_vram_mb = float(gpus[0].get("vram_free_mb", 6000.0)) if gpus else 6000.0

        model_size_mb = float(model.get("model_size_mb", 4000.0))

        budgets, active_policy, reasoning = self.planner.plan_budgets(
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            model_weights_mb=model_size_mb,
            health_status=health_status,
            pressure_level=pressure_level,
            policy_mode=cfg.policy_mode,
            learning_rec=learning_rec,
        )

        decision = BudgetDecision(
            total_vram_budget_gb=total_vram_mb / 1024.0,
            budgets=budgets,
            active_policy=active_policy,
            reasoning=reasoning,
        )

        self.store.record_decision(decision)

        if cfg.verbose:
            print(decision.format_cli_output())

        return decision

    def get_history(self) -> List[BudgetDecision]:
        return self.store.get_history()
