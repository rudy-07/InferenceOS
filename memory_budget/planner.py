"""
planner.py
----------
Budget planning and health-adaptation engine for Memory Budget Manager in InferenceOS.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .allocator import BudgetAllocator
from .interfaces import ComponentBudgets


class BudgetPlanner:
    """
    Adjusts component memory budgets dynamically based on RuntimeHealth observation.
    """

    def __init__(self) -> None:
        self.allocator = BudgetAllocator()

    def plan_budgets(
        self,
        total_vram_mb: float,
        free_vram_mb: float,
        model_weights_mb: float,
        health_status: str = "Good",
        pressure_level: str = "Low",
        policy_mode: str = "adaptive",
        learning_rec: Optional[Any] = None,
    ) -> Tuple[ComponentBudgets, str, List[str]]:
        """
        Plan and adjust component budgets using health and learning feedback.
        """
        reasoning: List[str] = []

        # Determine reserve ratio based on health status and pressure
        if health_status in ("Critical", "Failed") or pressure_level == "Critical":
            reserve_ratio = 0.30
            active_policy = "Low Memory"
            reasoning.append(f"Critical health ({health_status}) / pressure ({pressure_level}): reserve expanded to 30%.")
        elif health_status == "Warning" or pressure_level == "High":
            reserve_ratio = 0.22
            active_policy = "Conservative"
            reasoning.append(f"Warning status / high pressure: conservative reserve of 22% applied.")
        elif health_status == "Excellent" and pressure_level in ("Idle", "Low"):
            reserve_ratio = 0.10
            active_policy = "Adaptive"
            reasoning.append("Memory pressure low.")
        else:
            reserve_ratio = 0.15
            active_policy = "Balanced"
            reasoning.append("Balanced memory budget policy active.")

        budgets = self.allocator.allocate(
            total_vram_mb=total_vram_mb,
            free_vram_mb=free_vram_mb,
            model_weights_mb=model_weights_mb,
            reserve_ratio=reserve_ratio,
            policy_mode=policy_mode,
        )

        return budgets, active_policy, reasoning
