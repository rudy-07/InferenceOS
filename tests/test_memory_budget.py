"""
test_memory_budget.py
----------------------
Comprehensive test suite for Adaptive Memory Budget Manager in InferenceOS.
"""
import pytest

from memory_budget import (
    BudgetAllocator,
    BudgetDecision,
    BudgetPlanner,
    ComponentBudgets,
    MemoryBudgetManager,
)
from cli.budget_cli import handle_budget_cli


# ---------------------------------------------------------------------------
# 1. Component Budget Allocator Tests
# ---------------------------------------------------------------------------

def test_budget_allocator():
    allocator = BudgetAllocator()
    budgets = allocator.allocate(
        total_vram_mb=6144.0,
        free_vram_mb=5000.0,
        model_weights_mb=3800.0,
        reserve_ratio=0.15,
    )
    assert budgets.weights_mb == 3800.0
    assert budgets.safety_reserve_mb >= 500.0
    assert budgets.kv_cache_mb > 0
    assert budgets.microbatch_mb > 0


# ---------------------------------------------------------------------------
# 2. Budget Planner & Health Feedback Loop Tests
# ---------------------------------------------------------------------------

def test_budget_planner_health_feedback():
    planner = BudgetPlanner()

    # 1. Normal Health ("Good") -> Balanced reserve (~15%)
    budgets_good, pol_good, _ = planner.plan_budgets(
        total_vram_mb=8192.0, free_vram_mb=6000.0, model_weights_mb=4000.0, health_status="Good", pressure_level="Low"
    )
    assert pol_good in ("Balanced", "Adaptive")
    assert budgets_good.safety_reserve_mb < 2000.0

    # 2. Critical Health ("Critical") -> Low Memory reserve (~30%)
    budgets_crit, pol_crit, _ = planner.plan_budgets(
        total_vram_mb=8192.0, free_vram_mb=1000.0, model_weights_mb=4000.0, health_status="Critical", pressure_level="Critical"
    )
    assert pol_crit == "Low Memory"
    assert budgets_crit.safety_reserve_mb >= 2000.0


# ---------------------------------------------------------------------------
# 3. CLI Output Formatting & CLI Handler Tests
# ---------------------------------------------------------------------------

def test_budget_decision_cli_formatting():
    budgets = ComponentBudgets(
        weights_mb=3891.2,
        kv_cache_mb=900.0,
        microbatch_mb=300.0,
        context_mb=720.0,
        runtime_buffers_mb=200.0,
        safety_reserve_mb=500.0,
        future_expansion_mb=100.0,
    )
    decision = BudgetDecision(
        total_vram_budget_gb=6.0,
        budgets=budgets,
        active_policy="Adaptive",
        reasoning=["Memory pressure low."],
    )
    formatted = decision.format_cli_output()
    assert "Adaptive Memory Budget" in formatted
    assert "Weights" in formatted
    assert "3.8 GB" in formatted
    assert "KV Budget" in formatted
    assert "900 MB" in formatted
    assert "Microbatch Budget" in formatted
    assert "300 MB" in formatted
    assert "Reserve" in formatted
    assert "500 MB" in formatted
    assert "Policy" in formatted
    assert "Adaptive" in formatted
    assert "Reason" in formatted
    assert "Memory pressure low." in formatted


def test_budget_cli_handler():
    assert handle_budget_cli(["show"]) == 0
    assert handle_budget_cli(["history"]) == 0
