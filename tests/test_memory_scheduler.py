"""
test_memory_scheduler.py
-------------------------
Comprehensive test suite for the Adaptive Memory Scheduler in InferenceOS.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scheduler import (
    AggressivePolicy,
    BalancedPolicy,
    ConservativePolicy,
    MemoryDecision,
    MemoryPressureClassifier,
    MemoryPressureLevel,
    MemoryScheduler,
    MemorySchedulerConfig,
)
from inference_runtime.runtime_config import RuntimeConfig
from inference_runtime.argument_builder import ArgumentBuilder
from layer_placement.placement_plan import PlacementPlan, LayerPlacement, PlacementDevice, LayerCostBreakdown
from inference_runtime.backend_selector import BackendInfo


# ---------------------------------------------------------------------------
# 1. Memory Pressure Classification Tests
# ---------------------------------------------------------------------------

def test_memory_pressure_classifier():
    # Idle (< 30%)
    p_idle = MemoryPressureClassifier.classify(used_vram_mb=2000.0, total_vram_mb=16384.0)
    assert p_idle == MemoryPressureLevel.IDLE

    # Low (30% - 60%)
    p_low = MemoryPressureClassifier.classify(used_vram_mb=7000.0, total_vram_mb=16384.0)
    assert p_low == MemoryPressureLevel.LOW

    # Medium (60% - 80%)
    p_medium = MemoryPressureClassifier.classify(used_vram_mb=11000.0, total_vram_mb=16384.0)
    assert p_medium == MemoryPressureLevel.MEDIUM

    # High (80% - 92%)
    p_high = MemoryPressureClassifier.classify(used_vram_mb=14000.0, total_vram_mb=16384.0)
    assert p_high == MemoryPressureLevel.HIGH

    # Critical (> 92%)
    p_critical = MemoryPressureClassifier.classify(used_vram_mb=15500.0, total_vram_mb=16384.0)
    assert p_critical == MemoryPressureLevel.CRITICAL


# ---------------------------------------------------------------------------
# 2. Policy-driven Strategy Tests
# ---------------------------------------------------------------------------

def test_conservative_policy():
    policy = ConservativePolicy()
    vram_b, ram_b, res_vram, res_ram, oom, actions, reasoning = policy.evaluate_strategy(
        estimated_vram_mb=5000.0,
        estimated_ram_mb=2000.0,
        free_vram_mb=6000.0,
        free_ram_mb=16000.0,
        total_vram_mb=8192.0,
        total_ram_mb=32768.0,
        pressure=MemoryPressureLevel.MEDIUM,
        hardware_vram_gb=8.0,
    )
    assert res_vram >= 600.0  # Conservative policy reserves generous headroom
    assert "Conservative" in reasoning[0]


def test_balanced_policy():
    policy = BalancedPolicy()
    vram_b, ram_b, res_vram, res_ram, oom, actions, reasoning = policy.evaluate_strategy(
        estimated_vram_mb=5000.0,
        estimated_ram_mb=2000.0,
        free_vram_mb=12000.0,
        free_ram_mb=16000.0,
        total_vram_mb=16384.0,
        total_ram_mb=32768.0,
        pressure=MemoryPressureLevel.LOW,
        hardware_vram_gb=16.0,
    )
    assert not oom
    assert "Balanced" in reasoning[0]


def test_aggressive_policy():
    policy = AggressivePolicy()
    vram_b, ram_b, res_vram, res_ram, oom, actions, reasoning = policy.evaluate_strategy(
        estimated_vram_mb=18000.0,
        estimated_ram_mb=2000.0,
        free_vram_mb=22000.0,
        free_ram_mb=32000.0,
        total_vram_mb=24576.0,
        total_ram_mb=65536.0,
        pressure=MemoryPressureLevel.LOW,
        hardware_vram_gb=24.0,
    )
    assert res_vram < 2500.0  # Aggressive policy uses tighter reserved margin
    assert "Aggressive" in reasoning[0]


# ---------------------------------------------------------------------------
# 3. Hardware Adaptation & Strategy Selection Tests
# ---------------------------------------------------------------------------

def test_memory_scheduler_hardware_adaptation():
    scheduler = MemoryScheduler()

    # 1. Low VRAM GPU (6GB) -> Auto adapts to Conservative policy
    hw_low_vram = {
        "gpus": [{"vram_free_mb": 4000.0, "vram_total_mb": 6144.0}],
        "ram": {"available_gb": 8.0, "total_gb": 16.0},
    }
    dec_low = scheduler.schedule_memory(hw_profile=hw_low_vram)
    assert dec_low.strategy_selected == "Conservative Strategy"
    assert dec_low.safety_margin_level == "Conservative"

    # 2. High VRAM GPU (24GB) -> Auto adapts to Aggressive or Balanced strategy
    hw_high_vram = {
        "gpus": [{"vram_free_mb": 20000.0, "vram_total_mb": 24576.0}],
        "ram": {"available_gb": 32.0, "total_gb": 64.0},
    }
    dec_high = scheduler.schedule_memory(hw_profile=hw_high_vram)
    assert dec_high.strategy_selected in ("Aggressive Strategy", "Balanced Strategy")


# ---------------------------------------------------------------------------
# 4. Proactive OOM Prevention & Runtime Recommendations Tests
# ---------------------------------------------------------------------------

def test_proactive_oom_prevention_recommendations():
    scheduler = MemoryScheduler()
    # High pressure scenario (VRAM nearly full)
    hw_tight = {
        "gpus": [{"vram_free_mb": 800.0, "vram_total_mb": 8192.0}],
        "ram": {"available_gb": 4.0, "total_gb": 16.0},
    }
    decision = scheduler.schedule_memory(
        requested_context=32768,
        hw_profile=hw_tight,
    )
    assert decision.pressure_level in ("High", "Critical")
    assert decision.suggested_microbatch_override is not None
    assert decision.suggested_context_override is not None
    assert len(decision.warnings) > 0


# ---------------------------------------------------------------------------
# 5. CLI Output Formatting Tests
# ---------------------------------------------------------------------------

def test_memory_cli_formatting():
    decision = MemoryDecision(
        vram_budget_gb=5.5,
        ram_budget_gb=13.5,
        reserved_vram_gb=0.5,
        reserved_ram_gb=2.0,
        safety_margin_level="Balanced",
        pressure_level="Low",
        strategy_selected="Balanced Strategy",
        estimated_vram_gb=5.2,
        estimated_ram_gb=2.1,
        reasoning=["Enough headroom for stable execution."],
    )
    formatted = decision.format_cli_output()
    assert "Memory Scheduler" in formatted
    assert "Estimated VRAM" in formatted
    assert "5.2 GB" in formatted
    assert "Budget" in formatted
    assert "5.5 GB" in formatted
    assert "Reserved" in formatted
    assert "512 MB" in formatted
    assert "Pressure" in formatted
    assert "Low" in formatted
    assert "Decision" in formatted
    assert "Balanced Strategy" in formatted
    assert "Reason" in formatted
    assert "Enough headroom for stable execution." in formatted
