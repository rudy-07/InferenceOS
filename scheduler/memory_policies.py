"""
memory_policies.py
------------------
Modular pressure classification and memory allocation policies for the Adaptive Memory Scheduler in InferenceOS.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple


class MemoryPressureLevel(str, Enum):
    """Memory pressure levels reflecting current runtime resource saturation."""
    IDLE = "Idle"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

    def __str__(self) -> str:
        return self.value


class MemoryPressureClassifier:
    """Classifies system and VRAM pressure based on current utilization percentages."""

    @staticmethod
    def classify(
        used_vram_mb: float,
        total_vram_mb: float,
        used_ram_mb: float = 0.0,
        total_ram_mb: float = 0.0,
    ) -> MemoryPressureLevel:
        """
        Compute pressure level from memory utilization ratios.
        """
        vram_util = (used_vram_mb / max(1.0, total_vram_mb)) * 100.0 if total_vram_mb > 0 else 0.0
        ram_util = (used_ram_mb / max(1.0, total_ram_mb)) * 100.0 if total_ram_mb > 0 else 0.0

        max_util = max(vram_util, ram_util)

        if max_util > 92.0:
            return MemoryPressureLevel.CRITICAL
        elif max_util > 80.0:
            return MemoryPressureLevel.HIGH
        elif max_util > 60.0:
            return MemoryPressureLevel.MEDIUM
        elif max_util > 30.0:
            return MemoryPressureLevel.LOW
        else:
            return MemoryPressureLevel.IDLE


class MemorySchedulingPolicy(Protocol):
    """Protocol for modular memory allocation strategies."""
    name: str

    def evaluate_strategy(
        self,
        estimated_vram_mb: float,
        estimated_ram_mb: float,
        free_vram_mb: float,
        free_ram_mb: float,
        total_vram_mb: float,
        total_ram_mb: float,
        pressure: MemoryPressureLevel,
        hardware_vram_gb: float,
    ) -> Tuple[float, float, float, float, bool, List[str], List[str]]:
        """
        Returns (vram_budget_mb, ram_budget_mb, reserved_vram_mb, reserved_ram_mb, oom_risk, actions, reasoning).
        """
        ...


class ConservativePolicy:
    """
    Conservative Strategy: Prioritizes stability, larger reserved memory buffers,
    and early scaling recommendations under memory pressure.
    """
    name = "Conservative Strategy"

    def evaluate_strategy(
        self,
        estimated_vram_mb: float,
        estimated_ram_mb: float,
        free_vram_mb: float,
        free_ram_mb: float,
        total_vram_mb: float,
        total_ram_mb: float,
        pressure: MemoryPressureLevel,
        hardware_vram_gb: float,
    ) -> Tuple[float, float, float, float, bool, List[str], List[str]]:
        # Reserve 25% VRAM headroom for safety
        safety_ratio = 0.25
        reserved_vram_mb = max(600.0, total_vram_mb * safety_ratio)
        reserved_ram_mb = max(2048.0, total_ram_mb * 0.20)

        vram_budget_mb = max(0.0, (free_vram_mb if free_vram_mb > 0 else total_vram_mb) - reserved_vram_mb)
        ram_budget_mb = max(0.0, (free_ram_mb if free_ram_mb > 0 else total_ram_mb) - reserved_ram_mb)

        oom_risk = estimated_vram_mb > vram_budget_mb or estimated_ram_mb > ram_budget_mb

        actions: List[str] = []
        reasoning: List[str] = ["Conservative policy applied: generous 25% safety margin reserved."]

        if oom_risk or pressure in (MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL):
            actions.append("Recommend microbatch reduction (e.g. 256) to reduce activation VRAM spikes.")
            actions.append("Recommend context length scaling to maintain safe headroom.")
            reasoning.append("High pressure detected; proactive runtime scaling recommended.")

        return vram_budget_mb, ram_budget_mb, reserved_vram_mb, reserved_ram_mb, oom_risk, actions, reasoning


class BalancedPolicy:
    """
    Balanced Strategy: Standard operating mode balancing throughput with safe memory headroom (15%).
    """
    name = "Balanced Strategy"

    def evaluate_strategy(
        self,
        estimated_vram_mb: float,
        estimated_ram_mb: float,
        free_vram_mb: float,
        free_ram_mb: float,
        total_vram_mb: float,
        total_ram_mb: float,
        pressure: MemoryPressureLevel,
        hardware_vram_gb: float,
    ) -> Tuple[float, float, float, float, bool, List[str], List[str]]:
        safety_ratio = 0.15
        reserved_vram_mb = max(500.0, total_vram_mb * safety_ratio)
        reserved_ram_mb = max(1500.0, total_ram_mb * 0.15)

        vram_budget_mb = max(0.0, (free_vram_mb if free_vram_mb > 0 else total_vram_mb) - reserved_vram_mb)
        ram_budget_mb = max(0.0, (free_ram_mb if free_ram_mb > 0 else total_ram_mb) - reserved_ram_mb)

        oom_risk = estimated_vram_mb > vram_budget_mb or estimated_ram_mb > ram_budget_mb

        actions: List[str] = []
        reasoning: List[str] = ["Balanced policy applied: 15% safety margin reserved for optimal performance."]

        if oom_risk:
            actions.append("Recommend scaling context window or microbatch size to fit safe budget.")
            reasoning.append("Estimated memory exceeds balanced budget; runtime scaling recommended.")

        return vram_budget_mb, ram_budget_mb, reserved_vram_mb, reserved_ram_mb, oom_risk, actions, reasoning


class AggressivePolicy:
    """
    Aggressive Strategy: Maximizes GPU layer offload and throughput on modern high-capacity hardware (>= 24GB VRAM).
    Maintains tight 5-10% safety margin.
    """
    name = "Aggressive Strategy"

    def evaluate_strategy(
        self,
        estimated_vram_mb: float,
        estimated_ram_mb: float,
        free_vram_mb: float,
        free_ram_mb: float,
        total_vram_mb: float,
        total_ram_mb: float,
        pressure: MemoryPressureLevel,
        hardware_vram_gb: float,
    ) -> Tuple[float, float, float, float, bool, List[str], List[str]]:
        safety_ratio = 0.08 if hardware_vram_gb >= 20.0 else 0.12
        reserved_vram_mb = max(350.0, total_vram_mb * safety_ratio)
        reserved_ram_mb = max(1000.0, total_ram_mb * 0.10)

        vram_budget_mb = max(0.0, (free_vram_mb if free_vram_mb > 0 else total_vram_mb) - reserved_vram_mb)
        ram_budget_mb = max(0.0, (free_ram_mb if free_ram_mb > 0 else total_ram_mb) - reserved_ram_mb)

        oom_risk = estimated_vram_mb > vram_budget_mb or estimated_ram_mb > ram_budget_mb

        actions: List[str] = []
        reasoning: List[str] = ["Aggressive policy applied: tight safety margin reserved to maximize throughput."]

        if oom_risk:
            actions.append("Recommend falling back to Balanced policy under memory constraints.")
            reasoning.append("Aggressive allocation budget exceeded; falling back to safe limits.")

        return vram_budget_mb, ram_budget_mb, reserved_vram_mb, reserved_ram_mb, oom_risk, actions, reasoning
