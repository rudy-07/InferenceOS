"""
interfaces.py
-------------
Data structures and interface definitions for Adaptive Memory Budget Manager in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComponentBudgets:
    """Memory budget distribution across 7 runtime components (in MB)."""
    weights_mb: float = 0.0
    kv_cache_mb: float = 0.0
    microbatch_mb: float = 0.0
    context_mb: float = 0.0
    runtime_buffers_mb: float = 0.0
    safety_reserve_mb: float = 0.0
    future_expansion_mb: float = 0.0

    def total_allocated_mb(self) -> float:
        return (
            self.weights_mb
            + self.kv_cache_mb
            + self.microbatch_mb
            + self.runtime_buffers_mb
            + self.safety_reserve_mb
            + self.future_expansion_mb
        )


@dataclass
class BudgetDecision:
    """
    Result of an Adaptive Memory Budget decision.

    Attributes
    ----------
    total_vram_budget_gb : float
        Total allocated VRAM budget in GB.
    budgets : ComponentBudgets
        Breakdown across 7 component buckets.
    active_policy : str
        Policy mode ("Adaptive", "Balanced", "Conservative", "Aggressive", "Server", "Low Memory").
    reasoning : List[str]
        Human-readable rationale for budget allocation.
    timestamp : float
        Epoch timestamp when decision was computed.
    """

    total_vram_budget_gb: float
    budgets: ComponentBudgets
    active_policy: str = "Adaptive"
    reasoning: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a JSON-serializable dictionary."""
        return {
            "total_vram_budget_gb": round(self.total_vram_budget_gb, 3),
            "weights_mb": round(self.budgets.weights_mb, 1),
            "kv_cache_mb": round(self.budgets.kv_cache_mb, 1),
            "microbatch_mb": round(self.budgets.microbatch_mb, 1),
            "context_mb": round(self.budgets.context_mb, 1),
            "runtime_buffers_mb": round(self.budgets.runtime_buffers_mb, 1),
            "safety_reserve_mb": round(self.budgets.safety_reserve_mb, 1),
            "future_expansion_mb": round(self.budgets.future_expansion_mb, 1),
            "active_policy": self.active_policy,
            "reasoning": list(self.reasoning),
            "timestamp": self.timestamp,
        }

    def format_cli_output(self) -> str:
        """Format details into the exact CLI representation required by InferenceOS."""
        lines = [
            "Adaptive Memory Budget",
            "Weights",
            f"  {self.budgets.weights_mb / 1024.0:.1f} GB",
            "KV Budget",
            f"  {int(round(self.budgets.kv_cache_mb))} MB",
            "Microbatch Budget",
            f"  {int(round(self.budgets.microbatch_mb))} MB",
            "Reserve",
            f"  {int(round(self.budgets.safety_reserve_mb))} MB",
            "Policy",
            f"  {self.active_policy}",
            "Reason",
        ]
        if self.reasoning:
            for r in self.reasoning:
                lines.append(f"  {r}")
        else:
            lines.append("  Memory pressure low.")

        return "\n".join(lines)


@dataclass
class BudgetConfig:
    """Configuration options for Memory Budget Manager."""
    enabled: bool = True
    policy_mode: str = "adaptive"  # adaptive | balanced | conservative | aggressive | server | low_memory
    verbose: bool = False
