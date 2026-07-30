"""
memory_decision.py
------------------
Dataclass and formatting utilities for memory scheduling decisions in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryDecision:
    """
    Result of an adaptive memory scheduling decision.

    Attributes
    ----------
    vram_budget_gb : float
        Maximum safe VRAM budget allocated for inference (GB).
    ram_budget_gb : float
        Maximum safe system RAM budget allocated for inference (GB).
    reserved_vram_gb : float
        Unallocated VRAM headroom reserved for OS/driver/backend overhead (GB).
    reserved_ram_gb : float
        Unallocated RAM headroom reserved for OS overhead (GB).
    safety_margin_level : str
        Safety margin level ("Conservative", "Balanced", "Aggressive").
    pressure_level : str
        Memory pressure level ("Idle", "Low", "Medium", "High", "Critical").
    strategy_selected : str
        Selected memory strategy ("Conservative", "Balanced", "Aggressive").
    estimated_vram_gb : float
        Estimated total VRAM requirement for the run (GB).
    estimated_ram_gb : float
        Estimated total RAM requirement for the run (GB).
    suggested_microbatch_override : Optional[int]
        Recommended microbatch adjustment if memory pressure is elevated.
    suggested_context_override : Optional[int]
        Recommended context adjustment if memory pressure is elevated.
    suggested_gpu_layer_override : Optional[int]
        Recommended GPU offloaded layer count adjustment if VRAM is constrained.
    confidence : float
        Confidence score between 0.0 and 1.0.
    recommended_actions : List[str]
        List of recommended runtime scheduler adjustments.
    warnings : List[str]
        Warnings or memory pressure notices.
    reasoning : List[str]
        Human-readable decision rationale bullets.
    oom_risk : bool
        True if memory pressure presents an elevated risk of Out-Of-Memory.
    timestamp : float
        Epoch timestamp when decision was computed.
    """

    vram_budget_gb: float
    ram_budget_gb: float
    reserved_vram_gb: float
    reserved_ram_gb: float
    safety_margin_level: str = "Balanced"
    pressure_level: str = "Low"
    strategy_selected: str = "Balanced Strategy"
    estimated_vram_gb: float = 0.0
    estimated_ram_gb: float = 0.0
    suggested_microbatch_override: Optional[int] = None
    suggested_context_override: Optional[int] = None
    suggested_gpu_layer_override: Optional[int] = None
    confidence: float = 0.95
    recommended_actions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    oom_risk: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a JSON-serializable dictionary."""
        return {
            "vram_budget_gb": round(self.vram_budget_gb, 3),
            "ram_budget_gb": round(self.ram_budget_gb, 3),
            "reserved_vram_gb": round(self.reserved_vram_gb, 3),
            "reserved_ram_gb": round(self.reserved_ram_gb, 3),
            "safety_margin_level": self.safety_margin_level,
            "pressure_level": self.pressure_level,
            "strategy_selected": self.strategy_selected,
            "estimated_vram_gb": round(self.estimated_vram_gb, 3),
            "estimated_ram_gb": round(self.estimated_ram_gb, 3),
            "suggested_microbatch_override": self.suggested_microbatch_override,
            "suggested_context_override": self.suggested_context_override,
            "suggested_gpu_layer_override": self.suggested_gpu_layer_override,
            "confidence": round(self.confidence, 3),
            "recommended_actions": list(self.recommended_actions),
            "warnings": list(self.warnings),
            "reasoning": list(self.reasoning),
            "oom_risk": self.oom_risk,
            "timestamp": self.timestamp,
        }

    def format_cli_output(self) -> str:
        """
        Format decision details into the clean CLI representation required by InferenceOS.
        """
        reserved_mb = int(round(self.reserved_vram_gb * 1024.0))
        lines = [
            "Memory Scheduler",
            "Estimated VRAM",
            f"  {self.estimated_vram_gb:.1f} GB",
            "Budget",
            f"  {self.vram_budget_gb:.1f} GB",
            "Reserved",
            f"  {reserved_mb} MB",
            "Pressure",
            f"  {self.pressure_level}",
            "Decision",
            f"  {self.strategy_selected}",
            "Reason",
        ]
        if self.reasoning:
            for r in self.reasoning:
                lines.append(f"  {r}")
        else:
            lines.append("  Enough headroom for stable execution.")

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a single-line summary of the memory decision."""
        reason_str = self.reasoning[0] if self.reasoning else "Stable memory operating range"
        return (
            f"Memory: Budget={self.vram_budget_gb:.2f}GB VRAM / {self.ram_budget_gb:.2f}GB RAM "
            f"(Pressure={self.pressure_level}, Strategy={self.strategy_selected}, Reason: {reason_str})"
        )
