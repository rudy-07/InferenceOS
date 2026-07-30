"""
decision.py
-----------
Dataclass and formatting utilities for microbatch scheduling decisions in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SchedulingDecision:
    """
    Result of a microbatch scheduling decision.

    Attributes
    ----------
    microbatch : int
        Selected microbatch size (e.g., 512).
    confidence : float
        Confidence score between 0.0 and 1.0.
    optimization_goal : str
        Target goal ("throughput", "latency", "balanced").
    estimated_vram_gb : float
        Estimated VRAM consumption for prompt processing (GB).
    estimated_ram_gb : float
        Estimated RAM consumption for prompt processing (GB).
    safety_margin : float
        VRAM safety headroom fraction maintained (0.0 to 1.0).
    candidate_scores : Dict[int, float]
        Mapping of evaluated microbatch candidate sizes to total scores (0-100).
    candidate_reasons : Dict[int, str]
        Key rationale or rejection notice per candidate size.
    reasoning : List[str]
        Ordered human-readable explanation bullet points for final choice.
    validated_by_learning : bool
        True if the decision was recommended or validated by a Runtime Learning policy.
    timestamp : float
        Epoch timestamp when the decision was computed.
    """

    microbatch: int
    confidence: float = 0.90
    optimization_goal: str = "throughput"
    estimated_vram_gb: float = 0.0
    estimated_ram_gb: float = 0.0
    safety_margin: float = 0.15
    candidate_scores: Dict[int, float] = field(default_factory=dict)
    candidate_reasons: Dict[int, str] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    validated_by_learning: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a JSON-serializable dictionary."""
        return {
            "microbatch": self.microbatch,
            "confidence": round(self.confidence, 3),
            "optimization_goal": self.optimization_goal,
            "estimated_vram_gb": round(self.estimated_vram_gb, 3),
            "estimated_ram_gb": round(self.estimated_ram_gb, 3),
            "safety_margin": round(self.safety_margin, 3),
            "candidate_scores": {int(k): round(float(v), 1) for k, v in self.candidate_scores.items()},
            "candidate_reasons": {int(k): str(v) for k, v in self.candidate_reasons.items()},
            "reasoning": list(self.reasoning),
            "validated_by_learning": self.validated_by_learning,
            "timestamp": self.timestamp,
        }

    def format_cli_output(self) -> str:
        """
        Format decision details into the clean CLI representation required by InferenceOS.
        """
        lines = [
            "Dynamic Microbatch Scheduler",
            "Candidates:",
        ]
        sorted_candidates = sorted(self.candidate_scores.keys())
        for cand in sorted_candidates:
            score = self.candidate_scores[cand]
            reason = self.candidate_reasons.get(cand, "")
            if score <= 0:
                lines.append(f"  {cand:4d}: Score 0 ({reason or 'Rejected'})")
            else:
                extra = f" ({reason})" if reason else ""
                lines.append(f"  {cand:4d}: Score {int(round(score))}{extra}")

        lines.extend([
            "Selected:",
            f"  {self.microbatch}",
            "Reason:",
        ])
        if self.reasoning:
            for r in self.reasoning:
                lines.append(f"  {r}")
        else:
            lines.append("  Highest predicted throughput while remaining within VRAM budget.")

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a single-line summary of the decision."""
        reason_str = self.reasoning[0] if self.reasoning else "Optimal balance"
        return (
            f"Microbatch={self.microbatch} (Goal={self.optimization_goal}, "
            f"Est. VRAM={self.estimated_vram_gb:.2f}GB, Safety={self.safety_margin * 100:.0f}%, "
            f"Reason: {reason_str})"
        )
