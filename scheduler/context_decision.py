"""
context_decision.py
-------------------
Dataclass and formatting utilities for context scheduling decisions in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextDecision:
    """
    Result of a context scheduling decision.

    Attributes
    ----------
    requested_context : int
        User or API requested context length in tokens.
    recommended_context : int
        Recommended context length computed by the scheduler.
    effective_context : int
        Final selected context window size to be allocated in runtime.
    estimated_kv_memory_gb : float
        Analytical GQA-aware KV cache memory estimate in GB.
    estimated_total_memory_gb : float
        Combined model weights + KV cache + workspace memory estimate in GB.
    safety_margin_gb : float
        Reserved unallocated VRAM/RAM safety margin in GB.
    suggested_microbatch : Optional[int]
        Recommended microbatch size adjustment for Dynamic Microbatch Scheduler if context is tight.
    compression_mode : str
        Future-proof KV compression mode ("none", "fp8", "int4"). Default "none".
    confidence : float
        Confidence score between 0.0 and 1.0.
    candidate_evaluations : Dict[int, Dict[str, Any]]
        Per-candidate context assessment details and score mapping.
    warnings : List[str]
        Warnings or degradation notifications.
    reasoning : List[str]
        Human-readable bullet points explaining the decision.
    future_actions : List[str]
        Recommended actions for future runtime optimization (KV compression, sparse runtime, etc.).
    timestamp : float
        Epoch timestamp when the decision was computed.
    """

    requested_context: int
    recommended_context: int
    effective_context: int
    estimated_kv_memory_gb: float = 0.0
    estimated_total_memory_gb: float = 0.0
    safety_margin_gb: float = 0.0
    suggested_microbatch: Optional[int] = None
    compression_mode: str = "none"
    confidence: float = 0.95
    candidate_evaluations: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    future_actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a JSON-serializable dictionary."""
        return {
            "requested_context": self.requested_context,
            "recommended_context": self.recommended_context,
            "effective_context": self.effective_context,
            "estimated_kv_memory_gb": round(self.estimated_kv_memory_gb, 3),
            "estimated_total_memory_gb": round(self.estimated_total_memory_gb, 3),
            "safety_margin_gb": round(self.safety_margin_gb, 3),
            "suggested_microbatch": self.suggested_microbatch,
            "compression_mode": self.compression_mode,
            "confidence": round(self.confidence, 3),
            "candidate_evaluations": {
                int(k): dict(v) for k, v in self.candidate_evaluations.items()
            },
            "warnings": list(self.warnings),
            "reasoning": list(self.reasoning),
            "future_actions": list(self.future_actions),
            "timestamp": self.timestamp,
        }

    def format_cli_output(self) -> str:
        """
        Format decision details into the clean CLI representation required by InferenceOS.
        """
        lines = [
            "Context Scheduler",
            "Requested",
            f"  {self.requested_context}",
            "Candidates",
        ]
        sorted_candidates = sorted(self.candidate_evaluations.keys(), reverse=True)
        for cand in sorted_candidates:
            info = self.candidate_evaluations[cand]
            est_mem = info.get("estimated_total_memory_gb", 0.0)
            is_safe = info.get("is_safe", False)
            reason = info.get("reason", "")
            score = info.get("score", 0.0)

            if not is_safe:
                lines.append(f"  {cand:5d}: Estimated Memory {est_mem:.1f} GB (Rejected: {reason or 'Exceeds safe budget'})")
            elif cand == self.effective_context:
                lines.append(f"  {cand:5d}: Estimated Memory {est_mem:.1f} GB (Accepted: Score {int(round(score))})")
            else:
                lines.append(f"  {cand:5d}: Estimated Memory {est_mem:.1f} GB (Score {int(round(score))})")

        lines.extend([
            "Selected Context",
            f"  {self.effective_context}",
            "Reason",
        ])
        if self.reasoning:
            for r in self.reasoning:
                lines.append(f"  {r}")
        else:
            lines.append("  Largest context within safe memory budget.")

        return "\n".join(lines)

    def summary(self) -> str:
        """Return a single-line summary of the context decision."""
        reason_str = self.reasoning[0] if self.reasoning else "Optimal allocation"
        deg_str = f" [Degraded from {self.requested_context}]" if self.effective_context < self.requested_context else ""
        return (
            f"Context: {self.effective_context}{deg_str} (KV Mem={self.estimated_kv_memory_gb:.2f}GB, "
            f"Total Mem={self.estimated_total_memory_gb:.2f}GB, Reason: {reason_str})"
        )
