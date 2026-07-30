"""
context_config.py
------------------
Configuration dataclass and candidate generator for the Dynamic Context Scheduler in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextSchedulerConfig:
    """
    Configuration options governing dynamic context length scheduling.

    Parameters
    ----------
    enabled : bool
        Whether dynamic context scheduling is enabled. Default True.
    min_context : int
        Minimum context floor in tokens (must be >= 512). Default 512.
    max_context : int
        Absolute maximum context ceiling in tokens. Default 131072.
    preferred_context : Optional[int]
        Target preferred context size set by application profile.
    safety_margin : float
        Fraction of memory reserved as unallocated headroom (0.01 to 0.50). Default 0.15 (15%).
    optimization_goal : str
        Primary objective: "maximum_context", "maximum_speed", or "balanced". Default "maximum_context".
    manual_override : Optional[int]
        Explicit context override. Bypasses dynamic context selection when set.
    allow_graceful_degradation : bool
        Automatically scale down context when requested context exceeds safe memory. Default True.
    alignment_step : int
        GGML / llama.cpp block quantization context alignment step in tokens. Default 512.
    verbose : bool
        Print candidate evaluation table to CLI stdout when selecting context. Default False.
    """

    enabled: bool = True
    min_context: int = 512
    max_context: int = 131072
    preferred_context: Optional[int] = None
    safety_margin: float = 0.15
    optimization_goal: str = "maximum_context"
    manual_override: Optional[int] = None
    allow_graceful_degradation: bool = True
    alignment_step: int = 512
    verbose: bool = False

    def __post_init__(self) -> None:
        self.alignment_step = max(64, self.alignment_step)
        self.min_context = max(self.alignment_step, (self.min_context // self.alignment_step) * self.alignment_step)
        self.max_context = max(self.min_context, self.max_context)
        self.safety_margin = max(0.01, min(0.50, self.safety_margin))

        if self.optimization_goal not in ("maximum_context", "maximum_speed", "balanced"):
            self.optimization_goal = "maximum_context"

    def get_candidate_set(self, requested_context: int = 4096, max_model_context: int = 131072) -> List[int]:
        """
        Generate a candidate list of context lengths in descending order, floored to alignment step.

        Candidates start from min(requested_context, max_model_context, max_context) down to min_context.
        """
        if self.manual_override is not None and self.manual_override > 0:
            return [self.manual_override]

        upper_bound = min(requested_context, max_model_context, self.max_context)
        # Floor upper bound to alignment step
        upper_aligned = (max(self.min_context, upper_bound) // self.alignment_step) * self.alignment_step
        if upper_aligned < self.min_context:
            upper_aligned = self.min_context

        # Generate candidates in steps (e.g. 32768, 28672, 24576, 20480, 16384, 12288, 8192, 4096, 2048, 1024, 512)
        candidates: List[int] = []

        curr = upper_aligned
        while curr >= self.min_context:
            if curr not in candidates:
                candidates.append(curr)

            # Step down dynamically depending on scale
            if curr > 16384:
                step = 4096
            elif curr > 4096:
                step = 2048
            elif curr > 2048:
                step = 1024
            else:
                step = self.alignment_step

            curr -= step

        if self.min_context not in candidates:
            candidates.append(self.min_context)

        return sorted(candidates, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict representation of config."""
        return {
            "enabled": self.enabled,
            "min_context": self.min_context,
            "max_context": self.max_context,
            "preferred_context": self.preferred_context,
            "safety_margin": self.safety_margin,
            "optimization_goal": self.optimization_goal,
            "manual_override": self.manual_override,
            "allow_graceful_degradation": self.allow_graceful_degradation,
            "alignment_step": self.alignment_step,
            "verbose": self.verbose,
        }
