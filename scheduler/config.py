"""
config.py
---------
Configuration parameters for the Dynamic Microbatch Scheduler in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_DEFAULT_CANDIDATES: List[int] = [128, 256, 384, 512, 768, 1024, 2048]


@dataclass
class SchedulerConfig:
    """
    Configuration options governing dynamic microbatch selection.

    Parameters
    ----------
    enabled : bool
        Whether dynamic microbatching is enabled. Default True.
    min_microbatch : int
        Floor on candidate microbatch size. Default 128.
    max_microbatch : int
        Ceiling on candidate microbatch size. Default 2048.
    supported_candidates : List[int]
        Allowed discrete candidate step sizes.
    safety_margin : float
        Fraction of available VRAM reserved as unallocated headroom (0.0 to 0.5). Default 0.15 (15%).
    aggressiveness : float
        Weighting factor for throughput optimization (1.0 = standard, >1.0 = aggressive, <1.0 = conservative).
    optimization_goal : str
        Primary objective: "throughput", "latency", or "balanced". Default "throughput".
    manual_override : Optional[int]
        Explicit microbatch override set by user. Bypasses dynamic selection when set.
    verbose : bool
        Print candidate scoring table to CLI stdout when selecting microbatch. Default False.
    """

    enabled: bool = True
    min_microbatch: int = 128
    max_microbatch: int = 2048
    supported_candidates: List[int] = field(default_factory=lambda: list(_DEFAULT_CANDIDATES))
    safety_margin: float = 0.15
    aggressiveness: float = 1.0
    optimization_goal: str = "throughput"
    manual_override: Optional[int] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self.min_microbatch = max(16, self.min_microbatch)
        self.max_microbatch = max(self.min_microbatch, self.max_microbatch)
        self.safety_margin = max(0.01, min(0.50, self.safety_margin))
        self.aggressiveness = max(0.1, min(5.0, self.aggressiveness))

        if self.optimization_goal not in ("throughput", "latency", "balanced"):
            self.optimization_goal = "throughput"

        if not self.supported_candidates:
            self.supported_candidates = list(_DEFAULT_CANDIDATES)

    def get_candidate_set(self, context_length: int = 4096) -> List[int]:
        """
        Return the list of candidate microbatch sizes filtered by min/max boundaries
        and prompt context length.
        """
        if self.manual_override is not None and self.manual_override > 0:
            return [self.manual_override]

        # Clamp candidates between min and max bounds
        filtered = [
            c for c in sorted(self.supported_candidates)
            if self.min_microbatch <= c <= self.max_microbatch
        ]

        # If context_length is specified, ensure we don't return candidates vastly larger than context
        # unless context is smaller than min_microbatch
        upper_limit = max(self.min_microbatch, context_length)
        bounded = [c for c in filtered if c <= upper_limit]

        # If bounded set is empty, return at least min_microbatch or context_length
        if not bounded:
            val = min(self.max_microbatch, max(self.min_microbatch, context_length))
            return [val]

        return bounded

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict representation of config."""
        return {
            "enabled": self.enabled,
            "min_microbatch": self.min_microbatch,
            "max_microbatch": self.max_microbatch,
            "supported_candidates": list(self.supported_candidates),
            "safety_margin": self.safety_margin,
            "aggressiveness": self.aggressiveness,
            "optimization_goal": self.optimization_goal,
            "manual_override": self.manual_override,
            "verbose": self.verbose,
        }
