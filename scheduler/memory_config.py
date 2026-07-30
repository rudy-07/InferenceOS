"""
memory_config.py
----------------
Configuration options for the Adaptive Memory Scheduler in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MemorySchedulerConfig:
    """
    Configuration options governing adaptive memory scheduling decisions.

    Parameters
    ----------
    enabled : bool
        Whether adaptive memory scheduling is enabled. Default True.
    vram_safety_margin : float
        Fraction of VRAM reserved as unallocated headroom (0.01 to 0.50). Default 0.15 (15%).
    ram_safety_margin : float
        Fraction of system RAM reserved as unallocated headroom. Default 0.15 (15%).
    memory_strategy : str
        Target policy strategy: "auto", "conservative", "balanced", or "aggressive". Default "auto".
    oom_prevention : str
        Strictness mode for Out-Of-Memory prevention: "strict", "balanced", "disabled". Default "strict".
    manual_override : Optional[str]
        Explicit strategy override ("conservative", "balanced", "aggressive").
    verbose : bool
        Print formatted memory decision block to CLI stdout when selecting memory policy. Default False.
    """

    enabled: bool = True
    vram_safety_margin: float = 0.15
    ram_safety_margin: float = 0.15
    memory_strategy: str = "auto"
    oom_prevention: str = "strict"
    manual_override: Optional[str] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self.vram_safety_margin = max(0.01, min(0.50, self.vram_safety_margin))
        self.ram_safety_margin = max(0.01, min(0.50, self.ram_safety_margin))

        if self.memory_strategy not in ("auto", "conservative", "balanced", "aggressive"):
            self.memory_strategy = "auto"

        if self.oom_prevention not in ("strict", "balanced", "disabled"):
            self.oom_prevention = "strict"

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict representation of config."""
        return {
            "enabled": self.enabled,
            "vram_safety_margin": self.vram_safety_margin,
            "ram_safety_margin": self.ram_safety_margin,
            "memory_strategy": self.memory_strategy,
            "oom_prevention": self.oom_prevention,
            "manual_override": self.manual_override,
            "verbose": self.verbose,
        }
