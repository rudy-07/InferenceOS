"""
memory_tier.py
--------------
Defines hierarchical memory tiers for InferenceOS Phase 2.

Tiers:
  - Tier 1: GPU VRAM (Primary high-bandwidth inference memory)
  - Tier 2: System RAM (Secondary spillover inference memory)
  - Future Tier: Persistent Storage (Disk - strictly non-active for inference)
"""
from __future__ import annotations

from enum import Enum


class MemoryTier(str, Enum):
    """
    Hierarchical memory tiers.
    """
    TIER_1_VRAM = "VRAM"
    TIER_2_RAM = "RAM"
    FUTURE_TIER_STORAGE = "STORAGE"

    @property
    def tier_level(self) -> int:
        """Returns numeric priority level (1 highest, 3 lowest)."""
        if self == MemoryTier.TIER_1_VRAM:
            return 1
        elif self == MemoryTier.TIER_2_RAM:
            return 2
        return 3

    @property
    def is_active_inference_memory(self) -> bool:
        """
        Returns True if the tier is allowed for active inference execution.
        Disk storage is strictly forbidden for active inference memory.
        """
        return self in (MemoryTier.TIER_1_VRAM, MemoryTier.TIER_2_RAM)

    def __str__(self) -> str:
        return self.value
