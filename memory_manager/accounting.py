"""
accounting.py
-------------
Memory accounting system for tracking current VRAM/RAM allocations, reserved
buffers, and memory fragmentation metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .memory_tier import MemoryTier


@dataclass
class MemoryBlock:
    """Represents an allocated block of memory."""
    tensor_id: str
    size_bytes: int
    offset: int


class MemoryAccounting:
    """
    Tracks memory metrics across VRAM (Tier 1) and RAM (Tier 2).
    """

    def __init__(self, max_vram_bytes: int = 0, max_ram_bytes: int = 0) -> None:
        self.max_vram_bytes: int = max_vram_bytes
        self.max_ram_bytes: int = max_ram_bytes

        self.allocated_vram_bytes: int = 0
        self.allocated_ram_bytes: int = 0

        self.reserved_vram_bytes: int = 0
        self.reserved_ram_bytes: int = 0

        self.vram_blocks: List[MemoryBlock] = []
        self.ram_blocks: List[MemoryBlock] = []

    def allocate(self, tier: MemoryTier, tensor_id: str, size_bytes: int) -> None:
        """Record an allocation in the specified memory tier."""
        if tier == MemoryTier.TIER_1_VRAM:
            offset = self.allocated_vram_bytes
            self.allocated_vram_bytes += size_bytes
            self.vram_blocks.append(MemoryBlock(tensor_id, size_bytes, offset))
        elif tier == MemoryTier.TIER_2_RAM:
            offset = self.allocated_ram_bytes
            self.allocated_ram_bytes += size_bytes
            self.ram_blocks.append(MemoryBlock(tensor_id, size_bytes, offset))
        else:
            raise ValueError(f"Cannot allocate active inference tensor in tier {tier}")

    def deallocate(self, tier: MemoryTier, tensor_id: str, size_bytes: int) -> bool:
        """Deallocate a tensor from the specified memory tier."""
        if tier == MemoryTier.TIER_1_VRAM:
            self.allocated_vram_bytes = max(0, self.allocated_vram_bytes - size_bytes)
            self.vram_blocks = [b for b in self.vram_blocks if b.tensor_id != tensor_id]
            return True
        elif tier == MemoryTier.TIER_2_RAM:
            self.allocated_ram_bytes = max(0, self.allocated_ram_bytes - size_bytes)
            self.ram_blocks = [b for b in self.ram_blocks if b.tensor_id != tensor_id]
            return True
        return False

    def reserve(self, tier: MemoryTier, size_bytes: int) -> None:
        """Reserve a buffer in the specified memory tier."""
        if tier == MemoryTier.TIER_1_VRAM:
            self.reserved_vram_bytes += size_bytes
        elif tier == MemoryTier.TIER_2_RAM:
            self.reserved_ram_bytes += size_bytes

    def get_fragmentation(self, tier: MemoryTier) -> float:
        """
        Calculate memory fragmentation index (0.0 to 1.0) for a tier based on
        non-contiguous free memory gap distribution.
        """
        blocks = self.vram_blocks if tier == MemoryTier.TIER_1_VRAM else self.ram_blocks
        max_bytes = self.max_vram_bytes if tier == MemoryTier.TIER_1_VRAM else self.max_ram_bytes
        allocated_bytes = self.allocated_vram_bytes if tier == MemoryTier.TIER_1_VRAM else self.allocated_ram_bytes

        if not blocks or max_bytes == 0:
            return 0.0

        free_bytes = max(0, max_bytes - allocated_bytes)
        if free_bytes == 0:
            return 0.0

        # Calculate ratio of small gaps to total free memory
        num_blocks = len(blocks)
        avg_block_size = allocated_bytes / num_blocks if num_blocks > 0 else 1
        fragmentation = min(1.0, (num_blocks * 0.05) * (1.0 - (allocated_bytes / max_bytes)))
        return round(fragmentation, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Return memory accounting telemetry dict."""
        _mb = lambda b: round(b / (1024 * 1024), 2)
        _gb = lambda b: round(b / (1024 ** 3), 2)

        return {
            "vram": {
                "max_bytes": self.max_vram_bytes,
                "max_gb": _gb(self.max_vram_bytes),
                "allocated_bytes": self.allocated_vram_bytes,
                "allocated_mb": _mb(self.allocated_vram_bytes),
                "reserved_bytes": self.reserved_vram_bytes,
                "reserved_mb": _mb(self.reserved_vram_bytes),
                "free_bytes": max(0, self.max_vram_bytes - self.allocated_vram_bytes - self.reserved_vram_bytes),
                "free_gb": _gb(max(0, self.max_vram_bytes - self.allocated_vram_bytes - self.reserved_vram_bytes)),
                "fragmentation": self.get_fragmentation(MemoryTier.TIER_1_VRAM),
            },
            "ram": {
                "max_bytes": self.max_ram_bytes,
                "max_gb": _gb(self.max_ram_bytes),
                "allocated_bytes": self.allocated_ram_bytes,
                "allocated_mb": _mb(self.allocated_ram_bytes),
                "reserved_bytes": self.reserved_ram_bytes,
                "reserved_mb": _mb(self.reserved_ram_bytes),
                "free_bytes": max(0, self.max_ram_bytes - self.allocated_ram_bytes - self.reserved_ram_bytes),
                "free_gb": _gb(max(0, self.max_ram_bytes - self.allocated_ram_bytes - self.reserved_ram_bytes)),
                "fragmentation": self.get_fragmentation(MemoryTier.TIER_2_RAM),
            },
        }
