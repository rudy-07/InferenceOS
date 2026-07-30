"""
tensor.py
---------
Tensor metadata model for tracking memory placement, preferred tiers,
allowed tiers, size, and access frequency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .memory_tier import MemoryTier


@dataclass
class TensorMetadata:
    """
    Metadata describing a model tensor's size, memory placement, and access rate.
    """
    tensor_id: str
    size_bytes: int
    current_location: MemoryTier
    preferred_location: MemoryTier = MemoryTier.TIER_1_VRAM
    allowed_locations: List[MemoryTier] = field(
        default_factory=lambda: [MemoryTier.TIER_1_VRAM, MemoryTier.TIER_2_RAM]
    )
    access_frequency: float = 0.0
    dtype: str = "f16"
    shape: Optional[Tuple[int, ...]] = None

    def record_access(self, weight: float = 1.0) -> None:
        """Increment access frequency tracking metric."""
        self.access_frequency += weight

    @property
    def is_in_preferred_location(self) -> bool:
        """Check if tensor currently resides in its preferred memory tier."""
        return self.current_location == self.preferred_location

    def to_dict(self) -> Dict[str, Any]:
        """Convert TensorMetadata to a serializable dictionary."""
        return {
            "tensor_id": self.tensor_id,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "current_location": str(self.current_location),
            "preferred_location": str(self.preferred_location),
            "allowed_locations": [str(loc) for loc in self.allowed_locations],
            "access_frequency": self.access_frequency,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape else None,
        }
