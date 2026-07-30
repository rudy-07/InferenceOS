"""
placement_plan.py
-----------------
Output data structures for the Phase 3 Layer Placement Engine.

A PlacementPlan describes exactly where every model layer should reside
(GPU or CPU), groups them into contiguous segments, and stores the
per-layer cost breakdown produced by the optimizer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PlacementDevice(str, Enum):
    """Target execution device for a model layer."""
    GPU  = "GPU"
    CPU  = "CPU"
    IGPU = "IGPU"   # Phase 7: integrated GPU

    def __str__(self) -> str:
        return self.value


@dataclass
class LayerCostBreakdown:
    """
    Recorded cost components for a single layer's placement decision.

    Attributes
    ----------
    memory_pressure_cost : float
        Penalty [0, ∞) for pushing the target tier toward capacity.
    transfer_cost : float
        Penalty for crossing a GPU↔CPU boundary adjacent to this layer.
    compute_cost : float
        Relative slowdown vs. ideal all-GPU placement (0.0 = GPU speed).
    total_cost : float
        Weighted composite of the three sub-costs.
    """
    memory_pressure_cost: float = 0.0
    transfer_cost: float = 0.0
    compute_cost: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "memory_pressure_cost": round(self.memory_pressure_cost, 6),
            "transfer_cost": round(self.transfer_cost, 6),
            "compute_cost": round(self.compute_cost, 6),
            "total_cost": round(self.total_cost, 6),
        }


@dataclass
class LayerPlacement:
    """
    Placement decision for a single model layer.

    Attributes
    ----------
    layer_index : int
        Zero-based layer index within the full layer list.
    layer_type : str
        Functional type of this layer (transformer, embedding, lm_head, norm).
    device : PlacementDevice
        Where this layer will be executed (GPU or CPU).
    gpu_index : int, optional
        Index of the target GPU for GPU-placed layers; None for CPU.
    size_bytes : int
        Weight memory footprint of this layer.
    cost : LayerCostBreakdown
        Cost breakdown for this layer's placement.
    """
    layer_index: int
    layer_type: str
    device: PlacementDevice
    gpu_index: Optional[int]
    size_bytes: int
    cost: LayerCostBreakdown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_type": self.layer_type,
            "device": str(self.device),
            "gpu_index": self.gpu_index,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "cost": self.cost.to_dict(),
        }


@dataclass
class PlacementSegment:
    """
    A contiguous run of layers placed on the same device.

    Used for human-readable summaries and llama.cpp hint generation.

    Attributes
    ----------
    device : PlacementDevice
        Device this segment runs on.
    start_layer : int
        Inclusive first layer index (in overall layer list).
    end_layer : int
        Inclusive last layer index (in overall layer list).
    gpu_index : int, optional
        GPU index for GPU segments; None for CPU.
    total_size_bytes : int
        Combined weight footprint of all layers in this segment.
    layer_count : int
        Number of layers in this segment.
    """
    device: PlacementDevice
    start_layer: int
    end_layer: int
    gpu_index: Optional[int]
    total_size_bytes: int
    layer_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "gpu_index": self.gpu_index,
            "layer_count": self.layer_count,
            "total_size_bytes": self.total_size_bytes,
            "total_size_mb": round(self.total_size_bytes / (1024 * 1024), 2),
            "total_size_gb": round(self.total_size_bytes / (1024 ** 3), 3),
        }

    def label(self) -> str:
        """Human label: 'GPU (idx 0)', 'iGPU (idx 0)', or 'CPU'."""
        if self.device == PlacementDevice.GPU and self.gpu_index is not None:
            return f"GPU (device {self.gpu_index})"
        if self.device == PlacementDevice.IGPU and self.gpu_index is not None:
            return f"iGPU (device {self.gpu_index})"
        return "CPU"


@dataclass
class PlacementPlan:
    """
    The complete optimization result from the layer placement engine.

    Attributes
    ----------
    model_name : str
        Human-readable name of the model that was planned.
    architecture : str
        Architecture family (llama, mistral, etc.)
    total_layers : int
        Total layer count (including embedding, lm_head).
    n_gpu_layers : int
        Number of transformer blocks placed on GPU.
    n_cpu_layers : int
        Number of transformer blocks placed on CPU.
    gpu_layer_indices : List[int]
        Layer indices (within full list) placed on GPU.
    cpu_layer_indices : List[int]
        Layer indices (within full list) placed on CPU.
    layer_placements : List[LayerPlacement]
        Full per-layer placement decisions (ordered by layer_index).
    segments : List[PlacementSegment]
        Contiguous same-device runs (for human-readable display).
    total_cost : float
        Composite optimizer cost of this plan (lower is better).
    optimizer_iterations : int
        Number of simulated annealing iterations performed.
    estimated_vram_bytes : int
        Total VRAM bytes consumed by GPU-placed layers + KV cache.
    estimated_ram_bytes : int
        Total RAM bytes consumed by CPU-placed layers + KV cache.
    estimated_peak_bytes : int
        Peak total memory (VRAM + RAM + workspace scratch).
    context_length : int
        Context window length used for KV cache estimates.
    is_feasible : bool
        True if the plan fits within physical memory bounds.
    warnings : List[str]
        Non-fatal warnings (e.g. near-capacity tiers, fallback decisions).
    """
    model_name: str
    architecture: str
    total_layers: int
    n_gpu_layers: int
    n_cpu_layers: int
    # Phase 7: iGPU fields (default 0 / empty for backward compatibility)
    n_igpu_layers: int = 0
    gpu_layer_indices: List[int] = field(default_factory=list)
    igpu_layer_indices: List[int] = field(default_factory=list)
    cpu_layer_indices: List[int] = field(default_factory=list)
    layer_placements: List[LayerPlacement] = field(default_factory=list)
    segments: List[PlacementSegment] = field(default_factory=list)
    total_cost: float = 0.0
    optimizer_iterations: int = 0
    estimated_vram_bytes: int = 0
    estimated_igpu_vram_bytes: int = 0  # Phase 7
    estimated_ram_bytes: int = 0
    estimated_peak_bytes: int = 0
    context_length: int = 4096
    is_feasible: bool = True
    warnings: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------------------
    # Derived convenience properties
    # ---------------------------------------------------------------------------

    @property
    def gpu_offload_ratio(self) -> float:
        """Fraction of transformer blocks placed on dGPU (0.0–1.0)."""
        total_transformer = self.n_gpu_layers + self.n_cpu_layers
        if total_transformer == 0:
            return 0.0
        return self.n_gpu_layers / total_transformer

    @property
    def igpu_offload_ratio(self) -> float:
        """Fraction of all layers placed on iGPU (0.0–1.0). Phase 7."""
        if self.total_layers == 0:
            return 0.0
        return self.n_igpu_layers / self.total_layers

    @property
    def boundary_crossings(self) -> int:
        """Number of device transition boundaries (GPU↔CPU↔iGPU) in layer sequence."""
        if not self.layer_placements:
            return 0
        crossings = 0
        prev_device = self.layer_placements[0].device
        for lp in self.layer_placements[1:]:
            if lp.device != prev_device:
                crossings += 1
            prev_device = lp.device
        return crossings

    @property
    def llama_cpp_n_gpu_layers_hint(self) -> int:
        """
        Equivalent ``--n-gpu-layers`` value for llama.cpp.

        If all transformer layers are assigned to GPU, return 999 so llama.cpp
        offloads non-transformer layers (token_embd, output lm_head) to GPU VRAM.
        Otherwise, returns count of the contiguous GPU-starting segment.
        """
        if self.n_cpu_layers == 0 and getattr(self, "n_igpu_layers", 0) == 0:
            return 999

        count = 0
        for lp in self.layer_placements:
            if lp.layer_type not in ("transformer",):
                continue
            if lp.device == PlacementDevice.GPU:
                count += 1
            else:
                break
        return count

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization to a JSON-safe dict."""
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "total_layers": self.total_layers,
            "n_gpu_layers": self.n_gpu_layers,
            "n_igpu_layers": self.n_igpu_layers,
            "n_cpu_layers": self.n_cpu_layers,
            "gpu_offload_ratio": round(self.gpu_offload_ratio, 4),
            "igpu_offload_ratio": round(self.igpu_offload_ratio, 4),
            "boundary_crossings": self.boundary_crossings,
            "llama_cpp_n_gpu_layers_hint": self.llama_cpp_n_gpu_layers_hint,
            "context_length": self.context_length,
            "memory": {
                "estimated_vram_bytes": self.estimated_vram_bytes,
                "estimated_vram_gb": round(self.estimated_vram_bytes / (1024 ** 3), 3),
                "estimated_igpu_vram_bytes": self.estimated_igpu_vram_bytes,
                "estimated_igpu_vram_mb": self.estimated_igpu_vram_bytes // (1024 * 1024),
                "estimated_ram_bytes": self.estimated_ram_bytes,
                "estimated_ram_gb": round(self.estimated_ram_bytes / (1024 ** 3), 3),
                "estimated_peak_bytes": self.estimated_peak_bytes,
                "estimated_peak_gb": round(self.estimated_peak_bytes / (1024 ** 3), 3),
            },
            "optimizer": {
                "total_cost": round(self.total_cost, 6),
                "iterations": self.optimizer_iterations,
            },
            "is_feasible": self.is_feasible,
            "warnings": self.warnings,
            "segments": [s.to_dict() for s in self.segments],
            "layer_placements": [lp.to_dict() for lp in self.layer_placements],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """
        One-line placement summary (useful for logging / CLI output).

        Example output::

            Llama-3-8B  |  GPU: 48/80  iGPU: 3/80  CPU: 29/80  |  2 boundaries
        """
        total_transformer = self.n_gpu_layers + self.n_cpu_layers
        pct = self.gpu_offload_ratio * 100.0
        feasible_str = "" if self.is_feasible else " [INFEASIBLE]"
        igpu_str = f"  iGPU: {self.n_igpu_layers}" if self.n_igpu_layers > 0 else ""
        return (
            f"{self.model_name}  |  GPU: {self.n_gpu_layers}/{total_transformer} layers "
            f"({pct:.1f}%){igpu_str}  |  {self.boundary_crossings} boundary crossings  |  "
            f"cost={self.total_cost:.4f}{feasible_str}"
        )


# ---------------------------------------------------------------------------
# Internal helper: build segments from a flat placement list
# ---------------------------------------------------------------------------

def build_segments(placements: List[LayerPlacement]) -> List[PlacementSegment]:
    """
    Collapse a flat :class:`LayerPlacement` list into contiguous
    :class:`PlacementSegment` runs of the same device.

    Parameters
    ----------
    placements : List[LayerPlacement]
        Ordered (by layer_index) placement decisions.

    Returns
    -------
    List[PlacementSegment]
        Ordered list of segments, each grouping a contiguous same-device run.
    """
    if not placements:
        return []

    segments: List[PlacementSegment] = []
    current = placements[0]
    seg_start = current.layer_index
    seg_bytes = current.size_bytes
    seg_count = 1

    for lp in placements[1:]:
        same_device = lp.device == current.device
        same_gpu = lp.gpu_index == current.gpu_index
        if same_device and (lp.device == PlacementDevice.CPU or same_gpu):
            # Extend current segment
            seg_bytes += lp.size_bytes
            seg_count += 1
        else:
            # Flush current segment
            segments.append(PlacementSegment(
                device=current.device,
                start_layer=seg_start,
                end_layer=current.layer_index,
                gpu_index=current.gpu_index,
                total_size_bytes=seg_bytes,
                layer_count=seg_count,
            ))
            # Start new segment
            seg_start = lp.layer_index
            seg_bytes = lp.size_bytes
            seg_count = 1
        current = lp

    # Flush final segment
    segments.append(PlacementSegment(
        device=current.device,
        start_layer=seg_start,
        end_layer=current.layer_index,
        gpu_index=current.gpu_index,
        total_size_bytes=seg_bytes,
        layer_count=seg_count,
    ))

    return segments
