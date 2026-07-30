"""
buffer_planner.py
-----------------
Workspace and activation buffer sizing for Phase 6 Runtime Memory Optimization.

Computes the VRAM overhead beyond model weights that llama.cpp requires
for a single inference session:

  1. KV cache reservation  — reserved VRAM block for the full context KV
  2. Activation workspace  — rolling buffer for current-token activations
  3. Backend overhead      — GPU driver + CUDA/Vulkan context memory

These are separate from the model weights tracked by the Phase 3 PlacementPlan.

Activation workspace formula
-----------------------------
The activation buffer holds one layer's forward-pass intermediate values
at any point (pipeline-style execution):

    activation_bytes = hidden_size × batch_size × dtype_bytes × safety_factor

The safety_factor of 1.5 accounts for:
  - Attention score buffers (QK^T, softmax output)
  - FFN intermediate (gated models have 3x FFN tensors)
  - Layer norm buffers

Backend overhead
-----------------
GPU backend context memory is a fixed cost, roughly:
  CUDA: ~350 MB (driver + cuBLAS + cuDNN init + CUDA context)
  Vulkan: ~250 MB (Vulkan device + command pools + descriptor sets)
  CPU: ~10 MB

These are empirical estimates; actual values vary by driver version and model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from layer_placement.model_descriptor import ModelDescriptor
from layer_placement.placement_plan import PlacementPlan, PlacementDevice
from layer_placement.model_descriptor import LAYER_TYPE_TRANSFORMER

from .kv_estimator import KvCacheEstimator


# ---------------------------------------------------------------------------
# Backend overhead constants (bytes)
# ---------------------------------------------------------------------------

_BACKEND_OVERHEAD: Dict[str, int] = {
    "cuda":   350 * 1024 * 1024,   # 350 MB
    "vulkan": 250 * 1024 * 1024,   # 250 MB
    "metal":  200 * 1024 * 1024,   # 200 MB (Apple Silicon unified memory)
    "cpu":     10 * 1024 * 1024,   #  10 MB
    "unknown": 300 * 1024 * 1024,  # 300 MB safe default
}

# KV cache reservation safety multiplier (reserve slightly more than estimated
# to absorb speculative decoding / batch size variability)
_KV_RESERVATION_FACTOR = 1.05

# Activation workspace safety factor
_ACTIVATION_SAFETY_FACTOR = 1.5


# ---------------------------------------------------------------------------
# BufferPlan dataclass
# ---------------------------------------------------------------------------

@dataclass
class BufferPlan:
    """
    Memory buffer requirements for a single inference session.

    Attributes
    ----------
    activation_workspace_bytes : int
        Rolling activation buffer (one layer's intermediate values).
    backend_overhead_bytes : int
        GPU backend context and driver reservation.
    kv_reservation_bytes : int
        Reserved VRAM block for the KV cache at full context length.
    total_reserved_bytes : int
        Sum of all buffer components (non-weight overhead).
    backend : str
        Backend name used for overhead estimation.
    batch_size : int
        Batch size used for activation workspace calculation.
    dtype_bytes : int
        Data type size (2=fp16).
    """
    activation_workspace_bytes: int
    backend_overhead_bytes: int
    kv_reservation_bytes: int
    total_reserved_bytes: int
    backend: str
    batch_size: int
    dtype_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        def _mb(b: int) -> float:
            return round(b / (1024 * 1024), 2)

        return {
            "activation_workspace_mb": _mb(self.activation_workspace_bytes),
            "backend_overhead_mb": _mb(self.backend_overhead_bytes),
            "kv_reservation_mb": _mb(self.kv_reservation_bytes),
            "total_reserved_mb": _mb(self.total_reserved_bytes),
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# BufferPlanner
# ---------------------------------------------------------------------------

class BufferPlanner:
    """
    Computes activation workspace, KV reservation, and backend overhead
    for a given model / plan / context configuration.

    Parameters
    ----------
    dtype_bytes : int
        Bytes per element for KV and activation buffers. Default 2 (fp16).
    """

    def __init__(self, dtype_bytes: int = 2) -> None:
        self.dtype_bytes = max(1, dtype_bytes)
        self._kv_estimator = KvCacheEstimator(dtype_bytes=dtype_bytes)

    def plan(
        self,
        model: ModelDescriptor,
        placement_plan: PlacementPlan,
        context_length: int,
        batch_size: int = 512,
        backend: str = "unknown",
    ) -> BufferPlan:
        """
        Compute buffer requirements for the given configuration.

        Parameters
        ----------
        model : ModelDescriptor
            Full model descriptor (provides hidden_size for activation calc).
        placement_plan : PlacementPlan
            Phase 3 plan (determines GPU layer count for KV reservation).
        context_length : int
            Target context window in tokens.
        batch_size : int
            Prompt processing batch size. Default 512.
        backend : str
            Backend name for overhead lookup (``"cuda"``, ``"vulkan"``,
            ``"metal"``, ``"cpu"``). Default ``"unknown"``.

        Returns
        -------
        BufferPlan
            Complete buffer sizing result.
        """
        context_length = max(1, context_length)
        batch_size = max(1, batch_size)

        # 1. KV cache reservation
        kv_est = self._kv_estimator.estimate(model, placement_plan, context_length)
        kv_reservation = int(kv_est.kv_at_full_context * _KV_RESERVATION_FACTOR)

        # 2. Activation workspace
        activation = self._compute_activation_workspace(
            model=model,
            plan=placement_plan,
            batch_size=batch_size,
        )

        # 3. Backend overhead
        overhead = self._get_backend_overhead(backend)

        total = kv_reservation + activation + overhead

        return BufferPlan(
            activation_workspace_bytes=activation,
            backend_overhead_bytes=overhead,
            kv_reservation_bytes=kv_reservation,
            total_reserved_bytes=total,
            backend=backend,
            batch_size=batch_size,
            dtype_bytes=self.dtype_bytes,
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _compute_activation_workspace(
        self,
        model: ModelDescriptor,
        plan: PlacementPlan,
        batch_size: int,
    ) -> int:
        """
        Estimate the rolling activation buffer size.

        Formula:
            hidden_size × batch_size × dtype_bytes × _ACTIVATION_SAFETY_FACTOR
        """
        raw = model.hidden_size * batch_size * self.dtype_bytes
        return int(raw * _ACTIVATION_SAFETY_FACTOR)

    @staticmethod
    def _get_backend_overhead(backend: str) -> int:
        """Return backend GPU context overhead in bytes."""
        key = backend.lower().strip()
        return _BACKEND_OVERHEAD.get(key, _BACKEND_OVERHEAD["unknown"])
