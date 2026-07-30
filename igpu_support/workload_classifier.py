"""
workload_classifier.py
----------------------
Per-layer iGPU workload suitability classification for Phase 7.

Different transformer layer types have very different compute and memory
access patterns. The iGPU (particularly a shared-memory UMA device) is
well-suited to some workloads and harmful for others.

Workload taxonomy
-----------------

Layer type      | Compute intensity | Memory footprint | iGPU suitable?
─────────────── | ─────────────────── | ──────────────── | ──────────────
embedding       | Very low (lookup) | Small (vocab BW) | ✅ YES (always)
norm            | Very low (eltwise) | Tiny             | ✅ YES (always)
lm_head         | Low (one-shot)    | Medium           | ✅ YES (always)
transformer     | High (GEMM+attn)  | Large            | ⚠️ CONDITIONAL

For transformer blocks the decision depends on:
  1. Suitability level: must be FULL (not LIGHT) to allow transformer layers
  2. Layer size: must fit within the iGPU VRAM budget with headroom
  3. Effective bandwidth: must exceed COMPUTE_BW_THRESHOLD

Size constraints
----------------
  light_max_layer_bytes = igpu_vram_bytes × _LIGHT_BUDGET_FRACTION (0.30)
  full_max_layer_bytes  = igpu_vram_bytes × _FULL_BUDGET_FRACTION  (0.60)

Only transformer layers where size_bytes ≤ max_layer_bytes are assigned.
This prevents a single large block from filling all iGPU VRAM.

Compute bandwidth threshold
----------------------------
  Transformer layers on iGPU require at least _COMPUTE_BW_MIN_GBPS (10 GB/s)
  effective bandwidth to be worthwhile.  If contention makes effective BW
  fall below this threshold, transformer layers are not assigned.
"""
from __future__ import annotations

from enum import Enum
from typing import Set

from layer_placement.model_descriptor import (
    LAYER_TYPE_EMBEDDING,
    LAYER_TYPE_LM_HEAD,
    LAYER_TYPE_NORM,
    LAYER_TYPE_TRANSFORMER,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of total iGPU VRAM a single layer may occupy (light / full modes)
_LIGHT_BUDGET_FRACTION = 0.30
_FULL_BUDGET_FRACTION  = 0.60

# Minimum effective bandwidth (GB/s) to allow compute-heavy transformer layers
_COMPUTE_BW_MIN_GBPS = 10.0


# ---------------------------------------------------------------------------
# IgpuWorkloadSuitability
# ---------------------------------------------------------------------------

class IgpuWorkloadSuitability(str, Enum):
    """
    Three-level iGPU suitability descriptor.

    FULL
        iGPU is suitable for all workload types including transformer blocks
        (subject to per-layer size and bandwidth checks).
    LIGHT
        iGPU is suitable only for light, non-compute-intensive layers:
        embedding, layer norm, and lm_head.
    DISABLED
        iGPU should not be used for any inference workload.
    """
    FULL     = "FULL"
    LIGHT    = "LIGHT"
    DISABLED = "DISABLED"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# WorkloadClassifier
# ---------------------------------------------------------------------------

class WorkloadClassifier:
    """
    Determines per-layer iGPU assignment based on layer type, size, and
    the iGPU's suitability level.

    Layer types always suitable for iGPU (in LIGHT or FULL mode):
        embedding, norm, lm_head

    Layer types only suitable in FULL mode (with size/BW checks):
        transformer

    Parameters
    ----------
    suitability : IgpuWorkloadSuitability
        Overall suitability determined by :class:`SuitabilityEvaluator`.
    igpu_vram_bytes : int
        Total iGPU VRAM bytes (used for budget checks).
    effective_bw_gbps : float
        Effective iGPU bandwidth after bus contention.
    """

    # Layer types always assigned to iGPU when suitability >= LIGHT
    ALWAYS_SUITABLE: Set[str] = {
        LAYER_TYPE_EMBEDDING,
        LAYER_TYPE_NORM,
        LAYER_TYPE_LM_HEAD,
    }
    # Additional types allowed in FULL mode
    FULL_SUITABLE: Set[str] = {
        LAYER_TYPE_TRANSFORMER,
    }

    def __init__(
        self,
        suitability: IgpuWorkloadSuitability,
        igpu_vram_bytes: int,
        effective_bw_gbps: float,
    ) -> None:
        self.suitability = suitability
        self.igpu_vram_bytes = max(1, igpu_vram_bytes)
        self.effective_bw_gbps = effective_bw_gbps

        # Compute per-mode VRAM budget ceilings
        self._light_max_bytes = int(igpu_vram_bytes * _LIGHT_BUDGET_FRACTION)
        self._full_max_bytes  = int(igpu_vram_bytes * _FULL_BUDGET_FRACTION)

    def is_suitable_for_igpu(
        self,
        layer_type: str,
        layer_size_bytes: int,
    ) -> bool:
        """
        Determine whether a single layer should be assigned to the iGPU.

        Parameters
        ----------
        layer_type : str
            One of the LAYER_TYPE_* constants from ``model_descriptor``.
        layer_size_bytes : int
            Weight memory footprint of this layer.

        Returns
        -------
        bool
            ``True`` if this layer should run on the iGPU.
        """
        if self.suitability == IgpuWorkloadSuitability.DISABLED:
            return False

        # Always-suitable types (embedding, norm, lm_head): size check only
        if layer_type in self.ALWAYS_SUITABLE:
            return layer_size_bytes <= self._light_max_bytes

        # Transformer blocks: FULL mode only + bandwidth + size checks
        if layer_type in self.FULL_SUITABLE:
            if self.suitability != IgpuWorkloadSuitability.FULL:
                return False
            if self.effective_bw_gbps < _COMPUTE_BW_MIN_GBPS:
                return False
            return layer_size_bytes <= self._full_max_bytes

        # Unknown layer types: reject
        return False

    def describe(self) -> str:
        """Human-readable description of the current classification policy."""
        lines = [
            f"iGPU Workload Policy: {self.suitability}",
            f"  VRAM budget (light):  {self._light_max_bytes // (1024**2)} MB per layer",
            f"  VRAM budget (full):   {self._full_max_bytes  // (1024**2)} MB per layer",
            f"  Effective bandwidth:  {self.effective_bw_gbps:.1f} GB/s",
            f"  Transformer blocks:   {'Enabled' if self.suitability == IgpuWorkloadSuitability.FULL else 'Disabled'}",
        ]
        return "\n".join(lines)
