"""
kv_estimator.py
---------------
Formula-based KV cache growth projections for Phase 6 Runtime Memory Optimization.

The KV cache stores the Key and Value tensors for every token in the context
window, for every transformer layer. Its size grows linearly with:
  - Number of transformer layers placed on GPU
  - Context length (tokens stored so far)
  - KV head count (GQA reduces this vs full MHA)
  - Head dimension
  - Data type (fp16 = 2 bytes per element)

KV cache formula per layer per token:
    kv_bytes = 2 × num_kv_heads × head_dim × dtype_bytes
               │   └──── K and V ────┘

For the full GPU-placed context:
    total_kv = kv_bytes_per_layer_per_token × n_gpu_layers × context_length

GQA awareness
-------------
Models using Grouped Query Attention (Llama-3, Mistral, Phi-3) have
num_kv_heads < num_heads. The KV cache is proportionally smaller by factor
(num_kv_heads / num_heads) compared to full MHA. This module correctly
computes the GQA-reduced KV size from the ModelDescriptor.

Context-length alignment
------------------------
llama.cpp rounds context lengths internally to a multiple of 512 (the
GGML block quant alignment). max_safe_context() returns values floored to
the nearest 512-token boundary to avoid off-by-one OOM errors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from layer_placement.model_descriptor import ModelDescriptor, LAYER_TYPE_TRANSFORMER
from layer_placement.placement_plan import PlacementPlan, PlacementDevice


# ---------------------------------------------------------------------------
# KvEstimate output dataclass
# ---------------------------------------------------------------------------

@dataclass
class KvEstimate:
    """
    KV cache projections for a specific model / plan / context combination.

    Attributes
    ----------
    kv_bytes_per_token_per_layer : int
        KV cache bytes added per new token per GPU transformer layer.
        Formula: ``2 × num_kv_heads × head_dim × dtype_bytes``.
    kv_bytes_per_token : int
        Total KV bytes added per new token across all GPU-placed layers.
    kv_at_quarter_context : int
        KV cache bytes at 25% context fill.
    kv_at_half_context : int
        KV cache bytes at 50% context fill.
    kv_at_full_context : int
        KV cache bytes at 100% context fill (worst case).
    kv_growth_rate_gb_per_1k : float
        GB of KV cache added per 1000 tokens generated.
    context_length : int
        Context length used for this estimate.
    n_gpu_layers : int
        Number of GPU-placed transformer layers contributing to VRAM KV.
    num_kv_heads : int
        KV head count (may be less than num_heads for GQA models).
    head_dim : int
        Dimension per attention head.
    dtype_bytes : int
        Bytes per element (2 for fp16, 4 for fp32).
    per_layer_breakdown : List[dict]
        Per-layer KV byte breakdown (layer_index, layer_type, kv_bytes_at_full).
    """
    kv_bytes_per_token_per_layer: int
    kv_bytes_per_token: int
    kv_at_quarter_context: int
    kv_at_half_context: int
    kv_at_full_context: int
    kv_growth_rate_gb_per_1k: float
    context_length: int
    n_gpu_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int
    per_layer_breakdown: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict."""
        def _gb(b: int) -> float:
            return round(b / (1024 ** 3), 4)

        return {
            "kv_bytes_per_token_per_layer": self.kv_bytes_per_token_per_layer,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_at_quarter_context_gb": _gb(self.kv_at_quarter_context),
            "kv_at_half_context_gb": _gb(self.kv_at_half_context),
            "kv_at_full_context_gb": _gb(self.kv_at_full_context),
            "kv_growth_rate_gb_per_1k": round(self.kv_growth_rate_gb_per_1k, 4),
            "context_length": self.context_length,
            "n_gpu_layers": self.n_gpu_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
        }


# ---------------------------------------------------------------------------
# KvCacheEstimator
# ---------------------------------------------------------------------------

class KvCacheEstimator:
    """
    Computes KV cache size projections for a given model and placement plan.

    All computations are purely analytical — no subprocess or allocation needed.

    Parameters
    ----------
    dtype_bytes : int
        Bytes per KV element. 2 = fp16 (default), 4 = fp32.
        llama.cpp uses fp16 KV cache by default (``--flash-attn`` may use fp16
        internally; ``--kv-type f32`` would use fp32).
    """

    # llama.cpp aligns context to this granularity internally
    _CONTEXT_ALIGNMENT = 512

    def __init__(self, dtype_bytes: int = 2) -> None:
        self.dtype_bytes = max(1, dtype_bytes)

    def estimate(
        self,
        model: ModelDescriptor,
        plan: PlacementPlan,
        context_length: int,
    ) -> KvEstimate:
        """
        Compute KV cache projections for the given model, plan, and context.

        Parameters
        ----------
        model : ModelDescriptor
            Full model descriptor from Phase 3 (provides num_kv_heads, head_dim).
        plan : PlacementPlan
            Phase 3 placement plan (determines which layers are on GPU).
        context_length : int
            Target context window in tokens.

        Returns
        -------
        KvEstimate
            Complete KV cache projections.
        """
        context_length = max(1, context_length)

        # Count GPU-placed transformer layers
        gpu_transformer_layers = [
            lp for lp in plan.layer_placements
            if lp.device == PlacementDevice.GPU and lp.layer_type == LAYER_TYPE_TRANSFORMER
        ]
        n_gpu_transformer = len(gpu_transformer_layers)

        # KV cache formula (GQA-aware):
        #   2 (K+V) × num_kv_heads × head_dim × dtype_bytes
        kv_per_token_per_layer = 2 * model.num_kv_heads * model.head_dim * self.dtype_bytes

        # Total KV per token across all GPU layers
        kv_per_token = kv_per_token_per_layer * n_gpu_transformer

        # Projections at different fill levels
        kv_at_full = kv_per_token * context_length
        kv_at_half = kv_per_token * (context_length // 2)
        kv_at_quarter = kv_per_token * (context_length // 4)

        # Growth rate in GB per 1000 tokens
        growth_rate_gb_per_1k = (kv_per_token * 1000) / (1024 ** 3)

        # Per-layer breakdown
        breakdown = []
        for lp in gpu_transformer_layers:
            breakdown.append({
                "layer_index": lp.layer_index,
                "layer_type": lp.layer_type,
                "kv_bytes_per_token": kv_per_token_per_layer,
                "kv_bytes_at_full_context": kv_per_token_per_layer * context_length,
            })

        return KvEstimate(
            kv_bytes_per_token_per_layer=kv_per_token_per_layer,
            kv_bytes_per_token=kv_per_token,
            kv_at_quarter_context=kv_at_quarter,
            kv_at_half_context=kv_at_half,
            kv_at_full_context=kv_at_full,
            kv_growth_rate_gb_per_1k=growth_rate_gb_per_1k,
            context_length=context_length,
            n_gpu_layers=n_gpu_transformer,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            dtype_bytes=self.dtype_bytes,
            per_layer_breakdown=breakdown,
        )

    def max_safe_context(
        self,
        model: ModelDescriptor,
        plan: PlacementPlan,
        vram_headroom_bytes: int,
    ) -> int:
        """
        Compute the maximum context length whose full KV cache fits within
        ``vram_headroom_bytes``.

        Result is floored to the nearest multiple of
        ``_CONTEXT_ALIGNMENT`` (512) for llama.cpp compatibility.

        Parameters
        ----------
        model : ModelDescriptor
            Model descriptor.
        plan : PlacementPlan
            Active placement plan.
        vram_headroom_bytes : int
            Available VRAM bytes reserved for KV cache.

        Returns
        -------
        int
            Maximum safe context length (≥ 512, aligned to 512).
        """
        if vram_headroom_bytes <= 0:
            return self._CONTEXT_ALIGNMENT  # minimum possible context

        gpu_transformer_count = sum(
            1 for lp in plan.layer_placements
            if lp.device == PlacementDevice.GPU and lp.layer_type == LAYER_TYPE_TRANSFORMER
        )
        if gpu_transformer_count == 0:
            # No GPU layers → no VRAM KV → unlimited by VRAM
            return 131072  # practical maximum

        kv_per_token_total = (
            2 * model.num_kv_heads * model.head_dim * self.dtype_bytes
            * gpu_transformer_count
        )
        if kv_per_token_total == 0:
            return 131072

        max_tokens = vram_headroom_bytes // kv_per_token_total
        # Floor to alignment boundary, minimum 512
        aligned = (max(1, max_tokens) // self._CONTEXT_ALIGNMENT) * self._CONTEXT_ALIGNMENT
        return max(self._CONTEXT_ALIGNMENT, aligned)

    def estimate_from_params(
        self,
        num_kv_heads: int,
        head_dim: int,
        n_gpu_layers: int,
        context_length: int,
    ) -> KvEstimate:
        """
        Lightweight estimation without requiring a full ModelDescriptor.
        Useful for quick ad-hoc checks.

        Parameters
        ----------
        num_kv_heads : int
            Number of KV attention heads.
        head_dim : int
            Dimension per head.
        n_gpu_layers : int
            Number of GPU transformer layers.
        context_length : int
            Target context window.

        Returns
        -------
        KvEstimate
            KV projections with no per-layer breakdown.
        """
        context_length = max(1, context_length)
        kv_per_token_per_layer = 2 * num_kv_heads * head_dim * self.dtype_bytes
        kv_per_token = kv_per_token_per_layer * n_gpu_layers

        return KvEstimate(
            kv_bytes_per_token_per_layer=kv_per_token_per_layer,
            kv_bytes_per_token=kv_per_token,
            kv_at_quarter_context=kv_per_token * (context_length // 4),
            kv_at_half_context=kv_per_token * (context_length // 2),
            kv_at_full_context=kv_per_token * context_length,
            kv_growth_rate_gb_per_1k=(kv_per_token * 1000) / (1024 ** 3),
            context_length=context_length,
            n_gpu_layers=n_gpu_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=self.dtype_bytes,
        )
