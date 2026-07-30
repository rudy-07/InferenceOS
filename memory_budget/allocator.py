"""
allocator.py
------------
Dynamic component memory budget allocator for InferenceOS.
"""
from __future__ import annotations

from .interfaces import ComponentBudgets


class BudgetAllocator:
    """
    Divides available memory across 7 component buckets based on hardware profile and model requirements.
    """

    def allocate(
        self,
        total_vram_mb: float = 8192.0,
        free_vram_mb: float = 6000.0,
        model_weights_mb: float = 4000.0,
        reserve_ratio: float = 0.15,
        policy_mode: str = "adaptive",
    ) -> ComponentBudgets:
        """
        Allocate dynamic component budgets.
        """
        total_vram = max(1024.0, total_vram_mb)

        # 1. Safety Reserve
        safety_reserve = max(400.0, total_vram * reserve_ratio)

        # 2. Model Weights Budget
        weights_mb = min(model_weights_mb, max(0.0, total_vram - safety_reserve))

        # 3. Remaining usable memory for KV, microbatch, buffers
        usable = max(0.0, total_vram - weights_mb - safety_reserve)

        if usable > 2000.0:
            kv_mb = usable * 0.50
            mb_mb = usable * 0.25
            runtime_mb = usable * 0.15
            future_mb = usable * 0.10
        elif usable > 1000.0:
            kv_mb = usable * 0.55
            mb_mb = usable * 0.25
            runtime_mb = usable * 0.20
            future_mb = 0.0
        else:
            kv_mb = usable * 0.60
            mb_mb = usable * 0.30
            runtime_mb = usable * 0.10
            future_mb = 0.0

        return ComponentBudgets(
            weights_mb=round(weights_mb, 1),
            kv_cache_mb=round(kv_mb, 1),
            microbatch_mb=round(mb_mb, 1),
            context_mb=round(kv_mb * 0.8, 1),
            runtime_buffers_mb=round(runtime_mb, 1),
            safety_reserve_mb=round(safety_reserve, 1),
            future_expansion_mb=round(future_mb, 1),
        )
