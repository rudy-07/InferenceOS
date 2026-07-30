"""
context_heuristics.py
---------------------
Analytical memory prediction engine and candidate context scoring for the Dynamic Context Scheduler in InferenceOS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from runtime_memory.kv_estimator import KvCacheEstimator


@dataclass
class CandidateContextAssessment:
    """Detailed score and memory evaluation for a single candidate context length."""
    context_length: int
    total_score: float
    is_safe: bool
    estimated_kv_memory_mb: float
    estimated_total_memory_mb: float
    vram_headroom_score: float
    intent_satisfaction_score: float
    performance_score: float
    rejection_reason: Optional[str] = None
    reason_details: Optional[str] = None


class ContextScorer:
    """
    Analytical GQA-aware memory prediction and candidate context scoring engine.
    """

    def __init__(self, dtype_bytes: int = 2) -> None:
        self.kv_estimator = KvCacheEstimator(dtype_bytes=dtype_bytes)

    def evaluate_candidate(
        self,
        candidate_context: int,
        requested_context: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        n_gpu_layers: int,
        n_cpu_layers: int,
        hidden_size: int,
        model_vram_base_mb: float,
        model_ram_base_mb: float,
        free_vram_mb: float,
        free_ram_mb: float,
        total_vram_mb: float,
        total_ram_mb: float,
        microbatch_size: int = 512,
        safety_margin: float = 0.15,
        optimization_goal: str = "maximum_context",
        backend_overhead_mb: float = 500.0,
    ) -> CandidateContextAssessment:
        """
        Evaluate KV cache memory size and safety score for a candidate context length.
        """
        # 1. Compute GQA-aware KV cache size using KvCacheEstimator
        kv_estimate = self.kv_estimator.estimate_from_params(
            num_kv_heads=max(1, num_kv_heads),
            head_dim=max(1, head_dim),
            n_gpu_layers=max(0, n_gpu_layers),
            context_length=candidate_context,
        )
        kv_vram_mb = kv_estimate.kv_at_full_context / (1024 * 1024)

        # CPU layers KV cache
        cpu_kv_per_token_layer = 2 * max(1, num_kv_heads) * max(1, head_dim) * self.kv_estimator.dtype_bytes
        kv_ram_mb = (cpu_kv_per_token_layer * max(0, n_cpu_layers) * candidate_context) / (1024 * 1024)

        # 2. Prefill activation buffer for microbatch
        bytes_per_token_layer = max(2048, hidden_size * 4)
        activation_vram_mb = (microbatch_size * bytes_per_token_layer * max(1, n_gpu_layers)) / (1024 * 1024)
        activation_ram_mb = (microbatch_size * bytes_per_token_layer * max(0, n_cpu_layers)) / (1024 * 1024)

        # 3. Total memory estimates
        est_total_vram_mb = model_vram_base_mb + kv_vram_mb + activation_vram_mb + backend_overhead_mb
        est_total_ram_mb = model_ram_base_mb + kv_ram_mb + activation_ram_mb

        # 4. Safety Check
        safe_vram_limit = free_vram_mb * (1.0 - safety_margin) if free_vram_mb > 0 else (total_vram_mb * (1.0 - safety_margin))
        safe_ram_limit = free_ram_mb * 0.90 if free_ram_mb > 0 else (total_ram_mb * 0.90)

        if free_vram_mb > 0 and est_total_vram_mb > safe_vram_limit:
            return CandidateContextAssessment(
                context_length=candidate_context,
                total_score=0.0,
                is_safe=False,
                estimated_kv_memory_mb=kv_vram_mb,
                estimated_total_memory_mb=est_total_vram_mb,
                vram_headroom_score=0.0,
                intent_satisfaction_score=0.0,
                performance_score=0.0,
                rejection_reason="Exceeds safe VRAM budget",
                reason_details=f"Est. {est_total_vram_mb / 1024.0:.2f}GB VRAM exceeds safe limit ({safe_vram_limit / 1024.0:.2f}GB)",
            )

        if free_ram_mb > 0 and est_total_ram_mb > safe_ram_limit:
            return CandidateContextAssessment(
                context_length=candidate_context,
                total_score=0.0,
                is_safe=False,
                estimated_kv_memory_mb=kv_vram_mb,
                estimated_total_memory_mb=est_total_vram_mb,
                vram_headroom_score=0.0,
                intent_satisfaction_score=0.0,
                performance_score=0.0,
                rejection_reason="Exceeds safe System RAM budget",
                reason_details=f"Est. {est_total_ram_mb / 1024.0:.2f}GB RAM exceeds safe limit ({safe_ram_limit / 1024.0:.2f}GB)",
            )

        # 5. Compute Sub-Scores for Safe Candidates
        # Sub-score A: VRAM Headroom Score (0-100)
        if free_vram_mb > 0:
            headroom_ratio = (free_vram_mb - est_total_vram_mb) / max(1.0, free_vram_mb)
        else:
            headroom_ratio = 0.5
        vram_headroom_score = max(0.0, min(100.0, headroom_ratio * 100.0 * 1.2))

        # Sub-score B: Intent Satisfaction Score (0-100)
        intent_ratio = min(1.0, candidate_context / float(max(1, requested_context)))
        intent_satisfaction_score = max(10.0, intent_ratio * 100.0)

        # Sub-score C: Performance & Speed Score (0-100)
        # Smaller context = faster attention matrix compute & lower memory bandwidth pressure
        perf_score = max(10.0, 100.0 - (candidate_context / float(max(1, requested_context))) * 40.0)

        # Weighting based on optimization goal
        if optimization_goal == "maximum_context":
            w_intent = 0.60
            w_headroom = 0.30
            w_perf = 0.10
        elif optimization_goal == "maximum_speed":
            w_intent = 0.20
            w_headroom = 0.40
            w_perf = 0.40
        else:  # balanced
            w_intent = 0.45
            w_headroom = 0.35
            w_perf = 0.20

        raw_score = (intent_satisfaction_score * w_intent) + (vram_headroom_score * w_headroom) + (perf_score * w_perf)
        total_score = max(1.0, min(100.0, raw_score))

        reason_details = (
            f"KV Mem {kv_vram_mb / 1024.0:.2f}GB, Headroom Score {vram_headroom_score:.0f}, "
            f"Intent Score {intent_satisfaction_score:.0f}"
        )

        return CandidateContextAssessment(
            context_length=candidate_context,
            total_score=total_score,
            is_safe=True,
            estimated_kv_memory_mb=kv_vram_mb,
            estimated_total_memory_mb=est_total_vram_mb,
            vram_headroom_score=vram_headroom_score,
            intent_satisfaction_score=intent_satisfaction_score,
            performance_score=perf_score,
            rejection_reason=None,
            reason_details=reason_details,
        )
