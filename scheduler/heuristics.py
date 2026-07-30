"""
heuristics.py
-------------
Modular scoring engine and heuristic policies for the Dynamic Microbatch Scheduler.

Each candidate microbatch size receives sub-scores for:
  1. VRAM Memory Safety Headroom
  2. RAM Memory Safety Headroom
  3. GPU Compute Saturation / Occupancy
  4. Predicted Prompt TPS Throughput
  5. Interconnect & PCIe Pipeline Stability

All heuristic functions are isolated to allow future Runtime Learning policies to
override or re-weight individual components cleanly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CandidateAssessment:
    """Detailed score breakdown for a single candidate microbatch."""
    microbatch: int
    total_score: float
    is_safe: bool
    estimated_vram_mb: float
    estimated_ram_mb: float
    vram_headroom_score: float
    ram_headroom_score: float
    gpu_occupancy_score: float
    prompt_tps_score: float
    stability_score: float
    rejection_reason: Optional[str] = None
    reason_details: Optional[str] = None


class HeuristicScorer:
    """
    Modular scoring engine for microbatch candidate evaluation.
    """

    @staticmethod
    def estimate_prefill_vram_mb(
        microbatch: int,
        n_gpu_layers: int,
        hidden_size: int = 4096,
        model_vram_base_mb: float = 0.0,
        kv_cache_vram_mb: float = 0.0,
        backend_overhead_mb: float = 500.0,
    ) -> float:
        """
        Estimate total VRAM consumption (MB) during prefill for candidate microbatch size.

        Components:
          - Model base weights in VRAM
          - KV cache memory in VRAM
          - Backend/CUDA context overhead
          - Scratchpad & temporary activation tensors (scales with microbatch size)
        """
        # Activation and compute graph buffer scales linearly with microbatch size and offloaded layers
        # Standard transformer activation buffer approx: 8 * hidden_size bytes per token layer
        bytes_per_token_layer = max(2048, hidden_size * 4)
        prefill_activation_bytes = microbatch * bytes_per_token_layer * max(1, n_gpu_layers)
        activation_mb = prefill_activation_bytes / (1024 * 1024)

        # Graph execution overhead in llama.cpp scales with ubatch size
        graph_overhead_mb = 12.0 + (microbatch * 0.08)

        total_vram_mb = (
            model_vram_base_mb
            + kv_cache_vram_mb
            + backend_overhead_mb
            + activation_mb
            + graph_overhead_mb
        )
        return total_vram_mb

    @staticmethod
    def estimate_prefill_ram_mb(
        microbatch: int,
        n_cpu_layers: int,
        hidden_size: int = 4096,
        model_ram_base_mb: float = 0.0,
        kv_cache_ram_mb: float = 0.0,
    ) -> float:
        """
        Estimate system RAM consumption (MB) during prefill for candidate microbatch size.
        """
        bytes_per_token_layer = max(2048, hidden_size * 4)
        activation_mb = (microbatch * bytes_per_token_layer * max(0, n_cpu_layers)) / (1024 * 1024)
        total_ram_mb = model_ram_base_mb + kv_cache_ram_mb + activation_mb
        return total_ram_mb

    @classmethod
    def evaluate_candidate(
        self,
        candidate: int,
        total_vram_mb: float,
        free_vram_mb: float,
        total_ram_mb: float,
        free_ram_mb: float,
        n_gpu_layers: int,
        n_cpu_layers: int,
        total_layers: int,
        context_length: int,
        hidden_size: int,
        backend_name: str,
        boundary_crossings: int,
        gpu_bandwidth_gbps: float,
        pcie_bandwidth_gbps: float,
        safety_margin: float,
        aggressiveness: float,
        optimization_goal: str,
        model_vram_base_mb: float = 0.0,
        kv_cache_vram_mb: float = 0.0,
        model_ram_base_mb: float = 0.0,
        kv_cache_ram_mb: float = 0.0,
    ) -> CandidateAssessment:
        """
        Compute comprehensive assessment and total score (0-100) for a microbatch candidate.
        """
        # 1. Estimate VRAM and RAM
        est_vram_mb = self.estimate_prefill_vram_mb(
            microbatch=candidate,
            n_gpu_layers=n_gpu_layers,
            hidden_size=hidden_size,
            model_vram_base_mb=model_vram_base_mb,
            kv_cache_vram_mb=kv_cache_vram_mb,
        )

        est_ram_mb = self.estimate_prefill_ram_mb(
            microbatch=candidate,
            n_cpu_layers=n_cpu_layers,
            hidden_size=hidden_size,
            model_ram_base_mb=model_ram_base_mb,
            kv_cache_ram_mb=kv_cache_ram_mb,
        )

        # Compute safe budget headroom
        # Note: If free_vram_mb is provided and positive, use it. Otherwise use total_vram_mb * (1 - safety_margin)
        usable_vram_mb = free_vram_mb if free_vram_mb > 0 else (total_vram_mb * (1.0 - safety_margin))
        max_allowed_vram_mb = usable_vram_mb * (1.0 - safety_margin)

        # Safety Check
        if free_vram_mb > 0 and est_vram_mb > (free_vram_mb * (1.0 - safety_margin)):
            return CandidateAssessment(
                microbatch=candidate,
                total_score=0.0,
                is_safe=False,
                estimated_vram_mb=est_vram_mb,
                estimated_ram_mb=est_ram_mb,
                vram_headroom_score=0.0,
                ram_headroom_score=0.0,
                gpu_occupancy_score=0.0,
                prompt_tps_score=0.0,
                stability_score=0.0,
                rejection_reason="Exceeds VRAM budget",
                reason_details=f"Est. {est_vram_mb:.1f}MB VRAM exceeds safe available threshold ({free_vram_mb * (1.0 - safety_margin):.1f}MB)",
            )

        if free_ram_mb > 0 and est_ram_mb > (free_ram_mb * 0.90):
            return CandidateAssessment(
                microbatch=candidate,
                total_score=0.0,
                is_safe=False,
                estimated_vram_mb=est_vram_mb,
                estimated_ram_mb=est_ram_mb,
                vram_headroom_score=0.0,
                ram_headroom_score=0.0,
                gpu_occupancy_score=0.0,
                prompt_tps_score=0.0,
                stability_score=0.0,
                rejection_reason="Exceeds System RAM budget",
                reason_details=f"Est. {est_ram_mb:.1f}MB RAM exceeds safe system threshold ({free_ram_mb * 0.90:.1f}MB)",
            )

        # Sub-score 1: VRAM Headroom (0 - 100)
        if free_vram_mb > 0:
            headroom_ratio = (free_vram_mb - est_vram_mb) / max(1.0, free_vram_mb)
        else:
            headroom_ratio = 0.5
        vram_headroom_score = max(0.0, min(100.0, headroom_ratio * 100.0 * 1.2))

        # Sub-score 2: RAM Headroom (0 - 100)
        if free_ram_mb > 0:
            ram_headroom_ratio = (free_ram_mb - est_ram_mb) / max(1.0, free_ram_mb)
        else:
            ram_headroom_ratio = 0.5
        ram_headroom_score = max(0.0, min(100.0, ram_headroom_ratio * 100.0))

        # Sub-score 3: GPU Occupancy / Compute Saturation (0 - 100)
        # Tensor Cores and GPU execution units require sufficiently large microbatches to saturate
        # e.g., microbatch 128 has low GPU utilization; 512-1024 saturates GPU.
        if n_gpu_layers > 0 and backend_name.lower() in ("cuda", "vulkan", "metal"):
            # S-curve saturation centered around microbatch 512
            occupancy_ratio = 1.0 / (1.0 + math.exp(-0.005 * (candidate - 384)))
            gpu_occupancy_score = max(10.0, min(100.0, occupancy_ratio * 100.0))
        else:
            # CPU backend saturates at smaller microbatch (e.g. 128-256)
            occupancy_ratio = 1.0 / (1.0 + math.exp(-0.01 * (candidate - 256)))
            gpu_occupancy_score = max(20.0, min(80.0, occupancy_ratio * 80.0))

        # Sub-score 4: Prompt Throughput TPS Scaling (0 - 100)
        # TPS increases with microbatch size up to an optimal saturation knee point (e.g. 512-1024),
        # beyond which memory bandwidth or kernel launch overhead plateaus or diminishes.
        optimal_sweet_spot = 512 if gpu_bandwidth_gbps > 200 else 256
        if candidate <= optimal_sweet_spot:
            # Linear/logarithmic increase up to sweet spot
            tps_factor = math.log2(candidate / 64.0) / math.log2(optimal_sweet_spot / 64.0)
            prompt_tps_score = max(10.0, min(100.0, tps_factor * 95.0))
        else:
            # Gentle diminishing returns or slight cache thrashing penalty for huge microbatches
            overshoot = candidate / float(optimal_sweet_spot)
            prompt_tps_score = max(40.0, 95.0 - ((overshoot - 1.0) * 15.0))

        # Sub-score 5: Stability & Interconnect Transfer Penalty (0 - 100)
        # If model is split between GPU and CPU, large microbatches cause large PCIe transfer spikes
        stability_score = 100.0
        if boundary_crossings > 0 and n_cpu_layers > 0:
            # PCIe activation transfer volume scales with microbatch
            transfer_mb_per_batch = (candidate * hidden_size * 2 * boundary_crossings) / (1024 * 1024)
            pcie_stall_penalty = (transfer_mb_per_batch / max(1.0, pcie_bandwidth_gbps)) * 10.0
            stability_score = max(20.0, 100.0 - pcie_stall_penalty)

        # Adjust weights based on optimization goal & aggressiveness
        if optimization_goal == "throughput":
            w_tps = 0.45 * aggressiveness
            w_occ = 0.25 * aggressiveness
            w_headroom = 0.15 / max(0.5, aggressiveness)
            w_stability = 0.15
        elif optimization_goal == "latency":
            w_tps = 0.20
            w_occ = 0.15
            w_headroom = 0.35
            w_stability = 0.30
        else:  # balanced
            w_tps = 0.35
            w_occ = 0.20
            w_headroom = 0.25
            w_stability = 0.20

        total_weight = w_tps + w_occ + w_headroom + w_stability
        raw_score = (
            (vram_headroom_score * w_headroom)
            + (gpu_occupancy_score * w_occ)
            + (prompt_tps_score * w_tps)
            + (stability_score * w_stability)
        ) / total_weight

        total_score = max(1.0, min(100.0, raw_score))

        reason_summary = (
            f"VRAM Headroom {vram_headroom_score:.0f}, TPS Score {prompt_tps_score:.0f}, "
            f"Occupancy {gpu_occupancy_score:.0f}"
        )

        return CandidateAssessment(
            microbatch=candidate,
            total_score=total_score,
            is_safe=True,
            estimated_vram_mb=est_vram_mb,
            estimated_ram_mb=est_ram_mb,
            vram_headroom_score=vram_headroom_score,
            ram_headroom_score=ram_headroom_score,
            gpu_occupancy_score=gpu_occupancy_score,
            prompt_tps_score=prompt_tps_score,
            stability_score=stability_score,
            rejection_reason=None,
            reason_details=reason_summary,
        )
