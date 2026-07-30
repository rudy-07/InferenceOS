"""
context_scheduler.py
--------------------
Dynamic Context Scheduler facade for adaptive runtime context management in InferenceOS.

Transforms context length from a static user-defined parameter into an adaptive runtime decision.
Exposes schedule_context(...) as the single authority for context allocation across the runtime.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .context_config import ContextSchedulerConfig
from .context_decision import ContextDecision
from .context_heuristics import CandidateContextAssessment, ContextScorer

logger = logging.getLogger("InferenceOS.ContextScheduler")


class ContextScheduler:
    """
    Production-grade Dynamic Context Scheduler.

    Parameters
    ----------
    config : ContextSchedulerConfig, optional
        Default configuration settings.
    verbose : bool, optional
        Enable CLI display of candidate context evaluation tables. Default False.
    """

    def __init__(
        self,
        config: Optional[ContextSchedulerConfig] = None,
        verbose: bool = False,
    ) -> None:
        self.config = config or ContextSchedulerConfig()
        if verbose:
            self.config.verbose = True
        self.scorer = ContextScorer()

    def schedule_context(
        self,
        model_metadata: Optional[Dict[str, Any]] = None,
        requested_context: int = 4096,
        memory_plan: Optional[Any] = None,
        layer_placement: Optional[Any] = None,
        microbatch_decision: Optional[Any] = None,
        backend: Optional[Any] = None,
        hw_profile: Optional[Dict[str, Any]] = None,
        telemetry_history: Optional[Any] = None,
        vram_free_mb: float = 0.0,
        ram_free_mb: float = 0.0,
        config_override: Optional[ContextSchedulerConfig] = None,
        override_context: Optional[int] = None,
    ) -> ContextDecision:
        """
        Intelligently determine the optimal context window size and memory reservation
        for an inference request.

        Parameters
        ----------
        model_metadata : dict, optional
            Model architectural attributes (num_layers, hidden_size, num_heads, num_kv_heads, max_context_length).
        requested_context : int
            Requested context window in tokens.
        memory_plan : MemoryPlan, optional
            Output from orchestrator MemoryPlanner.
        layer_placement : PlacementPlan, optional
            Phase 3 layer placement plan.
        microbatch_decision : SchedulingDecision, optional
            Output from Dynamic Microbatch Scheduler.
        backend : BackendInfo or str, optional
            Resolved execution backend.
        hw_profile : dict, optional
            Hardware resource profile.
        telemetry_history : optional
            Historical telemetry collector or store.
        vram_free_mb : float
            Currently available VRAM in MB.
        ram_free_mb : float
            Currently available RAM in MB.
        config_override : ContextSchedulerConfig, optional
            Config overrides for this run.
        override_context : int, optional
            Explicit manual override value.

        Returns
        -------
        ContextDecision
            Complete decision containing requested, recommended, and effective context,
            memory estimates, warnings, microbatch suggestions, and reasoning.
        """
        effective_config = config_override or self.config
        hw_profile = hw_profile or {}
        meta = model_metadata or {}

        # Extract model architectural parameters
        hidden_size = int(meta.get("hidden_size", meta.get("embedding_length", 4096)))
        total_layers = int(meta.get("num_layers", meta.get("block_count", 32)))
        num_heads = int(meta.get("num_heads", meta.get("head_count", 32)))
        num_kv_heads = int(meta.get("num_kv_heads", meta.get("head_count_kv", num_heads)))
        head_dim = int(meta.get("head_dim", hidden_size // max(1, num_heads)))
        max_model_context = int(meta.get("max_context_length", meta.get("context_length", 131072)))

        # 1. Handle Manual Overrides (Explicit User Request)
        manual_val = override_context or effective_config.manual_override
        if manual_val is not None and manual_val > 0:
            kv_est = self.scorer.kv_estimator.estimate_from_params(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                n_gpu_layers=total_layers,
                context_length=manual_val,
            )
            kv_mem_gb = kv_est.kv_at_full_context / (1024 ** 3)
            decision = ContextDecision(
                requested_context=requested_context,
                recommended_context=manual_val,
                effective_context=manual_val,
                estimated_kv_memory_gb=kv_mem_gb,
                estimated_total_memory_gb=kv_mem_gb + 4.0,
                safety_margin_gb=1.0,
                confidence=1.0,
                candidate_evaluations={manual_val: {"estimated_total_memory_gb": kv_mem_gb + 4.0, "is_safe": True, "score": 100.0}},
                reasoning=[f"User explicitly requested fixed context length of {manual_val}."],
            )
            if effective_config.verbose:
                print(decision.format_cli_output())
            return decision

        if not effective_config.enabled:
            fallback_val = min(requested_context, max_model_context)
            return ContextDecision(
                requested_context=requested_context,
                recommended_context=fallback_val,
                effective_context=fallback_val,
                confidence=0.5,
                reasoning=["Dynamic context scheduler disabled; using static fallback."],
            )

        # 2. Extract Hardware & Placement Parameters
        gpus = hw_profile.get("gpus", [])
        avail_vram_mb = vram_free_mb
        if gpus and avail_vram_mb <= 0.0:
            avail_vram_mb = float(gpus[0].get("vram_free_mb", gpus[0].get("memory_free_mb", 0.0)))
        total_vram_mb = sum(float(g.get("vram_total_mb", 8192.0)) for g in gpus) if gpus else 8192.0

        avail_ram_mb = ram_free_mb
        ram_obj = hw_profile.get("ram", hw_profile.get("memory", {}))
        if avail_ram_mb <= 0.0:
            avail_ram_mb = float(ram_obj.get("available_gb", 8.0)) * 1024.0
        total_ram_mb = float(ram_obj.get("total_gb", 16.0)) * 1024.0

        backend_name = "cpu"
        if backend:
            backend_name = backend.name if hasattr(backend, "name") else str(backend)

        n_gpu_layers = 0
        n_cpu_layers = 0
        if layer_placement:
            n_gpu_layers = getattr(layer_placement, "n_gpu_layers", 0)
            n_cpu_layers = getattr(layer_placement, "n_cpu_layers", total_layers)
        elif memory_plan:
            n_gpu_layers = getattr(memory_plan, "n_gpu_layers", 0)
            n_cpu_layers = total_layers - n_gpu_layers
        else:
            if backend_name.lower() in ("cuda", "vulkan", "metal"):
                n_gpu_layers = total_layers
                n_cpu_layers = 0
            else:
                n_gpu_layers = 0
                n_cpu_layers = total_layers

        model_vram_base = getattr(memory_plan, "estimated_vram_mb", 0.0) if memory_plan else 0.0
        model_ram_base = getattr(memory_plan, "estimated_ram_mb", 0.0) if memory_plan else 0.0

        curr_microbatch = 512
        if microbatch_decision and hasattr(microbatch_decision, "microbatch"):
            curr_microbatch = microbatch_decision.microbatch

        # 3. Candidate Context Generation
        candidates = effective_config.get_candidate_set(requested_context, max_model_context)

        # 4. Evaluate Candidate Matrix
        evaluations: Dict[int, Dict[str, Any]] = {}
        best_cand: Optional[int] = None
        best_score: float = -1.0
        best_assessment: Optional[CandidateContextAssessment] = None

        for cand in candidates:
            asm = self.scorer.evaluate_candidate(
                candidate_context=cand,
                requested_context=requested_context,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_layers=total_layers,
                n_gpu_layers=n_gpu_layers,
                n_cpu_layers=n_cpu_layers,
                hidden_size=hidden_size,
                model_vram_base_mb=model_vram_base,
                model_ram_base_mb=model_ram_base,
                free_vram_mb=avail_vram_mb,
                free_ram_mb=avail_ram_mb,
                total_vram_mb=total_vram_mb,
                total_ram_mb=total_ram_mb,
                microbatch_size=curr_microbatch,
                safety_margin=effective_config.safety_margin,
                optimization_goal=effective_config.optimization_goal,
            )
            evaluations[cand] = {
                "estimated_kv_memory_gb": round(asm.estimated_kv_memory_mb / 1024.0, 3),
                "estimated_total_memory_gb": round(asm.estimated_total_memory_mb / 1024.0, 3),
                "is_safe": asm.is_safe,
                "score": round(asm.total_score, 1),
                "reason": asm.rejection_reason or asm.reason_details,
            }

            if asm.is_safe and asm.total_score > best_score:
                best_score = asm.total_score
                best_cand = cand
                best_assessment = asm

        # 5. Inter-Scheduler Coordination & Graceful Degradation
        suggested_mb: Optional[int] = None
        warnings: List[str] = []

        # If highest candidate (requested) was rejected, check if reducing microbatch can save requested context
        if candidates and not evaluations[candidates[0]]["is_safe"] and curr_microbatch > 256:
            # Re-evaluate top candidate with smaller microbatch (e.g. 256)
            smaller_mb = 256
            asm_retry = self.scorer.evaluate_candidate(
                candidate_context=candidates[0],
                requested_context=requested_context,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_layers=total_layers,
                n_gpu_layers=n_gpu_layers,
                n_cpu_layers=n_cpu_layers,
                hidden_size=hidden_size,
                model_vram_base_mb=model_vram_base,
                model_ram_base_mb=model_ram_base,
                free_vram_mb=avail_vram_mb,
                free_ram_mb=avail_ram_mb,
                total_vram_mb=total_vram_mb,
                total_ram_mb=total_ram_mb,
                microbatch_size=smaller_mb,
                safety_margin=effective_config.safety_margin,
                optimization_goal=effective_config.optimization_goal,
            )
            if asm_retry.is_safe:
                best_cand = candidates[0]
                best_score = asm_retry.total_score
                best_assessment = asm_retry
                suggested_mb = smaller_mb
                evaluations[candidates[0]] = {
                    "estimated_kv_memory_gb": round(asm_retry.estimated_kv_memory_mb / 1024.0, 3),
                    "estimated_total_memory_gb": round(asm_retry.estimated_total_memory_mb / 1024.0, 3),
                    "is_safe": True,
                    "score": round(asm_retry.total_score, 1),
                    "reason": f"Accepted with microbatch reduction to {smaller_mb}",
                }

        # Fallback if no safe candidate was found
        if best_cand is None:
            best_cand = candidates[-1]  # smallest min_context candidate
            best_assessment = self.scorer.evaluate_candidate(
                candidate_context=best_cand,
                requested_context=requested_context,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_layers=total_layers,
                n_gpu_layers=n_gpu_layers,
                n_cpu_layers=n_cpu_layers,
                hidden_size=hidden_size,
                model_vram_base_mb=model_vram_base,
                model_ram_base_mb=model_ram_base,
                free_vram_mb=avail_vram_mb,
                free_ram_mb=avail_ram_mb,
                total_vram_mb=total_vram_mb,
                total_ram_mb=total_ram_mb,
                microbatch_size=curr_microbatch,
                safety_margin=effective_config.safety_margin,
            )
            warnings.append("High memory pressure: minimal context allocated to avoid OOM crash.")

        if best_cand < requested_context:
            warnings.append(f"Context scaled down from requested {requested_context} to {best_cand} to stay within safe memory budget.")

        # 6. Formulate Decision Reasoning & Future Actions
        reasoning_bullets: List[str] = []
        if best_cand == requested_context:
            reasoning_bullets.append(f"Full requested context {requested_context} fits safely within memory budget.")
        else:
            reasoning_bullets.append(f"Adjusted context down to {best_cand} tokens to remain within safe memory budget.")

        if suggested_mb:
            reasoning_bullets.append(f"Coordinated with Dynamic Microbatch Scheduler: suggested microbatch reduction to {suggested_mb}.")

        future_actions: List[str] = []
        if best_cand > 16384:
            future_actions.append("Enable KV compression (FP8/INT4) if context window expands beyond 16k tokens.")
        if avail_vram_mb > 0 and (best_assessment.estimated_total_memory_mb > avail_vram_mb * 0.80):
            future_actions.append("Increase memory safety margin under high GPU memory pressure.")

        confidence = min(0.99, max(0.60, best_score / 100.0))
        est_kv_gb = best_assessment.estimated_kv_memory_mb / 1024.0
        est_tot_gb = best_assessment.estimated_total_memory_mb / 1024.0
        safety_gb = (avail_vram_mb * effective_config.safety_margin) / 1024.0 if avail_vram_mb > 0 else 0.5

        decision = ContextDecision(
            requested_context=requested_context,
            recommended_context=best_cand,
            effective_context=best_cand,
            estimated_kv_memory_gb=est_kv_gb,
            estimated_total_memory_gb=est_tot_gb,
            safety_margin_gb=safety_gb,
            suggested_microbatch=suggested_mb,
            compression_mode="none",
            confidence=confidence,
            candidate_evaluations=evaluations,
            warnings=warnings,
            reasoning=reasoning_bullets,
            future_actions=future_actions,
        )

        if effective_config.verbose:
            print(decision.format_cli_output())

        return decision
