"""
microbatch_scheduler.py
-----------------------
Dynamic Microbatch Scheduler facade for single-request prompt ingestion optimization in InferenceOS.

Replaces fixed microbatch configuration values (e.g., 512) with an intelligent, hardware-aware,
safety-bounded adaptive scheduler. Executes immediately before inference begins and consumes existing
telemetry, memory planner outputs, layer placement plans, and hardware profile.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from .config import SchedulerConfig
from .decision import SchedulingDecision
from .feedback_store import RuntimeFeedback, RuntimeFeedbackStore
from .heuristics import CandidateAssessment, HeuristicScorer

logger = logging.getLogger("InferenceOS.MicrobatchScheduler")


class RuntimeLearningPolicy(Protocol):
    """Protocol for future Runtime Learning module integration."""
    def recommend_microbatch(
        self,
        model_metadata: Dict[str, Any],
        context_length: int,
        hw_profile: Dict[str, Any],
        backend: str,
    ) -> Optional[int]:
        ...


class MicrobatchScheduler:
    """
    Production-grade Dynamic Microbatch Scheduler.

    Parameters
    ----------
    config : SchedulerConfig, optional
        Default configuration settings.
    feedback_store : RuntimeFeedbackStore, optional
        Store for recording runtime feedback. If None, instantiates a new store.
    verbose : bool, optional
        Enable CLI display of candidate evaluation tables. Default False.
    """

    def __init__(

        self,
        config: Optional[SchedulerConfig] = None,
        feedback_store: Optional[RuntimeFeedbackStore] = None,
        verbose: bool = False,
    ) -> None:
        self.config = config or SchedulerConfig()
        if verbose:
            self.config.verbose = True
        self.feedback_store = feedback_store or RuntimeFeedbackStore()
        self._learning_policy: Optional[RuntimeLearningPolicy] = None

    def register_learning_policy(self, policy: RuntimeLearningPolicy) -> None:
        """
        Register a Runtime Learning policy callback.
        Future architecture: Runtime Learning -> Recommended microbatch -> Scheduler validates -> Inference.
        """
        self._learning_policy = policy

    def select_microbatch(
        self,
        model_metadata: Optional[Dict[str, Any]] = None,
        context_length: int = 4096,
        memory_plan: Optional[Any] = None,
        layer_placement: Optional[Any] = None,
        hw_profile: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
        telemetry_cache: Optional[Any] = None,
        vram_free_mb: float = 0.0,
        ram_free_mb: float = 0.0,
        gpu_utilization: float = 0.0,
        config_override: Optional[SchedulerConfig] = None,
        override_microbatch: Optional[int] = None,
    ) -> SchedulingDecision:
        """
        Determine the largest safe microbatch size that maximizes prompt ingestion throughput
        while avoiding OOM, VRAM spikes, GPU starvation, and CPU bottlenecks.

        Executes immediately before inference begins.

        Parameters
        ----------
        model_metadata : dict, optional
            Model architectural attributes (num_layers, hidden_size, num_heads, bpw).
        context_length : int
            Target context window size in tokens.
        memory_plan : MemoryPlan, optional
            Output from orchestrator MemoryPlanner.
        layer_placement : PlacementPlan, optional
            Phase 3 layer placement plan.
        hw_profile : dict, optional
            System hardware resource profile.
        backend : BackendInfo or str, optional
            Resolved execution backend.
        telemetry_cache : ProfilerTelemetry, optional
            Recent runtime profiling metrics.
        vram_free_mb : float
            Currently available VRAM in MB.
        ram_free_mb : float
            Currently available system RAM in MB.
        gpu_utilization : float
            Current GPU utilization percentage.
        config_override : SchedulerConfig, optional
            Config overrides for this run.
        override_microbatch : int, optional
            Explicit manual override value.

        Returns
        -------
        SchedulingDecision
            Contains selected microbatch, confidence, candidate scores, and reasoning.
        """
        effective_config = config_override or self.config
        hw_profile = hw_profile or {}

        # 1. Handle Manual Overrides (Explicit User Request)
        manual_val = override_microbatch or effective_config.manual_override
        if manual_val is not None and manual_val > 0:
            decision = SchedulingDecision(
                microbatch=manual_val,
                confidence=1.0,
                optimization_goal=effective_config.optimization_goal,
                estimated_vram_gb=0.0,
                estimated_ram_gb=0.0,
                safety_margin=effective_config.safety_margin,
                candidate_scores={manual_val: 100.0},
                candidate_reasons={manual_val: "Explicit manual override requested"},
                reasoning=[f"User explicitly requested fixed microbatch size of {manual_val}."],
                validated_by_learning=False,
            )
            if effective_config.verbose:
                print(decision.format_cli_output())
            return decision

        # If dynamic scheduling is disabled in config, fallback to default 512
        if not effective_config.enabled:
            fallback_val = 512
            return SchedulingDecision(
                microbatch=fallback_val,
                confidence=0.5,
                optimization_goal=effective_config.optimization_goal,
                candidate_scores={fallback_val: 50.0},
                reasoning=["Dynamic microbatch scheduler disabled; using default 512."],
            )

        # 2. Extract Hardware & Placement Details
        gpus = hw_profile.get("gpus", [])
        avail_vram_mb = vram_free_mb
        gpu_bw_gbps = 300.0
        if gpus:
            gpu_obj = gpus[0]
            if avail_vram_mb <= 0.0:
                avail_vram_mb = float(gpu_obj.get("vram_free_mb", gpu_obj.get("memory_free_mb", 0.0)))
            gpu_bw_gbps = float(gpu_obj.get("vram_bandwidth_gbps", 300.0))

        total_vram_mb = sum(float(g.get("vram_total_mb", g.get("memory_total_mb", 0.0))) for g in gpus)
        if total_vram_mb <= 0.0:
            total_vram_mb = float(hw_profile.get("gpus", [{}])[0].get("vram_total_mb", 8192.0)) if gpus else 0.0

        avail_ram_mb = ram_free_mb
        ram_obj = hw_profile.get("ram", hw_profile.get("memory", {}))
        if avail_ram_mb <= 0.0:
            avail_ram_mb = float(ram_obj.get("available_gb", 8.0)) * 1024.0
        total_ram_mb = float(ram_obj.get("total_gb", 16.0)) * 1024.0

        pcie_bw_gbps = 16.0
        for ic in hw_profile.get("interconnects", []):
            if str(ic.get("type", "")).upper() == "PCIE":
                pcie_bw_gbps = float(ic.get("bandwidth", 16.0))

        # Extract layer count and hidden dim
        meta = model_metadata or {}
        hidden_size = int(meta.get("hidden_size", meta.get("embedding_length", 4096)))
        total_layers = int(meta.get("num_layers", meta.get("block_count", 32)))

        n_gpu_layers = 0
        n_cpu_layers = 0
        boundary_crossings = 0
        model_name = "model"

        backend_name = "cpu"
        if backend:
            backend_name = backend.name if hasattr(backend, "name") else str(backend)

        if layer_placement:
            n_gpu_layers = getattr(layer_placement, "n_gpu_layers", 0)
            n_cpu_layers = getattr(layer_placement, "n_cpu_layers", total_layers)
            boundary_crossings = getattr(layer_placement, "boundary_crossings", 0)
            total_layers = getattr(layer_placement, "total_layers", total_layers)
            model_name = getattr(layer_placement, "model_name", model_name)
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

        # Base model & KV cache estimates
        model_vram_base = getattr(memory_plan, "estimated_vram_mb", 0.0) if memory_plan else 0.0
        model_ram_base = getattr(memory_plan, "estimated_ram_mb", 0.0) if memory_plan else 0.0

        # 3. Candidate Set Generation
        candidates = effective_config.get_candidate_set(context_length)

        # 4. Check Runtime Learning Policy Override (Future Compatibility)
        if self._learning_policy is not None:
            try:
                rec_mb = self._learning_policy.recommend_microbatch(
                    model_metadata=meta,
                    context_length=context_length,
                    hw_profile=hw_profile,
                    backend=backend_name,
                )
                if rec_mb and rec_mb in candidates:
                    is_safe, msg, est_vram, est_ram = self.validate_candidate(
                        candidate=rec_mb,
                        free_vram_mb=avail_vram_mb,
                        free_ram_mb=avail_ram_mb,
                        n_gpu_layers=n_gpu_layers,
                        n_cpu_layers=n_cpu_layers,
                        hidden_size=hidden_size,
                        safety_margin=effective_config.safety_margin,
                    )
                    if is_safe:
                        decision = SchedulingDecision(
                            microbatch=rec_mb,
                            confidence=0.95,
                            optimization_goal=effective_config.optimization_goal,
                            estimated_vram_gb=est_vram / 1024.0,
                            estimated_ram_gb=est_ram / 1024.0,
                            safety_margin=effective_config.safety_margin,
                            candidate_scores={c: (100.0 if c == rec_mb else 50.0) for c in candidates},
                            candidate_reasons={rec_mb: "Recommended by Runtime Learning policy"},
                            reasoning=[
                                f"Selected microbatch {rec_mb} recommended by Runtime Learning policy.",
                                f"Validated memory safety (Est VRAM {est_vram / 1024.0:.2f}GB < available).",
                            ],
                            validated_by_learning=True,
                        )
                        if effective_config.verbose:
                            print(decision.format_cli_output())
                        return decision
            except Exception as e:
                logger.warning(f"Runtime Learning policy recommendation failed: {e}")

        # 5. Evaluate Candidate Scoring Matrix
        assessments: Dict[int, CandidateAssessment] = {}
        scores: Dict[int, float] = {}
        reasons: Dict[int, str] = {}

        best_cand: Optional[int] = None
        best_score: float = -1.0

        for cand in candidates:
            asm = HeuristicScorer.evaluate_candidate(
                candidate=cand,
                total_vram_mb=total_vram_mb,
                free_vram_mb=avail_vram_mb,
                total_ram_mb=total_ram_mb,
                free_ram_mb=avail_ram_mb,
                n_gpu_layers=n_gpu_layers,
                n_cpu_layers=n_cpu_layers,
                total_layers=total_layers,
                context_length=context_length,
                hidden_size=hidden_size,
                backend_name=backend_name,
                boundary_crossings=boundary_crossings,
                gpu_bandwidth_gbps=gpu_bw_gbps,
                pcie_bandwidth_gbps=pcie_bw_gbps,
                safety_margin=effective_config.safety_margin,
                aggressiveness=effective_config.aggressiveness,
                optimization_goal=effective_config.optimization_goal,
                model_vram_base_mb=model_vram_base,
                model_ram_base_mb=model_ram_base,
            )
            assessments[cand] = asm
            scores[cand] = asm.total_score
            reasons[cand] = asm.rejection_reason if not asm.is_safe else (asm.reason_details or "")

            if asm.is_safe and asm.total_score > best_score:
                best_score = asm.total_score
                best_cand = cand

        # Fallback if no safe candidate was found (pick smallest candidate safely)
        if best_cand is None:
            best_cand = candidates[0]
            scores[best_cand] = 10.0
            reasons[best_cand] = "Fallback minimal safe size"

        best_asm = assessments[best_cand]

        # 6. Formulate Decision Reasoning
        reasoning_bullets = []
        if best_asm.vram_headroom_score > 40:
            reasoning_bullets.append("VRAM headroom sufficient for prompt prefill activations.")
        else:
            reasoning_bullets.append("Tight VRAM headroom; conservative microbatch selected to prevent OOM.")

        if best_asm.prompt_tps_score >= 70:
            reasoning_bullets.append("High prompt processing throughput (TPS) predicted.")
        
        if best_asm.gpu_occupancy_score >= 80:
            reasoning_bullets.append(f"High GPU occupancy expected (saturation score {int(best_asm.gpu_occupancy_score)}%).")

        if boundary_crossings > 0:
            reasoning_bullets.append(f"Balanced size for split offload ({boundary_crossings} PCIe boundary crossings).")

        confidence = min(0.99, max(0.60, best_score / 100.0))

        decision = SchedulingDecision(
            microbatch=best_cand,
            confidence=confidence,
            optimization_goal=effective_config.optimization_goal,
            estimated_vram_gb=best_asm.estimated_vram_mb / 1024.0,
            estimated_ram_gb=best_asm.estimated_ram_mb / 1024.0,
            safety_margin=effective_config.safety_margin,
            candidate_scores=scores,
            candidate_reasons=reasons,
            reasoning=reasoning_bullets,
            validated_by_learning=False,
        )

        if effective_config.verbose:
            print(decision.format_cli_output())

        return decision

    def validate_candidate(
        self,
        candidate: int,
        free_vram_mb: float,
        free_ram_mb: float,
        n_gpu_layers: int,
        n_cpu_layers: int,
        hidden_size: int = 4096,
        safety_margin: float = 0.15,
    ) -> Tuple[bool, str, float, float]:
        """
        Validate whether a candidate microbatch satisfies VRAM & RAM safety margins.

        Returns
        -------
        Tuple[bool, str, float, float]
            (is_safe, reason_string, est_vram_mb, est_ram_mb)
        """
        est_vram = HeuristicScorer.estimate_prefill_vram_mb(
            microbatch=candidate,
            n_gpu_layers=n_gpu_layers,
            hidden_size=hidden_size,
        )
        est_ram = HeuristicScorer.estimate_prefill_ram_mb(
            microbatch=candidate,
            n_cpu_layers=n_cpu_layers,
            hidden_size=hidden_size,
        )

        if free_vram_mb > 0:
            max_vram = free_vram_mb * (1.0 - safety_margin)
            if est_vram > max_vram:
                return False, f"VRAM usage ({est_vram:.1f}MB) exceeds safe threshold ({max_vram:.1f}MB)", est_vram, est_ram

        if free_ram_mb > 0:
            max_ram = free_ram_mb * 0.90
            if est_ram > max_ram:
                return False, f"RAM usage ({est_ram:.1f}MB) exceeds safe threshold ({max_ram:.1f}MB)", est_vram, est_ram

        return True, "Safe within memory budget", est_vram, est_ram

    def record_runtime_feedback(
        self,
        prompt_tps: float,
        eval_tps: float,
        actual_vram_mb: float,
        actual_ram_mb: float,
        gpu_util_pct: float,
        cpu_util_pct: float,
        ttft_ms: float,
        microbatch: int,
        context_length: int,
        model_name: str,
        backend: str,
    ) -> None:
        """
        Collect post-inference runtime feedback and store for future Runtime Learning.
        """
        record = RuntimeFeedback(
            prompt_tps=prompt_tps,
            eval_tps=eval_tps,
            actual_vram_mb=actual_vram_mb,
            actual_ram_mb=actual_ram_mb,
            gpu_util_pct=gpu_util_pct,
            cpu_util_pct=cpu_util_pct,
            ttft_ms=ttft_ms,
            microbatch=microbatch,
            context_length=context_length,
            model_name=model_name,
            backend=backend,
            timestamp=time.time(),
        )
        self.feedback_store.add_feedback(record)
