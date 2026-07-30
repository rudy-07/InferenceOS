"""
memory_scheduler.py
-------------------
Adaptive Memory Scheduler facade for central authority memory allocation management in InferenceOS.

Transforms memory management from passive prediction ("Will this fit?") into active scheduling ("How should this run?").
Consumes predictions from MemoryPlanner, PlacementEngine, ContextScheduler, and MicrobatchScheduler to establish
safe operating budgets and allocation strategies immediately before inference begins.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .memory_config import MemorySchedulerConfig
from .memory_decision import MemoryDecision
from .memory_policies import (
    AggressivePolicy,
    BalancedPolicy,
    ConservativePolicy,
    MemoryPressureClassifier,
    MemoryPressureLevel,
    MemorySchedulingPolicy,
)

logger = logging.getLogger("InferenceOS.MemoryScheduler")


class MemoryScheduler:
    """
    Production-grade Adaptive Memory Scheduler.

    Parameters
    ----------
    config : MemorySchedulerConfig, optional
        Default configuration settings.
    verbose : bool, optional
        Enable CLI display of memory allocation decisions. Default False.
    """

    def __init__(
        self,
        config: Optional[MemorySchedulerConfig] = None,
        verbose: bool = False,
    ) -> None:
        self.config = config or MemorySchedulerConfig()
        if verbose:
            self.config.verbose = True

        self._policies: Dict[str, MemorySchedulingPolicy] = {
            "conservative": ConservativePolicy(),
            "balanced": BalancedPolicy(),
            "aggressive": AggressivePolicy(),
        }

    def schedule_memory(
        self,
        model_metadata: Optional[Dict[str, Any]] = None,
        requested_context: int = 4096,
        memory_plan: Optional[Any] = None,
        layer_placement: Optional[Any] = None,
        microbatch_decision: Optional[Any] = None,
        context_decision: Optional[Any] = None,
        hw_profile: Optional[Dict[str, Any]] = None,
        vram_free_mb: float = 0.0,
        ram_free_mb: float = 0.0,
        backend: Optional[Any] = None,
        telemetry_history: Optional[Any] = None,
        config_override: Optional[MemorySchedulerConfig] = None,
    ) -> MemoryDecision:
        """
        Actively schedule memory budgets and allocation strategies for an inference session.

        Parameters
        ----------
        model_metadata : dict, optional
            Model architectural attributes (num_layers, hidden_size, bpw, model_size_mb).
        requested_context : int
            Target context window size.
        memory_plan : MemoryPlan, optional
            Output from orchestrator MemoryPlanner.
        layer_placement : PlacementPlan, optional
            Phase 3 layer placement plan.
        microbatch_decision : SchedulingDecision, optional
            Output from Dynamic Microbatch Scheduler.
        context_decision : ContextDecision, optional
            Output from Dynamic Context Scheduler.
        hw_profile : dict, optional
            Hardware resource profile.
        vram_free_mb : float
            Currently available VRAM in MB.
        ram_free_mb : float
            Currently available system RAM in MB.
        backend : BackendInfo or str, optional
            Resolved execution backend.
        telemetry_history : optional
            Historical telemetry collector.
        config_override : MemorySchedulerConfig, optional
            Config overrides for this run.

        Returns
        -------
        MemoryDecision
            Central memory allocation strategy and safe operating budgets.
        """
        effective_config = config_override or self.config
        hw_profile = hw_profile or {}

        # 1. Extract Hardware Resources
        gpus = hw_profile.get("gpus", [])
        avail_vram = vram_free_mb
        if gpus and avail_vram <= 0.0:
            avail_vram = float(gpus[0].get("vram_free_mb", gpus[0].get("memory_free_mb", 0.0)))

        total_vram = sum(float(g.get("vram_total_mb", 8192.0)) for g in gpus) if gpus else 8192.0
        hw_vram_gb = total_vram / 1024.0

        used_vram = max(0.0, total_vram - avail_vram) if avail_vram > 0 else 0.0

        avail_ram = ram_free_mb
        ram_obj = hw_profile.get("ram", hw_profile.get("memory", {}))
        if avail_ram <= 0.0:
            avail_ram = float(ram_obj.get("available_gb", 8.0)) * 1024.0
        total_ram = float(ram_obj.get("total_gb", 16.0)) * 1024.0
        used_ram = max(0.0, total_ram - avail_ram)

        # 2. Classify Memory Pressure Level
        pressure = MemoryPressureClassifier.classify(
            used_vram_mb=used_vram,
            total_vram_mb=total_vram,
            used_ram_mb=used_ram,
            total_ram_mb=total_ram,
        )

        # 3. Extract Memory Estimates from Memory Planner & Placement Engine
        est_vram_mb = getattr(memory_plan, "estimated_vram_mb", 0.0) if memory_plan else 0.0
        est_ram_mb = getattr(memory_plan, "estimated_ram_mb", 0.0) if memory_plan else 0.0

        if context_decision and hasattr(context_decision, "estimated_total_memory_gb"):
            est_vram_mb = max(est_vram_mb, context_decision.estimated_total_memory_gb * 1024.0)

        # 4. Strategy Selection (Policy-driven)
        selected_strategy_name = effective_config.manual_override or effective_config.memory_strategy

        if selected_strategy_name == "auto":
            # Auto-adaptation based on hardware and pressure level
            if hw_vram_gb <= 8.0 or pressure in (MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL):
                target_key = "conservative"
            elif hw_vram_gb >= 20.0 and pressure in (MemoryPressureLevel.IDLE, MemoryPressureLevel.LOW):
                target_key = "aggressive"
            else:
                target_key = "balanced"
        else:
            target_key = selected_strategy_name.lower()
            if target_key not in self._policies:
                target_key = "balanced"

        policy = self._policies[target_key]

        vram_budget_mb, ram_budget_mb, res_vram_mb, res_ram_mb, oom_risk, policy_actions, policy_reasons = (
            policy.evaluate_strategy(
                estimated_vram_mb=est_vram_mb,
                estimated_ram_mb=est_ram_mb,
                free_vram_mb=avail_vram,
                free_ram_mb=avail_ram,
                total_vram_mb=total_vram,
                total_ram_mb=total_ram,
                pressure=pressure,
                hardware_vram_gb=hw_vram_gb,
            )
        )

        # 5. Formulate Proactive Runtime Recommendations
        rec_microbatch: Optional[int] = None
        rec_context: Optional[int] = None
        rec_gpu_layers: Optional[int] = None
        warnings: List[str] = []

        if oom_risk or pressure == MemoryPressureLevel.CRITICAL:
            warnings.append(f"Elevated memory pressure ({pressure.value}): proactive scaling active to prevent OOM.")
            rec_microbatch = 256
            rec_context = min(requested_context, 8192)
            if layer_placement and hasattr(layer_placement, "n_gpu_layers"):
                n_gpu = layer_placement.n_gpu_layers
                if n_gpu > 4:
                    rec_gpu_layers = n_gpu - 4
        elif pressure == MemoryPressureLevel.HIGH:
            warnings.append("High memory pressure detected: operating with conservative safety margin.")
            rec_microbatch = 384
            if requested_context > 8192:
                rec_context = 8192

        confidence = 0.98 if not oom_risk else 0.80

        decision = MemoryDecision(
            vram_budget_gb=vram_budget_mb / 1024.0,
            ram_budget_gb=ram_budget_mb / 1024.0,
            reserved_vram_gb=res_vram_mb / 1024.0,
            reserved_ram_gb=res_ram_mb / 1024.0,
            safety_margin_level="Conservative" if target_key == "conservative" else ("Aggressive" if target_key == "aggressive" else "Balanced"),
            pressure_level=pressure.value,
            strategy_selected=policy.name,
            estimated_vram_gb=est_vram_mb / 1024.0,
            estimated_ram_gb=est_ram_mb / 1024.0,
            suggested_microbatch_override=rec_microbatch,
            suggested_context_override=rec_context,
            suggested_gpu_layer_override=rec_gpu_layers,
            confidence=confidence,
            recommended_actions=policy_actions,
            warnings=warnings,
            reasoning=policy_reasons,
            oom_risk=oom_risk,
        )

        if effective_config.verbose:
            print(decision.format_cli_output())

        return decision
