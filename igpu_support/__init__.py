"""
InferenceOS Integrated GPU Support — Phase 7 Public API

Treats the integrated GPU (iGPU) as a third compute tier between the
discrete GPU and the CPU. Automatically determines which specific workloads
to delegate to the iGPU and disables iGPU use when it would cause shared
memory contention or performance regression.

Public exports:

    IgpuPlacementContributor   — primary facade: apply iGPU overlay to a plan
    IgpuPlacementResult        — result with updated plan + diagnostics
    apply_igpu_placement()     — convenience function

    IgpuProfiler               — reads iGPU hardware info from hw_profile
    IgpuProfile                — iGPU characteristics + suitability score
    profile_igpu()             — convenience function

    ContentionModel            — shared memory bus bandwidth estimator
    ContentionEstimate         — contention analysis result
    estimate_contention()      — convenience function

    WorkloadClassifier         — per-layer iGPU assignment policy
    IgpuWorkloadSuitability    — FULL / LIGHT / DISABLED enum

    SuitabilityEvaluator       — combines score + contention → suitability
    SuitabilityResult          — evaluation with reasoning chain
    evaluate_igpu_suitability() — convenience function

Usage
-----
    from igpu_support import IgpuPlacementContributor, apply_igpu_placement

    # Quick path
    result = apply_igpu_placement(model_descriptor, placement_plan, hw_profile)
    print(result.summary())

    if result.igpu_enabled:
        print(f"iGPU assigned {len(result.igpu_layer_indices)} layers")
        updated_plan = result.plan
    else:
        updated_plan = placement_plan   # original unchanged

    # Detailed path
    from igpu_support import IgpuProfiler, ContentionModel, SuitabilityEvaluator
    profile = IgpuProfiler(hw_profile).profile()
    contention = ContentionModel(hw_profile).estimate(profile)
    suitability = SuitabilityEvaluator().evaluate(profile, contention)
    print(f"iGPU score: {profile.suitability_score:.3f}")
    print(f"Suitability: {suitability.suitability}")
"""
from __future__ import annotations

from .contention_model import ContentionEstimate, ContentionModel, estimate_contention
from .igpu_profiler import IgpuProfile, IgpuProfiler, profile_igpu
from .placement_contributor import (
    IgpuPlacementContributor,
    IgpuPlacementResult,
    apply_igpu_placement,
)
from .suitability_evaluator import (
    SuitabilityEvaluator,
    SuitabilityResult,
    evaluate_igpu_suitability,
)
from .workload_classifier import IgpuWorkloadSuitability, WorkloadClassifier

__all__ = [
    # Primary facade
    "IgpuPlacementContributor",
    "IgpuPlacementResult",
    "apply_igpu_placement",
    # Profiling
    "IgpuProfiler",
    "IgpuProfile",
    "profile_igpu",
    # Contention
    "ContentionModel",
    "ContentionEstimate",
    "estimate_contention",
    # Workload classification
    "WorkloadClassifier",
    "IgpuWorkloadSuitability",
    # Suitability
    "SuitabilityEvaluator",
    "SuitabilityResult",
    "evaluate_igpu_suitability",
]
