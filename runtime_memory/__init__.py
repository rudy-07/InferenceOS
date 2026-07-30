"""
InferenceOS Runtime Memory Optimization — Phase 6 Public API

Exposes the complete predictive memory estimation and OOM prevention interface:

    PreflightGuard          — primary go/no-go gate (check() before inference)
    InferenceBlockedError   — raised on CRITICAL OOM risk
    MemoryBudget            — complete pre-flight analysis result + report()
    KvCacheEstimator        — formula-based KV cache projections (GQA-aware)
    KvEstimate              — KV estimate output (quarter/half/full context)
    BufferPlanner           — activation + backend + KV buffer sizing
    BufferPlan              — buffer plan output dataclass
    RiskClassifier          — LOW / MEDIUM / HIGH / CRITICAL OOM classification
    OomRisk                 — enum with severity(), is_actionable(), emoji()
    WarningGenerator        — structured warning message generation

Usage
-----
    from runtime_memory import PreflightGuard, MigrationConfig

    guard = PreflightGuard(hw_profile, auto_resize_buffers=True)

    try:
        budget = guard.check(model, plan, context_length=8192)
        print(budget.report())
    except InferenceBlockedError as e:
        print(e.budget.report())
        # → Adjust context or abort

    # Direct KV estimation
    from runtime_memory import KvCacheEstimator
    est = KvCacheEstimator()
    kv = est.estimate(model, plan, context_length=4096)
    print(f"KV at full context: {kv.kv_at_full_context / 1e9:.2f} GB")
    print(f"Max safe context:   {est.max_safe_context(model, plan, vram_headroom_bytes)}")
"""
from __future__ import annotations

from .buffer_planner import BufferPlan, BufferPlanner
from .kv_estimator import KvCacheEstimator, KvEstimate
from .memory_budget import MemoryBudget
from .preflight_guard import InferenceBlockedError, PreflightGuard
from .risk_classifier import OomRisk, RiskClassifier
from .warning_generator import WarningGenerator

__all__ = [
    # Primary facade
    "PreflightGuard",
    # Error type
    "InferenceBlockedError",
    # Output types
    "MemoryBudget",
    "KvEstimate",
    "BufferPlan",
    # Analysis components
    "KvCacheEstimator",
    "BufferPlanner",
    "RiskClassifier",
    "OomRisk",
    "WarningGenerator",
]
