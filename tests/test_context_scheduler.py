"""
test_context_scheduler.py
--------------------------
Comprehensive test suite for the Dynamic Context Scheduler in InferenceOS.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scheduler import (
    CandidateContextAssessment,
    ContextDecision,
    ContextScheduler,
    ContextSchedulerConfig,
    ContextScorer,
)
from inference_runtime.runtime_config import RuntimeConfig
from inference_runtime.argument_builder import ArgumentBuilder
from layer_placement.placement_plan import PlacementPlan, LayerPlacement, PlacementDevice, LayerCostBreakdown
from inference_runtime.backend_selector import BackendInfo


# ---------------------------------------------------------------------------
# 1. Candidate Set Generation & Alignment Tests
# ---------------------------------------------------------------------------

def test_context_candidate_set_generation():
    config = ContextSchedulerConfig(min_context=512, max_context=131072, alignment_step=512)
    candidates = config.get_candidate_set(requested_context=32768, max_model_context=131072)
    
    assert candidates[0] == 32768
    assert candidates[-1] == 512
    assert all(c % 512 == 0 for c in candidates)
    assert candidates == sorted(candidates, reverse=True)


def test_context_manual_override():
    config = ContextSchedulerConfig(manual_override=16384)
    candidates = config.get_candidate_set(requested_context=32768)
    assert candidates == [16384]


# ---------------------------------------------------------------------------
# 2. Analytical KV Cache & Memory Safety Tests
# ---------------------------------------------------------------------------

def test_context_scorer_evaluation():
    scorer = ContextScorer()
    assessment = scorer.evaluate_candidate(
        candidate_context=16384,
        requested_context=32768,
        num_kv_heads=8,
        head_dim=128,
        num_layers=32,
        n_gpu_layers=32,
        n_cpu_layers=0,
        hidden_size=4096,
        model_vram_base_mb=4000.0,
        model_ram_base_mb=0.0,
        free_vram_mb=12000.0,
        free_ram_mb=16384.0,
        total_vram_mb=16384.0,
        total_ram_mb=32768.0,
    )
    assert assessment.is_safe
    assert assessment.estimated_kv_memory_mb > 0
    assert assessment.total_score > 0.0


def test_context_scorer_rejection_on_tight_vram():
    scorer = ContextScorer()
    assessment = scorer.evaluate_candidate(
        candidate_context=32768,
        requested_context=32768,
        num_kv_heads=32,  # Full MHA (large KV cache)
        head_dim=128,
        num_layers=32,
        n_gpu_layers=32,
        n_cpu_layers=0,
        hidden_size=4096,
        model_vram_base_mb=4000.0,
        model_ram_base_mb=0.0,
        free_vram_mb=2000.0,  # Only 2GB free VRAM
        free_ram_mb=16384.0,
        total_vram_mb=16384.0,
        total_ram_mb=32768.0,
    )
    assert not assessment.is_safe
    assert assessment.total_score == 0.0
    assert "Exceeds safe VRAM budget" in assessment.rejection_reason


# ---------------------------------------------------------------------------
# 3. Graceful Degradation Tests
# ---------------------------------------------------------------------------

def test_context_scheduler_graceful_degradation():
    scheduler = ContextScheduler()
    # Free VRAM can hold ~16k context, but user requests 32768
    hw_profile = {
        "gpus": [{"vram_free_mb": 5000.0, "vram_total_mb": 16384.0}],
        "ram": {"available_gb": 16.0, "total_gb": 32.0},
    }
    decision = scheduler.schedule_context(
        model_metadata={"hidden_size": 4096, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
        requested_context=32768,
        hw_profile=hw_profile,
        backend="cuda",
    )
    assert isinstance(decision, ContextDecision)
    assert decision.effective_context < 32768
    assert decision.effective_context >= 512
    assert any("scaled down" in w.lower() for w in decision.warnings)


# ---------------------------------------------------------------------------
# 4. Inter-Scheduler Coordination & Microbatch Suggestions
# ---------------------------------------------------------------------------

def test_inter_scheduler_microbatch_suggestion():
    scheduler = ContextScheduler()
    # Tightly bounded VRAM where microbatch 1024 fails, but microbatch 256 succeeds
    hw_profile = {
        "gpus": [{"vram_free_mb": 3500.0, "vram_total_mb": 16384.0}],
        "ram": {"available_gb": 16.0, "total_gb": 32.0},
    }
    mock_mb_decision = MagicMock()
    mock_mb_decision.microbatch = 1024

    decision = scheduler.schedule_context(
        model_metadata={"hidden_size": 4096, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128},
        requested_context=16384,
        microbatch_decision=mock_mb_decision,
        hw_profile=hw_profile,
        backend="cuda",
    )
    assert isinstance(decision, ContextDecision)
    if decision.suggested_microbatch:
        assert decision.suggested_microbatch < 1024


# ---------------------------------------------------------------------------
# 5. CLI Output Formatting & Optimization Goals Tests
# ---------------------------------------------------------------------------

def test_context_cli_formatting():
    decision = ContextDecision(
        requested_context=32768,
        recommended_context=24576,
        effective_context=24576,
        estimated_kv_memory_gb=3.1,
        estimated_total_memory_gb=5.6,
        safety_margin_gb=0.7,
        candidate_evaluations={
            32768: {"estimated_total_memory_gb": 7.2, "is_safe": False, "reason": "Exceeds safe budget"},
            24576: {"estimated_total_memory_gb": 5.6, "is_safe": True, "score": 92.0, "reason": "Optimal"},
        },
        reasoning=["Largest context within safe memory budget."],
    )
    formatted = decision.format_cli_output()
    assert "Context Scheduler" in formatted
    assert "Requested" in formatted
    assert "32768" in formatted
    assert "24576" in formatted
    assert "Selected Context" in formatted


def test_context_optimization_goals():
    scheduler = ContextScheduler()
    hw = {
        "gpus": [{"vram_free_mb": 8000.0, "vram_total_mb": 16384.0}],
        "ram": {"available_gb": 16.0},
    }
    cfg_max_ctx = ContextSchedulerConfig(optimization_goal="maximum_context")
    cfg_speed = ContextSchedulerConfig(optimization_goal="maximum_speed")

    dec_ctx = scheduler.schedule_context(requested_context=32768, hw_profile=hw, backend="cuda", config_override=cfg_max_ctx)
    dec_spd = scheduler.schedule_context(requested_context=32768, hw_profile=hw, backend="cuda", config_override=cfg_speed)

    assert dec_ctx.effective_context >= dec_spd.effective_context


# ---------------------------------------------------------------------------
# 6. Integration with ArgumentBuilder
# ---------------------------------------------------------------------------

def test_argument_builder_effective_context(tmp_path):
    llama_exe = tmp_path / "llama.exe"
    llama_exe.touch()

    builder = ArgumentBuilder(llama_exe)
    plan = PlacementPlan(
        model_name="test_model",
        architecture="llama",
        total_layers=32,
        n_gpu_layers=32,
        n_cpu_layers=0,
        layer_placements=[
            LayerPlacement(i, "transformer", PlacementDevice.GPU, 0, 100, LayerCostBreakdown())
            for i in range(32)
        ],
        context_length=4096,
        is_feasible=True,
    )
    config = RuntimeConfig(context_length=24576)
    backend = BackendInfo(name="cuda", n_gpu_layers=32)

    args = builder.build(
        model_path=tmp_path / "model.gguf",
        prompt="Test prompt",
        plan=plan,
        config=config,
        backend=backend,
    )

    assert "-c" in args
    c_idx = args.index("-c")
    assert args[c_idx + 1] == "4096"  # min(config.context_length, plan.context_length)
