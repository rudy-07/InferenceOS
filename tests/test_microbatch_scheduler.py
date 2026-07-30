"""
test_microbatch_scheduler.py
-----------------------------
Comprehensive test suite for the Dynamic Microbatch Scheduler in InferenceOS.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scheduler import (
    CandidateAssessment,
    HeuristicScorer,
    MicrobatchScheduler,
    RuntimeFeedback,
    RuntimeFeedbackStore,
    SchedulerConfig,
    SchedulingDecision,
)
from inference_runtime.runtime_config import RuntimeConfig
from inference_runtime.argument_builder import ArgumentBuilder
from layer_placement.placement_plan import PlacementPlan, LayerPlacement, PlacementDevice, LayerCostBreakdown
from inference_runtime.backend_selector import BackendInfo


class DummyLearningPolicy:
    """Mock policy simulating future Runtime Learning policy recommendations."""
    def __init__(self, recommended_val: int):
        self.recommended_val = recommended_val

    def recommend_microbatch(self, model_metadata, context_length, hw_profile, backend):
        return self.recommended_val


# ---------------------------------------------------------------------------
# 1. Candidate Set Generation Tests
# ---------------------------------------------------------------------------

def test_candidate_set_generation():
    config = SchedulerConfig(min_microbatch=128, max_microbatch=2048)
    candidates = config.get_candidate_set(context_length=4096)
    assert candidates == [128, 256, 384, 512, 768, 1024, 2048]

    # Small context window restricts max candidate
    candidates_small_ctx = config.get_candidate_set(context_length=384)
    assert candidates_small_ctx == [128, 256, 384]

    # Manual override returns single candidate
    config_override = SchedulerConfig(manual_override=512)
    assert config_override.get_candidate_set(context_length=4096) == [512]


# ---------------------------------------------------------------------------
# 2. Heuristic Scoring & VRAM Safety Headroom Tests
# ---------------------------------------------------------------------------

def test_vram_estimation():
    est_vram = HeuristicScorer.estimate_prefill_vram_mb(
        microbatch=512,
        n_gpu_layers=32,
        hidden_size=4096,
        model_vram_base_mb=4000.0,
        kv_cache_vram_mb=500.0,
    )
    assert est_vram > 4500.0
    # Larger microbatch should yield higher estimated VRAM
    est_vram_large = HeuristicScorer.estimate_prefill_vram_mb(
        microbatch=2048,
        n_gpu_layers=32,
        hidden_size=4096,
        model_vram_base_mb=4000.0,
        kv_cache_vram_mb=500.0,
    )
    assert est_vram_large > est_vram


def test_vram_safety_rejection():
    # Evaluate candidate with very tight free VRAM
    assessment = HeuristicScorer.evaluate_candidate(
        candidate=2048,
        total_vram_mb=8192.0,
        free_vram_mb=1000.0,  # Only 1GB free
        total_ram_mb=16384.0,
        free_ram_mb=8192.0,
        n_gpu_layers=32,
        n_cpu_layers=0,
        total_layers=32,
        context_length=4096,
        hidden_size=4096,
        backend_name="cuda",
        boundary_crossings=0,
        gpu_bandwidth_gbps=300.0,
        pcie_bandwidth_gbps=16.0,
        safety_margin=0.15,
        aggressiveness=1.0,
        optimization_goal="throughput",
    )
    assert not assessment.is_safe
    assert assessment.total_score == 0.0
    assert "Exceeds VRAM budget" in assessment.rejection_reason


def test_scheduler_select_microbatch_normal():
    scheduler = MicrobatchScheduler()
    hw_profile = {
        "gpus": [{"vram_free_mb": 12000.0, "vram_total_mb": 16384.0, "vram_bandwidth_gbps": 500.0}],
        "ram": {"available_gb": 16.0, "total_gb": 32.0},
    }
    decision = scheduler.select_microbatch(
        model_metadata={"hidden_size": 4096, "num_layers": 32},
        context_length=4096,
        hw_profile=hw_profile,
        backend="cuda",
    )
    assert isinstance(decision, SchedulingDecision)
    assert decision.microbatch in [256, 384, 512, 768, 1024, 2048]
    assert decision.confidence > 0.5
    assert len(decision.candidate_scores) > 0


def test_scheduler_tight_memory_selects_small_microbatch():
    scheduler = MicrobatchScheduler()
    # Tight VRAM (e.g. 300 MB free VRAM headroom)
    hw_profile = {
        "gpus": [{"vram_free_mb": 300.0, "vram_total_mb": 8192.0, "vram_bandwidth_gbps": 300.0}],
        "ram": {"available_gb": 8.0, "total_gb": 16.0},
    }
    decision = scheduler.select_microbatch(
        model_metadata={"hidden_size": 4096, "num_layers": 32},
        context_length=4096,
        hw_profile=hw_profile,
        backend="cuda",
    )
    # Under tight VRAM, scheduler should reject large sizes and select small microbatch (<= 256)
    assert decision.microbatch <= 256


# ---------------------------------------------------------------------------
# 3. Optimization Goals & Manual Override Tests
# ---------------------------------------------------------------------------

def test_optimization_goals():
    scheduler = MicrobatchScheduler()
    hw = {
        "gpus": [{"vram_free_mb": 10000.0, "vram_total_mb": 16384.0, "vram_bandwidth_gbps": 400.0}],
        "ram": {"available_gb": 16.0, "total_gb": 32.0},
    }
    cfg_throughput = SchedulerConfig(optimization_goal="throughput")
    cfg_latency = SchedulerConfig(optimization_goal="latency")

    dec_tp = scheduler.select_microbatch(context_length=4096, hw_profile=hw, backend="cuda", config_override=cfg_throughput)
    dec_lat = scheduler.select_microbatch(context_length=4096, hw_profile=hw, backend="cuda", config_override=cfg_latency)

    assert dec_tp.optimization_goal == "throughput"
    assert dec_lat.optimization_goal == "latency"
    # Latency goal emphasizes safety headroom and lower latency, preferring equal or smaller microbatch
    assert dec_lat.microbatch <= dec_tp.microbatch


def test_manual_override():
    scheduler = MicrobatchScheduler()
    decision = scheduler.select_microbatch(override_microbatch=768)
    assert decision.microbatch == 768
    assert decision.confidence == 1.0
    assert "User explicitly requested" in decision.reasoning[0]


# ---------------------------------------------------------------------------
# 4. Future Runtime Learning Policy Integration Tests
# ---------------------------------------------------------------------------

def test_runtime_learning_policy_recommendation():
    scheduler = MicrobatchScheduler()
    policy = DummyLearningPolicy(recommended_val=768)
    scheduler.register_learning_policy(policy)

    hw = {
        "gpus": [{"vram_free_mb": 16000.0, "vram_total_mb": 24576.0, "vram_bandwidth_gbps": 800.0}],
        "ram": {"available_gb": 32.0, "total_gb": 64.0},
    }
    decision = scheduler.select_microbatch(context_length=4096, hw_profile=hw, backend="cuda")
    assert decision.microbatch == 768
    assert decision.validated_by_learning is True
    assert "Runtime Learning policy" in decision.reasoning[0]


def test_runtime_learning_policy_unsafe_recommendation_rejected():
    scheduler = MicrobatchScheduler()
    # Policy recommends 2048, but free VRAM is too low for 2048
    policy = DummyLearningPolicy(recommended_val=2048)
    scheduler.register_learning_policy(policy)

    hw = {
        "gpus": [{"vram_free_mb": 1000.0, "vram_total_mb": 8192.0}],
        "ram": {"available_gb": 8.0},
    }
    decision = scheduler.select_microbatch(context_length=4096, hw_profile=hw, backend="cuda")
    # Scheduler should reject unsafe 2048 and fall back to safe heuristic selection
    assert decision.microbatch < 2048
    assert decision.validated_by_learning is False


# ---------------------------------------------------------------------------
# 5. Feedback Store Tests
# ---------------------------------------------------------------------------

def test_feedback_store():
    store = RuntimeFeedbackStore()
    record = RuntimeFeedback(
        prompt_tps=350.0,
        eval_tps=45.0,
        actual_vram_mb=4500.0,
        actual_ram_mb=2000.0,
        gpu_util_pct=92.0,
        cpu_util_pct=15.0,
        ttft_ms=28.5,
        microbatch=512,
        context_length=4096,
        model_name="llama-3-8b",
        backend="cuda",
    )
    store.add_feedback(record)
    assert len(store) == 1

    history = store.get_history(model_name="llama-3-8b", backend="cuda")
    assert len(history) == 1
    assert history[0].prompt_tps == 350.0

    avg_tps = store.get_average_tps_for_microbatch(microbatch=512, model_name="llama-3-8b")
    assert avg_tps == 350.0


# ---------------------------------------------------------------------------
# 6. CLI Output Formatting Tests
# ---------------------------------------------------------------------------

def test_cli_output_formatting():
    decision = SchedulingDecision(
        microbatch=512,
        confidence=0.93,
        optimization_goal="throughput",
        candidate_scores={256: 72.0, 512: 91.0, 768: 84.0},
        candidate_reasons={256: "Sub-optimal GPU utilization", 512: "Optimal", 768: "Higher VRAM pressure"},
        reasoning=["Highest predicted throughput while remaining within VRAM budget."],
    )
    formatted = decision.format_cli_output()
    assert "Dynamic Microbatch Scheduler" in formatted
    assert "Candidates:" in formatted
    assert "256: Score 72" in formatted
    assert "512: Score 91" in formatted
    assert "Selected:" in formatted
    assert "512" in formatted
    assert "Reason:" in formatted


# ---------------------------------------------------------------------------
# 7. ArgumentBuilder & Integration Tests
# ---------------------------------------------------------------------------

def test_argument_builder_microbatch_flag(tmp_path):
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
            LayerPlacement(
                layer_index=i,
                layer_type="transformer",
                device=PlacementDevice.GPU,
                gpu_index=0,
                size_bytes=100 * 1024 * 1024,
                cost=LayerCostBreakdown(),
            ) for i in range(32)
        ],
        context_length=4096,
        is_feasible=True,
    )
    config = RuntimeConfig(batch_size=512)
    backend = BackendInfo(name="cuda", n_gpu_layers=32)

    args = builder.build(
        model_path=tmp_path / "model.gguf",
        prompt="Test prompt",
        plan=plan,
        config=config,
        backend=backend,
        microbatch=384,
    )

    assert "-ub" in args
    ub_idx = args.index("-ub")
    assert args[ub_idx + 1] == "384"
    assert "-b" in args
    b_idx = args.index("-b")
    assert args[b_idx + 1] == "512"  # batch size max(512, 384)
