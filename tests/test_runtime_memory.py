"""
test_runtime_memory.py
-----------------------
Comprehensive unit test suite for Phase 6 Runtime Memory Optimization.

Test classes:
  1. TestKvCacheEstimator — GQA math, growth rate, max_safe_context, alignment
  2. TestBufferPlanner — activation workspace, backend overhead, total reservation
  3. TestRiskClassifier — LOW/MEDIUM/HIGH/CRITICAL classification, OOM probability
  4. TestWarningGenerator — structured warning conditions (VRAM, RAM, KV ratio, etc.)
  5. TestMemoryBudget — report formatting, JSON serialization, peak calculations
  6. TestPreflightGuard — go/no-go gate, InferenceBlockedError, auto-resize
  7. TestInferenceSessionIntegration — preflight check opt-in in InferenceSession
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime_memory import (
    BufferPlan,
    BufferPlanner,
    InferenceBlockedError,
    KvCacheEstimator,
    KvEstimate,
    MemoryBudget,
    OomRisk,
    PreflightGuard,
    RiskClassifier,
    WarningGenerator,
)
from layer_placement.model_descriptor import ModelDescriptor, LayerDescriptor
from layer_placement.placement_plan import (
    LayerCostBreakdown,
    LayerPlacement,
    PlacementDevice,
    PlacementPlan,
    build_segments,
)
from inference_runtime.runtime_config import RuntimeConfig
from inference_runtime.inference_session import InferenceSession


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_model_descriptor() -> ModelDescriptor:
    """ModelDescriptor for a Llama-3-8B-like model (32 layers, GQA 32:8)."""
    metadata = {
        "arch": "llama",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,  # GQA ratio = 1:4
        "max_context_length": 8192,
        "vocab_size": 128256,
    }
    return ModelDescriptor.from_gguf_metadata(
        metadata=metadata,
        model_size_bytes=4_500_000_000,
        quant_type="Q4_K_M",
        model_name="Llama-3-8B-Q4_K_M",
    )


@pytest.fixture
def mock_placement_plan(mock_model_descriptor: ModelDescriptor) -> PlacementPlan:
    """PlacementPlan with 24 GPU layers and 8 CPU layers."""
    placements = []
    for layer in mock_model_descriptor.layers:
        device = PlacementDevice.GPU if layer.layer_index < 26 else PlacementDevice.CPU
        gpu_idx = 0 if device == PlacementDevice.GPU else None
        placements.append(
            LayerPlacement(
                layer_index=layer.layer_index,
                layer_type=layer.layer_type,
                device=device,
                gpu_index=gpu_idx,
                size_bytes=layer.size_bytes,
                cost=LayerCostBreakdown(),
            )
        )
    segs = build_segments(placements)
    return PlacementPlan(
        model_name=mock_model_descriptor.name,
        architecture=mock_model_descriptor.architecture,
        total_layers=len(mock_model_descriptor.layers),
        n_gpu_layers=24,
        n_cpu_layers=8,
        layer_placements=placements,
        segments=segs,
        estimated_vram_bytes=3_500_000_000,
        estimated_ram_bytes=1_000_000_000,
        context_length=8192,
    )


@pytest.fixture
def hw_profile_8gb() -> dict:
    return {
        "gpus": [{"model": "RTX 3070", "vendor": "nvidia", "vram_total_mb": 8192}],
        "ram": {"total_bytes": 16 * 1024 ** 3, "available_gb": 16.0},
        "cpu": {"physical_cores": 8},
    }


@pytest.fixture
def hw_profile_24gb() -> dict:
    return {
        "gpus": [{"model": "RTX 3090", "vendor": "nvidia", "vram_total_mb": 24576}],
        "ram": {"total_bytes": 32 * 1024 ** 3, "available_gb": 32.0},
        "cpu": {"physical_cores": 16},
    }


# ===========================================================================
# 1. TestKvCacheEstimator
# ===========================================================================

class TestKvCacheEstimator:

    def test_kv_per_token_formula_correctness(self, mock_model_descriptor, mock_placement_plan):
        estimator = KvCacheEstimator(dtype_bytes=2)
        est = estimator.estimate(mock_model_descriptor, mock_placement_plan, context_length=4096)
        
        # Formula: 2 * num_kv_heads (8) * head_dim (128) * dtype (2) = 4096 bytes/token/layer
        expected_per_token_per_layer = 2 * 8 * 128 * 2
        assert est.kv_bytes_per_token_per_layer == expected_per_token_per_layer
        
        # 25 GPU transformer layers in fixture (indices 1..25)
        assert est.kv_bytes_per_token == expected_per_token_per_layer * est.n_gpu_layers

    def test_gqa_reduces_kv_size(self):
        estimator = KvCacheEstimator(dtype_bytes=2)
        # GQA 32:8 heads vs Full MHA 32:32 heads
        est_gqa = estimator.estimate_from_params(num_kv_heads=8, head_dim=128, n_gpu_layers=32, context_length=4096)
        est_mha = estimator.estimate_from_params(num_kv_heads=32, head_dim=128, n_gpu_layers=32, context_length=4096)
        
        assert est_gqa.kv_bytes_per_token == est_mha.kv_bytes_per_token / 4

    def test_kv_grows_linearly_with_context(self, mock_model_descriptor, mock_placement_plan):
        estimator = KvCacheEstimator(dtype_bytes=2)
        est = estimator.estimate(mock_model_descriptor, mock_placement_plan, context_length=8192)
        
        assert est.kv_at_full_context == est.kv_bytes_per_token * 8192
        assert est.kv_at_half_context == est.kv_bytes_per_token * 4096
        assert est.kv_at_quarter_context == est.kv_bytes_per_token * 2048

    def test_max_safe_context_fits_in_headroom(self, mock_model_descriptor, mock_placement_plan):
        estimator = KvCacheEstimator(dtype_bytes=2)
        headroom = 4 * 1024 ** 3  # 4 GB
        safe_ctx = estimator.max_safe_context(mock_model_descriptor, mock_placement_plan, headroom)
        
        est = estimator.estimate(mock_model_descriptor, mock_placement_plan, safe_ctx)
        assert est.kv_at_full_context <= headroom

    def test_max_safe_context_multiple_of_512(self, mock_model_descriptor, mock_placement_plan):
        estimator = KvCacheEstimator(dtype_bytes=2)
        headroom = 3 * 1024 ** 3
        safe_ctx = estimator.max_safe_context(mock_model_descriptor, mock_placement_plan, headroom)
        assert safe_ctx % 512 == 0
        assert safe_ctx >= 512

    def test_zero_gpu_layers_zero_vram_kv(self, mock_model_descriptor):
        # Plan with 0 GPU layers
        placements = [
            LayerPlacement(
                layer_index=i, layer_type=l.layer_type, device=PlacementDevice.CPU,
                gpu_index=None, size_bytes=l.size_bytes, cost=LayerCostBreakdown(),
            ) for i, l in enumerate(mock_model_descriptor.layers)
        ]
        plan = PlacementPlan(
            model_name="CPUPlan", architecture="llama", total_layers=34,
            n_gpu_layers=0, n_cpu_layers=32, layer_placements=placements, segments=[],
            estimated_vram_bytes=0, estimated_ram_bytes=4_500_000_000, context_length=4096,
        )
        estimator = KvCacheEstimator()
        est = estimator.estimate(mock_model_descriptor, plan, 4096)
        assert est.kv_bytes_per_token == 0
        assert est.kv_at_full_context == 0


# ===========================================================================
# 2. TestBufferPlanner
# ===========================================================================

class TestBufferPlanner:

    def test_activation_workspace_positive(self, mock_model_descriptor, mock_placement_plan):
        planner = BufferPlanner()
        plan = planner.plan(mock_model_descriptor, mock_placement_plan, context_length=4096, batch_size=512)
        assert plan.activation_workspace_bytes > 0

    def test_backend_overhead_nonzero(self, mock_model_descriptor, mock_placement_plan):
        planner = BufferPlanner()
        plan_cuda = planner.plan(mock_model_descriptor, mock_placement_plan, 4096, backend="cuda")
        plan_vulkan = planner.plan(mock_model_descriptor, mock_placement_plan, 4096, backend="vulkan")
        assert plan_cuda.backend_overhead_bytes > 0
        assert plan_vulkan.backend_overhead_bytes > 0
        assert plan_cuda.backend_overhead_bytes != plan_vulkan.backend_overhead_bytes

    def test_kv_reservation_matches_estimator(self, mock_model_descriptor, mock_placement_plan):
        planner = BufferPlanner()
        plan = planner.plan(mock_model_descriptor, mock_placement_plan, context_length=4096)
        
        estimator = KvCacheEstimator()
        est = estimator.estimate(mock_model_descriptor, mock_placement_plan, 4096)
        assert plan.kv_reservation_bytes >= est.kv_at_full_context

    def test_total_reserved_is_sum(self, mock_model_descriptor, mock_placement_plan):
        planner = BufferPlanner()
        plan = planner.plan(mock_model_descriptor, mock_placement_plan, context_length=4096)
        expected = plan.activation_workspace_bytes + plan.backend_overhead_bytes + plan.kv_reservation_bytes
        assert plan.total_reserved_bytes == expected


# ===========================================================================
# 3. TestRiskClassifier
# ===========================================================================

class TestRiskClassifier:

    def test_low_risk_classification(self):
        classifier = RiskClassifier()
        # Ratio <= 0.70 -> LOW
        risk, prob = classifier.classify(vram_total_bytes=10_000_000_000, peak_vram_bytes=6_000_000_000, kv_fill_ratio=0.50)
        assert risk == OomRisk.LOW
        assert 0.0 <= prob <= 0.10

    def test_medium_risk_classification(self):
        classifier = RiskClassifier()
        # Ratio 0.80 -> MEDIUM
        risk, prob = classifier.classify(vram_total_bytes=10_000_000_000, peak_vram_bytes=8_000_000_000, kv_fill_ratio=0.80)
        assert risk == OomRisk.MEDIUM
        assert 0.10 < prob <= 0.30

    def test_high_risk_classification(self):
        classifier = RiskClassifier()
        # Ratio 0.90 -> HIGH
        risk, prob = classifier.classify(vram_total_bytes=10_000_000_000, peak_vram_bytes=9_000_000_000, kv_fill_ratio=0.90)
        assert risk == OomRisk.HIGH
        assert 0.30 < prob <= 0.80

    def test_critical_risk_classification(self):
        classifier = RiskClassifier()
        # Ratio 0.99 -> CRITICAL
        risk, prob = classifier.classify(vram_total_bytes=10_000_000_000, peak_vram_bytes=9_900_000_000, kv_fill_ratio=0.99)
        assert risk == OomRisk.CRITICAL
        assert prob > 0.70

    def test_oom_probability_increases_with_fill_ratio(self):
        classifier = RiskClassifier()
        r1, p1 = classifier.classify(100, 50, 0.50)
        r2, p2 = classifier.classify(100, 80, 0.80)
        r3, p3 = classifier.classify(100, 95, 0.95)
        assert p1 < p2 < p3

    def test_oom_probability_clamped_0_to_1(self):
        classifier = RiskClassifier()
        _, p_low = classifier.classify(100, 10, 0.0)
        _, p_high = classifier.classify(100, 200, 2.0)
        assert 0.0 <= p_low <= 1.0
        assert 0.0 <= p_high <= 1.0


# ===========================================================================
# 4. TestMemoryBudget
# ===========================================================================

class TestMemoryBudget:

    def test_budget_report_contains_required_fields(self):
        budget = MemoryBudget(
            model_name="Llama-3-8B-Q4_K_M",
            context_length=8192,
            effective_context=8192,
            kv_bytes_per_token=524288,
            kv_at_quarter_context=1_000_000_000,
            kv_at_half_context=2_000_000_000,
            kv_at_full_context=7_840_000_000,
            kv_growth_rate_gb_per_1k=0.50,
            weights_vram_bytes=4_500_000_000,
            weights_ram_bytes=0,
            activation_workspace_bytes=134_217_728,
            backend_overhead_bytes=314_572_800,
            kv_reservation_bytes=8_000_000_000,
            peak_vram_bytes=12_948_790_528,
            peak_ram_bytes=1_073_741_824,
            peak_total_bytes=14_022_532_352,
            risk_level="LOW",
            oom_probability=0.032,
            kv_fill_ratio=0.58,
            warnings=[],
            safe_context_length=16384,
            auto_resized=False,
            backend="vulkan",
        )
        report = budget.report()
        assert "Llama-3-8B-Q4_K_M" in report
        assert "8,192 tokens" in report
        assert "LOW" in report
        assert "3.2%" in report
        assert "0.58" in report

    def test_budget_to_dict_serializable(self):
        budget = MemoryBudget(
            model_name="TestModel",
            context_length=4096, effective_context=4096,
            kv_bytes_per_token=1000, kv_at_quarter_context=100, kv_at_half_context=200,
            kv_at_full_context=400, kv_growth_rate_gb_per_1k=0.001,
            weights_vram_bytes=1000, weights_ram_bytes=500,
            activation_workspace_bytes=100, backend_overhead_bytes=100, kv_reservation_bytes=400,
            peak_vram_bytes=1600, peak_ram_bytes=1500, peak_total_bytes=3100,
            risk_level="LOW", oom_probability=0.01, kv_fill_ratio=0.2,
        )
        json_str = budget.to_json()
        data = json.loads(json_str)
        assert data["model_name"] == "TestModel"
        assert data["risk"]["level"] == "LOW"

    def test_peak_vram_is_sum_of_components(self):
        weights = 4_000_000_000
        kv_res = 2_000_000_000
        act = 100_000_000
        backend = 300_000_000
        budget = MemoryBudget(
            model_name="M", context_length=4096, effective_context=4096,
            kv_bytes_per_token=100, kv_at_quarter_context=1, kv_at_half_context=2,
            kv_at_full_context=kv_res, kv_growth_rate_gb_per_1k=0.1,
            weights_vram_bytes=weights, weights_ram_bytes=0,
            activation_workspace_bytes=act, backend_overhead_bytes=backend, kv_reservation_bytes=kv_res,
            peak_vram_bytes=weights + kv_res + act + backend,
            peak_ram_bytes=1000, peak_total_bytes=weights + kv_res + act + backend + 1000,
            risk_level="LOW", oom_probability=0.01, kv_fill_ratio=0.3,
        )
        assert budget.peak_vram_bytes == weights + kv_res + act + backend


# ===========================================================================
# 5. TestWarningGenerator
# ===========================================================================

class TestWarningGenerator:

    def test_high_vram_generates_warning(self, hw_profile_8gb):
        warner = WarningGenerator()
        budget = MemoryBudget(
            model_name="M", context_length=8192, effective_context=8192,
            kv_bytes_per_token=1000, kv_at_quarter_context=1, kv_at_half_context=2,
            kv_at_full_context=4000, kv_growth_rate_gb_per_1k=0.1,
            weights_vram_bytes=5_000_000_000, weights_ram_bytes=0,
            activation_workspace_bytes=100_000_000, backend_overhead_bytes=300_000_000,
            kv_reservation_bytes=2_500_000_000,
            peak_vram_bytes=7_900_000_000,  # > 90% of 8GB (7.2GB)
            peak_ram_bytes=1_000_000_000, peak_total_bytes=8_900_000_000,
            risk_level="HIGH", oom_probability=0.6, kv_fill_ratio=0.89,
        )
        warnings = warner.generate(budget, hw_profile_8gb)
        assert any("VRAM utilisation is very high" in w for w in warnings)

    def test_no_gpu_layers_generates_warning(self, hw_profile_8gb):
        warner = WarningGenerator()
        budget = MemoryBudget(
            model_name="M", context_length=4096, effective_context=4096,
            kv_bytes_per_token=0, kv_at_quarter_context=0, kv_at_half_context=0,
            kv_at_full_context=0, kv_growth_rate_gb_per_1k=0.0,
            weights_vram_bytes=0, weights_ram_bytes=4_000_000_000,
            activation_workspace_bytes=100_000_000, backend_overhead_bytes=10_000_000,
            kv_reservation_bytes=0, peak_vram_bytes=110_000_000,
            peak_ram_bytes=5_000_000_000, peak_total_bytes=5_110_000_000,
            risk_level="LOW", oom_probability=0.0, kv_fill_ratio=0.0,
        )
        warnings = warner.generate(budget, hw_profile_8gb)
        assert any("No model layers are placed on GPU" in w for w in warnings)

    def test_auto_resized_generates_warning(self, hw_profile_8gb):
        warner = WarningGenerator()
        budget = MemoryBudget(
            model_name="M", context_length=8192, effective_context=4096,
            kv_bytes_per_token=1000, kv_at_quarter_context=1, kv_at_half_context=2,
            kv_at_full_context=1000, kv_growth_rate_gb_per_1k=0.1,
            weights_vram_bytes=3_000_000_000, weights_ram_bytes=0,
            activation_workspace_bytes=100_000_000, backend_overhead_bytes=300_000_000,
            kv_reservation_bytes=1_000_000_000, peak_vram_bytes=4_400_000_000,
            peak_ram_bytes=1_000_000_000, peak_total_bytes=5_400_000_000,
            risk_level="LOW", oom_probability=0.05, kv_fill_ratio=0.3,
            auto_resized=True, safe_context_length=4096,
        )
        warnings = warner.generate(budget, hw_profile_8gb)
        assert any("Context window automatically reduced" in w for w in warnings)

    def test_clean_run_no_warnings(self, hw_profile_24gb):
        warner = WarningGenerator()
        budget = MemoryBudget(
            model_name="M", context_length=2048, effective_context=2048,
            kv_bytes_per_token=1000, kv_at_quarter_context=1, kv_at_half_context=2,
            kv_at_full_context=1000, kv_growth_rate_gb_per_1k=0.1,
            weights_vram_bytes=4_000_000_000, weights_ram_bytes=0,
            activation_workspace_bytes=100_000_000, backend_overhead_bytes=300_000_000,
            kv_reservation_bytes=1_000_000_000, peak_vram_bytes=5_400_000_000,
            peak_ram_bytes=1_000_000_000, peak_total_bytes=6_400_000_000,
            risk_level="LOW", oom_probability=0.01, kv_fill_ratio=0.1,
        )
        warnings = warner.generate(budget, hw_profile_24gb)
        assert len(warnings) == 0


# ===========================================================================
# 6. TestPreflightGuard
# ===========================================================================

class TestPreflightGuard:

    def test_low_risk_returns_budget(self, mock_model_descriptor, mock_placement_plan, hw_profile_24gb):
        guard = PreflightGuard(hw_profile_24gb)
        budget = guard.check(mock_model_descriptor, mock_placement_plan, context_length=4096)
        assert isinstance(budget, MemoryBudget)
        assert budget.risk_level == "LOW"
        assert not budget.auto_resized

    def test_critical_risk_raises_inference_blocked(self, mock_model_descriptor, mock_placement_plan):
        # 2 GB VRAM profile -> guaranteed OOM with 24 GPU layers
        tiny_hw = {"gpus": [{"vram_total_mb": 2048}], "ram": {"total_bytes": 16 * 1024 ** 3}}
        guard = PreflightGuard(tiny_hw, block_on_critical=True)
        with pytest.raises(InferenceBlockedError) as exc_info:
            guard.check(mock_model_descriptor, mock_placement_plan, context_length=16384)
        assert exc_info.value.budget.risk_level == "CRITICAL"

    def test_high_risk_auto_resizes_context(self, mock_model_descriptor, mock_placement_plan):
        # 6 GB VRAM profile -> HIGH risk at 8192 context
        med_hw = {"gpus": [{"vram_total_mb": 6144}], "ram": {"total_bytes": 16 * 1024 ** 3}}
        guard = PreflightGuard(med_hw, auto_resize_buffers=True, block_on_critical=False)
        budget = guard.check(mock_model_descriptor, mock_placement_plan, context_length=8192)
        if budget.auto_resized:
            assert budget.effective_context < 8192
            assert budget.effective_context % 512 == 0

    def test_block_on_critical_false_does_not_raise(self, mock_model_descriptor, mock_placement_plan):
        tiny_hw = {"gpus": [{"vram_total_mb": 2048}], "ram": {"total_bytes": 16 * 1024 ** 3}}
        guard = PreflightGuard(tiny_hw, block_on_critical=False)
        budget = guard.check(mock_model_descriptor, mock_placement_plan, context_length=16384)
        assert budget.risk_level == "CRITICAL"

    def test_auto_resize_false_preserves_context(self, mock_model_descriptor, mock_placement_plan):
        med_hw = {"gpus": [{"vram_total_mb": 6144}], "ram": {"total_bytes": 16 * 1024 ** 3}}
        guard = PreflightGuard(med_hw, auto_resize_buffers=False, block_on_critical=False)
        budget = guard.check(mock_model_descriptor, mock_placement_plan, context_length=8192)
        assert not budget.auto_resized
        assert budget.effective_context == 8192

    def test_snake_case_alias_check_plan(self, mock_model_descriptor, mock_placement_plan, hw_profile_24gb):
        guard = PreflightGuard(hw_profile_24gb)
        budget = guard.check_plan(mock_model_descriptor, mock_placement_plan, context_length=2048)
        assert isinstance(budget, MemoryBudget)

    def test_inference_blocked_error_has_budget_attr(self, mock_model_descriptor, mock_placement_plan):
        tiny_hw = {"gpus": [{"vram_total_mb": 2048}], "ram": {"total_bytes": 16 * 1024 ** 3}}
        guard = PreflightGuard(tiny_hw, block_on_critical=True)
        try:
            guard.check(mock_model_descriptor, mock_placement_plan, context_length=16384)
        except InferenceBlockedError as e:
            assert hasattr(e, "budget")
            assert isinstance(e.budget, MemoryBudget)


# ===========================================================================
# 7. TestInferenceSessionIntegration
# ===========================================================================

class TestInferenceSessionIntegration:

    def test_inference_session_runs_preflight_when_configured(
        self, mock_placement_plan, hw_profile_24gb, tmp_path
    ):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"0" * 1000)
        exe_file = tmp_path / "llama.exe"
        exe_file.write_bytes(b"0" * 1000)

        cfg = RuntimeConfig(run_preflight_check=True)
        session = InferenceSession(
            model_path=model_file,
            plan=mock_placement_plan,
            config=cfg,
            hw_profile=hw_profile_24gb,
            llama_exe_path=exe_file,
        )
        assert session.memory_budget is not None
        assert isinstance(session.memory_budget, MemoryBudget)
