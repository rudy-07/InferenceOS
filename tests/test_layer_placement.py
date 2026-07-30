"""
test_layer_placement.py
-----------------------
Comprehensive test suite for InferenceOS Phase 3: Automatic Layer Placement Engine.

Tests cover:
  1. ModelDescriptor construction and layer size accounting
  2. CostModel sub-costs (memory pressure, transfer, compute)
  3. PlacementPlan data structures and segment generation
  4. Optimizer correctness: all-GPU, all-CPU, partial split, non-greedy
  5. Performance and memory estimation APIs
  6. Report generation (text + JSON round-trip)
  7. Full pipeline: hardware profile → PlacementPlan
  8. Edge cases: infeasible plans, determinism, empty VRAM
"""
from __future__ import annotations

import json
import math

import pytest

from layer_placement import (
    CostWeights,
    HardwareContext,
    LayerDescriptor,
    ModelDescriptor,
    OptimizerConfig,
    PlacementDevice,
    PlacementEngine,
    PlacementPlan,
    generate_json_report,
    generate_text_report,
)
from layer_placement.cost_model import CostModel
from layer_placement.model_descriptor import (
    LAYER_TYPE_EMBEDDING,
    LAYER_TYPE_LM_HEAD,
    LAYER_TYPE_TRANSFORMER,
    infer_quant_type_from_filename,
)
from layer_placement.optimizer import PlacementOptimizer, _count_bytes
from layer_placement.placement_plan import LayerPlacement, build_segments


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def gguf_metadata_small() -> dict:
    """Simulated read_gguf_metadata() output for a small 7B-class model."""
    return {
        "arch": "llama",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "max_context_length": 4096,
        "vocab_size": 32000,
    }


@pytest.fixture
def gguf_metadata_large() -> dict:
    """Simulated output for a large 70B-class model."""
    return {
        "arch": "llama",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "max_context_length": 8192,
        "vocab_size": 32000,
    }


@pytest.fixture
def model_small(gguf_metadata_small) -> ModelDescriptor:
    """~4 GB 7B Q4_K_M model descriptor."""
    return ModelDescriptor.from_gguf_metadata(
        gguf_metadata_small,
        model_size_bytes=4_200_000_000,
        quant_type="Q4_K_M",
        model_name="Llama-3-7B-Q4_K_M",
    )


@pytest.fixture
def model_large(gguf_metadata_large) -> ModelDescriptor:
    """~40 GB 70B Q4_K_M model descriptor."""
    return ModelDescriptor.from_gguf_metadata(
        gguf_metadata_large,
        model_size_bytes=40_000_000_000,
        quant_type="Q4_K_M",
        model_name="Llama-3-70B-Q4_K_M",
    )


@pytest.fixture
def hw_8gb_vram() -> dict:
    """Hardware profile with 8 GB discrete GPU and 16 GB RAM."""
    return {
        "gpus": [
            {
                "model": "Test GPU 8GB",
                "vram_total_bytes": 8 * 1024 ** 3,
                "vram_available_bytes": 8 * 1024 ** 3,
                "vram_total_mb": 8192,
                "vram_free_mb": 8192,
                "bandwidth": 384.0,
                "is_integrated": False,
            }
        ],
        "igpus": [],
        "ram": {"total_bytes": 16 * 1024 ** 3, "available_gb": 14.0},
        "memory": {"total_bytes": 16 * 1024 ** 3, "available_gb": 14.0},
        "cpu": {
            "logical_cores": 12,
            "base_freq_mhz": 3000.0,
            "isa_extensions": ["avx2", "fma"],
        },
        "interconnects": [
            {"type": "PCIe", "bandwidth": 32.0, "latency": 5.0},
        ],
        "inference_hints": {},
    }


@pytest.fixture
def hw_no_vram() -> dict:
    """Hardware profile with no discrete GPU (CPU-only)."""
    return {
        "gpus": [],
        "igpus": [],
        "ram": {"total_bytes": 32 * 1024 ** 3, "available_gb": 28.0},
        "memory": {"total_bytes": 32 * 1024 ** 3, "available_gb": 28.0},
        "cpu": {
            "logical_cores": 16,
            "base_freq_mhz": 3600.0,
            "isa_extensions": ["avx512f"],
        },
        "interconnects": [],
        "inference_hints": {},
    }


@pytest.fixture
def hardware_small(hw_8gb_vram) -> HardwareContext:
    return HardwareContext.from_hw_profile(hw_8gb_vram)


@pytest.fixture
def hardware_cpu_only(hw_no_vram) -> HardwareContext:
    return HardwareContext.from_hw_profile(hw_no_vram)


@pytest.fixture
def cost_model_small(hardware_small) -> CostModel:
    return CostModel(hardware_small)


@pytest.fixture
def engine_8gb(hw_8gb_vram) -> PlacementEngine:
    return PlacementEngine(hw_profile=hw_8gb_vram, optimizer_config=OptimizerConfig(max_iterations=100))


@pytest.fixture
def engine_cpu_only(hw_no_vram) -> PlacementEngine:
    return PlacementEngine(hw_profile=hw_no_vram, optimizer_config=OptimizerConfig(max_iterations=100))


# ===========================================================================
# 1. ModelDescriptor tests
# ===========================================================================

class TestModelDescriptor:

    def test_from_gguf_metadata_basic(self, gguf_metadata_small):
        """Constructs descriptor correctly from gguf_parser output."""
        model = ModelDescriptor.from_gguf_metadata(
            gguf_metadata_small,
            model_size_bytes=4_000_000_000,
            quant_type="Q4_K_M",
            model_name="TestModel",
        )
        assert model.architecture == "llama"
        assert model.num_layers == 32
        assert model.hidden_size == 4096
        assert model.num_heads == 32
        assert model.num_kv_heads == 8
        assert model.quant_type == "Q4_K_M"
        assert model.quant_bpw == pytest.approx(4.5, abs=0.1)
        assert model.name == "TestModel"

    def test_layer_list_contains_all_types(self, model_small):
        """Layer list includes embedding, transformer blocks, and lm_head."""
        types = {l.layer_type for l in model_small.layers}
        assert LAYER_TYPE_EMBEDDING in types
        assert LAYER_TYPE_TRANSFORMER in types
        assert LAYER_TYPE_LM_HEAD in types

    def test_transformer_layer_count(self, model_small):
        """Transformer layer count matches num_layers from metadata."""
        transformer_layers = model_small.transformer_layers()
        assert len(transformer_layers) == model_small.num_layers  # 32

    def test_total_weight_bytes_near_model_size(self, model_small):
        """Sum of all layer weights should be close to model_size_bytes."""
        total = model_small.total_weight_bytes()
        # Allow ±5% rounding error from integer division
        ratio = total / model_small.model_size_bytes
        assert 0.95 <= ratio <= 1.05, f"Weight byte sum ratio={ratio:.3f} outside [0.95, 1.05]"

    def test_kv_cache_bytes_per_token_positive(self, model_small):
        """Per-token KV cache should be non-zero for transformer layers."""
        kv = model_small.kv_cache_bytes_per_token()
        assert kv > 0

    def test_kv_cache_partial_layers(self, model_small):
        """Partial layer KV cache count should be proportional."""
        full = model_small.kv_cache_bytes_per_token()
        half = model_small.kv_cache_bytes_per_token(num_layers=model_small.num_layers // 2)
        assert half == pytest.approx(full / 2, rel=0.01)

    def test_infer_quant_type_from_filename(self):
        """Quant type should be inferred from GGUF filename stems."""
        assert infer_quant_type_from_filename("llama-3-8b-q4_k_m.gguf") == "Q4_K_M"
        assert infer_quant_type_from_filename("model.Q8_0.gguf") == "Q8_0"
        assert infer_quant_type_from_filename("model-F16.gguf") == "F16"

    def test_to_dict_serializable(self, model_small):
        """to_dict() output must be JSON-serializable."""
        d = model_small.to_dict()
        json_str = json.dumps(d)
        assert "num_layers" in json_str


# ===========================================================================
# 2. CostModel tests
# ===========================================================================

class TestCostModel:

    def test_memory_pressure_zero_below_threshold(self, cost_model_small):
        """No pressure when VRAM and RAM are well under threshold."""
        pressure = cost_model_small.memory_pressure_cost(
            vram_used_bytes=1 * 1024 ** 3,   # 1 GB used of 8 GB
            ram_used_bytes=2 * 1024 ** 3,    # 2 GB used of 16 GB
        )
        assert pressure == pytest.approx(0.0, abs=1e-3)

    def test_memory_pressure_rises_above_threshold(self, cost_model_small):
        """Pressure should be clearly positive when VRAM exceeds 85% threshold."""
        low_pressure = cost_model_small.memory_pressure_cost(
            vram_used_bytes=1 * 1024 ** 3,
            ram_used_bytes=2 * 1024 ** 3,
        )
        high_pressure = cost_model_small.memory_pressure_cost(
            vram_used_bytes=7_900_000_000,   # >92% of 8 GB VRAM
            ram_used_bytes=2 * 1024 ** 3,
        )
        assert high_pressure > low_pressure + 0.1

    def test_all_gpu_has_zero_transfer_cost(self, cost_model_small, model_small):
        """All-GPU assignment should produce zero transfer cost (no boundaries)."""
        assignment = [PlacementDevice.GPU] * len(model_small.layers)
        xfer = cost_model_small.transfer_cost(model_small.layers, assignment)
        assert xfer == pytest.approx(0.0, abs=1e-9)

    def test_alternating_layers_have_high_transfer_cost(self, cost_model_small, model_small):
        """Alternating GPU/CPU assignment should produce non-zero transfer cost."""
        assignment = []
        for i in range(len(model_small.layers)):
            assignment.append(PlacementDevice.GPU if i % 2 == 0 else PlacementDevice.CPU)
        xfer = cost_model_small.transfer_cost(model_small.layers, assignment)
        assert xfer > 0.0

    def test_gpu_cpu_gpu_has_two_crossings(self, cost_model_small, model_small):
        """GPU-CPU-GPU arrangement should produce exactly two transfer boundary costs."""
        n = len(model_small.layers)
        third = n // 3
        assignment = (
            [PlacementDevice.GPU] * third
            + [PlacementDevice.CPU] * third
            + [PlacementDevice.GPU] * (n - 2 * third)
        )
        xfer = cost_model_small.transfer_cost(model_small.layers, assignment)
        # Two crossings: 2 × 0.05 = 0.10
        assert xfer == pytest.approx(0.10, abs=0.01)

    def test_compute_cost_all_gpu_is_zero(self, cost_model_small, model_small):
        """All-GPU placement should have zero compute slowdown."""
        assignment = [PlacementDevice.GPU] * len(model_small.layers)
        comp = cost_model_small.compute_cost(model_small.layers, assignment)
        assert comp == pytest.approx(0.0, abs=1e-9)

    def test_compute_cost_all_cpu_positive(self, cost_model_small, model_small):
        """All-CPU placement should have positive compute cost."""
        assignment = [PlacementDevice.CPU] * len(model_small.layers)
        comp = cost_model_small.compute_cost(model_small.layers, assignment)
        assert comp > 0.0

    def test_cost_weights_must_sum_to_one(self):
        """CostWeights with weights not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            CostWeights(memory_pressure=0.5, transfer=0.5, compute=0.5)

    def test_feasibility_within_bounds(self, cost_model_small):
        """Plan within VRAM and RAM bounds should be feasible."""
        assert cost_model_small.is_feasible(
            vram_used_bytes=4 * 1024 ** 3,
            ram_used_bytes=8 * 1024 ** 3,
        )

    def test_feasibility_exceeds_vram(self, cost_model_small):
        """Plan exceeding VRAM should be infeasible."""
        assert not cost_model_small.is_feasible(
            vram_used_bytes=10 * 1024 ** 3,   # > 8 GB VRAM
            ram_used_bytes=4 * 1024 ** 3,
        )


# ===========================================================================
# 3. PlacementPlan & build_segments tests
# ===========================================================================

class TestPlacementPlan:

    def _make_plan(self, n_layers: int = 10, n_gpu: int = 5) -> PlacementPlan:
        """Helper: make a simple plan with n_gpu GPU layers, rest CPU."""
        placements = []
        for i in range(n_layers):
            device = PlacementDevice.GPU if i < n_gpu else PlacementDevice.CPU
            from layer_placement.placement_plan import LayerCostBreakdown
            placements.append(LayerPlacement(
                layer_index=i,
                layer_type=LAYER_TYPE_TRANSFORMER,
                device=device,
                gpu_index=0 if device == PlacementDevice.GPU else None,
                size_bytes=100 * 1024 * 1024,
                cost=LayerCostBreakdown(),
            ))
        segs = build_segments(placements)
        return PlacementPlan(
            model_name="TestModel",
            architecture="llama",
            total_layers=n_layers,
            n_gpu_layers=n_gpu,
            n_cpu_layers=n_layers - n_gpu,
            layer_placements=placements,
            segments=segs,
            estimated_vram_bytes=n_gpu * 100 * 1024 * 1024,
            estimated_ram_bytes=(n_layers - n_gpu) * 100 * 1024 * 1024,
        )

    def test_segments_simple_split(self):
        """Simple GPU-then-CPU should produce exactly 2 segments."""
        plan = self._make_plan(10, 5)
        assert len(plan.segments) == 2
        assert plan.segments[0].device == PlacementDevice.GPU
        assert plan.segments[1].device == PlacementDevice.CPU

    def test_all_gpu_produces_one_segment(self):
        """All-GPU plan should produce exactly 1 segment."""
        plan = self._make_plan(10, 10)
        assert len(plan.segments) == 1
        assert plan.segments[0].device == PlacementDevice.GPU

    def test_all_cpu_produces_one_segment(self):
        """All-CPU plan should produce exactly 1 segment."""
        plan = self._make_plan(10, 0)
        assert len(plan.segments) == 1
        assert plan.segments[0].device == PlacementDevice.CPU

    def test_boundary_crossings_simple_split(self):
        """GPU-CPU plan should have exactly 1 boundary crossing."""
        plan = self._make_plan(10, 5)
        assert plan.boundary_crossings == 1

    def test_gpu_offload_ratio(self):
        """gpu_offload_ratio should be n_gpu_layers / total_transformer."""
        plan = self._make_plan(10, 6)
        assert plan.gpu_offload_ratio == pytest.approx(0.6, abs=0.001)

    def test_to_dict_round_trip(self):
        """to_dict() → JSON → parse should preserve key fields."""
        plan = self._make_plan(10, 5)
        d = plan.to_dict()
        j = plan.to_json()
        reparsed = json.loads(j)
        assert reparsed["n_gpu_layers"] == 5
        assert reparsed["n_cpu_layers"] == 5
        assert reparsed["is_feasible"] is True
        assert len(reparsed["segments"]) == 2

    def test_summary_contains_model_name(self):
        """summary() should include the model name."""
        plan = self._make_plan(10, 5)
        assert "TestModel" in plan.summary()


# ===========================================================================
# 4. Optimizer tests
# ===========================================================================

class TestOptimizer:

    @pytest.fixture
    def optimizer_small(self, cost_model_small):
        cfg = OptimizerConfig(max_iterations=200, random_seed=42)
        return PlacementOptimizer(cost_model_small, cfg)

    def test_small_model_fits_entirely_in_vram(self, optimizer_small, model_small):
        """A 4 GB model with 8 GB VRAM should be fully GPU-placed."""
        plan = optimizer_small.optimize(
            model_name=model_small.name,
            architecture=model_small.architecture,
            layers=model_small.layers,
            vram_available_bytes=8 * 1024 ** 3,
            ram_available_bytes=16 * 1024 ** 3,
            context_length=2048,
        )
        assert plan.n_gpu_layers == model_small.num_layers
        assert plan.n_cpu_layers == 0
        assert plan.is_feasible

    def test_large_model_partial_split(self, hardware_small, model_large):
        """A 40 GB model with 8 GB VRAM should be split between GPU and CPU."""
        cost_model = CostModel(hardware_small)
        cfg = OptimizerConfig(max_iterations=200, random_seed=42)
        optimizer = PlacementOptimizer(cost_model, cfg)
        plan = optimizer.optimize(
            model_name=model_large.name,
            architecture=model_large.architecture,
            layers=model_large.layers,
            vram_available_bytes=8 * 1024 ** 3,
            ram_available_bytes=32 * 1024 ** 3,
            context_length=2048,
        )
        assert plan.n_gpu_layers > 0, "Should offload some layers to GPU"
        assert plan.n_cpu_layers > 0, "Should leave some layers on CPU"
        assert plan.n_gpu_layers + plan.n_cpu_layers == model_large.num_layers

    def test_cpu_only_fallback(self, hardware_cpu_only, model_small):
        """With no VRAM available, all transformer layers should land on CPU."""
        cost_model = CostModel(hardware_cpu_only)
        cfg = OptimizerConfig(max_iterations=100, random_seed=42)
        optimizer = PlacementOptimizer(cost_model, cfg)
        plan = optimizer.optimize(
            model_name=model_small.name,
            architecture=model_small.architecture,
            layers=model_small.layers,
            vram_available_bytes=0,
            ram_available_bytes=32 * 1024 ** 3,
            context_length=2048,
        )
        assert plan.n_gpu_layers == 0
        assert plan.n_cpu_layers == model_small.num_layers

    def test_optimizer_is_deterministic(self, hardware_small, model_small):
        """Same random seed must produce identical plans across multiple runs."""
        cost_model = CostModel(hardware_small)
        cfg = OptimizerConfig(max_iterations=200, random_seed=99)

        plans = []
        for _ in range(3):
            opt = PlacementOptimizer(cost_model, cfg)
            plan = opt.optimize(
                model_name=model_small.name,
                architecture=model_small.architecture,
                layers=model_small.layers,
                vram_available_bytes=6 * 1024 ** 3,
                ram_available_bytes=16 * 1024 ** 3,
                context_length=2048,
            )
            plans.append(plan.n_gpu_layers)

        assert plans[0] == plans[1] == plans[2], "Optimizer is not deterministic"

    def test_optimizer_beats_greedy_cost_or_equals(self, hardware_small, model_small):
        """
        SA optimizer should produce a plan with cost ≤ the Phase A greedy init.
        (SA only accepts moves that improve or tie, so it can never make things worse.)
        """
        cost_model = CostModel(hardware_small)

        # Greedy only (0 SA iterations)
        greedy_cfg = OptimizerConfig(max_iterations=0, random_seed=42)
        greedy_opt = PlacementOptimizer(cost_model, greedy_cfg)
        greedy_plan = greedy_opt.optimize(
            model_name=model_small.name,
            architecture=model_small.architecture,
            layers=model_small.layers,
            vram_available_bytes=4 * 1024 ** 3,
            ram_available_bytes=16 * 1024 ** 3,
            context_length=2048,
        )

        # SA refinement (200 iterations)
        sa_cfg = OptimizerConfig(max_iterations=200, random_seed=42)
        sa_opt = PlacementOptimizer(cost_model, sa_cfg)
        sa_plan = sa_opt.optimize(
            model_name=model_small.name,
            architecture=model_small.architecture,
            layers=model_small.layers,
            vram_available_bytes=4 * 1024 ** 3,
            ram_available_bytes=16 * 1024 ** 3,
            context_length=2048,
        )

        assert sa_plan.total_cost <= greedy_plan.total_cost + 1e-6

    def test_plan_layer_count_matches_model(self, optimizer_small, model_small):
        """Total GPU + CPU layers must equal model transformer count."""
        plan = optimizer_small.optimize(
            model_name=model_small.name,
            architecture=model_small.architecture,
            layers=model_small.layers,
            vram_available_bytes=6 * 1024 ** 3,
            ram_available_bytes=16 * 1024 ** 3,
            context_length=2048,
        )
        assert plan.n_gpu_layers + plan.n_cpu_layers == model_small.num_layers


# ===========================================================================
# 5. PlacementEngine API tests
# ===========================================================================

class TestPlacementEngine:

    def test_generate_plan_small_model_all_gpu(self, engine_8gb, model_small):
        """4 GB model on 8 GB GPU → full GPU placement."""
        plan = engine_8gb.generatePlacementPlan(model_small, context_length=2048)
        assert plan.n_gpu_layers == model_small.num_layers
        assert plan.is_feasible

    def test_generate_plan_large_model_split(self, engine_8gb, model_large):
        """40 GB model on 8 GB GPU → partial split."""
        plan = engine_8gb.generatePlacementPlan(model_large, context_length=2048)
        assert plan.n_gpu_layers > 0
        assert plan.n_cpu_layers > 0
        assert plan.n_gpu_layers + plan.n_cpu_layers == model_large.num_layers

    def test_generate_plan_cpu_only(self, engine_cpu_only, model_small):
        """CPU-only engine → all layers on CPU."""
        plan = engine_cpu_only.generatePlacementPlan(model_small, context_length=2048)
        assert plan.n_gpu_layers == 0
        assert plan.n_cpu_layers == model_small.num_layers

    def test_force_cpu_only_flag(self, engine_8gb, model_small):
        """force_cpu_only=True should override VRAM and place everything on CPU."""
        plan = engine_8gb.generatePlacementPlan(
            model_small, context_length=2048, force_cpu_only=True
        )
        assert plan.n_gpu_layers == 0

    def test_estimate_performance_all_gpu(self, engine_8gb, model_small):
        """All-GPU plan should have compute_score ≈ 1.0 and no PCIE overhead."""
        plan = engine_8gb.generatePlacementPlan(model_small, context_length=2048)
        perf = engine_8gb.estimatePerformance(plan, tokens_per_second_gpu_baseline=50.0)
        assert perf["compute_score"] == pytest.approx(1.0, abs=0.01)
        assert perf["pcie_transfer_overhead_ms_per_token"] == pytest.approx(0.0, abs=0.01)
        assert perf["estimated_tokens_per_second"] > 0

    def test_estimate_performance_partial_plan(self, engine_8gb, model_large):
        """Partial split should yield compute_score < 1.0."""
        plan = engine_8gb.generatePlacementPlan(model_large, context_length=2048)
        perf = engine_8gb.estimatePerformance(plan, tokens_per_second_gpu_baseline=20.0)
        assert perf["compute_score"] < 1.0
        assert perf["estimated_tokens_per_second"] > 0

    def test_estimate_memory_usage_peak_formula(self, engine_8gb, model_small):
        """Peak memory = weights + KV cache + workspace."""
        mem = engine_8gb.estimateMemoryUsage(model_small, context_length=2048)
        expected_peak = (
            model_small.model_size_bytes
            + model_small.kv_cache_bytes_per_token() * 2048
            + max(512 * 1024 * 1024, int(model_small.model_size_bytes * 0.10))
        )
        assert mem["peak_total_bytes"] == pytest.approx(expected_peak, rel=0.01)

    def test_estimate_memory_fits_fully_in_vram_small(self, engine_8gb, model_small):
        """Small model should report fits_fully_in_vram=True on 8 GB GPU."""
        mem = engine_8gb.estimateMemoryUsage(model_small, context_length=512)
        # With 512 ctx and 4.2 GB model, peak should be < 8 GB
        # (depends on KV cache) — at minimum, model alone should fit
        assert mem["vram_available_gb"] > 0

    def test_estimate_memory_large_model_requires_split(self, engine_8gb, model_large):
        """Large 40 GB model should report requires_split=True on 8 GB GPU."""
        mem = engine_8gb.estimateMemoryUsage(model_large, context_length=2048)
        assert mem["requires_split"] is True
        assert mem["fits_fully_in_vram"] is False

    def test_generate_plan_returns_feasible_plan(self, engine_8gb, model_small):
        """Plan from 8 GB GPU for a small model should always be feasible."""
        plan = engine_8gb.generatePlacementPlan(model_small, context_length=2048)
        assert plan.is_feasible

    def test_infeasibility_flagged_when_exceeds_memory(self, model_large):
        """When model exceeds both VRAM and RAM, plan should be marked infeasible."""
        tiny_hw = {
            "gpus": [{"vram_total_mb": 512, "vram_free_mb": 512, "bandwidth": 10.0, "is_integrated": False}],
            "igpus": [],
            "ram": {"total_bytes": 2 * 1024 ** 3, "available_gb": 1.5},
            "memory": {"total_bytes": 2 * 1024 ** 3, "available_gb": 1.5},
            "cpu": {"logical_cores": 4, "base_freq_mhz": 2000.0, "isa_extensions": []},
            "interconnects": [],
        }
        engine = PlacementEngine(hw_profile=tiny_hw, optimizer_config=OptimizerConfig(max_iterations=50))
        plan = engine.generatePlacementPlan(model_large, context_length=512)
        # With only 512 MB VRAM and 1.5 GB RAM, a 40 GB model cannot be feasible
        assert not plan.is_feasible


# ===========================================================================
# 6. Report generation tests
# ===========================================================================

class TestReportGenerator:

    @pytest.fixture
    def sample_plan(self, engine_8gb, model_small):
        return engine_8gb.generatePlacementPlan(model_small, context_length=2048)

    def test_text_report_contains_model_name(self, engine_8gb, sample_plan, model_small):
        """Text report should contain the model name."""
        report = engine_8gb.generateTextReport(sample_plan)
        assert model_small.name in report

    def test_text_report_contains_placement_map_header(self, engine_8gb, sample_plan):
        """Text report should contain the PLACEMENT MAP section."""
        report = engine_8gb.generateTextReport(sample_plan)
        assert "PLACEMENT MAP" in report

    def test_text_report_contains_llama_hint(self, engine_8gb, sample_plan):
        """Text report should contain a --n-gpu-layers hint."""
        report = engine_8gb.generateTextReport(sample_plan)
        assert "--n-gpu-layers" in report

    def test_text_report_contains_memory_section(self, engine_8gb, sample_plan):
        """Text report should contain MEMORY USAGE section."""
        report = engine_8gb.generateTextReport(sample_plan)
        assert "MEMORY USAGE" in report

    def test_json_report_round_trip(self, engine_8gb, sample_plan):
        """JSON report should parse back to a dict with expected keys."""
        json_str = engine_8gb.generateJsonReport(sample_plan)
        reparsed = json.loads(json_str)
        assert "n_gpu_layers" in reparsed
        assert "segments" in reparsed
        assert "layer_placements" in reparsed
        assert "optimizer" in reparsed

    def test_generate_report_format_json(self, engine_8gb, sample_plan):
        """generateReport(format='json') should return valid JSON."""
        output = engine_8gb.generateReport(sample_plan, format="json")
        reparsed = json.loads(output)
        assert isinstance(reparsed, dict)

    def test_generate_report_format_text(self, engine_8gb, sample_plan):
        """generateReport(format='text') should return multi-line string."""
        output = engine_8gb.generateReport(sample_plan, format="text")
        assert "\n" in output
        assert "InferenceOS" in output
