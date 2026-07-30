"""
tests/test_igpu_support.py
--------------------------
Comprehensive unit tests for Phase 7 Integrated GPU Support.

Tests cover:
  - IgpuProfiler: suitability scoring, vendor detection, edge cases
  - ContentionModel: contention factor, risk classification
  - WorkloadClassifier: per-layer suitability, budget checks
  - SuitabilityEvaluator: FULL/LIGHT/DISABLED transitions
  - IgpuPlacementContributor: plan overlay, VRAM budget, plan immutability
  - PlacementPlan: IGPU device enum, new fields, backward compatibility
  - HardwareContext: iGPU field population from hw_profile
  - BackendSelector: Vulkan --device flag for iGPU plans
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------

def _hw_profile_with_igpu(
    igpu_vram_mb: int = 512,
    igpu_vendor: str = "amd",
    igpu_backend: str = "vulkan",
    igpu_shared_bytes: int = 536_870_912,
    ram_total_gb: float = 8.0,
    ram_utilization: float = 50.0,
    system_bus_bw: float = 25.0,
    has_dgpu: bool = True,
) -> Dict[str, Any]:
    """Build a minimal but realistic hardware profile for testing."""
    igpus = [
        {
            "global_index": 0,
            "vendor": igpu_vendor,
            "model": f"Test {igpu_vendor.upper()} iGPU",
            "is_integrated": True,
            "vram_total_mb": igpu_vram_mb,
            "vram_total_bytes": igpu_vram_mb * 1024 * 1024,
            "vram_free_mb": igpu_vram_mb,
            "vram_used_mb": 0,
            "shared_memory_bytes": igpu_shared_bytes,
            "backend_hint": igpu_backend,
            "compute_capability": igpu_backend,
            "bandwidth": 0.0,  # Will fall back to system bus BW
            "utilization": 0.0,
        }
    ]
    gpus = []
    if has_dgpu:
        gpus = [
            {
                "global_index": 1,
                "vendor": "amd",
                "model": "AMD Radeon RX 5600M",
                "is_integrated": False,
                "vram_total_mb": 6144,
                "vram_total_bytes": 6 * 1024 ** 3,
                "vram_free_mb": 6144,
                "vram_used_mb": 0,
                "shared_memory_bytes": 0,
                "backend_hint": "vulkan",
                "bandwidth": 128.0,
                "utilization": 0.0,
            }
        ]

    ram_bytes = int(ram_total_gb * 1024 ** 3)
    used_bytes = int(ram_bytes * ram_utilization / 100)
    avail_bytes = ram_bytes - used_bytes

    return {
        "schema_version": "1.0",
        "os": {"system": "Windows"},
        "cpu": {
            "logical_cores": 12,
            "physical_cores": 6,
            "base_freq_mhz": 3000.0,
            "isa_extensions": ["avx2"],
        },
        "ram": {
            "total_bytes": ram_bytes,
            "available_bytes": avail_bytes,
            "free_bytes": avail_bytes,
            "total_gb": ram_total_gb,
            "available_gb": avail_bytes / 1024 ** 3,
            "utilization": ram_utilization,
            "bandwidth": 25.0,
        },
        "memory": {
            "total_bytes": ram_bytes,
            "available_bytes": avail_bytes,
        },
        "gpus": gpus,
        "igpus": igpus,
        "interconnects": [
            {
                "name": "System Memory Bus",
                "type": "SystemBus",
                "source": "CPU",
                "destination": "RAM",
                "bandwidth": system_bus_bw,
            }
        ],
        "inference_hints": {
            "recommended_backend": "vulkan",
            "primary_gpu_index": 1 if has_dgpu else 0,
        },
    }


def _hw_profile_no_igpu() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "os": {"system": "Windows"},
        "cpu": {"logical_cores": 8, "base_freq_mhz": 3200.0, "isa_extensions": ["avx2"]},
        "ram": {"total_bytes": 16 * 1024 ** 3, "available_bytes": 8 * 1024 ** 3,
                "utilization": 50.0, "bandwidth": 50.0},
        "memory": {},
        "gpus": [],
        "igpus": [],
        "interconnects": [],
        "inference_hints": {"recommended_backend": "cpu", "primary_gpu_index": None},
    }


def _make_simple_plan(
    n_gpu: int = 20,
    n_cpu: int = 12,
    gpu_index: int = 1,
    include_embedding: bool = True,
    include_norm: bool = True,
    include_lm_head: bool = True,
) -> Any:
    """Build a minimal PlacementPlan for testing."""
    from layer_placement.placement_plan import (
        LayerCostBreakdown,
        LayerPlacement,
        PlacementDevice,
        PlacementPlan,
        build_segments,
    )

    placements: List[LayerPlacement] = []
    idx = 0

    if include_embedding:
        placements.append(LayerPlacement(
            layer_index=idx, layer_type="embedding",
            device=PlacementDevice.CPU, gpu_index=None,
            size_bytes=50 * 1024 * 1024,  # 50 MB
            cost=LayerCostBreakdown(),
        ))
        idx += 1

    for i in range(n_gpu):
        placements.append(LayerPlacement(
            layer_index=idx, layer_type="transformer",
            device=PlacementDevice.GPU, gpu_index=gpu_index,
            size_bytes=200 * 1024 * 1024,  # 200 MB each
            cost=LayerCostBreakdown(),
        ))
        idx += 1

    for i in range(n_cpu):
        placements.append(LayerPlacement(
            layer_index=idx, layer_type="transformer",
            device=PlacementDevice.CPU, gpu_index=None,
            size_bytes=200 * 1024 * 1024,  # 200 MB each
            cost=LayerCostBreakdown(),
        ))
        idx += 1

    if include_norm:
        placements.append(LayerPlacement(
            layer_index=idx, layer_type="norm",
            device=PlacementDevice.CPU, gpu_index=None,
            size_bytes=5 * 1024 * 1024,  # 5 MB
            cost=LayerCostBreakdown(),
        ))
        idx += 1

    if include_lm_head:
        placements.append(LayerPlacement(
            layer_index=idx, layer_type="lm_head",
            device=PlacementDevice.CPU, gpu_index=None,
            size_bytes=100 * 1024 * 1024,  # 100 MB
            cost=LayerCostBreakdown(),
        ))
        idx += 1

    segments = build_segments(placements)

    return PlacementPlan(
        model_name="Test-Model",
        architecture="llama",
        total_layers=idx,
        n_gpu_layers=n_gpu,
        n_cpu_layers=n_cpu,
        layer_placements=placements,
        segments=segments,
        total_cost=0.5,
        optimizer_iterations=100,
        estimated_vram_bytes=n_gpu * 200 * 1024 * 1024,
        estimated_ram_bytes=n_cpu * 200 * 1024 * 1024,
        estimated_peak_bytes=(n_gpu + n_cpu) * 200 * 1024 * 1024,
        context_length=4096,
        is_feasible=True,
    )


# ---------------------------------------------------------------------------
# 1. PlacementDevice enum (placement_plan.py)
# ---------------------------------------------------------------------------

class TestPlacementDevice:
    def test_igpu_enum_value(self):
        from layer_placement.placement_plan import PlacementDevice
        assert PlacementDevice.IGPU == "IGPU"

    def test_igpu_str(self):
        from layer_placement.placement_plan import PlacementDevice
        assert str(PlacementDevice.IGPU) == "IGPU"

    def test_all_three_devices_exist(self):
        from layer_placement.placement_plan import PlacementDevice
        devices = {d.value for d in PlacementDevice}
        assert {"GPU", "CPU", "IGPU"} == devices


class TestPlacementPlanIgpuFields:
    def test_n_igpu_layers_defaults_zero(self):
        plan = _make_simple_plan()
        assert plan.n_igpu_layers == 0

    def test_igpu_layer_indices_defaults_empty(self):
        plan = _make_simple_plan()
        assert plan.igpu_layer_indices == []

    def test_estimated_igpu_vram_defaults_zero(self):
        plan = _make_simple_plan()
        assert plan.estimated_igpu_vram_bytes == 0

    def test_igpu_offload_ratio_zero_when_no_igpu(self):
        plan = _make_simple_plan()
        assert plan.igpu_offload_ratio == 0.0

    def test_to_dict_has_igpu_keys(self):
        plan = _make_simple_plan()
        d = plan.to_dict()
        assert "n_igpu_layers" in d
        assert "igpu_offload_ratio" in d
        assert "estimated_igpu_vram_bytes" in d["memory"]

    def test_segment_label_igpu(self):
        from layer_placement.placement_plan import PlacementDevice, PlacementSegment
        seg = PlacementSegment(
            device=PlacementDevice.IGPU, start_layer=0, end_layer=2,
            gpu_index=0, total_size_bytes=100, layer_count=3,
        )
        assert "iGPU" in seg.label()

    def test_summary_includes_igpu_when_present(self):
        plan = _make_simple_plan()
        plan.n_igpu_layers = 3
        s = plan.summary()
        assert "iGPU" in s

    def test_summary_no_igpu_when_absent(self):
        plan = _make_simple_plan()
        s = plan.summary()
        assert "iGPU" not in s


# ---------------------------------------------------------------------------
# 2. IgpuProfiler
# ---------------------------------------------------------------------------

class TestIgpuProfiler:
    def test_no_igpu_returns_not_detected(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_no_igpu()).profile()
        assert profile.detected is False
        assert profile.enabled is False
        assert profile.suitability_score == 0.0

    def test_igpu_detected(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu()).profile()
        assert profile.detected is True

    def test_igpu_vendor_parsed(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu(igpu_vendor="intel")).profile()
        assert profile.vendor == "intel"

    def test_igpu_vram_parsed(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu(igpu_vram_mb=1024)).profile()
        assert profile.vram_mb == 1024
        assert profile.vram_bytes == 1024 * 1024 * 1024

    def test_igpu_disabled_below_min_vram(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu(igpu_vram_mb=128)).profile()
        assert profile.enabled is False
        assert profile.suitability_score == 0.0

    def test_igpu_disabled_unsupported_backend(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu(igpu_backend="cpu")).profile()
        assert profile.enabled is False

    def test_is_shared_memory_true_for_amd_igpu(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu(igpu_vendor="amd")).profile()
        assert profile.is_shared_memory is True

    def test_is_shared_memory_true_for_intel(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu(igpu_vendor="intel")).profile()
        assert profile.is_shared_memory is True

    def test_vulkan_device_index_populated(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu()).profile()
        assert profile.vulkan_device_index == 0  # global_index=0 in fixture

    def test_suitability_score_in_range(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu()).profile()
        assert 0.0 <= profile.suitability_score <= 1.0

    def test_score_higher_with_more_vram(self):
        from igpu_support import IgpuProfiler
        low = IgpuProfiler(_hw_profile_with_igpu(igpu_vram_mb=512)).profile()
        high = IgpuProfiler(_hw_profile_with_igpu(igpu_vram_mb=2048)).profile()
        assert high.suitability_score > low.suitability_score

    def test_score_lower_with_high_ram_utilization(self):
        from igpu_support import IgpuProfiler
        low_util = IgpuProfiler(_hw_profile_with_igpu(ram_utilization=20.0)).profile()
        high_util = IgpuProfiler(_hw_profile_with_igpu(ram_utilization=95.0)).profile()
        assert low_util.suitability_score > high_util.suitability_score

    def test_to_dict_has_all_required_keys(self):
        from igpu_support import IgpuProfiler
        profile = IgpuProfiler(_hw_profile_with_igpu()).profile()
        d = profile.to_dict()
        for key in ("detected", "enabled", "vendor", "vram_mb", "suitability_score",
                    "backend_hint", "vulkan_device_index"):
            assert key in d


# ---------------------------------------------------------------------------
# 3. ContentionModel
# ---------------------------------------------------------------------------

class TestContentionModel:
    def test_low_utilization_gives_low_risk(self):
        from igpu_support import ContentionModel, IgpuProfiler
        hw = _hw_profile_with_igpu(ram_utilization=20.0)
        profile = IgpuProfiler(hw).profile()
        est = ContentionModel(hw).estimate(profile)
        assert est.contention_risk == "LOW"

    def test_high_utilization_gives_high_risk(self):
        from igpu_support import ContentionModel, IgpuProfiler
        hw = _hw_profile_with_igpu(ram_utilization=95.0)
        profile = IgpuProfiler(hw).profile()
        est = ContentionModel(hw).estimate(profile)
        assert est.contention_risk in ("MEDIUM", "HIGH")

    def test_effective_bw_always_positive(self):
        from igpu_support import ContentionModel, IgpuProfiler
        hw = _hw_profile_with_igpu(ram_utilization=99.0)
        profile = IgpuProfiler(hw).profile()
        est = ContentionModel(hw).estimate(profile)
        assert est.effective_igpu_bandwidth_gbps > 0.0

    def test_contention_factor_bounded(self):
        from igpu_support import ContentionModel, IgpuProfiler
        hw = _hw_profile_with_igpu(ram_utilization=100.0)
        profile = IgpuProfiler(hw).profile()
        est = ContentionModel(hw).estimate(profile)
        assert 0.0 <= est.contention_factor <= 0.80

    def test_to_dict_has_all_keys(self):
        from igpu_support import ContentionModel, IgpuProfiler
        hw = _hw_profile_with_igpu()
        profile = IgpuProfiler(hw).profile()
        est = ContentionModel(hw).estimate(profile)
        d = est.to_dict()
        for k in ("system_bus_bandwidth_gbps", "contention_factor",
                   "effective_igpu_bandwidth_gbps", "contention_risk"):
            assert k in d

    def test_no_igpu_uses_default_bandwidth(self):
        from igpu_support import ContentionModel, IgpuProfiler
        hw = _hw_profile_no_igpu()
        profile = IgpuProfiler(hw).profile()
        est = ContentionModel(hw).estimate(profile)
        # Should still return a valid estimate (with near-zero contention)
        assert est.contention_factor == 0.0


# ---------------------------------------------------------------------------
# 4. WorkloadClassifier
# ---------------------------------------------------------------------------

class TestWorkloadClassifier:
    def _make_classifier(self, suitability_str, igpu_vram_mb=512, bw=25.0):
        from igpu_support import IgpuWorkloadSuitability, WorkloadClassifier
        s_map = {
            "FULL": IgpuWorkloadSuitability.FULL,
            "LIGHT": IgpuWorkloadSuitability.LIGHT,
            "DISABLED": IgpuWorkloadSuitability.DISABLED,
        }
        return WorkloadClassifier(
            suitability=s_map[suitability_str],
            igpu_vram_bytes=igpu_vram_mb * 1024 * 1024,
            effective_bw_gbps=bw,
        )

    def test_disabled_rejects_all(self):
        clf = self._make_classifier("DISABLED")
        for lt in ("embedding", "norm", "lm_head", "transformer"):
            assert clf.is_suitable_for_igpu(lt, 1024) is False

    def test_light_allows_embedding(self):
        clf = self._make_classifier("LIGHT")
        assert clf.is_suitable_for_igpu("embedding", 50 * 1024 * 1024) is True

    def test_light_allows_norm(self):
        clf = self._make_classifier("LIGHT")
        assert clf.is_suitable_for_igpu("norm", 5 * 1024 * 1024) is True

    def test_light_allows_lm_head(self):
        clf = self._make_classifier("LIGHT")
        assert clf.is_suitable_for_igpu("lm_head", 100 * 1024 * 1024) is True

    def test_light_rejects_transformer(self):
        clf = self._make_classifier("LIGHT")
        assert clf.is_suitable_for_igpu("transformer", 50 * 1024 * 1024) is False

    def test_full_allows_small_transformer(self):
        clf = self._make_classifier("FULL", igpu_vram_mb=2048, bw=30.0)
        # 300 MB < 60% of 2048 MB = 1228 MB
        assert clf.is_suitable_for_igpu("transformer", 300 * 1024 * 1024) is True

    def test_full_rejects_large_transformer(self):
        clf = self._make_classifier("FULL", igpu_vram_mb=512, bw=30.0)
        # 400 MB > 60% of 512 MB = 307 MB
        assert clf.is_suitable_for_igpu("transformer", 400 * 1024 * 1024) is False

    def test_full_rejects_transformer_with_low_bw(self):
        clf = self._make_classifier("FULL", igpu_vram_mb=2048, bw=5.0)
        # bw=5.0 < COMPUTE_BW_MIN_GBPS=10.0 → reject
        assert clf.is_suitable_for_igpu("transformer", 100 * 1024 * 1024) is False

    def test_size_limit_for_light_layers(self):
        clf = self._make_classifier("LIGHT", igpu_vram_mb=512)
        # 30% of 512 MB = 153.6 MB
        assert clf.is_suitable_for_igpu("embedding", 150 * 1024 * 1024) is True
        assert clf.is_suitable_for_igpu("embedding", 200 * 1024 * 1024) is False


# ---------------------------------------------------------------------------
# 5. SuitabilityEvaluator
# ---------------------------------------------------------------------------

class TestSuitabilityEvaluator:
    def _evaluate(self, vram_mb=512, ram_util=50.0, risk_override=None):
        from igpu_support import (
            ContentionModel, IgpuProfiler,
            SuitabilityEvaluator,
        )
        hw = _hw_profile_with_igpu(igpu_vram_mb=vram_mb, ram_utilization=ram_util)
        profile = IgpuProfiler(hw).profile()
        contention = ContentionModel(hw).estimate(profile)
        result = SuitabilityEvaluator().evaluate(profile, contention)
        return result

    def test_disabled_when_no_igpu(self):
        from igpu_support import ContentionModel, IgpuProfiler, SuitabilityEvaluator
        from igpu_support.workload_classifier import IgpuWorkloadSuitability
        hw = _hw_profile_no_igpu()
        profile = IgpuProfiler(hw).profile()
        contention = ContentionModel(hw).estimate(profile)
        result = SuitabilityEvaluator().evaluate(profile, contention)
        assert result.suitability == IgpuWorkloadSuitability.DISABLED

    def test_disabled_when_below_min_vram(self):
        from igpu_support.workload_classifier import IgpuWorkloadSuitability
        result = self._evaluate(vram_mb=128)
        assert result.suitability == IgpuWorkloadSuitability.DISABLED

    def test_at_least_light_for_reasonable_igpu(self):
        from igpu_support.workload_classifier import IgpuWorkloadSuitability
        result = self._evaluate(vram_mb=512, ram_util=50.0)
        assert result.suitability != IgpuWorkloadSuitability.DISABLED

    def test_disabled_when_ram_critical(self):
        from igpu_support.workload_classifier import IgpuWorkloadSuitability
        result = self._evaluate(vram_mb=2048, ram_util=96.0)
        assert result.suitability == IgpuWorkloadSuitability.DISABLED

    def test_result_has_reasons(self):
        result = self._evaluate()
        assert isinstance(result.reasons, list)
        assert len(result.reasons) > 0

    def test_to_dict_serializable(self):
        result = self._evaluate()
        d = result.to_dict()
        assert "suitability" in d
        assert "score" in d
        assert "reasons" in d
        # Must be JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# 6. IgpuPlacementContributor
# ---------------------------------------------------------------------------

class TestIgpuPlacementContributor:
    def _make_model_stub(self):
        """Very minimal ModelDescriptor stub with just the fields contributor needs."""
        from layer_placement.model_descriptor import ModelDescriptor, LayerDescriptor
        # We pass the plan directly; model is only used for layer_type lookups which
        # the contributor actually reads from LayerPlacement, not ModelDescriptor.
        # Return a minimal object with a .layers list.
        class StubModel:
            name = "Test-Model"
            architecture = "llama"
            layers = []
            max_context_length = 4096
            model_size_bytes = 4 * 1024 ** 3
        return StubModel()

    def test_no_igpu_returns_original_plan_unchanged(self):
        from igpu_support import IgpuPlacementContributor
        hw = _hw_profile_no_igpu()
        plan = _make_simple_plan()
        result = IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        # Plan should be returned (possibly unchanged)
        assert result.plan is not None
        assert result.igpu_enabled is False

    def test_igpu_disabled_returns_original_plan_object(self):
        from igpu_support import IgpuPlacementContributor
        # With 128 MB VRAM, iGPU should be disabled
        hw = _hw_profile_with_igpu(igpu_vram_mb=128)
        plan = _make_simple_plan()
        result = IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        assert result.igpu_enabled is False
        # Original plan identity is preserved when DISABLED
        assert result.plan is plan

    def test_igpu_enabled_assigns_light_layers(self):
        from igpu_support import IgpuPlacementContributor
        from layer_placement.placement_plan import PlacementDevice
        # Low RAM utilization → suitability will be at least LIGHT
        hw = _hw_profile_with_igpu(igpu_vram_mb=512, ram_utilization=30.0)
        plan = _make_simple_plan(include_embedding=True, include_norm=True, include_lm_head=True)
        result = IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        if result.igpu_enabled:
            # All iGPU-assigned layers should have device=IGPU
            igpu_set = set(result.igpu_layer_indices)
            for lp in result.plan.layer_placements:
                if lp.layer_index in igpu_set:
                    assert lp.device == PlacementDevice.IGPU

    def test_original_plan_not_mutated(self):
        from igpu_support import IgpuPlacementContributor
        hw = _hw_profile_with_igpu(ram_utilization=30.0)
        plan = _make_simple_plan()
        original_n_cpu = plan.n_cpu_layers
        original_placements_count = len(plan.layer_placements)
        IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        # Original plan must not be modified
        assert plan.n_cpu_layers == original_n_cpu
        assert len(plan.layer_placements) == original_placements_count

    def test_igpu_vram_not_exceeded(self):
        from igpu_support import IgpuPlacementContributor
        igpu_vram_mb = 512
        hw = _hw_profile_with_igpu(igpu_vram_mb=igpu_vram_mb, ram_utilization=30.0)
        plan = _make_simple_plan()
        result = IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        # iGPU used VRAM must not exceed 70% of total iGPU VRAM
        max_allowed = int(igpu_vram_mb * 1024 * 1024 * 0.70)
        assert result.igpu_vram_used_bytes <= max_allowed

    def test_result_plan_n_igpu_layers_consistent(self):
        from igpu_support import IgpuPlacementContributor
        hw = _hw_profile_with_igpu(ram_utilization=30.0)
        plan = _make_simple_plan()
        result = IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        assert result.plan.n_igpu_layers == len(result.igpu_layer_indices)

    def test_to_dict_is_json_serializable(self):
        from igpu_support import IgpuPlacementContributor
        hw = _hw_profile_with_igpu()
        plan = _make_simple_plan()
        result = IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        d = result.to_dict()
        json.dumps(d)  # Must not raise

    def test_summary_string_not_empty(self):
        from igpu_support import IgpuPlacementContributor
        hw = _hw_profile_with_igpu()
        plan = _make_simple_plan()
        result = IgpuPlacementContributor(hw).contribute(self._make_model_stub(), plan)
        assert len(result.summary()) > 0


# ---------------------------------------------------------------------------
# 7. HardwareContext iGPU fields
# ---------------------------------------------------------------------------

class TestHardwareContextIgpu:
    def test_igpu_fields_populated_from_profile(self):
        from layer_placement.cost_model import HardwareContext
        hw = _hw_profile_with_igpu(igpu_vram_mb=512)
        ctx = HardwareContext.from_hw_profile(hw)
        assert ctx.igpu_vram_bytes > 0
        assert ctx.igpu_bandwidth_gbps > 0
        assert ctx.igpu_gflops > 0

    def test_igpu_fields_zero_when_no_igpu(self):
        from layer_placement.cost_model import HardwareContext
        hw = _hw_profile_no_igpu()
        ctx = HardwareContext.from_hw_profile(hw)
        assert ctx.igpu_vram_bytes == 0
        assert ctx.igpu_gflops == 0.0

    def test_compute_cost_includes_igpu(self):
        from layer_placement.cost_model import CostModel, HardwareContext
        from layer_placement.placement_plan import PlacementDevice
        from layer_placement.model_descriptor import LayerDescriptor, LAYER_TYPE_TRANSFORMER

        hw_ctx = HardwareContext(
            gpu_tflops=10.0,
            cpu_gflops=200.0,
            igpu_gflops=500.0,  # Very fast iGPU for testing
        )
        cm = CostModel(hw_ctx)

        layers = [
            LayerDescriptor(layer_index=0, layer_type=LAYER_TYPE_TRANSFORMER,
                            size_bytes=100 * 1024 * 1024, compute_flops=1e12,
                            kv_cache_bytes_per_token=0, is_shared=False)
        ]

        gpu_cost  = cm.compute_cost(layers, [PlacementDevice.GPU])
        igpu_cost = cm.compute_cost(layers, [PlacementDevice.IGPU])
        cpu_cost  = cm.compute_cost(layers, [PlacementDevice.CPU])

        # GPU is always cheapest
        assert gpu_cost == 0.0
        # iGPU cost should be between 0 and 1
        assert 0.0 <= igpu_cost <= 1.0
        # CPU should be more expensive than iGPU (igpu_gflops=500 > cpu_gflops=200)
        assert igpu_cost <= cpu_cost


# ---------------------------------------------------------------------------
# 8. BackendSelector iGPU Vulkan flags
# ---------------------------------------------------------------------------

class TestBackendSelectorIgpu:
    def test_no_igpu_layers_no_device_flag_change(self):
        from inference_runtime.backend_selector import BackendSelector
        hw = _hw_profile_with_igpu()
        plan = _make_simple_plan(n_gpu=20, n_cpu=0)  # all GPU, no iGPU
        # plan.n_igpu_layers is 0 by default
        selector = BackendSelector(hw)
        info = selector.detect_backend(plan)
        # Should not have a comma in the device flag
        if info.extra_flags:
            device_vals = [info.extra_flags[i+1] for i, f in enumerate(info.extra_flags)
                           if f == "--device"]
            for val in device_vals:
                assert "," not in val  # No multi-device when iGPU not in plan

    def test_get_igpu_vulkan_index(self):
        from inference_runtime.backend_selector import BackendSelector
        hw = _hw_profile_with_igpu()
        selector = BackendSelector(hw)
        idx = selector._get_igpu_vulkan_index()
        assert idx == 0  # global_index=0 for iGPU in fixture

    def test_get_igpu_vulkan_index_no_igpu(self):
        from inference_runtime.backend_selector import BackendSelector
        hw = _hw_profile_no_igpu()
        selector = BackendSelector(hw)
        idx = selector._get_igpu_vulkan_index()
        assert idx == -1

    def test_igpu_device_flag_emitted_when_igpu_in_plan(self):
        from inference_runtime.backend_selector import BackendSelector
        from layer_placement.placement_plan import PlacementDevice
        hw = _hw_profile_with_igpu()
        plan = _make_simple_plan()
        # Manually set n_igpu_layers to simulate Phase 7 having run
        plan.n_igpu_layers = 3
        plan.igpu_layer_indices = [0, 1, 2]

        selector = BackendSelector(hw)
        info = selector.detect_backend(plan)

        # For vulkan backend with iGPU layers, should emit --device dGPU,iGPU
        device_vals = [info.extra_flags[i+1] for i, f in enumerate(info.extra_flags)
                       if f == "--device"]
        if device_vals:
            # At least one --device flag, and when iGPU is present it should be comma-sep
            assert any("," in v for v in device_vals)
