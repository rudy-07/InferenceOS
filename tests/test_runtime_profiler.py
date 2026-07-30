"""
test_runtime_profiler.py
-------------------------
Comprehensive test suite for Phase 9: Runtime Profiler.

Test classes:
  TestTelemetryCollector         — Token latencies, bandwidth sampling, KV cache & layer breakdown
  TestFlameGraphGenerator        — Node tree creation, JSON, folded stack text, interactive HTML
  TestTimelineGenerator          — Event generation, tracks, Gantt HTML & JSON
  TestOptimizationAdvisor        — Bottleneck detection (GPU idle, PCIe, VRAM headroom, KV growth)
  TestPerformanceReportGenerator — Markdown & JSON report formatting
  TestProfilerEngine             — Context manager lifecycle, ProfilerResult export_all
  TestRuntimeEngineProfiler      — RuntimeEngine.profileRun integration
  TestRuntimeConfigPhase9        — RuntimeConfig Phase 9 fields
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from runtime_profiler import (
    FlameGraphGenerator,
    FlameNode,
    LayerTimingRecord,
    OptimizationAdvisor,
    OptimizationRecommendation,
    PerformanceReportGenerator,
    ProfilerEngine,
    ProfilerResult,
    ProfilerTelemetry,
    Severity,
    TelemetryCollector,
    TimelineEvent,
    TimelineGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hw(ram_gb=16, vram_gb=8, cores=8):
    return {
        "cpu": {"physical_cores": cores, "logical_cores": cores * 2},
        "gpus": [{"vendor": "NVIDIA", "vram_total_mb": vram_gb * 1024, "vram_bandwidth_gbps": 300.0}],
        "memory": {"total_bytes": ram_gb * 1024**3, "ram_bandwidth_gbps": 50.0},
        "interconnects": [{"type": "PCIE", "bandwidth": 16.0}],
    }


def _plan_mock(n_gpu=5, n_cpu=5):
    from layer_placement.placement_plan import (
        LayerCostBreakdown, LayerPlacement, PlacementDevice,
        PlacementPlan, PlacementSegment,
    )
    lc = LayerCostBreakdown()
    gpu_layers = [LayerPlacement(i, "transformer", PlacementDevice.GPU, 0, 50*1024*1024, lc) for i in range(n_gpu)]
    cpu_layers = [LayerPlacement(n_gpu + i, "transformer", PlacementDevice.CPU, None, 50*1024*1024, lc) for i in range(n_cpu)]
    segs = [
        PlacementSegment(PlacementDevice.GPU, 0, n_gpu - 1, 0, n_gpu * 50 * 1024 * 1024, n_gpu),
        PlacementSegment(PlacementDevice.CPU, n_gpu, n_gpu + n_cpu - 1, None, n_cpu * 50 * 1024 * 1024, n_cpu),
    ]
    return PlacementPlan(
        model_name="test-7b",
        architecture="llama",
        total_layers=n_gpu + n_cpu,
        n_gpu_layers=n_gpu,
        n_cpu_layers=n_cpu,
        context_length=4096,
        layer_placements=gpu_layers + cpu_layers,
        segments=segs,
        estimated_vram_bytes=n_gpu * 50 * 1024 * 1024,
        estimated_ram_bytes=n_cpu * 50 * 1024 * 1024,
    )


# ===========================================================================
# TestTelemetryCollector
# ===========================================================================

class TestTelemetryCollector(unittest.TestCase):

    def test_start_stop_lifecycle(self):
        collector = TelemetryCollector(sample_interval_ms=10.0, hw_profile=_hw())
        collector.start()
        time.sleep(0.05)
        collector.record_token()
        time.sleep(0.05)
        collector.record_token()
        collector.stop()

        telemetry = collector.finalize(
            stderr_info={"prompt_eval_tokens": 10, "eval_tokens": 5, "eval_ms": 100.0},
            model_name="test-model",
            backend="cuda",
        )
        self.assertEqual(telemetry.model_name, "test-model")
        self.assertEqual(telemetry.backend, "cuda")
        self.assertGreater(telemetry.total_wall_ms, 0)
        self.assertGreater(len(telemetry.inter_token_latencies_ms), 0)

    def test_latency_statistics_calculation(self):
        collector = TelemetryCollector()
        itls = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50, p90, p95, p99, stddev, jitter = collector._calc_latency_stats(itls)

        self.assertAlmostEqual(p50, 30.0)
        self.assertGreater(p95, p50)
        self.assertGreater(stddev, 0)
        self.assertGreater(jitter, 0)

    def test_empty_latency_stats(self):
        collector = TelemetryCollector()
        res = collector._calc_latency_stats([])
        self.assertEqual(res, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_layer_timings_computation(self):
        plan = _plan_mock(n_gpu=5, n_cpu=5)
        collector = TelemetryCollector(plan=plan)
        records = collector._compute_layer_timings(total_gen_ms=100.0)

        self.assertEqual(len(records), 10)
        self.assertIn("GPU", records[0].device)
        self.assertIn("CPU", records[5].device)
        # Boundary layer (layer 5) should have positive transfer_ms
        self.assertGreater(records[5].transfer_ms, 0)

    def test_telemetry_to_dict_serializable(self):
        t = ProfilerTelemetry(
            prompt_tokens=128,
            generation_tokens=32,
            prompt_tps=120.0,
            generation_tps=25.0,
            ttft_ms=45.0,
            p50_itl_ms=40.0,
            avg_cpu_util_pct=45.0,
            avg_gpu_util_pct=85.0,
        )
        d = t.to_dict()
        json.dumps(d)  # must not raise
        self.assertEqual(d["throughput"]["prompt_tokens"], 128)
        self.assertEqual(d["latency"]["ttft_ms"], 45.0)


# ===========================================================================
# TestFlameGraphGenerator
# ===========================================================================

class TestFlameGraphGenerator(unittest.TestCase):

    def setUp(self):
        self.telemetry = ProfilerTelemetry(
            model_name="llama-7b",
            backend="cuda",
            prompt_eval_ms=50.0,
            generation_eval_ms=200.0,
            total_wall_ms=260.0,
            n_gpu_layers=20,
            n_cpu_layers=10,
            total_layers=30,
            layer_timings=[
                LayerTimingRecord(0, "GPU", 10.0, 0.0, 50000),
                LayerTimingRecord(20, "CPU", 15.0, 1.2, 50000),
            ]
        )
        self.fg = FlameGraphGenerator(self.telemetry)

    def test_flame_tree_structure(self):
        root = self.fg.root
        self.assertEqual(root.category, "root")
        self.assertGreater(len(root.children), 0)

        # Check for expected stage names
        stage_names = [c.name for c in root.children]
        self.assertTrue(any("Prepare" in name for name in stage_names))
        self.assertTrue(any("Transfer" in name for name in stage_names))
        self.assertTrue(any("Compute" in name for name in stage_names))
        self.assertTrue(any("Sampling" in name for name in stage_names))

    def test_flame_graph_to_json(self):
        js_str = self.fg.to_json()
        d = json.loads(js_str)
        self.assertIn("children", d)
        self.assertEqual(d["category"], "root")

    def test_flame_graph_to_folded_text(self) -> None:
        text = self.fg.to_folded_text()
        self.assertIn("InferenceSession", text)
        self.assertIn("Stage:", text)

    def test_flame_graph_to_html(self):
        html = self.fg.to_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("InferenceOS Flame Graph", html)
        self.assertIn("treeData", html)


# ===========================================================================
# TestTimelineGenerator
# ===========================================================================

class TestTimelineGenerator(unittest.TestCase):

    def setUp(self):
        self.telemetry = ProfilerTelemetry(
            model_name="llama-7b",
            prompt_eval_ms=40.0,
            generation_eval_ms=160.0,
            generation_tokens=4,
            inter_token_latencies_ms=[40.0, 42.0, 38.0, 40.0],
            n_gpu_layers=20,
            gpu_idle_pct=25.0,
        )
        self.tg = TimelineGenerator(self.telemetry)

    def test_events_generation(self):
        events = self.tg.events
        self.assertGreater(len(events), 0)

        tracks = {e.track for e in events}
        self.assertIn("CPU Prepare", tracks)
        self.assertIn("Token Arrival", tracks)
        self.assertIn("GPU Execution", tracks)

    def test_timeline_to_json(self):
        js = self.tg.to_json()
        arr = json.loads(js)
        self.assertIsInstance(arr, list)
        self.assertGreater(len(arr), 0)

    def test_timeline_to_html(self):
        html = self.tg.to_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Execution Timeline", html)
        self.assertIn("events", html)


# ===========================================================================
# TestOptimizationAdvisor
# ===========================================================================

class TestOptimizationAdvisor(unittest.TestCase):

    def test_gpu_idle_recommendation(self):
        # Scenario: GPU idle 27%, free VRAM available -> recommend moving layer to GPU
        telemetry = ProfilerTelemetry(
            n_gpu_layers=40,
            n_cpu_layers=20,
            gpu_idle_pct=27.0,
            generation_tps=20.0,
        )
        advisor = OptimizationAdvisor(telemetry, vram_free_mb=2048.0)
        recs = advisor.analyze()

        self.assertGreater(len(recs), 0)
        layer_rec = next((r for r in recs if "Layer 40" in r.title), None)
        self.assertIsNotNone(layer_rec)
        self.assertIn("GPU idle 27%", layer_rec.bottleneck)
        self.assertGreater(layer_rec.estimated_tps_delta_pct, 0)
        self.assertEqual(layer_rec.severity, Severity.CRITICAL)

    def test_unused_vram_recommendation(self):
        # Scenario: 3 GB free VRAM, 10 CPU layers -> recommend offloading
        telemetry = ProfilerTelemetry(
            n_gpu_layers=20,
            n_cpu_layers=10,
            gpu_idle_pct=5.0,  # low idle
        )
        advisor = OptimizationAdvisor(telemetry, vram_free_mb=3072.0)
        recs = advisor.analyze()

        self.assertTrue(any("Unused VRAM" in r.title for r in recs))

    def test_kv_cache_growth_recommendation(self):
        # Scenario: 95% context window usage -> recommend KV quantization
        telemetry = ProfilerTelemetry(
            context_length=4096,
            context_utilization_pct=95.0,
        )
        advisor = OptimizationAdvisor(telemetry)
        recs = advisor.analyze()

        self.assertTrue(any("KV Cache Quantization" in r.title for r in recs))

    def test_recommendation_to_dict(self):
        rec = OptimizationRecommendation(
            title="Move Layer 61 to GPU",
            bottleneck="GPU idle 27%",
            action="Move Layer 61 to GPU",
            estimated_improvement="+9.2% tok/s",
            estimated_tps_delta_pct=9.2,
            severity=Severity.CRITICAL,
        )
        d = rec.to_dict()
        self.assertEqual(d["title"], "Move Layer 61 to GPU")
        self.assertEqual(d["severity"], "CRITICAL")


# ===========================================================================
# TestPerformanceReportGenerator
# ===========================================================================

class TestPerformanceReportGenerator(unittest.TestCase):

    def setUp(self):
        self.telemetry = ProfilerTelemetry(
            model_name="mistral-7b",
            backend="vulkan",
            total_wall_ms=500.0,
            generation_tps=32.5,
            ttft_ms=60.0,
            p50_itl_ms=30.0,
            gpu_idle_pct=12.0,
        )
        self.rec = OptimizationRecommendation(
            title="Enable Async Prefetch",
            bottleneck="Transfer delay",
            action="enable_async_scheduler=True",
            estimated_improvement="+8% tok/s",
            estimated_tps_delta_pct=8.0,
        )
        self.report = PerformanceReportGenerator(self.telemetry, [self.rec])

    def test_to_markdown(self):
        md = self.report.to_markdown()
        self.assertIn("# 🚀 InferenceOS Phase 9", md)
        self.assertIn("mistral-7b", md)
        self.assertIn("32.50 tokens/sec", md)
        self.assertIn("Enable Async Prefetch", md)

    def test_to_json(self):
        js = self.report.to_json()
        d = json.loads(js)
        self.assertEqual(d["summary"]["model_name"], "mistral-7b")
        self.assertEqual(len(d["recommendations"]), 1)


# ===========================================================================
# TestProfilerEngine
# ===========================================================================

class TestProfilerEngine(unittest.TestCase):

    def test_profile_session_context_manager(self):
        engine = ProfilerEngine(hw_profile=_hw())
        with engine.profile_session(model_name="test-model", backend="cuda") as collector:
            collector.record_token()
            collector.record_token()

        res = engine.get_last_result()
        self.assertIsNotNone(res)
        self.assertIsInstance(res, ProfilerResult)
        self.assertIn("tok/s", res.summary())

    def test_export_all_artifacts(self):
        engine = ProfilerEngine(hw_profile=_hw())
        with engine.profile_session(model_name="test-model") as collector:
            collector.record_token()

        res = engine.get_last_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_files = res.export_all(tmpdir, prefix="test_run")
            self.assertTrue(out_files["flame_graph_html"].exists())
            self.assertTrue(out_files["timeline_html"].exists())
            self.assertTrue(out_files["report_markdown"].exists())
            self.assertTrue(out_files["report_json"].exists())


# ===========================================================================
# TestRuntimeEngineProfiler
# ===========================================================================

class TestRuntimeEngineProfiler(unittest.TestCase):

    def test_runtime_engine_profile_run(self):
        from inference_runtime import RuntimeEngine
        plan = _plan_mock(n_gpu=5, n_cpu=5)
        engine = RuntimeEngine(hw_profile=_hw())

        def _mock_execute_plan(*args, **kwargs):
            res = MagicMock()
            res.stats.eval_tps = 25.0
            res.backend = "cpu"
            return res

        engine.executePlan = _mock_execute_plan

        res, prof_res = engine.profileRun(plan, Path("model.gguf"), "Hello prompt")
        self.assertIsNotNone(res)
        self.assertIsNotNone(prof_res)
        self.assertIsInstance(prof_res, ProfilerResult)

    def test_runtime_config_phase9_fields(self):
        from inference_runtime import RuntimeConfig
        cfg = RuntimeConfig(enable_profiler=True, profiler_sample_interval_ms=50.0)
        self.assertTrue(cfg.enable_profiler)
        self.assertEqual(cfg.profiler_sample_interval_ms, 50.0)

        d = cfg.to_dict()
        self.assertIn("enable_profiler", d)
        self.assertIn("profiler_sample_interval_ms", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
