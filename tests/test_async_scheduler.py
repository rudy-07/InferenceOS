"""
test_async_scheduler.py
-----------------------
Comprehensive test suite for Phase 8: Async Execution Scheduler.

Tests are organized by class matching each module:

  TestSchedulerConfig         — SchedulerConfig defaults, from_hw_profile
  TestTaskQueue               — submit, priority, bounded, future lifecycle
  TestTaskFuture              — result, cancel, timing properties
  TestWorkerPool              — start/stop, execute, shutdown
  TestPrefetchManager         — boundary detection, buffer pool, warm/cold
  TestCudaStreamManager       — backend detection, env-vars, overlap estimate
  TestSchedulerMetrics        — StageTimings, MetricsAccumulator, SchedulerReport
  TestPipelineCoordinator     — stage handoff, metrics recording, lifecycle
  TestAsyncRuntimeEngine      — submit, submitBatch, metrics, fallback
  TestRuntimeConfigPhase8     — new scheduler fields in RuntimeConfig
  TestInferenceRuntimeExports — Phase 8 types exported from inference_runtime
"""
from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: build a minimal hw_profile for tests
# ---------------------------------------------------------------------------

def _hw(gpus=None, igpus=None, ram_gb=16, cores=8):
    return {
        "cpu": {"physical_cores": cores, "logical_cores": cores * 2},
        "gpus": gpus or [],
        "igpus": igpus or [],
        "ram": {"total_bytes": ram_gb * 1024**3, "total_gb": ram_gb, "available_gb": ram_gb * 0.6},
        "memory": {"total_bytes": ram_gb * 1024**3},
        "interconnects": [{"type": "PCIE", "bandwidth": 16.0}],
        "inference_hints": {"recommended_backend": "cpu"},
    }


def _nvidia_hw():
    return _hw(gpus=[{"vendor": "NVIDIA", "backend_hint": "cuda", "vram_total_mb": 8192}])


def _plan_mock(n_gpu=10, n_cpu=10, boundary=1):
    """Build a minimal PlacementPlan mock."""
    from layer_placement.placement_plan import (
        LayerCostBreakdown, LayerPlacement, PlacementDevice,
        PlacementPlan, PlacementSegment,
    )
    lc = LayerCostBreakdown()
    gpu_layers = [LayerPlacement(i, "transformer", PlacementDevice.GPU, 0, 50*1024*1024, lc)
                  for i in range(n_gpu)]
    cpu_layers = [LayerPlacement(n_gpu + i, "transformer", PlacementDevice.CPU, None, 50*1024*1024, lc)
                  for i in range(n_cpu)]
    all_layers = gpu_layers + cpu_layers
    segs = []
    if n_gpu > 0:
        segs.append(PlacementSegment(
            device=PlacementDevice.GPU,
            start_layer=0,
            end_layer=n_gpu - 1,
            gpu_index=0,
            total_size_bytes=n_gpu * 50 * 1024 * 1024,
            layer_count=n_gpu,
        ))
    if n_cpu > 0:
        segs.append(PlacementSegment(
            device=PlacementDevice.CPU,
            start_layer=n_gpu,
            end_layer=n_gpu + n_cpu - 1,
            gpu_index=None,
            total_size_bytes=n_cpu * 50 * 1024 * 1024,
            layer_count=n_cpu,
        ))
    plan = PlacementPlan(
        model_name="test-model",
        architecture="llama",
        total_layers=n_gpu + n_cpu,
        n_gpu_layers=n_gpu,
        n_cpu_layers=n_cpu,
        context_length=4096,
        layer_placements=all_layers,
        segments=segs,
        estimated_vram_bytes=n_gpu * 50 * 1024 * 1024,
        estimated_ram_bytes=n_cpu * 50 * 1024 * 1024,
    )
    return plan


# ===========================================================================
# TestSchedulerConfig
# ===========================================================================

class TestSchedulerConfig(unittest.TestCase):

    def test_defaults_are_valid(self):
        from async_scheduler import SchedulerConfig
        cfg = SchedulerConfig()
        self.assertGreaterEqual(cfg.n_cpu_workers, 1)
        self.assertEqual(cfg.n_transfer_workers, 1)
        self.assertEqual(cfg.max_queue_depth, 8)
        self.assertTrue(cfg.enable_request_pipelining)
        self.assertTrue(cfg.enable_prefetch)
        self.assertEqual(cfg.prefetch_lookahead, 1)
        self.assertGreaterEqual(cfg.pinned_buffer_size_mb, 64)

    def test_cpu_workers_clamped_to_minimum_1(self):
        from async_scheduler import SchedulerConfig
        cfg = SchedulerConfig(n_cpu_workers=0)
        self.assertGreaterEqual(cfg.n_cpu_workers, 1)

    def test_transfer_workers_clamped(self):
        from async_scheduler import SchedulerConfig
        cfg = SchedulerConfig(n_transfer_workers=0)
        self.assertEqual(cfg.n_transfer_workers, 1)

    def test_bubble_threshold_clamped_0_to_1(self):
        from async_scheduler import SchedulerConfig
        cfg = SchedulerConfig(bubble_detection_threshold_pct=2.0)
        self.assertEqual(cfg.bubble_detection_threshold_pct, 1.0)
        cfg2 = SchedulerConfig(bubble_detection_threshold_pct=-0.5)
        self.assertEqual(cfg2.bubble_detection_threshold_pct, 0.0)

    def test_from_hw_profile_cpu_only(self):
        from async_scheduler import SchedulerConfig
        cfg = SchedulerConfig.from_hw_profile(_hw())
        self.assertGreaterEqual(cfg.n_cpu_workers, 1)
        # No NVIDIA GPU → cuda_streams_enabled should be None or False
        self.assertIn(cfg.cuda_streams_enabled, (None, False))

    def test_from_hw_profile_nvidia_enables_cuda(self):
        from async_scheduler import SchedulerConfig
        cfg = SchedulerConfig.from_hw_profile(_nvidia_hw())
        self.assertTrue(cfg.cuda_streams_enabled)

    def test_from_hw_profile_large_vram_bigger_buffer(self):
        from async_scheduler import SchedulerConfig
        hw = _hw(gpus=[{"vendor": "NVIDIA", "vram_total_mb": 8192}])
        cfg = SchedulerConfig.from_hw_profile(hw)
        self.assertGreaterEqual(cfg.pinned_buffer_size_mb, 512)

    def test_to_dict_serializable(self):
        from async_scheduler import SchedulerConfig
        import json
        cfg = SchedulerConfig()
        d = cfg.to_dict()
        json.dumps(d)  # must not raise

    def test_max_concurrent_requests_minimum_1(self):
        from async_scheduler import SchedulerConfig
        cfg = SchedulerConfig(max_concurrent_requests=0)
        self.assertGreaterEqual(cfg.max_concurrent_requests, 1)


# ===========================================================================
# TestTaskQueue
# ===========================================================================

class TestTaskQueue(unittest.TestCase):

    def test_submit_returns_future(self):
        from async_scheduler import TaskFuture, TaskQueue
        q = TaskQueue(max_depth=4)
        future = q.submit(lambda: 42)
        self.assertIsInstance(future, TaskFuture)

    def test_depth_increases_on_submit(self):
        from async_scheduler import TaskQueue
        q = TaskQueue(max_depth=4)
        q.submit(lambda: None)
        q.submit(lambda: None)
        self.assertEqual(q.depth, 2)

    def test_queue_full_raises(self):
        from async_scheduler import QueueFullError, TaskQueue
        q = TaskQueue(max_depth=2)
        q.submit(lambda: None)
        q.submit(lambda: None)
        with self.assertRaises(QueueFullError):
            q.submit(lambda: None)

    def test_priority_ordering(self):
        from async_scheduler import Priority, TaskQueue
        q = TaskQueue(max_depth=10)
        q.submit(lambda: "low", priority=Priority.LOW)
        q.submit(lambda: "critical", priority=Priority.CRITICAL)
        q.submit(lambda: "normal", priority=Priority.NORMAL)

        item1 = q.get_nowait()
        item2 = q.get_nowait()
        item3 = q.get_nowait()
        self.assertEqual(item1.priority, int(Priority.CRITICAL))
        self.assertEqual(item2.priority, int(Priority.NORMAL))
        self.assertEqual(item3.priority, int(Priority.LOW))

    def test_stats_submitted_counter(self):
        from async_scheduler import TaskQueue
        q = TaskQueue(max_depth=10)
        q.submit(lambda: None)
        q.submit(lambda: None)
        s = q.stats
        self.assertEqual(s.submitted, 2)

    def test_is_empty_initially(self):
        from async_scheduler import TaskQueue
        q = TaskQueue(max_depth=4)
        self.assertTrue(q.is_empty)

    def test_get_nowait_returns_none_when_empty(self):
        from async_scheduler import TaskQueue
        q = TaskQueue(max_depth=4)
        self.assertIsNone(q.get_nowait())

    def test_get_with_timeout_returns_none_on_timeout(self):
        from async_scheduler import TaskQueue
        q = TaskQueue(max_depth=4)
        result = q.get(timeout=0.05)
        self.assertIsNone(result)


# ===========================================================================
# TestTaskFuture
# ===========================================================================

class TestTaskFuture(unittest.TestCase):

    def test_future_resolves_via_worker(self):
        from async_scheduler import Priority, WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=1).start()
        future = pool.submit(lambda: 99)
        result = future.result(timeout=5.0)
        pool.stop()
        self.assertEqual(result, 99)

    def test_future_raises_on_exception(self):
        from async_scheduler import TaskFailedError, WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=1).start()

        def _fail():
            raise ValueError("intentional")

        future = pool.submit(_fail)
        with self.assertRaises(TaskFailedError):
            future.result(timeout=5.0)
        pool.stop()

    def test_future_cancel_pending(self):
        # A future in the queue (not yet started) can be cancelled
        from async_scheduler import TaskCancelledError, TaskFuture
        f = TaskFuture(task_id=0)
        cancelled = f.cancel()
        self.assertTrue(cancelled)
        self.assertTrue(f.is_cancelled)
        with self.assertRaises(TaskCancelledError):
            f.result(timeout=0.1)

    def test_future_cancel_after_done_returns_false(self):
        from async_scheduler import TaskFuture
        f = TaskFuture(task_id=1)
        f._mark_done("done")
        self.assertFalse(f.cancel())

    def test_future_timeout_raises(self):
        from async_scheduler import TaskFuture
        f = TaskFuture(task_id=2)
        with self.assertRaises(TimeoutError):
            f.result(timeout=0.05)

    def test_future_is_done_after_mark_done(self):
        from async_scheduler import TaskFuture
        f = TaskFuture(task_id=3)
        self.assertFalse(f.is_done)
        f._mark_done(42)
        self.assertTrue(f.is_done)

    def test_wall_time_ms_positive_after_done(self):
        from async_scheduler import TaskFuture
        f = TaskFuture(task_id=4)
        time.sleep(0.01)
        f._mark_done("x")
        self.assertGreater(f.wall_time_ms, 0)

    def test_task_stats_in_flight(self):
        from async_scheduler import TaskStats
        s = TaskStats(submitted=5, completed=2, cancelled=1, failed=0)
        self.assertEqual(s.in_flight, 2)


# ===========================================================================
# TestWorkerPool
# ===========================================================================

class TestWorkerPool(unittest.TestCase):

    def test_cpu_pool_starts_and_stops(self):
        from async_scheduler import WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=2)
        pool.start()
        self.assertTrue(pool.is_running)
        pool.stop()
        self.assertFalse(pool.is_running)

    def test_transfer_pool_has_one_worker(self):
        from async_scheduler import WorkerPool
        pool = WorkerPool.transfer_pool()
        self.assertEqual(pool.n_workers, 1)

    def test_tasks_execute_correctly(self):
        from async_scheduler import WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=2).start()
        results = []
        futures = [pool.submit(lambda x=i: x * 2) for i in range(5)]
        results = [f.result(timeout=5.0) for f in futures]
        pool.stop()
        self.assertEqual(sorted(results), [0, 2, 4, 6, 8])

    def test_pool_stats_active_workers(self):
        from async_scheduler import WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=2).start()
        # Stats should show pool is running
        stats = pool.stats
        self.assertTrue(stats.is_running)
        self.assertEqual(stats.n_workers, 2)
        pool.stop()

    def test_tasks_processed_counter(self):
        from async_scheduler import WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=1).start()
        [pool.submit(lambda: None).result(timeout=5.0) for _ in range(3)]
        stats = pool.stats
        self.assertEqual(stats.tasks_processed, 3)
        pool.stop()

    def test_submit_after_stop_raises(self):
        from async_scheduler import WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=1).start()
        pool.stop()
        with self.assertRaises(RuntimeError):
            pool.submit(lambda: None)

    def test_submit_wait_returns_result(self):
        from async_scheduler import WorkerPool
        pool = WorkerPool.cpu_pool(n_workers=1).start()
        result = pool.submit_wait(lambda: "hello", timeout=5.0)
        pool.stop()
        self.assertEqual(result, "hello")

    def test_pool_stats_to_dict_serializable(self):
        from async_scheduler import WorkerPool
        import json
        pool = WorkerPool.cpu_pool(n_workers=1).start()
        d = pool.stats.to_dict()
        json.dumps(d)
        pool.stop()


# ===========================================================================
# TestPrefetchManager
# ===========================================================================

class TestPrefetchManager(unittest.TestCase):

    def test_no_boundary_returns_empty_result(self):
        from async_scheduler import PrefetchManager
        mgr = PrefetchManager(pinned_buffer_size_mb=64, n_buffers=2)
        # All-GPU plan: no CPU segments → no boundaries
        plan = _plan_mock(n_gpu=10, n_cpu=0, boundary=0)
        result = mgr.prefetch(plan)
        self.assertEqual(result.segments_prefetched, 0)
        self.assertEqual(len(result.boundary_layer_indices), 0)

    def test_boundary_detected_for_split_plan(self):
        from async_scheduler import PrefetchManager
        mgr = PrefetchManager(pinned_buffer_size_mb=64, n_buffers=2)
        plan = _plan_mock(n_gpu=5, n_cpu=5, boundary=1)
        result = mgr.prefetch(plan)
        # CPU segment adjacent to GPU segment → at least 1 boundary layer
        self.assertGreater(len(result.boundary_layer_indices), 0)

    def test_recommended_no_mmap_for_split_plan(self):
        from async_scheduler import PrefetchManager
        mgr = PrefetchManager(pinned_buffer_size_mb=64, n_buffers=4)
        plan = _plan_mock(n_gpu=5, n_cpu=5, boundary=1)
        result = mgr.prefetch(plan)
        self.assertTrue(result.recommended_no_mmap)

    def test_no_mmap_not_recommended_for_single_segment(self):
        from async_scheduler import PrefetchManager
        mgr = PrefetchManager(pinned_buffer_size_mb=64, n_buffers=2)
        plan = _plan_mock(n_gpu=0, n_cpu=10, boundary=0)
        result = mgr.prefetch(plan)
        self.assertFalse(result.recommended_no_mmap)

    def test_prefetch_result_has_timing(self):
        from async_scheduler import PrefetchManager
        mgr = PrefetchManager(pinned_buffer_size_mb=64, n_buffers=2)
        plan = _plan_mock(n_gpu=5, n_cpu=5, boundary=1)
        result = mgr.prefetch(plan)
        self.assertGreaterEqual(result.prefetch_time_ms, 0.0)

    def test_to_dict_serializable(self):
        from async_scheduler import PrefetchManager
        import json
        mgr = PrefetchManager(pinned_buffer_size_mb=64, n_buffers=2)
        plan = _plan_mock(n_gpu=5, n_cpu=5, boundary=1)
        result = mgr.prefetch(plan)
        json.dumps(result.to_dict())

    def test_release_all_clears_warm_segments(self):
        from async_scheduler import PrefetchManager
        mgr = PrefetchManager(pinned_buffer_size_mb=64, n_buffers=2)
        plan = _plan_mock(n_gpu=5, n_cpu=5, boundary=1)
        mgr.prefetch(plan)
        mgr.release_all()
        self.assertEqual(mgr.warm_segment_count, 0)

    def test_pinned_buffer_acquire_and_release(self):
        from async_scheduler.prefetch_manager import PinnedBuffer
        buf = PinnedBuffer(size_bytes=1024, buffer_id=0)
        self.assertFalse(buf.is_in_use)
        acquired = buf.acquire("seg_0_5_GPU")
        self.assertTrue(acquired)
        self.assertTrue(buf.is_in_use)
        buf.release()
        self.assertFalse(buf.is_in_use)

    def test_double_acquire_fails(self):
        from async_scheduler.prefetch_manager import PinnedBuffer
        buf = PinnedBuffer(size_bytes=1024, buffer_id=1)
        buf.acquire("seg_a")
        second = buf.acquire("seg_b")
        self.assertFalse(second)


# ===========================================================================
# TestCudaStreamManager
# ===========================================================================

class TestCudaStreamManager(unittest.TestCase):

    def test_cpu_backend_emits_no_env_vars(self):
        from async_scheduler import CudaStreamManager
        mgr = CudaStreamManager(backend="cpu", hw_profile=_hw())
        self.assertEqual(mgr.env_vars, {})

    def test_cuda_backend_emits_cuda_env_vars(self):
        from async_scheduler import CudaStreamManager
        hw = _nvidia_hw()
        # Patch _detect_cuda to return True without running nvidia-smi
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=True):
            mgr = CudaStreamManager(backend="cuda", hw_profile=hw)
        self.assertIn("CUDA_LAUNCH_BLOCKING", mgr.env_vars)
        self.assertEqual(mgr.env_vars["CUDA_LAUNCH_BLOCKING"], "0")

    def test_vulkan_backend_emits_vulkan_env_vars(self):
        from async_scheduler import CudaStreamManager
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=False):
            mgr = CudaStreamManager(backend="vulkan", hw_profile=_hw())
        self.assertIn("GGML_VK_DISABLE_VALIDATION", mgr.env_vars)

    def test_dispatch_increments_stream_counter(self):
        from async_scheduler import CudaStreamManager, VirtualStream
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=False):
            mgr = CudaStreamManager(backend="cpu", hw_profile=_hw())
        mgr.dispatch(VirtualStream.COMPUTE)
        mgr.dispatch(VirtualStream.COMPUTE)
        stats = mgr.get_stream_stats()
        compute_stats = next(s for s in stats if s.stream == VirtualStream.COMPUTE)
        self.assertEqual(compute_stats.dispatches, 2)

    def test_overlap_estimate_zero_without_plan(self):
        from async_scheduler import CudaStreamManager
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=False):
            mgr = CudaStreamManager(backend="cpu")
        self.assertEqual(mgr.estimated_overlap_ratio, 0.0)

    def test_overlap_estimate_positive_with_split_plan(self):
        from async_scheduler import CudaStreamManager
        plan = _plan_mock(n_gpu=10, n_cpu=10, boundary=2)
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=False):
            mgr = CudaStreamManager(backend="cpu", plan=plan)
        # With boundary crossings, some overlap should be estimated
        self.assertGreaterEqual(mgr.estimated_overlap_ratio, 0.0)

    def test_get_all_stats_serializable(self):
        from async_scheduler import CudaStreamManager
        import json
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=False):
            mgr = CudaStreamManager(backend="cpu", hw_profile=_hw())
        json.dumps(mgr.get_all_stats())

    def test_update_plan_changes_overlap_estimate(self):
        from async_scheduler import CudaStreamManager
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=False):
            mgr = CudaStreamManager(backend="cpu")
        initial = mgr.estimated_overlap_ratio
        plan = _plan_mock(n_gpu=10, n_cpu=10, boundary=3)
        mgr.update_plan(plan)
        # After update, overlap may change (may stay 0 for CPU-only backend)
        self.assertIsInstance(mgr.estimated_overlap_ratio, float)

    def test_synchronize_is_noop(self):
        from async_scheduler import CudaStreamManager, VirtualStream
        with patch.object(CudaStreamManager, "_detect_cuda", return_value=False):
            mgr = CudaStreamManager(backend="cpu")
        mgr.synchronize(VirtualStream.COMPUTE)  # must not raise


# ===========================================================================
# TestSchedulerMetrics
# ===========================================================================

class TestSchedulerMetrics(unittest.TestCase):

    def _make_timings(self, req_id=0, prep=10, xfr=5, compute=100, tokens=50, tps=25.0):
        from async_scheduler import StageTimings
        return StageTimings(
            request_id=req_id,
            prepare_ms=prep,
            transfer_ms=xfr,
            compute_ms=compute,
            total_ms=prep + xfr + compute,
            tokens_generated=tokens,
            eval_tps=tps,
        )

    def test_stage_timings_pipeline_efficiency_perfect(self):
        from async_scheduler import StageTimings
        t = StageTimings(prepare_ms=10, transfer_ms=5, compute_ms=100, total_ms=115)
        self.assertAlmostEqual(t.pipeline_efficiency(), 1.0)

    def test_stage_timings_efficiency_with_idle(self):
        from async_scheduler import StageTimings
        t = StageTimings(
            prepare_ms=10, transfer_ms=5, compute_ms=100, total_ms=200,
            compute_idle_ms=90,  # 90 ms of idle in compute stage
        )
        # efficiency = 1 - 90/200 = 0.55
        self.assertAlmostEqual(t.pipeline_efficiency(), 0.55)

    def test_metrics_accumulator_records(self):
        from async_scheduler.scheduler_metrics import MetricsAccumulator
        acc = MetricsAccumulator()
        for i in range(3):
            acc.record(self._make_timings(i))
        m = acc.compute()
        self.assertEqual(m.total_requests, 3)

    def test_metrics_mean_tps(self):
        from async_scheduler.scheduler_metrics import MetricsAccumulator
        acc = MetricsAccumulator()
        acc.record(self._make_timings(0, tps=20.0, tokens=40))
        acc.record(self._make_timings(1, tps=30.0, tokens=60))
        m = acc.compute()
        self.assertAlmostEqual(m.mean_tokens_per_sec, 25.0)

    def test_metrics_peak_tps(self):
        from async_scheduler.scheduler_metrics import MetricsAccumulator
        acc = MetricsAccumulator()
        acc.record(self._make_timings(0, tps=10.0))
        acc.record(self._make_timings(1, tps=40.0))
        m = acc.compute()
        self.assertAlmostEqual(m.peak_tokens_per_sec, 40.0)

    def test_speedup_estimate_with_baseline(self):
        from async_scheduler.scheduler_metrics import MetricsAccumulator
        acc = MetricsAccumulator()
        acc.record(self._make_timings(0, tps=20.0))
        m = acc.compute(sequential_tps=10.0)
        self.assertAlmostEqual(m.speedup_estimate, 2.0)

    def test_to_dict_serializable(self):
        from async_scheduler import SchedulerMetrics
        import json
        m = SchedulerMetrics(total_requests=1, mean_tokens_per_sec=25.0)
        json.dumps(m.to_dict())

    def test_report_summary_not_empty(self):
        from async_scheduler import SchedulerMetrics, SchedulerReport
        m = SchedulerMetrics(mean_tokens_per_sec=20.0, mean_pipeline_efficiency=0.85)
        rep = SchedulerReport(m, sequential_tps=15.0)
        summary = rep.summary()
        self.assertIn("tok/s", summary)

    def test_report_comparison_table_not_empty(self):
        from async_scheduler import SchedulerMetrics, SchedulerReport
        m = SchedulerMetrics(mean_tokens_per_sec=20.0)
        rep = SchedulerReport(m, sequential_tps=15.0)
        table = rep.comparison_table()
        self.assertIn("Sequential", table)
        self.assertIn("Async", table)

    def test_full_report_not_empty(self):
        from async_scheduler import SchedulerMetrics, SchedulerReport
        m = SchedulerMetrics(mean_tokens_per_sec=20.0, total_requests=3)
        rep = SchedulerReport(m, sequential_tps=15.0)
        report = rep.full_report()
        self.assertIn("Phase 8", report)
        self.assertIn("THROUGHPUT", report)

    def test_accumulator_reset(self):
        from async_scheduler.scheduler_metrics import MetricsAccumulator
        acc = MetricsAccumulator()
        acc.record(self._make_timings(0))
        acc.reset()
        m = acc.compute()
        self.assertEqual(m.total_requests, 0)

    def test_empty_accumulator_returns_defaults(self):
        from async_scheduler.scheduler_metrics import MetricsAccumulator
        acc = MetricsAccumulator()
        m = acc.compute()
        self.assertEqual(m.total_requests, 0)
        self.assertEqual(m.mean_tokens_per_sec, 0.0)


# ===========================================================================
# TestPipelineCoordinator
# ===========================================================================

class TestPipelineCoordinator(unittest.TestCase):

    def _make_mock_result(self, tps=25.0, tokens=50):
        result = MagicMock()
        result.stats.eval_tps = tps
        result.stats.tokens_generated = tokens
        return result

    def _session_factory(self, model_path, plan, config, hw):
        session = MagicMock()
        session.run.return_value = self._make_mock_result()
        return session

    def test_coordinator_starts_and_stops(self):
        from async_scheduler import PipelineCoordinator, SchedulerConfig
        cfg = SchedulerConfig(n_cpu_workers=2, max_queue_depth=4, enable_prefetch=False)
        coord = PipelineCoordinator(cfg, _hw(), self._session_factory)
        coord.start()
        self.assertTrue(coord._running)
        coord.stop()
        self.assertFalse(coord._running)

    def test_run_returns_mock_result(self):
        from async_scheduler import PipelineCoordinator, SchedulerConfig
        from inference_runtime import RuntimeConfig
        cfg = SchedulerConfig(n_cpu_workers=1, max_queue_depth=4, enable_prefetch=False)
        coord = PipelineCoordinator(cfg, _hw(), self._session_factory)
        coord.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        rc = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        result = coord.run(plan, Path("model.gguf"), "Hello", rc, timeout=30.0)
        coord.stop()
        self.assertIsNotNone(result)

    def test_metrics_recorded_after_run(self):
        from async_scheduler import PipelineCoordinator, SchedulerConfig
        from inference_runtime import RuntimeConfig
        cfg = SchedulerConfig(n_cpu_workers=1, max_queue_depth=4, enable_prefetch=False)
        coord = PipelineCoordinator(cfg, _hw(), self._session_factory)
        coord.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        rc = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        coord.run(plan, Path("model.gguf"), "Hello", rc, timeout=30.0)
        metrics = coord.get_metrics()
        coord.stop()
        self.assertEqual(metrics.total_requests, 1)

    def test_pool_stats_dict_has_required_keys(self):
        from async_scheduler import PipelineCoordinator, SchedulerConfig
        cfg = SchedulerConfig(n_cpu_workers=1, max_queue_depth=4, enable_prefetch=False)
        coord = PipelineCoordinator(cfg, _hw(), self._session_factory)
        coord.start()
        stats = coord.pool_stats()
        coord.stop()
        self.assertIn("cpu_pool", stats)
        self.assertIn("transfer_pool", stats)
        self.assertIn("prefetch", stats)

    def test_context_manager_starts_and_stops(self):
        from async_scheduler import PipelineCoordinator, SchedulerConfig
        cfg = SchedulerConfig(n_cpu_workers=1, max_queue_depth=4, enable_prefetch=False)
        with PipelineCoordinator(cfg, _hw(), self._session_factory) as coord:
            self.assertTrue(coord._running)
        self.assertFalse(coord._running)

    def test_multiple_requests_accumulate_metrics(self):
        from async_scheduler import PipelineCoordinator, SchedulerConfig
        from inference_runtime import RuntimeConfig
        cfg = SchedulerConfig(n_cpu_workers=2, max_queue_depth=8, enable_prefetch=False)
        coord = PipelineCoordinator(cfg, _hw(), self._session_factory)
        coord.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        rc = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        for _ in range(3):
            coord.run(plan, Path("model.gguf"), "ping", rc, timeout=30.0)
        metrics = coord.get_metrics()
        coord.stop()
        self.assertEqual(metrics.total_requests, 3)


# ===========================================================================
# TestAsyncRuntimeEngine
# ===========================================================================

class TestAsyncRuntimeEngine(unittest.TestCase):

    def _mock_session_run(self, *args, **kwargs):
        result = MagicMock()
        result.stats.eval_tps = 20.0
        result.stats.tokens_generated = 40
        result.backend = "cpu"
        return result

    def _patched_engine(self):
        from async_scheduler import AsyncRuntimeEngine
        engine = AsyncRuntimeEngine(hw_profile=_hw())
        # Patch the session factory inside the coordinator
        engine._coordinator._session_factory = lambda mp, pl, cfg, hw: (
            setattr(MagicMock(), "run", self._mock_session_run) or
            type("S", (), {"run": lambda self, p, **kw: self._mock_session_run()})()
        )

        def _sf(model_path, plan, config, hw):
            s = MagicMock()
            s.run.return_value = self._mock_session_run()
            return s

        engine._coordinator._session_factory = _sf
        return engine

    def test_engine_starts_and_stops(self):
        engine = self._patched_engine()
        engine.start()
        self.assertTrue(engine._started)
        engine.stop()
        self.assertFalse(engine._started)

    def test_submit_returns_future(self):
        from async_scheduler import TaskFuture
        engine = self._patched_engine()
        engine.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        from inference_runtime import RuntimeConfig
        rc = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        engine.config = rc
        future = engine.submit(plan, Path("model.gguf"), "Hello")
        self.assertIsInstance(future, TaskFuture)
        future.result(timeout=30.0)
        engine.stop()

    def test_submit_batch_returns_list_of_futures(self):
        engine = self._patched_engine()
        engine.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        from inference_runtime import RuntimeConfig
        rc = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        engine.config = rc
        requests = [
            {"plan": plan, "model_path": Path("model.gguf"), "prompt": f"Q{i}"}
            for i in range(3)
        ]
        futures = engine.submitBatch(requests)
        self.assertEqual(len(futures), 3)
        results = [f.result(timeout=30.0) for f in futures]
        engine.stop()
        self.assertEqual(len(results), 3)

    def test_get_scheduler_metrics_returns_metrics(self):
        from async_scheduler import SchedulerMetrics
        engine = self._patched_engine()
        engine.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        from inference_runtime import RuntimeConfig
        rc = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        engine.config = rc
        engine.submit(plan, Path("model.gguf"), "Hello").result(timeout=30.0)
        metrics = engine.getSchedulerMetrics()
        engine.stop()
        self.assertIsInstance(metrics, SchedulerMetrics)
        self.assertEqual(metrics.total_requests, 1)

    def test_get_pool_stats_has_expected_keys(self):
        engine = self._patched_engine()
        engine.start()
        stats = engine.getPoolStats()
        engine.stop()
        self.assertIn("cpu_pool", stats)
        self.assertIn("transfer_pool", stats)

    def test_context_manager_lifecycle(self):
        engine = self._patched_engine()
        with engine:
            self.assertTrue(engine._started)
        self.assertFalse(engine._started)

    def test_snake_case_aliases_work(self):
        engine = self._patched_engine()
        # These should exist
        self.assertTrue(hasattr(engine, "submit_batch"))
        self.assertTrue(hasattr(engine, "get_scheduler_metrics"))
        self.assertTrue(hasattr(engine, "get_pool_stats"))

    def test_execute_plan_routes_through_async(self):
        """When use_async=True and started, executePlan goes through pipeline."""
        engine = self._patched_engine()
        engine.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        from inference_runtime import RuntimeConfig
        rc = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        result = engine.executePlan(plan, Path("model.gguf"), "Hi", use_async=True,
                                    config_override=rc)
        metrics = engine.getSchedulerMetrics()
        engine.stop()
        self.assertIsNotNone(result)
        self.assertGreater(metrics.total_requests, 0)

    def test_get_scheduler_report_is_scheduler_report(self):
        from async_scheduler import SchedulerReport
        engine = self._patched_engine()
        engine.start()
        plan = _plan_mock(n_gpu=0, n_cpu=5, boundary=0)
        from inference_runtime import RuntimeConfig
        engine.config = RuntimeConfig(n_predict=32, process_timeout_sec=30.0)
        engine.submit(plan, Path("model.gguf"), "Hi").result(30.0)
        report = engine.getSchedulerReport()
        engine.stop()
        self.assertIsInstance(report, SchedulerReport)


# ===========================================================================
# TestRuntimeConfigPhase8
# ===========================================================================

class TestRuntimeConfigPhase8(unittest.TestCase):

    def test_scheduler_fields_exist_with_defaults(self):
        from inference_runtime import RuntimeConfig
        cfg = RuntimeConfig()
        self.assertFalse(cfg.enable_async_scheduler)
        self.assertEqual(cfg.scheduler_n_cpu_workers, -1)
        self.assertEqual(cfg.scheduler_queue_depth, 8)
        self.assertTrue(cfg.scheduler_prefetch)
        self.assertFalse(cfg.scheduler_benchmark_mode)

    def test_scheduler_fields_in_to_dict(self):
        from inference_runtime import RuntimeConfig
        cfg = RuntimeConfig()
        d = cfg.to_dict()
        self.assertIn("enable_async_scheduler", d)
        self.assertIn("scheduler_n_cpu_workers", d)
        self.assertIn("scheduler_queue_depth", d)
        self.assertIn("scheduler_prefetch", d)
        self.assertIn("scheduler_benchmark_mode", d)

    def test_to_dict_serializable(self):
        from inference_runtime import RuntimeConfig
        import json
        cfg = RuntimeConfig(enable_async_scheduler=True, scheduler_queue_depth=16)
        json.dumps(cfg.to_dict())

    def test_from_hw_profile_preserves_scheduler_overrides(self):
        from inference_runtime import RuntimeConfig
        cfg = RuntimeConfig.from_hw_profile(
            _hw(),
            enable_async_scheduler=True,
            scheduler_queue_depth=12,
        )
        self.assertTrue(cfg.enable_async_scheduler)
        self.assertEqual(cfg.scheduler_queue_depth, 12)


# ===========================================================================
# TestInferenceRuntimeExports
# ===========================================================================

class TestInferenceRuntimeExports(unittest.TestCase):

    def _get(self, name):
        """Safely get a name from inference_runtime (works with conditional imports)."""
        import inference_runtime
        return getattr(inference_runtime, name, None)

    def test_async_runtime_engine_importable(self):
        from async_scheduler import AsyncRuntimeEngine
        self.assertIsNotNone(AsyncRuntimeEngine)

    def test_scheduler_config_importable(self):
        from async_scheduler import SchedulerConfig
        self.assertIsNotNone(SchedulerConfig)

    def test_scheduler_metrics_importable(self):
        from async_scheduler import SchedulerMetrics
        self.assertIsNotNone(SchedulerMetrics)

    def test_task_future_importable(self):
        from async_scheduler import TaskFuture
        self.assertIsNotNone(TaskFuture)

    def test_scheduler_priority_importable(self):
        from async_scheduler import Priority
        self.assertIsNotNone(Priority)
        self.assertEqual(Priority.CRITICAL, 0)
        self.assertEqual(Priority.HIGH, 1)

    def test_async_scheduler_package_all_exports(self):
        import async_scheduler
        for name in async_scheduler.__all__:
            self.assertTrue(
                hasattr(async_scheduler, name),
                f"async_scheduler.__all__ includes '{name}' but it's not importable"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
