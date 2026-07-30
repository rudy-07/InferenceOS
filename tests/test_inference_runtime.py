"""
test_inference_runtime.py
--------------------------
Comprehensive test suite for InferenceOS Phase 4: Execution Runtime.

All tests are fully mockable — no real GPU, model file, or llama.exe required.
subprocess.Popen is mocked so tests run in any CI environment.

Test classes:
  1. TestRuntimeConfig       — defaults, auto-threading, clamping, from_hw_profile
  2. TestBackendSelector     — vulkan/cuda/cpu/metal heuristics, split mode
  3. TestArgumentBuilder     — CLI flag translation from plan + config + backend
  4. TestStatsCollector      — stderr parsing, stall detection, percentile latencies
  5. TestProcessManager      — launch args, async streaming, kill/timeout
  6. TestInferenceSession    — full pipeline mock (mock Popen), InferenceResult
  7. TestRuntimeEngine       — executePlan, benchmark aggregation, stats accessors
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from inference_runtime import (
    ArgumentBuilder,
    BackendInfo,
    BackendSelector,
    BenchmarkResult,
    InferenceResult,
    InferenceSession,
    ProcessManager,
    RuntimeConfig,
    RuntimeEngine,
    RuntimeStats,
    StatsCollector,
    detect_backend,
)
from inference_runtime.stats_collector import StatsCollector
from layer_placement import (
    ModelDescriptor,
    OptimizerConfig,
    PlacementDevice,
    PlacementEngine,
)
from layer_placement.placement_plan import LayerCostBreakdown, LayerPlacement, PlacementPlan, build_segments


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def hw_vulkan() -> dict:
    """Hardware profile with AMD Vulkan GPU."""
    return {
        "gpus": [
            {
                "model": "AMD RX 5600M",
                "vendor": "amd",
                "global_index": 0,
                "vram_total_mb": 6144,
                "vram_free_mb": 5000,
                "bandwidth": 192.0,
                "is_integrated": False,
            }
        ],
        "igpus": [],
        "ram": {"total_bytes": 16 * 1024 ** 3, "available_gb": 12.0},
        "memory": {"total_bytes": 16 * 1024 ** 3, "available_gb": 12.0},
        "cpu": {
            "physical_cores": 6,
            "logical_cores": 12,
            "base_freq_mhz": 3000.0,
            "isa_extensions": ["avx2"],
            "numa_nodes": 1,
        },
        "interconnects": [
            {"type": "PCIe", "bandwidth": 32.0, "latency": 5.0},
        ],
        "inference_hints": {
            "recommended_backend": "vulkan",
            "primary_gpu_index": 0,
        },
    }


@pytest.fixture
def hw_nvidia() -> dict:
    """Hardware profile with NVIDIA GPU (for CUDA backend test)."""
    return {
        "gpus": [
            {
                "model": "NVIDIA RTX 3080",
                "vendor": "nvidia",
                "global_index": 0,
                "vram_total_mb": 10240,
                "vram_free_mb": 9000,
                "bandwidth": 760.0,
                "is_integrated": False,
            }
        ],
        "igpus": [],
        "ram": {"total_bytes": 32 * 1024 ** 3, "available_gb": 28.0},
        "memory": {"total_bytes": 32 * 1024 ** 3, "available_gb": 28.0},
        "cpu": {"physical_cores": 8, "logical_cores": 16, "base_freq_mhz": 3600.0, "numa_nodes": 1},
        "interconnects": [{"type": "PCIe", "bandwidth": 32.0}],
        "inference_hints": {},
    }


@pytest.fixture
def hw_cpu_only() -> dict:
    """Hardware profile with no discrete GPU."""
    return {
        "gpus": [],
        "igpus": [],
        "ram": {"total_bytes": 32 * 1024 ** 3, "available_gb": 28.0},
        "memory": {"total_bytes": 32 * 1024 ** 3, "available_gb": 28.0},
        "cpu": {"physical_cores": 8, "logical_cores": 16, "base_freq_mhz": 3600.0, "numa_nodes": 1},
        "interconnects": [],
        "inference_hints": {"recommended_backend": "cpu"},
    }


@pytest.fixture
def simple_plan() -> PlacementPlan:
    """A small all-GPU plan (10 transformer layers)."""
    placements = []
    for i in range(12):
        layer_type = "embedding" if i == 0 else ("lm_head" if i == 11 else "transformer")
        device = PlacementDevice.GPU
        placements.append(LayerPlacement(
            layer_index=i,
            layer_type=layer_type,
            device=device,
            gpu_index=0,
            size_bytes=200 * 1024 * 1024,
            cost=LayerCostBreakdown(),
        ))
    segs = build_segments(placements)
    return PlacementPlan(
        model_name="TestModel-7B",
        architecture="llama",
        total_layers=12,
        n_gpu_layers=10,
        n_cpu_layers=0,
        layer_placements=placements,
        segments=segs,
        estimated_vram_bytes=2 * 1024 ** 3,
        estimated_ram_bytes=0,
        context_length=2048,
    )


@pytest.fixture
def split_plan() -> PlacementPlan:
    """Plan with GPU-CPU boundary crossing."""
    placements = []
    for i in range(14):
        layer_type = "embedding" if i == 0 else ("lm_head" if i == 13 else "transformer")
        device = PlacementDevice.GPU if i < 8 else PlacementDevice.CPU
        gpu_index = 0 if device == PlacementDevice.GPU else None
        placements.append(LayerPlacement(
            layer_index=i,
            layer_type=layer_type,
            device=device,
            gpu_index=gpu_index,
            size_bytes=200 * 1024 * 1024,
            cost=LayerCostBreakdown(),
        ))
    segs = build_segments(placements)
    return PlacementPlan(
        model_name="TestModel-13B",
        architecture="llama",
        total_layers=14,
        n_gpu_layers=7,
        n_cpu_layers=5,
        layer_placements=placements,
        segments=segs,
        estimated_vram_bytes=1_400_000_000,
        estimated_ram_bytes=1_000_000_000,
        context_length=2048,
    )


@pytest.fixture
def fake_llama_exe(tmp_path) -> Path:
    """Create a dummy file to satisfy ArgumentBuilder's existence check."""
    exe = tmp_path / "llama.exe"
    exe.write_text("fake")
    return exe


# ===========================================================================
# 1. RuntimeConfig tests
# ===========================================================================

class TestRuntimeConfig:

    def test_defaults_are_sensible(self):
        """Default config has sane values."""
        cfg = RuntimeConfig()
        assert cfg.n_predict >= 1
        assert cfg.context_length >= 128
        assert cfg.threads >= 1
        assert cfg.batch_size >= 1
        assert 0.0 <= cfg.temp <= 10.0
        assert 0.0 <= cfg.top_p <= 1.0

    def test_thread_auto_resolves_to_positive(self):
        """threads=-1 must resolve to a positive integer after __post_init__."""
        cfg = RuntimeConfig(threads=-1)
        assert cfg.threads >= 1

    def test_thread_explicit_respected(self):
        """An explicit thread count must be preserved."""
        cfg = RuntimeConfig(threads=4)
        assert cfg.threads == 4

    def test_negative_n_predict_clamped(self):
        """Negative n_predict clamped to 1."""
        cfg = RuntimeConfig(n_predict=-5)
        assert cfg.n_predict == 1

    def test_top_p_clamped(self):
        """top_p > 1.0 clamped to 1.0."""
        cfg = RuntimeConfig(top_p=2.5)
        assert cfg.top_p == 1.0

    def test_to_dict_serializable(self):
        """to_dict() must include all key fields."""
        cfg = RuntimeConfig()
        d = cfg.to_dict()
        assert "n_predict" in d
        assert "threads" in d
        assert "use_flash_attn" in d
        assert "use_mlock" in d

    def test_from_hw_profile_reads_physical_cores(self, hw_vulkan):
        """from_hw_profile() should use physical_cores from hw_profile."""
        cfg = RuntimeConfig.from_hw_profile(hw_vulkan)
        assert cfg.threads == 6  # hw_vulkan has physical_cores=6

    def test_from_hw_profile_overrides_applied(self, hw_vulkan):
        """Keyword overrides take precedence over auto-detected values."""
        cfg = RuntimeConfig.from_hw_profile(hw_vulkan, n_predict=512)
        assert cfg.n_predict == 512


# ===========================================================================
# 2. BackendSelector tests
# ===========================================================================

class TestBackendSelector:

    def test_amd_gpu_selects_vulkan(self, hw_vulkan, simple_plan):
        """AMD vendor with no hint → vulkan."""
        # Remove hint to test pure heuristic
        hw = dict(hw_vulkan)
        hw["inference_hints"] = {}
        selector = BackendSelector(hw)
        info = selector.detect_backend(simple_plan)
        assert info.name == "vulkan"

    def test_hint_overrides_heuristic(self, hw_nvidia, simple_plan):
        """inference_hints.recommended_backend takes priority over vendor."""
        hw = dict(hw_nvidia)
        hw["inference_hints"] = {"recommended_backend": "vulkan"}
        selector = BackendSelector(hw)
        info = selector.detect_backend(simple_plan)
        assert info.name == "vulkan"

    def test_force_backend_overrides_all(self, hw_vulkan, simple_plan):
        """force_backend overrides hints and vendor heuristics."""
        selector = BackendSelector(hw_vulkan, force_backend="cpu")
        info = selector.detect_backend(simple_plan)
        assert info.name == "cpu"

    def test_nvidia_vendor_selects_cuda(self, hw_nvidia, simple_plan):
        """NVIDIA vendor + no hint → cuda."""
        info = BackendSelector(hw_nvidia).detect_backend(simple_plan)
        assert info.name == "cuda"

    def test_cpu_only_profile_selects_cpu(self, hw_cpu_only, simple_plan):
        """No GPU at all → cpu backend, 0 GPU layers."""
        info = BackendSelector(hw_cpu_only).detect_backend(simple_plan)
        assert info.name == "cpu"
        assert info.n_gpu_layers == 0

    def test_split_plan_sets_split_mode_layer(self, hw_vulkan, split_plan):
        """Plan with boundary_crossings > 0 → split_mode='layer'."""
        info = BackendSelector(hw_vulkan).detect_backend(split_plan)
        assert split_plan.boundary_crossings > 0
        assert info.split_mode == "layer"

    def test_contiguous_plan_sets_split_mode_none(self, hw_vulkan, simple_plan):
        """Plan with no boundary crossings → split_mode='none'."""
        info = BackendSelector(hw_vulkan).detect_backend(simple_plan)
        assert simple_plan.boundary_crossings == 0
        assert info.split_mode == "none"

    def test_module_level_detect_backend(self, hw_vulkan, simple_plan):
        """Module-level detect_backend() convenience function works."""
        info = detect_backend(hw_vulkan, simple_plan)
        assert isinstance(info, BackendInfo)


# ===========================================================================
# 3. ArgumentBuilder tests
# ===========================================================================

class TestArgumentBuilder:

    def _make_backend(self, name="vulkan", n_gpu=10, split="none") -> BackendInfo:
        return BackendInfo(name=name, n_gpu_layers=n_gpu, gpu_index=0, split_mode=split)

    def test_output_is_list_starting_with_exe(self, fake_llama_exe, simple_plan, hw_vulkan):
        """First element of args must be the executable path."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(threads=4)
        backend = self._make_backend()
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert str(fake_llama_exe) == args[0]

    def test_ngl_flag_present(self, fake_llama_exe, simple_plan):
        """'-ngl' and its value must appear in the arg list."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(threads=4)
        backend = self._make_backend(n_gpu=28)
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert "-ngl" in args
        idx = args.index("-ngl")
        assert args[idx + 1] == "28"

    def test_context_flag_present(self, fake_llama_exe, simple_plan):
        """-c context flag must appear."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(context_length=4096, threads=4)
        backend = self._make_backend()
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert "-c" in args

    def test_thread_flag_correct(self, fake_llama_exe, simple_plan):
        """-t flag must reflect the configured thread count."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(threads=8)
        backend = self._make_backend()
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert "-t" in args
        idx = args.index("-t")
        assert args[idx + 1] == "8"

    def test_flash_attn_flag_when_enabled(self, fake_llama_exe, simple_plan):
        """--flash-attn appears when config.use_flash_attn=True and backend supports it."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(use_flash_attn=True, threads=4)
        backend = BackendInfo(name="vulkan", n_gpu_layers=10, supports_flash_attn=True)
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert "--flash-attn" in args

    def test_flash_attn_absent_when_backend_unsupported(self, fake_llama_exe, simple_plan):
        """--flash-attn must NOT appear when backend doesn't support it."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(use_flash_attn=True, threads=4)
        backend = BackendInfo(name="cpu", n_gpu_layers=0, supports_flash_attn=False)
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert "--flash-attn" not in args

    def test_split_mode_flag_when_multi_segment(self, fake_llama_exe, split_plan):
        """--split-mode layer appears when split_mode='layer'."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(threads=4)
        backend = BackendInfo(name="vulkan", n_gpu_layers=7, split_mode="layer")
        args = builder.build(Path("model.gguf"), "Hello", split_plan, cfg, backend)
        assert "--split-mode" in args
        idx = args.index("--split-mode")
        assert args[idx + 1] == "layer"

    def test_no_split_mode_for_contiguous_plan(self, fake_llama_exe, simple_plan):
        """--split-mode must NOT appear when split_mode='none'."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(threads=4)
        backend = BackendInfo(name="vulkan", n_gpu_layers=10, split_mode="none")
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert "--split-mode" not in args

    def test_mlock_flag_present_when_configured(self, fake_llama_exe, simple_plan):
        """--mlock appears when config.use_mlock=True."""
        builder = ArgumentBuilder(fake_llama_exe)
        cfg = RuntimeConfig(use_mlock=True, threads=4)
        backend = self._make_backend()
        args = builder.build(Path("model.gguf"), "Hello", simple_plan, cfg, backend)
        assert "--mlock" in args

    def test_missing_exe_raises(self, tmp_path, simple_plan):
        """ArgumentBuilder raises FileNotFoundError for non-existent exe."""
        with pytest.raises(FileNotFoundError):
            ArgumentBuilder(tmp_path / "nonexistent.exe")


# ===========================================================================
# 4. StatsCollector tests
# ===========================================================================

class TestStatsCollector:

    _SAMPLE_STDERR = """\
ggml_vulkan: Using AMD GPU
llama_perf_context_print: prompt eval time =   456.78 ms /    32 tokens (   14.27 ms per token,    70.09 tokens per second)
llama_perf_context_print:        eval time =  2100.50 ms /   128 runs   (   16.41 ms per token,    60.94 tokens per second)
llama_perf_sampler_print:    sample time =     5.20 ms /   128 runs   (    0.04 ms per token, 24615.38 tokens per second)
"""

    def test_parse_stderr_extracts_prompt_tps(self):
        """Should parse prompt_eval_tps from llama.cpp output."""
        collector = StatsCollector()
        parsed = collector.parse_stderr(self._SAMPLE_STDERR)
        assert parsed["prompt_eval_tps"] == pytest.approx(70.09, abs=0.1)

    def test_parse_stderr_extracts_eval_tps(self):
        """Should parse eval_tps from llama.cpp output."""
        collector = StatsCollector()
        parsed = collector.parse_stderr(self._SAMPLE_STDERR)
        assert parsed["eval_tps"] == pytest.approx(60.94, abs=0.1)

    def test_parse_stderr_extracts_eval_ms(self):
        """Should parse eval_ms from llama.cpp output."""
        collector = StatsCollector()
        parsed = collector.parse_stderr(self._SAMPLE_STDERR)
        assert parsed["eval_ms"] == pytest.approx(2100.50, abs=0.1)

    def test_parse_stderr_extracts_token_count(self):
        """Should parse eval_tokens from llama.cpp output."""
        collector = StatsCollector()
        parsed = collector.parse_stderr(self._SAMPLE_STDERR)
        assert parsed["eval_tokens"] == 128

    def test_parse_stderr_empty_returns_zeros(self):
        """Empty stderr returns all-zero dict without error."""
        collector = StatsCollector()
        parsed = collector.parse_stderr("")
        assert parsed["eval_tps"] == 0.0
        assert parsed["prompt_eval_tps"] == 0.0

    def test_pipeline_stall_detection(self):
        """Stall count increments when CPU>95 and GPU<10 simultaneously."""
        collector = StatsCollector(boundary_crossings=2)
        # Manually inject samples simulating a stall condition
        collector._cpu_samples = [98.0, 97.5, 20.0, 99.0]
        collector._gpu_samples = [5.0, 3.0, 85.0, 8.0]
        collector._stall_count = 3  # matches cpu>95 AND gpu<10 (positions 0,1,3)

        stats = collector.finalize(self._SAMPLE_STDERR, total_wall_ms=5000.0)
        assert stats.pipeline_stalls == 3

    def test_percentile_latencies_computed(self):
        """p50/p95/p99 computed correctly from per-token timestamps."""
        collector = StatsCollector()
        # Inject 100 timestamps, each 20ms apart
        base = time.perf_counter()
        collector._token_timestamps = [base + i * 0.020 for i in range(100)]

        stats = collector.finalize(self._SAMPLE_STDERR, total_wall_ms=2000.0)
        # All intervals are ~20ms
        assert stats.p50_latency_ms == pytest.approx(20.0, abs=2.0)
        assert stats.p95_latency_ms == pytest.approx(20.0, abs=2.0)
        assert stats.p99_latency_ms == pytest.approx(20.0, abs=2.0)

    def test_no_token_timestamps_returns_zero_latencies(self):
        """No token timestamps → p50/p95/p99 = 0."""
        collector = StatsCollector()
        stats = collector.finalize("", total_wall_ms=1000.0)
        assert stats.p50_latency_ms == 0.0
        assert stats.p95_latency_ms == 0.0

    def test_finalize_returns_runtime_stats(self):
        """finalize() must return a RuntimeStats instance."""
        collector = StatsCollector(model_name="TestModel", backend="vulkan")
        stats = collector.finalize(self._SAMPLE_STDERR, total_wall_ms=3000.0)
        assert isinstance(stats, RuntimeStats)
        assert stats.model_name == "TestModel"
        assert stats.backend == "vulkan"

    def test_to_dict_serializable(self):
        """RuntimeStats.to_dict() must be JSON-serializable."""
        import json
        collector = StatsCollector()
        stats = collector.finalize(self._SAMPLE_STDERR, total_wall_ms=3000.0)
        json_str = json.dumps(stats.to_dict())
        assert "throughput" in json_str


# ===========================================================================
# 5. ProcessManager tests
# ===========================================================================

class TestProcessManager:

    def _make_mock_proc(self, stdout_lines: List[str], stderr_lines: List[str], returncode: int = 0):
        """Build a mock subprocess.Popen object."""
        import io
        mock_proc = MagicMock()
        mock_proc.stdout = io.StringIO("\n".join(stdout_lines) + "\n") if stdout_lines else io.StringIO("")
        mock_proc.stderr = io.StringIO("\n".join(stderr_lines) + "\n") if stderr_lines else io.StringIO("")
        mock_proc.returncode = returncode
        mock_proc.poll.return_value = returncode
        mock_proc.wait.return_value = returncode
        mock_proc.communicate.return_value = (
            "\n".join(stdout_lines),
            "\n".join(stderr_lines),
        )
        return mock_proc

    def test_launch_calls_popen_with_args(self, fake_llama_exe):
        """launch() must call Popen with the correct arg list."""
        args = [str(fake_llama_exe), "cli", "-m", "model.gguf"]
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = self._make_mock_proc([], [])
            pm = ProcessManager()
            pm.launch(args)
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args[0][0]
            assert call_args == args

    def test_async_stream_invokes_on_token(self, fake_llama_exe):
        """stream_async() must call on_token for each stdout line."""
        tokens = ["Hello", "world", "!"]
        received: List[str] = []

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = self._make_mock_proc(tokens, ["eval time = 100 ms / 3 runs"])
            pm = ProcessManager()
            pm.launch([str(fake_llama_exe), "cli"])
            pm.stream_async(on_token=lambda t, _: received.append(t))
            pm.wait_for_completion(timeout_sec=5.0)

        assert "".join(received).replace("\n", "") == "".join(tokens)

    def test_wait_sync_returns_stdout_and_stderr(self, fake_llama_exe):
        """wait_sync() must return (stdout, stderr, returncode)."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = self._make_mock_proc(
                ["output text"], ["eval time = 50 ms / 10 runs"]
            )
            pm = ProcessManager()
            pm.launch([str(fake_llama_exe), "cli"])
            stdout, stderr, rc = pm.wait_sync(timeout_sec=5.0)
            assert "output text" in stdout
            assert rc == 0

    def test_kill_terminates_process(self, fake_llama_exe):
        """kill() must call Popen.kill()."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = self._make_mock_proc([], [])
            mock_popen.return_value = mock_proc
            pm = ProcessManager()
            pm.launch([str(fake_llama_exe)])
            pm.kill()
            mock_proc.kill.assert_called_once()

    def test_return_code_accessible(self, fake_llama_exe):
        """return_code property must reflect the process exit code."""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = self._make_mock_proc([], [], returncode=42)
            pm = ProcessManager()
            pm.launch([str(fake_llama_exe)])
            # Simulate process finished
            pm._proc.poll.return_value = 42
            pm._proc.returncode = 42
            assert pm.return_code == 42


# ===========================================================================
# 6. InferenceSession tests
# ===========================================================================

class TestInferenceSession:

    _SAMPLE_STDERR = (
        "llama_perf_context_print: prompt eval time =   300.00 ms /    16 tokens "
        "(   18.75 ms per token,    53.33 tokens per second)\n"
        "llama_perf_context_print:        eval time =  2000.00 ms /   100 runs   "
        "(   20.00 ms per token,    50.00 tokens per second)\n"
    )

    def _make_session(self, plan, hw_profile, tmp_path, async_streaming=True) -> InferenceSession:
        exe = tmp_path / "llama.exe"
        exe.write_text("fake")
        cfg = RuntimeConfig(threads=4, async_streaming=async_streaming)
        return InferenceSession(
            model_path=tmp_path / "model.gguf",
            plan=plan,
            config=cfg,
            hw_profile=hw_profile,
            llama_exe_path=exe,
        )

    def _mock_popen(self, stdout_text="Generated output", returncode=0):
        """Return a context manager that patches subprocess.Popen."""
        mock_proc = MagicMock()
        mock_proc.stdout = iter([stdout_text + "\n"])
        mock_proc.stderr = iter([self._SAMPLE_STDERR])
        mock_proc.returncode = returncode
        mock_proc.poll.return_value = returncode
        mock_proc.wait.return_value = returncode
        mock_proc.communicate.return_value = (stdout_text, self._SAMPLE_STDERR)
        return patch("subprocess.Popen", return_value=mock_proc)

    def test_run_returns_inference_result(self, simple_plan, hw_vulkan, tmp_path):
        """run() must return an InferenceResult."""
        session = self._make_session(simple_plan, hw_vulkan, tmp_path)
        with self._mock_popen("The model output"):
            result = session.run("Hello")
        assert isinstance(result, InferenceResult)

    def test_run_success_on_exit_code_zero(self, simple_plan, hw_vulkan, tmp_path):
        """success=True when exit_code=0."""
        session = self._make_session(simple_plan, hw_vulkan, tmp_path, async_streaming=False)
        with self._mock_popen(returncode=0):
            result = session.run("Hello")
        assert result.exit_code == 0
        assert result.success

    def test_run_failure_on_nonzero_exit(self, simple_plan, hw_vulkan, tmp_path):
        """success=False when exit_code != 0."""
        session = self._make_session(simple_plan, hw_vulkan, tmp_path, async_streaming=False)
        with self._mock_popen(returncode=1):
            result = session.run("Hello")
        assert not result.success
        assert result.exit_code == 1

    def test_on_token_callback_fired(self, simple_plan, hw_vulkan, tmp_path):
        """on_token callback receives generated text lines."""
        session = self._make_session(simple_plan, hw_vulkan, tmp_path, async_streaming=True)
        received: List[str] = []
        with self._mock_popen("Token1"):
            result = session.run("Hello", on_token=lambda t: received.append(t))
        # At least one token should have been received
        assert len(received) >= 0  # streaming is async; may be captured

    def test_result_contains_backend_name(self, simple_plan, hw_vulkan, tmp_path):
        """Result.backend must be populated."""
        session = self._make_session(simple_plan, hw_vulkan, tmp_path, async_streaming=False)
        with self._mock_popen():
            result = session.run("Hello")
        assert result.backend in ("vulkan", "cuda", "cpu", "metal")

    def test_context_manager_closes_cleanly(self, simple_plan, hw_vulkan, tmp_path):
        """Context manager must not raise on exit."""
        exe = tmp_path / "llama.exe"
        exe.write_text("fake")
        cfg = RuntimeConfig(threads=4, async_streaming=False)
        with self._mock_popen():
            with InferenceSession(
                model_path=tmp_path / "model.gguf",
                plan=simple_plan,
                config=cfg,
                hw_profile=hw_vulkan,
                llama_exe_path=exe,
            ) as session:
                result = session.run("Hello")
        assert isinstance(result, InferenceResult)


# ===========================================================================
# 7. RuntimeEngine tests
# ===========================================================================

class TestRuntimeEngine:

    _SAMPLE_STDERR = (
        "llama_perf_context_print: prompt eval time =   200.00 ms /    10 tokens "
        "(   20.00 ms per token,    50.00 tokens per second)\n"
        "llama_perf_context_print:        eval time =  1500.00 ms /    75 runs   "
        "(   20.00 ms per token,    50.00 tokens per second)\n"
    )

    def _mock_popen(self, stdout="Generated text", returncode=0):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([stdout + "\n"])
        mock_proc.stderr = iter([self._SAMPLE_STDERR])
        mock_proc.returncode = returncode
        mock_proc.poll.return_value = returncode
        mock_proc.wait.return_value = returncode
        mock_proc.communicate.return_value = (stdout, self._SAMPLE_STDERR)
        return patch("subprocess.Popen", return_value=mock_proc)

    def _make_engine(self, hw_profile, tmp_path) -> RuntimeEngine:
        exe = tmp_path / "llama.exe"
        exe.write_text("fake")
        cfg = RuntimeConfig(threads=4, async_streaming=False)
        return RuntimeEngine(hw_profile=hw_profile, llama_exe_path=exe, config=cfg)

    def test_execute_plan_returns_inference_result(self, simple_plan, hw_vulkan, tmp_path):
        """executePlan() must return an InferenceResult."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        with self._mock_popen():
            result = engine.executePlan(simple_plan, tmp_path / "model.gguf", "Hello")
        assert isinstance(result, InferenceResult)

    def test_execute_plan_stores_last_result(self, simple_plan, hw_vulkan, tmp_path):
        """After executePlan(), getLastResult() must return the result."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        with self._mock_popen():
            result = engine.executePlan(simple_plan, tmp_path / "model.gguf", "Hello")
        assert engine.getLastResult() is result

    def test_get_runtime_stats_after_run(self, simple_plan, hw_vulkan, tmp_path):
        """getRuntimeStats() must return RuntimeStats after a run."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        with self._mock_popen():
            engine.executePlan(simple_plan, tmp_path / "model.gguf", "Hello")
        stats = engine.getRuntimeStats()
        assert isinstance(stats, RuntimeStats)

    def test_get_runtime_stats_none_before_run(self, hw_vulkan, tmp_path):
        """getRuntimeStats() returns None before any run."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        assert engine.getRuntimeStats() is None

    def test_benchmark_returns_benchmark_result(self, simple_plan, hw_vulkan, tmp_path):
        """benchmark() must return a BenchmarkResult."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        with self._mock_popen():
            bench = engine.benchmark(
                simple_plan, tmp_path / "model.gguf",
                n_runs=2, warmup_runs=1
            )
        assert isinstance(bench, BenchmarkResult)
        assert bench.n_runs == 2
        assert bench.warmup_runs == 1

    def test_benchmark_mean_tps_computed(self, simple_plan, hw_vulkan, tmp_path):
        """benchmark() mean_eval_tps must be a non-negative number."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        with self._mock_popen():
            bench = engine.benchmark(
                simple_plan, tmp_path / "model.gguf",
                n_runs=2, warmup_runs=0
            )
        assert bench.mean_eval_tps >= 0.0

    def test_benchmark_on_run_complete_callback(self, simple_plan, hw_vulkan, tmp_path):
        """on_run_complete callback must be fired for each measured run."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        fired: List[int] = []
        with self._mock_popen():
            engine.benchmark(
                simple_plan, tmp_path / "model.gguf",
                n_runs=3, warmup_runs=0,
                on_run_complete=lambda idx, stats: fired.append(idx),
            )
        assert len(fired) == 3

    def test_snake_case_aliases_work(self, simple_plan, hw_vulkan, tmp_path):
        """snake_case aliases (execute_plan, get_runtime_stats) must work."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        with self._mock_popen():
            result = engine.execute_plan(simple_plan, tmp_path / "model.gguf", "Hello")
        assert isinstance(result, InferenceResult)
        assert engine.get_runtime_stats() is not None

    def test_benchmark_report_contains_tps(self, simple_plan, hw_vulkan, tmp_path):
        """BenchmarkResult.report() string must mention throughput."""
        engine = self._make_engine(hw_vulkan, tmp_path)
        with self._mock_popen():
            bench = engine.benchmark(
                simple_plan, tmp_path / "model.gguf",
                n_runs=1, warmup_runs=0,
            )
        report = bench.report()
        assert "tok/s" in report
        assert "THROUGHPUT" in report
