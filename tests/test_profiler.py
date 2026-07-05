"""
test_profiler.py
----------------
Unit tests for the InferenceOS hardware profiler (Phase 1).

Tests are written to run without any real GPU drivers installed.
All external calls (nvidia-smi, rocm-smi, subprocess, pynvml) are mocked.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CPU Profiler tests
# ---------------------------------------------------------------------------

class TestCpuProfiler:
    def test_returns_required_keys(self):
        from profiler import cpu_profiler

        result = cpu_profiler.profile()
        required = {"brand", "architecture", "physical_cores", "logical_cores",
                    "base_freq_mhz", "cache_l2_kb", "cache_l3_mb", "isa_extensions"}
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_core_counts_are_positive_integers(self):
        from profiler import cpu_profiler

        result = cpu_profiler.profile()
        assert isinstance(result["physical_cores"], int)
        assert isinstance(result["logical_cores"], int)
        assert result["physical_cores"] >= 1
        assert result["logical_cores"] >= result["physical_cores"]

    def test_isa_extensions_is_list_of_strings(self):
        from profiler import cpu_profiler

        result = cpu_profiler.profile()
        assert isinstance(result["isa_extensions"], list)
        for ext in result["isa_extensions"]:
            assert isinstance(ext, str)

    def test_isa_filter_removes_irrelevant_flags(self):
        from profiler.cpu_profiler import _detect_isa_extensions

        raw = {"flags": ["avx2", "avx512f", "aes", "pclmul", "xsave"]}
        exts = _detect_isa_extensions(raw)
        assert "avx2" in exts
        assert "avx512f" in exts
        # aes is not in RELEVANT set
        assert "aes" not in exts

    def test_survives_missing_cpuinfo_library(self):
        """Should not raise even when py-cpuinfo is not installed."""
        with patch.dict("sys.modules", {"cpuinfo": None}):
            from profiler import cpu_profiler
            # Just verifying no exception is raised
            result = cpu_profiler.profile()
            assert "physical_cores" in result


# ---------------------------------------------------------------------------
# Memory Profiler tests
# ---------------------------------------------------------------------------

class TestMemoryProfiler:
    def test_returns_required_keys(self):
        from profiler import memory_profiler

        result = memory_profiler.profile()
        required = {"total_gb", "available_gb", "used_gb", "percent_used",
                    "swap_total_gb", "swap_used_gb"}
        assert required.issubset(result.keys())

    def test_values_are_non_negative(self):
        from profiler import memory_profiler

        result = memory_profiler.profile()
        for key, val in result.items():
            if key == "error":
                continue
            assert val >= 0, f"{key} should be non-negative, got {val}"

    def test_available_le_total(self):
        from profiler import memory_profiler

        result = memory_profiler.profile()
        if result.get("total_gb", 0) > 0:
            assert result["available_gb"] <= result["total_gb"]

    def test_survives_psutil_error(self):
        """Should return a dict with an error key, not raise."""
        with patch("psutil.virtual_memory", side_effect=RuntimeError("mock failure")):
            from profiler import memory_profiler
            result = memory_profiler.profile()
            assert "error" in result or "total_gb" in result


# ---------------------------------------------------------------------------
# NVIDIA backend tests
# ---------------------------------------------------------------------------

class TestNvidiaBackend:
    def test_returns_empty_list_when_no_nvidia(self):
        """All three detection tiers should fail gracefully."""
        with (
            patch.dict("sys.modules", {"pynvml": None, "GPUtil": None}),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            from profiler.gpu_backends import nvidia_backend
            result = nvidia_backend.detect()
            assert result == []

    def test_nvidia_smi_parsed_correctly(self):
        """Mock nvidia-smi CSV output and verify parsing."""
        mock_output = (
            " 0, NVIDIA GeForce RTX 4090, GPU-abc123, "
            "24576, 23000, 1576, 535.154.05, 8.9\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("subprocess.run", return_value=mock_result),
        ):
            from profiler.gpu_backends import nvidia_backend
            gpus = nvidia_backend._probe_nvidia_smi()

        assert gpus is not None
        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu["vendor"] == "nvidia"
        assert gpu["name"] == "NVIDIA GeForce RTX 4090"
        assert gpu["vram_total_mb"] == 24576
        assert gpu["vram_free_mb"] == 23000
        assert gpu["compute_capability"] == "8.9"
        assert gpu["backend_hint"] == "cuda"

    def test_nvidia_smi_failure_returns_none(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            from profiler.gpu_backends import nvidia_backend
            result = nvidia_backend._probe_nvidia_smi()
        assert result is None

    def test_gpu_descriptor_has_all_required_fields(self):
        """Any GPU dict returned must have these fields for downstream phases."""
        mock_output = (
            " 0, Tesla T4, GPU-xyz, 16384, 15000, 1384, 470.00, 7.5\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_output

        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("subprocess.run", return_value=mock_result),
        ):
            from profiler.gpu_backends import nvidia_backend
            gpus = nvidia_backend._probe_nvidia_smi()

        required = {"index", "vendor", "name", "vram_total_mb", "vram_free_mb",
                    "vram_used_mb", "driver_version", "compute_capability", "backend_hint"}
        assert required.issubset(gpus[0].keys())


# ---------------------------------------------------------------------------
# AMD backend tests
# ---------------------------------------------------------------------------

class TestAmdBackend:
    def test_returns_empty_list_when_no_rocm(self):
        with (
            patch.dict("sys.modules", {"amdsmi": None}),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            from profiler.gpu_backends import amd_backend
            result = amd_backend.detect()
            assert result == []

    def test_rocm_smi_json_parsed_correctly(self):
        mock_json = json.dumps({
            "card0": {
                "Card Series": "AMD Radeon RX 7900 XTX",
                "Driver version": "6.0.5",
                "VRAM Total Memory (B)": "24576",
                "VRAM Total Used Memory (B)": "512",
            }
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_json

        with (
            patch.dict("sys.modules", {"amdsmi": None}),
            patch("subprocess.run", return_value=mock_result),
        ):
            from profiler.gpu_backends import amd_backend
            gpus = amd_backend._probe_rocm_smi()

        assert gpus is not None
        assert gpus[0]["vendor"] == "amd"
        assert gpus[0]["backend_hint"] == "rocm"
        assert gpus[0]["name"] == "AMD Radeon RX 7900 XTX"


# ---------------------------------------------------------------------------
# Apple backend tests
# ---------------------------------------------------------------------------

class TestAppleBackend:
    def test_returns_empty_list_on_non_macos(self):
        with patch("platform.system", return_value="Windows"):
            from profiler.gpu_backends import apple_backend
            result = apple_backend.detect()
            assert result == []

    def test_system_profiler_parsed_correctly(self):
        mock_json = json.dumps({
            "SPDisplaysDataType": [
                {
                    "sppci_model": "Apple M2 Pro",
                    "spdisplays_vram": "16 GB",
                    "sppci_vendor": "Apple",
                }
            ]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_json

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", return_value=mock_result),
        ):
            from profiler.gpu_backends import apple_backend
            gpus = apple_backend._probe_system_profiler()

        assert gpus is not None
        assert gpus[0]["vendor"] == "apple"
        assert gpus[0]["vram_total_mb"] == 16384  # 16 GB → 16384 MB
        assert gpus[0]["backend_hint"] == "metal"


# ---------------------------------------------------------------------------
# Full profiler integration test
# ---------------------------------------------------------------------------

class TestHardwareProfiler:
    def test_run_profiler_returns_valid_schema(self):
        """
        Smoke-test: run the full profiler and verify schema structure.
        Does not mock anything — uses real system data.
        """
        from profiler.hardware_profiler import run_profiler

        profile = run_profiler(verbose=False)

        assert profile["schema_version"] == "1.0"
        assert "timestamp" in profile
        assert "os" in profile
        assert "cpu" in profile
        assert "memory" in profile
        assert "gpus" in profile
        assert "inference_hints" in profile

        hints = profile["inference_hints"]
        assert "recommended_backend" in hints
        assert "recommended_quant" in hints
        assert "max_gpu_layers" in hints
        assert "parallelism_threads" in hints

    def test_profile_is_json_serializable(self):
        from profiler.hardware_profiler import run_profiler

        profile = run_profiler(verbose=False)
        # Should not raise
        json_str = json.dumps(profile)
        restored = json.loads(json_str)
        assert restored["schema_version"] == "1.0"

    def test_inference_hints_cpu_only(self):
        from profiler.hardware_profiler import _derive_inference_hints

        cpu = {"logical_cores": 8, "isa_extensions": ["avx2"]}
        memory = {"available_gb": 16.0}
        hints = _derive_inference_hints(cpu, memory, gpus=[])

        assert hints["recommended_backend"] == "cpu"
        assert hints["max_gpu_layers"] == 0
        assert hints["parallelism_threads"] == 8
        assert hints["primary_gpu_index"] is None

    def test_inference_hints_nvidia_gpu(self):
        from profiler.hardware_profiler import _derive_inference_hints

        cpu = {"logical_cores": 16, "isa_extensions": ["avx2"]}
        memory = {"available_gb": 64.0}
        # 24576 MB free → meets the 24000 MB threshold → Q8_0
        gpus = [{
            "global_index": 0, "vendor": "nvidia",
            "vram_total_mb": 24576, "vram_free_mb": 24576,
            "backend_hint": "cuda",
        }]
        hints = _derive_inference_hints(cpu, memory, gpus)

        assert hints["recommended_backend"] == "cuda"
        assert hints["max_gpu_layers"] == -1
        assert hints["recommended_quant"] == "Q8_0"

    def test_inference_hints_small_vram(self):
        from profiler.hardware_profiler import _derive_inference_hints

        cpu = {"logical_cores": 8, "isa_extensions": []}
        memory = {"available_gb": 16.0}
        # 4096 MB free → hits the (4_000, "Q4_0") bucket (Q4_K_M needs ≥8000 MB)
        gpus = [{
            "global_index": 0, "vendor": "nvidia",
            "vram_total_mb": 4096, "vram_free_mb": 4096,
            "backend_hint": "cuda",
        }]
        hints = _derive_inference_hints(cpu, memory, gpus)
        assert hints["recommended_quant"] == "Q4_0"

    def test_no_crash_on_all_backends_absent(self):
        """Simulate a system where no GPU library or tool is available."""
        with (
            patch.dict("sys.modules", {"pynvml": None, "GPUtil": None, "amdsmi": None}),
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.system", return_value="Linux"),
        ):
            from profiler.hardware_profiler import run_profiler
            profile = run_profiler(verbose=False)
            assert profile["gpus"] == []
            assert profile["inference_hints"]["recommended_backend"] == "cpu"
