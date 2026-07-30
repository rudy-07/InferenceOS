"""
test_profiler.py
----------------
Unit tests for the InferenceOS hardware profiler and resource manager (Phase 1).

Tests cover multiple hardware configurations:
  - CPU only
  - CPU + NVIDIA
  - CPU + AMD
  - Laptop with iGPU
  - Missing GPU
  - Low memory
  - Unified Resource Model metrics (capacity, available, bandwidth, latency, utilization)
  - Public APIs (getSystemResources, getAvailableMemory, getGPUs, getCPUs, estimateBandwidth)

Run with:
    pytest tests/test_profiler.py -v
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from profiler import (
    BaseResource,
    CPUResource,
    GPUResource,
    RAMResource,
    StorageResource,
    SystemResources,
    estimateBandwidth,
    estimate_bandwidth,
    getAvailableMemory,
    getCPUs,
    getGPUs,
    getSystemResources,
    get_available_memory,
    get_cpus,
    get_gpus,
    get_system_resources,
    run_profiler,
)


# ---------------------------------------------------------------------------
# Resource Model tests
# ---------------------------------------------------------------------------

class TestResourceModel:
    def test_base_resource_contract(self):
        res = BaseResource(capacity=100.0, available=80.0, bandwidth=50.0, latency=0.1, utilization=20.0)
        assert res.capacity == 100.0
        assert res.available == 80.0
        assert res.bandwidth == 50.0
        assert res.latency == 0.1
        assert res.utilization == 20.0

    def test_cpu_resource_model(self):
        cpu = CPUResource(
            capacity=16.0, available=12.0, utilization=25.0,
            brand="AMD Ryzen 9", physical_cores=8, logical_cores=16,
            cache_l1_kb=512.0, cache_l2_kb=8192.0, cache_l3_mb=32.0,
            numa_nodes=1,
        )
        assert cpu.logical_cores == 16
        assert cpu.cache_l3_mb == 32.0
        d = cpu.to_dict()
        assert "capacity" in d
        assert "available" in d
        assert "bandwidth" in d
        assert "latency" in d
        assert "utilization" in d

    def test_ram_resource_model(self):
        ram = RAMResource(
            capacity=34359738368, available=17179869184, bandwidth=45.0,
            total_gb=32.0, available_gb=16.0, free_gb=12.0,
        )
        assert ram.total_gb == 32.0
        assert ram.available_gb == 16.0
        assert ram.bandwidth == 45.0

    def test_gpu_resource_model(self):
        gpu = GPUResource(
            capacity=25769803776, available=24000000000, bandwidth=31.5,
            vendor="nvidia", model="GeForce RTX 4090", vram_total_mb=24576,
            backend_hint="cuda", is_integrated=False,
        )
        assert gpu.vendor == "nvidia"
        assert gpu.backend_hint == "cuda"
        assert not gpu.is_integrated

    def test_system_resources_serializable(self):
        sys_res = SystemResources(
            cpus=[CPUResource(brand="Test CPU")],
            ram=RAMResource(total_gb=16.0),
            gpus=[GPUResource(model="Test GPU")],
        )
        d = sys_res.to_dict()
        assert "cpu" in d
        assert "ram" in d
        assert "gpus" in d
        assert "storage" in d

        json_str = sys_res.to_json()
        assert "Test CPU" in json_str


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
        assert "aes" not in exts

    def test_survives_missing_cpuinfo_library(self):
        with patch.dict("sys.modules", {"cpuinfo": None}):
            from profiler import cpu_profiler
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


# ---------------------------------------------------------------------------
# Specific Hardware Configurations
# ---------------------------------------------------------------------------

class TestConfigurations:
    def test_cpu_only_configuration(self):
        with (
            patch.dict("sys.modules", {"pynvml": None, "GPUtil": None, "amdsmi": None}),
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch("platform.system", return_value="Linux"),
        ):
            sys_res = get_system_resources()
            assert sys_res.gpus == []
            assert sys_res.igpus == []
            assert sys_res.inference_hints["recommended_backend"] == "cpu"
            assert sys_res.inference_hints["max_gpu_layers"] == 0

    def test_cpu_plus_nvidia_configuration(self):
        mock_output = " 0, NVIDIA GeForce RTX 4090, GPU-123, 24576, 23000, 1576, 535.10, 8.9, 5\n"
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = mock_output

        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("subprocess.run", return_value=mock_res),
        ):
            sys_res = get_system_resources()
            assert len(sys_res.gpus) >= 1
            gpu = sys_res.gpus[0]
            assert gpu.vendor == "nvidia"
            assert gpu.model == "NVIDIA GeForce RTX 4090"
            assert gpu.backend_hint == "cuda"
            assert not gpu.is_integrated
            assert sys_res.inference_hints["recommended_backend"] == "cuda"

    def test_cpu_plus_amd_configuration(self):
        mock_json = json.dumps({
            "card0": {
                "Card Series": "AMD Radeon RX 7900 XTX",
                "Driver version": "6.0.5",
                "VRAM Total Memory (B)": "24576",
                "VRAM Total Used Memory (B)": "512",
            }
        })
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = mock_json

        with (
            patch.dict("sys.modules", {"pynvml": None, "amdsmi": None}),
            patch("subprocess.run", return_value=mock_res),
        ):
            sys_res = get_system_resources()
            assert len(sys_res.gpus) >= 1
            gpu = sys_res.gpus[0]
            assert gpu.vendor == "amd"
            assert "7900" in gpu.model
            assert gpu.backend_hint == "rocm"

    def test_laptop_with_igpu_configuration(self):
        mock_json = json.dumps({
            "SPDisplaysDataType": [
                {
                    "sppci_model": "Apple M2 Pro",
                    "spdisplays_vram": "16 GB",
                    "sppci_vendor": "Apple",
                }
            ]
        })
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = mock_json

        with (
            patch("platform.system", return_value="Darwin"),
            patch("subprocess.run", return_value=mock_res),
        ):
            sys_res = get_system_resources()
            assert len(sys_res.igpus) >= 1
            igpu = sys_res.igpus[0]
            assert igpu.is_integrated
            assert igpu.vendor == "apple"
            assert igpu.backend_hint == "metal"

    def test_missing_gpu_drivers(self):
        with (
            patch.dict("sys.modules", {"pynvml": None, "GPUtil": None, "amdsmi": None, "Metal": None}),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            sys_res = get_system_resources()
            assert sys_res.gpus == []
            assert sys_res.igpus == []

    def test_low_memory_environment(self):
        mock_vm = MagicMock()
        mock_vm.total = 4 * 1024 * 1024 * 1024
        mock_vm.available = 1 * 1024 * 1024 * 1024
        mock_vm.free = 500 * 1024 * 1024
        mock_vm.used = 3 * 1024 * 1024 * 1024
        mock_vm.percent = 75.0

        with patch("psutil.virtual_memory", return_value=mock_vm):
            sys_res = get_system_resources()
            assert sys_res.ram.total_gb == 4.0
            assert sys_res.ram.available_gb == 1.0


# ---------------------------------------------------------------------------
# Public API & Storage Profiler tests
# ---------------------------------------------------------------------------

class TestPublicAPIs:
    def test_get_system_resources_api(self):
        res = getSystemResources()
        assert isinstance(res, SystemResources)
        assert len(res.cpus) > 0
        assert res.ram.capacity > 0

    def test_get_available_memory_api(self):
        mem = getAvailableMemory()
        assert "total_gb" in mem
        assert "available_gb" in mem

    def test_get_gpus_api(self):
        gpus = getGPUs()
        assert isinstance(gpus, list)

    def test_get_cpus_api(self):
        cpus = getCPUs()
        assert isinstance(cpus, list)
        assert len(cpus) == 1
        assert "physical_cores" in cpus[0]

    def test_estimate_bandwidth_api(self):
        bw = estimateBandwidth()
        assert "ram_bandwidth_gbps" in bw
        assert "max_gpu_bandwidth_gbps" in bw
        assert "max_storage_bandwidth_gbps" in bw

    def test_run_profiler_json_summary_structure(self):
        summary = run_profiler()
        assert "cpu" in summary
        assert "ram" in summary
        assert "gpus" in summary
        assert "igpus" in summary
        assert "storage" in summary
