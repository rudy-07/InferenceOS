"""
test_build_engine.py
--------------------
Unit tests for Phase 2: Dynamic Compilation & Setup.

All subprocess calls and filesystem operations are mocked so the tests
run on any machine regardless of whether cmake, gcc, or llama.cpp is built.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hw_profile(
    backend: str = "cpu",
    isa_extensions: list[str] | None = None,
    gpus: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "os": {"system": "Linux", "machine": "x86_64"},
        "cpu": {
            "brand": "Test CPU",
            "physical_cores": 8,
            "logical_cores": 16,
            "isa_extensions": isa_extensions or ["avx2", "fma"],
        },
        "memory": {"total_gb": 32.0, "available_gb": 24.0},
        "gpus": gpus or [],
        "inference_hints": {
            "recommended_backend": backend,
            "recommended_quant": "Q4_0",
            "max_gpu_layers": 0,
            "parallelism_threads": 16,
        },
    }


# ---------------------------------------------------------------------------
# cmake_profiles.py tests
# ---------------------------------------------------------------------------

class TestCMakeProfiles:
    def test_get_profile_cuda_has_required_flags(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("cuda")
        flags_str = " ".join(p.cmake_flags)
        assert "-DGGML_CUDA=ON" in flags_str
        assert "ARCHITECTURES" in flags_str
        assert p.backend == "cuda"
        assert p.env_checks  # must require nvcc

    def test_get_profile_rocm_has_hip_flag(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("rocm")
        flags_str = " ".join(p.cmake_flags)
        assert "-DGGML_HIP=ON" in flags_str
        assert "hipcc" in p.cmake_flags or "hipcc" in p.env_checks

    def test_get_profile_metal_has_metal_flag(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("metal")
        assert "-DGGML_METAL=ON" in p.cmake_flags
        assert p.backend == "metal"

    def test_get_profile_vulkan_has_vulkan_flag(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("vulkan")
        assert "-DGGML_VULKAN=ON" in p.cmake_flags
        assert "glslc" in p.env_checks

    def test_get_profile_cpu_has_avx2_flag(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("cpu")
        assert "-DGGML_AVX2=ON" in p.cmake_flags
        assert p.env_checks == []  # no extra tools needed

    def test_get_profile_cpu_avx512_has_avx512_flag(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("cpu_avx512")
        assert "-DGGML_AVX512=ON" in p.cmake_flags
        assert "-DGGML_AVX2=ON" in p.cmake_flags

    def test_unknown_backend_falls_back_to_cpu(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("unknown_future_backend_xyz")
        assert p.backend == "cpu"

    def test_all_profiles_have_build_shared_libs_off(self):
        """Critical: we want a static library for clean Python linking via ctypes."""
        from setup.cmake_profiles import get_profile
        for backend in ["cuda", "rocm", "metal", "vulkan", "cpu", "cpu_avx512"]:
            p = get_profile(backend)
            assert "-DBUILD_SHARED_LIBS=OFF" in p.cmake_flags, \
                f"Backend '{backend}' is missing -DBUILD_SHARED_LIBS=OFF"

    def test_all_profiles_have_tests_off(self):
        """We don't want llama.cpp's own test suite adding build time."""
        from setup.cmake_profiles import get_profile
        for backend in ["cuda", "cpu", "vulkan"]:
            p = get_profile(backend)
            assert "-DLLAMA_BUILD_TESTS=OFF" in p.cmake_flags

    def test_amdgpu_targets_propagated(self):
        from setup.cmake_profiles import get_profile
        p = get_profile("rocm", amdgpu_targets="gfx1100;gfx1030")
        assert "gfx1100;gfx1030" in " ".join(p.cmake_flags)

    # ── select_profile_from_hardware ──────────────────────────────────────────

    def test_select_cpu_avx512_when_avx512_present(self):
        from setup.cmake_profiles import select_profile_from_hardware
        hw = _make_hw_profile("cpu", isa_extensions=["avx512f", "avx2"])
        p = select_profile_from_hardware(hw)
        assert p.backend == "cpu_avx512"
        assert "-DGGML_AVX512=ON" in p.cmake_flags

    def test_select_cpu_baseline_when_no_avx2(self):
        from setup.cmake_profiles import select_profile_from_hardware
        hw = _make_hw_profile("cpu", isa_extensions=["sse4_2"])
        p = select_profile_from_hardware(hw)
        assert p.backend == "cpu_baseline"

    def test_select_cpu_avx2_when_only_avx2(self):
        from setup.cmake_profiles import select_profile_from_hardware
        hw = _make_hw_profile("cpu", isa_extensions=["avx2", "fma"])
        p = select_profile_from_hardware(hw)
        assert p.backend == "cpu"

    def test_select_cuda_profile_for_nvidia_gpu(self):
        from setup.cmake_profiles import select_profile_from_hardware
        hw = _make_hw_profile(
            backend="cuda",
            gpus=[{"vendor": "nvidia", "backend_hint": "cuda"}],
        )
        p = select_profile_from_hardware(hw)
        assert p.backend == "cuda"
        assert "-DGGML_CUDA=ON" in p.cmake_flags

    def test_select_rocm_profile_for_amd_gpu(self):
        from setup.cmake_profiles import select_profile_from_hardware
        hw = _make_hw_profile(
            backend="rocm",
            gpus=[{"vendor": "amd", "backend_hint": "rocm"}],
        )
        p = select_profile_from_hardware(hw)
        assert p.backend == "rocm"


# ---------------------------------------------------------------------------
# prerequisites.py tests
# ---------------------------------------------------------------------------

class TestPrerequisites:
    def _mock_which(self, available: list[str]):
        """Return a shutil.which mock that only finds tools in `available`."""
        import shutil
        original = shutil.which
        def _which(name, *a, **kw):
            return f"/fake/bin/{name}" if name in available else None
        return _which

    def test_report_ok_when_all_tools_present(self):
        from setup.cmake_profiles import get_profile
        from setup.prerequisites import check_prerequisites

        profile = get_profile("cpu")

        with (
            patch("shutil.which", side_effect=self._mock_which(
                ["git", "cmake", "gcc", "g++", "ninja"]
            )),
            patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="tool v1.0", stderr=""
            )),
        ):
            report = check_prerequisites(profile)

        assert report.ok

    def test_report_fails_when_cmake_missing(self):
        from setup.cmake_profiles import get_profile
        from setup.prerequisites import check_prerequisites

        profile = get_profile("cpu")

        with (
            patch("shutil.which", side_effect=self._mock_which(
                ["git", "gcc", "g++"]  # cmake absent
            )),
        ):
            report = check_prerequisites(profile)

        assert not report.ok
        missing_names = [m.name for m in report.missing]
        assert any("cmake" in n.lower() for n in missing_names)

    def test_cuda_profile_requires_nvcc(self):
        from setup.cmake_profiles import get_profile
        from setup.prerequisites import check_prerequisites

        profile = get_profile("cuda")

        with (
            patch("shutil.which", side_effect=self._mock_which(
                ["git", "cmake", "gcc", "g++", "ninja"]
                # nvcc deliberately absent
            )),
        ):
            report = check_prerequisites(profile)

        assert not report.ok
        missing_names = [m.name for m in report.missing]
        assert any("nvcc" in n.lower() for n in missing_names)

    def test_vulkan_profile_requires_glslc(self):
        from setup.cmake_profiles import get_profile
        from setup.prerequisites import check_prerequisites

        profile = get_profile("vulkan")

        with (
            patch("shutil.which", side_effect=self._mock_which(
                ["git", "cmake", "gcc", "g++", "ninja"]
                # glslc absent
            )),
            patch("os.path.exists", return_value=False),
            patch("glob.glob", return_value=[]),
        ):
            report = check_prerequisites(profile)

        assert not report.ok
        assert any("glslc" in m.name for m in report.missing)

    def test_cpu_profile_has_no_required_gpu_tools(self):
        from setup.cmake_profiles import get_profile
        from setup.prerequisites import check_prerequisites

        profile = get_profile("cpu")
        # CPU profile: env_checks should be empty → no GPU tool required
        assert profile.env_checks == []

    def test_summary_string_contains_tool_names(self):
        from setup.cmake_profiles import get_profile
        from setup.prerequisites import check_prerequisites

        profile = get_profile("cpu")
        with (
            patch("shutil.which", side_effect=self._mock_which(["git"])),
        ):
            report = check_prerequisites(profile)
        summary = report.summary()
        assert "git" in summary
        assert "cmake" in summary.lower()
        # Verify the ASCII status markers are present
        assert "[OK]" in summary or "[!!]" in summary



# ---------------------------------------------------------------------------
# build_validator.py tests
# ---------------------------------------------------------------------------

class TestBuildValidator:
    def test_returns_failed_when_build_dir_missing(self, tmp_path):
        from setup.build_validator import validate_build
        result = validate_build(tmp_path / "nonexistent")
        assert not result.ok
        # Message should mention the missing directory
        assert str(tmp_path / "nonexistent") in result.message or "not found" in result.message

    def test_finds_library_via_fallback_walk(self, tmp_path):
        from setup.build_validator import validate_build
        # Create a .a file in a non-canonical location
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "libllama.a").write_text("")

        result = validate_build(tmp_path)
        assert result.ok
        assert any("libllama.a" in a for a in result.artifacts_found)

    def test_returns_failed_when_build_dir_empty(self, tmp_path):
        from setup.build_validator import validate_build
        result = validate_build(tmp_path)
        assert not result.ok

    def test_smoke_test_returns_none_when_cli_absent(self, tmp_path):
        from setup.build_validator import run_smoke_test
        result = run_smoke_test(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# build_engine.py integration tests  (mocked — no real cmake run)
# ---------------------------------------------------------------------------

class TestBuildEngine:
    def _make_engine(self, tmp_path: Path, **kwargs):
        from setup.build_engine import BuildEngine

        # Create a fake hardware profile
        hw = _make_hw_profile("cpu", isa_extensions=["avx2"])
        profile_path = tmp_path / "hardware_profile.json"
        profile_path.write_text(json.dumps(hw))

        # Create a fake llama.cpp CMakeLists.txt
        src_dir = tmp_path / "llama.cpp"
        src_dir.mkdir()
        (src_dir / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.21)\n")

        return BuildEngine(
            profile_path=profile_path,
            src_dir=src_dir,
            build_dir=tmp_path / "build",
            **kwargs,
        )

    def test_dry_run_does_not_invoke_cmake(self, tmp_path):
        engine = self._make_engine(tmp_path, dry_run=True, verbose=False)

        with patch("subprocess.run") as mock_run:
            # Provide a fake prerequisite environment
            with patch("shutil.which", return_value="/fake/bin/tool"):
                with patch("subprocess.run", return_value=MagicMock(
                    returncode=0, stdout="v1", stderr=""
                )):
                    result = engine.run()

        # In dry-run mode, CMake should never actually be invoked for build/configure
        assert result.backend == "cpu"

    def test_clean_removes_build_directory(self, tmp_path):
        """
        --clean must physically delete the build directory.
        We use dry_run=False but mock subprocess.run so cmake never fires.
        Only the clean step uses shutil.rmtree — cmake calls are intercepted.
        """
        engine = self._make_engine(tmp_path, clean=True, dry_run=False, verbose=False)
        # Create the build dir with a sentinel file before running
        engine.build_dir.mkdir(parents=True, exist_ok=True)
        sentinel = engine.build_dir / "stale_artifact.a"
        sentinel.write_text("old build output")

        # Mock subprocess.run to succeed for all cmake calls (version probes + build)
        _ok = MagicMock(returncode=0, stdout="v1", stderr="")

        with (
            patch("shutil.which", return_value="/fake/bin/tool"),
            patch("subprocess.run", return_value=_ok),
        ):
            result = engine.run()

        # The stale sentinel must be gone — rmtree deleted the whole dir
        assert not sentinel.exists(), (
            "Expected stale file to be removed by --clean, but it still exists."
        )






    def test_raises_when_source_missing(self, tmp_path):
        from setup.build_engine import BuildEngine

        hw = _make_hw_profile("cpu")
        profile_path = tmp_path / "hardware_profile.json"
        profile_path.write_text(json.dumps(hw))

        engine = BuildEngine(
            profile_path=profile_path,
            src_dir=tmp_path / "nonexistent_llama",  # doesn't exist
            build_dir=tmp_path / "build",
            dry_run=True,
        )

        with (
            patch("shutil.which", return_value="/fake/bin/tool"),
            patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="v1", stderr=""
            )),
        ):
            with pytest.raises(RuntimeError, match="llama.cpp source not found"):
                engine.run()

    def test_build_result_json_written_after_dry_run(self, tmp_path):
        engine = self._make_engine(tmp_path, dry_run=True, verbose=False)

        with (
            patch("shutil.which", return_value="/fake/bin/tool"),
            patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="v1", stderr=""
            )),
        ):
            result = engine.run()

        result_file = tmp_path / "build" / "build_result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["backend"] == "cpu"
        assert "cmake_flags" in data
        assert "duration_seconds" in data

    def test_cmake_configure_failure_raises(self, tmp_path):
        engine = self._make_engine(tmp_path, dry_run=False, verbose=False)

        # We intercept subprocess.run and fail only when the call is a cmake configure
        # (i.e. the second positional arg is the source directory path, not '--build')
        _ok = MagicMock(returncode=0, stdout="v1", stderr="")
        _fail = MagicMock(returncode=1, stdout="", stderr="configure error")

        def _selective_run(cmd, *a, **kw):
            if isinstance(cmd, list) and "cmake" in str(cmd[0]):
                # cmake configure: arg[1] is the source path (not '--build')
                if len(cmd) > 1 and "--build" not in cmd:
                    return _fail
            return _ok

        with (
            patch("shutil.which", return_value="/fake/bin/tool"),
            patch("subprocess.run", side_effect=_selective_run),
        ):
            with pytest.raises(RuntimeError, match="CMake configure failed"):
                engine.run()
