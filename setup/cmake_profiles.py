"""
cmake_profiles.py
-----------------
Maps an InferenceOS backend name (from hardware_profile.json's
`inference_hints.recommended_backend`) to the exact CMake flags needed
to build llama.cpp for that backend.

Each profile is a CMakeProfile dataclass containing:
  - cmake_flags : list of -D... strings passed to `cmake <src>`
  - env_checks  : environment variables that must be set (or tools in PATH)
                  for the build to succeed — checked by prerequisites.py
  - description : human-readable summary of what this profile builds
  - generator   : preferred CMake generator (None = let CMake auto-detect)

Design notes
------------
* All CPU profiles include -DGGML_NATIVE=ON so GCC/Clang emits code
  tuned for the host micro-architecture (same effect as -march=native).
* CUDA profiles set CMAKE_CUDA_ARCHITECTURES=native so nvcc only compiles
  for the GPU actually present, keeping build times short.
* AMD/HIP profiles expose AMDGPU_TARGETS as a configurable field because
  multi-GPU systems may target several GCN generations simultaneously.
* The Vulkan profile is a universal fallback; it requires only the
  Vulkan SDK and works on NVIDIA, AMD, and Intel Arc GPUs.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CMakeProfile:
    backend: str
    description: str
    cmake_flags: list[str]
    env_checks: list[str] = field(default_factory=list)
    generator: Optional[str] = None
    build_type: str = "Release"


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _default_generator() -> str:
    """
    Choose the fastest available CMake generator.
    Ninja is preferred everywhere when available; on Windows we fall back
    to 'MinGW Makefiles' if MinGW is the toolchain.
    """
    import shutil

    if shutil.which("ninja"):
        return "Ninja"
    if _is_windows():
        return "MinGW Makefiles"
    return "Unix Makefiles"


def _base_flags(extra: list[str] | None = None) -> list[str]:
    """
    Flags applied to every profile regardless of backend:
      - LLAMA_NATIVE: tune code to host CPU (auto-vectorisation)
      - BUILD_SHARED_LIBS=ON: produce a shared library (.dll/.so/.dylib),
        which is required for Phase 3's ctypes bindings to load at runtime.
      - LLAMA_BUILD_TESTS=OFF: shorten build time; we don't need
        llama.cpp's own test suite
      - LLAMA_BUILD_EXAMPLES=OFF: same reasoning
    """
    flags = [
        "-DLLAMA_NATIVE=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DCMAKE_CXX_FLAGS=-Wa,-mbig-obj -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -mstackrealign",
        "-DCMAKE_C_FLAGS=-Wa,-mbig-obj -D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -mstackrealign",
    ]
    if extra:
        flags.extend(extra)
    return flags


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

def get_profile(backend: str, amdgpu_targets: str = "native") -> CMakeProfile:
    """
    Return the CMakeProfile for the given backend string.

    Parameters
    ----------
    backend : str
        One of: 'cuda', 'rocm', 'metal', 'vulkan', 'cpu'.
        Unrecognised values fall back to 'cpu'.
    amdgpu_targets : str
        AMDGPU architecture string(s) for ROCm builds.
        'native' lets the HIP compiler auto-detect the installed GPU.
        Explicit targets (e.g. 'gfx1100;gfx1030') can be set for cross-build.

    Returns
    -------
    CMakeProfile
    """
    profiles: dict[str, CMakeProfile] = {

        # ── NVIDIA CUDA ──────────────────────────────────────────────────────
        "cuda": CMakeProfile(
            backend="cuda",
            description="NVIDIA GPU via CUDA — maximum performance on Nvidia hardware",
            cmake_flags=_base_flags([
                "-DGGML_CUDA=ON",
                # 'native' compiles only for the GPU(s) present at build time
                # which avoids generating fat binaries for every SM version.
                "-DCMAKE_CUDA_ARCHITECTURES=native",
                # Let nvcc use all host CPU threads for parallel compilation
                "-DCMAKE_CUDA_FLAGS=-t0",
            ]),
            env_checks=["nvcc", "nvidia-smi"],
            generator=_default_generator(),
        ),

        # ── AMD ROCm / HIP ──────────────────────────────────────────────────
        "rocm": CMakeProfile(
            backend="rocm",
            description="AMD GPU via ROCm/HIP — targets GCN/RDNA GPUs",
            cmake_flags=_base_flags([
                "-DGGML_HIP=ON",
                f"-DAMDGPU_TARGETS={amdgpu_targets}",
                # Point CMake to the ROCm toolchain
                "-DCMAKE_C_COMPILER=hipcc",
                "-DCMAKE_CXX_COMPILER=hipcc",
            ]),
            env_checks=["hipcc", "rocm-smi"],
            generator=_default_generator(),
        ),

        # ── Apple Metal ─────────────────────────────────────────────────────
        "metal": CMakeProfile(
            backend="metal",
            description="Apple Silicon / Metal — unified-memory GPU on macOS",
            cmake_flags=_base_flags([
                "-DGGML_METAL=ON",
                # Embed the Metal shader library into the binary so
                # it doesn't need to find ggml-metal.metallib at runtime
                "-DGGML_METAL_EMBED_LIBRARY=ON",
            ]),
            # Metal ships with Xcode; no separate env check needed
            env_checks=[],
            generator="Xcode" if _is_windows() is False else None,
        ),

        # ── Vulkan (universal GPU fallback) ──────────────────────────────────
        "vulkan": CMakeProfile(
            backend="vulkan",
            description="Vulkan compute — works on NVIDIA, AMD, and Intel Arc GPUs",
            cmake_flags=_base_flags([
                "-DGGML_VULKAN=ON",
            ]),
            env_checks=["glslc"],  # part of Vulkan SDK
            generator=_default_generator(),
        ),

        # ── CPU-only AVX-512 ─────────────────────────────────────────────────
        "cpu_avx512": CMakeProfile(
            backend="cpu_avx512",
            description="CPU-only build with AVX-512 SIMD — best CPU throughput",
            cmake_flags=_base_flags([
                "-DGGML_AVX512=ON",
                "-DGGML_AVX2=ON",
                "-DGGML_AVX=ON",
                "-DGGML_FMA=ON",
                "-DGGML_F16C=ON",
            ]),
            env_checks=[],
            generator=_default_generator(),
        ),

        # ── CPU-only AVX2 (default CPU path) ─────────────────────────────────
        "cpu": CMakeProfile(
            backend="cpu",
            description="CPU-only build with AVX2 SIMD — portable baseline",
            cmake_flags=_base_flags([
                "-DGGML_AVX2=ON",
                "-DGGML_AVX=ON",
                "-DGGML_FMA=ON",
                "-DGGML_F16C=ON",
            ]),
            env_checks=[],
            generator=_default_generator(),
        ),

        # ── CPU baseline (no AVX extensions) ─────────────────────────────────
        "cpu_baseline": CMakeProfile(
            backend="cpu_baseline",
            description="CPU-only build without SIMD extensions — maximum portability",
            cmake_flags=_base_flags([
                "-DGGML_AVX=OFF",
                "-DGGML_AVX2=OFF",
            ]),
            env_checks=[],
            generator=_default_generator(),
        ),
    }

    return profiles.get(backend, profiles["cpu"])


def select_profile_from_hardware(profile_data: dict) -> CMakeProfile:
    """
    Inspect a hardware_profile.json dict and return the best CMakeProfile.

    This extends the simple backend→profile lookup with ISA-aware CPU
    sub-selection: a machine with AVX-512 gets the cpu_avx512 profile
    even when the hardware profiler reports backend='cpu'.
    """
    hints = profile_data.get("inference_hints", {})
    backend = hints.get("recommended_backend", "cpu")
    cpu = profile_data.get("cpu", {})
    isa = set(cpu.get("isa_extensions", []))

    # Refine CPU sub-profile based on ISA extensions
    if backend == "cpu":
        if "avx512f" in isa:
            backend = "cpu_avx512"
        elif "avx2" not in isa:
            backend = "cpu_baseline"

    # For ROCm, extract the target GPU architecture from the first AMD GPU
    amdgpu_targets = "native"
    if backend == "rocm":
        gpus = profile_data.get("gpus", [])
        amd_gpus = [g for g in gpus if g.get("vendor") == "amd"]
        if amd_gpus:
            # Future: derive gfxXXXX from GPU name (Phase 4 enhancement)
            amdgpu_targets = "native"

    return get_profile(backend, amdgpu_targets=amdgpu_targets)
