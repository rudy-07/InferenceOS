"""
build_engine.py
---------------
Main build orchestrator for InferenceOS Phase 2.

Reads hardware_profile.json, selects the optimal CMake profile, checks
prerequisites, configures the build, and compiles llama.cpp — all from a
single Python entry point.

Usage (CLI)
-----------
    # Standard run (auto-detect everything)
    python -m setup.build_engine

    # Force a specific backend regardless of hardware profile
    python -m setup.build_engine --backend cuda

    # Dry-run: print the CMake commands without executing them
    python -m setup.build_engine --dry-run --verbose

    # Clean previous build before recompiling
    python -m setup.build_engine --clean

Usage (API)
-----------
    from setup.build_engine import BuildEngine
    engine = BuildEngine()
    result = engine.run()

Design decisions
----------------
* The engine runs cmake configure + cmake build as two separate subprocess
  calls so that configure failures are distinguishable from compile errors.
* Parallel build jobs (-j N) are set to physical_cores from the hardware
  profile; this keeps the system responsive during long builds.
* Build state is persisted to build/build_result.json so Phase 3 can read
  the exact library paths without re-running the build.
* --clean deletes only the CMake binary dir (build/), never the source.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from .cmake_profiles import CMakeProfile, select_profile_from_hardware
from .prerequisites import PrerequisiteReport, check_prerequisites
from .build_validator import validate_build, run_smoke_test


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
LLAMA_SRC_DIR = PROJECT_ROOT / "llama.cpp"
BUILD_DIR = PROJECT_ROOT / "build"
PROFILE_JSON = PROJECT_ROOT / "hardware_profile.json"
BUILD_RESULT_JSON = BUILD_DIR / "build_result.json"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    success: bool
    backend: str
    profile_description: str
    cmake_flags: list[str]
    generator: str
    build_dir: str
    duration_seconds: float
    artifacts_found: list[str]
    artifacts_missing: list[str]
    error: Optional[str] = None
    smoke_test_passed: Optional[bool] = None


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class _Logger:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def info(self, msg: str) -> None:
        print(f"\033[36m[build]\033[0m {msg}")

    def success(self, msg: str) -> None:
        print(f"\033[32m[build]\033[0m [OK] {msg}")

    def warn(self, msg: str) -> None:
        print(f"\033[33m[build]\033[0m [!!] {msg}")

    def error(self, msg: str) -> None:
        print(f"\033[31m[build]\033[0m [XX] {msg}", file=sys.stderr)

    def debug(self, msg: str) -> None:
        if self.verbose:
            print(f"\033[90m[build]\033[0m     {msg}")

    def section(self, title: str) -> None:
        width = 60
        print(f"\n\033[1m{'-' * width}\033[0m")
        print(f"\033[1m  {title}\033[0m")
        print(f"\033[1m{'-' * width}\033[0m")



# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class BuildEngine:
    """
    Orchestrates the full build pipeline:
      1. Load hardware profile
      2. Select CMake profile
      3. Check prerequisites
      4. Ensure llama.cpp source is present
      5. cmake configure
      6. cmake build
      7. Validate artefacts
      8. Persist build_result.json
    """

    def __init__(
        self,
        *,
        profile_path: Path = PROFILE_JSON,
        src_dir: Path = LLAMA_SRC_DIR,
        build_dir: Path = BUILD_DIR,
        backend_override: Optional[str] = None,
        dry_run: bool = False,
        clean: bool = False,
        verbose: bool = False,
    ) -> None:
        self.profile_path = profile_path
        self.src_dir = src_dir
        self.build_dir = build_dir
        self.backend_override = backend_override
        self.dry_run = dry_run
        self.clean = clean
        self.log = _Logger(verbose=verbose)

    # ── Step 1: Load hardware profile ────────────────────────────────────────

    def _load_hardware_profile(self) -> dict[str, Any]:
        if not self.profile_path.exists():
            self.log.warn(
                f"hardware_profile.json not found at {self.profile_path}. "
                "Running the profiler now…"
            )
            self._run_profiler()

        with open(self.profile_path, encoding="utf-8") as f:
            return json.load(f)

    def _run_profiler(self) -> None:
        """Invoke Phase 1 profiler to generate the profile JSON."""
        result = subprocess.run(
            [sys.executable, "-m", "profiler.hardware_profiler",
             "--output", str(self.profile_path)],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            raise RuntimeError("Hardware profiler failed; cannot continue.")

    # ── Step 2: Select CMake profile ─────────────────────────────────────────

    def _select_profile(self, hw: dict[str, Any]) -> CMakeProfile:
        if self.backend_override:
            from .cmake_profiles import get_profile
            profile = get_profile(self.backend_override)
            self.log.info(f"Backend override: using '{self.backend_override}'")
        else:
            profile = select_profile_from_hardware(hw)
            self.log.info(f"Auto-selected backend: '{profile.backend}'")

        self.log.debug(f"Description: {profile.description}")
        self.log.debug(f"Flags: {' '.join(profile.cmake_flags)}")
        self.log.debug(f"Generator: {profile.generator}")
        return profile

    # ── Step 3: Prerequisites ─────────────────────────────────────────────────

    def _check_prerequisites(self, profile: CMakeProfile) -> PrerequisiteReport:
        report = check_prerequisites(profile)
        print(report.summary())

        if report.warnings:
            for w in report.warnings:
                self.log.warn(f"Optional tool missing: {w.name} — {w.message}")

        if not report.ok:
            for m in report.missing:
                self.log.error(f"Required tool missing: {m.name}")
                self.log.error(f"  {m.message}")
            raise RuntimeError(
                f"Prerequisites not met for backend '{profile.backend}'. "
                "See messages above."
            )

        self.log.success("All required prerequisites satisfied.")
        return report

    # ── Step 4: Ensure llama.cpp source ──────────────────────────────────────

    def _ensure_source(self) -> None:
        if not self.src_dir.exists() or not (self.src_dir / "CMakeLists.txt").exists():
            raise RuntimeError(
                f"llama.cpp source not found at {self.src_dir}.\n"
                "Run: git submodule update --init --recursive"
            )

        # Check for uninitialized submodule (empty directory)
        if not any(self.src_dir.iterdir()):
            self.log.info("Initialising llama.cpp submodule…")
            if not self.dry_run:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=PROJECT_ROOT, check=True,
                )

        self.log.success(f"Source ready: {self.src_dir}")

    # ── Step 5: Clean ─────────────────────────────────────────────────────────

    def _maybe_clean(self) -> None:
        if self.clean and self.build_dir.exists():
            self.log.info(f"Cleaning build directory: {self.build_dir}")
            if not self.dry_run:
                shutil.rmtree(self.build_dir)
                self.log.success("Build directory cleaned.")
            else:
                self.log.warn("[DRY RUN] Skipping clean.")

    # ── Step 6: CMake configure ───────────────────────────────────────────────

    def _cmake_configure(self, profile: CMakeProfile, hw: dict[str, Any]) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "cmake",
            "--fresh",          # wipe stale CMakeCache.txt; preserves .obj files
            str(self.src_dir),
            f"-DCMAKE_BUILD_TYPE={profile.build_type}",
        ]

        # Generator
        generator = profile.generator
        if generator:
            cmd += ["-G", generator]

        # MinGW on Windows: tell CMake to use gcc/g++
        if platform.system() == "Windows" and shutil.which("gcc"):
            cmd += [
                "-DCMAKE_C_COMPILER=gcc",
                "-DCMAKE_CXX_COMPILER=g++",
            ]

        # Backend-specific flags
        cmd += profile.cmake_flags

        self.log.info("CMake configure command:")
        self.log.info("  " + " ".join(cmd))

        if self.dry_run:
            self.log.warn("[DRY RUN] Skipping cmake configure.")
            return

        result = subprocess.run(cmd, cwd=self.build_dir)
        if result.returncode != 0:
            raise RuntimeError(
                f"CMake configure failed (exit {result.returncode}). "
                "Review the output above for errors."
            )
        self.log.success("CMake configure complete.")

    # ── Step 7: CMake build ───────────────────────────────────────────────────

    def _cmake_build(self, hw: dict[str, Any]) -> None:
        # Use physical CPU cores for parallel compilation
        n_jobs = hw.get("cpu", {}).get("physical_cores", os.cpu_count() or 4)

        cmd = [
            "cmake", "--build", str(self.build_dir),
            "--config", "Release",
            "--target", "llama-app",
            "--parallel", str(n_jobs),
        ]

        self.log.info(f"CMake build command (--parallel {n_jobs}):")
        self.log.info("  " + " ".join(cmd))


        if self.dry_run:
            self.log.warn("[DRY RUN] Skipping cmake build.")
            return

        result = subprocess.run(cmd, cwd=self.build_dir)
        if result.returncode != 0:
            raise RuntimeError(
                f"CMake build failed (exit {result.returncode}). "
                "Review the compiler output above."
            )
        self.log.success("CMake build complete.")

    # ── Step 8: Validate and persist ─────────────────────────────────────────

    def _validate_and_persist(
        self,
        profile: CMakeProfile,
        duration: float,
    ) -> BuildResult:
        validation = validate_build(self.build_dir)

        smoke_output = run_smoke_test(self.build_dir)
        smoke_passed = smoke_output is not None if not self.dry_run else None

        if smoke_output:
            self.log.debug(f"Smoke test output:\n{smoke_output[:300]}")

        result = BuildResult(
            success=validation.ok or self.dry_run,
            backend=profile.backend,
            profile_description=profile.description,
            cmake_flags=profile.cmake_flags,
            generator=profile.generator or "auto",
            build_dir=str(self.build_dir),
            duration_seconds=round(duration, 1),
            artifacts_found=validation.artifacts_found,
            artifacts_missing=validation.artifacts_missing,
            smoke_test_passed=smoke_passed,
        )

        # Persist so Phase 3 knows where the library lives
        self.build_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.build_dir / "build_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

        self.log.success(f"Build result written to: {result_path}")
        return result

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> BuildResult:
        start = time.time()

        self.log.section("Phase 2: Dynamic Compilation Engine")

        # 1. Load hardware profile
        self.log.info("Loading hardware profile…")
        hw = self._load_hardware_profile()
        os_info = hw.get("os", {})
        cpu_info = hw.get("cpu", {})
        self.log.info(
            f"System: {os_info.get('system')} | "
            f"CPU: {cpu_info.get('brand')} | "
            f"Backend hint: {hw.get('inference_hints', {}).get('recommended_backend')}"
        )

        # 2. Select profile
        self.log.section("Selecting CMake Profile")
        profile = self._select_profile(hw)

        # 3. Prerequisites
        self.log.section("Checking Prerequisites")
        self._check_prerequisites(profile)

        # 4. Source
        self.log.section("Verifying Source")
        self._ensure_source()

        # 5. Clean (optional)
        self._maybe_clean()

        # 6. Configure
        self.log.section("CMake Configure")
        self._cmake_configure(profile, hw)

        # 7. Build
        self.log.section("CMake Build")
        self._cmake_build(hw)

        # 8. Validate
        self.log.section("Validating Build")
        duration = time.time() - start
        result = self._validate_and_persist(profile, duration)

        if result.success:
            self.log.success(
                f"Build finished in {duration:.1f}s  "
                f"[{profile.backend}] [{profile.description}]"
            )
            if result.artifacts_found:
                for a in result.artifacts_found:
                    self.log.success(f"  Artefact: {a}")
        else:
            self.log.error(
                "Build completed but expected artefacts were not found. "
                "The compile step may have partially succeeded."
            )
            for m in result.artifacts_missing:
                self.log.error(f"  Missing: {m}")

        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="build_engine",
        description=(
            "InferenceOS Build Engine — "
            "reads hardware_profile.json and compiles llama.cpp with the "
            "optimal backend for this machine."
        ),
    )
    p.add_argument(
        "--backend", "-b",
        choices=["cuda", "rocm", "metal", "vulkan", "cpu", "cpu_avx512", "cpu_baseline"],
        default=None,
        help="Override the auto-detected backend.",
    )
    p.add_argument(
        "--profile", "-p",
        default=str(PROFILE_JSON),
        help=f"Path to hardware_profile.json (default: {PROFILE_JSON})",
    )
    p.add_argument(
        "--build-dir",
        default=str(BUILD_DIR),
        help=f"CMake binary directory (default: {BUILD_DIR})",
    )
    p.add_argument(
        "--clean", action="store_true",
        help="Delete the build directory before configuring.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print cmake commands without executing them.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print extra debug information.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    engine = BuildEngine(
        profile_path=Path(args.profile),
        build_dir=Path(args.build_dir),
        backend_override=args.backend,
        dry_run=args.dry_run,
        clean=args.clean,
        verbose=args.verbose,
    )

    try:
        result = engine.run()
        sys.exit(0 if result.success else 1)
    except RuntimeError as exc:
        print(f"\n\033[31m[build]\033[0m Fatal: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[build] Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
