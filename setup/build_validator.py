"""
build_validator.py
------------------
Post-build validation: confirms that the expected library/binary artefact
was produced and optionally runs a quick smoke-test via llama-cli.

This is intentionally separate from build_engine.py so it can be run
independently (e.g. in CI after a pre-built artefact is downloaded).
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ValidationResult:
    ok: bool
    artifacts_found: list[str]
    artifacts_missing: list[str]
    smoke_test_output: str = ""
    smoke_test_passed: Optional[bool] = None
    message: str = ""


def _expected_artifacts(build_dir: Path) -> list[Path]:
    """
    Return the list of file paths we expect CMake to produce.
    The exact name depends on the OS and whether we built a static library
    or an executable.
    """
    system = platform.system()
    candidates: list[Path] = []

    if system == "Windows":
        # Static lib (MinGW) + CLI tool (llama-app is the new name when server is disabled)
        candidates += [
            build_dir / "src" / "libllama.a",
            build_dir / "ggml" / "src" / "libggml.a",
            build_dir / "bin" / "llama-app.exe",
            build_dir / "bin" / "llama-cli.exe",   # fallback older name
        ]
    elif system == "Darwin":
        candidates += [
            build_dir / "src" / "libllama.a",
            build_dir / "ggml" / "src" / "libggml.a",
            build_dir / "bin" / "llama-app",
            build_dir / "bin" / "llama-cli",
        ]
    else:  # Linux
        candidates += [
            build_dir / "src" / "libllama.a",
            build_dir / "ggml" / "src" / "libggml.a",
            build_dir / "bin" / "llama-app",
            build_dir / "bin" / "llama-cli",
        ]

    return candidates


def _find_any_library(build_dir: Path) -> list[Path]:
    """
    Walk the build directory for any .a / .lib / .so / .dylib files.
    Used as a fallback when the canonical paths don't exist (CMake
    version differences can change output layout).
    """
    found: list[Path] = []
    extensions = {".a", ".lib", ".so", ".dylib", ".dll"}
    try:
        for root, _, files in os.walk(build_dir):
            for f in files:
                if Path(f).suffix.lower() in extensions:
                    found.append(Path(root) / f)
    except Exception:
        pass
    return found


def validate_build(build_dir: Path) -> ValidationResult:
    """
    Check that the build produced its expected artefacts.

    Parameters
    ----------
    build_dir : Path
        The CMake binary directory (i.e. where `cmake --build` was run).

    Returns
    -------
    ValidationResult
    """
    if not build_dir.exists():
        return ValidationResult(
            ok=False,
            artifacts_found=[],
            artifacts_missing=["<build directory does not exist>"],
            message=f"Build directory not found: {build_dir}",
        )

    expected = _expected_artifacts(build_dir)
    found = [str(p) for p in expected if p.exists()]
    missing = [str(p) for p in expected if not p.exists()]

    # Fallback: even if canonical paths changed, verify at least one library exists
    if not found:
        fallback_libs = _find_any_library(build_dir)
        if fallback_libs:
            found = [str(p) for p in fallback_libs]
            missing = []

    ok = len(found) > 0

    return ValidationResult(
        ok=ok,
        artifacts_found=found,
        artifacts_missing=missing,
        message=(
            f"Build validated: {len(found)} artefact(s) found."
            if ok
            else f"Build validation failed: expected artefacts not found in {build_dir}"
        ),
    )


def run_smoke_test(build_dir: Path, timeout: int = 10) -> Optional[str]:
    """
    Run `llama-cli --help` to confirm the binary is executable.
    Returns the output string on success, None on failure.
    """
    system = platform.system()
    cli_name = "llama-cli.exe" if system == "Windows" else "llama-cli"
    cli_path = build_dir / "bin" / cli_name

    if not cli_path.exists():
        return None

    try:
        result = subprocess.run(
            [str(cli_path), "--help"],
            capture_output=True, text=True, timeout=timeout,
        )
        # llama-cli --help exits with 0 or 1 depending on version
        if result.stdout or result.stderr:
            return (result.stdout or result.stderr)[:500]
    except (subprocess.TimeoutExpired, OSError):
        pass

    return None
