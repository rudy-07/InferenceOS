"""
prerequisites.py
----------------
Checks that all tools required for a given CMake build are available
on the host before attempting compilation.

Each check is independent and guarded — the module never raises; it
collects all findings into a PrerequisiteReport and returns it.

Public API
----------
    report = check_prerequisites(profile)
    if not report.ok:
        for item in report.missing:
            print(item.message)
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmake_profiles import CMakeProfile


@dataclass
class PrerequisiteItem:
    name: str
    found: bool
    version: str = ""
    path: str = ""
    message: str = ""
    required: bool = True


@dataclass
class PrerequisiteReport:
    items: list[PrerequisiteItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only if every *required* item was found."""
        return all(i.found for i in self.items if i.required)

    @property
    def missing(self) -> list[PrerequisiteItem]:
        return [i for i in self.items if not i.found and i.required]

    @property
    def warnings(self) -> list[PrerequisiteItem]:
        return [i for i in self.items if not i.found and not i.required]

    def summary(self) -> str:
        lines = []
        for item in self.items:
            status = "[OK]" if item.found else ("[!!]" if item.required else "[ ?]")
            line = f"  {status} {item.name}"
            if item.version:
                line += f" ({item.version})"
            if item.path:
                line += f"  ->  {item.path}"
            if not item.found and item.message:
                line += f"\n      `-- {item.message}"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual tool checkers
# ---------------------------------------------------------------------------

def _probe_tool(
    name: str,
    *args: str,
    version_flag: str = "--version",
    required: bool = True,
    install_hint: str = "",
) -> PrerequisiteItem:
    """Generic tool probe: check if it's in PATH and capture its version."""
    path = shutil.which(name)
    if not path:
        return PrerequisiteItem(
            name=name,
            found=False,
            required=required,
            message=install_hint or f"'{name}' not found in PATH.",
        )

    version_str = ""
    try:
        result = subprocess.run(
            [path, version_flag],
            capture_output=True, text=True, timeout=5,
        )
        # Use first non-empty line of stdout or stderr
        out = (result.stdout or result.stderr).strip()
        version_str = out.splitlines()[0] if out else ""
    except Exception:
        pass

    return PrerequisiteItem(
        name=name, found=True, version=version_str, path=path, required=required,
    )


def _check_git() -> PrerequisiteItem:
    return _probe_tool(
        "git",
        version_flag="--version",
        required=True,
        install_hint="Install git from https://git-scm.com/",
    )


def _check_cmake() -> PrerequisiteItem:
    return _probe_tool(
        "cmake",
        version_flag="--version",
        required=True,
        install_hint="Install CMake ≥ 3.21 from https://cmake.org/download/",
    )


def _check_ninja() -> PrerequisiteItem:
    return _probe_tool(
        "ninja",
        version_flag="--version",
        required=False,  # optional — CMake falls back to make
        install_hint="Install Ninja for faster builds: https://ninja-build.org/",
    )


def _check_compiler() -> PrerequisiteItem:
    """
    On Windows without MSVC, look for gcc/g++ (MinGW).
    On Linux/macOS, look for gcc or clang.
    Reports a combined 'C/C++ compiler' item.
    """
    sys = platform.system()

    if sys == "Windows":
        candidates = [("gcc", "g++"), ("cl", "cl")]
    else:
        candidates = [("gcc", "g++"), ("clang", "clang++")]

    for c_compiler, cxx_compiler in candidates:
        c_path = shutil.which(c_compiler)
        cxx_path = shutil.which(cxx_compiler)
        if c_path and cxx_path:
            version_str = ""
            try:
                res = subprocess.run(
                    [c_path, "--version"], capture_output=True, text=True, timeout=5,
                )
                version_str = (res.stdout or res.stderr).splitlines()[0].strip()
            except Exception:
                pass
            return PrerequisiteItem(
                name=f"C/C++ compiler ({c_compiler}/{cxx_compiler})",
                found=True,
                version=version_str,
                path=c_path,
                required=True,
            )

    hint = (
        "No C/C++ compiler found.\n"
        "      Windows: Install MinGW-w64 (https://www.mingw-w64.org/) or MSVC.\n"
        "      Linux:   sudo apt install build-essential\n"
        "      macOS:   xcode-select --install"
    )
    return PrerequisiteItem(
        name="C/C++ compiler", found=False, required=True, message=hint,
    )


def _check_make() -> PrerequisiteItem:
    """
    On Windows + MinGW, 'make' may be called 'mingw32-make'.
    We only need this when Ninja is absent.
    """
    for name in ("make", "mingw32-make", "gmake"):
        if shutil.which(name):
            return PrerequisiteItem(
                name=f"make ({name})",
                found=True,
                path=shutil.which(name) or "",
                required=False,
            )
    return PrerequisiteItem(
        name="make", found=False, required=False,
        message="No make tool found; Ninja will be used if available.",
    )


def _check_nvcc() -> PrerequisiteItem:
    return _probe_tool(
        "nvcc",
        version_flag="--version",
        required=True,
        install_hint=(
            "CUDA Toolkit not found. "
            "Install from https://developer.nvidia.com/cuda-downloads"
        ),
    )


def _check_hipcc() -> PrerequisiteItem:
    return _probe_tool(
        "hipcc",
        version_flag="--version",
        required=True,
        install_hint=(
            "ROCm HIP compiler not found. "
            "Install ROCm from https://rocm.docs.amd.com/en/latest/deploy/linux/"
        ),
    )


def _check_glslc() -> PrerequisiteItem:
    import glob
    import os

    # 1. Search standard PATH
    path = shutil.which("glslc")

    # 2. Search VULKAN_SDK env var
    if not path and "VULKAN_SDK" in os.environ:
        candidate = os.path.join(os.environ["VULKAN_SDK"], "Bin", "glslc.exe")
        if os.path.exists(candidate):
            path = candidate
        else:
            candidate = os.path.join(os.environ["VULKAN_SDK"], "bin", "glslc")
            if os.path.exists(candidate):
                path = candidate

    # 3. Search local project setup/vulkan_sdk
    if not path:
        local_candidates = [
            os.path.abspath("setup/vulkan_sdk/Bin/glslc.exe"),
            os.path.abspath("setup/vulkan_sdk/bin/glslc.exe"),
            os.path.abspath("setup/vulkan_sdk/glslc.exe"),
        ]
        for c in local_candidates:
            if os.path.exists(c):
                path = c
                break

    # 4. Search C:\VulkanSDK\*
    if not path:
        matches = glob.glob("C:/VulkanSDK/*/Bin/glslc.exe") + glob.glob("C:/VulkanSDK/*/bin/glslc.exe")
        if matches:
            path = matches[-1]

    if path:
        bin_dir = os.path.dirname(os.path.abspath(path))
        sdk_dir = os.path.dirname(bin_dir)
        os.environ["VULKAN_SDK"] = sdk_dir
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

        version_str = ""
        try:
            res = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
            version_str = (res.stdout or res.stderr).splitlines()[0].strip() if (res.stdout or res.stderr) else ""
        except Exception:
            pass

        return PrerequisiteItem(
            name="glslc",
            found=True,
            version=version_str,
            path=path,
            required=True,
        )

    return PrerequisiteItem(
        name="glslc",
        found=False,
        required=True,
        message=(
            "Vulkan SDK (glslc) not found. "
            "Install from https://vulkan.lunarg.com/sdk/home"
        ),
    )



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Map env_check strings → checker functions
_CHECKER_MAP = {
    "nvcc":       _check_nvcc,
    "nvidia-smi": lambda: _probe_tool("nvidia-smi", version_flag="--version",
                                      required=False,
                                      install_hint="nvidia-smi missing; install NVIDIA drivers."),
    "hipcc":      _check_hipcc,
    "rocm-smi":   lambda: _probe_tool("rocm-smi", version_flag="--version",
                                      required=False,
                                      install_hint="rocm-smi missing; install ROCm."),
    "glslc":      _check_glslc,
}


def check_prerequisites(profile: "CMakeProfile") -> PrerequisiteReport:
    """
    Verify all tools required by the given CMakeProfile.

    Always checks: git, cmake, C/C++ compiler, ninja (optional), make (optional).
    Additionally checks backend-specific tools listed in profile.env_checks.

    Returns a PrerequisiteReport.  Never raises.
    """
    report = PrerequisiteReport()

    # ── Universal tools ──
    report.items.append(_check_git())
    report.items.append(_check_cmake())
    report.items.append(_check_compiler())
    report.items.append(_check_ninja())
    report.items.append(_check_make())

    # ── Backend-specific tools ──
    for tool in profile.env_checks:
        checker = _CHECKER_MAP.get(tool)
        if checker:
            report.items.append(checker())
        else:
            # Generic fallback for unlisted tools
            report.items.append(_probe_tool(tool, required=True))

    return report
