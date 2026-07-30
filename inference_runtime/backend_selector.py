"""
backend_selector.py
--------------------
Detects the appropriate llama.cpp execution backend from the hardware
profile and translates a PlacementPlan into backend-specific flags.

Supports:
  - CUDA   (NVIDIA GPU, requires CUDA-enabled llama.exe build)
  - Vulkan (AMD/Intel/NVIDIA GPU, default for this project)
  - Metal  (Apple Silicon, macOS only)
  - CPU    (fallback, no GPU offload)

Backend selection priority:
  1. RuntimeConfig.force_backend (explicit override)
  2. hardware_profile.inference_hints.recommended_backend
  3. GPU vendor heuristic (nvidia → cuda, amd/intel → vulkan, apple → metal)
  4. CPU fallback
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from layer_placement.placement_plan import PlacementDevice, PlacementPlan


# ---------------------------------------------------------------------------
# BackendInfo
# ---------------------------------------------------------------------------

@dataclass
class BackendInfo:
    """
    Resolved backend configuration ready for CLI argument generation.

    Attributes
    ----------
    name : str
        Backend identifier: ``"cuda"``, ``"vulkan"``, ``"metal"``, ``"cpu"``.
    n_gpu_layers : int
        Number of transformer layers to offload to GPU, derived from the
        placement plan's ``llama_cpp_n_gpu_layers_hint``.
    gpu_index : int
        Primary GPU device index for single-GPU setups. Default 0.
    split_mode : str
        llama.cpp split mode flag value. ``"none"`` for single-segment plans,
        ``"layer"`` for multi-segment (boundary crossings > 0) plans.
    extra_flags : List[str]
        Additional raw CLI flags appended to the argument list, e.g.
        ``["--tensor-split", "0.5,0.5"]`` for multi-GPU splits.
    supports_flash_attn : bool
        Whether this backend/build supports ``--flash-attn``. Vulkan and
        CUDA builds generally do; CPU-only does not.
    env_vars : Dict[str, str]
        Environment variable overrides for the subprocess (e.g.
        ``{"CUDA_VISIBLE_DEVICES": "0"}``).
    """
    name: str
    n_gpu_layers: int
    gpu_index: int = 0
    split_mode: str = "none"
    extra_flags: List[str] = field(default_factory=list)
    supports_flash_attn: bool = True
    env_vars: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_gpu_layers": self.n_gpu_layers,
            "gpu_index": self.gpu_index,
            "split_mode": self.split_mode,
            "extra_flags": self.extra_flags,
            "supports_flash_attn": self.supports_flash_attn,
        }


# ---------------------------------------------------------------------------
# BackendSelector
# ---------------------------------------------------------------------------

class BackendSelector:
    """
    Resolves the optimal llama.cpp execution backend for the current hardware.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile from ``hardware_profile.json``.
    force_backend : str, optional
        Override all heuristics. One of ``"cuda"``, ``"vulkan"``,
        ``"metal"``, ``"cpu"``.
    """

    # Maps vendor strings to preferred backend
    _VENDOR_BACKEND_MAP: Dict[str, str] = {
        "nvidia": "cuda",
        "amd": "vulkan",
        "intel": "vulkan",
        "apple": "metal",
    }

    def __init__(
        self,
        hw_profile: Dict[str, Any],
        force_backend: Optional[str] = None,
    ) -> None:
        self.hw_profile = hw_profile
        self.force_backend = force_backend

    def detect_backend(self, plan: PlacementPlan) -> BackendInfo:
        """
        Determine the execution backend and translate plan parameters into
        a complete :class:`BackendInfo` configuration.

        Parameters
        ----------
        plan : PlacementPlan
            The placement plan from Phase 3 to translate.

        Returns
        -------
        BackendInfo
            Resolved backend with GPU layer count, split mode, and flags.
        """
        backend_name = self._resolve_backend_name()
        n_gpu_layers = plan.llama_cpp_n_gpu_layers_hint
        gpu_index = self._get_primary_gpu_index()

        # Split mode: use "layer" when there are boundary crossings
        # This enables llama.cpp's tensor-split layer mode which allows
        # non-contiguous GPU/CPU layer interleaving.
        split_mode = "layer" if plan.boundary_crossings > 0 else "none"

        extra_flags: List[str] = []
        env_vars: Dict[str, str] = {}
        supports_flash_attn = True

        if backend_name == "cuda":
            # For CUDA, set device visibility
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            if plan.boundary_crossings > 0:
                # Multi-segment: hint to CUDA to use async copies
                extra_flags.append("--override-kv")
                extra_flags.append("general.split_mode=layer")

        elif backend_name == "vulkan":
            # Vulkan: specify the Vulkan device index (e.g. Vulkan1 for discrete GPU)
            vk_dev_name = f"Vulkan{gpu_index}" if isinstance(gpu_index, int) or str(gpu_index).isdigit() else str(gpu_index)
            if gpu_index > 0 or "Vulkan" in str(gpu_index):
                extra_flags.extend(["--device", vk_dev_name])

            # Phase 7: if iGPU layers exist in the plan, add the iGPU device index
            # so llama.cpp can use both the dGPU and iGPU via multi-device Vulkan.
            if getattr(plan, "n_igpu_layers", 0) > 0:
                igpu_vk_idx = self._get_igpu_vulkan_index()
                if igpu_vk_idx >= 0 and igpu_vk_idx != gpu_index:
                    igpu_vk_name = f"Vulkan{igpu_vk_idx}" if isinstance(igpu_vk_idx, int) or str(igpu_vk_idx).isdigit() else str(igpu_vk_idx)
                    # Replace the single-device flag with a comma-separated device list
                    if extra_flags and extra_flags[-1] == vk_dev_name:
                        extra_flags.pop()  # remove device index value
                        extra_flags.pop()  # remove "--device" key
                    extra_flags.extend(["--device", f"{vk_dev_name},{igpu_vk_name}"])

        elif backend_name == "metal":
            # Metal: Apple Silicon unified memory, always supports flash attn
            supports_flash_attn = True

        elif backend_name == "cpu":
            # CPU only: disable GPU offload
            n_gpu_layers = 0
            split_mode = "none"
            supports_flash_attn = False

        return BackendInfo(
            name=backend_name,
            n_gpu_layers=n_gpu_layers,
            gpu_index=gpu_index,
            split_mode=split_mode,
            extra_flags=extra_flags,
            supports_flash_attn=supports_flash_attn,
            env_vars=env_vars,
        )

    detect = detect_backend  # snake_case alias

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _resolve_backend_name(self) -> str:
        """Determine backend name via override → hints → vendor heuristic."""
        # 1. Explicit override from RuntimeConfig
        if self.force_backend:
            return self.force_backend.lower()

        # 2. Hardware profile inference_hints
        hints = self.hw_profile.get("inference_hints", {})
        recommended = hints.get("recommended_backend", "").lower()
        if recommended and recommended in ("cuda", "vulkan", "metal", "cpu"):
            return recommended

        # 3. Vendor heuristic from discrete GPUs
        gpus = self.hw_profile.get("gpus", [])
        if gpus:
            vendor = str(gpus[0].get("vendor", "")).lower()
            if vendor in self._VENDOR_BACKEND_MAP:
                return self._VENDOR_BACKEND_MAP[vendor]

        # 4. iGPU fallback
        igpus = self.hw_profile.get("igpus", [])
        if igpus:
            vendor = str(igpus[0].get("vendor", "")).lower()
            if vendor in self._VENDOR_BACKEND_MAP:
                return self._VENDOR_BACKEND_MAP[vendor]

        return "cpu"

    def _get_primary_gpu_index(self) -> int:
        """Get the primary GPU device index from the hardware profile."""
        hints = self.hw_profile.get("inference_hints", {})
        # Phase 1 profiler populates this key
        idx = hints.get("primary_gpu_index", None)
        if idx is not None:
            return int(idx)

        gpus = self.hw_profile.get("gpus", [])
        if gpus:
            return int(gpus[0].get("global_index", 0))
        return 0

    def _get_igpu_vulkan_index(self) -> int:
        """
        Return the Vulkan logical device index for the first iGPU.

        Phase 7: iGPUs are listed under ``hw_profile["igpus"]``. Their
        ``global_index`` maps to the Vulkan device enumeration order produced
        by the Phase 1 GPU profiler.

        Returns -1 when no iGPU is present.
        """
        igpus = self.hw_profile.get("igpus", [])
        if not igpus:
            return -1
        # Prefer the iGPU with the most VRAM
        best = max(igpus, key=lambda g: g.get("vram_total_mb", 0))
        return int(best.get("global_index", -1))


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def detect_backend(
    hw_profile: Dict[str, Any],
    plan: PlacementPlan,
    force_backend: Optional[str] = None,
) -> BackendInfo:
    """
    Convenience wrapper around :class:`BackendSelector`.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile dictionary.
    plan : PlacementPlan
        Phase 3 placement plan.
    force_backend : str, optional
        Override backend name.

    Returns
    -------
    BackendInfo
        Resolved backend configuration.
    """
    return BackendSelector(hw_profile, force_backend).detect_backend(plan)
