"""
cuda_stream_manager.py
----------------------
CUDA stream and async-kernel dispatch management for Phase 8.

Scope
-----
Since InferenceOS wraps llama.cpp as a subprocess, we cannot call
``cudaStreamCreate()`` or ``cudaMemcpyAsync()`` directly from Python.

Instead, CudaStreamManager operates at the **env-var level**: it prepares
the set of environment variables that influence how the llama.cpp process
schedules its own CUDA kernels. The subprocess inherits these env-vars
and benefits from them without any C extension required.

Additionally, CudaStreamManager acts as the **interface abstraction** for
the Phase 9 library-mode future: when llama.cpp is eventually embedded as
a shared library, this manager will be upgraded to hold real CUDA stream
handles. Callers use the same API in either case.

Stream model
-----------
Three virtual streams are defined:

  COMPUTE  — transformer GEMM and attention kernels (highest priority)
  TRANSFER — PCIe DMA for weight and KV cache transfers
  SAMPLING — top-k/top-p sampling (lowest priority, can run concurrently)

Each stream maps to a ``StreamStats`` record tracking dispatch counts and
estimated overlap ratio.

Env-var strategy
----------------
For CUDA backends (NVIDIA, CUDA):
  CUDA_LAUNCH_BLOCKING=0       — async kernel launch (default, but explicit)
  GGML_CUDA_FORCE_CUBLAS=1     — prefer cuBLAS over GGML custom kernels
  GGML_CUDA_ENABLE_UNIFIED_MEMORY=0  — disable unified memory (keep tensors on device)
  CUDA_DEVICE_ORDER=FASTEST    — auto-select fastest GPU if multiple present
  OMP_NUM_THREADS=1            — prevent OpenMP from competing with CUDA streams

For Vulkan backends:
  GGML_VK_DISABLE_VALIDATION=1 — skip validation layers in production
  GGML_VK_SHADER_COMPILE_ASYNC=1 — async shader compilation (llama.cpp ≥ 3.x)

For CPU:
  No CUDA env-vars emitted. OMP_NUM_THREADS unchanged.

Overlap estimation
------------------
Since we cannot observe actual CUDA stream overlap, we estimate it from
the placement plan's boundary crossings and PCIe bandwidth:

  overlap_ratio ≈ min(1.0, PCIe_BW / (boundary_crossings × layer_transfer_rate))

This is a coarse heuristic but gives a directionally-correct metric.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from layer_placement.placement_plan import PlacementPlan


# ---------------------------------------------------------------------------
# VirtualStream
# ---------------------------------------------------------------------------

class VirtualStream(Enum):
    """Named virtual execution streams."""
    COMPUTE  = auto()
    TRANSFER = auto()
    SAMPLING = auto()


# ---------------------------------------------------------------------------
# StreamStats
# ---------------------------------------------------------------------------

@dataclass
class StreamStats:
    """
    Per-stream dispatch statistics.

    Attributes
    ----------
    stream : VirtualStream
        Which stream this record describes.
    dispatches : int
        Number of work items issued on this stream since creation.
    estimated_overlap_ratio : float
        Fraction of stream time estimated to overlap with other streams (0.0–1.0).
    total_dispatch_ms : float
        Cumulative time spent dispatching to this stream in milliseconds.
    """
    stream: VirtualStream
    dispatches: int = 0
    estimated_overlap_ratio: float = 0.0
    total_dispatch_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream": self.stream.name,
            "dispatches": self.dispatches,
            "estimated_overlap_ratio": round(self.estimated_overlap_ratio, 4),
            "total_dispatch_ms": round(self.total_dispatch_ms, 2),
        }


# ---------------------------------------------------------------------------
# CudaStreamManager
# ---------------------------------------------------------------------------

class CudaStreamManager:
    """
    Manages CUDA/Vulkan environment hints and virtual stream tracking.

    Parameters
    ----------
    backend : str
        Resolved backend name: ``"cuda"``, ``"vulkan"``, ``"metal"``, ``"cpu"``.
    hw_profile : dict, optional
        Hardware profile for GPU vendor detection.
    plan : PlacementPlan, optional
        Placement plan used for overlap estimation.
    pcie_bandwidth_gbps : float
        PCIe bandwidth in GB/s. Default 16.0 (PCIe 3.0 x16).
    """

    def __init__(
        self,
        backend: str = "cpu",
        hw_profile: Optional[Dict[str, Any]] = None,
        plan: Optional[PlacementPlan] = None,
        pcie_bandwidth_gbps: float = 16.0,
    ) -> None:
        self.backend = backend.lower()
        self.hw_profile = hw_profile or {}
        self.plan = plan
        self.pcie_bandwidth_gbps = pcie_bandwidth_gbps

        # Detect CUDA availability
        self._cuda_available = self._detect_cuda()

        # Stream statistics
        self._stream_stats: Dict[VirtualStream, StreamStats] = {
            s: StreamStats(stream=s) for s in VirtualStream
        }

        # Compute the env-var set at construction time
        self._env_vars: Dict[str, str] = self._build_env_vars()

        # Compute overlap estimate at construction
        self._overlap_estimate = self._estimate_overlap()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def env_vars(self) -> Dict[str, str]:
        """
        Environment variables to inject into the llama.cpp subprocess.

        Returns an empty dict when the backend is CPU or no CUDA is available.
        """
        return dict(self._env_vars)

    @property
    def cuda_available(self) -> bool:
        """True when CUDA is detected on this system."""
        return self._cuda_available

    @property
    def estimated_overlap_ratio(self) -> float:
        """
        Estimated fraction of GPU compute time that overlaps with PCIe transfers.
        A higher value means more effective pipelining.
        """
        return self._overlap_estimate

    def dispatch(
        self,
        stream: VirtualStream,
        work_description: str = "",
    ) -> None:
        """
        Record a dispatch on the given virtual stream.

        In the current subprocess model this is a bookkeeping-only call.
        In a future library-mode implementation this would call
        ``cudaStreamAddCallback()`` or similar.

        Parameters
        ----------
        stream : VirtualStream
            Target stream.
        work_description : str
            Human-readable description for logging.
        """
        t_start = time.perf_counter()
        stats = self._stream_stats[stream]
        stats.dispatches += 1
        elapsed = (time.perf_counter() - t_start) * 1000.0
        stats.total_dispatch_ms += elapsed

    def synchronize(self, stream: Optional[VirtualStream] = None) -> None:
        """
        Wait for the given stream to complete all pending work.

        In the current model this is a no-op (the subprocess handles its
        own synchronization). Included as the future-library-mode stub.
        """
        pass

    def get_stream_stats(self) -> List[StreamStats]:
        """Return per-stream statistics for all virtual streams."""
        return list(self._stream_stats.values())

    def get_all_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "cuda_available": self._cuda_available,
            "estimated_overlap_ratio": round(self._overlap_estimate, 4),
            "pcie_bandwidth_gbps": self.pcie_bandwidth_gbps,
            "env_vars": self._env_vars,
            "streams": [s.to_dict() for s in self._stream_stats.values()],
        }

    def update_plan(self, plan: PlacementPlan) -> None:
        """Update the placement plan and recompute overlap estimate."""
        self.plan = plan
        self._overlap_estimate = self._estimate_overlap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_cuda(self) -> bool:
        """
        Detect CUDA availability by querying nvidia-smi.

        Returns True if nvidia-smi exits with code 0 and reports at least
        one GPU. Falls back to hw_profile check if nvidia-smi is unavailable.
        """
        if self.backend not in ("cuda",):
            # For non-CUDA backends check hw_profile for NVIDIA GPUs
            gpus = self.hw_profile.get("gpus", [])
            return any(
                str(g.get("vendor", "")).lower() == "nvidia" or
                str(g.get("backend_hint", "")).lower() == "cuda"
                for g in gpus
            )
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # nvidia-smi not found — check hw_profile
            gpus = self.hw_profile.get("gpus", [])
            return any(
                str(g.get("vendor", "")).lower() == "nvidia"
                for g in gpus
            )

    def _build_env_vars(self) -> Dict[str, str]:
        """Build the environment variable set for the target backend."""
        env: Dict[str, str] = {}

        if self.backend == "cuda" or self._cuda_available:
            env.update({
                "CUDA_LAUNCH_BLOCKING":           "0",     # async kernel launch
                "GGML_CUDA_FORCE_CUBLAS":         "1",     # prefer cuBLAS
                "GGML_CUDA_ENABLE_UNIFIED_MEMORY":"0",     # keep tensors on device
                "CUDA_DEVICE_ORDER":              "FASTEST", # fastest GPU first
                "OMP_NUM_THREADS":                "1",     # don't fight CUDA streams
            })
        elif self.backend == "vulkan":
            env.update({
                "GGML_VK_DISABLE_VALIDATION":    "1",     # skip validation in prod
                "GGML_VK_SHADER_COMPILE_ASYNC":  "1",     # async shader compile
            })

        return env

    def _estimate_overlap(self) -> float:
        """
        Estimate compute/transfer overlap ratio from the placement plan.

        Returns 0.0 when no plan is available or no boundary crossings exist.
        """
        if self.plan is None:
            return 0.0
        crossings = self.plan.boundary_crossings
        if crossings == 0:
            return 0.0

        # Rough estimate: each boundary crossing requires transferring
        # avg_cpu_segment_mb MB over PCIe. Transfer time = size / BW.
        # If transfer_time < avg_layer_compute_time, overlap > 0.
        n_cpu = max(1, self.plan.n_cpu_layers)
        n_gpu = max(1, self.plan.n_gpu_layers)
        avg_layer_bytes = self.plan.estimated_ram_bytes / n_cpu

        # Estimated transfer time per boundary (seconds)
        transfer_sec = (avg_layer_bytes / (1024 ** 3)) / max(self.pcie_bandwidth_gbps, 0.1)

        # Assume GPU computes at 1 TFLOP/s for a typical layer:
        # ~200M params × 2 FLOPs/param = 400 GFLOP, at 1 TFLOP/s = 0.4 s
        # (this is a rough order-of-magnitude heuristic only)
        compute_sec_per_layer = 0.02  # 20 ms: conservative single-token latency

        overlap_ratio = min(1.0, transfer_sec / max(compute_sec_per_layer, 1e-6))
        return round(overlap_ratio, 4)
