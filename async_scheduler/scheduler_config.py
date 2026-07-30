"""
scheduler_config.py
-------------------
Configuration dataclass for Phase 8 Asynchronous Execution Scheduler.

All fields are optional with production-ready defaults. The config is
designed to be constructed once (from a hardware profile) and treated as
immutable during scheduler operation.

Tuning guidelines
-----------------
n_cpu_workers
    Set to physical_cores // 2. This leaves the other half of the CPU
    cores for llama.cpp's own thread pool (--threads flag). Using all
    cores here would starve the inference process.

n_transfer_workers
    Keep at 1. PCIe is a shared bus — multiple concurrent transfer threads
    fight for the same bandwidth and increase contention. One dedicated
    transfer worker is optimal.

max_queue_depth
    Controls backpressure. When all workers are busy and the queue reaches
    this depth, submit() raises QueueFullError rather than growing unboundedly.
    8 is sufficient for all practical multi-request workloads.

prefetch_lookahead
    How many requests ahead to prefetch model segments. 1 is safe; higher
    values improve throughput but consume more pinned memory.

pinned_buffer_size_mb
    Total pinned (page-locked) memory budget for model segment prefetching.
    Each segment copy costs zero CPU cycles once issued — the DMA engine
    handles it while the GPU computes. 256 MB is a good default; increase
    to 512 MB if your model has many CPU↔GPU boundary segments.

cuda_streams_enabled
    Auto-detected. When True, the scheduler emits CUDA env-var hints to
    llama.cpp that encourage async kernel scheduling. When False (Vulkan,
    Metal, CPU), no CUDA env-vars are emitted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SchedulerConfig:
    """
    Complete configuration for the Phase 8 Async Execution Scheduler.

    Parameters
    ----------
    n_cpu_workers : int
        Number of CPU worker threads for the Prepare stage.
        ``-1`` = auto-detect as ``physical_cores // 2`` (minimum 1).
    n_transfer_workers : int
        Number of dedicated transfer worker threads (PCIe/memcpy operations).
        Keep at 1 to avoid bus contention. Default 1.
    max_queue_depth : int
        Maximum pending tasks per queue before backpressure is applied.
        Default 8.
    enable_request_pipelining : bool
        When True, overlap multiple sequential inference requests so that
        while one request runs on the GPU, the next request's CPU preparation
        is already in progress. Default True.
    enable_prefetch : bool
        When True, PrefetchManager pre-loads model boundary segments into
        pinned memory before the GPU needs them. Default True.
    prefetch_lookahead : int
        How many requests ahead to prefetch. Default 1.
    pinned_buffer_size_mb : int
        Total pinned-memory budget for prefetch buffers in megabytes.
        Default 256.
    cuda_streams_enabled : bool or None
        ``True`` = emit CUDA async env-vars to llama.cpp subprocess.
        ``None`` = auto-detect from hardware profile. Default None.
    benchmark_mode : bool
        When True, collect fine-grained per-stage timing. Adds slight
        overhead (extra timestamps). Default False.
    bubble_detection_threshold_pct : float
        If a pipeline stage is idle for more than this fraction of total
        time, it is flagged as a bubble. Default 0.20 (20%).
    max_concurrent_requests : int
        Maximum number of llama.cpp processes that may run simultaneously.
        Gated by VRAM headroom check. Default 2.
    """

    # Worker pool sizing
    n_cpu_workers: int = -1
    n_transfer_workers: int = 1
    max_queue_depth: int = 8

    # Pipeline features
    enable_request_pipelining: bool = True
    enable_prefetch: bool = True
    prefetch_lookahead: int = 1

    # Memory
    pinned_buffer_size_mb: int = 256

    # CUDA
    cuda_streams_enabled: Optional[bool] = None

    # Diagnostics
    benchmark_mode: bool = False
    bubble_detection_threshold_pct: float = 0.20

    # Multi-request
    max_concurrent_requests: int = 2

    def __post_init__(self) -> None:
        if self.n_cpu_workers == -1:
            try:
                import psutil
                phys = psutil.cpu_count(logical=False) or os.cpu_count() or 2
            except ImportError:
                phys = os.cpu_count() or 2
            self.n_cpu_workers = max(1, phys // 2)

        self.n_cpu_workers = max(1, self.n_cpu_workers)
        self.n_transfer_workers = max(1, self.n_transfer_workers)
        self.max_queue_depth = max(1, self.max_queue_depth)
        self.prefetch_lookahead = max(0, self.prefetch_lookahead)
        self.pinned_buffer_size_mb = max(64, self.pinned_buffer_size_mb)
        self.max_concurrent_requests = max(1, self.max_concurrent_requests)
        self.bubble_detection_threshold_pct = max(0.0, min(1.0, self.bubble_detection_threshold_pct))

    @classmethod
    def from_hw_profile(cls, hw_profile: Dict[str, Any], **overrides: Any) -> "SchedulerConfig":
        """
        Build a SchedulerConfig with sensible defaults derived from a hardware profile.

        Parameters
        ----------
        hw_profile : dict
            Hardware profile from Phase 1 profiler.
        **overrides
            Keyword arguments that override auto-detected values.

        Returns
        -------
        SchedulerConfig
        """
        cpu = hw_profile.get("cpu", {})
        phys_cores = int(cpu.get("physical_cores", -1))
        n_cpu_workers = max(1, phys_cores // 2) if phys_cores > 0 else -1

        # CUDA detection: look for NVIDIA discrete GPU
        gpus = hw_profile.get("gpus", [])
        has_cuda = any(
            str(g.get("vendor", "")).lower() == "nvidia" or
            str(g.get("backend_hint", "")).lower() == "cuda"
            for g in gpus
        )

        # Pinned buffer: larger if we have GPU layers crossing boundaries
        hints = hw_profile.get("inference_hints", {})
        vram_total_mb = sum(
            int(g.get("vram_total_mb", 0)) for g in gpus
        )
        pinned_mb = 512 if vram_total_mb >= 6144 else 256

        kwargs: Dict[str, Any] = {
            "n_cpu_workers": n_cpu_workers,
            "cuda_streams_enabled": has_cuda if has_cuda else None,
            "pinned_buffer_size_mb": pinned_mb,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_cpu_workers": self.n_cpu_workers,
            "n_transfer_workers": self.n_transfer_workers,
            "max_queue_depth": self.max_queue_depth,
            "enable_request_pipelining": self.enable_request_pipelining,
            "enable_prefetch": self.enable_prefetch,
            "prefetch_lookahead": self.prefetch_lookahead,
            "pinned_buffer_size_mb": self.pinned_buffer_size_mb,
            "cuda_streams_enabled": self.cuda_streams_enabled,
            "benchmark_mode": self.benchmark_mode,
            "bubble_detection_threshold_pct": self.bubble_detection_threshold_pct,
            "max_concurrent_requests": self.max_concurrent_requests,
        }
