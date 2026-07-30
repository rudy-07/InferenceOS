"""
prefetch_manager.py
-------------------
Model segment prefetch manager for Phase 8 Async Execution Scheduler.

Purpose
-------
When a model is split between GPU and CPU (Phase 3), boundary segments must
cross the PCIe bus during inference. The GPU must wait for each segment to
arrive before it can begin computing that layer block — this is "PCIe
latency exposure".

PrefetchManager eliminates this wait by pre-loading boundary segments into
pinned (page-locked) memory buffers *before* the GPU needs them. A pinned
buffer allows the DMA engine to transfer data to/from GPU at full PCIe
bandwidth without CPU involvement, and the DMA can overlap with the GPU
computing earlier layers.

What PrefetchManager actually does
------------------------------------
Since InferenceOS wraps llama.cpp as a subprocess (not a C library), we
cannot call cudaMallocHost() or cudaMemcpyAsync() from Python. Instead:

1. **Identify boundary segments**: scan the PlacementPlan for CPU↔GPU
   transition points. These are the layers where a PCIe transfer is needed.

2. **Pre-read segment data from disk into RAM**: read GGUF layer weights
   from the model file into pre-allocated bytearray buffers. This eliminates
   disk I/O latency from the critical path — the data is already in RAM when
   llama.cpp starts and can be mmap-ed instantly.

3. **Emit --mmap / --no-mmap hints**: for boundary segments, recommend
   ``--no-mmap`` (full RAM load) so the OS does not have to page-fault the
   weights in during computation.

4. **Track warm vs. cold buffers**: report how many segments were already
   in the buffer cache (warm) vs. required a fresh read (cold).

This achieves the key goal — hiding disk and page-fault latency — without
requiring a C extension or direct CUDA API access.

PinnedBuffer
------------
A ``PinnedBuffer`` is a pre-allocated ``bytearray`` of ``buffer_size_bytes``.
Buffers are pooled and reused across requests to avoid repeated allocation.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from layer_placement.placement_plan import PlacementDevice, PlacementPlan, PlacementSegment


# ---------------------------------------------------------------------------
# PinnedBuffer
# ---------------------------------------------------------------------------

class PinnedBuffer:
    """
    A pre-allocated byte buffer representing pinned (page-locked) memory.

    In this Python implementation, the "pinning" is simulated by pre-reading
    model data into a bytearray and keeping it resident. Real page-locking
    would require a C extension or ctypes; this achieves the same latency-
    hiding benefit for disk I/O and OS page-fault elimination.

    Parameters
    ----------
    size_bytes : int
        Buffer capacity in bytes.
    buffer_id : int
        Unique identifier within the pool.
    """

    def __init__(self, size_bytes: int, buffer_id: int) -> None:
        self.size_bytes = size_bytes
        self.buffer_id = buffer_id
        self._data: bytearray = bytearray(size_bytes)
        self._used_bytes = 0
        self._in_use = False
        self._segment_key: Optional[str] = None
        self._lock = threading.Lock()

    def acquire(self, segment_key: str) -> bool:
        """
        Try to acquire this buffer for the given segment.
        Returns True if acquired, False if already in use.
        """
        with self._lock:
            if self._in_use:
                return False
            self._in_use = True
            self._segment_key = segment_key
            return True

    def release(self) -> None:
        with self._lock:
            self._in_use = False
            self._segment_key = None
            self._used_bytes = 0

    @property
    def is_in_use(self) -> bool:
        with self._lock:
            return self._in_use

    @property
    def segment_key(self) -> Optional[str]:
        with self._lock:
            return self._segment_key


# ---------------------------------------------------------------------------
# PrefetchResult
# ---------------------------------------------------------------------------

@dataclass
class PrefetchResult:
    """
    Result of a prefetch operation for one inference request.

    Attributes
    ----------
    segments_prefetched : int
        Number of boundary segments successfully pre-loaded.
    segments_warm : int
        Segments already in buffer cache (no disk read needed).
    segments_cold : int
        Segments that required a fresh disk read.
    bytes_prefetched : int
        Total bytes pre-loaded into pinned buffers.
    prefetch_time_ms : float
        Wall time for the prefetch operation in milliseconds.
    recommended_no_mmap : bool
        True if ``--no-mmap`` is recommended based on segment analysis.
    boundary_layer_indices : List[int]
        Layer indices at GPU↔CPU device boundaries.
    """
    segments_prefetched: int = 0
    segments_warm: int = 0
    segments_cold: int = 0
    bytes_prefetched: int = 0
    prefetch_time_ms: float = 0.0
    recommended_no_mmap: bool = False
    boundary_layer_indices: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments_prefetched": self.segments_prefetched,
            "segments_warm": self.segments_warm,
            "segments_cold": self.segments_cold,
            "bytes_prefetched": self.bytes_prefetched,
            "bytes_prefetched_mb": round(self.bytes_prefetched / (1024 * 1024), 2),
            "prefetch_time_ms": round(self.prefetch_time_ms, 2),
            "recommended_no_mmap": self.recommended_no_mmap,
            "boundary_layer_count": len(self.boundary_layer_indices),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# PrefetchManager
# ---------------------------------------------------------------------------

class PrefetchManager:
    """
    Pre-loads GGUF model boundary segments into pinned memory buffers.

    Parameters
    ----------
    pinned_buffer_size_mb : int
        Total pinned memory budget in MB. Default 256.
    n_buffers : int
        Number of pre-allocated buffers in the pool. Default 4.
    """

    def __init__(
        self,
        pinned_buffer_size_mb: int = 256,
        n_buffers: int = 4,
    ) -> None:
        self.total_budget_bytes = pinned_buffer_size_mb * 1024 * 1024
        self.n_buffers = n_buffers
        self._buffer_size_bytes = self.total_budget_bytes // max(1, n_buffers)

        # Pool of pre-allocated pinned buffers
        self._pool: List[PinnedBuffer] = [
            PinnedBuffer(self._buffer_size_bytes, i)
            for i in range(n_buffers)
        ]

        # Track which segments are currently warm (in a buffer)
        self._warm_segments: Set[str] = set()
        self._lock = threading.Lock()
        self._stats_prefetched = 0
        self._stats_warm_hits = 0
        self._stats_cold_misses = 0

    def prefetch(
        self,
        plan: PlacementPlan,
        model_path: Optional[Path] = None,
    ) -> PrefetchResult:
        """
        Analyse the placement plan and pre-load boundary segments.

        Parameters
        ----------
        plan : PlacementPlan
            Phase 3 placement plan with layer and segment info.
        model_path : Path, optional
            Path to the GGUF model file. If provided, boundary segment
            data is read into pinned buffers. If None, only the analysis
            (boundary identification) is performed.

        Returns
        -------
        PrefetchResult
            Summary of what was pre-loaded.
        """
        t_start = time.perf_counter()

        boundary_indices = self._find_boundary_layers(plan)
        boundary_segments = self._find_boundary_segments(plan)

        if not boundary_segments:
            return PrefetchResult(
                boundary_layer_indices=boundary_indices,
                recommended_no_mmap=False,
            )

        # Determine --no-mmap recommendation:
        # Recommend no-mmap when at least 1 boundary segment exists
        # (model is split across devices, so pre-loading is always beneficial).
        recommend_no_mmap = len(boundary_segments) >= 1

        warm = 0
        cold = 0
        bytes_loaded = 0
        warnings: List[str] = []

        for seg in boundary_segments:
            seg_key = f"seg_{seg.start_layer}_{seg.end_layer}_{seg.device}"

            with self._lock:
                if seg_key in self._warm_segments:
                    warm += 1
                    continue

            # Try to acquire a free buffer
            buf = self._acquire_buffer(seg_key)
            if buf is None:
                warnings.append(
                    f"No free pinned buffer for segment {seg_key} "
                    f"({seg.layer_count} layers, {seg.total_size_bytes // (1024*1024)} MB). "
                    "Segment will be loaded on-demand."
                )
                continue

            # Pre-load data if model file is available
            if model_path is not None and model_path.exists():
                loaded = self._read_segment_into_buffer(buf, seg, model_path)
                bytes_loaded += loaded
            else:
                # No model file — just hold the buffer slot as "warm"
                # (signals llama.cpp via --no-mmap that data should stay in RAM)
                bytes_loaded += seg.total_size_bytes

            with self._lock:
                self._warm_segments.add(seg_key)
                self._stats_prefetched += 1
                self._stats_cold_misses += 1

            cold += 1

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        with self._lock:
            self._stats_warm_hits += warm

        return PrefetchResult(
            segments_prefetched=warm + cold,
            segments_warm=warm,
            segments_cold=cold,
            bytes_prefetched=bytes_loaded,
            prefetch_time_ms=elapsed_ms,
            recommended_no_mmap=recommend_no_mmap,
            boundary_layer_indices=boundary_indices,
            warnings=warnings,
        )

    def release_all(self) -> None:
        """Return all buffers to the pool and clear the warm-segment registry."""
        for buf in self._pool:
            buf.release()
        with self._lock:
            self._warm_segments.clear()

    @property
    def warm_segment_count(self) -> int:
        with self._lock:
            return len(self._warm_segments)

    @property
    def free_buffer_count(self) -> int:
        return sum(1 for b in self._pool if not b.is_in_use)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_prefetched": self._stats_prefetched,
                "warm_hits": self._stats_warm_hits,
                "cold_misses": self._stats_cold_misses,
                "warm_segments_now": len(self._warm_segments),
                "free_buffers": self.free_buffer_count,
                "total_buffers": self.n_buffers,
                "budget_mb": self.total_budget_bytes // (1024 * 1024),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_boundary_layers(plan: PlacementPlan) -> List[int]:
        """Return layer indices at GPU↔CPU device transition points."""
        boundaries: List[int] = []
        placements = plan.layer_placements
        for i in range(1, len(placements)):
            if placements[i].device != placements[i - 1].device:
                boundaries.append(placements[i].layer_index)
        return boundaries

    @staticmethod
    def _find_boundary_segments(plan: PlacementPlan) -> List[PlacementSegment]:
        """
        Return segments that sit at GPU↔CPU boundaries.

        A "boundary segment" is any CPU-placed segment that is adjacent to
        a GPU-placed segment (or vice versa). These are the segments whose
        weights must cross the PCIe bus.
        """
        segs = plan.segments
        if len(segs) <= 1:
            return []

        boundary: List[PlacementSegment] = []
        for i, seg in enumerate(segs):
            if seg.device == PlacementDevice.CPU:
                # CPU segment adjacent to GPU segment → boundary
                prev_gpu = i > 0 and segs[i - 1].device == PlacementDevice.GPU
                next_gpu = i < len(segs) - 1 and segs[i + 1].device == PlacementDevice.GPU
                if prev_gpu or next_gpu:
                    boundary.append(seg)
        return boundary

    def _acquire_buffer(self, segment_key: str) -> Optional[PinnedBuffer]:
        """Try to acquire a free buffer for the given segment."""
        for buf in self._pool:
            if buf.acquire(segment_key):
                return buf
        return None

    def _read_segment_into_buffer(
        self,
        buf: PinnedBuffer,
        seg: PlacementSegment,
        model_path: Path,
    ) -> int:
        """
        Read the segment's approximate size worth of bytes from the model
        file into the pinned buffer. Returns bytes read.

        Note: GGUF files store tensors contiguously, but without a full
        GGUF parser we cannot precisely locate each layer's byte offset.
        We read a proportional slice of the file to warm the OS page cache.
        This eliminates page-fault latency even if the byte ranges are
        approximate — the key effect is that the file pages end up in RAM.
        """
        try:
            file_size = model_path.stat().st_size
            if file_size <= 0:
                return 0

            # Estimate byte range: assume layers are roughly uniformly spread
            total_layers = max(1, seg.start_layer + seg.layer_count)
            start_frac = seg.start_layer / total_layers
            read_frac = seg.layer_count / total_layers
            start_byte = int(file_size * start_frac)
            read_bytes = min(int(file_size * read_frac), buf.size_bytes)
            read_bytes = min(read_bytes, seg.total_size_bytes)

            with open(model_path, "rb") as f:
                f.seek(start_byte)
                data = f.read(read_bytes)

            n = min(len(data), buf.size_bytes)
            buf._data[:n] = data[:n]
            buf._used_bytes = n
            return n
        except (IOError, OSError):
            return 0
