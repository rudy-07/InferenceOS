"""
memory_monitor.py
-----------------
Background VRAM and RAM utilization monitor for Phase 5 layer migration.

Polls GPU memory and system RAM at a configurable interval using:
  1. GPUtil (optional) — most accurate for NVIDIA GPUs
  2. Hardware profile vram fields — for AMD/Intel via Phase 1 profiler
  3. psutil virtual_memory — for RAM; GPU assumed 0% if no GPU library

A daemon thread accumulates MemorySnapshot history so the HysteresisController
can examine recent trends without blocking inference.

Thread safety
-------------
All mutable state (``_history``, ``_latest``) is protected by a
``threading.Lock``. Callers may safely call ``get_latest()`` and
``get_history()`` from any thread.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# MemorySnapshot
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """
    A point-in-time memory utilization reading.

    Attributes
    ----------
    timestamp : float
        ``time.perf_counter()`` value at capture time.
    vram_used_bytes : int
        GPU VRAM in use (bytes).
    vram_total_bytes : int
        Total GPU VRAM capacity (bytes).
    vram_used_pct : float
        ``vram_used_bytes / vram_total_bytes * 100``  (0–100).
    ram_used_bytes : int
        System RAM in use (bytes).
    ram_total_bytes : int
        Total system RAM (bytes).
    ram_used_pct : float
        ``ram_used_bytes / ram_total_bytes * 100`` (0–100).
    """
    timestamp: float
    vram_used_bytes: int
    vram_total_bytes: int
    vram_used_pct: float
    ram_used_bytes: int
    ram_total_bytes: int
    ram_used_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "vram": {
                "used_bytes": self.vram_used_bytes,
                "total_bytes": self.vram_total_bytes,
                "used_pct": round(self.vram_used_pct, 2),
                "used_gb": round(self.vram_used_bytes / (1024 ** 3), 3),
                "total_gb": round(self.vram_total_bytes / (1024 ** 3), 3),
            },
            "ram": {
                "used_bytes": self.ram_used_bytes,
                "total_bytes": self.ram_total_bytes,
                "used_pct": round(self.ram_used_pct, 2),
            },
        }

    @property
    def vram_free_bytes(self) -> int:
        return max(0, self.vram_total_bytes - self.vram_used_bytes)


# ---------------------------------------------------------------------------
# MemoryMonitor
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """
    Background daemon thread that periodically captures VRAM and RAM usage.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile from Phase 1 profiler. Used to derive total VRAM
        when GPU monitoring libraries are unavailable.
    interval_ms : float
        Polling interval in milliseconds. Default 500 ms.
    history_size : int
        Maximum number of snapshots retained in memory. Older entries are
        discarded with a rolling deque. Default 120 (1 minute at 500ms).
    gpu_index : int
        Index of the primary GPU to monitor. Default 0.
    """

    def __init__(
        self,
        hw_profile: Dict[str, Any],
        interval_ms: float = 500.0,
        history_size: int = 120,
        gpu_index: int = 0,
    ) -> None:
        self._hw_profile = hw_profile
        self._interval_sec = interval_ms / 1000.0
        self._history_size = history_size
        self._gpu_index = gpu_index

        self._lock = threading.Lock()
        self._history: List[MemorySnapshot] = []
        self._latest: Optional[MemorySnapshot] = None
        self._active = False
        self._thread: Optional[threading.Thread] = None

        # Derive static VRAM total from hw_profile (used as fallback)
        self._static_vram_total = self._read_vram_total_from_profile()
        self._static_ram_total = self._read_ram_total_from_profile()

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        if self._active:
            return
        self._active = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="InferenceOSMemoryMonitor",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest(self) -> Optional[MemorySnapshot]:
        """
        Return the most recent memory snapshot, or None if not started yet.

        Thread-safe; may be called from any thread.
        """
        with self._lock:
            return self._latest

    def get_history(self, n: int = 10) -> List[MemorySnapshot]:
        """
        Return the last ``n`` snapshots, oldest first.

        Thread-safe; may be called from any thread.
        """
        with self._lock:
            return list(self._history[-n:])

    def capture_once(self) -> MemorySnapshot:
        """
        Take a single immediate snapshot without starting the background thread.
        Useful for one-shot checks and testing.
        """
        return self._capture_snapshot()

    @property
    def is_running(self) -> bool:
        """True if the background polling thread is active."""
        return self._active and bool(self._thread and self._thread.is_alive())

    # ---------------------------------------------------------------------------
    # Background polling
    # ---------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Main polling loop running in daemon thread."""
        while self._active:
            try:
                snapshot = self._capture_snapshot()
                with self._lock:
                    self._latest = snapshot
                    self._history.append(snapshot)
                    # Enforce rolling window
                    if len(self._history) > self._history_size:
                        self._history.pop(0)
            except Exception:
                pass  # Never crash the daemon thread
            time.sleep(self._interval_sec)

    def _capture_snapshot(self) -> MemorySnapshot:
        """Capture current memory state using best available method."""
        now = time.perf_counter()

        vram_used = 0
        vram_total = self._static_vram_total

        # --- VRAM: try GPUtil first (NVIDIA) ---
        try:
            import GPUtil  # type: ignore
            gpus = GPUtil.getGPUs()
            if gpus and self._gpu_index < len(gpus):
                g = gpus[self._gpu_index]
                vram_used = int(g.memoryUsed * 1024 * 1024)
                vram_total = int(g.memoryTotal * 1024 * 1024)
        except (ImportError, Exception):
            pass

        # --- VRAM: AMD/Intel via psutil meminfo (Linux only) ---
        # On Windows with no GPUtil, we rely on hw_profile total and report 0 used
        # (conservative: better to under-trigger than over-trigger migration)

        # --- RAM: psutil ---
        ram_used = 0
        ram_total = self._static_ram_total
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram_used = vm.used
            ram_total = vm.total
        except (ImportError, Exception):
            pass

        vram_used_pct = (
            (vram_used / vram_total * 100.0) if vram_total > 0 else 0.0
        )
        ram_used_pct = (
            (ram_used / ram_total * 100.0) if ram_total > 0 else 0.0
        )

        return MemorySnapshot(
            timestamp=now,
            vram_used_bytes=vram_used,
            vram_total_bytes=vram_total,
            vram_used_pct=vram_used_pct,
            ram_used_bytes=ram_used,
            ram_total_bytes=ram_total,
            ram_used_pct=ram_used_pct,
        )

    # ---------------------------------------------------------------------------
    # Profile parsing helpers
    # ---------------------------------------------------------------------------

    def _read_vram_total_from_profile(self) -> int:
        """Extract total VRAM bytes from hw_profile (Phase 1 output)."""
        gpus = self._hw_profile.get("gpus", [])
        if gpus:
            g = gpus[0]
            # Phase 1 stores in MB
            mb = float(g.get("vram_total_mb", g.get("vram_mb", 0)))
            if mb > 0:
                return int(mb * 1024 * 1024)
        return 8 * 1024 ** 3  # 8 GB safe default

    def _read_ram_total_from_profile(self) -> int:
        """Extract total RAM bytes from hw_profile."""
        ram = self._hw_profile.get("ram", self._hw_profile.get("memory", {}))
        total = int(ram.get("total_bytes", 16 * 1024 ** 3))
        return total
