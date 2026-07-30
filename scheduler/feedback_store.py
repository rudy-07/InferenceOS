"""
feedback_store.py
------------------
Stores post-inference runtime performance feedback for future Runtime Learning.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeFeedback:
    """
    Performance feedback collected after a single inference pass completes.

    Attributes
    ----------
    prompt_tps : float
        Measured prompt processing throughput (tokens/sec).
    eval_tps : float
        Measured token generation speed (tokens/sec).
    actual_vram_mb : float
        Peak VRAM consumption measured during inference (MB).
    actual_ram_mb : float
        Peak RAM consumption measured during inference (MB).
    gpu_util_pct : float
        Average GPU utilization percentage during inference.
    cpu_util_pct : float
        Average CPU utilization percentage during inference.
    ttft_ms : float
        Time To First Token (ms).
    microbatch : int
        Microbatch size used during this run.
    context_length : int
        Context window size used during this run.
    model_name : str
        Model identifier or filename.
    backend : str
        Execution backend used (e.g. "cuda", "vulkan", "cpu").
    timestamp : float
        Epoch timestamp when execution completed.
    """

    prompt_tps: float
    eval_tps: float
    actual_vram_mb: float
    actual_ram_mb: float
    gpu_util_pct: float
    cpu_util_pct: float
    ttft_ms: float
    microbatch: int
    context_length: int
    model_name: str
    backend: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback record to dict."""
        return {
            "prompt_tps": round(self.prompt_tps, 2),
            "eval_tps": round(self.eval_tps, 2),
            "actual_vram_mb": round(self.actual_vram_mb, 2),
            "actual_ram_mb": round(self.actual_ram_mb, 2),
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "cpu_util_pct": round(self.cpu_util_pct, 1),
            "ttft_ms": round(self.ttft_ms, 2),
            "microbatch": self.microbatch,
            "context_length": self.context_length,
            "model_name": self.model_name,
            "backend": self.backend,
            "timestamp": self.timestamp,
        }


class RuntimeFeedbackStore:
    """
    Thread-safe repository for accumulating runtime execution feedback entries.
    Provides statistical lookup mechanisms for historical calibration and future
    Runtime Learning policy integration.
    """

    def __init__(self, max_records: int = 1000) -> None:
        self.max_records = max_records
        self._records: List[RuntimeFeedback] = []
        self._lock = threading.Lock()

    def add_feedback(self, record: RuntimeFeedback) -> None:
        """Append a new runtime feedback record to the store."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_records:
                self._records.pop(0)

    def get_history(
        self,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        limit: int = 100,
    ) -> List[RuntimeFeedback]:
        """Retrieve recent feedback records matching criteria."""
        with self._lock:
            filtered = list(self._records)

        if model_name:
            filtered = [r for r in filtered if r.model_name.lower() == model_name.lower()]
        if backend:
            filtered = [r for r in filtered if r.backend.lower() == backend.lower()]

        return filtered[-limit:]

    def get_average_tps_for_microbatch(
        self,
        microbatch: int,
        model_name: Optional[str] = None,
    ) -> Optional[float]:
        """Compute mean prompt TPS for a given microbatch size if historical data exists."""
        history = self.get_history(model_name=model_name)
        matching = [r.prompt_tps for r in history if r.microbatch == microbatch and r.prompt_tps > 0]
        if not matching:
            return None
        return sum(matching) / len(matching)

    def clear(self) -> None:
        """Clear stored feedback history."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
