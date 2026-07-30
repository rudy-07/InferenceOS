"""
hardware.py
-----------
Hardware Knowledge Base for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interfaces import ExecutionRecord, HardwareFingerprint


@dataclass
class HardwareKnowledgeProfile:
    gpu_name: str = "GPU"
    total_executions: int = 0
    stable_vram_mb: float = 6000.0
    stable_ram_mb: float = 14000.0
    best_placement_layers: int = 32
    best_microbatch: int = 512
    best_context: int = 8192
    avg_tps: float = 45.0
    avg_ttft_ms: float = 350.0
    confidence_pct: float = 80.0


class HardwareKnowledgeBase:
    """
    Maintains persistent knowledge about hardware architectures and capabilities.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, HardwareKnowledgeProfile] = {}

    def update_from_records(self, records: List[ExecutionRecord]) -> None:
        """Aggregate execution records to build hardware profiles."""
        if not records:
            return

        grouped: Dict[str, List[ExecutionRecord]] = {}
        for r in records:
            key = r.hardware_fp.gpu_name
            grouped.setdefault(key, []).append(r)

        for key, recs in grouped.items():
            valid = [r for r in recs if r.success]
            if not valid:
                continue

            n = len(valid)
            best_tps_rec = max(valid, key=lambda r: r.eval_tps)
            avg_tps = sum(r.eval_tps for r in valid) / float(n)
            avg_ttft = sum(r.ttft_ms for r in valid) / float(n)

            # Calculate confidence score (scaled up to 98% with 100+ executions)
            conf = min(98.0, 50.0 + (n * 0.30))

            self._profiles[key] = HardwareKnowledgeProfile(
                gpu_name=key,
                total_executions=n,
                stable_vram_mb=round(max(r.memory_used_mb for r in valid), 1),
                best_placement_layers=best_tps_rec.gpu_layers,
                best_microbatch=best_tps_rec.microbatch_size,
                best_context=max(r.context_length for r in valid),
                avg_tps=round(avg_tps, 1),
                avg_ttft_ms=round(avg_ttft, 1),
                confidence_pct=round(conf, 1),
            )

    def get_profile(self, gpu_name: str) -> HardwareKnowledgeProfile:
        return self._profiles.get(gpu_name, HardwareKnowledgeProfile(gpu_name=gpu_name))
