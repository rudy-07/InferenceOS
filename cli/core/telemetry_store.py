"""
telemetry_store.py
-------------------
Telemetry persistence store for InferenceOS historical metrics.

Saves benchmark runs, memory prediction accuracy, latency timelines, and throughput
statistics in ~/.inferenceos/telemetry/telemetry_history.json.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from .config_manager import ConfigManager, get_config_manager


class TelemetryStore:
    """
    Manages storage and historical analysis of telemetry records.
    """

    def __init__(self, config_mgr: Optional[ConfigManager] = None) -> None:
        self.config_mgr = config_mgr or get_config_manager()
        self.history_file = self.config_mgr.telemetry_dir / "telemetry_history.json"
        self._history: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        """Load telemetry history from disk."""
        if not self.history_file.exists():
            self._history = []
            return

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                self._history = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load telemetry store from {self.history_file}: {e}")
            self._history = []

    def save(self) -> None:
        """Save telemetry history to disk (capped at 500 records)."""
        try:
            # Keep latest 500 runs
            trimmed = self._history[-500:]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, indent=2)
        except Exception as e:
            print(f"[Error] Failed to save telemetry history: {e}")

    def record_run(self, run_dict: Dict[str, Any]) -> None:
        """Add a run record with timestamp."""
        record = {
            "timestamp": datetime.now().isoformat(),
            **run_dict,
        }
        self._history.append(record)
        self.save()

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return historical run records up to limit."""
        return self._history[-limit:]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregated summary statistics over historical runs."""
        if not self._history:
            return {
                "total_runs": 0,
                "avg_generation_tps": 0.0,
                "avg_prompt_tps": 0.0,
                "avg_ttft_ms": 0.0,
                "models_run": 0,
            }

        total_runs = len(self._history)
        gen_tps_list = [r.get("generation_tps", 0.0) for r in self._history if "generation_tps" in r]
        prompt_tps_list = [r.get("prompt_tps", 0.0) for r in self._history if "prompt_tps" in r]
        ttft_list = [r.get("ttft_ms", 0.0) for r in self._history if "ttft_ms" in r]
        models_set = {r.get("model_name", "unknown") for r in self._history}

        return {
            "total_runs": total_runs,
            "avg_generation_tps": round(sum(gen_tps_list) / max(1, len(gen_tps_list)), 2),
            "avg_prompt_tps": round(sum(prompt_tps_list) / max(1, len(prompt_tps_list)), 2),
            "avg_ttft_ms": round(sum(ttft_list) / max(1, len(ttft_list)), 1),
            "models_run": len(models_set),
        }

    def clear_history(self) -> None:
        """Clear all historical telemetry."""
        self._history = []
        self.save()


_telemetry_store_instance: Optional[TelemetryStore] = None


def get_telemetry_store() -> TelemetryStore:
    global _telemetry_store_instance
    if _telemetry_store_instance is None:
        _telemetry_store_instance = TelemetryStore()
    return _telemetry_store_instance
