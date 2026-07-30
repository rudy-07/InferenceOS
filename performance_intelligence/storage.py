"""
storage.py
----------
Persistent storage engine for Performance Intelligence Engine in InferenceOS.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("InferenceOS.PIE.StorageEngine")


class PIEStorageEngine:
    """
    Manages persistent SQLite storage under ~/.inferenceos/performance/.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        raw_dir = base_dir or "~/.inferenceos/performance"
        self.dir_path = Path(os.path.expanduser(raw_dir)).resolve()
        self.dir_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.dir_path / "history.db"
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_history (
                        record_id TEXT PRIMARY KEY,
                        model_name TEXT,
                        gpu_name TEXT,
                        prompt_tps REAL,
                        eval_tps REAL,
                        ttft_ms REAL,
                        latency_ms REAL,
                        gpu_utilization REAL,
                        vram_used_mb REAL,
                        score INTEGER,
                        timestamp REAL
                    )
                    """
                )
        finally:
            conn.close()

    def record_run(self, data: Dict[str, Any]) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO performance_history (
                        record_id, model_name, gpu_name, prompt_tps, eval_tps,
                        ttft_ms, latency_ms, gpu_utilization, vram_used_mb, score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["record_id"],
                        data.get("model_name", "Qwen3-4B"),
                        data.get("gpu_name", "GPU"),
                        data.get("prompt_tps", 180.0),
                        data.get("eval_tps", 50.0),
                        data.get("ttft_ms", 330.0),
                        data.get("latency_ms", 1200.0),
                        data.get("gpu_utilization", 90.0),
                        data.get("vram_used_mb", 4500.0),
                        data.get("score", 94),
                        data.get("timestamp", 0.0),
                    ),
                )
        finally:
            conn.close()

    def fetch_all(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(self.db_path))
        results: List[Dict[str, Any]] = []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT record_id, model_name, gpu_name, prompt_tps, eval_tps, ttft_ms, latency_ms, gpu_utilization, vram_used_mb, score, timestamp FROM performance_history ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            for r in rows:
                results.append(
                    {
                        "record_id": r[0],
                        "model_name": r[1],
                        "gpu_name": r[2],
                        "prompt_tps": r[3],
                        "eval_tps": r[4],
                        "ttft_ms": r[5],
                        "latency_ms": r[6],
                        "gpu_utilization": r[7],
                        "vram_used_mb": r[8],
                        "score": r[9],
                        "timestamp": r[10],
                    }
                )
        finally:
            conn.close()
        return results

    def clear(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute("DELETE FROM performance_history")
        finally:
            conn.close()
