"""
storage.py
----------
Persistent storage engine for Runtime Knowledge Base in InferenceOS.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("InferenceOS.RKB.StorageEngine")


class RKBStorageEngine:
    """
    Manages persistent SQLite databases under ~/.inferenceos/knowledge/.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        raw_dir = base_dir or "~/.inferenceos/knowledge"
        self.dir_path = Path(os.path.expanduser(raw_dir)).resolve()
        self.dir_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.dir_path / "history.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS executions (
                        record_id TEXT PRIMARY KEY,
                        hardware_fp TEXT,
                        model_fp TEXT,
                        backend TEXT,
                        gpu_layers INTEGER,
                        microbatch_size INTEGER,
                        context_length INTEGER,
                        memory_used_mb REAL,
                        eval_tps REAL,
                        ttft_ms REAL,
                        latency_ms REAL,
                        gpu_utilization_pct REAL,
                        health_status TEXT,
                        success INTEGER,
                        raw_json TEXT,
                        timestamp REAL
                    )
                    """
                )
        finally:
            conn.close()

    def insert_execution(self, record_dict: Dict[str, Any]) -> None:
        """Insert execution record into history.db."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO executions (
                        record_id, hardware_fp, model_fp, backend, gpu_layers, microbatch_size,
                        context_length, memory_used_mb, eval_tps, ttft_ms, latency_ms,
                        gpu_utilization_pct, health_status, success, raw_json, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_dict["record_id"],
                        record_dict.get("hardware_fp", "unknown"),
                        record_dict.get("model_fp", "unknown"),
                        record_dict.get("backend", "unknown"),
                        record_dict.get("gpu_layers", 0),
                        record_dict.get("microbatch_size", 512),
                        record_dict.get("context_length", 4096),
                        record_dict.get("memory_used_mb", 0.0),
                        record_dict.get("eval_tps", 0.0),
                        record_dict.get("ttft_ms", 0.0),
                        record_dict.get("latency_ms", 0.0),
                        record_dict.get("gpu_utilization_pct", 0.0),
                        record_dict.get("health_status", "Good"),
                        1 if record_dict.get("success", True) else 0,
                        json.dumps(record_dict),
                        record_dict.get("timestamp", 0.0),
                    ),
                )
        finally:
            conn.close()

    def fetch_all_executions(self) -> List[Dict[str, Any]]:
        """Fetch all execution records as JSON dictionaries."""
        conn = sqlite3.connect(str(self.db_path))
        records: List[Dict[str, Any]] = []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_json FROM executions ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            for r in rows:
                try:
                    records.append(json.loads(r[0]))
                except Exception:
                    pass
        finally:
            conn.close()
        return records

    def clear(self) -> None:
        """Clear all records from database."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute("DELETE FROM executions")
        finally:
            conn.close()
