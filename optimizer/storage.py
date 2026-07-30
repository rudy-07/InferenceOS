"""
storage.py
----------
Persistent storage engine for Automatic Performance Optimizer in InferenceOS.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interfaces import CandidateConfig, OptimizationProfile

logger = logging.getLogger("InferenceOS.APO.StorageEngine")


class OptimizationStorageEngine:
    """
    Manages persistent SQLite storage under ~/.inferenceos/optimization/.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        raw_dir = base_dir or "~/.inferenceos/optimization"
        self.dir_path = Path(os.path.expanduser(raw_dir)).resolve()
        self.dir_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.dir_path / "profiles.db"
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS profiles (
                        profile_key TEXT PRIMARY KEY,
                        model_name TEXT,
                        model_hash TEXT,
                        gpu_name TEXT,
                        expected_tps REAL,
                        confidence_pct REAL,
                        goal TEXT,
                        version INTEGER,
                        is_valid INTEGER,
                        raw_json TEXT,
                        timestamp REAL
                    )
                    """
                )
        finally:
            conn.close()

    def save_profile(self, profile: OptimizationProfile) -> None:
        key = f"{profile.model_hash}_{profile.gpu_name}"
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO profiles (
                        profile_key, model_name, model_hash, gpu_name,
                        expected_tps, confidence_pct, goal, version, is_valid, raw_json, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        profile.model_name,
                        profile.model_hash,
                        profile.gpu_name,
                        profile.expected_tps,
                        profile.confidence_pct,
                        profile.goal,
                        profile.version,
                        1 if profile.is_valid else 0,
                        json.dumps(profile.to_dict()),
                        profile.timestamp,
                    ),
                )
        finally:
            conn.close()

    def load_profile(self, model_hash: str, gpu_name: str) -> Optional[OptimizationProfile]:
        key = f"{model_hash}_{gpu_name}"
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_json FROM profiles WHERE profile_key = ?", (key,))
            row = cursor.fetchone()
            if not row:
                return None

            d = json.loads(row[0])
            cand_d = d.get("best_candidate", {})
            cand = CandidateConfig(
                gpu_layers=cand_d.get("gpu_layers", 32),
                microbatch_size=cand_d.get("microbatch_size", 512),
                context_length=cand_d.get("context_length", 4096),
                memory_strategy=cand_d.get("memory_strategy", "balanced"),
                thread_count=cand_d.get("thread_count", 8),
                score=cand_d.get("score", 0.0),
            )
            return OptimizationProfile(
                model_name=d.get("model_name", "Model"),
                model_hash=d.get("model_hash", model_hash),
                gpu_name=d.get("gpu_name", gpu_name),
                best_candidate=cand,
                expected_tps=d.get("expected_tps", 50.0),
                expected_ttft_ms=d.get("expected_ttft_ms", 300.0),
                expected_latency_ms=d.get("expected_latency_ms", 1000.0),
                expected_memory_mb=d.get("expected_memory_mb", 4000.0),
                confidence_pct=d.get("confidence_pct", 95.0),
                goal=d.get("goal", "Balanced"),
                version=d.get("version", 1),
                is_valid=d.get("is_valid", True),
                reasoning=d.get("reasoning", []),
                timestamp=d.get("timestamp", 0.0),
            )
        finally:
            conn.close()

    def fetch_all_profiles(self) -> List[OptimizationProfile]:
        conn = sqlite3.connect(str(self.db_path))
        profiles: List[OptimizationProfile] = []
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_json FROM profiles ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            for r in rows:
                try:
                    d = json.loads(r[0])
                    cand_d = d.get("best_candidate", {})
                    cand = CandidateConfig(
                        gpu_layers=cand_d.get("gpu_layers", 32),
                        microbatch_size=cand_d.get("microbatch_size", 512),
                        context_length=cand_d.get("context_length", 4096),
                        memory_strategy=cand_d.get("memory_strategy", "balanced"),
                        thread_count=cand_d.get("thread_count", 8),
                    )
                    profiles.append(
                        OptimizationProfile(
                            model_name=d.get("model_name", "Model"),
                            model_hash=d.get("model_hash", "hash"),
                            gpu_name=d.get("gpu_name", "GPU"),
                            best_candidate=cand,
                            expected_tps=d.get("expected_tps", 50.0),
                            expected_ttft_ms=d.get("expected_ttft_ms", 300.0),
                            expected_latency_ms=d.get("expected_latency_ms", 1000.0),
                            expected_memory_mb=d.get("expected_memory_mb", 4000.0),
                            confidence_pct=d.get("confidence_pct", 95.0),
                            goal=d.get("goal", "Balanced"),
                        )
                    )
                except Exception:
                    pass
        finally:
            conn.close()
        return profiles

    def clear(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute("DELETE FROM profiles")
        finally:
            conn.close()
