"""
engine.py
---------
Runtime Learning Engine and Adaptive Runtime Intelligence (ARTI) main facade for InferenceOS.

Continuously learns from every inference execution and automatically improves future scheduling decisions.
Provides recommendations to schedulers while maintaining strict separation of authority (schedulers hold final authority).
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .confidence import ConfidenceEngine
from .database import LearningDatabase, compute_hardware_fingerprint, compute_model_fingerprint
from .history import ExecutionHistoryStore, ExecutionRecord
from .knowledge import HardwareKnowledge, ModelKnowledge, WorkloadKnowledge
from .prediction import PerformancePredictor, RegressionReport
from .recommendation import LearningRecommendation
from .statistics import LearningStatistics

logger = logging.getLogger("InferenceOS.RuntimeLearningEngine")


class RuntimeLearningEngine:
    """
    Production-grade Runtime Learning Engine.

    Parameters
    ----------
    db_path : Optional[str or Path]
        Path to custom SQLite database file. Defaults to ~/.inferenceos/learning/knowledge.db.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self.db = LearningDatabase(db_path=db_path)
        self.history_store = ExecutionHistoryStore()
        self.confidence_engine = ConfidenceEngine()
        self.predictor = PerformancePredictor()

    def get_recommendation(
        self,
        model_metadata: Dict[str, Any],
        hw_profile: Dict[str, Any],
        requested_context: int = 4096,
        backend: str = "cpu",
        vram_free_mb: float = 0.0,
        ram_free_mb: float = 0.0,
    ) -> LearningRecommendation:
        """
        Query Adaptive Runtime Intelligence for optimal execution recommendations based on historical performance.

        Returns
        -------
        LearningRecommendation
            Object containing recommended placement, microbatch, context, expected TPS, TTFT, and confidence score.
        """
        hw_fp = compute_hardware_fingerprint(hw_profile, backend=backend)
        model_fp = compute_model_fingerprint(model_metadata)

        # Context bucket floored to nearest 4096 tokens (e.g. 4096, 8192, 16384, 32768)
        context_bucket = max(512, (requested_context // 4096) * 4096) if requested_context >= 4096 else requested_context
        workload_key = f"{hw_fp}_{model_fp}_{context_bucket}"

        # Fetch historical records
        records = self.db.get_executions(hardware_fingerprint=hw_fp, model_fingerprint=model_fp, limit=100)
        sample_count = len(records)
        success_count = sum(1 for r in records if r.success)
        failure_count = sum(1 for r in records if not r.success)

        # Cold Start Check (No historical runs yet)
        if sample_count == 0:
            return LearningRecommendation(
                recommended_microbatch=None,
                recommended_context=None,
                recommended_gpu_layers=None,
                recommended_safety_margin_mb=None,
                confidence=0.0,
                sample_count=0,
                decision_source="Heuristic Fallback (Cold Start)",
                reasoning=["Cold start: no previous executions recorded for this hardware and model combination."],
            )

        # Compute Confidence Score
        confidence = self.confidence_engine.compute_confidence(
            sample_count=sample_count,
            success_count=success_count,
            failure_count=failure_count,
            records=records,
        )

        # Predict performance metrics using PerformancePredictor
        metrics = self.predictor.predict_metrics(records)

        # Determine optimal historical microbatch & layer placement from fastest successful executions
        successful_records = [r for r in records if r.success and r.eval_tps > 0]
        if successful_records:
            fastest_run = max(successful_records, key=lambda r: r.eval_tps)
            opt_microbatch = fastest_run.microbatch_size
            opt_gpu_layers = fastest_run.n_gpu_layers
            opt_context = max(requested_context, fastest_run.context_length)
        else:
            opt_microbatch = 512
            opt_gpu_layers = int(model_metadata.get("num_layers", 32))
            opt_context = requested_context

        gpu_name = hw_profile.get("gpus", [{}])[0].get("name", "GPU") if hw_profile.get("gpus") else "CPU"
        model_name = str(model_metadata.get("model_name", "Model"))

        reasoning = [
            f"Evaluated {sample_count} previous executions for {model_name} on {gpu_name}.",
            f"Historical peak generation throughput: {metrics['expected_eval_tps']:.1f} TPS.",
            f"Confidence score of {int(round(confidence * 100))}% based on {sample_count} samples and low variance.",
        ]

        rec = LearningRecommendation(
            recommended_microbatch=opt_microbatch,
            recommended_context=opt_context,
            recommended_gpu_layers=opt_gpu_layers,
            recommended_safety_margin_mb=500.0,
            expected_prompt_tps=metrics["expected_prompt_tps"],
            expected_eval_tps=metrics["expected_eval_tps"],
            expected_ttft_ms=metrics["expected_ttft_ms"],
            expected_vram_mb=metrics["expected_vram_mb"],
            expected_ram_mb=metrics["expected_ram_mb"],
            expected_gpu_utilization=metrics["expected_gpu_util"],
            expected_cpu_utilization=metrics["expected_cpu_util"],
            confidence=confidence,
            sample_count=sample_count,
            decision_source=f"Runtime Learning ({sample_count} previous executions)",
            reasoning=reasoning,
        )

        # Save workload knowledge cache
        wk = WorkloadKnowledge(
            workload_key=workload_key,
            hardware_fingerprint=hw_fp,
            model_fingerprint=model_fp,
            context_bucket=context_bucket,
            optimal_microbatch=opt_microbatch,
            optimal_gpu_layers=opt_gpu_layers,
            optimal_context=opt_context,
            expected_prompt_tps=metrics["expected_prompt_tps"],
            expected_eval_tps=metrics["expected_eval_tps"],
            expected_ttft_ms=metrics["expected_ttft_ms"],
            expected_vram_mb=metrics["expected_vram_mb"],
            expected_ram_mb=metrics["expected_ram_mb"],
            confidence_score=confidence,
            sample_count=sample_count,
            success_count=success_count,
            failure_count=failure_count,
        )
        self.db.save_workload_knowledge(wk)

        return rec

    def record_execution(
        self,
        model_metadata: Dict[str, Any],
        hw_profile: Dict[str, Any],
        stats: Any,
        args: List[str],
        backend: str = "cpu",
        n_gpu_layers: int = 0,
        n_cpu_layers: int = 0,
        n_igpu_layers: int = 0,
        microbatch_size: int = 512,
        context_length: int = 4096,
        scheduler_decisions: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        success: bool = True,
        duration_sec: float = 0.0,
    ) -> ExecutionRecord:
        """
        Record telemetry from a completed inference pass into the persistent learning database.
        """
        hw_fp = compute_hardware_fingerprint(hw_profile, backend=backend)
        model_fp = compute_model_fingerprint(model_metadata)

        gpus = hw_profile.get("gpus", [])
        gpu_name = gpus[0].get("name", "GPU") if gpus else "CPU"
        model_name = str(model_metadata.get("model_name", "Model"))

        eval_tps = getattr(stats, "eval_tps", getattr(stats, "tokens_per_second", 0.0))
        prompt_tps = getattr(stats, "prompt_tps", getattr(stats, "prompt_tokens_per_second", 0.0))
        ttft_ms = getattr(stats, "time_to_first_token_ms", getattr(stats, "ttft_ms", 0.0))
        total_latency_ms = getattr(stats, "total_time_ms", duration_sec * 1000.0)
        vram_used_mb = getattr(stats, "gpu_vram_used_mb", getattr(stats, "peak_vram_mb", 0.0))
        ram_used_mb = getattr(stats, "system_ram_used_mb", getattr(stats, "peak_ram_mb", 0.0))
        gpu_util = getattr(stats, "gpu_utilization_pct", 0.0)
        cpu_util = getattr(stats, "cpu_utilization_pct", 0.0)

        record = ExecutionRecord(
            record_id=str(uuid.uuid4())[:8],
            hardware_fingerprint=hw_fp,
            model_fingerprint=model_fp,
            model_name=model_name,
            gpu_name=gpu_name,
            backend=backend,
            n_gpu_layers=n_gpu_layers,
            n_cpu_layers=n_cpu_layers,
            n_igpu_layers=n_igpu_layers,
            microbatch_size=microbatch_size,
            context_length=context_length,
            thread_count=getattr(stats, "thread_count", 4),
            vram_used_mb=vram_used_mb,
            ram_used_mb=ram_used_mb,
            prompt_tps=prompt_tps,
            eval_tps=eval_tps,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            gpu_utilization_pct=gpu_util,
            cpu_utilization_pct=cpu_util,
            memory_pressure_level="Low",
            scheduler_decisions=scheduler_decisions or {},
            warnings=warnings or [],
            success=success,
            duration_sec=duration_sec,
        )

        self.db.save_execution(record)
        self.history_store.add_record(record)
        return record

    def detect_regression(
        self,
        model_metadata: Dict[str, Any],
        hw_profile: Dict[str, Any],
        backend: str = "cpu",
    ) -> RegressionReport:
        """Detect if recent executions exhibit performance regressions."""
        hw_fp = compute_hardware_fingerprint(hw_profile, backend=backend)
        model_fp = compute_model_fingerprint(model_metadata)
        records = self.db.get_executions(hardware_fingerprint=hw_fp, model_fingerprint=model_fp, limit=100)
        return self.predictor.detect_regression(records)

    def get_statistics(self) -> LearningStatistics:
        """Get overall system learning statistics."""
        return self.db.get_statistics()

    def reset_database(self) -> None:
        """Clear all knowledge and historical execution records."""
        self.db.clear()
        self.history_store.clear()

    def export_database(self, export_path: Union[str, Path]) -> None:
        """Export learning database to JSON file."""
        self.db.export_json(export_path)

    def import_database(self, import_path: Union[str, Path]) -> None:
        """Import learning database from JSON file."""
        self.db.import_json(import_path)
