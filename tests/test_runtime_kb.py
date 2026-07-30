"""
test_runtime_kb.py
-------------------
Comprehensive test suite for Runtime Knowledge Base (RKB) in InferenceOS.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from runtime_kb import (
    ExecutionRecord,
    HardwareFingerprint,
    HardwareKnowledgeBase,
    HistoricalExecutionDatabase,
    KnowledgeQueryEngine,
    KnowledgeRecommendationEngine,
    ModelFingerprint,
    ModelKnowledgeBase,
    PerformanceKnowledgeBase,
    RKBStorageEngine,
    RuntimeKnowledgeBase,
    SchedulerKnowledgeBase,
)
from cli.knowledge_cli import handle_knowledge_cli


# ---------------------------------------------------------------------------
# 1. Fingerprint & Database Tests
# ---------------------------------------------------------------------------

def test_fingerprints():
    hw_fp = HardwareFingerprint(gpu_name="RX5600M", vram_gb=6.0, backend="vulkan")
    assert hw_fp.to_string() == "RX5600M_6GB_vulkan"

    mod_fp = ModelFingerprint(model_name="Qwen3-4B", quantization="Q4_K_M")
    assert mod_fp.to_string() == "Qwen3-4B_Q4_K_M"


def test_historical_execution_database(tmp_path):
    storage = RKBStorageEngine(base_dir=str(tmp_path / "knowledge"))
    db = HistoricalExecutionDatabase(storage=storage)

    rec = ExecutionRecord(
        record_id="rec001",
        hardware_fp=HardwareFingerprint(gpu_name="RX5600M"),
        model_fp=ModelFingerprint(model_name="Qwen3-4B"),
        backend="vulkan",
        gpu_layers=38,
        microbatch_size=768,
        context_length=16384,
        memory_used_mb=4700.0,
        eval_tps=50.7,
        ttft_ms=338.0,
        latency_ms=1200.0,
        gpu_utilization_pct=92.0,
        health_status="Good",
        success=True,
    )
    db.record_execution(rec)
    assert db.count() == 1
    assert db.get_all_records()[0].record_id == "rec001"


# ---------------------------------------------------------------------------
# 2. Knowledge Stores & Aggregation Tests
# ---------------------------------------------------------------------------

def test_knowledge_stores_aggregation():
    hw_kb = HardwareKnowledgeBase()
    mod_kb = ModelKnowledgeBase()
    perf_kb = PerformanceKnowledgeBase()

    records = [
        ExecutionRecord(
            record_id="rec1",
            hardware_fp=HardwareFingerprint(gpu_name="RX5600M"),
            model_fp=ModelFingerprint(model_name="Qwen3-4B"),
            backend="vulkan",
            gpu_layers=38,
            microbatch_size=768,
            context_length=16384,
            memory_used_mb=4700.0,
            eval_tps=51.3,
            ttft_ms=338.0,
            latency_ms=1200.0,
            gpu_utilization_pct=92.0,
            health_status="Good",
            success=True,
        )
    ]

    hw_kb.update_from_records(records)
    mod_kb.update_from_records(records)
    perf_kb.update_from_records(records)

    hw_prof = hw_kb.get_profile("RX5600M")
    assert hw_prof.best_microbatch == 768

    mod_prof = mod_kb.get_profile("Qwen3-4B")
    assert mod_prof.best_tps == 51.3

    summary = perf_kb.get_summary()
    assert summary.avg_tps == 51.3


# ---------------------------------------------------------------------------
# 3. Query Engine & Recommendation Tests
# ---------------------------------------------------------------------------

def test_query_engine_and_recommendations():
    hw_kb = HardwareKnowledgeBase()
    mod_kb = ModelKnowledgeBase()
    perf_kb = PerformanceKnowledgeBase()
    recommender = KnowledgeRecommendationEngine()

    query_engine = KnowledgeQueryEngine(hw_kb, mod_kb, perf_kb, recommender)
    mb, conf = query_engine.get_recommended_microbatch("Qwen3-4B", "RX5600M")
    assert mb > 0
    assert conf > 0.0


# ---------------------------------------------------------------------------
# 4. RKB Facade & CLI Output Formatting Tests
# ---------------------------------------------------------------------------

def test_rkb_facade_cli_formatting(tmp_path):
    rkb = RuntimeKnowledgeBase(base_dir=str(tmp_path / "rkb"))
    rec = ExecutionRecord(
        record_id="rec_fmt",
        hardware_fp=HardwareFingerprint(gpu_name="RX5600M"),
        model_fp=ModelFingerprint(model_name="Qwen3-4B"),
        backend="vulkan",
        gpu_layers=38,
        microbatch_size=768,
        context_length=16384,
        memory_used_mb=4700.0,
        eval_tps=50.7,
        ttft_ms=338.0,
        latency_ms=1200.0,
        gpu_utilization_pct=92.0,
        health_status="Good",
        success=True,
    )
    rkb.record_execution(rec)
    formatted = rkb.format_cli_output(gpu_name="RX5600M", model_name="Qwen3-4B")

    assert "Runtime Knowledge Base" in formatted
    assert "Hardware" in formatted
    assert "RX5600M" in formatted
    assert "Executions" in formatted
    assert "Confidence" in formatted
    assert "Best Placement" in formatted
    assert "38 GPU" in formatted
    assert "Best Microbatch" in formatted
    assert "768" in formatted
    assert "Best Context" in formatted
    assert "16384" in formatted
    assert "Average TPS" in formatted
    assert "Average TTFT" in formatted
    assert "Model" in formatted
    assert "Qwen3-4B" in formatted


def test_knowledge_cli_handlers():
    assert handle_knowledge_cli(["show"]) == 0
    assert handle_knowledge_cli(["hardware"]) == 0
    assert handle_knowledge_cli(["models"]) == 0
    assert handle_knowledge_cli(["performance"]) == 0
    assert handle_knowledge_cli(["history"]) == 0
    assert handle_knowledge_cli(["recommend"]) == 0
    assert handle_knowledge_cli(["reset"]) == 0
