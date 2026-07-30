"""
test_runtime_learning.py
-------------------------
Comprehensive unit and integration test suite for Runtime Learning Engine & ARTI in InferenceOS.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from runtime_learning import (
    ConfidenceEngine,
    ExecutionRecord,
    LearningDatabase,
    LearningRecommendation,
    PerformancePredictor,
    RuntimeLearningEngine,
    compute_hardware_fingerprint,
    compute_model_fingerprint,
)
from cli.learning_cli import handle_learning_cli


# ---------------------------------------------------------------------------
# 1. Fingerprinting Tests
# ---------------------------------------------------------------------------

def test_fingerprinting_determinism():
    hw1 = {"gpus": [{"name": "RTX 4090", "vram_total_mb": 24576}], "cpu": {"name": "Ryzen 9"}, "ram": {"total_gb": 64}}
    hw2 = {"gpus": [{"name": "RTX 4090", "vram_total_mb": 24576}], "cpu": {"name": "Ryzen 9"}, "ram": {"total_gb": 64}}
    fp1 = compute_hardware_fingerprint(hw1, backend="vulkan")
    fp2 = compute_hardware_fingerprint(hw2, backend="vulkan")
    assert fp1 == fp2

    m1 = {"model_name": "Llama-3-8B", "architecture": "llama", "num_layers": 32, "hidden_size": 4096, "quantization": "Q4_K_M"}
    m2 = {"model_name": "Llama-3-8B", "architecture": "llama", "num_layers": 32, "hidden_size": 4096, "quantization": "Q4_K_M"}
    mfp1 = compute_model_fingerprint(m1)
    mfp2 = compute_model_fingerprint(m2)
    assert mfp1 == mfp2


# ---------------------------------------------------------------------------
# 2. Database Persistence, Reset, Export, and Import Tests
# ---------------------------------------------------------------------------

def test_learning_database_operations(tmp_path):
    db_file = tmp_path / "test_knowledge.db"
    db = LearningDatabase(db_path=db_file)

    rec = ExecutionRecord(
        record_id="rec001",
        hardware_fingerprint="hw123",
        model_fingerprint="m123",
        model_name="Llama-3-8B",
        gpu_name="RTX 4090",
        backend="vulkan",
        n_gpu_layers=32,
        n_cpu_layers=0,
        n_igpu_layers=0,
        microbatch_size=768,
        context_length=8192,
        thread_count=8,
        vram_used_mb=5000.0,
        ram_used_mb=2000.0,
        prompt_tps=120.0,
        eval_tps=65.0,
        ttft_ms=250.0,
        total_latency_ms=1000.0,
        gpu_utilization_pct=95.0,
        cpu_utilization_pct=15.0,
        memory_pressure_level="Low",
        success=True,
    )
    db.save_execution(rec)

    fetched = db.get_executions(hardware_fingerprint="hw123", model_fingerprint="m123")
    assert len(fetched) == 1
    assert fetched[0].eval_tps == 65.0

    # Export & Import
    json_export = tmp_path / "export.json"
    db.export_json(json_export)
    assert json_export.exists()

    db.clear()
    assert len(db.get_executions("hw123", "m123")) == 0

    db.import_json(json_export)
    assert len(db.get_executions("hw123", "m123")) == 1


# ---------------------------------------------------------------------------
# 3. Cold Start vs Continuous Learning Confidence Scaling
# ---------------------------------------------------------------------------

def test_cold_start_and_confidence_progression(tmp_path):
    engine = RuntimeLearningEngine(db_path=tmp_path / "engine.db")

    hw = {"gpus": [{"name": "RX5600M", "vram_total_mb": 6144}], "ram": {"total_gb": 16}}
    model = {"model_name": "Qwen3-4B", "architecture": "qwen2", "num_layers": 28, "hidden_size": 3584}

    # 1. Cold start
    rec_cold = engine.get_recommendation(model_metadata=model, hw_profile=hw, backend="vulkan")
    assert rec_cold.confidence == 0.0
    assert "Cold Start" in rec_cold.decision_source

    # 2. Record 15 successful runs
    for i in range(15):
        mock_stats = MagicMock()
        mock_stats.eval_tps = 48.0 + (i % 3)
        mock_stats.prompt_tps = 150.0
        mock_stats.time_to_first_token_ms = 330.0
        mock_stats.total_time_ms = 800.0
        mock_stats.gpu_vram_used_mb = 4500.0
        mock_stats.system_ram_used_mb = 1800.0
        mock_stats.gpu_utilization_pct = 90.0
        mock_stats.cpu_utilization_pct = 20.0
        mock_stats.thread_count = 8
        mock_stats.warnings = []

        engine.record_execution(
            model_metadata=model,
            hw_profile=hw,
            stats=mock_stats,
            args=["-c", "8192"],
            backend="vulkan",
            n_gpu_layers=28,
            microbatch_size=768,
            context_length=8192,
        )

    rec_learned = engine.get_recommendation(model_metadata=model, hw_profile=hw, backend="vulkan")
    assert rec_learned.confidence > 0.60
    assert "Runtime Learning" in rec_learned.decision_source
    assert rec_learned.recommended_microbatch == 768


# ---------------------------------------------------------------------------
# 4. Performance Predictor & Regression Detection
# ---------------------------------------------------------------------------

def test_performance_regression_detection(tmp_path):
    engine = RuntimeLearningEngine(db_path=tmp_path / "reg.db")
    hw = {"gpus": [{"name": "RTX 3080", "vram_total_mb": 10240}]}
    model = {"model_name": "Mistral-7B"}

    # Record 10 baseline runs at ~60 TPS
    for _ in range(10):
        mock_stats = MagicMock(eval_tps=60.0, prompt_tps=200.0, time_to_first_token_ms=200.0, total_time_ms=500.0, gpu_vram_used_mb=6000.0, system_ram_used_mb=2000.0, gpu_utilization_pct=95.0, cpu_utilization_pct=10.0, thread_count=4)
        engine.record_execution(model, hw, mock_stats, ["-c", "4096"], backend="cuda")

    # Record 5 degraded runs at ~45 TPS (25% drop)
    for _ in range(5):
        mock_stats = MagicMock(eval_tps=45.0, prompt_tps=160.0, time_to_first_token_ms=280.0, total_time_ms=700.0, gpu_vram_used_mb=6000.0, system_ram_used_mb=2000.0, gpu_utilization_pct=95.0, cpu_utilization_pct=10.0, thread_count=4)
        engine.record_execution(model, hw, mock_stats, ["-c", "4096"], backend="cuda")

    reg = engine.detect_regression(model, hw, backend="cuda")
    assert reg.is_regression
    assert reg.drop_percentage >= 15.0
    assert "Performance regression detected" in reg.message


# ---------------------------------------------------------------------------
# 5. CLI Output Formatting & CLI Commands Tests
# ---------------------------------------------------------------------------

def test_learning_recommendation_cli_output():
    rec = LearningRecommendation(
        recommended_gpu_layers=38,
        recommended_microbatch=768,
        recommended_context=16384,
        expected_eval_tps=50.8,
        expected_ttft_ms=335.0,
        confidence=0.94,
        sample_count=143,
        decision_source="Runtime Learning (143 previous executions)",
    )
    formatted = rec.format_cli_output(gpu_name="RX5600M", model_name="Qwen3-4B")
    assert "Adaptive Runtime Intelligence" in formatted
    assert "RX5600M" in formatted
    assert "Qwen3-4B" in formatted
    assert "94%" in formatted
    assert "768" in formatted
    assert "16384" in formatted
    assert "50.8" in formatted
    assert "335 ms" in formatted


def test_learning_cli_handler(tmp_path):
    ret_stats = handle_learning_cli(["stats"])
    assert ret_stats == 0

    json_path = tmp_path / "export_cli.json"
    ret_export = handle_learning_cli(["export", str(json_path)])
    assert ret_export == 0
    assert json_path.exists()

    ret_reset = handle_learning_cli(["reset"])
    assert ret_reset == 0
