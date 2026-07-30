"""
test_optimizer.py
------------------
Comprehensive test suite for Model Fingerprinting and Automatic Performance Optimizer (APO) in InferenceOS.
"""
import pytest
from pathlib import Path

from optimizer import (
    APOConfig,
    AutomaticPerformanceOptimizer,
    CandidateConfig,
    FingerprintEngine,
    OptimizationConfidenceEngine,
    OptimizationProfile,
    OptimizationStorageEngine,
    OptimizationValidator,
    ProfileManager,
)
from cli.optimizer_cli import handle_fingerprint_cli, handle_optimizer_cli


# ---------------------------------------------------------------------------
# 1. Fingerprint Engine Tests
# ---------------------------------------------------------------------------

def test_fingerprint_engine(tmp_path):
    # Test model fingerprint on dummy file
    model_file = tmp_path / "dummy_model.gguf"
    model_file.write_bytes(b"GGUF_HEADER_DATA_1234567890")

    fp = FingerprintEngine.compute_model_fingerprint(model_path=str(model_file))
    assert fp.model_name == "dummy_model"
    assert len(fp.sha256_hash) == 64
    assert fp.get_fingerprint_hash() == fp.sha256_hash[:12]

    # Test hardware fingerprint
    hw_fp = FingerprintEngine.compute_hardware_fingerprint(
        hw_profile={"gpus": [{"name": "RX5600M", "vram_total_mb": 6144.0}]}, backend="vulkan"
    )
    assert hw_fp.gpu_name == "RX5600M"
    assert hw_fp.vram_gb == 6.0


# ---------------------------------------------------------------------------
# 2. Storage & Profile Manager Tests
# ---------------------------------------------------------------------------

def test_optimization_storage_and_profile_manager(tmp_path):
    storage = OptimizationStorageEngine(base_dir=str(tmp_path / "opt"))
    manager = ProfileManager(storage=storage)

    profile = OptimizationProfile(
        model_name="Qwen3-4B",
        model_hash="9b42a1f8",
        gpu_name="RX5600M",
        best_candidate=CandidateConfig(gpu_layers=38, microbatch_size=768, thread_count=8),
        expected_tps=51.2,
        expected_ttft_ms=338.0,
        expected_latency_ms=1200.0,
        expected_memory_mb=4700.0,
        confidence_pct=96.0,
        goal="Balanced",
    )
    manager.save_profile(profile)

    loaded = storage.load_profile("9b42a1f8", "RX5600M")
    assert loaded is not None
    assert loaded.model_name == "Qwen3-4B"
    assert loaded.best_candidate.gpu_layers == 38
    assert loaded.best_candidate.microbatch_size == 768


# ---------------------------------------------------------------------------
# 3. Validation & Confidence Engine Tests
# ---------------------------------------------------------------------------

def test_validation_and_confidence():
    validator = OptimizationValidator()
    model_fp = FingerprintEngine.compute_model_fingerprint()
    hw_fp = FingerprintEngine.compute_hardware_fingerprint()

    cand_valid = CandidateConfig(gpu_layers=32, microbatch_size=512)
    valid, reason = validator.validate_candidate(cand_valid, model_fp, hw_fp)
    assert valid

    cand_invalid = CandidateConfig(gpu_layers=32, microbatch_size=2048)
    valid_inv, _ = validator.validate_candidate(cand_invalid, model_fp, hw_fp)
    assert not valid_inv

    conf_engine = OptimizationConfidenceEngine()
    conf = conf_engine.compute_confidence(validation_passed=True, execution_count=5)
    assert conf >= 92.5


# ---------------------------------------------------------------------------
# 4. APO Facade & CLI Output Formatting Tests
# ---------------------------------------------------------------------------

def test_apo_facade_and_cli_formatting(tmp_path):
    apo = AutomaticPerformanceOptimizer(config=APOConfig(storage_dir=str(tmp_path / "apo")))
    profile = apo.get_or_create_profile(
        model_metadata={"model_name": "Qwen3-4B", "num_layers": 38},
        hw_profile={"gpus": [{"name": "RX5600M", "vram_total_mb": 6144.0}]},
        backend="vulkan",
    )
    assert isinstance(profile, OptimizationProfile)
    assert profile.model_name == "Qwen3-4B"
    assert profile.gpu_name == "RX5600M"

    formatted = profile.format_cli_output()
    assert "Automatic Performance Optimizer" in formatted
    assert "Model" in formatted
    assert "Qwen3-4B" in formatted
    assert "Fingerprint" in formatted
    assert "Hardware" in formatted
    assert "RX5600M" in formatted
    assert "Optimization Goal" in formatted
    assert "Balanced" in formatted
    assert "Testing" in formatted
    assert "Placement" in formatted
    assert "Microbatch" in formatted
    assert "Result" in formatted
    assert "Validation" in formatted
    assert "Passed" in formatted
    assert "Confidence" in formatted
    assert "Optimization Profile Saved" in formatted


def test_optimizer_cli_handlers():
    assert handle_optimizer_cli(["status"]) == 0
    assert handle_optimizer_cli(["profiles"]) == 0
    assert handle_fingerprint_cli([]) == 0
