"""
test_kv_manager.py
-------------------
Comprehensive test suite for Intelligent KV Manager in InferenceOS.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from kv_manager import (
    AdaptiveKVCompressor,
    ImportanceAnalyzer,
    IntelligentKVEvictor,
    IntelligentKVManager,
    KVDecision,
    KVPolicyConfig,
    KVPolicyEngine,
    KVQuantizer,
)
from cli.kv_cli import handle_kv_cli


# ---------------------------------------------------------------------------
# 1. Importance Analyzer Tests
# ---------------------------------------------------------------------------

def test_importance_analyzer_scoring():
    analyzer = ImportanceAnalyzer()
    blocks = analyzer.analyze_blocks(total_tokens=4096, block_size=512, system_prompt_tokens=256)
    
    assert len(blocks) == 8
    # First block (system prompt) should have maximum structural protection
    assert blocks[0].role == "system"
    assert blocks[0].structure_score == 1.0

    # Last block (newest assistant response) should have highest recency score
    assert blocks[-1].recency_score == 1.0


# ---------------------------------------------------------------------------
# 2. KV Quantizer & Adaptive Compressor Tests
# ---------------------------------------------------------------------------

def test_kv_quantizer_memory_reduction():
    orig_mb = 1000.0
    new_int8_mb, saved_int8 = KVQuantizer.calculate_memory_reduction(orig_mb, target_format="INT8")
    assert new_int8_mb == 500.0
    assert saved_int8 == 500.0

    new_int4_mb, saved_int4 = KVQuantizer.calculate_memory_reduction(orig_mb, target_format="INT4")
    assert new_int4_mb == 250.0
    assert saved_int4 == 750.0


def test_adaptive_kv_compressor_modes():
    compressor = AdaptiveKVCompressor()

    # 1. Disabled
    res_dis = compressor.evaluate_compression(current_kv_mb=1000.0, requested_mode="disabled")
    assert res_dis.mode_selected == "Disabled"
    assert res_dis.memory_saved_mb == 0.0

    # 2. Adaptive under High pressure -> selects Aggressive (INT4)
    res_high = compressor.evaluate_compression(current_kv_mb=1000.0, pressure_level="High", requested_mode="adaptive")
    assert res_high.mode_selected == "Aggressive"
    assert res_high.target_format == "INT4"
    assert res_high.memory_saved_mb == 750.0

    # 3. Adaptive under Medium pressure -> selects Balanced (INT8)
    res_med = compressor.evaluate_compression(current_kv_mb=1000.0, pressure_level="Medium", requested_mode="adaptive")
    assert res_med.mode_selected == "Balanced"
    assert res_med.target_format == "INT8"
    assert res_med.memory_saved_mb == 500.0


# ---------------------------------------------------------------------------
# 3. Intelligent Evictor Tests
# ---------------------------------------------------------------------------

def test_intelligent_kv_evictor_protection():
    evictor = IntelligentKVEvictor()
    # High pressure eviction request
    res = evictor.evaluate_eviction(
        total_tokens=4096,
        current_kv_mb=1000.0,
        target_free_mb=300.0,
        eviction_policy="adaptive",
        pressure_level="Critical",
    )
    assert res.tokens_evicted > 0
    assert res.memory_freed_mb >= 250.0
    # Block index 0 (system prompt) must NOT be evicted
    assert 0 not in res.evicted_block_indices


# ---------------------------------------------------------------------------
# 4. Policy Engine Thresholds Tests
# ---------------------------------------------------------------------------

def test_kv_policy_engine_thresholds():
    policy = KVPolicyEngine()

    # Normal utilization -> Adaptive policy
    eval_low = policy.evaluate_policy(current_kv_mb=1000.0, vram_free_mb=8000.0, vram_total_mb=16384.0, pressure_level="Low")
    assert not eval_low.eviction_required

    # Critical pressure -> Critical policy (compression & eviction active)
    eval_crit = policy.evaluate_policy(current_kv_mb=1000.0, vram_free_mb=500.0, vram_total_mb=16384.0, pressure_level="Critical")
    assert eval_crit.compression_required
    assert eval_crit.eviction_required
    assert eval_crit.suggested_compression_mode == "aggressive"


# ---------------------------------------------------------------------------
# 5. CLI Output Formatting & CLI Handler Tests
# ---------------------------------------------------------------------------

def test_kv_decision_cli_formatting():
    decision = KVDecision(
        current_kv_gb=2.8,
        pressure_level="Medium",
        compression_strategy="Balanced (INT8)",
        memory_saved_mb=420.0,
        eviction_action="None",
        policy_name="Adaptive Policy",
        reasoning=["Memory pressure exceeded balanced threshold."],
    )
    formatted = decision.format_cli_output()
    assert "Intelligent KV Manager" in formatted
    assert "Current KV" in formatted
    assert "2.8 GB" in formatted
    assert "Pressure" in formatted
    assert "Medium" in formatted
    assert "Compression" in formatted
    assert "Balanced (INT8)" in formatted
    assert "Memory Saved" in formatted
    assert "420 MB" in formatted
    assert "Eviction" in formatted
    assert "None" in formatted
    assert "Policy" in formatted
    assert "Adaptive Policy" in formatted
    assert "Reason" in formatted
    assert "Memory pressure exceeded balanced threshold." in formatted


def test_kv_cli_handler():
    assert handle_kv_cli(["show"]) == 0
    assert handle_kv_cli(["stats"]) == 0
    assert handle_kv_cli(["policies"]) == 0
    assert handle_kv_cli(["reset"]) == 0


# ---------------------------------------------------------------------------
# 6. Intelligent KV Manager Integration
# ---------------------------------------------------------------------------

def test_intelligent_kv_manager_facade():
    manager = IntelligentKVManager()
    model = {"num_kv_heads": 8, "head_dim": 128, "num_layers": 32, "model_name": "Llama-3-8B"}
    decision = manager.evaluate_kv_state(
        model_metadata=model,
        context_length=8192,
        vram_free_mb=2000.0,
        vram_total_mb=8192.0,
        pressure_level="High",
    )
    assert isinstance(decision, KVDecision)
    assert decision.current_kv_gb > 0
    assert decision.memory_saved_mb > 0.0
    assert "Aggressive" in decision.compression_strategy or "Balanced" in decision.compression_strategy
