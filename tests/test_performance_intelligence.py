"""
test_performance_intelligence.py
---------------------------------
Comprehensive test suite for Performance Intelligence Engine (PIE) in InferenceOS.
"""
import pytest
from unittest.mock import MagicMock

from performance_intelligence import (
    BaselineEngine,
    PerformanceBaseline,
    PerformanceIntelligenceEngine,
    PerformanceScoreCalculator,
    PIEConfig,
    RegressionDetector,
    RootCauseAnalyzer,
    TrendAnalyzer,
)
from cli.performance_cli import handle_performance_cli


# ---------------------------------------------------------------------------
# 1. Baseline Engine & Outlier Rejection Tests
# ---------------------------------------------------------------------------

def test_baseline_engine_outlier_rejection():
    engine = BaselineEngine()
    records = [
        {"eval_tps": 50.0, "prompt_tps": 180.0, "ttft_ms": 330.0, "latency_ms": 1200.0, "gpu_utilization": 90.0, "vram_used_mb": 4500.0},
        {"eval_tps": 52.0, "prompt_tps": 182.0, "ttft_ms": 328.0, "latency_ms": 1190.0, "gpu_utilization": 92.0, "vram_used_mb": 4500.0},
        {"eval_tps": 0.0, "prompt_tps": 0.0, "ttft_ms": 0.0},  # Failed run (outlier)
    ]
    baseline = engine.compute_baseline(records)
    assert baseline.sample_count == 2
    assert baseline.avg_eval_tps == 51.0


# ---------------------------------------------------------------------------
# 2. Regression Detector Tests
# ---------------------------------------------------------------------------

def test_regression_detector():
    detector = RegressionDetector()
    baseline = PerformanceBaseline(avg_eval_tps=50.0, avg_ttft_ms=330.0, sample_count=10)

    # 1. Normal run (no regression)
    report_ok = detector.detect_regression(current_eval_tps=49.0, current_ttft_ms=335.0, baseline=baseline)
    assert not report_ok.is_regression

    # 2. Regression run (TPS drops > 10%)
    report_reg = detector.detect_regression(current_eval_tps=41.0, current_ttft_ms=510.0, baseline=baseline)
    assert report_reg.is_regression
    assert report_reg.confidence_pct >= 80.0


# ---------------------------------------------------------------------------
# 3. Root Cause Analyzer & Trend Analyzer Tests
# ---------------------------------------------------------------------------

def test_root_cause_and_trend_analysis():
    rc_analyzer = RootCauseAnalyzer()
    detector = RegressionDetector()
    baseline = PerformanceBaseline(avg_eval_tps=50.0, avg_ttft_ms=330.0, sample_count=5)

    reg = detector.detect_regression(current_eval_tps=40.0, current_ttft_ms=500.0, baseline=baseline)
    cause = rc_analyzer.analyze_cause(reg, health_status="Good")

    assert cause.regression_detected
    assert cause.severity == "WARNING"

    trend_analyzer = TrendAnalyzer()
    history = [{"eval_tps": 40.0}, {"eval_tps": 45.0}, {"eval_tps": 50.0}]
    assert trend_analyzer.analyze_trend(history) == "Improving"


# ---------------------------------------------------------------------------
# 4. Performance Score Calculator Tests
# ---------------------------------------------------------------------------

def test_performance_score_calculator():
    calc = PerformanceScoreCalculator()
    baseline = PerformanceBaseline(avg_eval_tps=50.0, sample_count=5)
    score = calc.calculate_score(current_eval_tps=50.0, current_ttft_ms=330.0, baseline=baseline)

    assert 0 <= score.overall_score <= 100
    assert score.overall_score >= 90


# ---------------------------------------------------------------------------
# 5. PIE Facade & CLI Output Tests
# ---------------------------------------------------------------------------

def test_pie_facade_and_cli_output(tmp_path):
    pie = PerformanceIntelligenceEngine(config=PIEConfig(storage_dir=str(tmp_path / "pie")))
    score = pie.record_run(
        model_name="Qwen3-4B",
        gpu_name="RX5600M",
        prompt_tps=187.0,
        eval_tps=51.4,
        ttft_ms=334.0,
        latency_ms=1200.0,
    )
    assert score.overall_score > 0

    formatted = pie.format_cli_output()
    assert "Performance Intelligence" in formatted
    assert "Performance Score" in formatted
    assert "Generation TPS" in formatted
    assert "Prompt TPS" in formatted
    assert "TTFT" in formatted
    assert "Trend" in formatted
    assert "Regression" in formatted
    assert "Recommendation" in formatted


def test_performance_cli_handlers():
    assert handle_performance_cli(["show"]) == 0
    assert handle_performance_cli(["history"]) == 0
    assert handle_performance_cli(["trends"]) == 0
    assert handle_performance_cli(["regressions"]) == 0
    assert handle_performance_cli(["score"]) == 0
    assert handle_performance_cli(["analyze"]) == 0
