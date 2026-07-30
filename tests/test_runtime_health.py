"""
test_runtime_health.py
-----------------------
Comprehensive test suite for Runtime Health Monitor in InferenceOS.
"""
import pytest
from unittest.mock import MagicMock

from runtime_health import (
    AlertEngine,
    HealthSensorSuite,
    HealthStatus,
    RuntimeHealth,
    RuntimeHealthMonitor,
)
from cli.health_cli import handle_health_cli


# ---------------------------------------------------------------------------
# 1. Sensor Sampling Tests
# ---------------------------------------------------------------------------

def test_health_sensor_suite():
    sensors = HealthSensorSuite()
    hw = {
        "gpus": [{"name": "RTX 4090", "vram_total_mb": 24576.0, "vram_free_mb": 12000.0, "temp_c": 62.0}],
        "ram": {"total_gb": 64.0, "available_gb": 32.0},
    }
    gpu = sensors.sample_gpu(hw)
    assert gpu.gpu_name == "RTX 4090"
    assert gpu.vram_used_mb == 12576.0
    assert gpu.temp_c == 62.0

    ram = sensors.sample_ram(hw)
    assert ram.ram_used_gb == 32.0


# ---------------------------------------------------------------------------
# 2. Alert Engine Tests
# ---------------------------------------------------------------------------

def test_alert_engine_thermal_throttling():
    engine = AlertEngine()
    gpu = MagicMock(is_throttling=True, temp_c=88.0, pressure_level="Low", vram_used_mb=1000.0, vram_total_mb=8192.0)
    cpu = MagicMock()
    ram = MagicMock()

    alerts = engine.evaluate_alerts(gpu, cpu, ram)
    assert len(alerts) >= 1
    assert any("thermal throttling" in a.message.lower() for a in alerts)


# ---------------------------------------------------------------------------
# 3. Monitor Facade & CLI Output Tests
# ---------------------------------------------------------------------------

def test_runtime_health_monitor_facade():
    monitor = RuntimeHealthMonitor()
    hw = {
        "gpus": [{"name": "RX5600M", "vram_total_mb": 6144.0, "vram_free_mb": 4000.0, "temp_c": 55.0}],
        "ram": {"total_gb": 16.0, "available_gb": 10.0},
    }
    health = monitor.observe_health(hw_profile=hw)
    assert isinstance(health, RuntimeHealth)
    assert health.status in ("Good", "Excellent")
    assert health.gpu_memory_gb > 0


def test_runtime_health_cli_output():
    health = RuntimeHealth(
        status="Good",
        gpu_utilization_pct=92.0,
        gpu_memory_gb=5.1,
        temperature_c=74.0,
        memory_pressure="Low",
        performance_stability="Stable",
    )
    formatted = health.format_cli_output()
    assert "Runtime Health" in formatted
    assert "GPU Utilization" in formatted
    assert "92%" in formatted
    assert "5.1 GB" in formatted
    assert "Low" in formatted
    assert "74°C" in formatted
    assert "Good" in formatted


def test_health_cli_handler():
    assert handle_health_cli(["show"]) == 0
    assert handle_health_cli(["monitor"]) == 0
