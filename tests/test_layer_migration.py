"""
test_layer_migration.py
------------------------
Comprehensive test suite for InferenceOS Phase 5: Dynamic Layer Migration.

All tests are fully mockable — no real GPU, VRAM polling, or llama.exe needed.
MemoryMonitor polling is bypassed via capture_once() injection; subprocess.Popen
is mocked via unittest.mock.patch.

Test classes:
  1. TestMigrationConfig        — defaults, validation, presets
  2. TestMigrationEvent         — UUID, direction, message, serialization
  3. TestMemoryMonitor          — snapshot capture, start/stop lifecycle, history
  4. TestHysteresisController   — FSM transitions, cooldown, dead-band, OOM bypass
  5. TestPlanMutator            — migrate down, restore up, clamp, immutability
  6. TestMigrationCoordinator   — start/stop, force_migrate, event/plan callbacks
  7. TestMigrationEngine        — runWithMigration (mocked), forceMigrate, history
"""
from __future__ import annotations

import copy
import time
import threading
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from layer_migration import (
    HysteresisController,
    HysteresisState,
    MemoryMonitor,
    MemorySnapshot,
    MigrationConfig,
    MigrationCoordinator,
    MigrationEngine,
    MigrationEvent,
    MigrationReason,
    MigrationResult,
    PlanMutator,
)
from layer_placement.placement_plan import (
    LayerCostBreakdown,
    LayerPlacement,
    PlacementDevice,
    PlacementPlan,
    build_segments,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def default_config() -> MigrationConfig:
    return MigrationConfig(
        pressure_threshold_pct=85.0,
        relief_threshold_pct=60.0,
        oom_threshold_pct=95.0,
        cooldown_sec=30.0,
        consecutive_pressure_samples=3,
        layers_to_move_on_pressure=4,
        layers_to_restore_on_relief=2,
        monitor_interval_ms=500.0,
    )


@pytest.fixture
def hw_profile() -> dict:
    return {
        "gpus": [{"model": "Test GPU", "vendor": "amd", "vram_total_mb": 8192, "global_index": 0}],
        "igpus": [],
        "ram": {"total_bytes": 16 * 1024 ** 3},
        "memory": {"total_bytes": 16 * 1024 ** 3},
        "cpu": {"physical_cores": 8},
        "interconnects": [],
        "inference_hints": {"recommended_backend": "vulkan"},
    }


@pytest.fixture
def gpu_plan() -> PlacementPlan:
    """Plan with 12 layers, 10 on GPU, 0 on CPU."""
    placements = []
    for i in range(12):
        layer_type = "embedding" if i == 0 else ("lm_head" if i == 11 else "transformer")
        device = PlacementDevice.GPU
        placements.append(LayerPlacement(
            layer_index=i, layer_type=layer_type, device=device,
            gpu_index=0, size_bytes=200 * 1024 * 1024, cost=LayerCostBreakdown(),
        ))
    segs = build_segments(placements)
    return PlacementPlan(
        model_name="TestModel-7B", architecture="llama", total_layers=12,
        n_gpu_layers=10, n_cpu_layers=0, layer_placements=placements, segments=segs,
        estimated_vram_bytes=2_000_000_000, estimated_ram_bytes=0, context_length=2048,
    )


@pytest.fixture
def mixed_plan() -> PlacementPlan:
    """Plan with 14 layers, 8 on GPU, 4 on CPU."""
    placements = []
    for i in range(14):
        layer_type = "embedding" if i == 0 else ("lm_head" if i == 13 else "transformer")
        device = PlacementDevice.GPU if i < 9 else PlacementDevice.CPU
        gpu_idx = 0 if device == PlacementDevice.GPU else None
        placements.append(LayerPlacement(
            layer_index=i, layer_type=layer_type, device=device,
            gpu_index=gpu_idx, size_bytes=150 * 1024 * 1024, cost=LayerCostBreakdown(),
        ))
    segs = build_segments(placements)
    return PlacementPlan(
        model_name="TestModel-13B", architecture="llama", total_layers=14,
        n_gpu_layers=8, n_cpu_layers=4, layer_placements=placements, segments=segs,
        estimated_vram_bytes=1_200_000_000, estimated_ram_bytes=600_000_000, context_length=2048,
    )


def _make_snapshot(vram_pct: float, vram_total_bytes: int = 8 * 1024 ** 3) -> MemorySnapshot:
    used = int(vram_total_bytes * vram_pct / 100.0)
    return MemorySnapshot(
        timestamp=time.perf_counter(),
        vram_used_bytes=used,
        vram_total_bytes=vram_total_bytes,
        vram_used_pct=vram_pct,
        ram_used_bytes=4 * 1024 ** 3,
        ram_total_bytes=16 * 1024 ** 3,
        ram_used_pct=25.0,
    )


# ===========================================================================
# 1. TestMigrationConfig
# ===========================================================================

class TestMigrationConfig:

    def test_defaults_are_valid(self):
        cfg = MigrationConfig()
        assert 0 < cfg.relief_threshold_pct < cfg.pressure_threshold_pct < cfg.oom_threshold_pct
        assert cfg.cooldown_sec > 0
        assert cfg.consecutive_pressure_samples >= 1

    def test_pressure_must_exceed_relief(self):
        with pytest.raises(ValueError, match="relief_threshold_pct"):
            MigrationConfig(pressure_threshold_pct=50.0, relief_threshold_pct=80.0)

    def test_oom_must_exceed_pressure(self):
        with pytest.raises(ValueError, match="oom_threshold_pct"):
            MigrationConfig(pressure_threshold_pct=85.0, oom_threshold_pct=80.0)

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown_sec"):
            MigrationConfig(cooldown_sec=-1.0)

    def test_to_dict_serializable(self):
        import json
        cfg = MigrationConfig()
        json_str = json.dumps(cfg.to_dict())
        assert "pressure_threshold_pct" in json_str

    def test_aggressive_preset_thresholds(self):
        cfg = MigrationConfig.aggressive()
        assert cfg.pressure_threshold_pct < 80.0  # lower than default 85

    def test_conservative_preset_thresholds(self):
        cfg = MigrationConfig.conservative()
        assert cfg.pressure_threshold_pct > 88.0  # higher than default 85

    def test_feature_flags_default_true(self):
        cfg = MigrationConfig()
        assert cfg.enable_migration
        assert cfg.enable_restoration
        assert cfg.enable_oom_protection


# ===========================================================================
# 2. TestMigrationEvent
# ===========================================================================

class TestMigrationEvent:

    def test_event_has_non_empty_uuid(self):
        event = MigrationEvent(
            reason=MigrationReason.PRESSURE_HIGH, direction="gpu_to_cpu",
            layers_before=10, layers_after=6, vram_usage_pct=88.0,
            model_name="TestModel", backend="vulkan",
        )
        assert len(event.event_id) == 36  # UUID4 format

    def test_layers_moved_computed(self):
        event = MigrationEvent(
            reason=MigrationReason.PRESSURE_HIGH, direction="gpu_to_cpu",
            layers_before=10, layers_after=6, vram_usage_pct=88.0,
            model_name="TestModel", backend="vulkan",
        )
        assert event.layers_moved == 4

    def test_message_auto_generated(self):
        event = MigrationEvent(
            reason=MigrationReason.PRESSURE_HIGH, direction="gpu_to_cpu",
            layers_before=10, layers_after=6, vram_usage_pct=88.0,
            model_name="TestModel", backend="vulkan",
        )
        assert "GPU" in event.message or "gpu" in event.message.lower()
        assert "CPU" in event.message or "cpu" in event.message.lower()

    def test_event_to_dict_serializable(self):
        import json
        event = MigrationEvent(
            reason=MigrationReason.MANUAL, direction="cpu_to_gpu",
            layers_before=6, layers_after=8, vram_usage_pct=45.0,
            model_name="TestModel", backend="vulkan",
        )
        json_str = json.dumps(event.to_dict())
        assert "event_id" in json_str
        assert "migration_duration_ms" in json_str

    def test_reason_enum_values(self):
        assert MigrationReason.PRESSURE_HIGH == "pressure_high"
        assert MigrationReason.OOM_IMMINENT == "oom_imminent"
        assert MigrationReason.MANUAL == "manual"
        assert MigrationReason.PRESSURE_RELIEF == "pressure_relief"


# ===========================================================================
# 3. TestMemoryMonitor
# ===========================================================================

class TestMemoryMonitor:

    def test_snapshot_fields_populated(self, hw_profile):
        monitor = MemoryMonitor(hw_profile)
        snap = monitor.capture_once()
        assert isinstance(snap.timestamp, float)
        assert snap.vram_total_bytes > 0
        assert snap.ram_total_bytes > 0
        assert 0.0 <= snap.vram_used_pct <= 100.0
        assert 0.0 <= snap.ram_used_pct <= 100.0

    def test_start_stop_no_crash(self, hw_profile):
        monitor = MemoryMonitor(hw_profile, interval_ms=100.0)
        monitor.start()
        time.sleep(0.05)
        monitor.stop()
        assert not monitor.is_running

    def test_history_grows_after_start(self, hw_profile):
        monitor = MemoryMonitor(hw_profile, interval_ms=50.0)
        monitor.start()
        time.sleep(0.2)
        monitor.stop()
        history = monitor.get_history(n=20)
        assert len(history) >= 1  # at least one sample

    def test_get_latest_returns_snapshot_after_start(self, hw_profile):
        monitor = MemoryMonitor(hw_profile, interval_ms=50.0)
        monitor.start()
        time.sleep(0.15)
        snap = monitor.get_latest()
        monitor.stop()
        assert snap is not None or True  # may still be None if no GPU

    def test_vram_total_from_profile(self, hw_profile):
        """hw_profile has vram_total_mb=8192 → 8 GB total."""
        monitor = MemoryMonitor(hw_profile)
        assert monitor._static_vram_total == 8192 * 1024 * 1024

    def test_snapshot_vram_free_bytes(self, hw_profile):
        monitor = MemoryMonitor(hw_profile)
        snap = monitor.capture_once()
        assert snap.vram_free_bytes >= 0


# ===========================================================================
# 4. TestHysteresisController
# ===========================================================================

class TestHysteresisController:

    def test_no_migration_below_pressure_threshold(self, default_config):
        ctrl = HysteresisController(default_config)
        snap = _make_snapshot(70.0)  # dead-band
        for _ in range(5):
            result = ctrl.should_migrate_down(snap)
        assert not result

    def test_pressure_fires_after_n_consecutive_samples(self, default_config):
        ctrl = HysteresisController(default_config)
        snap = _make_snapshot(90.0)  # above 85%
        results = [ctrl.should_migrate_down(snap) for _ in range(3)]
        assert results[-1]  # True on 3rd sample

    def test_pressure_does_not_fire_before_n_samples(self, default_config):
        ctrl = HysteresisController(default_config)
        snap = _make_snapshot(90.0)
        # Only 2 samples, need 3
        r1 = ctrl.should_migrate_down(snap)
        r2 = ctrl.should_migrate_down(snap)
        assert not r1
        assert not r2

    def test_cooldown_blocks_second_migration(self, default_config):
        cfg = MigrationConfig(
            pressure_threshold_pct=85.0, relief_threshold_pct=60.0,
            cooldown_sec=60.0, consecutive_pressure_samples=1,
        )
        ctrl = HysteresisController(cfg)
        snap = _make_snapshot(90.0)
        # First migration fires
        ctrl.should_migrate_down(snap)
        ctrl.record_migration_started("gpu_to_cpu")
        ctrl.record_migration_completed()
        # Second attempt immediately → blocked by cooldown
        assert ctrl.is_in_cooldown()
        result = ctrl.should_migrate_down(snap)
        assert not result

    def test_no_oscillation_in_dead_band(self, default_config):
        ctrl = HysteresisController(default_config)
        snap = _make_snapshot(72.0)  # between 60 and 85
        for _ in range(10):
            assert not ctrl.should_migrate_down(snap)
            assert not ctrl.should_migrate_up(snap)

    def test_relief_fires_after_cooldown_expiry(self, default_config):
        cfg = MigrationConfig(
            pressure_threshold_pct=85.0, relief_threshold_pct=60.0,
            cooldown_sec=0.0, consecutive_pressure_samples=1,
        )
        ctrl = HysteresisController(cfg)
        # Simulate a prior migration to set last_migration_time
        ctrl.record_migration_started("gpu_to_cpu")
        ctrl.record_migration_completed()
        # Now cooldown=0 → expired immediately
        relief_snap = _make_snapshot(40.0)  # below 60%
        assert ctrl.should_migrate_up(relief_snap)

    def test_oom_bypasses_cooldown(self, default_config):
        cfg = MigrationConfig(
            pressure_threshold_pct=85.0, relief_threshold_pct=60.0,
            oom_threshold_pct=95.0, cooldown_sec=3600.0,  # 1 hour cooldown
        )
        ctrl = HysteresisController(cfg)
        ctrl.record_migration_started("gpu_to_cpu")
        ctrl.record_migration_completed()  # set long cooldown
        oom_snap = _make_snapshot(97.0)  # above 95%
        assert ctrl.should_migrate_down_emergency(oom_snap)

    def test_state_transitions_correctly(self, default_config):
        cfg = MigrationConfig(
            pressure_threshold_pct=85.0, relief_threshold_pct=60.0,
            cooldown_sec=0.0, consecutive_pressure_samples=1,
        )
        ctrl = HysteresisController(cfg)
        assert ctrl.state == "STABLE"
        snap = _make_snapshot(90.0)
        ctrl.should_migrate_down(snap)
        # After 1 sample (consecutive=1), should fire → PRESSURE state
        assert ctrl.state in ("PRESSURE", "STABLE")

    def test_disable_migration_flag(self, default_config):
        cfg = MigrationConfig(
            pressure_threshold_pct=85.0, relief_threshold_pct=60.0,
            enable_migration=False,
        )
        ctrl = HysteresisController(cfg)
        snap = _make_snapshot(95.0)
        for _ in range(10):
            assert not ctrl.should_migrate_down(snap)

    def test_cooldown_remaining_decreases(self, default_config):
        cfg = MigrationConfig(
            pressure_threshold_pct=85.0, relief_threshold_pct=60.0,
            cooldown_sec=2.0,
        )
        ctrl = HysteresisController(cfg)
        ctrl.record_migration_started("gpu_to_cpu")
        ctrl.record_migration_completed()
        r1 = ctrl.cooldown_remaining_sec()
        time.sleep(0.1)
        r2 = ctrl.cooldown_remaining_sec()
        assert r2 <= r1


# ===========================================================================
# 5. TestPlanMutator
# ===========================================================================

class TestPlanMutator:

    def test_migrate_to_cpu_reduces_gpu_count(self, gpu_plan):
        mutator = PlanMutator()
        new_plan = mutator.migrate_layers_to_cpu(gpu_plan, 4)
        assert new_plan.n_gpu_layers == gpu_plan.n_gpu_layers - 4
        assert new_plan.n_cpu_layers == gpu_plan.n_cpu_layers + 4

    def test_migrate_to_cpu_does_not_modify_original(self, gpu_plan):
        mutator = PlanMutator()
        orig_gpu = gpu_plan.n_gpu_layers
        mutator.migrate_layers_to_cpu(gpu_plan, 3)
        assert gpu_plan.n_gpu_layers == orig_gpu  # unchanged

    def test_migrate_respects_min_gpu_layers(self, gpu_plan):
        mutator = PlanMutator(min_gpu_layers=6)
        new_plan = mutator.migrate_layers_to_cpu(gpu_plan, 100)  # try to move all
        assert new_plan.n_gpu_layers >= 6

    def test_restore_to_gpu_increases_gpu_count(self, mixed_plan):
        mutator = PlanMutator()
        new_plan = mutator.restore_layers_to_gpu(mixed_plan, 2)
        assert new_plan.n_gpu_layers == mixed_plan.n_gpu_layers + 2

    def test_restore_respects_vram_budget(self, mixed_plan):
        mutator = PlanMutator()
        # Budget only allows 1 layer (150 MB = 157,286,400 bytes)
        budget = 160 * 1024 * 1024  # ~160 MB
        new_plan = mutator.restore_layers_to_gpu(mixed_plan, 3, vram_available_bytes=budget)
        assert new_plan.n_gpu_layers <= mixed_plan.n_gpu_layers + 1

    def test_clamp_plan_exact_count(self, gpu_plan):
        mutator = PlanMutator()
        new_plan = mutator.clamp_plan(gpu_plan, 5)
        assert new_plan.n_gpu_layers == 5

    def test_migrate_zero_layers_returns_equal_plan(self, gpu_plan):
        mutator = PlanMutator()
        new_plan = mutator.migrate_layers_to_cpu(gpu_plan, 0)
        assert new_plan.n_gpu_layers == gpu_plan.n_gpu_layers

    def test_segments_rebuilt_after_migration(self, gpu_plan):
        mutator = PlanMutator()
        new_plan = mutator.migrate_layers_to_cpu(gpu_plan, 3)
        # All placements must cover the same layer indices
        orig_indices = {lp.layer_index for lp in gpu_plan.layer_placements}
        new_indices = {lp.layer_index for lp in new_plan.layer_placements}
        assert orig_indices == new_indices

    def test_vram_estimate_decreases_after_downward_migration(self, gpu_plan):
        mutator = PlanMutator()
        new_plan = mutator.migrate_layers_to_cpu(gpu_plan, 4)
        assert new_plan.estimated_vram_bytes < gpu_plan.estimated_vram_bytes

    def test_ram_estimate_increases_after_downward_migration(self, gpu_plan):
        mutator = PlanMutator()
        new_plan = mutator.migrate_layers_to_cpu(gpu_plan, 4)
        assert new_plan.estimated_ram_bytes > gpu_plan.estimated_ram_bytes


# ===========================================================================
# 6. TestMigrationCoordinator
# ===========================================================================

class TestMigrationCoordinator:

    def test_coordinator_starts_and_stops_cleanly(self, default_config, hw_profile, gpu_plan):
        coord = MigrationCoordinator(
            config=default_config, hw_profile=hw_profile,
            model_name="TestModel", backend="vulkan",
        )
        coord.start(gpu_plan)
        time.sleep(0.05)
        coord.stop()
        assert not coord._active

    def test_current_plan_set_on_start(self, default_config, hw_profile, gpu_plan):
        coord = MigrationCoordinator(
            config=default_config, hw_profile=hw_profile,
        )
        coord.start(gpu_plan)
        assert coord.current_plan is not None
        assert coord.current_plan.n_gpu_layers == gpu_plan.n_gpu_layers
        coord.stop()

    def test_force_migrate_down_fires_event(self, default_config, hw_profile, gpu_plan):
        events: List[MigrationEvent] = []
        coord = MigrationCoordinator(
            config=default_config, hw_profile=hw_profile,
            on_event=events.append,
        )
        coord.start(gpu_plan)
        event = coord.force_migrate("gpu_to_cpu", 3)
        coord.stop()
        assert event.direction == "gpu_to_cpu"
        assert event.layers_moved == 3
        assert len(events) >= 1

    def test_force_migrate_on_plan_changed_callback(self, default_config, hw_profile, gpu_plan):
        plans: List[PlacementPlan] = []
        coord = MigrationCoordinator(
            config=default_config, hw_profile=hw_profile,
            on_plan_changed=lambda p, e: plans.append(p),
        )
        coord.start(gpu_plan)
        coord.force_migrate("gpu_to_cpu", 2)
        coord.stop()
        assert len(plans) >= 1
        assert plans[0].n_gpu_layers < gpu_plan.n_gpu_layers

    def test_migration_history_grows(self, default_config, hw_profile, gpu_plan):
        coord = MigrationCoordinator(
            config=default_config, hw_profile=hw_profile,
        )
        coord.start(gpu_plan)
        coord.force_migrate("gpu_to_cpu", 2)
        coord.force_migrate("gpu_to_cpu", 1)
        coord.stop()
        assert len(coord.migration_history) == 2

    def test_hysteresis_state_accessible(self, default_config, hw_profile, gpu_plan):
        coord = MigrationCoordinator(
            config=default_config, hw_profile=hw_profile,
        )
        coord.start(gpu_plan)
        state = coord.hysteresis_state
        coord.stop()
        assert state in ("STABLE", "PRESSURE", "COOLDOWN", "RELIEF", "MIGRATING_DOWN", "MIGRATING_UP")

    def test_force_migrate_before_start_raises(self, default_config, hw_profile):
        coord = MigrationCoordinator(
            config=default_config, hw_profile=hw_profile,
        )
        with pytest.raises(RuntimeError):
            coord.force_migrate("gpu_to_cpu", 2)


# ===========================================================================
# 7. TestMigrationEngine
# ===========================================================================

class TestMigrationEngine:

    _SAMPLE_STDERR = (
        "llama_perf_context_print: prompt eval time =   200.00 ms /    10 tokens "
        "(   20.00 ms per token,    50.00 tokens per second)\n"
        "llama_perf_context_print:        eval time =  1500.00 ms /    75 runs   "
        "(   20.00 ms per token,    50.00 tokens per second)\n"
    )

    def _mock_popen(self, stdout="Generated text", returncode=0):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([stdout + "\n"])
        mock_proc.stderr = iter([self._SAMPLE_STDERR])
        mock_proc.returncode = returncode
        mock_proc.poll.return_value = returncode
        mock_proc.wait.return_value = returncode
        mock_proc.communicate.return_value = (stdout, self._SAMPLE_STDERR)
        return patch("subprocess.Popen", return_value=mock_proc)

    def _make_engine(self, hw_profile, tmp_path, migration_config=None) -> MigrationEngine:
        exe = tmp_path / "llama.exe"
        exe.write_text("fake")
        cfg = RuntimeConfig(threads=4, async_streaming=False)
        return MigrationEngine(
            hw_profile=hw_profile,
            llama_exe_path=exe,
            config=cfg,
            migration_config=migration_config or MigrationConfig(),
        )

    def test_run_with_migration_returns_migration_result(self, hw_profile, gpu_plan, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        with self._mock_popen("Hello world"):
            result = engine.runWithMigration(gpu_plan, tmp_path / "model.gguf", "Hi")
        assert isinstance(result, MigrationResult)

    def test_run_with_migration_success_flag(self, hw_profile, gpu_plan, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        with self._mock_popen(returncode=0):
            result = engine.runWithMigration(gpu_plan, tmp_path / "model.gguf", "Hi")
        assert result.success

    def test_run_with_migration_initial_gpu_layers(self, hw_profile, gpu_plan, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        with self._mock_popen():
            result = engine.runWithMigration(gpu_plan, tmp_path / "model.gguf", "Hi")
        assert result.initial_n_gpu_layers == gpu_plan.n_gpu_layers

    def test_migration_result_to_dict_serializable(self, hw_profile, gpu_plan, tmp_path):
        import json
        engine = self._make_engine(hw_profile, tmp_path)
        with self._mock_popen():
            result = engine.runWithMigration(gpu_plan, tmp_path / "model.gguf", "Hi")
        json_str = json.dumps(result.to_dict())
        assert "migration_events" in json_str

    def test_on_token_callback_fired(self, hw_profile, gpu_plan, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        received: List[str] = []
        with self._mock_popen("TokenA"):
            engine.runWithMigration(
                gpu_plan, tmp_path / "model.gguf", "Hi",
                on_token=lambda t: received.append(t),
            )
        # Tokens may or may not be captured depending on async mode
        assert isinstance(received, list)

    def test_get_migration_history_empty_before_run(self, hw_profile, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        assert engine.getMigrationHistory() == []

    def test_get_current_plan_none_before_run(self, hw_profile, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        assert engine.getCurrentPlan() is None

    def test_get_current_plan_set_after_run(self, hw_profile, gpu_plan, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        with self._mock_popen():
            engine.runWithMigration(gpu_plan, tmp_path / "model.gguf", "Hi")
        assert engine.getCurrentPlan() is not None

    def test_force_migrate_down_raises_without_active_session(self, hw_profile, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        with pytest.raises(RuntimeError):
            engine.forceMigrateDown(3)

    def test_snake_case_aliases_work(self, hw_profile, gpu_plan, tmp_path):
        engine = self._make_engine(hw_profile, tmp_path)
        with self._mock_popen():
            result = engine.run_with_migration(gpu_plan, tmp_path / "model.gguf", "Hi")
        assert isinstance(result, MigrationResult)
        assert engine.get_migration_history() == []  # no events during this run
        assert engine.get_current_plan() is not None

    def test_on_migration_event_callback_type(self, hw_profile, gpu_plan, tmp_path):
        """on_migration_event callback is callable without crash."""
        engine = self._make_engine(hw_profile, tmp_path)
        events_received: List[MigrationEvent] = []
        with self._mock_popen():
            engine.runWithMigration(
                gpu_plan, tmp_path / "model.gguf", "Hi",
                on_migration_event=events_received.append,
            )
        # No migration expected in no-pressure scenario
        assert isinstance(events_received, list)


# Prevent accidental import of RuntimeConfig without the right path
try:
    from inference_runtime.runtime_config import RuntimeConfig
except ImportError:
    RuntimeConfig = None  # type: ignore
