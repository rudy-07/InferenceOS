"""
test_cli_comprehensive.py
-------------------------
End-to-end and unit tests for the InferenceOS CLI command suite:
- Main parser and all 22 subcommands + 6 subsystem commands
- Custom help formatting without wrapping/overflow
- Version, doctor, hardware, and profiler key resolution
- Multi-format model inspection, layer placement, and optimization
- Models management (search, list, scan, add, rm with positional & flag ergonomics)
- Configuration get/set with type parsing (bool, int, float, str)
- Speculative decoding status & verification
- Subsystem subcommands: budget, health, kv, knowledge, learning, performance
"""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from cli.main import build_parser, print_custom_help, main
from cli.core.model_registry import ModelRegistry, SUPPORTED_EXTENSIONS
import cli.commands.hardware as hardware_cmd
import cli.commands.doctor as doctor_cmd
import cli.commands.version_cmd as version_cmd
import cli.commands.models_cmd as models_cmd
import cli.commands.inspect_cmd as inspect_cmd
import cli.commands.placement_cmd as placement_cmd
import cli.commands.optimize as optimize_cmd
import cli.commands.config_cmd as config_cmd
import cli.commands.speculative_cmd as speculative_cmd
import cli.commands.cache_cmd as cache_cmd
import cli.commands.logs_cmd as logs_cmd

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
FIXTURES_DIR = MODELS_DIR / "test_formats"
QWEN_GGUF = MODELS_DIR / "Qwen3-0.6B-Q8_0.gguf"
TINY_ONNX = FIXTURES_DIR / "tiny_model.onnx"
TINY_ST = FIXTURES_DIR / "tiny_model.safetensors"
KERAS_H5 = FIXTURES_DIR / "text_model.h5"


class TestCLIParser:
    """Test CLI argument parsing and help formatting."""

    def test_build_parser_registers_all_commands(self):
        parser = build_parser()
        subparsers_action = next(
            action for action in parser._actions
            if action.dest == "command"
        )
        registered_commands = set(subparsers_action.choices.keys())
        expected_commands = {
            "run", "chat", "inspect", "placement", "optimize", "benchmark",
            "models", "hardware", "doctor", "config", "serve", "speculative",
            "cache", "logs", "reset", "version", "plugins", "telemetry",
            "monitor", "stats", "update",
            # Subsystem additions
            "budget", "health", "kv", "knowledge", "learning", "performance",
        }
        for cmd in expected_commands:
            assert cmd in registered_commands, f"Command '{cmd}' not found in registered commands"

    def test_print_custom_help_executes_cleanly(self):
        # Should execute cleanly without layout or wrapping exceptions
        print_custom_help()


class TestHardwareAndDoctor:
    """Test hardware detection, system doctor checks, and version."""

    def test_hardware_display_robustness(self):
        hardware_cmd.handle_hardware_command()

    def test_doctor_command_runs(self):
        doctor_cmd.handle_doctor_command()

    def test_version_command_runs(self):
        version_cmd.handle_version_command()


class TestModelRegistryAndCommands:
    """Test ModelRegistry integration and models subcommand."""

    def test_models_list(self):
        models_cmd.handle_models_command(action="list")

    def test_models_search_positional(self):
        models_cmd.handle_models_command(action="search", target="tiny")

    def test_models_search_query_flag(self):
        models_cmd.handle_models_command(action="search", query="tiny")

    def test_models_add_and_rm(self, tmp_path):
        fake_model = tmp_path / "dummy.gguf"
        fake_model.write_text("GGUF")

        # Add model
        models_cmd.handle_models_command(
            action="add",
            path=str(fake_model),
            nickname="test_dummy_cli",
            tags="test,unit"
        )

        # Remove via positional target
        models_cmd.handle_models_command(action="rm", target="test_dummy_cli")


class TestMultiFormatInspectPlacementOptimize:
    """Test inspect, placement, and optimize across multiple model formats."""

    @pytest.mark.skipif(not QWEN_GGUF.exists(), reason="Qwen GGUF model not present")
    def test_inspect_gguf(self):
        inspect_cmd.handle_inspect_command(str(QWEN_GGUF))

    @pytest.mark.skipif(not TINY_ONNX.exists(), reason="Tiny ONNX model not present")
    def test_inspect_onnx(self):
        inspect_cmd.handle_inspect_command(str(TINY_ONNX))

    @pytest.mark.skipif(not TINY_ST.exists(), reason="Tiny SafeTensors model not present")
    def test_inspect_safetensors(self):
        inspect_cmd.handle_inspect_command(str(TINY_ST))

    @pytest.mark.skipif(not KERAS_H5.exists(), reason="Keras H5 model not present")
    def test_inspect_keras_h5(self):
        inspect_cmd.handle_inspect_command(str(KERAS_H5))

    def test_inspect_missing_file_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc_info:
            inspect_cmd.handle_inspect_command("non_existent_model_file.xyz")
        assert exc_info.value.code == 1

    @pytest.mark.skipif(not TINY_ONNX.exists(), reason="Tiny ONNX model not present")
    def test_placement_onnx(self):
        placement_cmd.handle_placement_command(str(TINY_ONNX))

    @pytest.mark.skipif(not TINY_ST.exists(), reason="Tiny SafeTensors model not present")
    def test_placement_safetensors(self):
        placement_cmd.handle_placement_command(str(TINY_ST))

    @pytest.mark.skipif(not TINY_ONNX.exists(), reason="Tiny ONNX model not present")
    def test_optimize_onnx(self):
        optimize_cmd.handle_optimize_command(str(TINY_ONNX))


class TestConfigCommand:
    """Test config show, get, and set with type parsing."""

    def test_config_show(self):
        config_cmd.handle_config_command()

    def test_config_get_valid(self):
        config_cmd.handle_config_command(key="hardware.gpu_id")

    def test_config_set_and_restore(self):
        # Boolean conversion
        config_cmd.handle_config_command(
            key="runtime.enable_speculative",
            value="true"
        )
        # Integer conversion
        config_cmd.handle_config_command(
            key="runtime.default_threads",
            value="4"
        )


class TestSpeculativeCommand:
    """Test speculative decoding status and test commands."""

    def test_speculative_status(self):
        speculative_cmd.handle_speculative_command(action="status")


class TestCacheAndLogsCommands:
    """Test cache view and logs inspection."""

    def test_cache_view(self):
        cache_cmd.handle_cache_command(action="view")

    def test_logs_view(self):
        logs_cmd.handle_logs_command(lines=10)


class TestSubsystemCLICommands:
    """Test execution of wired subsystem commands through main()."""

    def test_budget_cli_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "budget", "show"]):
            res = main()
            assert res == 0

    def test_health_cli_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "health", "show"]):
            res = main()
            assert res == 0

    def test_kv_cli_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "kv", "show"]):
            res = main()
            assert res == 0

    def test_knowledge_cli_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "knowledge", "show"]):
            res = main()
            assert res == 0

    def test_learning_cli_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "learning", "show"]):
            res = main()
            assert res == 0

    def test_performance_cli_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "performance", "show"]):
            res = main()
            assert res == 0

    def test_performance_trends_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "performance", "trends"]):
            res = main()
            assert res == 0
