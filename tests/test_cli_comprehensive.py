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
import cli.commands.run as run_cmd
import cli.commands.telemetry_cmd as telemetry_cmd
import cli.commands.stats_cmd as stats_cmd
import cli.commands.update_cmd as update_cmd
import cli.commands.plugins_cmd as plugins_cmd
import cli.commands.reset_cmd as reset_cmd
from cli.tui.chat_ui import ChatInterface
from cli.tui.settings_menu import InteractiveSettingsMenu
from server.runtime_adapter.adapter import get_runtime_adapter
from server.models.openai import ChatMessage
from inference_runtime.multiformat_engine import _HAS_ORT, _HAS_TORCH, _HAS_SAFETENSORS


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

    def test_knowledge_export_output_flag_forwarding(self, tmp_path):
        out_file = tmp_path / "custom_rkb.json"
        with patch.object(sys, "argv", ["inferenceos", "knowledge", "export", "--output", str(out_file)]):
            res = main()
            assert res == 0
            assert out_file.exists()

    def test_kv_policies_and_stats_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "kv", "policies"]):
            assert main() == 0
        with patch.object(sys, "argv", ["inferenceos", "kv", "stats"]):
            assert main() == 0

    def test_budget_history_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "budget", "history"]):
            assert main() == 0

    def test_health_monitor_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "health", "monitor"]):
            assert main() == 0

    def test_learning_stats_dispatch(self):
        with patch.object(sys, "argv", ["inferenceos", "learning", "stats"]):
            assert main() == 0

    def test_performance_detailed_actions_dispatch(self):
        for sub in ("score", "regressions", "analyze"):
            with patch.object(sys, "argv", ["inferenceos", "performance", sub]):
                assert main() == 0


class TestRunCommand:
    """Test single-shot model execution command."""

    @pytest.mark.skipif(not TINY_ONNX.exists() or not _HAS_ORT, reason="Tiny ONNX model or onnxruntime not present")
    def test_run_onnx_single_shot(self):
        run_cmd.handle_run_command(
            model_query=str(TINY_ONNX),
            prompt="Test prompt ONNX execution",
            max_tokens=5,
        )

    @pytest.mark.skipif(not TINY_ST.exists() or not (_HAS_TORCH and _HAS_SAFETENSORS), reason="Tiny SafeTensors model or PyTorch/SafeTensors not present")
    def test_run_safetensors_single_shot(self):
        run_cmd.handle_run_command(
            model_query=str(TINY_ST),
            prompt="Test prompt SafeTensors execution",
            max_tokens=5,
        )

    def test_run_missing_model_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            run_cmd.handle_run_command(model_query="non_existent_model_xyz.gguf")
        assert exc_info.value.code == 1

    def test_run_empty_prompt_exits(self):
        with patch("rich.console.Console.input", return_value=""):
            with pytest.raises(SystemExit) as exc_info:
                run_cmd.handle_run_command(
                    model_query=str(TINY_ONNX) if TINY_ONNX.exists() else str(QWEN_GGUF),
                    prompt=None,
                )
            assert exc_info.value.code == 0


class TestChatInterfaceSlashCommands:
    """Test ChatInterface slash commands and interaction loops."""

    def test_chat_interface_initialization(self):
        chat = ChatInterface()
        assert chat.theme_mgr is not None
        assert chat.history == []

    def test_chat_slash_commands(self, tmp_path):
        chat = ChatInterface()
        chat.history = [
            {"role": "user", "content": "Hello InferenceOS"},
            {"role": "assistant", "content": "Hello! How can I assist with local inference today?"},
        ]

        # /help
        assert chat._handle_slash_command("/help") is True

        # /history
        assert chat._handle_slash_command("/history") is True

        # /stats
        assert chat._handle_slash_command("/stats") is True

        # /context
        assert chat._handle_slash_command("/context") is True

        # /memory
        assert chat._handle_slash_command("/memory") is True

        # /config
        assert chat._handle_slash_command("/config") is True

        # /telemetry
        assert chat._handle_slash_command("/telemetry") is True

        # /save & /load
        save_file = tmp_path / "saved_session.json"
        assert chat._handle_slash_command(f"/save {save_file}") is True
        assert save_file.exists()

        chat.history = []
        assert chat._handle_slash_command(f"/load {save_file}") is True
        assert len(chat.history) == 2

        # /export
        export_file = tmp_path / "chat_export.md"
        assert chat._handle_slash_command(f"/export {export_file}") is True
        assert export_file.exists()

        # /reset
        assert chat._handle_slash_command("/reset") is True
        assert chat.history == []

        # /quit
        assert chat._handle_slash_command("/quit") is False


class TestModelsCommandEnhanced:
    """Test enhanced models add, tags parsing, and removal."""

    def test_models_add_positional_file_path(self, tmp_path):
        fake_model = tmp_path / "positional_model.safetensors"
        fake_model.write_text("DUMMY_SAFETENSORS")

        # Add using target as file path directly without --path or --nickname
        models_cmd.handle_models_command(action="add", target=str(fake_model))

        # Search to verify it was registered with stem as nickname
        registry = models_cmd.get_model_registry()
        model_entry = registry.get_model("positional_model")
        assert model_entry is not None
        assert model_entry["nickname"] == "positional_model"

        # Remove by file path
        models_cmd.handle_models_command(action="rm", target=str(fake_model))
        assert registry.get_model("positional_model") is None

    def test_models_add_with_comma_delimited_tags(self, tmp_path):
        fake_model = tmp_path / "tagged_model.gguf"
        fake_model.write_text("GGUF")

        models_cmd.handle_models_command(
            action="add",
            path=str(fake_model),
            nickname="tagged_model_nick",
            tags="quantized,production,edge",
        )

        registry = models_cmd.get_model_registry()
        entry = registry.get_model("tagged_model_nick")
        assert entry is not None
        assert "quantized" in entry["tags"]
        assert "production" in entry["tags"]
        assert "edge" in entry["tags"]

        # Clean up
        models_cmd.handle_models_command(action="rm", target="tagged_model_nick")


class TestUtilitiesAndTUI:
    """Test utility commands, logging, and settings menu."""

    def test_telemetry_command_runs(self):
        telemetry_cmd.handle_telemetry_command()

    def test_stats_command_runs(self):
        stats_cmd.handle_stats_command()

    def test_update_command_runs(self):
        update_cmd.handle_update_command()

    def test_plugins_command_runs(self):
        plugins_cmd.handle_plugins_command()

    def test_reset_command_clears_cache_and_config(self, tmp_path):
        from cli.core.config_manager import get_config_manager
        cfg = get_config_manager()
        # Create a dummy cache file
        dummy_cache = cfg.cache_dir / "test_cache.json"
        dummy_cache.write_text("{}")
        assert dummy_cache.exists()

        reset_cmd.handle_reset_command()
        assert not dummy_cache.exists()

    def test_logs_command_with_brackets(self, tmp_path):
        from cli.core.config_manager import get_config_manager
        cfg = get_config_manager()
        test_log = cfg.logs_dir / "test_markup.log"
        test_log.write_text("[2026-09-21 23:59:59] [INFO] [task-123] Server initialized with [special brackets]")
        logs_cmd.handle_logs_command(lines=5)

    def test_settings_menu_numbered_display(self):
        menu = InteractiveSettingsMenu()
        menu.display()


class TestServerRuntimeAdapterMultiFormat:
    """Test server runtime adapter with multi-format models."""

    @pytest.mark.skipif(not TINY_ONNX.exists() or not _HAS_ORT, reason="Tiny ONNX model or onnxruntime not present")
    def test_server_adapter_load_and_chat_onnx(self):
        adapter = get_runtime_adapter()
        loaded = adapter.load_model_if_needed(str(TINY_ONNX))
        assert loaded is not None
        assert loaded["model_path"] == TINY_ONNX

        msg = ChatMessage(role="user", content="Hello server ONNX test")
        res = adapter.execute_chat(
            model_query=str(TINY_ONNX),
            messages=[msg],
            request_overrides={"max_tokens": 5},
        )
        assert res is not None
        assert res.generated_text is not None
