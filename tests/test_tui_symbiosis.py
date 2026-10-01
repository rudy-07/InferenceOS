"""
test_tui_symbiosis.py
----------------------
Unit and integration tests for TUI ChatInterface across Standard, System 1, and Hybrid Symbiosis modes.

Validates:
1. Auto-discovery of System 1 and System 2 models on startup.
2. Format auto-detection routing (e.g. launching with System 1 model enters system1 mode).
3. Interactive mode switching via /mode (standard, system1, hybrid).
4. Dual-model selection via /s1 and /s2.
5. Escalation toggle (/escalate on|off) and threshold setting (/tau <float>).
6. Instant guardrail evaluation (/guardrail <action>) and option tool (/tool <options>).
7. LiveStatusPanel dynamic rendering across all three modes (0 KV cache for S1, dual-model for Hybrid).
8. Prompt processing in System 1 and Hybrid modes.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from cli.tui.chat_ui import ChatInterface, SLASH_COMMANDS
from cli.tui.live_status import LiveStatusPanel

LAYA_CHECKPOINT = Path(r"D:\Projects\llm\laya-browser\v14s")
QWEN_GGUF = Path(r"D:\Projects\mySphere projects\InferenceOS\models\Qwen3-0.6B-Q8_0.gguf")


def test_tui_initialization_defaults():
    """TUI auto-discovers available System 1 and System 2 models from registry."""
    chat = ChatInterface()
    assert chat.mode in ("standard", "system1", "hybrid")
    assert "/mode" in SLASH_COMMANDS
    assert "/s1" in SLASH_COMMANDS
    assert "/s2" in SLASH_COMMANDS
    assert "/escalate" in SLASH_COMMANDS
    assert "/tau" in SLASH_COMMANDS
    assert "/guardrail" in SLASH_COMMANDS
    assert "/tool" in SLASH_COMMANDS


def test_tui_initialization_with_system1_model():
    """Launching TUI with a System 1 model query enters system1 mode automatically."""
    if not LAYA_CHECKPOINT.exists():
        pytest.skip("laya:v14s not found")

    chat = ChatInterface(model_query="laya:v14s")
    assert chat.mode == "system1"
    assert chat.s1_model_path == LAYA_CHECKPOINT
    assert chat.model_path == LAYA_CHECKPOINT


def test_tui_mode_switching_slash_commands():
    """Switching modes via /mode command updates internal state and engine mode."""
    chat = ChatInterface()

    chat._handle_slash_command("/mode system1")
    assert chat.mode == "system1"

    chat._handle_slash_command("/mode hybrid")
    assert chat.mode == "hybrid"

    chat._handle_slash_command("/mode standard")
    assert chat.mode == "standard"


def test_tui_dual_model_selection_commands():
    """Assigning System 1 reflex model via /s1 and System 2 teacher model via /s2."""
    chat = ChatInterface()

    if LAYA_CHECKPOINT.exists():
        chat._handle_slash_command(f"/s1 {LAYA_CHECKPOINT}")
        assert chat.s1_model_path == LAYA_CHECKPOINT

    if QWEN_GGUF.exists():
        chat._handle_slash_command(f"/s2 {QWEN_GGUF}")
        assert chat.s2_model_path == QWEN_GGUF


def test_tui_escalation_and_tau_controls():
    """Toggling escalation on/off and updating tau via slash commands."""
    chat = ChatInterface()

    chat._handle_slash_command("/escalate off")
    assert chat.escalate is False

    chat._handle_slash_command("/escalate on")
    assert chat.escalate is True

    chat._handle_slash_command("/tau 0.88")
    assert chat.escalate_tau == 0.88

    chat._handle_slash_command("/teacher http://localhost:11434/v1")
    assert chat.s2_base_url == "http://localhost:11434/v1"


def test_tui_guardrail_and_tool_slash_commands(tmp_path, monkeypatch):
    """Executing /guardrail and /tool invokes the orchestrator methods correctly."""
    mock_file = tmp_path / "dummy_s1"
    mock_file.write_text("dummy")

    chat = ChatInterface()
    chat.s1_model_path = mock_file

    mock_guardrail = MagicMock(return_value={
        "allowed": True,
        "safety_score": 0.99,
        "latency_ms": 14.5,
    })
    mock_tool = MagicMock(return_value={
        "chosen": "Proceed",
        "confidence": 0.92,
        "latency_ms": 17.0,
    })

    monkeypatch.setattr(chat.orchestrator, "system1_guardrail", mock_guardrail)
    monkeypatch.setattr(chat.orchestrator, "system1_tool_decide", mock_tool)

    # Test /guardrail
    chat._handle_slash_command("/guardrail git commit -m 'update'")
    mock_guardrail.assert_called_once()
    assert mock_guardrail.call_args[1]["proposed_action"] == "git commit -m 'update'"

    # Test /tool
    chat._handle_slash_command("/tool Option A, Option B, Option C")
    mock_tool.assert_called_once()
    assert mock_tool.call_args[1]["options"] == ["Option A", "Option B", "Option C"]


def test_live_status_panel_rendering_all_modes():
    """Verify LiveStatusPanel renders valid Rich Panels across Standard, System 1, and Hybrid modes."""
    panel_mgr = LiveStatusPanel()

    # 1. Standard mode panel
    p_std = panel_mgr.render(
        mode="standard",
        model_name="Llama-3-8B.gguf",
        gen_tps=35.2,
        ttft_ms=18.0,
    )
    assert p_std is not None

    # 2. System 1 mode panel
    p_s1 = panel_mgr.render(
        mode="system1",
        s1_model_name="laya:v14s",
        last_decision="YES",
        last_confidence=0.88,
        last_latency_ms=18.4,
    )
    assert p_s1 is not None

    # 3. Hybrid Symbiosis mode panel
    p_hyb = panel_mgr.render(
        mode="hybrid",
        s1_model_name="laya:v14s",
        s2_model_name="Qwen3-0.6B-Q8_0.gguf",
        escalate_enabled=True,
        escalate_tau=0.75,
        last_decision="SUBMIT",
        last_confidence=0.95,
        last_latency_ms=25.0,
        last_escalated=True,
    )
    assert p_hyb is not None


def test_tui_prompt_processing_system1_mode(tmp_path, monkeypatch):
    """Submitting a prompt in System 1 mode invokes reflex prediction and updates TUI metrics."""
    mock_s1 = tmp_path / "mock_s1"
    mock_s1.write_text("dummy")

    chat = ChatInterface()
    chat.mode = "system1"
    chat.s1_model_path = mock_s1

    mock_predict = MagicMock(return_value={
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "YES",
                "confidence": 0.85,
                "probabilities": {"YES": 0.85, "NO": 0.15},
            }
        },
        "latency_ms": 19.2,
        "passes": 1,
        "backend": "system1-mock",
    })
    monkeypatch.setattr(chat.orchestrator.s1_engine, "predict", mock_predict)

    chat._process_user_prompt("Is the user clicking the confirm button?")
    assert chat.last_decision == "YES"
    assert chat.last_confidence == 0.85
    assert chat.last_latency_ms == 19.2
    assert chat.last_escalated is False


def test_tui_prompt_processing_hybrid_mode(tmp_path, monkeypatch):
    """Submitting a prompt in Hybrid mode invokes decide_with_escalation and updates TUI metrics."""
    mock_s1 = tmp_path / "mock_s1"
    mock_s1.write_text("dummy")
    mock_s2 = tmp_path / "mock_s2"
    mock_s2.write_text("dummy")

    chat = ChatInterface()
    chat.mode = "hybrid"
    chat.s1_model_path = mock_s1
    chat.s2_model_path = mock_s2

    mock_escalate = MagicMock(return_value={
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "SUBMIT",
                "confidence": 0.95,
                "system2": True,
                "rationale": "Teacher resolved ambiguity",
            }
        },
        "escalated": True,
        "escalate_reason": "Confidence below threshold",
        "latency_ms": 32.5,
        "s1_latency_ms": 18.0,
        "s2_latency_ms": 14.5,
        "generated_text": "Proceeding with purchase.",
    })
    monkeypatch.setattr(chat.orchestrator, "decide_with_escalation", mock_escalate)

    chat._process_user_prompt("Complete checkout with credit card")
    assert chat.last_decision == "SUBMIT"
    assert chat.last_confidence == 0.95
    assert chat.last_latency_ms == 32.5
    assert chat.last_escalated is True


def test_tui_smart_prompt_parsing():
    """TUI intelligently parses JSON, bracketed options, and freeform text."""
    chat = ChatInterface()

    # 1. Bracketed options parsing
    state, questions = chat._parse_prompt_to_state_and_questions("Options: [BUY, SELL, HOLD] Stock dropped 5%")
    crit = questions["decision"]["criteria"]
    assert "BUY" in crit
    assert "SELL" in crit
    assert "HOLD" in crit
    assert "Stock dropped 5%" in state or "BUY" in questions["decision"]["instructions"]["goal"]

    # 2. JSON input parsing
    json_prompt = '{"state": "User on checkout page", "questions": {"act": {"type": "choice", "criteria": {"A": "1", "B": "2"}}}}'
    s_json, q_json = chat._parse_prompt_to_state_and_questions(json_prompt)
    assert s_json == "User on checkout page"
    assert "act" in q_json

    # 3. Freeform fallback
    s_free, q_free = chat._parse_prompt_to_state_and_questions("Is this safe?")
    assert s_free == "Is this safe?"
    assert "YES" in q_free["decision"]["criteria"]
    assert "NO" in q_free["decision"]["criteria"]
    assert "UNCERTAIN" in q_free["decision"]["criteria"]


def test_tui_dagger_logging_controls(tmp_path):
    """Controlling DAgger dataset logging via /dagger on/off/status/path."""
    chat = ChatInterface()
    assert chat.dagger_log_path is None

    log_file = tmp_path / "custom_distill.jsonl"
    chat._handle_slash_command(f"/dagger on {log_file}")
    assert chat.dagger_log_path == log_file

    chat._handle_slash_command("/dagger status")
    chat._handle_slash_command("/dagger off")
    assert chat.dagger_log_path is None


def test_tui_subsystem_slash_commands(monkeypatch):
    """Verify integration of /doctor, /benchmark, /inspect, /health, /kv, /budget, /settings."""
    chat = ChatInterface()

    mock_doctor = MagicMock()
    mock_benchmark = MagicMock()
    mock_inspect = MagicMock()
    mock_health = MagicMock()
    mock_kv = MagicMock()
    mock_budget = MagicMock()

    monkeypatch.setattr("cli.commands.doctor.handle_doctor_command", mock_doctor)
    monkeypatch.setattr("cli.commands.benchmark.handle_benchmark_command", mock_benchmark)
    monkeypatch.setattr("cli.commands.inspect_cmd.handle_inspect_command", mock_inspect)
    monkeypatch.setattr("cli.health_cli.handle_health_cli", mock_health)
    monkeypatch.setattr("cli.kv_cli.handle_kv_cli", mock_kv)
    monkeypatch.setattr("cli.budget_cli.handle_budget_cli", mock_budget)

    chat._handle_slash_command("/doctor")
    mock_doctor.assert_called_once()

    chat._handle_slash_command("/benchmark dummy_model")
    mock_benchmark.assert_called_once()

    chat._handle_slash_command("/inspect dummy_model")
    mock_inspect.assert_called_once()

    chat._handle_slash_command("/health")
    mock_health.assert_called_once()

    chat._handle_slash_command("/kv")
    mock_kv.assert_called_once()

    chat._handle_slash_command("/budget")
    mock_budget.assert_called_once()


def test_tui_history_reset_and_save_load(tmp_path):
    """Test message history, /reset, /save, and /load for symbiosis state."""
    chat = ChatInterface()
    chat.mode = "hybrid"
    chat.escalate = True
    chat.escalate_tau = 0.82
    chat.history = [
        {"role": "user", "content": "Hello InferenceOS"},
        {"role": "assistant", "content": "Ready to assist."},
    ]

    # Verify history rendering
    chat._handle_slash_command("/history")

    # Test /save
    session_file = tmp_path / "test_session.json"
    chat._handle_slash_command(f"/save {session_file}")
    assert session_file.exists()

    # Test /reset
    chat._handle_slash_command("/reset")
    assert len(chat.history) == 0
    assert chat.last_decision is None

    # Test /load
    chat._handle_slash_command(f"/load {session_file}")
    assert len(chat.history) == 2
    assert chat.mode == "hybrid"
    assert chat.escalate_tau == 0.82


def test_tui_ctrl_c_interrupt_handling(tmp_path, monkeypatch):
    """Pressing Ctrl+C during streaming halts generation cleanly without crashing."""
    mock_s2 = tmp_path / "mock_s2.gguf"
    mock_s2.write_text("dummy")

    chat = ChatInterface()
    chat.mode = "standard"
    chat.model_path = mock_s2

    mock_session = MagicMock()
    def mock_run_interrupted(prompt, on_token=None):
        if on_token:
            on_token("Partial response before interrupt...")
        raise KeyboardInterrupt()

    mock_session.run = mock_run_interrupted
    chat.session = mock_session

    # Should not raise exception
    chat._process_user_prompt("Explain quantum computing")

    # Verify the partial token was recorded in history
    assert any("Partial response" in msg.get("content", "") for msg in chat.history)


def test_tui_error_resilience_and_crash_proofing():
    """Verify that malformed commands and edge cases do not crash the TUI."""
    chat = ChatInterface()

    # 1. Invalid tau values
    chat._handle_slash_command("/tau not_a_float")
    assert chat.escalate_tau == 0.70  # Default remains
    chat._handle_slash_command("/tau -5.0")
    assert chat.escalate_tau == 0.0  # Clamped
    chat._handle_slash_command("/tau 15.0")
    assert chat.escalate_tau == 1.0  # Clamped

    # 2. Invalid modes
    chat._handle_slash_command("/mode invalid_mode_xyz")
    assert chat.mode != "invalid_mode_xyz"

    # 3. Invalid model resolutions
    chat._handle_slash_command("/s1 non_existent_s1_model_xyz")
    chat._handle_slash_command("/s2 non_existent_s2_model_xyz")
    chat._handle_slash_command("/model non_existent_model_xyz")

    # 4. Guardrail and tool with missing models or empty args
    chat.s1_model_path = None
    chat._handle_slash_command("/guardrail")
    chat._handle_slash_command("/guardrail sudo rm -rf /")
    chat._handle_slash_command("/tool")
    chat._handle_slash_command("/tool opt1, opt2")

    # 5. Missing /load file
    chat._handle_slash_command("/load non_existent_file.json")

