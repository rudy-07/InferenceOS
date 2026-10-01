"""
test_modern_tui.py
------------------
Exhaustive test suite for Next-Gen Modern TUI revamp in InferenceOS.

Validates:
1. Dynamic Floating Autocompleter (InferenceOSCompleter):
   - Slash command completions with categories & metadata
   - Sub-argument completions (/mode, /s1_device, /escalate, /profile, /backend)
   - Dynamic model nickname completions queried from ModelRegistry (/model, /s1, /s2)
2. Interactive Modal Selectors (InteractivePicker):
   - Non-interactive TTY fallback & matching
   - pick_model, pick_mode, pick_device, pick_profile
3. Docked Bottom Status Toolbar (BottomToolbarBuilder):
   - HTML rendering across Standard, System 1, and Hybrid modes
   - Dynamic VRAM, tok/s, and hotkey indicator updates
4. Next-Gen Chat Renderer (ChatRenderer):
   - User message accent rail
   - System 1 reflex cards with probability bars
   - Hybrid deliberation flowcards
   - Token streaming performance pill
5. Full ChatInterface integration with PromptSession & KeyBindings
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from prompt_toolkit.document import Document
from rich.console import Console

from cli.tui.completer import InferenceOSCompleter, COMMAND_METADATA
from cli.tui.interactive_picker import (
    InteractivePicker,
    pick_model,
    pick_mode,
    pick_device,
    pick_profile,
)
from cli.tui.bottom_toolbar import BottomToolbarBuilder
from cli.tui.chat_renderer import ChatRenderer
from cli.tui.chat_ui import ChatInterface


# ---------------------------------------------------------------------------
# 1. Dynamic Floating Completer Tests
# ---------------------------------------------------------------------------

def test_completer_slash_prefix_matches():
    """Typing '/' returns all available slash commands with category tags."""
    completer = InferenceOSCompleter()
    doc = Document(text="/", cursor_position=1)
    completions = list(completer.get_completions(doc, None))

    assert len(completions) == len(COMMAND_METADATA)
    texts = [c.text.strip() for c in completions]
    assert "/mode" in texts
    assert "/model" in texts
    assert "/s1" in texts
    assert "/s1_device" in texts
    assert "/escalate" in texts


def test_completer_fuzzy_filtering():
    """Typing '/m' narrows completions down to /mode, /model, /models, /memory."""
    completer = InferenceOSCompleter()
    doc = Document(text="/m", cursor_position=2)
    completions = list(completer.get_completions(doc, None))

    texts = [c.text.strip() for c in completions]
    assert "/mode" in texts
    assert "/model" in texts
    assert "/models" in texts
    assert "/memory" in texts
    assert "/s1" not in texts


def test_completer_sub_arguments():
    """Typing '/mode ' offers standard, system1, and hybrid options."""
    completer = InferenceOSCompleter()
    doc = Document(text="/mode ", cursor_position=6)
    completions = list(completer.get_completions(doc, None))

    texts = [c.text for c in completions]
    assert "standard" in texts
    assert "system1" in texts
    assert "hybrid" in texts

    # Filtered sub-arg
    doc_hy = Document(text="/mode hy", cursor_position=8)
    comp_hy = list(completer.get_completions(doc_hy, None))
    assert len(comp_hy) == 1
    assert comp_hy[0].text == "hybrid"


def test_completer_s1_device_sub_arguments():
    """Typing '/s1_device ' offers auto, gpu, cpu."""
    completer = InferenceOSCompleter()
    doc = Document(text="/s1_device ", cursor_position=11)
    completions = list(completer.get_completions(doc, None))

    texts = [c.text for c in completions]
    assert "auto" in texts
    assert "gpu" in texts
    assert "cpu" in texts


def test_completer_dynamic_models_from_registry():
    """Typing '/s1 ' suggests only System 1 models from registry."""
    mock_registry = MagicMock()
    mock_registry.list_models.return_value = [
        {"nickname": "laya:v14s", "format": "system1", "size_bytes": 600 * 1024**2, "location": "/path/laya"},
        {"nickname": "qwen3-4b", "format": "gguf", "size_bytes": 2400 * 1024**2, "location": "/path/qwen"},
    ]

    completer = InferenceOSCompleter(registry_getter=lambda: mock_registry)

    # For /s1, should only return laya:v14s
    doc_s1 = Document(text="/s1 ", cursor_position=4)
    comp_s1 = list(completer.get_completions(doc_s1, None))
    texts_s1 = [c.text for c in comp_s1]
    assert "laya:v14s" in texts_s1
    assert "qwen3-4b" not in texts_s1

    # For /s2, should only return qwen3-4b
    doc_s2 = Document(text="/s2 ", cursor_position=4)
    comp_s2 = list(completer.get_completions(doc_s2, None))
    texts_s2 = [c.text for c in comp_s2]
    assert "qwen3-4b" in texts_s2
    assert "laya:v14s" not in texts_s2


# ---------------------------------------------------------------------------
# 2. Interactive Picker Tests
# ---------------------------------------------------------------------------

def test_interactive_picker_fallback_mode():
    """Picker in non-interactive environment returns matched current_value or first item."""
    picker = InteractivePicker()
    items = [
        {"label": "Standard", "value": "standard"},
        {"label": "System 1", "value": "system1"},
        {"label": "Hybrid", "value": "hybrid"},
    ]

    # Non-interactive matches current_value
    chosen = picker.select("Select Mode", items, current_value="hybrid")
    assert chosen is not None
    assert chosen["value"] == "hybrid"

    # Matches label if value not exact
    chosen_label = picker.select("Select Mode", items, current_value="System 1")
    assert chosen_label is not None
    assert chosen_label["value"] == "system1"


def test_pick_mode_domain_helper():
    """pick_mode helper returns valid mode string."""
    mode = pick_mode(current_mode="hybrid")
    assert mode == "hybrid"

    mode_s1 = pick_mode(current_mode="system1")
    assert mode_s1 == "system1"


def test_pick_device_domain_helper():
    """pick_device helper returns valid device string."""
    dev = pick_device(current_device="gpu")
    assert dev == "gpu"

    dev_cpu = pick_device(current_device="cpu")
    assert dev_cpu == "cpu"


def test_pick_profile_domain_helper():
    """pick_profile helper returns valid performance profile."""
    prof = pick_profile(current_profile="low-latency")
    assert prof == "low-latency"


def test_pick_model_domain_helper():
    """pick_model queries registry and resolves Path in fallback mode."""
    mock_reg = MagicMock()
    mock_reg.list_models.return_value = [
        {"nickname": "test_s1", "format": "system1", "size_bytes": 500, "location": "/mock/test_s1"},
        {"nickname": "test_s2", "format": "gguf", "size_bytes": 1000, "location": "/mock/test_s2"},
    ]

    chosen_s1 = pick_model(mock_reg, filter_kind="system1", current_path="/mock/test_s1")
    assert chosen_s1 == Path("/mock/test_s1")

    chosen_s2 = pick_model(mock_reg, filter_kind="system2", current_path="/mock/test_s2")
    assert chosen_s2 == Path("/mock/test_s2")


# ---------------------------------------------------------------------------
# 3. Docked Bottom Status Toolbar Tests
# ---------------------------------------------------------------------------

def test_bottom_toolbar_html_generation_standard():
    """Bottom toolbar formats HTML correctly for Standard chat mode."""
    state = {
        "mode": "standard",
        "s2_model_name": "Qwen3-4B.gguf",
        "backend": "VULKAN",
        "vram_used_mb": 3120.0,
        "vram_total_mb": 6144.0,
        "last_tps": 28.5,
    }
    builder = BottomToolbarBuilder(lambda: state)
    html_out = builder()
    raw = html_out.value

    assert "STANDARD CHAT" in raw
    assert "Qwen3-4B.gguf" in raw
    assert "VULKAN" in raw
    assert "28.5 t/s" in raw
    assert "3.0/6.0GB" in raw
    assert "^O" in raw
    assert "^T" in raw


def test_bottom_toolbar_html_generation_hybrid():
    """Bottom toolbar formats HTML correctly for Hybrid Symbiosis mode."""
    state = {
        "mode": "hybrid",
        "s1_model_name": "laya:v14s",
        "s2_model_name": "Qwen3-4B.gguf",
        "s1_device": "GPU",
        "vram_used_mb": 3720.0,
        "vram_total_mb": 6144.0,
        "last_tps": 32.1,
        "last_latency_ms": 380.0,
        "tau": 0.85,
    }
    builder = BottomToolbarBuilder(lambda: state)
    html_out = builder()
    raw = html_out.value

    assert "HYBRID SYMBIOSIS" in raw
    assert "laya:v14s" in raw
    assert "Qwen3-4B.gguf" in raw
    assert "0.85" in raw
    assert "32.1 t/s" in raw


# ---------------------------------------------------------------------------
# 4. Next-Gen Chat Renderer Tests
# ---------------------------------------------------------------------------

def test_chat_renderer_system1_card():
    """ChatRenderer renders System 1 reflex card with probability bars without errors."""
    mock_console = MagicMock()
    renderer = ChatRenderer(mock_console)

    res = {
        "latency_ms": 132.5,
        "answers": {
            "action": {
                "type": "choice",
                "choice": "PROCEED",
                "confidence": 0.94,
                "probabilities": {"PROCEED": 0.94, "CANCEL": 0.04, "WAIT": 0.02},
            },
            "safety": {
                "type": "noul",
                "noul": 0.88,
                "confidence": 0.88,
            },
            "score": {
                "type": "score",
                "score": 4.5,
                "confidence": 0.91,
            },
        },
    }

    renderer.render_system1_card(res, state_summary="User on checkout page")
    assert mock_console.print.called


def test_chat_renderer_hybrid_flowcard():
    """ChatRenderer renders multi-stage Kahneman deliberation flowcard without errors."""
    mock_console = MagicMock()
    renderer = ChatRenderer(mock_console)

    res = {
        "escalated": True,
        "escalate_reason": "Low confidence (0.62 < tau=0.85)",
        "total_latency_ms": 375.0,
        "system1_latency_ms": 128.0,
        "system2_latency_ms": 247.0,
        "answers": {
            "action": {"choice": "PROCEED", "confidence": 0.98, "rationale": "Order verified."}
        },
        "generated_text": "Proceeding with order confirmation.",
    }

    renderer.render_hybrid_flowcard(res, state_summary="Cart contains 2 items.")
    assert mock_console.print.called


def test_chat_renderer_streaming_pill():
    """ChatRenderer renders streaming performance pill with tok/s and TTFT."""
    mock_console = MagicMock()
    renderer = ChatRenderer(mock_console)

    stats = MagicMock()
    stats.eval_tps = 32.4
    stats.prompt_eval_tps = 180.2
    stats.prompt_eval_ms = 41.5
    stats.tokens_generated = 120

    renderer.render_streaming_pill(stats, backend="vulkan")
    assert mock_console.print.called


# ---------------------------------------------------------------------------
# 5. Full ChatInterface Modern Integration Tests
# ---------------------------------------------------------------------------

def test_chat_interface_initializes_prompt_session():
    """ChatInterface initializes PromptSession, completer, renderer, and bottom toolbar."""
    chat = ChatInterface()

    assert hasattr(chat, "renderer")
    assert isinstance(chat.renderer, ChatRenderer)
    assert hasattr(chat, "completer")
    assert isinstance(chat.completer, InferenceOSCompleter)
    assert hasattr(chat, "prompt_session")
    assert chat.prompt_session is not None
    assert hasattr(chat, "bottom_toolbar")


def test_chat_interface_slash_command_picker_fallbacks():
    """Slash commands (/mode, /s1_device, /profile) route through pickers gracefully."""
    chat = ChatInterface()

    # /mode without args invokes picker
    chat._handle_slash_command("/mode")
    assert chat.mode in ("standard", "system1", "hybrid")

    # /s1_device without args invokes picker
    chat._handle_slash_command("/s1_device")
    assert chat.s1_device in ("auto", "gpu", "cpu")

    # /profile without args invokes picker
    chat._handle_slash_command("/profile")


def test_chat_interface_ctrl_t_and_keybindings_registered():
    """Verify that c-t (mode toggle) and c-o (model picker) are properly registered in prompt_session."""
    chat = ChatInterface()
    assert chat.prompt_session is not None
    bindings = chat.prompt_session.key_bindings
    # Check that c-t and c-o are registered
    key_tuples = [b.keys for b in bindings.bindings]
    has_ct = any(("c-t",) == k for k in key_tuples)
    has_co = any(("c-o",) == k for k in key_tuples)
    has_cl = any(("c-l",) == k for k in key_tuples)
    assert has_ct, "Expected c-t to be registered for mode switcher"
    assert has_co, "Expected c-o to be registered for model picker"
    assert has_cl, "Expected c-l to be registered for clear screen"


def test_chat_interface_enter_and_empty_prompts_non_crashing():
    """ChatInterface handles empty and whitespace-only prompts without crashing or state corruption."""
    chat = ChatInterface()
    initial_len = len(chat.history)
    chat._process_user_prompt("")
    chat._process_user_prompt("   ")
    assert len(chat.history) == initial_len


def test_chat_interface_unknown_slash_command():
    """Unknown slash command displays error and does not terminate session."""
    chat = ChatInterface()
    result = chat._handle_slash_command("/non_existent_command_123")
    assert result is True  # True means continue session loop


def test_interactive_picker_adversarial_inputs():
    """InteractivePicker gracefully handles empty items, missing fields, and bad types."""
    picker = InteractivePicker()

    # Empty list
    assert picker.select("Test Empty", []) is None

    # Items with missing keys
    bad_items = [{"foo": "bar"}, {"label": "Valid Item", "value": 42}]
    res = picker.select("Test Bad Items", bad_items, current_value="42")
    assert res is not None
    assert res.get("value") == 42


def test_completer_all_known_commands_have_categories():
    """Every command in COMMAND_METADATA has valid category and description."""
    for cmd, info in COMMAND_METADATA.items():
        assert cmd.startswith("/"), f"Command {cmd} must start with /"
        assert "desc" in info and len(info["desc"]) > 0
        assert "category" in info and len(info["category"]) > 0


def test_chat_interface_all_modes_process_prompt(tmp_path, monkeypatch):
    """Verify prompt processing across System 1, Hybrid, and Standard modes with renderer."""
    chat = ChatInterface()

    # 1. System 1 mode
    chat.orchestrator.s1_engine.predict = MagicMock(return_value={
        "answers": {
            "action": {
                "type": "choice",
                "choice": "PROCEED",
                "confidence": 0.95,
                "probabilities": {"PROCEED": 0.95, "CANCEL": 0.05},
            }
        },
        "latency_ms": 15.0,
    })
    chat.s1_model_path = tmp_path / "mock_s1"
    chat.s1_model_path.write_text("mock")
    chat.mode = "system1"

    chat._process_user_prompt("Options: [PROCEED, CANCEL] Can user checkout?")
    assert chat.last_decision == "PROCEED"
    assert chat.last_confidence == 0.95

    # 2. Hybrid mode
    chat.mode = "hybrid"
    chat.orchestrator.decide_with_escalation = MagicMock(return_value={
        "answers": {"decision": {"choice": "CONFIRM", "confidence": 0.99}},
        "escalated": False,
        "latency_ms": 20.0,
        "s1_latency_ms": 20.0,
    })
    chat._process_user_prompt("Confirm transaction")
    assert chat.last_decision == "CONFIRM"


def test_picker_interactive_search_and_navigation(monkeypatch):
    """Test interactive key simulation: 'q' search works, digits jump in non-searchable, home/end navigate."""
    from cli.tui import interactive_picker

    items = [
        {"label": "Qwen3-4B", "value": "qwen3-4b"},
        {"label": "Laya-v14s", "value": "laya-v14s"},
        {"label": "DeepSeek-R1", "value": "deepseek-r1"},
    ]

    picker = InteractivePicker()

    # 1. Searchable mode: simulate typing 'q' then Enter -> must select Qwen3-4B, NOT exit with None
    key_sequence = iter(["q", "enter"])
    monkeypatch.setattr(interactive_picker, "_read_key", lambda: next(key_sequence))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    res = picker.select("Model Picker", items, searchable=True)
    assert res is not None
    assert res["value"] == "qwen3-4b"

    # 2. Non-searchable mode: simulate typing 'q' -> must cancel (return None)
    key_seq_q = iter(["q"])
    monkeypatch.setattr(interactive_picker, "_read_key", lambda: next(key_seq_q))
    res_cancel = picker.select("Mode Picker", items, searchable=False)
    assert res_cancel is None

    # 3. Non-searchable mode: simulate digit '2' then Enter -> must jump to Laya-v14s and confirm
    key_seq_jump = iter(["2", "enter"])
    monkeypatch.setattr(interactive_picker, "_read_key", lambda: next(key_seq_jump))
    res_jump = picker.select("Mode Picker", items, searchable=False)
    assert res_jump is not None
    assert res_jump["value"] == "laya-v14s"

    # 4. Navigation: End key jumps to last item (DeepSeek-R1)
    key_seq_nav = iter(["end", "enter"])
    monkeypatch.setattr(interactive_picker, "_read_key", lambda: next(key_seq_nav))
    res_nav = picker.select("Mode Picker", items, searchable=False)
    assert res_nav is not None
    assert res_nav["value"] == "deepseek-r1"


def test_bottom_toolbar_html_injection_and_crash_proofing():
    """Bottom toolbar handles XML tags, ampersands, and state provider errors without raising."""
    # State with potential HTML injection characters
    evil_state = {
        "mode": "standard",
        "s2_model_name": "<Dangerous&Model>",
        "s1_device": "<GPU&CUDA>",
        "backend": "<VULKAN>",
        "vram_used_mb": 2048.0,
        "vram_total_mb": 6144.0,
        "last_tps": 30.0,
    }
    builder = BottomToolbarBuilder(lambda: evil_state)
    html_out = builder()
    assert html_out is not None
    assert "&lt;Dangerous&amp;Model&gt;" in html_out.value

    # Crashing state provider
    def crashing_provider():
        raise RuntimeError("Hardware monitor exploded!")

    crash_builder = BottomToolbarBuilder(crashing_provider)
    crash_out = crash_builder()
    assert crash_out is not None
    assert ("STANDARD CHAT" in crash_out.value or "InferenceOS" in crash_out.value)


def test_chat_renderer_malformed_and_edge_case_inputs():
    """ChatRenderer handles dict stats, None objects, and extreme probability values safely."""
    renderer = ChatRenderer()

    # 1. streaming pill with dictionary
    renderer.render_streaming_pill({"eval_tps": 45.2, "prompt_eval_ms": 12.0, "tokens_generated": 100}, backend="cuda")

    # 2. streaming pill with None
    renderer.render_streaming_pill(None, backend="vulkan")

    # 3. reflex card with string confidence and abnormal probability floats
    renderer.render_system1_card({
        "latency_ms": 14.2,
        "answers": {
            "test_q": {
                "type": "choice",
                "choice": "A",
                "confidence": "0.92",
                "probabilities": {"A": 1.5, "B": -0.2, "C": None},
            },
            "score_q": {"type": "score", "score": "4.5", "confidence": None},
            "gate_q": {"type": "noul", "noul": 0.88},
        }
    })

    # 4. hybrid flowcard with empty answers
    renderer.render_hybrid_flowcard({
        "answers": {},
        "escalated": True,
        "escalation_reason": "Low confidence",
        "generated_text": "# Markdown synthesized text\n```python\nprint(1)\n```",
    })


def test_chat_interface_export_json_and_md(tmp_path):
    """Test /export command outputs valid JSON and Markdown files."""
    chat = ChatInterface()
    chat.history = [
        {"role": "user", "content": "Hello InferenceOS"},
        {"role": "assistant", "content": "Hello! How can I assist?"},
    ]

    # Export to JSON
    json_path = tmp_path / "test_session.json"
    chat._handle_slash_command(f"/export {json_path}")
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["mode"] == "standard"
        assert len(data["history"]) == 2

    # Export to Markdown
    md_path = tmp_path / "test_session.md"
    chat._handle_slash_command(f"/export {md_path}")
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "InferenceOS Chat Session Export" in content
    assert "Hello InferenceOS" in content


def test_chat_interface_status_and_mode_aliases():
    """Verify /status works as alias for /stats and mode aliases switch properly."""
    chat = ChatInterface()

    # /status alias
    assert chat._handle_slash_command("/status") is True

    # Mode aliases
    chat._handle_slash_command("/mode reflex")
    assert chat.mode == "system1"

    chat._handle_slash_command("/mode symbiosis")
    assert chat.mode == "hybrid"

    chat._handle_slash_command("/mode chat")
    assert chat.mode == "standard"


def test_chat_interface_ctrl_c_buffer_logic():
    """Verify Ctrl+C keybinding clears buffer when text is present, or exits if empty."""
    chat = ChatInterface()
    assert chat.prompt_session is not None

    kb = chat.prompt_session.key_bindings
    # Find c-c binding
    c_c_binding = None
    for b in kb.bindings:
        for k in b.keys:
            if str(k) in ("c-c", "Keys.ControlC"):
                c_c_binding = b
                break

    assert c_c_binding is not None

    # Simulate buffer with text
    mock_app = MagicMock()
    mock_app.current_buffer.text = "some pending text"
    mock_event = MagicMock(app=mock_app)

    c_c_binding.handler(mock_event)
    assert mock_app.current_buffer.text == ""  # Cleared!
    mock_app.exit.assert_not_called()

    # Simulate empty buffer
    mock_app.current_buffer.text = ""
    c_c_binding.handler(mock_event)
    mock_app.exit.assert_called_with(result="/exit")


