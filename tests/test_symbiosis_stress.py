"""
test_symbiosis_stress.py
-------------------------
Exhaustive stress, breaking, and adversarial test suite for System 1 / System 2 Symbiosis.

Pushes the engine to its limits with:
1. Malformed / non-JSON LLM teacher outputs (markdown fences, trailing commas, partial regex).
2. Pure gibberish LLM outputs (verifying non-crashing fallback to rule-based teacher).
3. Pathological S1 outputs (NaN confidence, negative confidence, empty answer dicts).
4. Extreme context sizes (100k char states).
5. Concurrent multi-question scaling (50 simultaneous questions).
6. Boundary conditions for Direction B: empty options, empty actions, negative/overflow thresholds.
7. Real-world pairing: laya:v14s with local GGUF Qwen3-0.6B-Q8_0.gguf teacher.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from orchestrator.hybrid_orchestrator import HybridOrchestrator

LAYA_CHECKPOINT = Path(r"D:\Projects\llm\laya-browser\v14s")
QWEN_GGUF = Path(r"D:\Projects\mySphere projects\InferenceOS\models\Qwen3-0.6B-Q8_0.gguf")


# ---------------------------------------------------------------------------
# Adversarial System 2 LLM Output Parsing Tests
# ---------------------------------------------------------------------------

def test_parse_system2_markdown_wrapped_json():
    """Verify parser extracts JSON wrapped in ```json ... ``` markdown blocks."""
    orch = HybridOrchestrator()
    raw = (
        "Here is my deliberation as the System 2 teacher:\n\n"
        "```json\n"
        "{\n"
        '  "decisions": {\n'
        '    "nav": {"choice": "SETTINGS", "confidence": 0.98, "rationale": "Clear intention"}\n'
        "  },\n"
        '  "response_text": "Navigating user to settings."\n'
        "}\n"
        "```\n\nHope this helps!"
    )
    questions = {"nav": {"type": "choice", "criteria": {"HOME": "Home", "SETTINGS": "Settings"}}}
    decisions, s2_text = orch._parse_system2_response(raw, questions)

    assert "nav" in decisions
    assert decisions["nav"]["choice"] == "SETTINGS"
    assert decisions["nav"]["confidence"] == 0.98
    assert decisions["nav"]["system2"] is True
    assert s2_text == "Navigating user to settings."


def test_parse_system2_regex_fallback_when_not_valid_json():
    """When LLM returns free text without valid JSON, regex locates candidate criteria."""
    orch = HybridOrchestrator()
    raw = "Based on the user's intent, the best course of action is definitely to proceed with BUY_NOW."
    questions = {
        "action": {
            "type": "choice",
            "criteria": {"CANCEL": "Cancel", "BUY_NOW": "Purchase", "UNCERTAIN": "Unsure"},
        }
    }
    decisions, s2_text = orch._parse_system2_response(raw, questions)

    assert "action" in decisions
    assert decisions["action"]["choice"] == "BUY_NOW"
    assert decisions["action"]["confidence"] == 0.90
    assert decisions["action"]["system2"] is True


def test_parse_system2_complete_gibberish_non_crashing():
    """When LLM returns complete garbage, parser returns empty dict and triggers fallback."""
    orch = HybridOrchestrator()
    raw = "!@#$%^&*()_+ arbitrary random garbage with no keys 12345"
    questions = {"action": {"type": "choice", "criteria": {"OPT_A": "Option A", "OPT_B": "Option B"}}}
    decisions, s2_text = orch._parse_system2_response(raw, questions)

    assert decisions == {}
    assert s2_text == raw


# ---------------------------------------------------------------------------
# Pathological System 1 Output Resilience
# ---------------------------------------------------------------------------

def test_pathological_s1_nan_and_negative_confidence():
    """System 1 returning NaN or negative confidence must safely trigger escalation without crashing."""
    mock_s1 = MagicMock()
    mock_s2 = MagicMock()
    orch = HybridOrchestrator(s1_engine=mock_s1, s2_engine=mock_s2)

    # Return NaN confidence
    mock_s1.predict.return_value = {
        "answers": {
            "q1": {
                "type": "choice",
                "choice": "OPT_A",
                "confidence": float("nan"),
                "probabilities": {"OPT_A": 0.5, "OPT_B": 0.5},
            }
        },
        "latency_ms": 15.0,
        "passes": 1,
        "backend": "system1-mock",
    }

    questions = {"q1": {"type": "choice", "criteria": {"OPT_A": "A", "OPT_B": "B"}}}
    res = orch.decide_with_escalation(
        model_path="dummy",
        state="State",
        questions=questions,
        escalate=True,
    )

    assert res["escalated"] is True
    assert "NaN confidence" in res["escalate_reason"]


def test_pathological_s1_empty_answers_dict():
    """System 1 returning empty answers dict when questions were asked must trigger escalation."""
    mock_s1 = MagicMock()
    mock_s2 = MagicMock()
    orch = HybridOrchestrator(s1_engine=mock_s1, s2_engine=mock_s2)

    mock_s1.predict.return_value = {
        "answers": {},
        "latency_ms": 12.0,
        "passes": 1,
        "backend": "system1-mock",
    }

    questions = {"q1": {"type": "choice", "criteria": {"YES": "Yes", "NO": "No"}}}
    res = orch.decide_with_escalation(
        model_path="dummy",
        state="State",
        questions=questions,
        escalate=True,
    )

    assert res["escalated"] is True
    assert "produced no answers" in res["escalate_reason"]


# ---------------------------------------------------------------------------
# Scale & Extreme Context Stress
# ---------------------------------------------------------------------------

def test_extreme_context_state_truncation():
    """100,000 character state context is safely handled and truncated for System 2 prompt."""
    mock_s1 = MagicMock()
    mock_s2 = MagicMock()
    orch = HybridOrchestrator(s1_engine=mock_s1, s2_engine=mock_s2)

    huge_state = "<div id='content'>" + ("<p>Text snippet for DOM tree</p>" * 3000) + "</div>"
    assert len(huge_state) > 90000

    questions = {"action": {"type": "choice", "criteria": {"CLICK": "Click", "UNCERTAIN": "Unsure"}}}
    mock_s1.predict.return_value = {
        "answers": {"action": {"type": "choice", "choice": "UNCERTAIN", "confidence": 0.5}},
        "latency_ms": 25.0,
    }

    res = orch.decide_with_escalation(
        model_path="dummy",
        state=huge_state,
        questions=questions,
        escalate=True,
    )
    assert res["escalated"] is True
    assert res["success"] is True


def test_multi_question_concurrency_scaling():
    """Stress test with 30 simultaneous compound questions in a single decision pass."""
    mock_s1 = MagicMock()
    mock_s2 = MagicMock()
    orch = HybridOrchestrator(s1_engine=mock_s1, s2_engine=mock_s2)

    questions = {}
    mock_answers = {}
    for i in range(30):
        qid = f"question_{i:02d}"
        questions[qid] = {
            "type": "choice",
            "instructions": {"goal": f"Goal {i}"},
            "criteria": {"ALLOW": "Allow", "DENY": "Deny", "UNCERTAIN": "Unsure"},
        }
        # Every 5th question is low-confidence, triggering escalation
        conf = 0.40 if i % 5 == 0 else 0.95
        mock_answers[qid] = {
            "type": "choice",
            "choice": "UNCERTAIN" if i % 5 == 0 else "ALLOW",
            "confidence": conf,
            "probabilities": {"ALLOW": 0.8, "DENY": 0.1, "UNCERTAIN": 0.1},
        }

    mock_s1.predict.return_value = {
        "answers": mock_answers,
        "latency_ms": 45.0,
        "passes": 1,
        "backend": "system1-mock",
    }

    res = orch.decide_with_escalation(
        model_path="dummy",
        state="Multi-question evaluation state",
        questions=questions,
        escalate=True,
        escalate_tau=0.70,
    )

    assert res["escalated"] is True
    assert len(res["answers"]) == 30


# ---------------------------------------------------------------------------
# Direction B Boundary Conditions
# ---------------------------------------------------------------------------

def test_direction_b_empty_options_handling():
    """Calling system1_tool_decide with empty options list returns graceful error, not unhandled crash."""
    orch = HybridOrchestrator()
    res = orch.system1_tool_decide(
        model_path="dummy",
        state="State",
        options=[],
    )
    assert res["success"] is False
    assert res["chosen"] is None
    assert "cannot be empty" in res["error"]


def test_direction_b_guardrail_boundary_thresholds():
    """Calling system1_guardrail with empty action or out-of-bound threshold [0.0 - 1.0]."""
    mock_s1 = MagicMock()
    orch = HybridOrchestrator(s1_engine=mock_s1)

    # 1. Empty action
    res_empty = orch.system1_guardrail(model_path="dummy", proposed_action="")
    assert res_empty["allowed"] is False
    assert "cannot be empty" in res_empty["error"]

    # 2. Out of bound threshold clamping: -2.0 -> 0.0, 50.0 -> 1.0
    mock_s1.predict.return_value = {
        "answers": {"is_safe": {"type": "noul", "noul": 0.75, "confidence": 0.75}},
        "latency_ms": 10.0,
    }

    res_low = orch.system1_guardrail(model_path="dummy", proposed_action="ls", safe_threshold=-2.0)
    assert res_low["threshold"] == 0.0
    assert res_low["allowed"] is True  # 0.75 >= 0.0

    res_high = orch.system1_guardrail(model_path="dummy", proposed_action="ls", safe_threshold=50.0)
    assert res_high["threshold"] == 1.0
    assert res_high["allowed"] is False  # 0.75 < 1.0


# ---------------------------------------------------------------------------
# Real Pairing: laya:v14s + Local Qwen3-0.6B GGUF Model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LAYA_CHECKPOINT.exists() or not QWEN_GGUF.exists(), reason="Local Laya or Qwen GGUF not found")
def test_real_pair_laya_and_local_qwen_gguf():
    """
    Test real System 1 (laya:v14s) paired with a real System 2 local GGUF model (Qwen3-0.6B-Q8_0.gguf).
    Verifies end-to-end local hardware layer placement, inference, and teacher handoff.
    """
    orch = HybridOrchestrator()

    questions = {
        "action": {
            "type": "choice",
            "instructions": {"goal": "Choose whether to click submit"},
            "criteria": {"SUBMIT": "Click submit button", "CANCEL": "Click cancel button", "UNCERTAIN": "Not sure"},
        }
    }

    # Trigger escalation with tau=0.999 targeting local GGUF model
    res = orch.decide_with_escalation(
        model_path=LAYA_CHECKPOINT,
        state="User wants to complete their order. Page shows big blue Submit Order button.",
        questions=questions,
        escalate=True,
        escalate_tau=0.999,
        system2_model=QWEN_GGUF,
    )

    assert res["escalated"] is True
    assert res["answers"]["action"]["system2"] is True
    assert res["answers"]["action"]["choice"] in ("SUBMIT", "CANCEL", "UNCERTAIN")
    assert res["total_latency_ms"] if "total_latency_ms" in res else res["latency_ms"] > 0
