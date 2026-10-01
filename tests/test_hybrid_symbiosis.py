"""
test_hybrid_symbiosis.py
-------------------------
Comprehensive test suite for Phase 13: Bi-Directional System 1 / System 2 Symbiosis.

Validates:
1. Fast-path reflex bypass (no escalation when confidence >= tau).
2. Direction A (S1 -> S2): Confidence gating (confidence < tau triggers escalation).
3. Direction A (S1 -> S2): Uncertainty fallback (choice in UNCERTAIN, AMBIGUOUS triggers escalation).
4. Direction A (S1 -> S2): Generative synthesis handoff (needs_generation=True invokes S2 text generator).
5. Strict Toggle-Off: Disabling escalation (escalate=False) preserves pure S1 reflex regardless of confidence.
6. DAgger Distillation Logging: Appends JSONL demonstration dataset on escalation.
7. Direction B (S2 -> S1): System 1 Decision Tool (system1_tool_decide) evaluating option arrays.
8. Direction B (S2 -> S1): System 1 Safety Guardrail (system1_guardrail) checking safe vs destructive actions.
9. Server API Routes: /v1/systemone escalation headers & body toggle, /v1/systemone/tool, /v1/systemone/guardrail.
10. Live Checkpoint Integration: Real-world execution with laya:v14s.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from orchestrator.hybrid_orchestrator import HybridOrchestrator
from server.api.app import create_app

LAYA_CHECKPOINT = Path(r"D:\Projects\llm\laya-browser\v14s")


# ---------------------------------------------------------------------------
# Unit Tests with Mocked S1 Engine (Isolation & Determinism)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_orchestrator():
    mock_s1 = MagicMock()
    mock_s2 = MagicMock()
    orch = HybridOrchestrator(s1_engine=mock_s1, s2_engine=mock_s2)
    return orch, mock_s1, mock_s2


def test_fast_path_reflex_bypass_when_confident(mock_orchestrator):
    """When System 1 confidence is >= tau and no generation needed, return immediately."""
    orch, mock_s1, _ = mock_orchestrator

    mock_s1.predict.return_value = {
        "answers": {
            "action": {
                "type": "choice",
                "choice": "CLICK_SUBMIT",
                "confidence": 0.94,
                "probabilities": {"CLICK_SUBMIT": 0.94, "CLICK_CANCEL": 0.04, "UNCERTAIN": 0.02},
            }
        },
        "latency_ms": 18.5,
        "passes": 1,
        "backend": "system1-mock",
        "model": "mock-s1",
    }

    questions = {
        "action": {
            "type": "choice",
            "instructions": {"goal": "Submit form"},
            "criteria": {"CLICK_SUBMIT": "Submit", "CLICK_CANCEL": "Cancel", "UNCERTAIN": "Unsure"},
        }
    }

    res = orch.decide_with_escalation(
        model_path="dummy/path",
        state="Form filled",
        questions=questions,
        escalate=True,
        escalate_tau=0.70,
    )

    assert res["escalated"] is False
    assert res["escalate_reason"] is None
    assert res["answers"]["action"]["choice"] == "CLICK_SUBMIT"
    assert res["latency_ms"] == 18.5
    assert res["backend"] == "system1-mock"


def test_escalation_triggers_on_low_confidence(mock_orchestrator):
    """When confidence is below tau, System 2 teacher is invoked to resolve the decision."""
    orch, mock_s1, _ = mock_orchestrator

    mock_s1.predict.return_value = {
        "answers": {
            "action": {
                "type": "choice",
                "choice": "CLICK_SUBMIT",
                "confidence": 0.52,  # Below tau=0.70
                "probabilities": {"CLICK_SUBMIT": 0.52, "CLICK_CANCEL": 0.48},
            }
        },
        "latency_ms": 22.0,
        "passes": 1,
        "backend": "system1-mock",
        "model": "mock-s1",
    }

    questions = {
        "action": {
            "type": "choice",
            "instructions": {"goal": "Submit form"},
            "criteria": {"CLICK_SUBMIT": "Submit", "CLICK_CANCEL": "Cancel"},
        }
    }

    res = orch.decide_with_escalation(
        model_path="dummy/path",
        state="Form filled with ambiguous inputs",
        questions=questions,
        escalate=True,
        escalate_tau=0.70,
    )

    assert res["escalated"] is True
    assert "below threshold tau=0.70" in res["escalate_reason"]
    assert res["answers"]["action"]["system2"] is True
    assert res["answers"]["action"]["confidence"] == 0.95
    assert "system2-escalated" in res["backend"]


def test_escalation_triggers_on_uncertain_choice(mock_orchestrator):
    """When S1 returns UNCERTAIN, escalation triggers even if confidence is nominally high."""
    orch, mock_s1, _ = mock_orchestrator

    mock_s1.predict.return_value = {
        "answers": {
            "nav": {
                "type": "choice",
                "choice": "UNCERTAIN",
                "confidence": 0.88,
                "probabilities": {"NAV_HOME": 0.06, "NAV_SETTINGS": 0.06, "UNCERTAIN": 0.88},
            }
        },
        "latency_ms": 21.0,
        "passes": 1,
        "backend": "system1-mock",
        "model": "mock-s1",
    }

    questions = {
        "nav": {
            "type": "choice",
            "instructions": {"goal": "Navigate somewhere"},
            "criteria": {"NAV_HOME": "Home", "NAV_SETTINGS": "Settings", "UNCERTAIN": "Need context"},
        }
    }

    res = orch.decide_with_escalation(
        model_path="dummy/path",
        state="Ambiguous navigation state",
        questions=questions,
        escalate=True,
        escalate_tau=0.70,
    )

    assert res["escalated"] is True
    assert "uncertain choice: 'UNCERTAIN'" in res["escalate_reason"]
    assert res["answers"]["nav"]["system2"] is True
    # Teacher resolves away from UNCERTAIN
    assert res["answers"]["nav"]["choice"] != "UNCERTAIN"


def test_strict_toggle_off_suppresses_escalation(mock_orchestrator):
    """When escalate=False, pure System 1 reflex is preserved unconditionally."""
    orch, mock_s1, _ = mock_orchestrator

    # S1 returns very low confidence and UNCERTAIN
    mock_s1.predict.return_value = {
        "answers": {
            "action": {
                "type": "choice",
                "choice": "UNCERTAIN",
                "confidence": 0.15,
                "probabilities": {"UNCERTAIN": 0.15, "OPTION_A": 0.10},
            }
        },
        "latency_ms": 15.0,
        "passes": 1,
        "backend": "system1-mock",
        "model": "mock-s1",
    }

    questions = {
        "action": {
            "type": "choice",
            "instructions": {"goal": "Do something"},
            "criteria": {"UNCERTAIN": "Unsure", "OPTION_A": "Do A"},
        }
    }

    # Strict toggle: escalate=False
    res = orch.decide_with_escalation(
        model_path="dummy/path",
        state="Whatever",
        questions=questions,
        escalate=False,
        escalate_tau=0.99,
    )

    assert res["escalated"] is False
    assert res["escalate_reason"] is None
    assert res["answers"]["action"]["choice"] == "UNCERTAIN"
    assert res["answers"]["action"]["confidence"] == 0.15


def test_generation_handoff_invokes_system2_synthesis(mock_orchestrator):
    """When a question requests text generation, System 2 generates the natural language text."""
    orch, mock_s1, _ = mock_orchestrator

    mock_s1.predict.return_value = {
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "REPLY_SUPPORT",
                "confidence": 0.96,
                "probabilities": {"REPLY_SUPPORT": 0.96},
            }
        },
        "latency_ms": 19.0,
        "passes": 1,
        "backend": "system1-mock",
        "model": "mock-s1",
    }

    questions = {
        "intent": {
            "type": "choice",
            "instructions": {"goal": "Classify user message"},
            "criteria": {"REPLY_SUPPORT": "Send customer support answer"},
            "needs_generation": True,  # Flag demanding text synthesis
        }
    }

    res = orch.decide_with_escalation(
        model_path="dummy/path",
        state="Customer asked: Where is my shipment?",
        questions=questions,
        escalate=True,
        escalate_tau=0.70,
    )

    assert res["escalated"] is True
    assert "requires generative response synthesis" in res["escalate_reason"]
    assert res["generated_text"] is not None
    assert len(res["generated_text"]) > 0


def test_dagger_distillation_logging(tmp_path, mock_orchestrator):
    """When escalation triggers with a dagger_log_path, a JSONL pair is appended."""
    orch, mock_s1, _ = mock_orchestrator
    log_file = tmp_path / "dagger" / "escalations.jsonl"

    mock_s1.predict.return_value = {
        "answers": {
            "q1": {
                "type": "choice",
                "choice": "OPTION_A",
                "confidence": 0.40,
                "probabilities": {"OPTION_A": 0.40, "OPTION_B": 0.60},
            }
        },
        "latency_ms": 20.0,
        "passes": 1,
        "backend": "system1-mock",
        "model": "mock-s1",
    }

    questions = {
        "q1": {
            "type": "choice",
            "instructions": {"goal": "Choose option"},
            "criteria": {"OPTION_A": "A", "OPTION_B": "B"},
        }
    }

    orch.decide_with_escalation(
        model_path="dummy/path",
        state="Complex test state",
        questions=questions,
        escalate=True,
        escalate_tau=0.75,
        dagger_log_path=log_file,
    )

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert "timestamp" in entry
    assert entry["state"] == "Complex test state"
    assert "q1" in entry["system1_answers"]
    assert "q1" in entry["system2_answers"]
    assert "below threshold tau=0.75" in entry["escalate_reason"]


def test_direction_b_system1_tool_decide(mock_orchestrator):
    """System 2 agent invokes System 1 as a fast tool to evaluate candidate options."""
    orch, mock_s1, _ = mock_orchestrator

    mock_s1.predict.return_value = {
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "opt_2",
                "confidence": 0.89,
                "probabilities": {"opt_0": 0.05, "opt_1": 0.06, "opt_2": 0.89},
            }
        },
        "latency_ms": 16.2,
    }

    tool_result = orch.system1_tool_decide(
        model_path="dummy/path",
        state="Page with multiple buttons",
        options=["Submit", "Reset", "Confirm Payment"],
        goal="Select button to complete checkout",
    )

    assert tool_result["success"] is True
    assert tool_result["chosen"] == "opt_2"
    assert tool_result["confidence"] == 0.89
    assert tool_result["latency_ms"] == 16.2


def test_direction_b_system1_guardrail_gate(mock_orchestrator):
    """System 2 passes proposed action through System 1 noul safety check before running."""
    orch, mock_s1, _ = mock_orchestrator

    # Case 1: Safe action
    mock_s1.predict.return_value = {
        "answers": {
            "is_safe": {
                "type": "noul",
                "noul": 0.98,
                "confidence": 0.98,
            }
        },
        "latency_ms": 14.1,
    }

    safe_check = orch.system1_guardrail(
        model_path="dummy/path",
        proposed_action="grep 'TODO' src/main.py",
        safe_threshold=0.85,
    )
    assert safe_check["allowed"] is True
    assert safe_check["safety_score"] == 0.98

    # Case 2: Dangerous action
    mock_s1.predict.return_value = {
        "answers": {
            "is_safe": {
                "type": "noul",
                "noul": 0.12,
                "confidence": 0.88,
            }
        },
        "latency_ms": 13.9,
    }

    danger_check = orch.system1_guardrail(
        model_path="dummy/path",
        proposed_action="rm -rf /",
        safe_threshold=0.85,
    )
    assert danger_check["allowed"] is False
    assert danger_check["safety_score"] == 0.12


# ---------------------------------------------------------------------------
# Server API Integration Tests (Endpoints & HTTP Headers)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_api_systemone_escalation_toggle_and_headers(client, monkeypatch):
    """Test HTTP API /v1/systemone with body parameters and header overrides."""
    mock_orch = MagicMock()
    mock_orch.decide_with_escalation.return_value = {
        "model": "laya:v14s",
        "answers": {"decision": {"choice": "YES", "confidence": 0.95, "system2": True}},
        "escalated": True,
        "escalate_reason": "Confidence below threshold",
        "latency_ms": 25.0,
        "passes": 1,
        "backend": "system1-laya + system2-escalated",
    }
    monkeypatch.setattr("orchestrator.hybrid_orchestrator.HybridOrchestrator.decide_with_escalation", mock_orch.decide_with_escalation)

    # 1. Request with escalate=True
    resp = client.post(
        "/v1/systemone",
        json={
            "model": "laya:v14s",
            "state": "Evaluate checkout button",
            "questions": {
                "decision": {
                    "type": "choice",
                    "criteria": {"YES": "Yes", "NO": "No"},
                }
            },
            "escalate": True,
            "escalate_tau": 0.90,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["escalated"] is True
    assert "system2" in data["answers"]["decision"]

    # 2. Header override: x-system1-escalate: false forces no escalation
    mock_orch.decide_with_escalation.return_value = {
        "model": "laya:v14s",
        "answers": {"decision": {"choice": "YES", "confidence": 0.60}},
        "escalated": False,
        "escalate_reason": None,
        "latency_ms": 18.0,
        "passes": 1,
        "backend": "system1-laya",
    }

    resp = client.post(
        "/v1/systemone",
        json={
            "model": "laya:v14s",
            "state": "Evaluate checkout button",
            "questions": {"decision": {"type": "choice", "criteria": {"YES": "Yes", "NO": "No"}}},
            "escalate": True,
        },
        headers={"x-system1-escalate": "false"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["escalated"] is False


def test_api_systemone_tool_endpoint(client, monkeypatch):
    """Test /v1/systemone/tool endpoint for System 2 tool calls."""
    mock_orch = MagicMock()
    mock_orch.system1_tool_decide.return_value = {
        "chosen": "1",
        "confidence": 0.91,
        "probabilities": {"0": 0.05, "1": 0.91, "2": 0.04},
        "latency_ms": 17.5,
        "success": True,
    }
    monkeypatch.setattr("orchestrator.hybrid_orchestrator.HybridOrchestrator.system1_tool_decide", mock_orch.system1_tool_decide)

    resp = client.post(
        "/v1/systemone/tool",
        json={
            "model": "laya:v14s",
            "state": "Select from navbar",
            "options": ["Dashboard", "Settings", "Logout"],
            "goal": "Go to settings",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chosen"] == "1"
    assert data["confidence"] == 0.91
    assert data["success"] is True


def test_api_systemone_guardrail_endpoint(client, monkeypatch):
    """Test /v1/systemone/guardrail endpoint for System 2 safety checks."""
    mock_orch = MagicMock()
    mock_orch.system1_guardrail.return_value = {
        "allowed": True,
        "safety_score": 0.96,
        "threshold": 0.85,
        "latency_ms": 15.0,
        "proposed_action": "git status",
    }
    monkeypatch.setattr("orchestrator.hybrid_orchestrator.HybridOrchestrator.system1_guardrail", mock_orch.system1_guardrail)

    resp = client.post(
        "/v1/systemone/guardrail",
        json={
            "model": "laya:v14s",
            "proposed_action": "git status",
            "safe_threshold": 0.85,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["allowed"] is True
    assert data["safety_score"] == 0.96


# ---------------------------------------------------------------------------
# Real Checkpoint Live Symbiosis Test (if laya:v14s present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LAYA_CHECKPOINT.exists(), reason="Local laya:v14s checkpoint not found")
def test_real_laya_live_symbiosis(tmp_path):
    """Test real Laya model executing both fast reflex path and forced teacher escalation."""
    orch = HybridOrchestrator()
    dagger_file = tmp_path / "live_dagger.jsonl"

    questions = {
        "decision": {
            "type": "choice",
            "instructions": {"goal": "Evaluate if the user clicked the buy button"},
            "criteria": {
                "YES": "User clicked buy",
                "NO": "User clicked something else",
                "UNCERTAIN": "Cannot tell from state",
            },
        }
    }

    # Pass 1: escalate=False -> pure fast System 1 reflex
    res_s1 = orch.decide_with_escalation(
        model_path=LAYA_CHECKPOINT,
        state="Button label: Complete Purchase",
        questions=questions,
        escalate=False,
    )
    assert res_s1["escalated"] is False
    assert res_s1["answers"]["decision"]["choice"] in ("YES", "NO", "UNCERTAIN")
    assert res_s1["latency_ms"] < 1000.0  # Real CPU inference < 1 sec

    # Pass 2: Forced escalation with tau=0.999 and DAgger logging
    res_hybrid = orch.decide_with_escalation(
        model_path=LAYA_CHECKPOINT,
        state="Button label: Complete Purchase",
        questions=questions,
        escalate=True,
        escalate_tau=0.999,  # Guarantees escalation
        dagger_log_path=dagger_file,
    )
    assert res_hybrid["escalated"] is True
    assert "below threshold" in res_hybrid["escalate_reason"]
    assert res_hybrid["answers"]["decision"]["system2"] is True
    assert dagger_file.exists()
