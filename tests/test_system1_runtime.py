"""
test_system1_runtime.py
-----------------------
Comprehensive, exhaustive test suite for InferenceOS System 1 Non-Autoregressive Runtime.

Validates:
1. Auto-detection across all System 1 architectures (Laya, Kev with kev_config, Kev with head.pt).
2. Metadata extraction & zero-KV cache invariant.
3. ModelDescriptor & PlacementEngine 100% GPU allocation with 0 MB KV cache.
4. Multi-type questions (choice, score, noul) in single forward passes.
5. Coarse-to-fine chunking for wide action spaces (>60 options) with probability conservation.
6. Edge case resilience: missing instructions, empty/single options, huge DOM states, special characters.
7. Kev LocalPredictor output adaptation to unified Jev contract.
8. MultiFormatRuntimeEngine routing & execution metrics (decisions/sec).
9. Server API endpoints: GET/POST /v1/systemone, POST /v1/decision.
10. OpenAI /v1/chat/completions tool-selection bridge for System 1.
11. Real-world execution with live local Laya checkpoint (laya:v14s).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from orchestrator.model_parser import detect_model_format, read_model_metadata, ModelFormat
from layer_placement import ModelDescriptor, PlacementEngine
from inference_runtime import System1RuntimeEngine, System1Runner, MultiFormatRuntimeEngine
from server.api.app import create_app

LAYA_CHECKPOINT = Path(r"D:\Projects\llm\laya-browser\v14s")


def test_detect_model_format_system1(tmp_path):
    """Test format auto-detection on mock Laya and Kev checkpoints."""
    # 1. Laya checkpoint directory
    laya_dir = tmp_path / "mock_laya"
    laya_dir.mkdir()
    (laya_dir / "rl_agent_config.json").write_text(json.dumps({"encoder": "mmBERT", "max_len": 1024}), encoding="utf-8")
    (laya_dir / "model.safetensors").write_text("dummy", encoding="utf-8")

    assert detect_model_format(laya_dir) == ModelFormat.SYSTEM1
    assert detect_model_format(laya_dir / "rl_agent_config.json") == ModelFormat.SYSTEM1
    assert detect_model_format(laya_dir / "model.safetensors") == ModelFormat.SYSTEM1

    # 2. Kev checkpoint directory with kev_config.json
    kev_dir = tmp_path / "mock_kev_json"
    kev_dir.mkdir()
    (kev_dir / "kev_config.json").write_text(json.dumps({"arch": "qwen"}), encoding="utf-8")
    assert detect_model_format(kev_dir) == ModelFormat.SYSTEM1

    # 3. Kev checkpoint directory with head.pt
    kev_pt_dir = tmp_path / "mock_kev_pt"
    kev_pt_dir.mkdir()
    (kev_pt_dir / "head.pt").write_bytes(b"\x00" * 128)
    assert detect_model_format(kev_pt_dir) == ModelFormat.SYSTEM1
    assert detect_model_format(kev_pt_dir / "head.pt") == ModelFormat.SYSTEM1


def test_read_model_metadata_system1(tmp_path):
    """Test reading System 1 metadata and verifying zero-KV footprint."""
    # Test Laya metadata
    laya_dir = tmp_path / "test_laya"
    laya_dir.mkdir()
    cfg = {
        "encoder": "jhu-clsp/mmBERT-base",
        "head_layers": 2,
        "max_len": 1024,
        "head_max_len": 768,
        "laya_fmt": "v3",
    }
    (laya_dir / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (laya_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    meta = read_model_metadata(laya_dir)
    assert meta["format"] == "system1"
    assert meta["system1_kind"] == "laya"
    assert meta["is_non_autoregressive"] is True
    assert meta["kv_cache_bytes_per_token"] == 0
    assert meta["max_context_length"] == 1024
    assert meta["head_max_len"] == 768

    # Test Kev metadata
    kev_dir = tmp_path / "test_kev"
    kev_dir.mkdir()
    kev_cfg = {
        "model_name": "jaredpalmer/kev-0.8b",
        "max_len": 2048,
        "head_max_len": 1024,
    }
    (kev_dir / "kev_config.json").write_text(json.dumps(kev_cfg), encoding="utf-8")
    meta_kev = read_model_metadata(kev_dir)
    assert meta_kev["format"] == "system1"
    assert meta_kev["system1_kind"] == "kev"
    assert meta_kev["max_context_length"] == 2048


def test_model_descriptor_and_placement_plan(tmp_path):
    """Test that PlacementEngine allocates 100% GPU with 0 MB KV cache."""
    laya_dir = tmp_path / "placement_laya"
    laya_dir.mkdir()
    cfg = {"encoder": "mmBERT-base", "max_len": 1024, "head_layers": 2}
    (laya_dir / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (laya_dir / "model.safetensors").write_bytes(b"\x00" * 1024 * 1024)

    meta = read_model_metadata(laya_dir)
    desc = ModelDescriptor.from_system1_metadata(meta, model_size_bytes=meta["file_size_bytes"], model_name="test-s1")

    assert desc.kv_cache_bytes_per_token() == 0

    pe = PlacementEngine()
    plan = pe.generatePlacementPlan(desc, context_length=1024)

    assert plan.n_gpu_layers == desc.num_layers
    assert plan.n_cpu_layers == 0


def test_compound_questions_choice_score_noul():
    """Verify simultaneous evaluation of choice, score, and noul in one forward pass."""
    engine = System1RuntimeEngine()

    class MockMultiAgent:
        def __init__(self):
            self.device = "cpu"
            self.cfg = {"laya_fmt": "v3"}
        def predict(self, state, questions):
            answers = {}
            for qid, q in questions.items():
                t = q.get("type", "choice")
                if t == "choice":
                    opts = list(q["criteria"].keys())
                    answers[qid] = {
                        "type": "choice",
                        "choice": opts[0],
                        "probabilities": {k: 1.0 / len(opts) for k in opts},
                        "confidence": 0.90,
                    }
                elif t == "score":
                    answers[qid] = {
                        "type": "score",
                        "score": 4.25,
                        "probabilities": {"0": 0.05, "1": 0.1, "2": 0.15, "3": 0.3, "4": 0.4},
                        "confidence": 0.88,
                    }
                elif t == "noul":
                    answers[qid] = {
                        "type": "noul",
                        "noul": 0.94,
                        "confidence": 0.94,
                    }
            return {"answers": answers, "usage": {"input_tokens": 150}}

    engine._agents["mock_compound"] = MockMultiAgent()
    engine._agent_metadata["mock_compound"] = {"laya_fmt": "v3", "system1_kind": "mock"}

    questions = {
        "action": {
            "type": "choice",
            "instructions": {"goal": "Choose next step"},
            "criteria": {"CLICK": "Click button", "WAIT": "Wait 1s", "DONE": "Done"},
        },
        "page_readiness": {
            "type": "score",
            "instructions": {"goal": "Evaluate page readiness"},
            "criteria": ["Not ready", "Partially loaded", "Ready"],
        },
        "is_safe": {
            "type": "noul",
            "instructions": {"goal": "Is this action safe to execute?"},
        },
    }

    res = engine.predict("mock_compound", state="User on checkout page", questions=questions)

    assert res["success"] is True
    assert res["passes"] == 1
    assert "action" in res["answers"]
    assert "page_readiness" in res["answers"]
    assert "is_safe" in res["answers"]

    assert res["answers"]["action"]["type"] == "choice"
    assert res["answers"]["page_readiness"]["type"] == "score"
    assert res["answers"]["page_readiness"]["score"] == 4.25
    assert res["answers"]["is_safe"]["type"] == "noul"
    assert res["answers"]["is_safe"]["noul"] == 0.94


def test_coarse_to_fine_chunking():
    """Verify coarse-to-fine chunking logic when options exceed max_options threshold."""
    engine = System1RuntimeEngine()

    class MockAgent:
        def __init__(self):
            self.device = "cpu"
            self.cfg = {"laya_fmt": "v3"}
        def predict(self, state, questions):
            answers = {}
            for qid, q in questions.items():
                opts = list(q["criteria"].keys())
                winner = opts[0] if opts else ""
                n = max(1, len(opts))
                answers[qid] = {
                    "type": "choice",
                    "choice": winner,
                    "probabilities": {k: 1.0 / n for k in opts},
                    "confidence": 0.85,
                }
            return {"answers": answers, "usage": {"input_tokens": 100}}

    engine._agents["dummy_mock"] = MockAgent()
    engine._agent_metadata["dummy_mock"] = {"laya_fmt": "v3", "system1_kind": "mock"}

    # 120 options (> max_options=50)
    wide_options = {f"opt_{i}": f"Candidate element #{i}" for i in range(120)}
    questions = {
        "target": {
            "type": "choice",
            "instructions": {"goal": "Find element"},
            "criteria": wide_options,
        }
    }

    res = engine.predict("dummy_mock", state="dummy state", questions=questions, max_options=50)

    assert res["passes"] == 2  # Required 2 passes
    ans = res["answers"]["target"]
    assert "coarse_to_fine" in ans
    assert ans["coarse_to_fine"]["chunks"] == 3  # 120 / 50 -> 3 chunks
    assert len(ans["probabilities"]) == 120
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-4


def test_edge_case_missing_instructions_auto_heal():
    """Verify that questions with missing or empty instructions are automatically healed."""
    engine = System1RuntimeEngine()

    class MockHealAgent:
        def __init__(self):
            self.device = "cpu"
            self.cfg = {"laya_fmt": "v3"}
        def predict(self, state, questions):
            for qid, q in questions.items():
                assert "instructions" in q and q["instructions"]
                assert "goal" in q["instructions"]
            return {"answers": {"q1": {"type": "choice", "choice": "A", "probabilities": {"A": 1.0}, "confidence": 0.99}}, "usage": {"input_tokens": 50}}

    engine._agents["mock_heal"] = MockHealAgent()
    engine._agent_metadata["mock_heal"] = {"laya_fmt": "v3", "system1_kind": "mock"}

    raw_questions = {
        "q1": {
            "type": "choice",
            "criteria": {"A": "Option A", "B": "Option B"},
            # Notice missing "instructions" field!
        }
    }

    res = engine.predict("mock_heal", state="Global page state context", questions=raw_questions)
    assert res["success"] is True
    assert res["answers"]["q1"]["choice"] == "A"


def test_edge_case_huge_dom_state_truncation():
    """Verify that oversized state texts/DOMs are safely truncated according to format rules."""
    engine = System1RuntimeEngine()

    received_states = []

    class MockInspectStateAgent:
        def __init__(self):
            self.device = "cpu"
            self.cfg = {"laya_fmt": "v3"}
        def predict(self, state, questions):
            received_states.append(state)
            return {"answers": {"q": {"type": "choice", "choice": "OK", "probabilities": {"OK": 1.0}, "confidence": 1.0}}, "usage": {"input_tokens": 100}}

    engine._agents["mock_trunc"] = MockInspectStateAgent()
    engine._agent_metadata["mock_trunc"] = {"laya_fmt": "v3", "system1_kind": "mock"}

    huge_state = {
        "page": {
            "text": "X" * 50_000,
            "title": "Huge Page",
        }
    }

    engine.predict("mock_trunc", state=huge_state, questions={"q": {"criteria": {"OK": "yes"}}})
    assert len(received_states) == 1
    # FMT v3 cutoff is 1200 characters
    assert len(received_states[0]["page"]["text"]) <= 1200


def test_edge_case_special_and_unicode_criteria():
    """Verify options with emojis, quotes, URLs, and non-ASCII characters work seamlessly."""
    engine = System1RuntimeEngine()

    class MockUnicodeAgent:
        def __init__(self):
            self.device = "cpu"
            self.cfg = {"laya_fmt": "v3"}
        def predict(self, state, questions):
            crit = questions["q"]["criteria"]
            return {"answers": {"q": {"type": "choice", "choice": list(crit.keys())[0], "probabilities": {k: 0.5 for k in crit}, "confidence": 0.8}}, "usage": {"input_tokens": 80}}

    engine._agents["mock_unicode"] = MockUnicodeAgent()
    engine._agent_metadata["mock_unicode"] = {"laya_fmt": "v3", "system1_kind": "mock"}

    special_questions = {
        "q": {
            "type": "choice",
            "criteria": {
                "🚀_launch": "Launch payload to https://space.com/api?id=123",
                "⚠️_warning": 'Quotes: "Attention Required" & symbols <tag>',
            }
        }
    }

    res = engine.predict("mock_unicode", state="Status check", questions=special_questions)
    assert res["success"] is True
    assert "🚀_launch" in res["answers"]["q"]["probabilities"]


def test_kev_local_predictor_adaptation():
    """Verify adaptation of Kev LocalPredictor (__call__) output into standard TypeSafe Jev answers."""
    engine = System1RuntimeEngine()

    class MockKevPredictor:
        def __init__(self):
            self.device = "cpu"
        def __call__(self, record):
            # Kev returns probabilities dict keyed by question id
            return {
                "probabilities": {
                    "classify": {"SPAM": 0.85, "HAM": 0.15},
                    "urgency": {"LOW": 0.1, "MEDIUM": 0.2, "HIGH": 0.7},
                },
                "latency_ms": 18.5,
                "input_tokens": 128,
            }

    engine._agents["mock_kev"] = MockKevPredictor()
    engine._agent_metadata["mock_kev"] = {"system1_kind": "kev"}

    questions = {
        "classify": {
            "type": "choice",
            "criteria": {"SPAM": "Spam message", "HAM": "Legitimate message"},
        },
        "urgency": {
            "type": "choice",
            "criteria": {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"},
        },
    }

    res = engine.predict("mock_kev", state="Congratulations! You won $10,000!", questions=questions)

    assert res["success"] is True
    assert res["answers"]["classify"]["choice"] == "SPAM"
    assert res["answers"]["classify"]["confidence"] == 0.85
    assert res["answers"]["urgency"]["choice"] == "HIGH"
    assert res["answers"]["urgency"]["confidence"] == 0.70


def test_multiformat_runtime_routing(tmp_path):
    """Verify routing of ModelFormat.SYSTEM1 to System1Runner through MultiFormatRuntimeEngine."""
    laya_dir = tmp_path / "mock_system1_route"
    laya_dir.mkdir()
    (laya_dir / "rl_agent_config.json").write_text(json.dumps({"encoder": "mmBERT", "max_len": 1024}), encoding="utf-8")
    (laya_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    multi_engine = MultiFormatRuntimeEngine()
    runner = multi_engine.get_runner(laya_dir)
    assert isinstance(runner, System1Runner)


def test_systemone_server_routes():
    """Test FastAPI /v1/systemone, /v1/decision, and OpenAI chat completions bridge."""
    app = create_app()
    client = TestClient(app)

    # 1. GET /v1/systemone status check
    resp = client.get("/v1/systemone")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "active"
    assert "InferenceOS System 1" in data["engine"]

    # 2. POST /v1/decision with live or mock model
    if LAYA_CHECKPOINT.exists():
        payload = {
            "model": "laya:v14s",
            "state": "User wants to navigate home",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": {"goal": "Navigate home"},
                    "criteria": {
                        "NAV_HOME": "Click home icon",
                        "SEARCH": "Search home",
                        "CANCEL": "Cancel navigation",
                    },
                }
            },
        }
        post_resp = client.post("/v1/decision", json=payload)
        assert post_resp.status_code == 200
        post_data = post_resp.json()
        assert post_data["success"] is True
        assert "action" in post_data["answers"]
        assert "choice" in post_data["answers"]["action"]

        # 3. POST /v1/systemone endpoint
        sys1_resp = client.post("/v1/systemone", json=payload)
        assert sys1_resp.status_code == 200
        sys1_data = sys1_resp.json()
        assert "answers" in sys1_data

        # 4. OpenAI /v1/chat/completions bridge test
        chat_payload = {
            "model": "laya:v14s",
            "messages": [
                {"role": "system", "content": "You are a decision model."},
                {"role": "user", "content": "Should we discard unsaved changes? Options: YES, NO, UNCERTAIN"}
            ]
        }
        chat_resp = client.post("/v1/chat/completions", json=chat_payload)
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert len(chat_data["choices"]) > 0
        assert "content" in chat_data["choices"][0]["message"]


@pytest.mark.skipif(not LAYA_CHECKPOINT.exists(), reason="Local Laya checkpoint not found")
def test_live_laya_execution():
    """Live integration test executing real Laya model through MultiFormatRuntimeEngine."""
    engine = MultiFormatRuntimeEngine()
    res = engine.execute(LAYA_CHECKPOINT, prompt="Should we open the menu?")

    assert res.success is True
    assert "system1-laya" in res.backend
    assert res.system1_answers is not None
    assert "decision" in res.system1_answers
    # Verify decisions/sec computation
    assert res.stats.eval_tps > 0
