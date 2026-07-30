"""
test_openai_server.py
---------------------
Tests for InferenceOS OpenAI-compatible HTTP API server endpoints.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from cli.server.openai_server import OpenAIServerApp


@pytest.fixture
def mock_server_app(tmp_path):
    dummy_model = tmp_path / "dummy.gguf"
    dummy_model.write_bytes(b"GGUF_HEADER")

    with patch("cli.server.openai_server.read_gguf_metadata") as mock_gguf, \
         patch("cli.server.openai_server.InferenceSession") as mock_session_cls, \
         patch("cli.server.openai_server.profiler.get_system_resources") as mock_sys_res:

        mock_gguf.return_value = {"arch": "llama", "num_layers": 16, "max_context_length": 2048}
        mock_sys_res.return_value = MagicMock(to_dict=lambda: {"cpu": {"logical_cores": 4}, "gpus": []})
        
        session_inst = MagicMock()
        session_inst.backend_info.name = "vulkan"
        mock_session_cls.return_value = session_inst

        app_inst = OpenAIServerApp(model_path=dummy_model)
        app_inst.initialize_session()
        return app_inst


def test_health_endpoint(mock_server_app):
    client = TestClient(mock_server_app.app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["backend"] == "vulkan"


def test_list_models_endpoint(mock_server_app):
    client = TestClient(mock_server_app.app)
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_chat_completions_json(mock_server_app):
    mock_run_res = MagicMock()
    mock_run_res.generated_text = " Hello! How can I assist you?"
    mock_run_res.stats.tokens_generated = 7
    mock_server_app.session.run.return_value = mock_run_res

    client = TestClient(mock_server_app.app)
    payload = {
        "model": "dummy",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello! How can I assist you?"


def test_completions_json(mock_server_app):
    mock_run_res = MagicMock()
    mock_run_res.generated_text = " Once upon a time"
    mock_run_res.stats.tokens_generated = 5
    mock_server_app.session.run.return_value = mock_run_res

    client = TestClient(mock_server_app.app)
    payload = {
        "model": "dummy",
        "prompt": "Once upon",
        "stream": False,
    }
    res = client.post("/v1/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == " Once upon a time"
