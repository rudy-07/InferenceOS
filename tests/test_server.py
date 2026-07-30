from __future__ import annotations

import sys
import os
from pathlib import Path

# Add global user site-packages for fastapi/uvicorn/httpx if needed
user_site = r"C:\Users\gamin\AppData\Roaming\Python\Python310\site-packages"
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from server.config import ServerConfig
from server.api.app import create_app
from server.sessions.session_manager import get_session_manager


@pytest.fixture
def client():
    config = ServerConfig(auth_api_keys=[])
    app = create_app(config)
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "loaded_models" in data


def test_version_endpoint(client):
    res = client.get("/version")
    assert res.status_code == 200
    data = res.json()
    assert data["server"] == "InferenceOS Server"
    assert "version" in data


def test_list_models_endpoint(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_metrics_endpoints(client):
    # Test JSON format
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "uptime_seconds" in data
    assert "total_requests" in data

    # Test Prometheus format
    res_prom = client.get("/metrics?format=prometheus")
    assert res_prom.status_code == 200
    assert "inferenceos_uptime_seconds" in res_prom.text


def test_session_lifecycle_api(client):
    # 1. Create Session
    res = client.post(
        "/v1/sessions",
        json={
            "model": "test-model.gguf",
            "system_prompt": "You are a helpful assistant.",
        },
    )
    assert res.status_code == 200
    session_data = res.json()
    sid = session_data["session_id"]
    assert sid.startswith("sess_")
    assert session_data["model"] == "test-model.gguf"

    # 2. List Sessions
    res_list = client.get("/v1/sessions")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] >= 1
    assert any(s["session_id"] == sid for s in list_data["data"])

    # 3. Get Session Details
    res_detail = client.get(f"/v1/sessions/{sid}")
    assert res_detail.status_code == 200
    detail_data = res_detail.json()
    assert detail_data["session_id"] == sid
    assert len(detail_data["messages"]) == 1
    assert detail_data["messages"][0]["role"] == "system"

    # 4. Reset Session
    res_reset = client.post(f"/v1/sessions/{sid}/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "reset"

    # 5. Save Session
    res_save = client.post(f"/v1/sessions/{sid}/save")
    assert res_save.status_code == 200
    save_data = res_save.json()
    assert save_data["status"] == "saved"
    assert "saved_path" in save_data

    # 6. Load Session
    res_load = client.post(
        "/v1/sessions/load",
        json={"session_id": sid},
    )
    assert res_load.status_code == 200
    assert res_load.json()["session_id"] == sid

    # 7. Delete Session
    res_del = client.delete(f"/v1/sessions/{sid}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"

    # Verify deleted
    res_verify = client.get(f"/v1/sessions/{sid}")
    assert res_verify.status_code == 404


def test_embeddings_placeholder(client):
    res = client.post(
        "/v1/embeddings",
        json={"model": "test-model", "input": "hello world"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 128


def test_responses_placeholder(client):
    res = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hello"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "completed"


@patch("server.api.routes.chat.get_runtime_adapter")
def test_chat_completions_non_stream(mock_get_adapter, client):
    mock_adapter = MagicMock()
    mock_result = MagicMock()
    mock_result.generated_text = "Hello! I am InferenceOS."
    mock_result.stats.prompt_tokens = 10
    mock_result.stats.tokens_generated = 8
    mock_result.stats.prompt_eval_ms = 15.0
    mock_result.stats.prompt_eval_tps = 100.0
    mock_result.stats.eval_tps = 45.0
    mock_adapter.execute_chat.return_value = mock_result
    mock_get_adapter.return_value = mock_adapter

    res = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello! I am InferenceOS."
    assert data["usage"]["completion_tokens"] == 8
