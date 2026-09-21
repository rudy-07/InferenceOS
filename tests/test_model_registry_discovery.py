"""
test_model_registry_discovery.py
--------------------------------
Unit tests for cli/core/model_registry.py auto-discovery across multiple formats and folders.
"""
from pathlib import Path
import pytest

from cli.core.model_registry import ModelRegistry, SUPPORTED_EXTENSIONS

FIXTURES_DIR = Path(__file__).parent.parent / "models" / "test_formats"


def test_supported_extensions():
    assert ".gguf" in SUPPORTED_EXTENSIONS
    assert ".onnx" in SUPPORTED_EXTENSIONS
    assert ".safetensors" in SUPPORTED_EXTENSIONS
    assert ".pt" in SUPPORTED_EXTENSIONS
    assert ".obx" in SUPPORTED_EXTENSIONS
    assert ".pkl" in SUPPORTED_EXTENSIONS
    assert ".h5" in SUPPORTED_EXTENSIONS
    assert ".keras" in SUPPORTED_EXTENSIONS


def test_registry_discovery_test_formats():
    reg = ModelRegistry()
    discovered = reg.auto_discover(root_dir=FIXTURES_DIR)
    assert len(discovered) >= 4

    models = reg.list_models()
    formats_found = {m.get("format") for m in models}
    assert "onnx" in formats_found
    assert "safetensors" in formats_found
    assert "pytorch" in formats_found
    assert "obx" in formats_found


def test_resolve_model_path_formats():
    reg = ModelRegistry()
    onnx_p = reg.resolve_model_path(str(FIXTURES_DIR / "tiny_model.onnx"))
    assert onnx_p is not None
    assert onnx_p.exists()

    st_p = reg.resolve_model_path(str(FIXTURES_DIR / "tiny_model.safetensors"))
    assert st_p is not None
    assert st_p.exists()
