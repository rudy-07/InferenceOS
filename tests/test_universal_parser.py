"""
test_universal_parser.py
------------------------
Unit tests for orchestrator/model_parser.py: format detection and metadata extraction.
"""
from pathlib import Path
import pytest

from orchestrator.model_parser import (
    ModelFormat,
    detect_model_format,
    read_model_metadata,
)

FIXTURES_DIR = Path(__file__).parent.parent / "models" / "test_formats"


def test_detect_model_format_onnx():
    onnx_file = FIXTURES_DIR / "tiny_model.onnx"
    assert onnx_file.exists()
    assert detect_model_format(onnx_file) == ModelFormat.ONNX


def test_detect_model_format_safetensors():
    st_file = FIXTURES_DIR / "tiny_model.safetensors"
    assert st_file.exists()
    assert detect_model_format(st_file) == ModelFormat.SAFETENSORS


def test_detect_model_format_pytorch():
    pt_file = FIXTURES_DIR / "tiny_model.pt"
    assert pt_file.exists()
    assert detect_model_format(pt_file) == ModelFormat.PYTORCH


def test_detect_model_format_pickle():
    pkl_file = FIXTURES_DIR / "tiny_model.pkl"
    assert pkl_file.exists()
    assert detect_model_format(pkl_file) == ModelFormat.PICKLE


def test_detect_model_format_obx():
    obx_file = FIXTURES_DIR / "sample.obx"
    assert obx_file.exists()
    assert detect_model_format(obx_file) == ModelFormat.OBX


def test_read_onnx_metadata():
    onnx_file = FIXTURES_DIR / "tiny_model.onnx"
    meta = read_model_metadata(onnx_file)
    assert meta["format"] == "onnx"
    assert meta["total_params"] > 0
    assert "format_details" in meta
    assert "opset_version" in meta["format_details"]


def test_read_safetensors_metadata():
    st_file = FIXTURES_DIR / "tiny_model.safetensors"
    meta = read_model_metadata(st_file)
    assert meta["format"] == "safetensors"
    assert meta["arch"] == "transformer"
    assert meta["total_params"] == 20480
    assert meta["tensor_count"] == 6


def test_read_pytorch_metadata():
    pt_file = FIXTURES_DIR / "tiny_model.pt"
    meta = read_model_metadata(pt_file)
    assert meta["format"] == "pytorch"
    assert meta["arch"] == "linear_mlp"
    assert meta["total_params"] == 212


def test_read_obx_metadata():
    obx_file = FIXTURES_DIR / "sample.obx"
    meta = read_model_metadata(obx_file)
    assert meta["format"] == "obx"
    assert "underlying_type" in meta["format_details"]
