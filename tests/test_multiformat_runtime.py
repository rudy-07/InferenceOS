"""
test_multiformat_runtime.py
---------------------------
Unit tests for inference_runtime/multiformat_engine.py.
"""
from pathlib import Path
import pytest

from inference_runtime.multiformat_engine import MultiFormatRuntimeEngine
from inference_runtime.inference_session import InferenceResult

FIXTURES_DIR = Path(__file__).parent.parent / "models" / "test_formats"


@pytest.fixture
def engine():
    return MultiFormatRuntimeEngine()


def test_execute_onnx(engine):
    onnx_file = FIXTURES_DIR / "tiny_model.onnx"
    tokens = []
    res = engine.execute(onnx_file, prompt="test prompt", on_token=tokens.append)
    assert isinstance(res, InferenceResult)
    assert res.success is True
    assert "onnxruntime" in res.backend.lower()
    assert len(tokens) > 0
    assert res.stats.eval_tps > 0


def test_execute_obx(engine):
    obx_file = FIXTURES_DIR / "sample.obx"
    tokens = []
    res = engine.execute(obx_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "onnxruntime" in res.backend.lower()
    assert len(tokens) > 0


def test_execute_safetensors(engine):
    st_file = FIXTURES_DIR / "tiny_model.safetensors"
    tokens = []
    res = engine.execute(st_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "torch" in res.backend.lower()
    assert len(tokens) > 0


def test_execute_pytorch(engine):
    pt_file = FIXTURES_DIR / "tiny_model.pt"
    tokens = []
    res = engine.execute(pt_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "torch" in res.backend.lower()
    assert len(tokens) > 0


def test_execute_pickle(engine):
    pkl_file = FIXTURES_DIR / "tiny_model.pkl"
    tokens = []
    res = engine.execute(pkl_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "pickle" in res.backend.lower()
    assert len(tokens) > 0


def test_execute_pth_slm(engine):
    pth_file = FIXTURES_DIR / "tinystories_8m.pth"
    if pth_file.exists():
        tokens = []
        res = engine.execute(pth_file, prompt="Once upon a time", max_tokens=10, on_token=tokens.append)
        assert res.success is True
        assert len(tokens) > 0


def test_execute_pkl_slm(engine):
    pkl_file = FIXTURES_DIR / "tinystories_8m.pkl"
    if pkl_file.exists():
        tokens = []
        res = engine.execute(pkl_file, prompt="Once upon a time", max_tokens=10, on_token=tokens.append)
        assert res.success is True
        assert len(tokens) > 0


def test_execute_tflite(engine):
    tflite_file = FIXTURES_DIR / "text_classification.tflite"
    if tflite_file.exists():
        res = engine.execute(tflite_file, prompt="This is great")
        assert res.success is True
        assert "litert" in res.backend.lower()


def test_execute_openvino(engine):
    xml_file = FIXTURES_DIR / "tinystories_8m.xml"
    if xml_file.exists():
        tokens = []
        res = engine.execute(xml_file, prompt="Once upon a time", max_tokens=10, on_token=tokens.append)
        assert res.success is True
        assert "openvino" in res.backend.lower()
        assert len(tokens) > 0


def test_execute_keras(engine):
    keras_file = FIXTURES_DIR / "text_model.keras"
    if keras_file.exists():
        res = engine.execute(keras_file, prompt="InferenceOS is fast")
        assert res.success is True
        assert "keras" in res.backend.lower()


def test_execute_keras_h5(engine):
    h5_file = FIXTURES_DIR / "text_model.h5"
    if h5_file.exists():
        res = engine.execute(h5_file, prompt="Legacy format test")
        assert res.success is True
        assert "keras" in res.backend.lower()


def test_execute_int8_quantized(engine):
    int8_file = FIXTURES_DIR / "tinystories_8m_int8.pt"
    if int8_file.exists():
        res = engine.execute(int8_file, prompt="Once upon a time", max_tokens=10)
        assert res.success is True
        assert "quantized" in res.backend.lower()


