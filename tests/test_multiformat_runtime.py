"""
test_multiformat_runtime.py
---------------------------
Unit tests for inference_runtime/multiformat_engine.py.
"""
from pathlib import Path
import pytest

from inference_runtime.multiformat_engine import (
    MultiFormatRuntimeEngine,
    _HAS_ORT,
    _HAS_TORCH,
    _HAS_SAFETENSORS,
    _HAS_PICKLE,
    _HAS_OPENVINO,
    _HAS_KERAS,
)
from inference_runtime.inference_session import InferenceResult
from inference_runtime.stats_collector import RuntimeStats

FIXTURES_DIR = Path(__file__).parent.parent / "models" / "test_formats"


@pytest.fixture
def engine():
    return MultiFormatRuntimeEngine()


def test_execute_nonexistent_model_raises(engine):
    with pytest.raises(FileNotFoundError):
        engine.execute("nonexistent_file_98765.onnx", prompt="test")


@pytest.mark.skipif(not _HAS_ORT or not (FIXTURES_DIR / "tiny_model.onnx").exists(), reason="onnxruntime or tiny_model.onnx not available")
def test_execute_onnx(engine):
    onnx_file = FIXTURES_DIR / "tiny_model.onnx"
    tokens = []
    res = engine.execute(onnx_file, prompt="test prompt", on_token=tokens.append)
    assert isinstance(res, InferenceResult)
    assert res.success is True
    assert "onnxruntime" in res.backend.lower()
    assert len(tokens) > 0
    assert res.stats.eval_tps > 0


@pytest.mark.skipif(not _HAS_ORT or not (FIXTURES_DIR / "sample.obx").exists(), reason="onnxruntime or sample.obx not available")
def test_execute_obx(engine):
    obx_file = FIXTURES_DIR / "sample.obx"
    tokens = []
    res = engine.execute(obx_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "onnxruntime" in res.backend.lower()
    assert len(tokens) > 0


@pytest.mark.skipif(not (_HAS_TORCH and _HAS_SAFETENSORS) or not (FIXTURES_DIR / "tiny_model.safetensors").exists(), reason="PyTorch/SafeTensors or tiny_model.safetensors not available")
def test_execute_safetensors(engine):
    st_file = FIXTURES_DIR / "tiny_model.safetensors"
    tokens = []
    res = engine.execute(st_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "torch" in res.backend.lower()
    assert len(tokens) > 0


@pytest.mark.skipif(not _HAS_TORCH or not (FIXTURES_DIR / "tiny_model.pt").exists(), reason="PyTorch or tiny_model.pt not available")
def test_execute_pytorch(engine):
    pt_file = FIXTURES_DIR / "tiny_model.pt"
    tokens = []
    res = engine.execute(pt_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "torch" in res.backend.lower()
    assert len(tokens) > 0


@pytest.mark.skipif(not _HAS_PICKLE or not (FIXTURES_DIR / "tiny_model.pkl").exists(), reason="pickle or tiny_model.pkl not available")
def test_execute_pickle(engine):
    pkl_file = FIXTURES_DIR / "tiny_model.pkl"
    tokens = []
    res = engine.execute(pkl_file, prompt="test prompt", on_token=tokens.append)
    assert res.success is True
    assert "pickle" in res.backend.lower()
    assert len(tokens) > 0


@pytest.mark.skipif(not _HAS_TORCH or not (FIXTURES_DIR / "tinystories_8m.pth").exists(), reason="tinystories_8m.pth or PyTorch not available")
def test_execute_pth_slm(engine):
    pth_file = FIXTURES_DIR / "tinystories_8m.pth"
    tokens = []
    res = engine.execute(pth_file, prompt="Once upon a time", max_tokens=10, on_token=tokens.append)
    assert res.success is True
    assert len(tokens) > 0


@pytest.mark.skipif(not _HAS_PICKLE or not (FIXTURES_DIR / "tinystories_8m.pkl").exists(), reason="tinystories_8m.pkl or pickle not available")
def test_execute_pkl_slm(engine):
    pkl_file = FIXTURES_DIR / "tinystories_8m.pkl"
    tokens = []
    res = engine.execute(pkl_file, prompt="Once upon a time", max_tokens=10, on_token=tokens.append)
    assert res.success is True
    assert len(tokens) > 0


@pytest.mark.skipif(not (FIXTURES_DIR / "text_classification.tflite").exists(), reason="text_classification.tflite not available")
def test_execute_tflite(engine):
    tflite_file = FIXTURES_DIR / "text_classification.tflite"
    res = engine.execute(tflite_file, prompt="This is great")
    assert res.success is True
    assert "litert" in res.backend.lower()


@pytest.mark.skipif(not _HAS_OPENVINO or not (FIXTURES_DIR / "tinystories_8m.xml").exists(), reason="OpenVINO or tinystories_8m.xml not available")
def test_execute_openvino(engine):
    xml_file = FIXTURES_DIR / "tinystories_8m.xml"
    tokens = []
    res = engine.execute(xml_file, prompt="Once upon a time", max_tokens=10, on_token=tokens.append)
    assert res.success is True
    assert "openvino" in res.backend.lower()
    assert len(tokens) > 0


@pytest.mark.skipif(not _HAS_KERAS or not (FIXTURES_DIR / "text_model.keras").exists(), reason="Keras or text_model.keras not available")
def test_execute_keras(engine):
    keras_file = FIXTURES_DIR / "text_model.keras"
    res = engine.execute(keras_file, prompt="InferenceOS is fast")
    assert res.success is True
    assert "keras" in res.backend.lower()


@pytest.mark.skipif(not _HAS_KERAS or not (FIXTURES_DIR / "text_model.h5").exists(), reason="Keras or text_model.h5 not available")
def test_execute_keras_h5(engine):
    h5_file = FIXTURES_DIR / "text_model.h5"
    res = engine.execute(h5_file, prompt="Legacy format test")
    assert res.success is True
    assert "keras" in res.backend.lower()


@pytest.mark.skipif(not _HAS_TORCH or not (FIXTURES_DIR / "tinystories_8m_int8.pt").exists(), reason="tinystories_8m_int8.pt or PyTorch not available")
def test_execute_int8_quantized(engine):
    int8_file = FIXTURES_DIR / "tinystories_8m_int8.pt"
    res = engine.execute(int8_file, prompt="Once upon a time", max_tokens=10)
    assert res.success is True
    assert "quantized" in res.backend.lower()



