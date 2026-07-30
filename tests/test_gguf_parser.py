"""
test_gguf_parser.py
-------------------
Unit tests for orchestrator/gguf_parser.py
"""
import struct
from pathlib import Path
import pytest
from orchestrator.gguf_parser import GGUFMetadataReader, read_gguf_metadata


def create_mock_gguf_file(path: Path, arch="qwen2", layers=28, hidden=3584, heads=28, kv_heads=4, ctx=32768):
    """Utility to build a minimal valid binary GGUF v3 file."""
    def encode_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def encode_kv(key: str, val_type: int, val_bytes: bytes) -> bytes:
        return encode_str(key) + struct.pack("<I", val_type) + val_bytes

    kv_data = bytearray()

    # 1. general.architecture = arch (STRING = 8)
    kv_data.extend(encode_kv("general.architecture", 8, encode_str(arch)))

    # 2. {arch}.block_count = layers (UINT32 = 4)
    kv_data.extend(encode_kv(f"{arch}.block_count", 4, struct.pack("<I", layers)))

    # 3. {arch}.embedding_length = hidden (UINT32 = 4)
    kv_data.extend(encode_kv(f"{arch}.embedding_length", 4, struct.pack("<I", hidden)))

    # 4. {arch}.attention.head_count = heads (UINT32 = 4)
    kv_data.extend(encode_kv(f"{arch}.attention.head_count", 4, struct.pack("<I", heads)))

    # 5. {arch}.attention.head_count_kv = kv_heads (UINT32 = 4)
    kv_data.extend(encode_kv(f"{arch}.attention.head_count_kv", 4, struct.pack("<I", kv_heads)))

    # 6. {arch}.context_length = ctx (UINT32 = 4)
    kv_data.extend(encode_kv(f"{arch}.context_length", 4, struct.pack("<I", ctx)))

    metadata_count = 6
    tensor_count = 0

    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count) + struct.pack("<Q", metadata_count)

    with open(path, "wb") as f:
        f.write(header)
        f.write(kv_data)


def test_gguf_parser_extracts_metadata(tmp_path):
    gguf_path = tmp_path / "test_model.gguf"
    create_mock_gguf_file(gguf_path, arch="qwen2", layers=28, hidden=3584, heads=28, kv_heads=4, ctx=32768)

    reader = GGUFMetadataReader(gguf_path)
    params = reader.get_model_params()

    assert params["arch"] == "qwen2"
    assert params["num_layers"] == 28
    assert params["hidden_size"] == 3584
    assert params["num_heads"] == 28
    assert params["num_kv_heads"] == 4
    assert params["max_context_length"] == 32768


def test_gguf_parser_helper_func(tmp_path):
    gguf_path = tmp_path / "llama_model.gguf"
    create_mock_gguf_file(gguf_path, arch="llama", layers=32, hidden=4096, heads=32, kv_heads=8, ctx=8192)

    params = read_gguf_metadata(gguf_path)

    assert params["arch"] == "llama"
    assert params["num_layers"] == 32
    assert params["hidden_size"] == 4096
    assert params["num_heads"] == 32
    assert params["num_kv_heads"] == 8
    assert params["max_context_length"] == 8192


def test_gguf_parser_reads_real_file_if_present():
    real_gguf = Path(__file__).parent.parent / "models" / "Qwen3-4B-Thinking-2507.Q5_K_M.gguf"
    if real_gguf.exists():
        params = read_gguf_metadata(real_gguf)
        assert params["num_layers"] > 0
        assert params["hidden_size"] > 0
