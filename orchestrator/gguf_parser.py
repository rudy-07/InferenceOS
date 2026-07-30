"""
gguf_parser.py
--------------
Pure-Python GGUF binary header parser for extracting architecture metadata
such as layer count, hidden size, attention heads, and context window length.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


class GGUFMetadataReader:
    """Reads metadata key-value pairs from a GGUF binary header."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.metadata: dict[str, Any] = {}

    def _read_string(self, f: BinaryIO) -> str:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            raise EOFError("Unexpected end of file while reading string length")
        length = struct.unpack("<Q", length_bytes)[0]
        str_bytes = f.read(length)
        if len(str_bytes) < length:
            raise EOFError("Unexpected end of file while reading string content")
        return str_bytes.decode("utf-8", errors="replace")

    def _read_value(self, f: BinaryIO, val_type: int) -> Any:
        if val_type == GGUF_TYPE_UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif val_type == GGUF_TYPE_INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif val_type == GGUF_TYPE_UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif val_type == GGUF_TYPE_INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif val_type == GGUF_TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif val_type == GGUF_TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif val_type == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif val_type == GGUF_TYPE_BOOL:
            return bool(struct.unpack("<B", f.read(1))[0])
        elif val_type == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif val_type == GGUF_TYPE_ARRAY:
            elem_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            return [self._read_value(f, elem_type) for _ in range(count)]
        elif val_type == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif val_type == GGUF_TYPE_INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif val_type == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        else:
            raise ValueError(f"Unknown GGUF value type: {val_type}")

    def parse(self, max_kv_pairs: int = 256) -> dict[str, Any]:
        """Parses up to max_kv_pairs metadata elements from the GGUF header."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")

        with open(self.file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Invalid GGUF magic bytes: {magic!r}")

            version_bytes = f.read(4)
            version = struct.unpack("<I", version_bytes)[0]
            if version not in (2, 3):
                # We log warning or attempt best-effort for version >= 2
                pass

            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            kv_count_to_read = min(metadata_kv_count, max_kv_pairs)

            for _ in range(kv_count_to_read):
                try:
                    key = self._read_string(f)
                    val_type = struct.unpack("<I", f.read(4))[0]
                    val = self._read_value(f, val_type)
                    self.metadata[key] = val
                except (EOFError, ValueError, struct.error):
                    break

        return self.metadata

    def get_architecture_summary(self) -> dict[str, Any]:
        return self.get_model_params()

    def get_model_params(self) -> dict[str, Any]:
        """
        Extracts key model parameters: num_layers, hidden_size, num_heads, num_kv_heads, max_context_length.
        Applies architectural defaults if individual keys are missing.
        """
        if not self.metadata:
            try:
                self.parse()
            except Exception:
                pass

        arch = str(self.metadata.get("general.architecture", "llama"))

        # Look up keys with prefix or fallback
        num_layers = int(
            self.metadata.get(f"{arch}.block_count")
            or self.metadata.get("llama.block_count")
            or 32
        )
        hidden_size = int(
            self.metadata.get(f"{arch}.embedding_length")
            or self.metadata.get("llama.embedding_length")
            or 4096
        )
        num_heads = int(
            self.metadata.get(f"{arch}.attention.head_count")
            or self.metadata.get("llama.attention.head_count")
            or 32
        )
        num_kv_heads = int(
            self.metadata.get(f"{arch}.attention.head_count_kv")
            or self.metadata.get("llama.attention.head_count_kv")
            or num_heads
        )
        max_context = int(
            self.metadata.get(f"{arch}.context_length")
            or self.metadata.get("llama.context_length")
            or 4096
        )

        return {
            "arch": arch,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "max_context_length": max_context,
        }


def read_gguf_metadata(file_path: Path) -> dict[str, Any]:
    """Helper utility to parse a GGUF file and return extracted architecture metadata."""
    reader = GGUFMetadataReader(file_path)
    return reader.get_model_params()
