"""
model_parser.py
---------------
Universal Model Parser & Format Detection Engine for InferenceOS.

Auto-detects and extracts metadata from diverse deep learning and LLM formats:
  - GGUF (.gguf)
  - SafeTensors (.safetensors)
  - ONNX (.onnx, .ort)
  - PyTorch Checkpoints (.pt, .pth, .bin)
  - Python / JAX Pickle (.pkl, .pickle, .joblib)
  - OpenBinary / Orbax (.obx)
  - TensorFlow Lite (.tflite)
"""
from __future__ import annotations

import enum
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Lazy/conditional imports
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

try:
    import onnx
    _HAS_ONNX = True
except ImportError:
    onnx = None
    _HAS_ONNX = False

try:
    import safetensors
    _HAS_SAFETENSORS = True
except ImportError:
    safetensors = None
    _HAS_SAFETENSORS = False

try:
    import pickle
    _HAS_PICKLE = True
except ImportError:
    pickle = None
    _HAS_PICKLE = False


class ModelFormat(str, enum.Enum):
    GGUF = "gguf"
    ONNX = "onnx"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    PICKLE = "pickle"
    OBX = "obx"
    TFLITE = "tflite"
    KERAS = "keras"
    OPENVINO = "openvino"
    COREML = "coreml"
    TENSORRT = "tensorrt"
    UNKNOWN = "unknown"


def detect_model_format(path: Union[str, Path]) -> ModelFormat:
    """
    Detect the model format using file extension and binary magic numbers.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ModelFormat.UNKNOWN

    ext = p.suffix.lower()

    # Read binary header (up to 64 bytes)
    header = b""
    try:
        with open(p, "rb") as f:
            header = f.read(64)
    except Exception:
        pass

    # 1. GGUF magic check: "GGUF"
    if header.startswith(b"GGUF"):
        return ModelFormat.GGUF

    # 2. TFLite magic check: "TFL3" at byte offset 4
    if len(header) >= 8 and header[4:8] == b"TFL3":
        return ModelFormat.TFLITE

    # 3. SafeTensors check: 8-byte uint64 followed by JSON header starting with '{'
    if len(header) >= 9:
        try:
            header_len = struct.unpack("<Q", header[:8])[0]
            if 0 < header_len < 100 * 1024 * 1024 and header[8:9] == b"{":
                if ext == ".obx":
                    return ModelFormat.OBX
                return ModelFormat.SAFETENSORS
        except Exception:
            pass

    # 4. ZIP Archive magic (PK\x03\x04): PyTorch v1.6+ or TorchScript
    if header.startswith(b"PK\x03\x04"):
        if ext in (".pt", ".pth", ".bin"):
            # Check if it's TorchScript or state_dict
            try:
                import zipfile
                with zipfile.ZipFile(p, "r") as zf:
                    namelist = zf.namelist()
                    if any("model.json" in n or "code/" in n for n in namelist):
                        return ModelFormat.TORCHSCRIPT
            except Exception:
                pass
            return ModelFormat.PYTORCH
        if ext == ".obx":
            return ModelFormat.OBX

    # 5. Pickle protocol check
    if len(header) >= 2 and header[0] == 0x80 and header[1] in (2, 3, 4, 5):
        if ext in (".pt", ".pth", ".bin"):
            return ModelFormat.PYTORCH
        if ext in (".pkl", ".pickle", ".joblib"):
            return ModelFormat.PICKLE
        if ext == ".obx":
            return ModelFormat.OBX

    # 6. ONNX Protobuf check: ONNX files typically start with protobuf field 1 (ir_version) varint
    # (0x08 followed by version, e.g. 0x08 0x08, 0x08 0x07, etc.)
    if len(header) >= 2 and header[0] == 0x08 and header[1] in range(1, 20):
        if ext == ".obx":
            return ModelFormat.OBX
        return ModelFormat.ONNX

    # 7. Extension-based fallback
    if ext == ".gguf":
        return ModelFormat.GGUF
    elif ext == ".safetensors":
        return ModelFormat.SAFETENSORS
    elif ext in (".onnx", ".ort"):
        return ModelFormat.ONNX
    elif ext in (".pt", ".pth"):
        return ModelFormat.PYTORCH
    elif ext in (".pkl", ".pickle", ".joblib"):
        return ModelFormat.PICKLE
    elif ext == ".obx":
        return ModelFormat.OBX
    elif ext == ".tflite":
        return ModelFormat.TFLITE
    elif ext in (".h5", ".keras"):
        return ModelFormat.KERAS
    elif ext == ".xml":
        if p.with_suffix(".bin").exists():
            return ModelFormat.OPENVINO
    elif ext in (".mlmodel", ".mlpackage"):
        return ModelFormat.COREML
    elif ext in (".engine", ".plan"):
        return ModelFormat.TENSORRT
    elif ext == ".bin":
        # Could be PyTorch, GGUF, or SafeTensors
        return ModelFormat.PYTORCH

    return ModelFormat.UNKNOWN


# ---------------------------------------------------------------------------
# Metadata Extractors
# ---------------------------------------------------------------------------

def _read_adjacent_config(model_path: Path) -> Dict[str, Any]:
    """Look for an adjacent config.json (common in Hugging Face checkpoints)."""
    candidates = [
        model_path.parent / "config.json",
        model_path.parent / "generation_config.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _extract_safetensors_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from SafeTensors file without loading full tensor weights."""
    file_size = path.stat().st_size
    tensors_info = []
    total_params = 0
    embedded_meta = {}

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Invalid SafeTensors file: shorter than 8 bytes.")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len > file_size or header_len > 100 * 1024 * 1024:
            raise ValueError(f"SafeTensors header too large ({header_len} bytes).")
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode("utf-8"))

    if "__metadata__" in header:
        embedded_meta = header.pop("__metadata__")

    for name, info in header.items():
        shape = info.get("shape", [])
        dtype = info.get("dtype", "unknown")
        num_elements = math.prod(shape) if shape else 1
        total_params += num_elements
        tensors_info.append({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "params": num_elements,
        })

    # Read adjacent config.json if available
    cfg = _read_adjacent_config(path)

    arch = (
        embedded_meta.get("arch")
        or cfg.get("model_type")
        or (cfg.get("architectures", [None])[0] if cfg.get("architectures") else None)
        or "transformer"
    )
    num_layers = int(
        embedded_meta.get("num_layers")
        or cfg.get("num_hidden_layers")
        or cfg.get("n_layer")
        or max([int(p.split(".")[2]) for p in header.keys() if "layers." in p and p.split(".")[2].isdigit()] + [-1]) + 1
        or 1
    )
    hidden_size = int(
        embedded_meta.get("hidden_size")
        or cfg.get("hidden_size")
        or cfg.get("n_embd")
        or 4096
    )
    num_heads = int(
        embedded_meta.get("num_heads")
        or cfg.get("num_attention_heads")
        or cfg.get("n_head")
        or 32
    )
    num_kv_heads = int(
        embedded_meta.get("num_kv_heads")
        or cfg.get("num_key_value_heads")
        or num_heads
    )
    max_context_length = int(
        embedded_meta.get("max_context_length")
        or cfg.get("max_position_embeddings")
        or 4096
    )

    return {
        "format": "safetensors",
        "file_name": path.name,
        "file_size_bytes": file_size,
        "arch": str(arch),
        "num_layers": max(1, num_layers),
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "max_context_length": max_context_length,
        "total_params": total_params,
        "tensor_count": len(tensors_info),
        "tensors": tensors_info[:20],
        "quantization": _detect_quantization(cfg, path),
        "format_details": {
            "embedded_metadata": embedded_meta,
            "has_external_config": bool(cfg),
        },
    }


def _extract_onnx_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from ONNX model using onnx module or protobuf inspection."""
    file_size = path.stat().st_size
    inputs_info = []
    outputs_info = []
    total_params = 0
    op_counts = {}
    ir_version = 0
    opset_version = 0
    producer_name = "unknown"

    if _HAS_ONNX:
        model = onnx.load(str(path), load_external_data=False)
        ir_version = model.ir_version
        producer_name = model.producer_name or "unknown"
        if model.opset_import:
            opset_version = model.opset_import[0].version

        graph = model.graph
        for inp in graph.input:
            shape = []
            if inp.type.tensor_type.shape:
                for d in inp.type.tensor_type.shape.dim:
                    shape.append(d.dim_value if d.dim_value else d.dim_param or -1)
            inputs_info.append({"name": inp.name, "shape": shape})

        for out in graph.output:
            shape = []
            if out.type.tensor_type.shape:
                for d in out.type.tensor_type.shape.dim:
                    shape.append(d.dim_value if d.dim_value else d.dim_param or -1)
            outputs_info.append({"name": out.name, "shape": shape})

        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        for init in graph.initializer:
            dims = list(init.dims)
            total_params += math.prod(dims) if dims else 1

        node_count = len(graph.node)
    else:
        node_count = 0

    return {
        "format": "onnx",
        "file_name": path.name,
        "file_size_bytes": file_size,
        "arch": "onnx_graph",
        "num_layers": op_counts.get("MatMul", op_counts.get("Gemm", op_counts.get("Conv", 1))),
        "hidden_size": inputs_info[0]["shape"][-1] if inputs_info and inputs_info[0]["shape"] else 512,
        "num_heads": 8,
        "num_kv_heads": 8,
        "max_context_length": 2048,
        "total_params": total_params,
        "tensor_count": len(inputs_info) + len(outputs_info),
        "tensors": inputs_info + outputs_info,
        "format_details": {
            "ir_version": ir_version,
            "opset_version": opset_version,
            "producer_name": producer_name,
            "node_count": node_count,
            "inputs": inputs_info,
            "outputs": outputs_info,
            "top_operators": sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        },
    }


def _extract_pytorch_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from PyTorch (.pt, .pth, .bin) checkpoint."""
    file_size = path.stat().st_size
    cfg = _read_adjacent_config(path)
    total_params = 0
    tensors_info = []
    arch = cfg.get("model_type", "pytorch_model")
    num_layers = int(cfg.get("num_hidden_layers", cfg.get("n_layer", 1)))
    hidden_size = int(cfg.get("hidden_size", cfg.get("n_embd", 512)))

    if _HAS_TORCH:
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = {}

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            if not isinstance(state_dict, dict):
                state_dict = {}

            # Read user-defined metadata keys if present
            arch = str(checkpoint.get("arch", checkpoint.get("model_type", "pytorch_nn")))
            num_layers = int(checkpoint.get("num_layers", checkpoint.get("n_layers", 1)))
            hidden_size = int(checkpoint.get("hidden_size", checkpoint.get("dim", 512)))

            for k, v in state_dict.items():
                if hasattr(v, "shape"):
                    shape = list(v.shape)
                    num_el = math.prod(shape)
                    total_params += num_el
                    tensors_info.append({
                        "name": str(k),
                        "shape": shape,
                        "dtype": str(v.dtype).replace("torch.", ""),
                        "params": num_el,
                    })

            # Detect layers from keys if not explicit
            layer_indices = set()
            for k in state_dict.keys():
                for part in k.split("."):
                    if part.isdigit():
                        layer_indices.add(int(part))
            if layer_indices and num_layers == 1:
                num_layers = max(layer_indices) + 1

    return {
        "format": "pytorch",
        "file_name": path.name,
        "file_size_bytes": file_size,
        "arch": arch,
        "num_layers": max(1, num_layers),
        "hidden_size": hidden_size,
        "num_heads": 8,
        "num_kv_heads": 8,
        "max_context_length": 2048,
        "total_params": total_params,
        "tensor_count": len(tensors_info),
        "tensors": tensors_info[:20],
        "quantization": _detect_quantization(cfg, path),
        "format_details": {
            "checkpoint_type": "state_dict" if tensors_info else "general_torch_object",
        },
    }


def _extract_pickle_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from Pickle / JAX checkpoint file."""
    file_size = path.stat().st_size
    keys = []
    model_name = "pickle_serialized_model"
    total_params = 0

    if _HAS_PICKLE:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                keys = list(data.keys())
                model_name = data.get("model_name", data.get("name", "pickle_model"))
                # Calculate params if arrays are present
                for v in data.values():
                    if hasattr(v, "shape"):
                        total_params += math.prod(v.shape)
                    elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                        total_params += len(v)
        except Exception as e:
            keys = [f"inspection_error: {e}"]

    return {
        "format": "pickle",
        "file_name": path.name,
        "file_size_bytes": file_size,
        "arch": str(model_name),
        "num_layers": 1,
        "hidden_size": 256,
        "num_heads": 4,
        "num_kv_heads": 4,
        "max_context_length": 1024,
        "total_params": total_params,
        "tensor_count": len(keys),
        "tensors": [{"name": k} for k in keys[:20]],
        "format_details": {
            "keys": keys[:10],
        },
    }


def _extract_obx_metadata(path: Path) -> Dict[str, Any]:
    """
    Extract metadata from .obx (OpenBinary / Orbax) model.
    Sniffs underlying format and wraps with OBX attributes.
    """
    # Try ONNX first
    try:
        meta = _extract_onnx_metadata(path)
        meta["format"] = "obx"
        meta["format_details"]["underlying_type"] = "onnx_protobuf"
        return meta
    except Exception:
        pass

    # Try SafeTensors
    try:
        meta = _extract_safetensors_metadata(path)
        meta["format"] = "obx"
        meta["format_details"]["underlying_type"] = "safetensors_binary"
        return meta
    except Exception:
        pass

    # Try PyTorch
    try:
        meta = _extract_pytorch_metadata(path)
        meta["format"] = "obx"
        meta["format_details"]["underlying_type"] = "pytorch_serialized"
        return meta
    except Exception:
        pass

    return {
        "format": "obx",
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "arch": "open_binary_bundle",
        "num_layers": 1,
        "hidden_size": 512,
        "num_heads": 8,
        "num_kv_heads": 8,
        "max_context_length": 2048,
        "total_params": 0,
        "tensor_count": 0,
        "tensors": [],
        "format_details": {
            "underlying_type": "unstructured_binary",
        },
    }


def _extract_tflite_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from a TensorFlow Lite (.tflite) model."""
    file_size = path.stat().st_size
    inputs_info = []
    outputs_info = []
    total_params = 0
    tensor_count = 0

    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=str(path))
        interp.allocate_tensors()
        for inp in interp.get_input_details():
            inputs_info.append({"name": inp["name"], "shape": list(inp["shape"]), "dtype": str(inp["dtype"])})
        for out in interp.get_output_details():
            outputs_info.append({"name": out["name"], "shape": list(out["shape"]), "dtype": str(out["dtype"])})
        tensor_details = interp.get_tensor_details()
        tensor_count = len(tensor_details)
        for t in tensor_details:
            shape = list(t.get("shape", []))
            if shape:
                total_params += math.prod(shape)
    except Exception:
        pass

    return {
        "format": "tflite",
        "file_name": path.name,
        "file_size_bytes": file_size,
        "arch": "tflite_neural_network",
        "num_layers": max(1, len(outputs_info)),
        "hidden_size": inputs_info[0]["shape"][-1] if inputs_info and inputs_info[0]["shape"] else 256,
        "num_heads": 4,
        "num_kv_heads": 4,
        "max_context_length": 1024,
        "total_params": total_params or (file_size // 4),
        "tensor_count": tensor_count,
        "tensors": inputs_info + outputs_info,
        "format_details": {
            "inputs": inputs_info,
            "outputs": outputs_info,
        },
    }


def _detect_quantization(cfg: Dict[str, Any], path: Path) -> str:
    """Detect quantization method (bitsandbytes, awq, gptq, int8) from config or filename."""
    if "quantization_config" in cfg:
        q_cfg = cfg["quantization_config"]
        method = q_cfg.get("quant_method", "").lower()
        if method:
            return method
        if q_cfg.get("load_in_4bit") or q_cfg.get("load_in_8bit"):
            return "bitsandbytes"
    stem = path.stem.lower()
    if "int8" in stem or "q8" in stem:
        return "int8"
    if "int4" in stem or "q4" in stem:
        return "int4"
    if "awq" in stem:
        return "awq"
    if "gptq" in stem:
        return "gptq"
    return "none"


def _extract_openvino_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from OpenVINO Intermediate Representation (.xml)."""
    file_size = path.stat().st_size
    bin_path = path.with_suffix(".bin")
    total_size = file_size + (bin_path.stat().st_size if bin_path.exists() else 0)

    inputs_info = []
    outputs_info = []
    num_layers = 1
    total_params = 0

    try:
        import openvino as ov
        core = ov.Core()
        model = core.read_model(str(path))
        for inp in model.inputs:
            inputs_info.append({"name": inp.get_any_name(), "shape": list(inp.shape), "dtype": str(inp.element_type)})
        for out in model.outputs:
            outputs_info.append({"name": out.get_any_name(), "shape": list(out.shape), "dtype": str(out.element_type)})
        num_layers = len(model.get_ordered_ops())
        if bin_path.exists():
            total_params = bin_path.stat().st_size // 2
    except Exception:
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(path))
            root = tree.getroot()
            layers = root.findall(".//layer")
            num_layers = max(1, len(layers))
            if bin_path.exists():
                total_params = bin_path.stat().st_size // 2
        except Exception:
            pass

    return {
        "format": "openvino",
        "file_name": path.name,
        "file_size_bytes": total_size,
        "arch": "openvino_ir",
        "num_layers": num_layers,
        "hidden_size": inputs_info[0]["shape"][-1] if inputs_info and inputs_info[0]["shape"] else 768,
        "num_heads": 12,
        "num_kv_heads": 12,
        "max_context_length": 2048,
        "total_params": total_params or (total_size // 2),
        "tensor_count": len(inputs_info) + len(outputs_info),
        "tensors": inputs_info + outputs_info,
        "quantization": _detect_quantization({}, path),
        "format_details": {
            "xml_path": str(path),
            "bin_path": str(bin_path) if bin_path.exists() else None,
            "inputs": inputs_info,
            "outputs": outputs_info,
        },
    }


def _extract_keras_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from Keras 3 (.keras) or legacy Keras (.h5) models."""
    file_size = path.stat().st_size
    inputs_info = []
    outputs_info = []
    arch_name = "keras_model"
    total_params = file_size // 4
    num_layers = 1

    try:
        import os
        os.environ["KERAS_BACKEND"] = "torch"
        import keras
        model = keras.saving.load_model(str(path))
        arch_name = getattr(model, "name", "keras_model")
        num_layers = len(model.layers) if hasattr(model, "layers") else 1
        total_params = model.count_params() if hasattr(model, "count_params") else file_size // 4
        if hasattr(model, "input_shape"):
            inputs_info.append({"name": "input", "shape": list(model.input_shape or [])})
        if hasattr(model, "output_shape"):
            outputs_info.append({"name": "output", "shape": list(model.output_shape or [])})
    except Exception:
        pass

    return {
        "format": "keras",
        "file_name": path.name,
        "file_size_bytes": file_size,
        "arch": arch_name,
        "num_layers": num_layers,
        "hidden_size": inputs_info[0]["shape"][-1] if inputs_info and inputs_info[0]["shape"] else 256,
        "num_heads": 4,
        "num_kv_heads": 4,
        "max_context_length": 1024,
        "total_params": total_params,
        "tensor_count": num_layers,
        "tensors": inputs_info + outputs_info,
        "quantization": _detect_quantization({}, path),
        "format_details": {
            "model_type": path.suffix.lstrip("."),
            "inputs": inputs_info,
            "outputs": outputs_info,
        },
    }


def read_model_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Universal model metadata extractor. Auto-detects the format and returns
    a standardized metadata dictionary compatible with ModelDescriptor.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    fmt = detect_model_format(p)

    if fmt == ModelFormat.GGUF:
        from orchestrator.gguf_parser import read_gguf_metadata
        try:
            gguf_meta = read_gguf_metadata(p)
        except Exception:
            gguf_meta = {}
        file_size = p.stat().st_size
        return {
            "format": "gguf",
            "file_name": p.name,
            "file_size_bytes": file_size,
            "arch": str(gguf_meta.get("arch", "llama")),
            "num_layers": int(gguf_meta.get("num_layers", 32)),
            "hidden_size": int(gguf_meta.get("hidden_size", 4096)),
            "num_heads": int(gguf_meta.get("num_heads", 32)),
            "num_kv_heads": int(gguf_meta.get("num_kv_heads", 8)),
            "max_context_length": int(gguf_meta.get("max_context_length", 4096)),
            "total_params": int(gguf_meta.get("total_params", file_size // 2)),
            "tensor_count": int(gguf_meta.get("tensor_count", 0)),
            "tensors": [],
            "quantization": _detect_quantization(gguf_meta, p),
            "format_details": gguf_meta,
        }

    elif fmt == ModelFormat.SAFETENSORS:
        return _extract_safetensors_metadata(p)

    elif fmt == ModelFormat.ONNX:
        return _extract_onnx_metadata(p)

    elif fmt in (ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT):
        return _extract_pytorch_metadata(p)

    elif fmt == ModelFormat.PICKLE:
        return _extract_pickle_metadata(p)

    elif fmt == ModelFormat.OBX:
        return _extract_obx_metadata(p)

    elif fmt == ModelFormat.TFLITE:
        return _extract_tflite_metadata(p)

    elif fmt == ModelFormat.OPENVINO:
        return _extract_openvino_metadata(p)

    elif fmt == ModelFormat.KERAS:
        return _extract_keras_metadata(p)

    else:
        # Fallback generic metadata
        file_size = p.stat().st_size
        return {
            "format": str(fmt.value),
            "file_name": p.name,
            "file_size_bytes": file_size,
            "arch": "generic_model",
            "num_layers": 1,
            "hidden_size": 512,
            "num_heads": 8,
            "num_kv_heads": 8,
            "max_context_length": 2048,
            "total_params": file_size // 4,
            "tensor_count": 0,
            "tensors": [],
            "format_details": {"extension": p.suffix},
        }
