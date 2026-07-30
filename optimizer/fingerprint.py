"""
fingerprint.py
--------------
Model and Hardware Fingerprint engine for InferenceOS.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .interfaces import HardwareFingerprintData, ModelFingerprintData


class FingerprintEngine:
    """
    Computes stable model and hardware fingerprints.
    """

    @classmethod
    def compute_model_fingerprint(
        self,
        model_path: Optional[str] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelFingerprintData:
        """
        Compute model fingerprint using SHA256 hash prefix and GGUF metadata.
        """
        meta = model_metadata or {}
        name = str(meta.get("model_name", Path(model_path).stem if model_path else "Qwen3-4B"))

        sha256_hash = ""
        if model_path and os.path.exists(model_path):
            try:
                # Read first 64KB for fast stable SHA256 hashing
                hasher = hashlib.sha256()
                with open(model_path, "rb") as f:
                    chunk = f.read(65536)
                    hasher.update(chunk)
                sha256_hash = hasher.hexdigest()
            except Exception:
                sha256_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()
        else:
            sha256_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()

        return ModelFingerprintData(
            sha256_hash=sha256_hash,
            model_name=name,
            param_count_b=float(meta.get("param_count_b", 4.0)),
            quantization=str(meta.get("quantization", "Q4_K_M")),
            n_layers=int(meta.get("num_layers", 32)),
            hidden_dim=int(meta.get("hidden_size", 4096)),
            vocab_size=int(meta.get("vocab_size", 152064)),
            n_heads=int(meta.get("num_heads", 32)),
            context_capability=int(meta.get("context_length", 32768)),
            gguf_version=int(meta.get("gguf_version", 3)),
        )

    @classmethod
    def compute_hardware_fingerprint(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        backend: str = "vulkan",
    ) -> HardwareFingerprintData:
        """
        Compute hardware fingerprint.
        """
        hw = hw_profile or {}
        gpus = hw.get("gpus", [])
        gpu_name = gpus[0].get("name", "GPU") if gpus else "CPU"
        vram_gb = (float(gpus[0].get("vram_total_mb", 8192.0)) / 1024.0) if gpus else 0.0

        ram_info = hw.get("ram", {})
        ram_gb = float(ram_info.get("total_gb", 16.0))

        return HardwareFingerprintData(
            gpu_name=gpu_name,
            vram_gb=round(vram_gb, 1),
            cpu_name="CPU",
            ram_gb=round(ram_gb, 1),
            backend=backend,
            driver_version="latest",
            os_name="Windows",
            inferenceos_version="1.0.0",
        )
