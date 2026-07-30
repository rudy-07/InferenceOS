"""
compression.py
--------------
Adaptive KV cache compression engine for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .quantization import KVQuantizer


@dataclass
class CompressionResult:
    mode_selected: str
    target_format: str
    original_kv_mb: float
    compressed_kv_mb: float
    memory_saved_mb: float
    compression_ratio: float
    reasoning: str


class AdaptiveKVCompressor:
    """
    Adaptive KV compression engine deciding when and how much to compress KV cache.
    """

    def evaluate_compression(
        self,
        current_kv_mb: float,
        pressure_level: str = "Low",
        requested_mode: str = "adaptive",
        vram_headroom_mb: float = 4000.0,
    ) -> CompressionResult:
        """
        Determine optimal compression mode, target quantization format, and memory reduction.
        """
        if current_kv_mb <= 0.0 or requested_mode.lower() == "disabled":
            return CompressionResult(
                mode_selected="Disabled",
                target_format="FP16",
                original_kv_mb=current_kv_mb,
                compressed_kv_mb=current_kv_mb,
                memory_saved_mb=0.0,
                compression_ratio=1.0,
                reasoning="KV compression is disabled.",
            )

        mode = requested_mode.lower()

        if mode == "adaptive":
            if pressure_level in ("Critical", "High"):
                effective_mode = "aggressive"
            elif pressure_level == "Medium":
                effective_mode = "balanced"
            elif vram_headroom_mb < 2000.0:
                effective_mode = "balanced"
            else:
                effective_mode = "lossless"
        else:
            effective_mode = mode

        if effective_mode == "aggressive":
            target_fmt = "INT4"
            desc = "Aggressive compression (INT4) active under high memory pressure."
        elif effective_mode == "balanced":
            target_fmt = "INT8"
            desc = "Balanced compression (INT8) active to maintain safe memory headroom."
        elif effective_mode == "lossless":
            target_fmt = "BF16"
            desc = "Lossless compression (BF16) active for maximum quality."
        else:
            target_fmt = "FP16"
            desc = "Standard uncompressed FP16 baseline."

        comp_kv_mb, saved_mb = KVQuantizer.calculate_memory_reduction(current_kv_mb, target_format=target_fmt)
        ratio = current_kv_mb / max(0.1, comp_kv_mb)

        return CompressionResult(
            mode_selected=effective_mode.capitalize(),
            target_format=target_fmt,
            original_kv_mb=current_kv_mb,
            compressed_kv_mb=comp_kv_mb,
            memory_saved_mb=saved_mb,
            compression_ratio=round(ratio, 2),
            reasoning=desc,
        )
