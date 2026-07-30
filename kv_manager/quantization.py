"""
quantization.py
---------------
KV cache quantization manager for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class QuantizationFormatInfo:
    format_name: str
    bits_per_element: int
    memory_ratio: float  # vs FP16 baseline (1.0 = FP16)
    quality_retention: float  # 0.0 to 1.0


class KVQuantizer:
    """
    Manages KV cache quantization formats and theoretical memory reduction factors.
    """

    FORMATS: Dict[str, QuantizationFormatInfo] = {
        "FP16": QuantizationFormatInfo("FP16", 16, 1.0, 1.0),
        "BF16": QuantizationFormatInfo("BF16", 16, 1.0, 0.999),
        "INT8": QuantizationFormatInfo("INT8", 8, 0.50, 0.992),
        "INT4": QuantizationFormatInfo("INT4", 4, 0.25, 0.970),
        "MIXED": QuantizationFormatInfo("MIXED", 6, 0.38, 0.985),
    }

    @classmethod
    def calculate_memory_reduction(
        cls,
        current_kv_mb: float,
        target_format: str = "INT8",
    ) -> Tuple[float, float]:
        """
        Calculate (new_kv_mb, memory_saved_mb) when converting to target quantization format.
        """
        fmt = cls.FORMATS.get(target_format.upper(), cls.FORMATS["FP16"])
        new_kv_mb = current_kv_mb * fmt.memory_ratio
        saved_mb = max(0.0, current_kv_mb - new_kv_mb)
        return round(new_kv_mb, 2), round(saved_mb, 2)
