"""
validation.py
-------------
Optimization candidate validator for InferenceOS.
"""
from __future__ import annotations

from typing import Tuple

from .interfaces import CandidateConfig, HardwareFingerprintData, ModelFingerprintData


class OptimizationValidator:
    """
    Validates candidate runtime configurations.
    """

    def validate_candidate(
        self,
        candidate: CandidateConfig,
        model_fp: ModelFingerprintData,
        hw_fp: HardwareFingerprintData,
    ) -> Tuple[bool, str]:
        """
        Run multi-pass validation checks on a candidate configuration.
        """
        # 1. Memory Headroom Check
        if candidate.microbatch_size > 1024 and hw_fp.vram_gb < 6.0:
            return False, "Microbatch size exceeds VRAM headroom."

        if candidate.gpu_layers > model_fp.n_layers:
            return False, "GPU layer offload exceeds total model layers."

        return True, "Passed all validation checks."
