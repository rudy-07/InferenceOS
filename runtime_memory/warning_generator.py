"""
warning_generator.py
--------------------
Structured warning message generation for Phase 6 Runtime Memory Optimization.

Inspects a MemoryBudget and hardware profile to produce human-readable
warning strings. Warnings are attached to the MemoryBudget and also
printed in the report() output.

Warning conditions
------------------
The following conditions trigger warnings (each fires at most once):

  VRAM_HIGH         peak_vram > 90% of total VRAM
  KV_RATIO_HIGH     kv_fill_ratio > 0.80 (approaching HIGH threshold)
  RAM_HIGH          peak_ram > 85% of total RAM
  AUTO_RESIZED      context was automatically reduced to avoid OOM
  KV_GROWTH_FAST    kv_growth_rate > 1.0 GB per 1000 tokens
  NO_GPU_LAYERS     all layers placed on CPU (GPU unused)
  CONTEXT_NEAR_SAFE effective_context > safe_context * 0.95
  LOW_RAM_HEADROOM  RAM free after inference < 2 GB

Design: stateless
-----------------
WarningGenerator has no state. generate() is a pure function over
(budget, hw_profile). This makes it trivially testable and composable.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .memory_budget import MemoryBudget


class WarningGenerator:
    """
    Generates structured warning messages from a completed MemoryBudget.

    All methods are stateless; the class acts as a namespace only.
    """

    # Thresholds for warning conditions
    _VRAM_UTIL_WARN       = 0.90    # peak VRAM > 90% → warn
    _RAM_UTIL_WARN        = 0.85    # peak RAM > 85% → warn
    _KV_RATIO_WARN        = 0.80    # kv_fill_ratio > 0.80 → warn
    _KV_GROWTH_WARN_GB    = 1.0     # > 1 GB per 1K tokens → warn
    _CTX_NEAR_SAFE        = 0.95    # effective > 95% of safe limit → warn
    _RAM_HEADROOM_MIN_GB  = 2.0     # < 2 GB free RAM after inference → warn

    def generate(
        self,
        budget: MemoryBudget,
        hw_profile: Dict[str, Any],
    ) -> List[str]:
        """
        Generate warning strings for a completed MemoryBudget.

        Parameters
        ----------
        budget : MemoryBudget
            Completed pre-flight analysis result.
        hw_profile : dict
            Hardware profile from Phase 1 profiler. Used to read total
            VRAM and RAM capacity for utilisation calculations.

        Returns
        -------
        List[str]
            Zero or more warning message strings (no duplicates).
        """
        warnings: List[str] = []

        vram_total = self._get_vram_total(hw_profile)
        ram_total  = self._get_ram_total(hw_profile)

        # 1. High VRAM utilisation
        if vram_total > 0:
            vram_util = budget.peak_vram_bytes / vram_total
            if vram_util > self._VRAM_UTIL_WARN:
                warnings.append(
                    f"VRAM utilisation is very high ({vram_util * 100:.1f}% of "
                    f"{vram_total / (1024**3):.1f} GB). "
                    "Consider reducing context length or using a more quantized model."
                )

        # 2. High KV fill ratio (approaching HIGH threshold)
        if budget.kv_fill_ratio > self._KV_RATIO_WARN:
            warnings.append(
                f"KV cache fill ratio is high ({budget.kv_fill_ratio:.3f}). "
                f"At full context ({budget.effective_context:,} tokens), "
                f"VRAM headroom will be very tight."
            )

        # 3. High RAM utilisation
        if ram_total > 0 and budget.peak_ram_bytes > 0:
            ram_util = budget.peak_ram_bytes / ram_total
            if ram_util > self._RAM_UTIL_WARN:
                warnings.append(
                    f"System RAM utilisation is high ({ram_util * 100:.1f}%). "
                    "The system may swap under load, severely degrading performance."
                )

        # 4. Auto-resized context
        if budget.auto_resized:
            warnings.append(
                f"Context window automatically reduced from {budget.context_length:,} "
                f"→ {budget.effective_context:,} tokens to prevent OOM. "
                f"Maximum safe context: {budget.safe_context_length:,} tokens."
            )

        # 5. High KV growth rate
        if budget.kv_growth_rate_gb_per_1k > self._KV_GROWTH_WARN_GB:
            warnings.append(
                f"KV cache grows rapidly ({budget.kv_growth_rate_gb_per_1k:.2f} GB per 1K tokens). "
                "Long prompts or extended generation will consume significant VRAM quickly."
            )

        # 6. No GPU layers (CPU-only inference)
        gpu_weight_bytes = budget.weights_vram_bytes
        if gpu_weight_bytes == 0 and budget.kv_bytes_per_token == 0:
            warnings.append(
                "No model layers are placed on GPU. Inference will run entirely on CPU "
                "and will be significantly slower. Consider enabling GPU offload."
            )

        # 7. Effective context close to safe limit
        if budget.safe_context_length > 0:
            ctx_ratio = budget.effective_context / budget.safe_context_length
            if ctx_ratio > self._CTX_NEAR_SAFE and not budget.auto_resized:
                warnings.append(
                    f"Requested context ({budget.effective_context:,} tokens) is "
                    f"{ctx_ratio * 100:.0f}% of the safe limit "
                    f"({budget.safe_context_length:,} tokens). "
                    "Memory pressure may peak near end of generation."
                )

        # 8. Low RAM headroom
        if ram_total > 0:
            ram_free_after = ram_total - budget.peak_ram_bytes
            if 0 < ram_free_after < self._RAM_HEADROOM_MIN_GB * (1024 ** 3):
                warnings.append(
                    f"Less than {self._RAM_HEADROOM_MIN_GB:.0f} GB RAM will remain free "
                    f"during inference ({ram_free_after / (1024**3):.2f} GB). "
                    "Other applications may be affected."
                )

        return warnings

    # ---------------------------------------------------------------------------
    # Hardware profile helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _get_vram_total(hw_profile: Dict[str, Any]) -> int:
        """Extract total VRAM bytes from hw_profile."""
        gpus = hw_profile.get("gpus", [])
        if gpus:
            total_mb = sum(float(g.get("vram_total_mb", g.get("vram_mb", 0))) for g in gpus)
            if total_mb > 0:
                return int(total_mb * 1024 * 1024)
        return 0

    @staticmethod
    def _get_ram_total(hw_profile: Dict[str, Any]) -> int:
        """Extract total RAM bytes from hw_profile."""
        ram = hw_profile.get("ram", hw_profile.get("memory", {}))
        total = int(ram.get("total_bytes", 0))
        if total == 0:
            # Fallback: available_gb field (Phase 1 format)
            avail_gb = float(ram.get("available_gb", 0))
            total = int(avail_gb * 1024 ** 3)
        return total
