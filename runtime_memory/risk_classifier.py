"""
risk_classifier.py
------------------
OOM risk classification for Phase 6 Runtime Memory Optimization.

Classifies the probability of an Out-of-Memory error during inference
based on the ratio of projected KV cache (at full context fill) to the
available VRAM headroom after weights are loaded.

KV fill ratio = kv_at_full_context / vram_headroom_after_weights

Risk thresholds
---------------
  ratio ≤ 0.70   LOW       ≥30% VRAM headroom remains after KV fills — safe
  ratio ≤ 0.85   MEDIUM    10–30% headroom — generally safe, watch closely
  ratio ≤ 0.97   HIGH      <10% headroom — very likely OOM near full context
  ratio > 0.97   CRITICAL  <3% headroom — guaranteed OOM, block inference

OOM probability model
---------------------
A sigmoid-like piecewise linear function maps the fill ratio to an
OOM probability percentage. This is an engineering approximation that
produces human-intuitive values:

  ratio 0.0   →  0%
  ratio 0.70  → ~5%     (LOW boundary)
  ratio 0.85  → ~25%    (MEDIUM boundary)
  ratio 0.97  → ~70%    (HIGH boundary)
  ratio 1.00  → ~97%    (CRITICAL — not quite 100% due to runtime headroom)
  ratio > 1.0 →  99%    (certain OOM)

The model acknowledges that actual OOM depends on system runtime behaviour
(OS page cache, driver overhead) so we never return exactly 100%.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Tuple


# ---------------------------------------------------------------------------
# OomRisk
# ---------------------------------------------------------------------------

class OomRisk(str, Enum):
    """
    Four-level OOM risk classification.

    Values are ordered by severity (LOW < MEDIUM < HIGH < CRITICAL).
    Comparison is not meaningful via Enum ordering; compare via
    ``OomRisk.severity()``.
    """
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.value

    def severity(self) -> int:
        """Return numeric severity index (0=LOW, 3=CRITICAL)."""
        return {
            OomRisk.LOW:      0,
            OomRisk.MEDIUM:   1,
            OomRisk.HIGH:     2,
            OomRisk.CRITICAL: 3,
        }[self]

    def is_actionable(self) -> bool:
        """True if this risk level warrants automatic intervention."""
        return self in (OomRisk.HIGH, OomRisk.CRITICAL)

    def emoji(self) -> str:
        """Short emoji indicator for terminal output."""
        return {
            OomRisk.LOW:      "✅",
            OomRisk.MEDIUM:   "⚠️",
            OomRisk.HIGH:     "🔴",
            OomRisk.CRITICAL: "💀",
        }[self]


# ---------------------------------------------------------------------------
# Risk thresholds (module-level constants for easy tuning)
# ---------------------------------------------------------------------------

_THRESHOLD_LOW      = 0.70   # ratio ≤ this → LOW
_THRESHOLD_MEDIUM   = 0.85   # ratio ≤ this → MEDIUM
_THRESHOLD_HIGH     = 0.97   # ratio ≤ this → HIGH
                              # ratio > HIGH → CRITICAL


# ---------------------------------------------------------------------------
# RiskClassifier
# ---------------------------------------------------------------------------

class RiskClassifier:
    """
    Classifies OOM risk and estimates OOM probability from memory metrics.

    All methods are stateless and may be called directly or via the
    class constructor (no required parameters).
    """

    # Piecewise linear segments for OOM probability model
    # Each tuple: (ratio_lo, ratio_hi, prob_lo, prob_hi)
    _PROB_SEGMENTS = [
        (0.00, _THRESHOLD_LOW,    0.00,  0.05),   # 0% → 5%
        (_THRESHOLD_LOW,    _THRESHOLD_MEDIUM, 0.05,  0.25),   # 5% → 25%
        (_THRESHOLD_MEDIUM, _THRESHOLD_HIGH,   0.25,  0.70),   # 25% → 70%
        (_THRESHOLD_HIGH,   1.00,              0.70,  0.97),   # 70% → 97%
    ]

    def classify(
        self,
        vram_total_bytes: int,
        peak_vram_bytes: int,
        kv_fill_ratio: float,
    ) -> Tuple[OomRisk, float]:
        """
        Classify OOM risk and compute modelled OOM probability.

        Parameters
        ----------
        vram_total_bytes : int
            Total VRAM capacity of the GPU (bytes).
        peak_vram_bytes : int
            Estimated peak VRAM usage (weights + KV at full context + buffers).
        kv_fill_ratio : float
            ``kv_at_full_context / vram_headroom_after_weights``.
            Values > 1.0 indicate guaranteed OOM.

        Returns
        -------
        Tuple[OomRisk, float]
            ``(risk_level, oom_probability_0_to_1)``
        """
        ratio = kv_fill_ratio

        # Classify
        if ratio <= _THRESHOLD_LOW:
            risk = OomRisk.LOW
        elif ratio <= _THRESHOLD_MEDIUM:
            risk = OomRisk.MEDIUM
        elif ratio <= _THRESHOLD_HIGH:
            risk = OomRisk.HIGH
        else:
            risk = OomRisk.CRITICAL

        # Also consider absolute VRAM usage
        if vram_total_bytes > 0:
            abs_util = peak_vram_bytes / vram_total_bytes
            if abs_util > 1.0 and risk != OomRisk.CRITICAL:
                risk = OomRisk.CRITICAL
        
        # Compute OOM probability
        oom_prob = self._compute_probability(ratio)

        return risk, oom_prob

    def classify_from_bytes(
        self,
        vram_total_bytes: int,
        weights_vram_bytes: int,
        kv_at_full_context_bytes: int,
        extra_buffers_bytes: int = 0,
    ) -> Tuple[OomRisk, float]:
        """
        Convenience wrapper that computes the fill ratio from raw byte values.

        Parameters
        ----------
        vram_total_bytes : int
            Total VRAM capacity.
        weights_vram_bytes : int
            VRAM consumed by GPU-placed model weights.
        kv_at_full_context_bytes : int
            KV cache bytes at 100% context fill.
        extra_buffers_bytes : int
            Additional overhead (activation workspace + backend context).

        Returns
        -------
        Tuple[OomRisk, float]
            ``(risk_level, oom_probability_0_to_1)``
        """
        # Headroom = VRAM remaining after weights and fixed buffers
        headroom = vram_total_bytes - weights_vram_bytes - extra_buffers_bytes
        headroom = max(1, headroom)  # avoid division by zero

        ratio = kv_at_full_context_bytes / headroom
        peak = weights_vram_bytes + kv_at_full_context_bytes + extra_buffers_bytes

        return self.classify(
            vram_total_bytes=vram_total_bytes,
            peak_vram_bytes=peak,
            kv_fill_ratio=ratio,
        )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _compute_probability(self, ratio: float) -> float:
        """
        Map a KV fill ratio to an OOM probability using piecewise linear interpolation.

        Returns a value in [0.0, 1.0].
        """
        if ratio >= 1.0:
            return 0.99  # Near-certain, but never exactly 1.0

        for r_lo, r_hi, p_lo, p_hi in self._PROB_SEGMENTS:
            if r_lo <= ratio <= r_hi:
                if r_hi == r_lo:
                    return p_lo
                # Linear interpolation within segment
                t = (ratio - r_lo) / (r_hi - r_lo)
                return p_lo + t * (p_hi - p_lo)

        # Below 0 → 0% probability
        return 0.0
