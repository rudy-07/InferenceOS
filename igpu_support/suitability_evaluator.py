"""
suitability_evaluator.py
------------------------
Overall iGPU suitability evaluation for Phase 7 Integrated GPU Support.

Combines the iGPU suitability score (from IgpuProfiler) with the memory
bus contention estimate (from ContentionModel) to produce a single
IgpuWorkloadSuitability decision: FULL, LIGHT, or DISABLED.

Decision table
--------------
The evaluator applies thresholds to both score and contention risk:

  Score       Contention    →  Suitability
  ──────────  ────────────     ───────────
  ≥ 0.60      LOW or MEDIUM →  FULL
  ≥ 0.60      HIGH          →  LIGHT    (transformer blocks blocked by contention)
  ≥ 0.35      any           →  LIGHT    (only embedding/norm/lm_head)
  < 0.35      any           →  DISABLED

Additional override conditions that force DISABLED regardless of score:
  - igpu_profile.enabled == False  (set by IgpuProfiler on min-VRAM/backend failure)
  - contention_risk == HIGH AND score < 0.50  (high contention + mediocre score)
  - RAM utilisation > 95%  (system critically low on memory)

These thresholds are module-level constants so they can be tuned without
touching business logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .contention_model import ContentionEstimate
from .igpu_profiler import IgpuProfile
from .workload_classifier import IgpuWorkloadSuitability


# ---------------------------------------------------------------------------
# Thresholds (module-level constants for easy tuning)
# ---------------------------------------------------------------------------

FULL_SCORE_THRESHOLD      = 0.60   # score ≥ this → potentially FULL
LIGHT_SCORE_THRESHOLD     = 0.35   # score ≥ this → LIGHT
RAM_CRITICAL_THRESHOLD    = 95.0   # RAM% above which iGPU is disabled
CONTENTION_OVERRIDE_SCORE = 0.50   # if HIGH contention AND score < this → DISABLED


# ---------------------------------------------------------------------------
# SuitabilityResult
# ---------------------------------------------------------------------------

@dataclass
class SuitabilityResult:
    """
    Output of the suitability evaluation.

    Attributes
    ----------
    suitability : IgpuWorkloadSuitability
        Final decision: FULL, LIGHT, or DISABLED.
    score : float
        Raw iGPU suitability score (0.0–1.0).
    contention_risk : str
        Contention risk level from ContentionModel.
    effective_bw_gbps : float
        Effective iGPU bandwidth after contention.
    reasons : List[str]
        Human-readable explanation of why this suitability was chosen.
    """
    suitability: IgpuWorkloadSuitability
    score: float
    contention_risk: str
    effective_bw_gbps: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suitability": str(self.suitability),
            "score": round(self.score, 4),
            "contention_risk": self.contention_risk,
            "effective_bw_gbps": round(self.effective_bw_gbps, 2),
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# SuitabilityEvaluator
# ---------------------------------------------------------------------------

class SuitabilityEvaluator:
    """
    Combines IgpuProfile and ContentionEstimate into a final
    :class:`IgpuWorkloadSuitability` decision.

    All thresholds are module-level constants and can be overridden per-call.
    """

    def evaluate(
        self,
        igpu_profile: IgpuProfile,
        contention: ContentionEstimate,
        ram_utilization_pct: float = 0.0,
    ) -> SuitabilityResult:
        """
        Evaluate overall iGPU suitability.

        Parameters
        ----------
        igpu_profile : IgpuProfile
            Profile from :class:`IgpuProfiler`.
        contention : ContentionEstimate
            Contention analysis from :class:`ContentionModel`.
        ram_utilization_pct : float, optional
            Current system RAM utilization. If 0.0, read from contention.

        Returns
        -------
        SuitabilityResult
            Final decision and supporting context.
        """
        score = igpu_profile.suitability_score
        risk  = contention.contention_risk
        bw    = contention.effective_igpu_bandwidth_gbps
        ram_util = ram_utilization_pct or contention.ram_utilization_pct
        reasons: List[str] = []

        # ----------------------------------------------------------------
        # Override conditions → DISABLED
        # ----------------------------------------------------------------
        if not igpu_profile.enabled:
            reasons.append(f"iGPU disabled by profiler: {' | '.join(igpu_profile.warnings)}")
            return SuitabilityResult(
                IgpuWorkloadSuitability.DISABLED, score, risk, bw, reasons
            )

        if ram_util >= RAM_CRITICAL_THRESHOLD:
            reasons.append(
                f"System RAM critically low ({ram_util:.0f}% used ≥ {RAM_CRITICAL_THRESHOLD:.0f}% threshold). "
                "iGPU disabled to avoid memory pressure."
            )
            return SuitabilityResult(
                IgpuWorkloadSuitability.DISABLED, score, risk, bw, reasons
            )

        if risk == "HIGH" and score < CONTENTION_OVERRIDE_SCORE:
            reasons.append(
                f"HIGH bus contention AND low score ({score:.3f} < {CONTENTION_OVERRIDE_SCORE}). "
                "iGPU disabled to prevent performance regression."
            )
            return SuitabilityResult(
                IgpuWorkloadSuitability.DISABLED, score, risk, bw, reasons
            )

        # ----------------------------------------------------------------
        # Normal threshold evaluation
        # ----------------------------------------------------------------
        if score < LIGHT_SCORE_THRESHOLD:
            reasons.append(
                f"iGPU score {score:.3f} below LIGHT threshold {LIGHT_SCORE_THRESHOLD}. "
                "iGPU disabled."
            )
            return SuitabilityResult(
                IgpuWorkloadSuitability.DISABLED, score, risk, bw, reasons
            )

        # Score ≥ LIGHT_THRESHOLD → at least LIGHT
        if score >= FULL_SCORE_THRESHOLD and risk != "HIGH":
            reasons.append(
                f"Score {score:.3f} ≥ FULL threshold {FULL_SCORE_THRESHOLD} "
                f"with {risk} contention risk. FULL mode enabled."
            )
            suitability = IgpuWorkloadSuitability.FULL
        else:
            if score >= FULL_SCORE_THRESHOLD and risk == "HIGH":
                reasons.append(
                    f"Score {score:.3f} qualifies for FULL but HIGH contention "
                    "forces LIGHT mode (transformer blocks disabled)."
                )
            else:
                reasons.append(
                    f"Score {score:.3f} in [LIGHT, FULL) range → LIGHT mode. "
                    "Only embedding, norm, lm_head assigned to iGPU."
                )
            suitability = IgpuWorkloadSuitability.LIGHT

        return SuitabilityResult(suitability, score, risk, bw, reasons)


def evaluate_igpu_suitability(
    igpu_profile: IgpuProfile,
    contention: ContentionEstimate,
    ram_utilization_pct: float = 0.0,
) -> SuitabilityResult:
    """
    Module-level convenience wrapper around :class:`SuitabilityEvaluator`.

    Parameters
    ----------
    igpu_profile : IgpuProfile
        iGPU profile from :func:`profile_igpu`.
    contention : ContentionEstimate
        Contention estimate from :func:`estimate_contention`.
    ram_utilization_pct : float, optional
        Override for RAM utilisation reading.

    Returns
    -------
    SuitabilityResult
    """
    return SuitabilityEvaluator().evaluate(igpu_profile, contention, ram_utilization_pct)
