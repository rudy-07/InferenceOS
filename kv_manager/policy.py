"""
policy.py
---------
Policy Engine for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .interfaces import KVPolicyConfig


@dataclass
class PolicyEvaluation:
    policy_name: str
    compression_required: bool
    eviction_required: bool
    suggested_compression_mode: str
    suggested_eviction_policy: str
    reasoning: List[str]


class KVPolicyEngine:
    """
    Evaluates memory pressure thresholds and applies configurable policies.
    """

    def evaluate_policy(
        self,
        current_kv_mb: float,
        vram_free_mb: float,
        vram_total_mb: float,
        pressure_level: str = "Low",
        config: Optional[KVPolicyConfig] = None,
    ) -> PolicyEvaluation:
        """
        Evaluate memory pressure against policy thresholds.
        """
        cfg = config or KVPolicyConfig()

        if not cfg.enabled:
            return PolicyEvaluation(
                policy_name="Disabled",
                compression_required=False,
                eviction_required=False,
                suggested_compression_mode="disabled",
                suggested_eviction_policy="disabled",
                reasoning=["KV Policy Engine is disabled."],
            )

        used_pct = ((vram_total_mb - vram_free_mb) / max(1.0, vram_total_mb)) * 100.0 if vram_total_mb > 0 else 0.0

        reasoning: List[str] = []
        comp_req = False
        evict_req = False

        if pressure_level == "Critical" or used_pct >= cfg.eviction_threshold_pct:
            comp_req = cfg.compression_enabled
            evict_req = cfg.eviction_enabled
            comp_mode = "aggressive"
            evict_pol = cfg.eviction_policy
            reasoning.append(f"Critical memory pressure ({used_pct:.1f}% used): aggressive compression & eviction active.")
            policy_name = "Critical Memory Policy"
        elif pressure_level == "High" or used_pct >= cfg.compression_threshold_pct:
            comp_req = cfg.compression_enabled
            comp_mode = "balanced"
            evict_pol = cfg.eviction_policy
            reasoning.append(f"High memory pressure ({used_pct:.1f}% used): balanced compression active.")
            policy_name = "High Memory Policy"
        elif pressure_level == "Medium":
            comp_req = cfg.compression_enabled
            comp_mode = "lossless"
            evict_pol = cfg.eviction_policy
            reasoning.append("Medium memory pressure: lossless compression recommended for safe headroom.")
            policy_name = "Balanced Policy"
        else:
            comp_mode = "lossless" if cfg.compression_mode == "adaptive" else cfg.compression_mode
            evict_pol = cfg.eviction_policy
            reasoning.append("Memory pressure within normal operational boundaries.")
            policy_name = "Adaptive Policy"

        return PolicyEvaluation(
            policy_name=policy_name,
            compression_required=comp_req,
            eviction_required=evict_req,
            suggested_compression_mode=comp_mode,
            suggested_eviction_policy=evict_pol,
            reasoning=reasoning,
        )
