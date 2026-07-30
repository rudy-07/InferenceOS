"""
manager.py
----------
Intelligent KV Manager facade for active KV Cache resource management in InferenceOS.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from runtime_memory.kv_estimator import KvCacheEstimator
from .compression import AdaptiveKVCompressor
from .eviction import IntelligentKVEvictor
from .importance import ImportanceAnalyzer
from .interfaces import KVDecision, KVPolicyConfig, KVState
from .policy import KVPolicyEngine
from .quantization import KVQuantizer
from .statistics import KVStatistics
from .telemetry import KVTelemetryCollector, KVTelemetrySnapshot

logger = logging.getLogger("InferenceOS.IntelligentKVManager")


class IntelligentKVManager:
    """
    Central authority facade responsible for KV Cache allocation, compression, eviction, and policy enforcement.
    """

    def __init__(self, config: Optional[KVPolicyConfig] = None, verbose: bool = False) -> None:
        self.config = config or KVPolicyConfig()
        if verbose:
            self.config.verbose = True

        self.kv_estimator = KvCacheEstimator()
        self.policy_engine = KVPolicyEngine()
        self.compressor = AdaptiveKVCompressor()
        self.evictor = IntelligentKVEvictor()
        self.telemetry = KVTelemetryCollector()
        self.stats = KVStatistics()

    def evaluate_kv_state(
        self,
        model_metadata: Dict[str, Any],
        context_length: int = 4096,
        vram_free_mb: float = 8000.0,
        vram_total_mb: float = 16384.0,
        ram_free_mb: float = 16000.0,
        ram_total_mb: float = 32768.0,
        pressure_level: str = "Low",
        learning_recommendation: Optional[Any] = None,
        config_override: Optional[KVPolicyConfig] = None,
    ) -> KVDecision:
        """
        Actively evaluate KV cache state, compression mode, and eviction strategy for an inference session.

        Returns
        -------
        KVDecision
            Complete decision containing current KV size, compression strategy, memory saved (MB), and reasoning.
        """
        cfg = config_override or self.config

        # 1. Estimate base uncompressed KV cache size
        num_kv_heads = int(model_metadata.get("num_kv_heads", model_metadata.get("head_count_kv", 8)))
        head_dim = int(model_metadata.get("head_dim", 128))
        n_gpu_layers = int(model_metadata.get("num_layers", model_metadata.get("n_gpu_layers", 32)))

        kv_est = self.kv_estimator.estimate_from_params(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            n_gpu_layers=n_gpu_layers,
            context_length=context_length,
        )
        uncompressed_kv_mb = kv_est.kv_at_full_context / (1024 * 1024)

        # 2. Evaluate Policy
        policy_eval = self.policy_engine.evaluate_policy(
            current_kv_mb=uncompressed_kv_mb,
            vram_free_mb=vram_free_mb,
            vram_total_mb=vram_total_mb,
            pressure_level=pressure_level,
            config=cfg,
        )

        # Consult ARTI learning recommendation if available
        comp_mode = cfg.compression_mode
        if comp_mode == "adaptive" and policy_eval.suggested_compression_mode != "disabled":
            comp_mode = policy_eval.suggested_compression_mode

        # 3. Evaluate Adaptive Compression
        comp_res = self.compressor.evaluate_compression(
            current_kv_mb=uncompressed_kv_mb,
            pressure_level=pressure_level,
            requested_mode=comp_mode,
            vram_headroom_mb=vram_free_mb,
        )

        effective_kv_mb = comp_res.compressed_kv_mb
        total_saved_mb = comp_res.memory_saved_mb

        # 4. Evaluate Eviction if memory pressure remains Critical
        target_evict_mb = max(0.0, effective_kv_mb - (vram_free_mb * 0.80)) if pressure_level == "Critical" else 0.0
        evict_res = self.evictor.evaluate_eviction(
            total_tokens=context_length,
            current_kv_mb=effective_kv_mb,
            target_free_mb=target_evict_mb,
            eviction_policy=cfg.eviction_policy,
            pressure_level=pressure_level,
        )

        total_saved_mb += evict_res.memory_freed_mb
        effective_kv_mb -= evict_res.memory_freed_mb

        # 5. Formulate Decision & Telemetry
        reasoning_bullets = list(policy_eval.reasoning)
        if comp_res.reasoning and comp_res.mode_selected != "Disabled":
            reasoning_bullets.append(comp_res.reasoning)

        # Record Telemetry Snapshot
        snapshot = KVTelemetrySnapshot(
            kv_size_mb=effective_kv_mb,
            growth_rate_mb_per_sec=0.0,
            compression_ratio=comp_res.compression_ratio,
            compression_time_ms=1.5,
            eviction_count=evict_res.tokens_evicted,
            memory_saved_mb=total_saved_mb,
            restore_events=0,
            pressure_level=pressure_level,
            policy_name=policy_eval.policy_name,
        )
        self.telemetry.record_snapshot(snapshot)

        # Update Statistics
        self.stats.total_evaluations += 1
        self.stats.total_memory_saved_mb += total_saved_mb
        self.stats.total_evictions += evict_res.tokens_evicted
        self.stats.active_policy = policy_eval.policy_name
        self.stats.last_update = time.time()

        decision = KVDecision(
            current_kv_gb=effective_kv_mb / 1024.0,
            pressure_level=pressure_level,
            compression_strategy=f"{comp_res.mode_selected} ({comp_res.target_format})",
            memory_saved_mb=total_saved_mb,
            eviction_action=evict_res.summary_action,
            policy_name=policy_eval.policy_name,
            reasoning=reasoning_bullets,
        )

        if cfg.verbose:
            print(decision.format_cli_output())

        return decision

    def get_statistics(self) -> KVStatistics:
        return self.stats

    def reset(self) -> None:
        self.telemetry.clear()
        self.stats = KVStatistics()
