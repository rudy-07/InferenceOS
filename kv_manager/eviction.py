"""
eviction.py
-----------
Intelligent KV cache eviction engine for InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .importance import ImportanceAnalyzer, TokenBlockImportance


@dataclass
class EvictionResult:
    policy_used: str
    tokens_evicted: int
    memory_freed_mb: float
    evicted_block_indices: List[int]
    summary_action: str


class IntelligentKVEvictor:
    """
    Evaluates KV cache eviction candidates based on recency, importance scores, and memory pressure.
    """

    def __init__(self) -> None:
        self.importance_analyzer = ImportanceAnalyzer()

    def evaluate_eviction(
        self,
        total_tokens: int,
        current_kv_mb: float,
        target_free_mb: float,
        eviction_policy: str = "adaptive",
        pressure_level: str = "Low",
    ) -> EvictionResult:
        """
        Determine which token blocks to evict if target memory recovery is required.
        """
        if target_free_mb <= 0.0 or pressure_level not in ("High", "Critical"):
            return EvictionResult(
                policy_used=eviction_policy.capitalize(),
                tokens_evicted=0,
                memory_freed_mb=0.0,
                evicted_block_indices=[],
                summary_action="None",
            )

        block_size = 512
        blocks = self.importance_analyzer.analyze_blocks(total_tokens, block_size=block_size)
        if not blocks:
            return EvictionResult(eviction_policy.capitalize(), 0, 0.0, [], "None")

        pol = eviction_policy.lower()

        # Sort blocks for eviction candidate selection (lowest score first)
        if pol == "lru":
            # Sort by recency (oldest first)
            candidates = sorted(blocks, key=lambda b: b.recency_score)
        elif pol == "fifo":
            # Sort by block index (first created first)
            candidates = sorted(blocks, key=lambda b: b.block_index)
        elif pol == "lfu":
            # Sort by reuse score
            candidates = sorted(blocks, key=lambda b: b.reuse_score)
        else:  # adaptive / attention_aware
            # Sort by overall combined total importance score
            candidates = sorted(blocks, key=lambda b: b.total_importance)

        # Do not evict system prompt blocks (role == "system")
        evictable = [b for b in candidates if b.role != "system"]

        mb_per_block = current_kv_mb / max(1, len(blocks))
        freed_mb = 0.0
        evicted_indices: List[int] = []
        evicted_tokens = 0

        for b in evictable:
            if freed_mb >= target_free_mb:
                break
            evicted_indices.append(b.block_index)
            freed_mb += mb_per_block
            evicted_tokens += (b.end_token - b.start_token)

        action_summary = f"{pol.upper()} Eviction ({evicted_tokens} tokens, {freed_mb:.1f} MB freed)" if evicted_tokens > 0 else "None"

        return EvictionResult(
            policy_used=pol.capitalize(),
            tokens_evicted=evicted_tokens,
            memory_freed_mb=round(freed_mb, 2),
            evicted_block_indices=evicted_indices,
            summary_action=action_summary,
        )
