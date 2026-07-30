"""
search.py
---------
Intelligent search engine for Automatic Performance Optimizer in InferenceOS.
"""
from __future__ import annotations

from typing import List, Tuple

from .interfaces import CandidateConfig, HardwareFingerprintData, ModelFingerprintData


class IntelligentSearchEngine:
    """
    Intelligently generates and scores candidate configurations.
    """

    def generate_and_rank_candidates(
        self,
        model_fp: ModelFingerprintData,
        hw_fp: HardwareFingerprintData,
        goal: str = "Balanced",
    ) -> List[CandidateConfig]:
        """
        Generate candidate search space and rank candidates based on optimization goal.
        """
        max_layers = model_fp.n_layers
        candidates: List[CandidateConfig] = []

        # Candidate 1: Full GPU offload with 768 microbatch
        candidates.append(
            CandidateConfig(
                gpu_layers=max_layers,
                microbatch_size=768 if hw_fp.vram_gb >= 6.0 else 512,
                context_length=min(16384, model_fp.context_capability),
                memory_strategy="balanced",
                thread_count=8,
                score=95.0,
            )
        )

        # Candidate 2: High microbatch (1024)
        if hw_fp.vram_gb >= 8.0:
            candidates.append(
                CandidateConfig(
                    gpu_layers=max_layers,
                    microbatch_size=1024,
                    context_length=min(16384, model_fp.context_capability),
                    memory_strategy="aggressive",
                    thread_count=8,
                    score=92.0,
                )
            )

        # Candidate 3: Conservative microbatch (256)
        candidates.append(
            CandidateConfig(
                gpu_layers=max(1, max_layers - 2),
                microbatch_size=256,
                context_length=min(8192, model_fp.context_capability),
                memory_strategy="conservative",
                thread_count=6,
                score=88.0,
            )
        )

        # Goal-based ranking adjustment
        g_lower = goal.lower()
        if "throughput" in g_lower:
            candidates.sort(key=lambda c: c.microbatch_size, reverse=True)
        elif "memory" in g_lower:
            candidates.sort(key=lambda c: c.gpu_layers)
        else:  # Balanced
            candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates
