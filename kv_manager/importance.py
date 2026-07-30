"""
importance.py
-------------
Token importance analysis engine for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TokenBlockImportance:
    """Importance score for a block of KV tokens."""
    block_index: int
    start_token: int
    end_token: int
    role: str  # "system", "user", "assistant"
    recency_score: float
    structure_score: float
    reuse_score: float
    total_importance: float


class ImportanceAnalyzer:
    """
    Analyzes conversation structure and recency to assign importance scores to KV cache blocks.
    """

    def analyze_blocks(
        self,
        total_tokens: int,
        block_size: int = 512,
        system_prompt_tokens: int = 256,
    ) -> List[TokenBlockImportance]:
        """
        Divide context into blocks and calculate importance scores.
        """
        if total_tokens <= 0:
            return []

        blocks: List[TokenBlockImportance] = []
        num_blocks = (total_tokens + block_size - 1) // block_size

        for i in range(num_blocks):
            start = i * block_size
            end = min(total_tokens, (i + 1) * block_size)

            # Recency Score (0.1 to 1.0, newest block gets 1.0)
            recency = 0.1 + (i / max(1, num_blocks - 1)) * 0.9 if num_blocks > 1 else 1.0

            # Structural Protection Score (System prompt = 1.0, early user prompt = 0.7, assistant = 0.5)
            if start < system_prompt_tokens:
                role = "system"
                structure = 1.0
            elif i % 2 == 0:
                role = "user"
                structure = 0.75
            else:
                role = "assistant"
                structure = 0.50

            # Reuse Probability Score
            reuse = 0.9 if role == "system" else (0.6 if role == "user" else 0.4)

            total_score = (recency * 0.40) + (structure * 0.40) + (reuse * 0.20)

            blocks.append(
                TokenBlockImportance(
                    block_index=i,
                    start_token=start,
                    end_token=end,
                    role=role,
                    recency_score=round(recency, 3),
                    structure_score=round(structure, 3),
                    reuse_score=round(reuse, 3),
                    total_importance=round(total_score, 3),
                )
            )

        return blocks
