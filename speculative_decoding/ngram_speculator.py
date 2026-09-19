"""
ngram_speculator.py
-------------------
N-gram suffix speculator for Phase 5 Speculative Decoding Suite in InferenceOS.

Searches the model's own previously generated token sequence for n-gram suffix
matches and returns candidate continuation token IDs as draft sequences.

Design decisions
----------------
- **Zero GPU memory overhead**: operates entirely on Python collections and
  numpy integer arrays.  No additional VRAM is allocated.
- **Ring-buffer context**: maintains a sliding window of the last
  ``max_context_tokens`` generated token IDs using a ``collections.deque``.
  Older tokens are discarded to bound memory usage.
- **Greedy longest-match first**: tries n-gram sizes from ``ngram_size`` down
  to ``min_match`` to maximise draft quality before falling back to shorter
  matches.
- **Draft deduplication**: if the same candidate continuation appears multiple
  times in the context, the most recent occurrence is preferred (recency bias).
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

logger = logging.getLogger("InferenceOS.SpeculativeDecoding.NGram")


class NGramSpeculator:
    """
    N-gram context speculator.

    Maintains a rolling history of generated token IDs and uses suffix-matching
    to predict the next ``n_drafts`` tokens without any additional model forward
    passes.

    Parameters
    ----------
    ngram_size : int
        Primary n-gram length to match (2–8).  Larger sizes reduce false-positive
        matches but require a longer history to find hits.
    min_match_length : int
        Minimum n-gram size to attempt before giving up.  Must be ≥ 1.
    max_context_tokens : int
        Rolling window length for the generated token history.  Default 2048 is
        sufficient for most use-cases; increase for very long generation loops.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        min_match_length: int = 2,
        max_context_tokens: int = 2048,
    ) -> None:
        if ngram_size < 1:
            raise ValueError("ngram_size must be ≥ 1")
        if min_match_length < 1:
            min_match_length = 1
        if min_match_length > ngram_size:
            min_match_length = ngram_size

        self.ngram_size = ngram_size
        self.min_match_length = min_match_length

        # Rolling ring-buffer of generated token IDs (int)
        self._history: Deque[int] = deque(maxlen=max_context_tokens)

        # Counters for telemetry
        self.total_calls: int = 0
        self.total_hits: int = 0
        self.total_drafts_generated: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, token_ids: Sequence[int]) -> None:
        """
        Append newly generated token IDs to the rolling history.

        Call this after each verified/accepted generation step so the speculator's
        context stays in sync with the base model's output.

        Parameters
        ----------
        token_ids : sequence of int
            One or more token IDs produced by the model in the last step.
        """
        self._history.extend(token_ids)

    def generate_drafts(
        self,
        n_drafts: int = 5,
    ) -> List[int]:
        """
        Generate up to ``n_drafts`` speculative token IDs from the rolling history.

        Searches the history for the longest n-gram suffix match starting at
        ``ngram_size`` and working down to ``min_match_length``.

        Parameters
        ----------
        n_drafts : int
            Number of draft tokens to return.

        Returns
        -------
        list of int
            Speculative token IDs.  Empty list if no match is found.
        """
        self.total_calls += 1
        history = list(self._history)

        if len(history) < self.min_match_length + 1:
            # Not enough history to match any n-gram suffix
            return []

        # Try matching from largest to smallest n-gram size
        for n in range(min(self.ngram_size, len(history) - 1), self.min_match_length - 1, -1):
            suffix = history[-n:]          # last n tokens = search key
            drafts = self._find_continuations(history, suffix, n_drafts)
            if drafts:
                self.total_hits += 1
                self.total_drafts_generated += len(drafts)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "NGram hit: n=%d  suffix=%s  drafts=%s",
                        n, suffix[-3:], drafts[:3],
                    )
                return drafts

        return []

    def reset(self) -> None:
        """Clear the rolling history (call between unrelated sessions)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Fraction of generate_drafts() calls that produced ≥ 1 draft token."""
        return self.total_hits / max(1, self.total_calls)

    def get_stats(self) -> dict:
        """Return telemetry statistics as a plain dict."""
        return {
            "total_calls": self.total_calls,
            "total_hits": self.total_hits,
            "total_drafts_generated": self.total_drafts_generated,
            "hit_rate": round(self.hit_rate, 4),
            "history_length": len(self._history),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_continuations(
        self,
        history: List[int],
        suffix: List[int],
        n_drafts: int,
    ) -> List[int]:
        """
        Search ``history`` for the most recent occurrence of ``suffix`` and
        return the ``n_drafts`` tokens that immediately follow it.

        Parameters
        ----------
        history : list of int
            Full flattened token history.
        suffix : list of int
            N-gram pattern to search for.
        n_drafts : int
            How many continuation tokens to extract.

        Returns
        -------
        list of int
            Continuation tokens, possibly fewer than ``n_drafts`` if the match
            is near the end of history.
        """
        n = len(suffix)
        # Search backward from the second-to-last possible position
        # (the last ``n`` tokens are the suffix itself)
        search_end = len(history) - n  # exclusive upper bound for match start
        best_match_start: Optional[int] = None

        for i in range(search_end - 1, -1, -1):
            if history[i : i + n] == suffix:
                best_match_start = i
                break   # most recent match wins

        if best_match_start is None:
            return []

        # Extract up to n_drafts tokens after the match
        cont_start = best_match_start + n
        cont_end = min(cont_start + n_drafts, len(history))
        return history[cont_start:cont_end]
