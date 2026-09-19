"""
prompt_lookup_speculator.py
----------------------------
Prompt-Lookup Decoding speculator for Phase 5 Speculative Decoding Suite.

Implements the Prompt-Lookup Decoding algorithm (Saxena 2023):
instead of searching previously *generated* tokens (as in n-gram speculation),
this speculator searches the original *input prompt* for suffix matches to the
last K generated tokens, then returns the prompt continuation as draft tokens.

This is particularly effective for workloads that exhibit high overlap between
generation and input content:
  - Document summarisation (model paraphrases input phrases)
  - RAG-grounded answers (model cites retrieved passages verbatim)
  - Code completion (model continues from function signatures in the prompt)
  - JSON/schema structured outputs (model copies field names / boilerplate)

Design decisions
----------------
- Entirely CPU-side operation — no GPU memory allocated.
- Operates on raw token IDs (integers), backend-agnostic.
- Lookup window is configurable to limit search cost on very long prompts.
- Returns the longest matching continuation found; shorter fallbacks are tried
  automatically if the primary match produces insufficient draft tokens.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger("InferenceOS.SpeculativeDecoding.PromptLookup")


class PromptLookupSpeculator:
    """
    Prompt-Lookup Decoding speculator.

    Searches the input prompt token sequence for occurrences of the last
    ``match_length`` generated tokens and returns the subsequent prompt tokens
    as draft candidates.

    Parameters
    ----------
    min_match_length : int
        Minimum suffix length (in tokens) required to trigger a draft.  Longer
        values improve precision but reduce hit rate.  Default: 3.
    lookup_window : int
        Maximum number of prompt tokens to search backward from the end.
        0 = search the entire prompt (default).  Limiting the window reduces
        search cost on very long prompts but may miss earlier matches.
    """

    def __init__(
        self,
        min_match_length: int = 3,
        lookup_window: int = 0,
    ) -> None:
        self.min_match_length = max(1, min_match_length)
        self.lookup_window = lookup_window  # 0 = unbounded

        # The original prompt token IDs — set once per inference call
        self._prompt_ids: List[int] = []

        # Rolling suffix of generated token IDs (last K tokens)
        self._generated_suffix: List[int] = []

        # Telemetry
        self.total_calls: int = 0
        self.total_hits: int = 0
        self.total_drafts_generated: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def set_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """
        Register the input prompt for this inference session.

        Must be called before ``generate_drafts()`` for each new session.

        Parameters
        ----------
        prompt_token_ids : sequence of int
            Full tokenised input prompt as a flat integer sequence.
        """
        self._prompt_ids = list(prompt_token_ids)
        self._generated_suffix.clear()
        logger.debug("PromptLookup: prompt registered, %d tokens", len(self._prompt_ids))

    def update(self, token_ids: Sequence[int]) -> None:
        """
        Append newly verified token IDs to the generated suffix buffer.

        Parameters
        ----------
        token_ids : sequence of int
            Token IDs accepted by the base model in the last step.
        """
        self._generated_suffix.extend(token_ids)
        # Keep only the tail needed for matching — bounded by 2× the match length
        # to avoid unbounded growth on long outputs
        max_tail = max(64, self.min_match_length * 4)
        if len(self._generated_suffix) > max_tail:
            self._generated_suffix = self._generated_suffix[-max_tail:]

    def reset(self) -> None:
        """Clear prompt and generated suffix (call between sessions)."""
        self._prompt_ids.clear()
        self._generated_suffix.clear()

    # ------------------------------------------------------------------
    # Core speculation
    # ------------------------------------------------------------------

    def generate_drafts(
        self,
        n_drafts: int = 5,
    ) -> List[int]:
        """
        Generate up to ``n_drafts`` speculative token IDs from the prompt.

        Algorithm
        ---------
        1. Take the last ``match_length`` generated tokens as the search key.
        2. Search the prompt (within ``lookup_window``) for the most recent
           occurrence of that key.
        3. Return the ``n_drafts`` tokens immediately following the match in the
           prompt as draft candidates.
        4. If no match is found with ``match_length``, decrement and retry down
           to ``min_match_length``.

        Parameters
        ----------
        n_drafts : int
            Desired number of draft tokens.

        Returns
        -------
        list of int
            Draft token IDs.  May be shorter than ``n_drafts`` if the match
            is near the end of the prompt.  Empty if no match found.
        """
        self.total_calls += 1

        if not self._prompt_ids:
            return []
        if len(self._generated_suffix) < self.min_match_length:
            return []

        # Determine the searchable portion of the prompt
        prompt = self._prompt_ids
        if self.lookup_window > 0:
            prompt = prompt[-self.lookup_window:]

        # Try progressively shorter suffixes
        max_try = min(len(self._generated_suffix), 8)  # cap at 8 to bound search cost
        for match_len in range(max_try, self.min_match_length - 1, -1):
            suffix = self._generated_suffix[-match_len:]
            drafts = self._search_prompt(prompt, suffix, n_drafts)
            if drafts:
                self.total_hits += 1
                self.total_drafts_generated += len(drafts)
                logger.debug(
                    "PromptLookup hit: match_len=%d  suffix=%s  drafts=%s",
                    match_len, suffix[-3:], drafts[:3],
                )
                return drafts

        return []

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Fraction of generate_drafts() calls that produced ≥ 1 draft token."""
        return self.total_hits / max(1, self.total_calls)

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_hits": self.total_hits,
            "total_drafts_generated": self.total_drafts_generated,
            "hit_rate": round(self.hit_rate, 4),
            "prompt_length": len(self._prompt_ids),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search_prompt(
        self,
        prompt: List[int],
        suffix: List[int],
        n_drafts: int,
    ) -> List[int]:
        """
        Linear scan of ``prompt`` for the most recent occurrence of ``suffix``.

        Returns the ``n_drafts`` tokens immediately after the match.  Searches
        backward (most-recent match) so the continuation is contextually closest
        to the current generation position.
        """
        n = len(suffix)
        search_end = len(prompt) - n   # last valid start index for a full match

        if search_end < 0:
            return []

        for i in range(search_end - 1, -1, -1):
            if prompt[i : i + n] == suffix:
                cont_start = i + n
                cont_end = min(cont_start + n_drafts, len(prompt))
                return prompt[cont_start:cont_end]

        return []
