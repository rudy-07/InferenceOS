"""
config.py
---------
Configuration dataclass for Phase 5 Speculative Decoding Suite in InferenceOS.

All fields map 1-to-1 with RuntimeConfig.spec_* keys, allowing the
SpeculativeOrchestrator to be constructed from either a SpeculativeDecoderConfig
or directly from a RuntimeConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpeculativeDecoderConfig:
    """
    Configuration for the Speculative Decoding Orchestrator.

    Parameters
    ----------
    enabled : bool
        Master switch. Default False (opt-in).
    mode : str
        Active speculation strategy.
        - ``"ngram"``         — N-gram suffix matching from prior generated text.
        - ``"prompt_lookup"`` — Match suffixes against the original input prompt.
        - ``"eagle"``         — Eagle-2 / external draft GGUF model heads.
        - ``"medusa"``        — Medusa-style parallel self-draft heads (stub for Phase 5).
        - ``"auto"``          — Auto-select: ngram when no draft model is available,
                                eagle when ``eagle_draft_model_path`` is provided.
    draft_tokens : int
        Number of draft tokens generated per speculation step. Increasing this
        raises the speculation window but may reduce acceptance rate if the model
        is highly variable. Good default: 5.
    ngram_size : int
        N-gram window size (2–8). Larger values reduce false matches but may lower
        hit rate on short contexts. Good default: 3.
    min_match_length : int
        Minimum matching prefix length for prompt-lookup decoding (tokens).
        Below this length, no draft is generated. Default: 3.
    prompt_lookup_window : int
        Maximum number of tokens to search backward in the prompt during
        prompt-lookup speculation. 0 = search entire prompt. Default: 0.
    acceptance_threshold : float
        Minimum token acceptance probability (0–1). Tokens with a base model
        probability below this threshold are rejected. Default: 0.0 (always
        accept if the draft matches the greedy top-1 token).
    acceptance_strategy : str
        Token acceptance algorithm.
        - ``"greedy"``      — Accept draft token only if it equals the base model's
                              argmax token (lossless, fastest verification).
        - ``"speculative"`` — Standard speculative sampling: accept with probability
                              min(1, p_base / p_draft) (matches base distribution).
    eagle_draft_model_path : str, optional
        Filesystem path to a compatible Eagle-2 draft GGUF model. Required when
        ``mode="eagle"``. Ignored for ngram / prompt_lookup modes.
    max_speculation_rounds : int
        Safety cap on consecutive speculation cycles per generation step.
        Prevents infinite draft loops on very long outputs. Default: 8.
    fallback_to_greedy : bool
        When True (default), fall back to standard greedy decoding if the
        speculator produces zero draft tokens (e.g. context too short for ngram).
    record_telemetry : bool
        Write per-run acceptance rate and speedup statistics to the Runtime
        Learning SQLite database for trend analysis. Default: True.
    verbose : bool
        Print per-step speculation statistics to console. Default: False.
    """

    enabled: bool = False

    # Strategy
    mode: str = "auto"                  # "ngram" | "prompt_lookup" | "eagle" | "medusa" | "auto"

    # Draft parameters
    draft_tokens: int = 5
    ngram_size: int = 3
    min_match_length: int = 3
    prompt_lookup_window: int = 0       # 0 = entire prompt

    # Acceptance
    acceptance_threshold: float = 0.0
    acceptance_strategy: str = "greedy" # "greedy" | "speculative"

    # Eagle mode
    eagle_draft_model_path: Optional[str] = None

    # Safety
    max_speculation_rounds: int = 8
    fallback_to_greedy: bool = True

    # Observability
    record_telemetry: bool = True
    verbose: bool = False

    # ── Derived / internal ────────────────────────────────────────────────

    @classmethod
    def from_runtime_config(cls, cfg: object) -> "SpeculativeDecoderConfig":
        """
        Construct from a ``RuntimeConfig`` instance using the ``spec_*`` fields.

        Parameters
        ----------
        cfg : RuntimeConfig
            The top-level inference runtime configuration object.

        Returns
        -------
        SpeculativeDecoderConfig
        """
        return cls(
            enabled=getattr(cfg, "enable_speculative_decoding", False),
            mode=getattr(cfg, "spec_mode", "auto"),
            draft_tokens=getattr(cfg, "spec_draft_tokens", 5),
            ngram_size=getattr(cfg, "spec_ngram_size", 3),
            min_match_length=getattr(cfg, "spec_min_match_length", 3),
            prompt_lookup_window=getattr(cfg, "spec_prompt_lookup_window", 0),
            acceptance_threshold=getattr(cfg, "spec_acceptance_threshold", 0.0),
            acceptance_strategy=getattr(cfg, "spec_acceptance_strategy", "greedy"),
            eagle_draft_model_path=getattr(cfg, "spec_eagle_draft_model", None),
            max_speculation_rounds=getattr(cfg, "spec_max_rounds", 8),
            fallback_to_greedy=getattr(cfg, "spec_fallback_to_greedy", True),
            record_telemetry=getattr(cfg, "spec_record_telemetry", True),
            verbose=getattr(cfg, "verbose_spec_decoding", False),
        )

    def resolve_mode(self) -> str:
        """
        Resolve ``"auto"`` to a concrete mode based on available resources.

        Returns
        -------
        str
            One of: ``"ngram"``, ``"prompt_lookup"``, ``"eagle"``.
        """
        if self.mode != "auto":
            return self.mode
        if self.eagle_draft_model_path:
            return "eagle"
        return "ngram"
