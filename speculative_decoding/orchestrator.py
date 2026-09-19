"""
orchestrator.py
---------------
SpeculativeOrchestrator — central controller for Phase 5 Speculative Decoding.

The orchestrator is the single entry point for all speculative decoding logic.
It is instantiated once per ``InferenceSession`` and acts as a **strategy wrapper**
around the base model's text generation loop.

How it integrates with the InferenceOS subprocess architecture
--------------------------------------------------------------
InferenceOS runs llama.cpp as an external process via ``ProcessManager``.  Unlike
in-process libraries (e.g. llama-python), we cannot intercept individual forward
passes.

The orchestrator therefore operates at the **text level** using the following
two-phase protocol:

  Phase A — Draft generation (zero base model cost)
    The NGramSpeculator or PromptLookupSpeculator generates candidate token strings
    from the rolling context without touching the GPU.  Eagle mode runs the draft
    GGUF model as a separate (fast) subprocess.

  Phase B — Verification (base model forward pass)
    The base model's llama.cpp subprocess is invoked with the original prompt
    PLUS the draft tokens appended.  The model verifies (or refutes) the draft
    in a single forward pass by checking whether its output matches the draft
    prefix.  Accepted tokens reduce the total number of needed base model calls.

Speedup model
-------------
In the best case (100% acceptance), each speculation cycle produces ``draft_tokens``
tokens at the cost of 1 base model verification call.  The theoretical speedup
over standard autoregressive decoding is:

    speedup ≈ 1 + draft_tokens × acceptance_rate

At 80% acceptance with 5 draft tokens:  1 + 5 × 0.8 = 5× speedup
At 60% acceptance with 5 draft tokens:  1 + 5 × 0.6 = 4× speedup

In practice, draft latency is not zero (n-gram is O(context_len), Eagle has
subprocess overhead), so real-world speedups are typically 1.3×–2.5×.

Configuration
-------------
All behaviour is controlled by ``SpeculativeDecoderConfig``.  The orchestrator is
disabled (no-op passthrough) when ``config.enabled = False``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .acceptance_sampler import AcceptanceSampler
from .config import SpeculativeDecoderConfig
from .ngram_speculator import NGramSpeculator
from .prompt_lookup_speculator import PromptLookupSpeculator
from .telemetry import SpecDecisionSummary, SpecRunRecord, SpecTelemetryCollector

logger = logging.getLogger("InferenceOS.SpeculativeDecoding.Orchestrator")


class SpeculativeOrchestrator:
    """
    Central controller for the Phase 5 Speculative Decoding Suite.

    Parameters
    ----------
    config : SpeculativeDecoderConfig
        Fully-resolved speculative decoding configuration.
    llama_exe_path : Path, optional
        Path to the llama.cpp executable.  Required for Eagle mode.
    hw_profile : dict, optional
        Hardware profile.  Used by EagleDraftEngine for backend/thread decisions.
    """

    def __init__(
        self,
        config: SpeculativeDecoderConfig,
        llama_exe_path: Optional[Path] = None,
        hw_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.llama_exe_path = llama_exe_path
        self.hw_profile = hw_profile or {}
        self._active_mode: str = config.resolve_mode()

        # ── Instantiate speculators ────────────────────────────────────
        self._ngram = NGramSpeculator(
            ngram_size=config.ngram_size,
            min_match_length=config.min_match_length,
        )
        self._prompt_lookup = PromptLookupSpeculator(
            min_match_length=config.min_match_length,
            lookup_window=config.prompt_lookup_window,
        )
        self._eagle: Optional[Any] = None  # lazy-init below if needed

        if self._active_mode == "eagle" and config.eagle_draft_model_path:
            try:
                from .eagle_draft_engine import EagleDraftEngine
                if llama_exe_path and llama_exe_path.exists():
                    self._eagle = EagleDraftEngine(
                        draft_model_path=config.eagle_draft_model_path,
                        llama_exe_path=llama_exe_path,
                        hw_profile=self.hw_profile,
                        draft_tokens=config.draft_tokens,
                        verbose=config.verbose,
                    )
                else:
                    logger.warning(
                        "Eagle mode requested but llama_exe not found — "
                        "falling back to ngram speculation."
                    )
                    self._active_mode = "ngram"
            except Exception as exc:
                logger.warning("EagleDraftEngine init failed (%s) — falling back to ngram.", exc)
                self._active_mode = "ngram"

        # ── Acceptance sampler ─────────────────────────────────────────
        self._sampler = AcceptanceSampler(
            strategy=config.acceptance_strategy,
            acceptance_threshold=config.acceptance_threshold,
        )

        # ── Telemetry ─────────────────────────────────────────────────
        self._telemetry = SpecTelemetryCollector()

        # ── Per-session state ──────────────────────────────────────────
        self._current_prompt: str = ""
        self._generated_tokens: List[str] = []
        self._total_draft_tokens: int = 0
        self._total_accepted_tokens: int = 0
        self._spec_rounds: int = 0
        self._session_start: float = 0.0

        if config.verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(
            "SpeculativeOrchestrator ready: mode=%s  draft_tokens=%d  strategy=%s",
            self._active_mode, config.draft_tokens, config.acceptance_strategy,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def begin_session(self, prompt: str) -> None:
        """
        Initialise a new inference session.

        Must be called once per ``InferenceSession.run()`` invocation before
        any speculation steps.

        Parameters
        ----------
        prompt : str
            The input prompt text for this inference call.
        """
        self._current_prompt = prompt
        self._generated_tokens.clear()
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0
        self._spec_rounds = 0
        self._session_start = time.perf_counter()

        # Reset sub-speculator state
        self._ngram.reset()
        self._prompt_lookup.reset()

        # Register the prompt with the prompt-lookup speculator
        # We tokenise by whitespace as a lightweight approximation
        # (actual token IDs are not available without a tokenizer)
        prompt_pseudo_ids = [hash(w) & 0xFFFF for w in prompt.split()]
        self._prompt_lookup.set_prompt(prompt_pseudo_ids)

    # ------------------------------------------------------------------
    # Core API — called by InferenceSession
    # ------------------------------------------------------------------

    def generate_draft_strings(
        self,
        current_context: str,
    ) -> List[str]:
        """
        Generate speculative draft tokens as strings from the current context.

        This is the **draft phase** (Phase A).  No base model forward pass is
        triggered here.

        Parameters
        ----------
        current_context : str
            Full context text: original prompt + all generated tokens so far.

        Returns
        -------
        list of str
            Draft token strings.  Empty list if speculation is not possible.
        """
        if not self.config.enabled:
            return []

        n = self.config.draft_tokens

        if self._active_mode == "ngram":
            # Convert running text to pseudo-IDs for n-gram matching
            pseudo_ids = [hash(w) & 0xFFFF for w in current_context.split()]
            self._ngram.update(pseudo_ids[-50:])   # only feed recent tokens
            draft_pseudo_ids = self._ngram.generate_drafts(n_drafts=n)
            # Convert back to word strings (approximate)
            words = current_context.split()
            drafts = []
            for did in draft_pseudo_ids:
                # Find the word that produced this hash
                for w in reversed(words):
                    if (hash(w) & 0xFFFF) == did:
                        drafts.append(" " + w)
                        break
            return drafts[:n]

        elif self._active_mode == "prompt_lookup":
            pseudo_ids = [hash(w) & 0xFFFF for w in current_context.split()]
            gen_suffix = pseudo_ids[-min(8, len(pseudo_ids)):]
            self._prompt_lookup.update(gen_suffix)
            draft_pseudo_ids = self._prompt_lookup.generate_drafts(n_drafts=n)
            prompt_words = self._current_prompt.split()
            drafts = []
            for did in draft_pseudo_ids:
                for w in prompt_words:
                    if (hash(w) & 0xFFFF) == did:
                        drafts.append(" " + w)
                        break
            return drafts[:n]

        elif self._active_mode == "eagle" and self._eagle is not None:
            draft_text = self._eagle.generate_drafts(context_text=current_context, n_drafts=n)
            if draft_text:
                # Split at word boundaries to produce discrete draft tokens
                return [" " + w for w in draft_text.split()[:n]]
            return []

        return []

    def verify_and_accept(
        self,
        draft_strings: List[str],
        model_output_text: str,
    ) -> "VerificationResult":
        """
        Verify draft tokens against the base model's output text.

        This is the **verification phase** (Phase B).  The base model has already
        produced ``model_output_text`` in a single forward pass that included the
        draft tokens as context hints.

        Parameters
        ----------
        draft_strings : list of str
            Draft token strings from ``generate_draft_strings()``.
        model_output_text : str
            The actual text generated by the base model.

        Returns
        -------
        VerificationResult
            Acceptance count, bonus token, and updated context.
        """
        self._spec_rounds += 1
        self._total_draft_tokens += len(draft_strings)

        n_accepted, bonus_token = self._sampler.verify_greedy_text(
            draft_token_strings=draft_strings,
            model_output_text=model_output_text,
        )
        self._total_accepted_tokens += n_accepted

        # Build accepted text
        accepted_text = "".join(draft_strings[:n_accepted])
        if bonus_token:
            accepted_text += bonus_token

        if self.config.verbose:
            logger.debug(
                "Verify: %d/%d drafts accepted  bonus=%r",
                n_accepted, len(draft_strings), bonus_token[:20] if bonus_token else "",
            )

        return VerificationResult(
            n_accepted=n_accepted,
            bonus_token=bonus_token,
            accepted_text=accepted_text,
            total_accepted_so_far=self._total_accepted_tokens,
        )

    def end_session(
        self,
        total_tokens_generated: int,
        eval_tps: float,
        baseline_tps: float = 0.0,
        model_name: str = "",
        backend: str = "cpu",
    ) -> SpecDecisionSummary:
        """
        Finalise the session and return a summary for the InferenceResult.

        Parameters
        ----------
        total_tokens_generated : int
            Total tokens produced in this run.
        eval_tps : float
            Observed generation speed (tokens/sec).
        baseline_tps : float
            Expected baseline speed without speculation (from Runtime Learning DB).
        model_name : str
            Model name for telemetry.
        backend : str
            Backend name for telemetry.

        Returns
        -------
        SpecDecisionSummary
            Summary object for attaching to ``InferenceResult``.
        """
        speedup = eval_tps / max(1.0, baseline_tps) if baseline_tps > 0 else 1.0
        acceptance_rate = (
            self._total_accepted_tokens / max(1, self._total_draft_tokens)
        )
        fallback_used = self._spec_rounds == 0

        summary = SpecDecisionSummary(
            enabled=self.config.enabled,
            mode=self._active_mode,
            draft_tokens_per_step=self.config.draft_tokens,
            acceptance_rate=acceptance_rate,
            speedup_ratio=speedup,
            speculative_rounds=self._spec_rounds,
            total_draft_tokens=self._total_draft_tokens,
            total_accepted_tokens=self._total_accepted_tokens,
            fallback_used=fallback_used,
        )

        # Record telemetry
        if self.config.record_telemetry and self._spec_rounds > 0:
            record = SpecRunRecord(
                mode=self._active_mode,
                draft_tokens_requested=self.config.draft_tokens,
                total_tokens_generated=total_tokens_generated,
                total_draft_tokens=self._total_draft_tokens,
                total_accepted_tokens=self._total_accepted_tokens,
                speculative_rounds=self._spec_rounds,
                eval_tps_speculative=eval_tps,
                eval_tps_baseline=baseline_tps,
                speedup_ratio=speedup,
                model_name=model_name,
                backend=backend,
            )
            self._telemetry.record(record)
            self._persist_telemetry(record)

        if self.config.verbose:
            print(summary.format_cli_output())

        return summary

    # ------------------------------------------------------------------
    # Telemetry accessors
    # ------------------------------------------------------------------

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all sessions in this orchestrator instance."""
        return {
            "orchestrator_mode": self._active_mode,
            "ngram": self._ngram.get_stats(),
            "prompt_lookup": self._prompt_lookup.get_stats(),
            "eagle": self._eagle.get_stats() if self._eagle else None,
            "sampler": self._sampler.get_stats(),
            "session_aggregate": self._telemetry.get_aggregate(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_telemetry(self, record: SpecRunRecord) -> None:
        """
        Write speculative decoding telemetry to the Runtime Learning SQLite DB.

        Uses the ``speculative_stats`` table (created via schema migration in
        ``runtime_learning/database.py``).  Fails silently on DB errors to
        ensure inference is never blocked by telemetry writes.
        """
        try:
            from runtime_learning.database import LearningDatabase
            db = LearningDatabase()
            db.save_spec_record(record.to_dict())
        except Exception as exc:
            logger.debug("Spec telemetry persist skipped: %s", exc)


# ---------------------------------------------------------------------------
# VerificationResult — returned by verify_and_accept()
# ---------------------------------------------------------------------------

class VerificationResult:
    """Outcome of one speculative verification step."""

    __slots__ = ("n_accepted", "bonus_token", "accepted_text", "total_accepted_so_far")

    def __init__(
        self,
        n_accepted: int,
        bonus_token: str,
        accepted_text: str,
        total_accepted_so_far: int,
    ) -> None:
        self.n_accepted = n_accepted
        self.bonus_token = bonus_token
        self.accepted_text = accepted_text
        self.total_accepted_so_far = total_accepted_so_far
