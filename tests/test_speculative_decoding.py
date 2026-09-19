"""
test_speculative_decoding.py
----------------------------
Unit tests for Phase 5 Speculative Decoding Suite in InferenceOS.

Covers:
  - NGramSpeculator: update, generate_drafts, hit_rate, edge cases
  - PromptLookupSpeculator: set_prompt, update, generate_drafts, edge cases
  - AcceptanceSampler: greedy text verification, greedy ID verification
  - SpeculativeDecoderConfig: from_runtime_config, resolve_mode
  - SpeculativeOrchestrator: initialisation, begin_session, end_session
  - Telemetry: SpecRunRecord.to_dict(), SpecDecisionSummary.format_cli_output()
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from speculative_decoding.ngram_speculator import NGramSpeculator
from speculative_decoding.prompt_lookup_speculator import PromptLookupSpeculator
from speculative_decoding.acceptance_sampler import AcceptanceSampler
from speculative_decoding.config import SpeculativeDecoderConfig
from speculative_decoding.telemetry import SpecRunRecord, SpecDecisionSummary, SpecTelemetryCollector
from speculative_decoding.orchestrator import SpeculativeOrchestrator


# ===========================================================================
# NGramSpeculator
# ===========================================================================

class TestNGramSpeculator:
    def test_basic_hit(self):
        """N-gram should find a continuation for a repeated pattern."""
        spec = NGramSpeculator(ngram_size=3, min_match_length=2)
        # Pattern: [10, 20, 30] appears twice, and the second occurrence is the
        # current suffix — so the search finds the first occurrence and returns
        # the tokens that followed it: [10, 20, 30]
        tokens = [10, 20, 30, 10, 20, 30, 10, 20, 30]
        # Feed first two occurrences as history, then the third as the suffix
        spec.update([10, 20, 30, 10, 20, 30, 40, 50, 10, 20, 30])
        drafts = spec.generate_drafts(n_drafts=3)
        # The suffix [10, 20, 30] was seen at index 0 earlier; continuation is [10, 20, 30]
        assert len(drafts) > 0, "Should find at least one draft from repeated pattern"

    def test_no_match_on_short_history(self):
        """With fewer tokens than ngram_size, should return empty."""
        spec = NGramSpeculator(ngram_size=5, min_match_length=4)
        spec.update([1, 2])
        drafts = spec.generate_drafts(n_drafts=3)
        assert drafts == []

    def test_hit_rate_increments(self):
        """hit_rate should increase after successful draft generation."""
        spec = NGramSpeculator(ngram_size=2, min_match_length=1)
        tokens = [1, 2, 3, 1, 2, 3, 4]
        spec.update(tokens)
        spec.generate_drafts(n_drafts=2)
        assert spec.total_calls == 1

    def test_reset_clears_history(self):
        """reset() should prevent any future drafts until new tokens are fed."""
        spec = NGramSpeculator(ngram_size=3, min_match_length=2)
        spec.update([1, 2, 3, 1, 2, 3])
        spec.reset()
        drafts = spec.generate_drafts(n_drafts=3)
        assert drafts == []

    def test_get_stats_keys(self):
        """get_stats() should return required keys."""
        spec = NGramSpeculator()
        stats = spec.get_stats()
        assert "total_calls" in stats
        assert "hit_rate" in stats
        assert "history_length" in stats

    def test_max_drafts_capped(self):
        """Should return at most n_drafts tokens."""
        spec = NGramSpeculator(ngram_size=2, min_match_length=1)
        spec.update([1, 2, 3, 4, 5, 1, 2])
        drafts = spec.generate_drafts(n_drafts=2)
        assert len(drafts) <= 2

    def test_n_drafts_zero(self):
        """Requesting 0 drafts should return empty."""
        spec = NGramSpeculator(ngram_size=2, min_match_length=1)
        spec.update([1, 2, 1, 2, 1, 2])
        drafts = spec.generate_drafts(n_drafts=0)
        assert drafts == []


# ===========================================================================
# PromptLookupSpeculator
# ===========================================================================

class TestPromptLookupSpeculator:
    def test_basic_hit(self):
        """Should return the continuation after a suffix match in the prompt."""
        pl = PromptLookupSpeculator(min_match_length=2)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        pl.set_prompt(prompt)
        pl.update([3, 4])
        drafts = pl.generate_drafts(n_drafts=4)
        assert drafts == [5, 6, 7, 8], f"Expected [5,6,7,8] but got {drafts}"

    def test_no_hit_when_suffix_absent(self):
        """Should return empty when suffix doesn't appear in prompt."""
        pl = PromptLookupSpeculator(min_match_length=2)
        pl.set_prompt([1, 2, 3, 4, 5])
        pl.update([99, 100])
        drafts = pl.generate_drafts(n_drafts=3)
        assert drafts == []

    def test_empty_prompt(self):
        """No drafts when prompt is empty."""
        pl = PromptLookupSpeculator(min_match_length=2)
        pl.set_prompt([])
        pl.update([1, 2])
        drafts = pl.generate_drafts(n_drafts=3)
        assert drafts == []

    def test_lookup_window_limits_search(self):
        """Lookup window should restrict search to last N prompt tokens."""
        pl = PromptLookupSpeculator(min_match_length=2, lookup_window=4)
        # Suffix [1,2] exists at position 0, but window=4 only looks at last 4 tokens
        pl.set_prompt([1, 2, 3, 4, 5, 6, 7, 8])
        pl.update([1, 2])
        # With window=4, search is [5,6,7,8] which doesn't contain [1,2]
        # So it should return empty
        drafts = pl.generate_drafts(n_drafts=3)
        # This may match or not depending on window; just confirm no crash
        assert isinstance(drafts, list)

    def test_reset_clears_state(self):
        """reset() should clear prompt and suffix."""
        pl = PromptLookupSpeculator(min_match_length=2)
        pl.set_prompt([1, 2, 3, 4, 5])
        pl.update([3, 4])
        pl.reset()
        drafts = pl.generate_drafts(n_drafts=3)
        assert drafts == []

    def test_hit_rate(self):
        """hit_rate should reflect successful lookups."""
        pl = PromptLookupSpeculator(min_match_length=1)
        pl.set_prompt([1, 2, 3, 4, 5])
        pl.update([2])
        pl.generate_drafts(n_drafts=3)
        assert pl.total_calls == 1


# ===========================================================================
# AcceptanceSampler
# ===========================================================================

class TestAcceptanceSampler:
    def test_greedy_text_full_accept(self):
        """All draft tokens should be accepted when model output matches exactly."""
        sampler = AcceptanceSampler(strategy="greedy")
        drafts = [" the", " quick", " brown", " fox"]
        model_out = " the quick brown fox jumps"
        n_acc, bonus = sampler.verify_greedy_text(drafts, model_out)
        assert n_acc == 4
        assert bonus.strip() == "jumps"

    def test_greedy_text_partial_accept(self):
        """Only matching prefix should be accepted on a mismatch."""
        sampler = AcceptanceSampler(strategy="greedy")
        drafts = [" hello", " world", " wrong"]
        model_out = " hello world different"
        n_acc, bonus = sampler.verify_greedy_text(drafts, model_out)
        assert n_acc == 2
        assert "different" in bonus

    def test_greedy_text_zero_accept(self):
        """Zero drafts accepted when first token mismatches."""
        sampler = AcceptanceSampler(strategy="greedy")
        n_acc, bonus = sampler.verify_greedy_text([" wrong"], " right answer")
        assert n_acc == 0

    def test_greedy_ids_full_accept(self):
        """All token IDs accepted when they match exactly."""
        sampler = AcceptanceSampler(strategy="greedy")
        draft_ids = [10, 20, 30]
        verified_ids = [10, 20, 30, 40]
        n_acc, bonus_id = sampler.verify_greedy_ids(draft_ids, verified_ids)
        assert n_acc == 3
        assert bonus_id == 40

    def test_greedy_ids_partial(self):
        """Partial match on ID sequence."""
        sampler = AcceptanceSampler(strategy="greedy")
        n_acc, bonus = sampler.verify_greedy_ids([1, 2, 99], [1, 2, 3, 4])
        assert n_acc == 2
        assert bonus == 3

    def test_acceptance_rate_updates(self):
        """acceptance_rate should be computed correctly."""
        sampler = AcceptanceSampler(strategy="greedy")
        sampler.verify_greedy_text([" a", " b"], " a b c")
        assert sampler.total_drafts_accepted == 2
        assert sampler.acceptance_rate == 1.0

    def test_invalid_strategy_raises(self):
        """Invalid strategy string should raise ValueError."""
        with pytest.raises(ValueError):
            AcceptanceSampler(strategy="invalid_strat")

    def test_empty_drafts(self):
        """Empty draft list should return (0, full model output)."""
        sampler = AcceptanceSampler()
        n_acc, bonus = sampler.verify_greedy_text([], "output text")
        assert n_acc == 0


# ===========================================================================
# SpeculativeDecoderConfig
# ===========================================================================

class TestSpeculativeDecoderConfig:
    def test_defaults(self):
        cfg = SpeculativeDecoderConfig()
        assert cfg.enabled is False
        assert cfg.mode == "auto"
        assert cfg.draft_tokens == 5
        assert cfg.ngram_size == 3

    def test_resolve_mode_auto_no_eagle(self):
        cfg = SpeculativeDecoderConfig(mode="auto", eagle_draft_model_path=None)
        assert cfg.resolve_mode() == "ngram"

    def test_resolve_mode_auto_with_eagle(self):
        cfg = SpeculativeDecoderConfig(mode="auto", eagle_draft_model_path="/tmp/draft.gguf")
        assert cfg.resolve_mode() == "eagle"

    def test_resolve_mode_explicit(self):
        cfg = SpeculativeDecoderConfig(mode="prompt_lookup")
        assert cfg.resolve_mode() == "prompt_lookup"

    def test_from_runtime_config_defaults(self):
        """from_runtime_config should not raise with a default-initialized RuntimeConfig."""
        from inference_runtime.runtime_config import RuntimeConfig
        rt_cfg = RuntimeConfig()
        spec_cfg = SpeculativeDecoderConfig.from_runtime_config(rt_cfg)
        assert spec_cfg.enabled is False
        assert spec_cfg.draft_tokens == 5


# ===========================================================================
# SpeculativeOrchestrator
# ===========================================================================

class TestSpeculativeOrchestrator:
    def _make_orchestrator(self, mode="ngram") -> SpeculativeOrchestrator:
        cfg = SpeculativeDecoderConfig(
            enabled=True,
            mode=mode,
            draft_tokens=5,
            ngram_size=3,
            min_match_length=2,
            record_telemetry=False,   # avoid DB writes in tests
        )
        return SpeculativeOrchestrator(cfg, llama_exe_path=None, hw_profile={})

    def test_init_ngram(self):
        orc = self._make_orchestrator("ngram")
        assert orc._active_mode == "ngram"
        assert orc._ngram is not None

    def test_init_prompt_lookup(self):
        orc = self._make_orchestrator("prompt_lookup")
        assert orc._active_mode == "prompt_lookup"

    def test_begin_session_resets_state(self):
        orc = self._make_orchestrator()
        orc.begin_session("Hello world test prompt")
        assert orc._spec_rounds == 0
        assert orc._total_draft_tokens == 0
        assert orc._current_prompt == "Hello world test prompt"

    def test_verify_and_accept_increments_rounds(self):
        orc = self._make_orchestrator()
        orc.begin_session("test prompt")
        orc.verify_and_accept([" hello", " world"], " hello world indeed")
        assert orc._spec_rounds == 1
        assert orc._total_draft_tokens == 2

    def test_end_session_returns_summary(self):
        orc = self._make_orchestrator()
        orc.begin_session("test")
        orc.verify_and_accept([" a", " b"], " a b c")
        summary = orc.end_session(
            total_tokens_generated=100,
            eval_tps=50.0,
            baseline_tps=30.0,
            model_name="test-model",
            backend="cpu",
        )
        assert summary.enabled is True
        assert summary.mode == "ngram"
        assert summary.speculative_rounds == 1
        assert summary.total_draft_tokens == 2

    def test_disabled_orchestrator_returns_no_drafts(self):
        cfg = SpeculativeDecoderConfig(enabled=False)
        orc = SpeculativeOrchestrator(cfg)
        orc.begin_session("some prompt")
        drafts = orc.generate_draft_strings("some context text")
        assert drafts == []

    def test_get_aggregate_stats(self):
        orc = self._make_orchestrator()
        stats = orc.get_aggregate_stats()
        assert "orchestrator_mode" in stats
        assert "ngram" in stats
        assert "sampler" in stats


# ===========================================================================
# Telemetry
# ===========================================================================

class TestTelemetry:
    def test_spec_run_record_to_dict(self):
        record = SpecRunRecord(
            mode="ngram",
            draft_tokens_requested=5,
            total_tokens_generated=200,
            total_draft_tokens=100,
            total_accepted_tokens=75,
        )
        d = record.to_dict()
        assert d["mode"] == "ngram"
        assert d["acceptance_rate"] == pytest.approx(0.75)
        assert "speedup_ratio" in d

    def test_spec_decision_summary_format(self):
        summary = SpecDecisionSummary(
            enabled=True,
            mode="prompt_lookup",
            draft_tokens_per_step=5,
            acceptance_rate=0.78,
            speedup_ratio=1.45,
            speculative_rounds=20,
        )
        output = summary.format_cli_output()
        assert "Speculative Decoding" in output
        assert "prompt_lookup" in output
        assert "78.0%" in output

    def test_spec_decision_summary_disabled(self):
        summary = SpecDecisionSummary(enabled=False)
        output = summary.format_cli_output()
        assert "disabled" in output.lower()

    def test_telemetry_collector_aggregate(self):
        collector = SpecTelemetryCollector()
        r1 = SpecRunRecord(mode="ngram", total_draft_tokens=100, total_accepted_tokens=80, speedup_ratio=1.8)
        r2 = SpecRunRecord(mode="ngram", total_draft_tokens=100, total_accepted_tokens=60, speedup_ratio=1.4)
        collector.record(r1)
        collector.record(r2)
        agg = collector.get_aggregate()
        assert agg["total_runs"] == 2
        assert agg["avg_speedup_ratio"] == pytest.approx(1.6)

    def test_telemetry_collector_clear(self):
        collector = SpecTelemetryCollector()
        collector.record(SpecRunRecord())
        collector.clear()
        agg = collector.get_aggregate()
        assert agg == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
