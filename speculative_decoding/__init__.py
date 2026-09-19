"""
speculative_decoding/__init__.py
---------------------------------
Phase 5 Speculative Decoding Suite for InferenceOS.

Public exports
--------------
SpeculativeDecoderConfig
    Configuration dataclass (maps to RuntimeConfig.spec_* fields).

SpeculativeOrchestrator
    Central controller that drives draft generation and verification.

NGramSpeculator
    N-gram suffix matching from the model's own generated context.

PromptLookupSpeculator
    Suffix matching against the original input prompt.

EagleDraftEngine
    External GGUF draft model (Eagle-2 / small GGUF) runner.

AcceptanceSampler
    Greedy and speculative sampling acceptance logic.

SpecDecisionSummary
    Per-run summary attached to InferenceResult.spec_decision.

SpecTelemetryCollector
    Aggregate telemetry collector for multi-run statistics.
"""
from .acceptance_sampler import AcceptanceSampler
from .config import SpeculativeDecoderConfig
from .eagle_draft_engine import EagleDraftEngine
from .ngram_speculator import NGramSpeculator
from .orchestrator import SpeculativeOrchestrator, VerificationResult
from .prompt_lookup_speculator import PromptLookupSpeculator
from .telemetry import SpecDecisionSummary, SpecRunRecord, SpecTelemetryCollector

__all__ = [
    "SpeculativeDecoderConfig",
    "SpeculativeOrchestrator",
    "VerificationResult",
    "NGramSpeculator",
    "PromptLookupSpeculator",
    "EagleDraftEngine",
    "AcceptanceSampler",
    "SpecDecisionSummary",
    "SpecRunRecord",
    "SpecTelemetryCollector",
]
