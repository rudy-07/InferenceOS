"""
telemetry.py
------------
Telemetry collection and reporting for Phase 5 Speculative Decoding Suite.

Captures per-run and aggregate statistics from the SpeculativeOrchestrator,
formats them for CLI display, and serialises them for the Runtime Learning DB.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpecRunRecord:
    """
    Statistics from a single inference run that used speculative decoding.

    Attributes
    ----------
    mode : str
        Speculation strategy used (``"ngram"``, ``"prompt_lookup"``, ``"eagle"``).
    draft_tokens_requested : int
        Draft tokens requested per speculation step.
    total_tokens_generated : int
        Total tokens produced by the run (including accepted drafts).
    total_draft_tokens : int
        Total draft tokens evaluated (accepted + rejected).
    total_accepted_tokens : int
        Draft tokens accepted by the base model.
    speculative_rounds : int
        Number of speculation cycles executed during this run.
    eval_tps_speculative : float
        Effective token generation speed with speculation (tokens/sec).
    eval_tps_baseline : float
        Baseline generation speed without speculation (tokens/sec, from DB).
    speedup_ratio : float
        ``eval_tps_speculative / max(1, eval_tps_baseline)``.
    model_name : str
        Model file name.
    backend : str
        Inference backend.
    timestamp : float
        Unix timestamp of the run.
    """
    mode: str = "ngram"
    draft_tokens_requested: int = 5
    total_tokens_generated: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    speculative_rounds: int = 0
    eval_tps_speculative: float = 0.0
    eval_tps_baseline: float = 0.0
    speedup_ratio: float = 1.0
    model_name: str = ""
    backend: str = "cpu"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "draft_tokens_requested": self.draft_tokens_requested,
            "total_tokens_generated": self.total_tokens_generated,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "acceptance_rate": self.acceptance_rate,
            "speculative_rounds": self.speculative_rounds,
            "eval_tps_speculative": round(self.eval_tps_speculative, 2),
            "eval_tps_baseline": round(self.eval_tps_baseline, 2),
            "speedup_ratio": round(self.speedup_ratio, 3),
            "model_name": self.model_name,
            "backend": self.backend,
            "timestamp": self.timestamp,
        }

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted_tokens / max(1, self.total_draft_tokens)


@dataclass
class SpecDecisionSummary:
    """
    Summary of speculative decoding participation for an InferenceResult.

    Attached to ``InferenceResult.spec_decision`` so the CLI and telemetry
    pipeline can display and store speculation statistics.
    """
    enabled: bool = False
    mode: str = "disabled"
    draft_tokens_per_step: int = 0
    acceptance_rate: float = 0.0
    speedup_ratio: float = 1.0
    speculative_rounds: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "draft_tokens_per_step": self.draft_tokens_per_step,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "speedup_ratio": round(self.speedup_ratio, 3),
            "speculative_rounds": self.speculative_rounds,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "fallback_used": self.fallback_used,
        }

    def format_cli_output(self) -> str:
        """Format a Rich-compatible summary string for CLI display."""
        if not self.enabled:
            return "[dim]Speculative Decoding: disabled[/dim]"
        status = "[green]✓[/green]" if self.acceptance_rate >= 0.5 else "[yellow]~[/yellow]"
        return (
            f"[bold cyan]Speculative Decoding[/bold cyan]  {status}\n"
            f"  Mode:            [green]{self.mode}[/green]\n"
            f"  Draft tokens:    {self.draft_tokens_per_step} / step\n"
            f"  Acceptance rate: [bold]{self.acceptance_rate * 100:.1f}%[/bold]\n"
            f"  Speedup ratio:   [bold yellow]{self.speedup_ratio:.2f}×[/bold yellow]\n"
            f"  Spec rounds:     {self.speculative_rounds}\n"
            f"  Drafts evaluated:{self.total_draft_tokens}  "
            f"accepted: {self.total_accepted_tokens}"
        )


class SpecTelemetryCollector:
    """
    Accumulates per-run speculative decoding statistics for reporting and
    persistence into the Runtime Learning database.
    """

    def __init__(self) -> None:
        self._records: List[SpecRunRecord] = []

    def record(self, record: SpecRunRecord) -> None:
        """Append a completed run record."""
        self._records.append(record)

    def get_aggregate(self) -> Dict[str, Any]:
        """Return aggregate statistics across all recorded runs."""
        if not self._records:
            return {}
        total_runs = len(self._records)
        avg_accept = sum(r.acceptance_rate for r in self._records) / total_runs
        avg_speedup = sum(r.speedup_ratio for r in self._records) / total_runs
        return {
            "total_runs": total_runs,
            "avg_acceptance_rate": round(avg_accept, 4),
            "avg_speedup_ratio": round(avg_speedup, 3),
            "last_mode": self._records[-1].mode if self._records else "none",
        }

    def clear(self) -> None:
        self._records.clear()
