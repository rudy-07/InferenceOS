"""
memory_budget.py
----------------
Primary output dataclass for Phase 6 Runtime Memory Optimization.

A MemoryBudget is the complete pre-flight memory analysis result, containing
all estimated footprints, KV cache projections, OOM risk classification,
warnings, and whether context was auto-resized.

Callers use:
    budget = guard.check(model, plan, context_length=8192)
    print(budget.report())
    if budget.risk_level == "CRITICAL":
        raise InferenceBlockedError(budget)

The ``report()`` method produces the human-readable output format shown
in the Phase 6 requirements.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryBudget:
    """
    Complete pre-flight memory analysis result.

    Attributes
    ----------
    model_name : str
        Model name annotated from ModelDescriptor.
    context_length : int
        Requested context length (before any auto-resize).
    effective_context : int
        Context length that will actually be used. May be reduced from
        ``context_length`` if ``auto_resized=True``.

    KV Cache
    --------
    kv_bytes_per_token : int
        Total KV bytes added per new token (all GPU layers combined).
    kv_at_quarter_context : int
        KV cache size at 25% fill.
    kv_at_half_context : int
        KV cache size at 50% fill.
    kv_at_full_context : int
        KV cache size at 100% fill (worst-case, used for OOM risk).
    kv_growth_rate_gb_per_1k : float
        GB of KV cache added per 1000 tokens.

    Weights
    -------
    weights_vram_bytes : int
        GPU-placed model weight bytes.
    weights_ram_bytes : int
        CPU-placed model weight bytes.

    Buffers
    -------
    activation_workspace_bytes : int
        Rolling activation buffer estimate.
    backend_overhead_bytes : int
        GPU backend / driver context overhead.
    kv_reservation_bytes : int
        Reserved VRAM block for KV cache.

    Totals
    ------
    peak_vram_bytes : int
        ``weights_vram + kv_reservation + activation_workspace + backend_overhead``
    peak_ram_bytes : int
        ``weights_ram + OS_overhead_estimate``
    peak_total_bytes : int
        ``peak_vram + peak_ram``

    Risk
    ----
    risk_level : str
        One of ``"LOW"`` | ``"MEDIUM"`` | ``"HIGH"`` | ``"CRITICAL"``.
    oom_probability : float
        Modelled OOM probability (0.0–1.0).
    kv_fill_ratio : float
        ``kv_at_full_context / vram_headroom_after_weights``.

    Recommendations
    ---------------
    warnings : List[str]
        Human-readable warning messages.
    safe_context_length : int
        Maximum context that fits safely within hardware limits.
    auto_resized : bool
        True if context was reduced automatically (HIGH risk + auto_resize=True).
    """
    # Identity
    model_name: str
    context_length: int
    effective_context: int

    # KV cache estimates
    kv_bytes_per_token: int
    kv_at_quarter_context: int
    kv_at_half_context: int
    kv_at_full_context: int
    kv_growth_rate_gb_per_1k: float

    # Weight footprints
    weights_vram_bytes: int
    weights_ram_bytes: int

    # Buffer estimates
    activation_workspace_bytes: int
    backend_overhead_bytes: int
    kv_reservation_bytes: int

    # Peak totals
    peak_vram_bytes: int
    peak_ram_bytes: int
    peak_total_bytes: int

    # Risk
    risk_level: str
    oom_probability: float
    kv_fill_ratio: float

    # Recommendations
    warnings: List[str] = field(default_factory=list)
    safe_context_length: int = 0
    auto_resized: bool = False

    # Backend (for display)
    backend: str = "unknown"

    # ---------------------------------------------------------------------------
    # report(): the human-readable summary matching Phase 6 requirements
    # ---------------------------------------------------------------------------

    def report(self) -> str:
        """
        Generate a formatted multi-line memory budget report.

        Output format::

            ╔════════════════════════════════════════════════╗
            ║   InferenceOS  ·  Phase 6 Memory Budget        ║
            ╚════════════════════════════════════════════════╝

              Model:              TestModel
              Estimated context:  8192 tokens
              ...
        """
        def _gb(b: int) -> str:
            return f"{b / (1024 ** 3):.2f} GB"

        def _mb(b: int) -> str:
            return f"{b / (1024 * 1024):.0f} MB"

        def _fmt_bytes(b: int) -> str:
            if b >= 1024 ** 3:
                return _gb(b)
            return _mb(b)

        risk_emoji = {
            "LOW":      "✅ LOW",
            "MEDIUM":   "⚠️  MEDIUM",
            "HIGH":     "🔴 HIGH",
            "CRITICAL": "💀 CRITICAL",
        }.get(self.risk_level, self.risk_level)

        ctx_note = ""
        if self.auto_resized:
            ctx_note = f"  (reduced from {self.context_length} — auto-resize)"

        lines = [
            "╔════════════════════════════════════════════════════╗",
            "║      InferenceOS  ·  Phase 6 Memory Budget         ║",
            "╚════════════════════════════════════════════════════╝",
            "",
            f"  Model:                {self.model_name}",
            f"  Estimated context:    {self.effective_context:,} tokens{ctx_note}",
            f"  KV bytes/token:       {self.kv_bytes_per_token:,} B  "
            f"({self.kv_bytes_per_token / (1024*1024):.2f} MB)",
            f"  Expected KV usage:    {_gb(self.kv_at_full_context)}"
            f"   (at full context)",
            f"  KV growth rate:       {self.kv_growth_rate_gb_per_1k:.3f} GB"
            f" per 1K tokens",
            "",
            "MEMORY FOOTPRINT",
            "─" * 52,
            f"  Weights (VRAM):       {_fmt_bytes(self.weights_vram_bytes)}",
            f"  Weights (RAM):        {_fmt_bytes(self.weights_ram_bytes)}",
            f"  KV reservation:       {_fmt_bytes(self.kv_reservation_bytes)}",
            f"  Activation workspace: {_fmt_bytes(self.activation_workspace_bytes)}",
            f"  Backend overhead:     {_fmt_bytes(self.backend_overhead_bytes)}",
            f"  Peak VRAM:            {_fmt_bytes(self.peak_vram_bytes)}",
            f"  Peak RAM:             {_fmt_bytes(self.peak_ram_bytes)}",
            f"  Peak total:           {_fmt_bytes(self.peak_total_bytes)}",
            "",
            "RISK ASSESSMENT",
            "─" * 52,
            f"  Risk level:           {risk_emoji}",
            f"  OOM probability:      {self.oom_probability * 100:.1f}%",
            f"  KV fill ratio:        {self.kv_fill_ratio:.3f}",
            f"  Safe context limit:   {self.safe_context_length:,} tokens",
            "",
            "WARNINGS",
            "─" * 52,
        ]

        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        else:
            lines.append("  (none)")

        lines.append("")
        return "\n".join(lines)

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of all budget values."""
        def _gb(b: int) -> float:
            return round(b / (1024 ** 3), 4)

        return {
            "model_name": self.model_name,
            "context": {
                "requested": self.context_length,
                "effective": self.effective_context,
                "auto_resized": self.auto_resized,
                "safe_limit": self.safe_context_length,
            },
            "kv_cache": {
                "bytes_per_token": self.kv_bytes_per_token,
                "at_quarter_context_gb": _gb(self.kv_at_quarter_context),
                "at_half_context_gb": _gb(self.kv_at_half_context),
                "at_full_context_gb": _gb(self.kv_at_full_context),
                "growth_rate_gb_per_1k": round(self.kv_growth_rate_gb_per_1k, 4),
            },
            "weights": {
                "vram_gb": _gb(self.weights_vram_bytes),
                "ram_gb": _gb(self.weights_ram_bytes),
            },
            "buffers": {
                "activation_workspace_mb": round(self.activation_workspace_bytes / (1024*1024), 2),
                "backend_overhead_mb": round(self.backend_overhead_bytes / (1024*1024), 2),
                "kv_reservation_gb": _gb(self.kv_reservation_bytes),
            },
            "peak": {
                "vram_gb": _gb(self.peak_vram_bytes),
                "ram_gb": _gb(self.peak_ram_bytes),
                "total_gb": _gb(self.peak_total_bytes),
            },
            "risk": {
                "level": self.risk_level,
                "oom_probability_pct": round(self.oom_probability * 100, 2),
                "kv_fill_ratio": round(self.kv_fill_ratio, 4),
            },
            "warnings": self.warnings,
            "backend": self.backend,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
