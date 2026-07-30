"""
preflight_guard.py
------------------
Pre-flight memory gate for Phase 6 Runtime Memory Optimization.

The PreflightGuard is the primary entry point for Phase 6. It orchestrates
the full memory budget analysis and enforces a go/no-go decision before
any llama.cpp subprocess is launched.

Workflow
--------
  1. Compute GPU/CPU weight distribution from PlacementPlan
  2. Estimate KV cache at the requested context (KvCacheEstimator)
  3. Size all ancillary buffers (BufferPlanner)
  4. Classify OOM risk (RiskClassifier)
  5. If HIGH risk + auto_resize_buffers → reduce context, repeat steps 2–4
  6. Generate warnings (WarningGenerator)
  7. Assemble and return MemoryBudget
  8. If risk is CRITICAL + block_on_critical → raise InferenceBlockedError

Auto-resize behaviour
---------------------
When ``auto_resize_buffers=True`` and risk is HIGH:
  - KvCacheEstimator.max_safe_context() computes the largest context that
    fits in available VRAM headroom
  - A 10% safety buffer is applied: effective_ctx = safe * 0.90
  - The analysis is re-run at the reduced context
  - MemoryBudget.auto_resized = True and a warning is appended

The minimum context after resize is 512 tokens (llama.cpp alignment minimum).

InferenceBlockedError
---------------------
Raised by PreflightGuard when risk = CRITICAL and block_on_critical = True.
The exception carries the full MemoryBudget so callers can inspect what
would have happened:

    try:
        budget = guard.check(model, plan, context_length=16384)
    except InferenceBlockedError as e:
        print(e.budget.report())
        # Decide: reduce context, swap model, or abort
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from layer_placement.model_descriptor import ModelDescriptor
from layer_placement.placement_plan import PlacementPlan, PlacementDevice
from layer_placement.model_descriptor import LAYER_TYPE_TRANSFORMER

from .buffer_planner import BufferPlanner
from .kv_estimator import KvCacheEstimator
from .memory_budget import MemoryBudget
from .risk_classifier import OomRisk, RiskClassifier
from .warning_generator import WarningGenerator


# ---------------------------------------------------------------------------
# InferenceBlockedError
# ---------------------------------------------------------------------------

class InferenceBlockedError(RuntimeError):
    """
    Raised by :class:`PreflightGuard` when OOM risk is CRITICAL and
    inference would certainly fail.

    Attributes
    ----------
    budget : MemoryBudget
        The full pre-flight analysis result, including risk level,
        OOM probability, and all memory estimates.
    """

    def __init__(self, budget: MemoryBudget, message: str = "") -> None:
        self.budget = budget
        if not message:
            message = (
                f"Inference blocked: OOM risk is CRITICAL "
                f"(kv_fill_ratio={budget.kv_fill_ratio:.3f}, "
                f"oom_probability={budget.oom_probability * 100:.1f}%). "
                f"Peak VRAM estimate: {budget.peak_vram_bytes / (1024**3):.2f} GB. "
                f"Safe context limit: {budget.safe_context_length:,} tokens."
            )
        super().__init__(message)


# ---------------------------------------------------------------------------
# OS RAM overhead constant
# ---------------------------------------------------------------------------

_OS_RAM_OVERHEAD_BYTES = 1 * 1024 ** 3   # 1 GB reserved for OS


# ---------------------------------------------------------------------------
# PreflightGuard
# ---------------------------------------------------------------------------

class PreflightGuard:
    """
    Runs the full memory budget analysis and enforces the inference go/no-go gate.

    Parameters
    ----------
    hw_profile : dict
        Hardware profile from Phase 1 profiler.
    auto_resize_buffers : bool
        If True, automatically reduce context_length when risk is HIGH.
        The safe context is computed as ``max_safe_context × 0.90``.
        Default True.
    block_on_critical : bool
        If True, raise :class:`InferenceBlockedError` when risk = CRITICAL.
        If False, the CRITICAL budget is returned without raising.
        Default True.
    dtype_bytes : int
        Bytes per KV/activation element. Default 2 (fp16).
    backend : str
        Backend name for buffer overhead lookup. Default ``"unknown"``.
    """

    # Auto-resize applies a 10% safety buffer below the computed safe context
    _AUTO_RESIZE_SAFETY = 0.90
    _MIN_CONTEXT = 512

    def __init__(
        self,
        hw_profile: Dict[str, Any],
        auto_resize_buffers: bool = True,
        block_on_critical: bool = True,
        dtype_bytes: int = 2,
        backend: str = "unknown",
    ) -> None:
        self.hw_profile = hw_profile
        self.auto_resize_buffers = auto_resize_buffers
        self.block_on_critical = block_on_critical
        self.backend = backend

        self._kv_estimator = KvCacheEstimator(dtype_bytes=dtype_bytes)
        self._buffer_planner = BufferPlanner(dtype_bytes=dtype_bytes)
        self._classifier = RiskClassifier()
        self._warner = WarningGenerator()

    # ---------------------------------------------------------------------------
    # Primary API
    # ---------------------------------------------------------------------------

    def check(
        self,
        model: ModelDescriptor,
        plan: PlacementPlan,
        context_length: int,
        batch_size: int = 512,
    ) -> MemoryBudget:
        """
        Run the full pre-flight memory analysis and enforce the go/no-go gate.

        Parameters
        ----------
        model : ModelDescriptor
            Full model descriptor (architecture, KV head counts, hidden size).
        plan : PlacementPlan
            Phase 3 placement plan (determines GPU/CPU weight distribution).
        context_length : int
            Requested context window in tokens.
        batch_size : int
            Prompt processing batch size. Default 512.

        Returns
        -------
        MemoryBudget
            Complete pre-flight analysis. If ``auto_resize_buffers=True`` and
            risk was HIGH, ``budget.auto_resized`` will be True and
            ``budget.effective_context < context_length``.

        Raises
        ------
        InferenceBlockedError
            If risk is CRITICAL and ``block_on_critical=True``.
        """
        context_length = max(self._MIN_CONTEXT, context_length)
        budget = self._analyse(model, plan, context_length, batch_size)

        risk = OomRisk(budget.risk_level)

        # Auto-resize on HIGH risk
        if risk == OomRisk.HIGH and self.auto_resize_buffers:
            budget = self._auto_resize(model, plan, batch_size, budget)
            risk = OomRisk(budget.risk_level)

        # Block on CRITICAL risk
        if risk == OomRisk.CRITICAL and self.block_on_critical:
            raise InferenceBlockedError(budget)

        return budget

    check_plan = check  # snake_case alias

    # ---------------------------------------------------------------------------
    # Internal analysis
    # ---------------------------------------------------------------------------

    def _analyse(
        self,
        model: ModelDescriptor,
        plan: PlacementPlan,
        context_length: int,
        batch_size: int,
    ) -> MemoryBudget:
        """Run one full analysis cycle at the given context length."""
        vram_total = self._get_vram_total()
        ram_total  = self._get_ram_total()

        # Weight distribution from plan
        weights_vram = sum(
            lp.size_bytes for lp in plan.layer_placements
            if lp.device == PlacementDevice.GPU
        )
        weights_ram = sum(
            lp.size_bytes for lp in plan.layer_placements
            if lp.device == PlacementDevice.CPU
        )

        # KV cache estimate
        kv_est = self._kv_estimator.estimate(model, plan, context_length)

        # Buffer plan (activation + backend overhead + KV reservation)
        buf = self._buffer_planner.plan(
            model=model,
            placement_plan=plan,
            context_length=context_length,
            batch_size=batch_size,
            backend=self.backend,
        )

        # Peak VRAM: weights + KV reservation + activation + backend overhead
        peak_vram = (
            weights_vram
            + buf.kv_reservation_bytes
            + buf.activation_workspace_bytes
            + buf.backend_overhead_bytes
        )
        # Peak RAM: weights + OS overhead
        peak_ram = weights_ram + _OS_RAM_OVERHEAD_BYTES
        peak_total = peak_vram + peak_ram

        # KV fill ratio: KV at full context / VRAM headroom after weights+buffers
        fixed_vram = weights_vram + buf.activation_workspace_bytes + buf.backend_overhead_bytes
        headroom = max(1, vram_total - fixed_vram)
        kv_fill_ratio = kv_est.kv_at_full_context / headroom

        # Risk classification
        risk, oom_prob = self._classifier.classify(
            vram_total_bytes=vram_total,
            peak_vram_bytes=peak_vram,
            kv_fill_ratio=kv_fill_ratio,
        )

        # Compute safe context limit (for display and auto-resize)
        safe_ctx = self._kv_estimator.max_safe_context(
            model=model,
            plan=plan,
            vram_headroom_bytes=max(0, headroom),
        )

        # Assemble budget (warnings added below)
        budget = MemoryBudget(
            model_name=model.name,
            context_length=context_length,
            effective_context=context_length,
            kv_bytes_per_token=kv_est.kv_bytes_per_token,
            kv_at_quarter_context=kv_est.kv_at_quarter_context,
            kv_at_half_context=kv_est.kv_at_half_context,
            kv_at_full_context=kv_est.kv_at_full_context,
            kv_growth_rate_gb_per_1k=kv_est.kv_growth_rate_gb_per_1k,
            weights_vram_bytes=weights_vram,
            weights_ram_bytes=weights_ram,
            activation_workspace_bytes=buf.activation_workspace_bytes,
            backend_overhead_bytes=buf.backend_overhead_bytes,
            kv_reservation_bytes=buf.kv_reservation_bytes,
            peak_vram_bytes=peak_vram,
            peak_ram_bytes=peak_ram,
            peak_total_bytes=peak_total,
            risk_level=str(risk),
            oom_probability=oom_prob,
            kv_fill_ratio=round(kv_fill_ratio, 6),
            safe_context_length=safe_ctx,
            auto_resized=False,
            backend=self.backend,
        )

        # Generate and attach warnings
        budget.warnings = self._warner.generate(budget, self.hw_profile)

        return budget

    def _auto_resize(
        self,
        model: ModelDescriptor,
        plan: PlacementPlan,
        batch_size: int,
        original_budget: MemoryBudget,
    ) -> MemoryBudget:
        """
        Reduce context to the safe limit × safety factor and re-run analysis.

        Returns a new MemoryBudget with auto_resized=True.
        """
        safe_ctx = original_budget.safe_context_length
        # Apply 10% safety buffer and align to 512
        resized_ctx = max(
            self._MIN_CONTEXT,
            int(safe_ctx * self._AUTO_RESIZE_SAFETY)
            // KvCacheEstimator._CONTEXT_ALIGNMENT
            * KvCacheEstimator._CONTEXT_ALIGNMENT
        )
        resized_ctx = max(self._MIN_CONTEXT, resized_ctx)

        new_budget = self._analyse(model, plan, resized_ctx, batch_size)
        new_budget.context_length = original_budget.context_length  # preserve original
        new_budget.auto_resized = True

        # Ensure the auto-resize warning is present
        warnings = list(new_budget.warnings)
        resize_warning = (
            f"Context window automatically reduced from "
            f"{original_budget.context_length:,} → {resized_ctx:,} tokens "
            f"to prevent OOM. Maximum safe context: "
            f"{original_budget.safe_context_length:,} tokens."
        )
        # Avoid duplicate if WarningGenerator already added it
        if not any("automatically reduced" in w for w in warnings):
            warnings.insert(0, resize_warning)
        new_budget.warnings = warnings

        return new_budget

    # ---------------------------------------------------------------------------
    # Hardware helpers
    # ---------------------------------------------------------------------------

    def _get_vram_total(self) -> int:
        """Read total VRAM bytes from hw_profile."""
        gpus = self.hw_profile.get("gpus", [])
        if gpus:
            total_mb = sum(float(g.get("vram_total_mb", g.get("vram_mb", 0))) for g in gpus)
            if total_mb > 0:
                return int(total_mb * 1024 * 1024)
        return 8 * 1024 ** 3  # 8 GB safe fallback

    def _get_ram_total(self) -> int:
        """Read total RAM bytes from hw_profile."""
        ram = self.hw_profile.get("ram", self.hw_profile.get("memory", {}))
        total = int(ram.get("total_bytes", 0))
        if total == 0:
            avail_gb = float(ram.get("available_gb", 16.0))
            total = int(avail_gb * 1024 ** 3)
        return total
