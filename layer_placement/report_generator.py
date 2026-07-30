"""
report_generator.py
--------------------
Human-readable report generation for Phase 3 layer placement plans.

Produces two output formats:

1. **Text Report (ASCII art)**
   Formatted for console output or log files. Includes:
   - Model summary header
   - Visual placement map with layer ranges and device labels
   - Memory usage bars
   - Performance estimate
   - llama.cpp command hint

2. **JSON Report**
   Full machine-readable serialization of the PlacementPlan including
   all metadata, per-layer decisions, segments, and cost breakdown.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict

from .placement_plan import PlacementDevice, PlacementPlan, PlacementSegment


# Box-drawing characters for the text report
_BOX_TOP    = "╔══════════════════════════════════════════════════════════════╗"
_BOX_TITLE  = "║         InferenceOS  ·  Layer Placement Report               ║"
_BOX_BTM    = "╚══════════════════════════════════════════════════════════════╝"
_DIVIDER    = "──────────────────────────────────────────────────────────────"
_WIDTH      = 64  # printable width inside the box


class ReportGenerator:
    """
    Generates human-readable text and JSON placement reports.

    Parameters
    ----------
    plan : PlacementPlan
        The placement plan to report on.
    gpu_model_name : str, optional
        Human-readable GPU model name (e.g. "AMD RX 5600M") for labels.
    """

    def __init__(
        self,
        plan: PlacementPlan,
        gpu_model_name: str = "GPU",
    ) -> None:
        self.plan = plan
        self.gpu_label = gpu_model_name

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def generate_text_report(self) -> str:
        """
        Produce a full ASCII text report.

        Returns
        -------
        str
            Multi-line formatted report string suitable for console/log output.
        """
        lines = []
        plan = self.plan

        # Header
        lines.append(_BOX_TOP)
        lines.append(_BOX_TITLE)
        lines.append(_BOX_BTM)
        lines.append("")

        # Model summary
        lines.append(self._field("Model",        plan.model_name))
        lines.append(self._field("Architecture", plan.architecture))
        quant_line = f"{plan.total_layers} total layers"
        lines.append(self._field("Layers", quant_line))
        lines.append(self._field("Context Window", f"{plan.context_length:,} tokens"))
        lines.append(self._field("Feasible",     "Yes ✓" if plan.is_feasible else "No ✗ (memory exceeded)"))
        lines.append("")

        # Placement map
        lines.append("PLACEMENT MAP")
        lines.append(_DIVIDER)
        transformer_offset = self._transformer_layer_offset()
        for seg in plan.segments:
            lines.append(self._segment_line(seg, transformer_offset))
        lines.append("")

        # Optimizer summary
        lines.append("OPTIMIZER")
        lines.append(_DIVIDER)
        total_transformer = plan.n_gpu_layers + plan.n_cpu_layers
        pct = plan.gpu_offload_ratio * 100.0
        lines.append(self._field("GPU Transformer Layers", f"{plan.n_gpu_layers} / {total_transformer} ({pct:.1f}%)"))
        lines.append(self._field("CPU Transformer Layers", str(plan.n_cpu_layers)))
        lines.append(self._field("Boundary Crossings",     str(plan.boundary_crossings)))
        lines.append(self._field("Total Optimizer Cost",   f"{plan.total_cost:.4f}"))
        lines.append(self._field("SA Iterations",          str(plan.optimizer_iterations)))
        lines.append("")

        # Memory usage
        lines.append("MEMORY USAGE")
        lines.append(_DIVIDER)
        lines.append(self._memory_bar(
            "VRAM", plan.estimated_vram_bytes,
            plan.estimated_vram_bytes + plan.estimated_ram_bytes,
        ))
        lines.append(self._memory_bar(
            " RAM", plan.estimated_ram_bytes,
            plan.estimated_vram_bytes + plan.estimated_ram_bytes,
        ))
        lines.append(self._field("Peak Total", self._fmt_bytes(plan.estimated_peak_bytes)))
        lines.append("")

        # Warnings
        if plan.warnings:
            lines.append("WARNINGS")
            lines.append(_DIVIDER)
            for w in plan.warnings:
                lines.append(f"  ⚠  {w}")
            lines.append("")

        # llama.cpp hint
        lines.append("llama.cpp HINT")
        lines.append(_DIVIDER)
        n_hint = plan.llama_cpp_n_gpu_layers_hint
        lines.append(f"  --n-gpu-layers {n_hint}")
        if plan.boundary_crossings > 2:
            lines.append("  Note: Non-contiguous placement detected.")
            lines.append("        Use --split-mode layer for full segment support.")
        if plan.n_cpu_layers == 0:
            lines.append("  Full GPU offload — no CPU fallback needed.")
        elif plan.n_gpu_layers == 0:
            lines.append("  CPU-only inference — no GPU offload.")
        lines.append("")

        return "\n".join(lines)

    def generate_json_report(self, indent: int = 2) -> str:
        """
        Produce a full machine-readable JSON report.

        Returns
        -------
        str
            Formatted JSON string of the complete PlacementPlan.
        """
        return self.plan.to_json(indent=indent)

    def generate_summary_line(self) -> str:
        """One-line summary for inline logging."""
        return self.plan.summary()

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _field(self, label: str, value: str, label_width: int = 22) -> str:
        return f"  {label:<{label_width}}  {value}"

    def _fmt_bytes(self, b: int) -> str:
        if b >= 1024 ** 3:
            return f"{b / (1024**3):.2f} GB"
        elif b >= 1024 ** 2:
            return f"{b / (1024**2):.1f} MB"
        return f"{b / 1024:.0f} KB"

    def _transformer_layer_offset(self) -> int:
        """Return the layer_index of the first transformer block."""
        for lp in self.plan.layer_placements:
            if lp.layer_type == "transformer":
                return lp.layer_index
        return 0

    def _segment_line(self, seg: PlacementSegment, transformer_offset: int) -> str:
        """Format a single PlacementSegment into a display line."""
        start = seg.start_layer
        end = seg.end_layer
        device_label = (
            f"GPU  [{self.gpu_label}]"
            if seg.device == PlacementDevice.GPU
            else "CPU  [System RAM]"
        )
        size_str = self._fmt_bytes(seg.total_size_bytes)
        bar_width = 18
        # Build a mini fill bar proportional to segment size
        plan_total = max(
            self.plan.estimated_vram_bytes + self.plan.estimated_ram_bytes, 1
        )
        fill_ratio = seg.total_size_bytes / plan_total
        filled = max(1, int(fill_ratio * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        return f"  Layer {start:>4} – {end:>4}  │  {device_label:<22}  │  [{bar}]  {size_str}"

    def _memory_bar(self, label: str, used_bytes: int, total_bytes: int) -> str:
        """Render a horizontal memory utilisation bar."""
        bar_width = 24
        ratio = used_bytes / max(total_bytes, 1)
        filled = int(ratio * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = ratio * 100.0
        return f"  {label}  [{bar}]  {self._fmt_bytes(used_bytes)}  ({pct:.1f}% of total memory)"


# ---------------------------------------------------------------------------
# Convenience factory function
# ---------------------------------------------------------------------------

def generate_text_report(
    plan: PlacementPlan,
    gpu_model_name: str = "GPU",
) -> str:
    """
    Convenience wrapper — generate a text report from a :class:`PlacementPlan`.

    Parameters
    ----------
    plan : PlacementPlan
        The plan to report on.
    gpu_model_name : str
        Human-friendly GPU model label.

    Returns
    -------
    str
        Full ASCII text report.
    """
    return ReportGenerator(plan, gpu_model_name).generate_text_report()


def generate_json_report(
    plan: PlacementPlan,
    indent: int = 2,
) -> str:
    """
    Convenience wrapper — generate a JSON report from a :class:`PlacementPlan`.

    Parameters
    ----------
    plan : PlacementPlan
        The plan to report on.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        Formatted JSON string.
    """
    return ReportGenerator(plan).generate_json_report(indent=indent)
