"""
performance_report.py
----------------------
Performance report generator for Phase 9 Runtime Profiler.

Synthesizes telemetry, flame graph summaries, timeline events, and optimization
recommendations into formatted Markdown reports and structured JSON documents.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .flame_graph import FlameGraphGenerator
from .optimization_advisor import OptimizationAdvisor, OptimizationRecommendation
from .telemetry_collector import ProfilerTelemetry
from .timeline_view import TimelineGenerator


# ---------------------------------------------------------------------------
# PerformanceReportGenerator
# ---------------------------------------------------------------------------

class PerformanceReportGenerator:
    """
    Generates Markdown and JSON reports for inference runs.
    """

    def __init__(
        self,
        telemetry: ProfilerTelemetry,
        recommendations: Optional[List[OptimizationRecommendation]] = None,
        flame_graph: Optional[FlameGraphGenerator] = None,
        timeline: Optional[TimelineGenerator] = None,
    ) -> None:
        self.telemetry = telemetry
        self.recommendations = recommendations or []
        self.flame_graph = flame_graph
        self.timeline = timeline

    def to_dict(self) -> Dict[str, Any]:
        """Return structured JSON object representation of the report."""
        t = self.telemetry
        return {
            "summary": {
                "model_name": t.model_name,
                "backend": t.backend,
                "total_wall_ms": round(t.total_wall_ms, 2),
                "generation_tps": round(t.generation_tps, 2),
                "prompt_tps": round(t.prompt_tps, 2),
                "ttft_ms": round(t.ttft_ms, 2),
                "p50_itl_ms": round(t.p50_itl_ms, 2),
                "p95_itl_ms": round(t.p95_itl_ms, 2),
                "gpu_idle_pct": round(t.gpu_idle_pct, 1),
            },
            "telemetry": t.to_dict(),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "flame_graph": self.flame_graph.root.to_dict() if self.flame_graph else None,
            "timeline": [e.to_dict() for e in self.timeline.events] if self.timeline else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export full report as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate human-readable GitHub-flavored Markdown report."""
        t = self.telemetry
        lines = [
            "# 🚀 InferenceOS Phase 9 — Runtime Performance Report",
            "",
            "## Executive Summary",
            "",
            f"- **Model**: `{t.model_name or 'N/A'}`",
            f"- **Backend**: `{t.backend.upper()}`",
            f"- **Total Wall Time**: `{t.total_wall_ms:.1f} ms`",
            f"- **Generation Throughput**: **`{t.generation_tps:.2f} tokens/sec`**",
            f"- **Prompt Processing**: `{t.prompt_tps:.2f} tokens/sec`",
            f"- **Time To First Token (TTFT)**: `{t.ttft_ms:.1f} ms`",
            "",
            "---",
            "",
            "## ⏱️ Token Latency Breakdown (Inter-Token Latency)",
            "",
            "| Metric | Latency (ms) | Description |",
            "|---|---|---|",
            f"| **p50 (Median)** | `{t.p50_itl_ms:.2f} ms` | Expected per-token generation time |",
            f"| **p90** | `{t.p90_itl_ms:.2f} ms` | 90th percentile arrival |",
            f"| **p95** | `{t.p95_itl_ms:.2f} ms` | 95th percentile tail latency |",
            f"| **p99** | `{t.p99_itl_ms:.2f} ms` | Worst-case spike latency |",
            f"| **StdDev** | `{t.stddev_itl_ms:.2f} ms` | Latency standard deviation |",
            f"| **Jitter** | `{t.jitter_itl_ms:.2f} ms` | Mean inter-token variation |",
            "",
            "---",
            "",
            "## 💻 Hardware & Bandwidth Utilization",
            "",
            "| Subsystem | Measured Metric | Status / Peak |",
            "|---|---|---|",
            f"| **GPU Compute** | `{t.avg_gpu_util_pct:.1f}% avg` | `{t.peak_gpu_util_pct:.1f}% peak` |",
            f"| **GPU Idle Time** | `{t.gpu_idle_pct:.1f}%` | {'⚠️ Stalling' if t.gpu_idle_pct >= 15 else '✅ Efficient'} |",
            f"| **CPU Core Load** | `{t.avg_cpu_util_pct:.1f}% avg` | `{t.peak_cpu_util_pct:.1f}% peak` |",
            f"| **VRAM Bandwidth** | `{t.vram_bandwidth_gbps:.1f} GB/s` | Memory bus throughput |",
            f"| **RAM Bandwidth** | `{t.ram_bandwidth_gbps:.1f} GB/s` | System memory throughput |",
            f"| **PCIe Bandwidth** | `{t.pcie_bandwidth_gbps:.1f} GB/s` | Transfer bus throughput |",
            f"| **PCIe Transferred** | `{t.pcie_bytes_transferred / (1024*1024):.1f} MB` | Cross-boundary layer data |",
            f"| **KV Cache Size** | `{t.kv_cache_size_mb:.1f} MB` | Context: `{t.context_utilization_pct:.1f}%` used |",
            "",
            "---",
            "",
            "## 🎯 Optimization Recommendations",
            "",
        ]

        if self.recommendations:
            for i, r in enumerate(self.recommendations, start=1):
                badge = "🔴 **[CRITICAL]**" if r.severity.name == "CRITICAL" else ("🟡 **[WARNING]**" if r.severity.name == "WARNING" else "🔵 **[INFO]**")
                lines.extend([
                    f"### {i}. {r.title}",
                    f"- {badge} **Bottleneck**: {r.bottleneck}",
                    f"- 🛠️ **Recommended Action**: `{r.action}`",
                    f"- 📈 **Estimated Improvement**: **`{r.estimated_improvement}`**",
                    "",
                ])
        else:
            lines.append("✅ *No bottlenecks detected! Current layer placement and runtime execution are fully optimal.*")

        lines.extend([
            "---",
            "",
            "## 🧱 Layer Placement Topology",
            "",
            f"- **dGPU Layers**: `{t.n_gpu_layers}`",
            f"- **CPU Layers**: `{t.n_cpu_layers}`",
            f"- **iGPU Layers**: `{t.n_igpu_layers}`",
            f"- **Total Model Layers**: `{t.total_layers}`",
            "",
        ])

        return "\n".join(lines)
