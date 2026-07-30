"""
InferenceOS Runtime Profiler — Phase 9 Public API

Provides comprehensive inference telemetry, flame graph visualizers, interactive Gantt timelines,
performance reports, and optimization recommendations.

Exports
-------
    ProfilerEngine               — Main facade & context manager
    ProfilerResult               — Container holding telemetry, flamegraph, timeline, report
    TelemetryCollector           — Live metrics sampling engine
    ProfilerTelemetry            — Full metrics dataclass
    FlameGraphGenerator          — Interactive HTML/SVG & JSON flame graph generator
    TimelineGenerator            — Interactive Gantt timeline generator
    OptimizationAdvisor          — Recommendation engine
    OptimizationRecommendation   — Structured recommendation object with estimated impact
    PerformanceReportGenerator   — Markdown & JSON report generator
"""
from __future__ import annotations

from .flame_graph import FlameGraphGenerator, FlameNode
from .optimization_advisor import (
    OptimizationAdvisor,
    OptimizationRecommendation,
    Severity,
)
from .performance_report import PerformanceReportGenerator
from .profiler_engine import ProfilerEngine, ProfilerResult
from .telemetry_collector import (
    LayerTimingRecord,
    ProfilerTelemetry,
    TelemetryCollector,
)
from .timeline_view import TimelineEvent, TimelineGenerator

__all__ = [
    "ProfilerEngine",
    "ProfilerResult",
    "TelemetryCollector",
    "ProfilerTelemetry",
    "LayerTimingRecord",
    "FlameGraphGenerator",
    "FlameNode",
    "TimelineGenerator",
    "TimelineEvent",
    "OptimizationAdvisor",
    "OptimizationRecommendation",
    "Severity",
    "PerformanceReportGenerator",
]
