"""
profiler_engine.py
------------------
ProfilerEngine facade for Phase 9 Runtime Profiler.

High-level interface for profiling inference sessions:
  - Context manager `with engine.profile_session(): ...`
  - `profile_run()` helper for profiling execution runs.
  - Generates Flame Graphs, Timeline views, Reports, and Recommendations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .flame_graph import FlameGraphGenerator
from .optimization_advisor import OptimizationAdvisor, OptimizationRecommendation
from .performance_report import PerformanceReportGenerator
from .telemetry_collector import ProfilerTelemetry, TelemetryCollector
from .timeline_view import TimelineGenerator


# ---------------------------------------------------------------------------
# ProfilerResult Output Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfilerResult:
    """
    Combined container for all Phase 9 profiling artifacts.
    """
    telemetry: ProfilerTelemetry
    recommendations: List[OptimizationRecommendation]
    flame_graph: FlameGraphGenerator
    timeline: TimelineGenerator
    report: PerformanceReportGenerator

    def summary(self) -> str:
        t = self.telemetry
        top_rec = self.recommendations[0].summary_line() if self.recommendations else "Optimal"
        return (
            f"Profiler Report: {t.generation_tps:.1f} tok/s  |  "
            f"TTFT {t.ttft_ms:.0f}ms  |  GPU Idle {t.gpu_idle_pct:.0f}%  |  "
            f"Top Recommendation: {top_rec}"
        )

    def export_all(self, output_dir: Path, prefix: str = "profile") -> Dict[str, Path]:
        """
        Export HTML flame graph, HTML timeline, Markdown report, and JSON report files.

        Returns dict mapping file type to Path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fg_path = output_dir / f"{prefix}_flamegraph.html"
        tl_path = output_dir / f"{prefix}_timeline.html"
        md_path = output_dir / f"{prefix}_report.md"
        json_path = output_dir / f"{prefix}_report.json"

        fg_path.write_text(self.flame_graph.to_html(), encoding="utf-8")
        tl_path.write_text(self.timeline.to_html(), encoding="utf-8")
        md_path.write_text(self.report.to_markdown(), encoding="utf-8")
        json_path.write_text(self.report.to_json(), encoding="utf-8")

        return {
            "flame_graph_html": fg_path,
            "timeline_html": tl_path,
            "report_markdown": md_path,
            "report_json": json_path,
        }


# ---------------------------------------------------------------------------
# ProfilerEngine
# ---------------------------------------------------------------------------

class ProfilerEngine:
    """
    Primary manager for Phase 9 Runtime Profiler.

    Parameters
    ----------
    hw_profile : dict, optional
        Hardware profile.
    sample_interval_ms : float
        Telemetry sampling interval in ms. Default 100.
    """

    def __init__(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        sample_interval_ms: float = 100.0,
    ) -> None:
        self.hw_profile = hw_profile or {}
        self.sample_interval_ms = sample_interval_ms

    @contextmanager
    def profile_session(
        self,
        plan: Optional[Any] = None,
        model_name: str = "",
        backend: str = "cpu",
        context_length: int = 4096,
        vram_free_mb: float = 0.0,
    ):
        """
        Context manager for profiling an inference session.

        Yields the :class:`TelemetryCollector` instance so tokens can be recorded.

        Example
        -------
            with profiler.profile_session(plan, model_name="7B") as collector:
                # Run inference session
                collector.record_token()
            result = profiler.get_result()
        """
        collector = TelemetryCollector(
            sample_interval_ms=self.sample_interval_ms,
            plan=plan,
            hw_profile=self.hw_profile,
        )
        collector.start()
        try:
            yield collector
        finally:
            collector.stop()

        # Build results
        self._last_telemetry = collector.finalize(
            model_name=model_name,
            backend=backend,
            context_length=context_length,
        )

        advisor = OptimizationAdvisor(
            telemetry=self._last_telemetry,
            hw_profile=self.hw_profile,
            vram_free_mb=vram_free_mb,
        )
        recs = advisor.analyze()

        fg = FlameGraphGenerator(self._last_telemetry)
        tl = TimelineGenerator(self._last_telemetry)
        rep = PerformanceReportGenerator(
            telemetry=self._last_telemetry,
            recommendations=recs,
            flame_graph=fg,
            timeline=tl,
        )

        self._last_result = ProfilerResult(
            telemetry=self._last_telemetry,
            recommendations=recs,
            flame_graph=fg,
            timeline=tl,
            report=rep,
        )

    def get_last_result(self) -> Optional[ProfilerResult]:
        return getattr(self, "_last_result", None)
