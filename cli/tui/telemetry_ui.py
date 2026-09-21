"""
telemetry_ui.py
---------------
Terminal Telemetry Dashboard for InferenceOS.

Displays historical TPS averages, latency distributions, prediction accuracy,
and benchmark trends rendered in rich terminal tables and ascii graphs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from .themes import get_theme
from cli.core.telemetry_store import get_telemetry_store


class TelemetryInterface:
    """
    Renders terminal dashboard summarizing performance metrics & history.
    """

    def __init__(self, theme_name: str = "nord") -> None:
        self.console = Console()
        self.theme_mgr = get_theme(theme_name)
        self.store = get_telemetry_store()

    def display(self) -> None:
        """Render formatted telemetry overview."""
        self.console.clear()
        tm = self.theme_mgr
        c_primary = tm.color("primary")
        c_secondary = tm.color("secondary")
        c_success = tm.color("success")
        c_warning = tm.color("warning")

        summary = self.store.get_summary_stats()
        history = self.store.get_history(limit=15)

        # Overview Cards
        overview_table = Table(box=None, expand=True)
        overview_table.add_column("TOTAL RUNS", justify="center", style=f"bold {c_primary}")
        overview_table.add_column("AVG GEN TPS", justify="center", style=f"bold {c_success}")
        overview_table.add_column("AVG PROMPT TPS", justify="center", style=f"bold {c_secondary}")
        overview_table.add_column("AVG TTFT (MS)", justify="center", style=f"bold {c_warning}")

        overview_table.add_row(
            str(summary["total_runs"]),
            f"{summary['avg_generation_tps']} tok/s",
            f"{summary['avg_prompt_tps']} tok/s",
            f"{summary['avg_ttft_ms']} ms",
        )

        # History Table
        hist_table = Table(title="Recent Telemetry Runs", box=None, expand=True)
        hist_table.add_column("Timestamp", style="dim")
        hist_table.add_column("Model", style="bold cyan")
        hist_table.add_column("Backend", style="magenta")
        hist_table.add_column("Gen TPS", justify="right", style="bold green")
        hist_table.add_column("Prompt TPS", justify="right", style="cyan")
        hist_table.add_column("TTFT", justify="right", style="yellow")
        hist_table.add_column("Tokens", justify="right", style="white")

        for r in reversed(history):
            ts = str(r.get("timestamp", ""))[:19].replace("T", " ")
            hist_table.add_row(
                ts,
                str(r.get("model_name", "unknown"))[:20],
                str(r.get("backend", "auto")).upper(),
                f"{r.get('generation_tps', 0.0):.2f}",
                f"{r.get('prompt_tps', 0.0):.1f}",
                f"{r.get('ttft_ms', 0.0):.1f} ms",
                str(r.get("tokens_generated", 0)),
            )

        if not history:
            hist_table.add_row("-", "No telemetry runs recorded yet", "-", "0.00", "0.0", "0 ms", "0")

        grid = Table(box=None, expand=True)
        grid.add_column("METRICS SUMMARY")
        grid.add_row(overview_table)
        grid.add_row(Text("\n"))
        grid.add_row(hist_table)

        header = Text("⚡ InferenceOS Telemetry & Metric Dashboard", style=f"bold {c_primary}")
        panel = Panel(grid, title=header, border_style=c_primary, padding=(1, 2))
        self.console.print(panel)
