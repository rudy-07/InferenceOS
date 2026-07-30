"""
stats_cmd.py
------------
Command handler for 'inferenceos stats'.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from cli.core.telemetry_store import get_telemetry_store


def handle_stats_command() -> None:
    """Display overall summary performance statistics."""
    console = Console()
    store = get_telemetry_store()
    stats = store.get_summary_stats()

    table = Table(title="InferenceOS Summary Performance Statistics", box=None, expand=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Total Inference Runs", str(stats["total_runs"]))
    table.add_row("Average Generation TPS", f"{stats['avg_generation_tps']} tok/s")
    table.add_row("Average Prompt Evaluation TPS", f"{stats['avg_prompt_tps']} tok/s")
    table.add_row("Average Time To First Token (TTFT)", f"{stats['avg_ttft_ms']} ms")
    table.add_row("Unique Models Executed", str(stats["models_run"]))

    console.print(Panel(table, border_style="cyan"))
