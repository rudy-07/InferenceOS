"""
profile.py
----------
Command handler for 'inferenceos profile model.gguf'.
"""
from __future__ import annotations

import sys
from typing import Optional
from rich.console import Console
from cli.core.model_registry import get_model_registry
from run_e2e_integration import PipelineConfig, run_end_to_end_validation


def handle_profile_command(model_query: str) -> None:
    """Run full runtime profiling and export flamegraph / timeline HTMLs."""
    console = Console()
    registry = get_model_registry()

    model_path = registry.resolve_model_path(model_query)
    if not model_path or not model_path.exists():
        console.print(f"[bold red]Error: Could not resolve model '{model_query}'[/bold red]")
        sys.exit(1)

    console.print(f"[bold cyan]Profiling InferenceOS Execution for [bold green]{model_path.name}[/bold green]...[/bold cyan]")

    pipe_cfg = PipelineConfig()
    pipe_cfg.model_name = model_path.name
    pipe_cfg.enable_profiler_telemetry = True

    run_end_to_end_validation(pipe_cfg)
    console.print("[bold green]✔ Profiling complete! Flamegraph & Timeline HTML reports saved to benchmarks/ directory.[/bold green]")
