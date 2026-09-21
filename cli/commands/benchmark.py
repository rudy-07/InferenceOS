"""
benchmark.py
------------
Command handler for 'inferenceos benchmark [model]'.
"""
from __future__ import annotations

import sys
from typing import Optional
from rich.console import Console
from cli.core.model_registry import get_model_registry
from run_e2e_integration import PipelineConfig, run_end_to_end_validation


def handle_benchmark_command(
    model_query: Optional[str] = None,
    gpu_layers: Optional[int] = None,
    threads: Optional[int] = None,
) -> None:
    """Execute complete end-to-end benchmark suite."""
    console = Console()
    pipe_cfg = PipelineConfig()

    if model_query:
        registry = get_model_registry()
        mp = registry.resolve_model_path(model_query)
        if not mp or not mp.exists():
            console.print(f"[bold red]Error: Could not resolve model '{model_query}'[/bold red]")
            sys.exit(1)
        pipe_cfg.model_name = mp.name

    if gpu_layers is not None:
        pipe_cfg.gpu_layers_override = gpu_layers
    if threads is not None:
        pipe_cfg.threads = threads

    console.print("[bold cyan]Running InferenceOS End-to-End Benchmark Suite...[/bold cyan]\n")
    run_end_to_end_validation(pipe_cfg)
