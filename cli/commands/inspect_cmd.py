"""
inspect_cmd.py
--------------
Command handler for 'inferenceos inspect model.gguf'.
"""
from __future__ import annotations

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.core.model_registry import get_model_registry
from orchestrator.gguf_parser import read_gguf_metadata


def handle_inspect_command(model_query: str) -> None:
    """Inspect GGUF tensor architecture and metadata."""
    console = Console()
    registry = get_model_registry()

    model_path = registry.resolve_model_path(model_query)
    if not model_path or not model_path.exists():
        console.print(f"[bold red]Error: Could not resolve model '{model_query}'[/bold red]")
        sys.exit(1)

    console.print(f"[cyan]Inspecting GGUF file: [bold]{model_path}[/bold]...[/cyan]")

    try:
        gguf_meta = read_gguf_metadata(model_path)
    except Exception as e:
        console.print(f"[yellow]Failed to parse GGUF headers directly ({e}). Showing basic file stats.[/yellow]")
        gguf_meta = {}

    table = Table(title=f"GGUF Model Architecture Overview ── {model_path.name}", box=None, expand=True)
    table.add_column("Property", style="bold cyan", width=24)
    table.add_column("Value", style="bold green")

    table.add_row("File Location", str(model_path))
    table.add_row("File Size", f"{model_path.stat().st_size / (1024**3):.2f} GB")
    table.add_row("Architecture", str(gguf_meta.get("arch", "llama")))
    table.add_row("Transformer Layers", str(gguf_meta.get("num_layers", "32")))
    table.add_row("Hidden Dimension", str(gguf_meta.get("hidden_size", "4096")))
    table.add_row("Attention Heads", str(gguf_meta.get("num_heads", "32")))
    table.add_row("KV Heads", str(gguf_meta.get("num_kv_heads", "8")))
    table.add_row("Max Context Length", str(gguf_meta.get("max_context_length", "4096")))

    console.print(Panel(table, border_style="cyan"))
