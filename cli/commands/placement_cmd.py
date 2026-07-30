"""
placement_cmd.py
----------------
Command handler for 'inferenceos placement model.gguf'.
"""
from __future__ import annotations

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.core.model_registry import get_model_registry
import profiler
from layer_placement import ModelDescriptor, PlacementEngine
from orchestrator.gguf_parser import read_gguf_metadata


def handle_placement_command(model_query: str) -> None:
    """Compute and visualize layer placement distribution plan."""
    console = Console()
    registry = get_model_registry()

    model_path = registry.resolve_model_path(model_query)
    if not model_path or not model_path.exists():
        console.print(f"[bold red]Error: Could not resolve model '{model_query}'[/bold red]")
        sys.exit(1)

    console.print(f"[cyan]Computing placement plan for [bold]{model_path.name}[/bold]...[/cyan]")

    sys_res = profiler.get_system_resources()
    hw_profile = sys_res.to_dict() if hasattr(sys_res, "to_dict") else sys_res

    try:
        gguf_meta = read_gguf_metadata(model_path)
    except Exception:
        gguf_meta = {"arch": "llama", "num_layers": 32, "max_context_length": 4096}

    model_desc = ModelDescriptor.from_gguf_metadata(
        metadata=gguf_meta,
        model_size_bytes=model_path.stat().st_size,
        model_name=model_path.stem,
    )

    placement_engine = PlacementEngine(hw_profile=hw_profile)
    plan = placement_engine.generatePlacementPlan(model=model_desc, context_length=4096)

    table = Table(title=f"Layer Placement Plan ── {model_path.name}", box=None, expand=True)
    table.add_column("Layer Distribution", style="bold cyan")
    table.add_column("Layer Count", justify="right", style="bold green")
    table.add_column("Percentage", justify="right", style="bold yellow")

    tot = max(1, plan.total_layers)
    table.add_row("Discrete GPU Layers (dGPU)", str(plan.n_gpu_layers), f"{(plan.n_gpu_layers / tot)*100:.1f}%")
    table.add_row("Integrated GPU Layers (iGPU)", str(plan.n_igpu_layers), f"{(plan.n_igpu_layers / tot)*100:.1f}%")
    table.add_row("System CPU Layers (RAM)", str(plan.n_cpu_layers), f"{(plan.n_cpu_layers / tot)*100:.1f}%")
    table.add_row("Total Transformer Blocks", str(plan.total_layers), "100.0%")

    console.print(Panel(table, border_style="cyan"))
    console.print(f"  Est. VRAM Memory: [bold green]{plan.estimated_vram_bytes / (1024**2):.1f} MB[/bold green]")
    console.print(f"  Est. System RAM:  [bold yellow]{plan.estimated_ram_bytes / (1024**2):.1f} MB[/bold yellow]")
    console.print(f"  Boundary Crossings: [cyan]{plan.boundary_crossings}[/cyan]")
